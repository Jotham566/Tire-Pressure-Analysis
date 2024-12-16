import os
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from tqdm import tqdm
import logging

# Constants and Configuration
FULL_SEGMENTS_FOLDER = "Full_Segments"
OUTPUT_FOLDER = "Processed_csv"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Known pressures for filename parsing
KNOWN_PRESSURES = ['500', '600', '700', '800', '850', '900']

# Desired number of samples per hit for CSV
DESIRED_LENGTH = 512  # Adjust based on your data

# Setup Logging
logging.basicConfig(
    filename='process_audio.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_tire_info(full_segment_filename):
    """
    Extract tire number, pressure, TireSize, and Ver from the full segment filename.

    Args:
        full_segment_filename (str): The name of the full segment WAV file.

    Returns:
        tuple: (tire_num, pressure, tire_size, ver, wear, rim)

    Raises:
        ValueError: If necessary information cannot be extracted.
    """
    base = os.path.splitext(os.path.basename(full_segment_filename))[0]
    parts = base.split('_')

    if len(parts) < 2:
        raise ValueError(f"Unexpected full segment filename format: {full_segment_filename}")

    tire_num = parts[0]
    remaining_info = '_'.join(parts[1:])

    # Regular expression to extract TireSize and Ver
    # Assuming TireSize is between the first '_' and the first '-' before pressure
    # Ver is between pressure and '_H.wav'

    # Extract TireSize
    tire_size_pattern = r'_(.+?)-(' + '|'.join(KNOWN_PRESSURES) + r')-'
    tire_size_match = re.search(tire_size_pattern, full_segment_filename)
    if tire_size_match:
        tire_size = tire_size_match.group(1)
    else:
        raise ValueError(f"Could not extract TireSize from filename: {full_segment_filename}")

    # Extract Pressure
    pressure_found = None
    for p in KNOWN_PRESSURES:
        pattern = r"(?:^|\D)" + re.escape(p) + r"(?:\D|$)"
        if re.search(pattern, remaining_info):
            pressure_found = p
            break

    if pressure_found is None:
        raise ValueError(f"Could not extract a known pressure from filename: {full_segment_filename}")

    pressure = pressure_found

    # Extract Ver
    ver_pattern = r'-' + re.escape(pressure) + r'-(.+?)_H$'
    ver_match = re.search(ver_pattern, base)
    if ver_match:
        ver = ver_match.group(1)
    else:
        raise ValueError(f"Could not extract Ver from filename: {full_segment_filename}")

    # Determine Rim
    rim = determine_rim(tire_num)

    # Determine Wear
    wear = determine_wear(full_segment_filename)

    return tire_num, pressure, tire_size, ver, wear, rim

def determine_rim(tire_num):
    """
    Determine the Rim type based on the Tire Number.

    Args:
        tire_num (str): Tire number extracted from the filename.

    Returns:
        str: 'Al' for Aluminum or 'Ir' for Iron.
    """
    al_rims = {str(i) for i in range(42, 54)}  # 42 to 53 inclusive
    if tire_num in al_rims:
        return 'Al'
    else:
        return 'Ir'

def determine_wear(full_segment_filename):
    """
    Determine the Wear level based on the filename.

    Args:
        full_segment_filename (str): The name of the full segment WAV file.

    Returns:
        str: '0%', '50%', or '100%'.
    """
    if re.search(r'バフ100_H\.wav$', full_segment_filename):
        return '100%'
    elif re.search(r'バフ50_H\.wav$', full_segment_filename):
        return '50%'
    else:
        return '0%'

def load_audio(file_path):
    """
    Load the WAV file using scipy.io.wavfile and return raw samples as a NumPy array (mono).

    Args:
        file_path (str): Path to the WAV file.

    Returns:
        tuple: (sample_rate, samples)
    """
    try:
        sample_rate, data = wavfile.read(file_path)
        # If stereo, convert to mono by averaging channels
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
        samples = data.astype(np.float32)
        return sample_rate, samples
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")

def detect_hits(samples, sample_rate, num_hits=5):
    """
    Detect hits using peak detection and energy-based criteria.

    Args:
        samples (np.ndarray): The full audio samples.
        sample_rate (int): Sampling rate of the audio.
        num_hits (int): Approximate number of hits to detect.

    Returns:
        list of tuples: List of (start_index, end_index) for each hit.
    """
    # Preprocess: take absolute value and square to emphasize energy
    processed_signal = np.abs(samples) ** 2

    # Use a sliding window to smooth the signal
    window_size = int(sample_rate * 0.01)  # 10ms window
    smoothed_signal = np.convolve(processed_signal, np.ones(window_size)/window_size, mode='same')

    # Find peaks with prominence to identify hits
    peaks, _ = find_peaks(smoothed_signal, 
                          height=np.mean(smoothed_signal), 
                          distance=int(sample_rate * 0.1),  # min 100ms between hits
                          prominence=np.std(smoothed_signal))

    # If not enough peaks, adjust detection
    if len(peaks) < num_hits:
        # Try with lower threshold
        peaks, _ = find_peaks(smoothed_signal, 
                              height=np.mean(smoothed_signal) * 0.5, 
                              distance=int(sample_rate * 0.05),
                              prominence=np.std(smoothed_signal) * 0.5)

    # Ensure we have desired number of hits
    if len(peaks) > num_hits:
        # If too many peaks, take the most prominent ones
        peak_prominences = np.zeros_like(peaks, dtype=float)
        for i, peak in enumerate(peaks):
            left_base = max(0, peak - int(sample_rate * 0.05))
            right_base = min(len(smoothed_signal), peak + int(sample_rate * 0.05))
            peak_prominences[i] = smoothed_signal[peak] - np.min(smoothed_signal[left_base:right_base])

        # Sort peaks by prominence and take top num_hits
        top_peak_indices = np.argsort(peak_prominences)[-num_hits:]
        peaks = peaks[top_peak_indices]

    # Extract hits with context around peaks
    hit_segments = []
    context_width = int(sample_rate * 0.05)  # 50ms context on each side
    for peak in peaks:
        start = max(0, peak - context_width)
        end = min(len(samples), peak + context_width)
        hit_segments.append((start, end))

    # Ensure no overlap between segments
    hit_segments = sorted(hit_segments, key=lambda x: x[0])
    non_overlapping_segments = []
    for segment in hit_segments:
        # If this segment overlaps with previous, adjust its start
        if non_overlapping_segments and segment[0] < non_overlapping_segments[-1][1]:
            segment = (non_overlapping_segments[-1][1], segment[1])
        non_overlapping_segments.append(segment)

    return non_overlapping_segments

def process_hit(samples, hit_segment, desired_length=DESIRED_LENGTH):
    """
    Process a single hit by extracting the segment and preparing for CSV.

    Args:
        samples (np.ndarray): Full audio samples.
        hit_segment (tuple): (start, end) indices of the hit.
        desired_length (int): The desired number of samples.

    Returns:
        np.ndarray: Processed hit with the desired length.
    """
    # Extract hit segment
    hit = samples[hit_segment[0]:hit_segment[1]]
    
    # Zero-offset correction
    hit_corrected = hit - np.mean(hit)
    
    # Square the signal to get intensity
    hit_intensity = hit_corrected ** 2
    
    # Truncate or pad to desired_length
    if len(hit_intensity) >= desired_length:
        return hit_intensity[:desired_length]
    else:
        padding = desired_length - len(hit_intensity)
        hit_padded = np.pad(hit_intensity, (0, padding), 'constant')
        # Normalize so that the sum of the segment is 1
        hit_normalized = hit_padded / np.sum(hit_padded)
        return hit_normalized

def save_to_csv(tire_num, pressure, tire_size, ver, wear, rim, segments, output_dir):
    """
    Save the processed segments to a CSV file.

    Args:
        tire_num (str): Tire number.
        pressure (str): Tire pressure.
        tire_size (str): Tire size.
        ver (str): Version.
        wear (str): Wear level.
        rim (str): Rim type.
        segments (list of np.ndarray): List of processed hit segments.
        output_dir (str): Directory to save the CSV file.
    """
    data = []
    for idx, seg in enumerate(segments, start=1):
        row = {
            "Segment ID / Value index": f"signal segment {idx}",
            "Tire Number": tire_num,
            "Pressure": pressure,
            "TireSize": tire_size,
            "Ver": ver,
            "Wear": wear,
            "Rim": rim
        }
        # Add signal values
        for i, value in enumerate(seg, start=1):
            row[f"Signal Value {i}"] = value
        data.append(row)

    # Define column order
    signal_columns = [f"Signal Value {i}" for i in range(1, DESIRED_LENGTH + 1)]
    columns = ["Segment ID / Value index", "Tire Number", "Pressure", "TireSize", "Ver", "Wear", "Rim"] + signal_columns
    df = pd.DataFrame(data, columns=columns)

    # Define CSV filename
    filename = f"{tire_num}_{tire_size}-{pressure}-{ver}.csv"
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)

def main():
    # Get list of full segment WAV files
    try:
        full_segment_files = [f for f in os.listdir(FULL_SEGMENTS_FOLDER) if f.lower().endswith('.wav')]
        if not full_segment_files:
            print("⚠️ No full segment WAV files found to process.")
            logging.warning("No full segment WAV files found to process.")
            return
    except FileNotFoundError:
        print(f"❌ The directory '{FULL_SEGMENTS_FOLDER}' does not exist.")
        logging.error(f"The directory '{FULL_SEGMENTS_FOLDER}' does not exist.")
        return

    # Process each full segment file with a progress bar
    for full_segment_filename in tqdm(full_segment_files, desc="Processing Full Segments"):
        try:
            # Parse tire information
            tire_num, pressure, tire_size, ver, wear, rim = parse_tire_info(full_segment_filename)
            logging.info(f"Processing file: {full_segment_filename} (Tire: {tire_num}, Pressure: {pressure}, TireSize: {tire_size}, Ver: {ver}, Wear: {wear}, Rim: {rim})")
        except ValueError as ve:
            logging.warning(f"Skipping '{full_segment_filename}': {ve}")
            print(f"⚠️ Skipping '{full_segment_filename}': {ve}")
            continue

        # Load audio samples
        file_path = os.path.join(FULL_SEGMENTS_FOLDER, full_segment_filename)
        try:
            sample_rate, samples = load_audio(file_path)
            logging.info(f"Loaded audio file '{full_segment_filename}' with sample rate {sample_rate} Hz.")
        except RuntimeError as e:
            logging.error(f"Failed to load '{full_segment_filename}': {e}")
            print(f"⚠️ {e}. Skipping.")
            continue

        # Detect and segment hits
        try:
            hit_segments = detect_hits(samples, sample_rate, num_hits=5)
            logging.info(f"Detected {len(hit_segments)} hits in '{full_segment_filename}'.")
        except Exception as e:
            logging.warning(f"Error detecting hits in '{full_segment_filename}': {e}")
            print(f"⚠️ Error detecting hits in '{full_segment_filename}': {e}")
            continue

        # Process each hit
        processed_hits = []
        for hit_segment in hit_segments:
            processed_hit = process_hit(samples, hit_segment, desired_length=DESIRED_LENGTH)
            processed_hits.append(processed_hit)

        # Save to CSV
        try:
            save_to_csv(tire_num, pressure, tire_size, ver, wear, rim, processed_hits, OUTPUT_FOLDER)
            logging.info(f"Saved processed hits to '{tire_num}_{pressure}.csv'.")
        except Exception as e:
            logging.error(f"Failed to save CSV for '{full_segment_filename}': {e}")
            print(f"⚠️ Failed to save CSV for '{full_segment_filename}': {e}")
            continue

        # Calculate actual duration
        actual_duration = len(samples) / sample_rate  # in seconds
        actual_duration = round(actual_duration, 3)

        # Display duration comparison (optional)
        print(f"\n📄 CSV: {tire_num}_{pressure}.csv")
        print(f"🎧 Actual Full Segment Duration: {actual_duration} seconds")
        print("Detected Hits:")
        for i, (start, end) in enumerate(hit_segments, 1):
            hit_duration = (end - start) / sample_rate
            print(f"  Hit {i}: {round(start/sample_rate, 3)} - {round(end/sample_rate, 3)} seconds (Duration: {round(hit_duration, 3)} s)")
        print()

    print("✅ Processing complete.")
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
