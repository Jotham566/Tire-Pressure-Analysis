import os
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from tqdm import tqdm
import logging
from typing import Dict, List

# Constants and Configuration
FULL_SEGMENTS_FOLDER = "full_5_hits_tread_weak_person_a"
BASE_OUTPUT_FOLDER = "Processed_CSV_Files"
TRUCK_WHEEL_FOLDERS = {
    '6W': "6 wheels",
    '10W': "10 wheels",
    '12W': "12 wheels"
}

def count_files_in_directory(directory):
    """Count number of CSV files in a directory including the segments subfolder."""
    try:
        segments_dir = os.path.join(directory, "Strong-Tread Time-Domain Segments")
        return len([f for f in os.listdir(segments_dir) if f.endswith('.csv')])
    except FileNotFoundError:
        return 0

# Known pressures for filename parsing
KNOWN_PRESSURES = ['500', '600', '700', '800', '850', '900']

# Desired number of samples per hit for CSV
DESIRED_LENGTH = 256 # Upto 256 samples per hit

# Setup Logging
logging.basicConfig(
    filename='process_audio.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse the truck tire filename into its components.
    Format: TireNumber_TruckWheel-TireSize-POS{i}_Pressure_truckload_H
    """
    try:
        # Remove _H suffix and .wav extension
        base = filename.replace('_H.wav', '')
        
        # Split main components
        parts = base.split('_')
        if len(parts) < 4:
            raise ValueError(f"Invalid filename format: {filename}")
            
        tire_number = parts[0]
        
        # Parse wheel-size-position part
        wheel_parts = parts[1].split('-')
        if len(wheel_parts) < 3:
            raise ValueError(f"Invalid wheel-size-position format: {parts[1]}")
            
        truck_wheel = wheel_parts[0]
        position = wheel_parts[-1]
        tire_size = '-'.join(wheel_parts[1:-1])
        
        # Get pressure
        pressure = parts[2]

        # Get truck load without the .wav extension
        truck_load = parts[3].replace('.wav', '')
        
        # Determine tire type based on truck wheel
        tire_type = "702ZE-i" if truck_wheel == "10W" else ""
        
        return {
            'tire_number': tire_number,
            'truck_wheel': truck_wheel,
            'tire_size': tire_size,
            'position': position,
            'pressure': pressure,
            'truck_load': truck_load,
            'tire_type': tire_type
        }
    except Exception as e:
        raise ValueError(f"Error parsing filename {filename}: {str(e)}")

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

def save_to_csv(file_info: Dict[str, str], segments: List[np.ndarray], base_output_dir: str):
    """
    Save the processed segments to a CSV file in the appropriate directory structure:
    base_output_dir/[wheel_type]/Strong-Tread Time-Domain Segments/[csv_files]
    """
    # Create truck wheel directory if it exists in the data
    truck_wheel_dir = TRUCK_WHEEL_FOLDERS.get(file_info['truck_wheel'])
    if not truck_wheel_dir:
        raise ValueError(f"Invalid truck wheel type: {file_info['truck_wheel']}")
        
    # Create path including the segments subfolder
    wheel_dir = os.path.join(base_output_dir, truck_wheel_dir)
    segments_dir = os.path.join(wheel_dir, "Strong-Tread Time-Domain Segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    data = []
    for idx, seg in enumerate(segments, start=1):
        row = {
            "Segment ID / Value index": f"signal segment {idx}",
            "Tire Number": file_info['tire_number'],
            "Pressure": file_info['pressure'],
            "TireSize": file_info['tire_size'],
            "Tire_Type": file_info['tire_type'],
            "Truck_Load": file_info['truck_load']
        }
        # Add signal values
        for i, value in enumerate(seg, start=1):
            row[f"Signal Value {i}"] = value
        data.append(row)

    # Define column order
    signal_columns = [f"Signal Value {i}" for i in range(1, DESIRED_LENGTH + 1)]
    columns = ["Segment ID / Value index", "Tire Number", "Pressure", "TireSize", 
              "Tire_Type", "Truck_Load"] + signal_columns
    
    df = pd.DataFrame(data, columns=columns)

    # Create filename: Strong Tread Time Domain segment pressure-POS-{position}.csv
    filename = f"Strong Tread Time Domain segment {file_info['pressure']}-POS-{file_info['position']}.csv"
    csv_path = os.path.join(segments_dir, filename)
    df.to_csv(csv_path, index=False)
    
    return csv_path

def main():
    # Create base output directory only
    os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)

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

    # Track which truck wheel types we've seen
    seen_truck_wheels = set()

    # First pass - identify which wheel types exist in the data
    for full_segment_filename in full_segment_files:
        try:
            file_info = parse_filename(full_segment_filename)
            seen_truck_wheels.add(file_info['truck_wheel'])
        except ValueError:
            continue

    # Create only the needed directories with their segments subdirectory
    created_directories = {}
    for wheel_type in seen_truck_wheels:
        if wheel_type in TRUCK_WHEEL_FOLDERS:
            wheel_dir = os.path.join(BASE_OUTPUT_FOLDER, TRUCK_WHEEL_FOLDERS[wheel_type])
            segments_dir = os.path.join(wheel_dir, "Strong-Tread Time-Domain Segments")
            os.makedirs(segments_dir, exist_ok=True)
            created_directories[wheel_type] = wheel_dir

    # Track successful and failed processing
    successful_files = 0
    failed_files = 0

    # Process each full segment file with a progress bar
    for full_segment_filename in tqdm(full_segment_files, desc="Processing Full Segments"):
        try:
            # Parse file information
            file_info = parse_filename(full_segment_filename)
            logging.info(f"Processing file: {full_segment_filename}")
        except ValueError as ve:
            logging.warning(f"Skipping '{full_segment_filename}': {ve}")
            print(f"⚠️ Skipping '{full_segment_filename}': {ve}")
            failed_files += 1
            continue

        # Load audio samples
        file_path = os.path.join(FULL_SEGMENTS_FOLDER, full_segment_filename)
        try:
            sample_rate, samples = load_audio(file_path)
        except RuntimeError as e:
            logging.error(f"Failed to load '{full_segment_filename}': {e}")
            print(f"⚠️ {e}. Skipping.")
            failed_files += 1
            continue

        # Detect and segment hits
        try:
            hit_segments = detect_hits(samples, sample_rate, num_hits=5)
        except Exception as e:
            logging.warning(f"Error detecting hits in '{full_segment_filename}': {e}")
            failed_files += 1
            continue

        # Process each hit
        processed_hits = []
        for hit_segment in hit_segments:
            processed_hit = process_hit(samples, hit_segment)
            processed_hits.append(processed_hit)

        # Save to CSV in appropriate directory
        try:
            csv_path = save_to_csv(file_info, processed_hits, BASE_OUTPUT_FOLDER)
            logging.info(f"Saved processed hits to '{csv_path}'")
            successful_files += 1
        except Exception as e:
            logging.error(f"Failed to save CSV for '{full_segment_filename}': {e}")
            print(f"⚠️ Failed to save CSV for '{full_segment_filename}': {e}")
            failed_files += 1
            continue

    # Print summary with file counts
    print("\n✅ Processing complete.")
    print(f"Successfully processed: {successful_files} files")
    if failed_files > 0:
        print(f"Failed to process: {failed_files} files")
    
    print("\nProcessed files by truck wheel type:")
    for wheel_type in sorted(seen_truck_wheels):
        if wheel_type in TRUCK_WHEEL_FOLDERS:
            wheel_dir = os.path.join(BASE_OUTPUT_FOLDER, TRUCK_WHEEL_FOLDERS[wheel_type])
            file_count = count_files_in_directory(wheel_dir)
            print(f"  • {TRUCK_WHEEL_FOLDERS[wheel_type]}: {file_count} CSV files")
            logging.info(f"Generated {file_count} CSV files for {TRUCK_WHEEL_FOLDERS[wheel_type]}")

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
