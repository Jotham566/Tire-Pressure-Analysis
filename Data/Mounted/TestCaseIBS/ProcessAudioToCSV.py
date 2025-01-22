import os
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

# Constants and Configuration
FULL_SEGMENTS_FOLDER = "IBS@Kanada_Segments"
BASE_OUTPUT_FOLDER = "Processed_CSV_Files(IBS@Kanada)"
TRUCK_WHEEL_FOLDERS = {
    '6W': "6 wheels"
}

# Create dictionary for hit position folders
HIT_POSITION_FOLDERS = {
    'Tread': "Strong-Tread Time-Domain Segments",
    'Side': "Strong-Side Time-Domain Segments"
}

def count_files_in_directories(directory):
    """Count number of CSV files in all segment subdirectories."""
    counts = {}
    for hit_pos, folder_name in HIT_POSITION_FOLDERS.items():
        try:
            segments_dir = os.path.join(directory, folder_name)
            counts[hit_pos] = len([f for f in os.listdir(segments_dir) if f.endswith('.csv')])
        except FileNotFoundError:
            counts[hit_pos] = 0
    return counts

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
    Format: TireNumber_TruckWheel-Unknown-POS{i}_{HitPosition}.wav
    Example: 1902_6W-Unknown-POS2_Side.wav
    Returns dict with all metadata fields, empty strings for unknown values.
    """
    try:
        # Remove .wav extension
        base = filename.replace('.wav', '')
        
        # Split parts
        parts = base.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid filename format: {filename}")
            
        tire_number = parts[0]
        
        # Parse wheel-position part
        wheel_parts = parts[1].split('-')
        if len(wheel_parts) < 3:
            raise ValueError(f"Invalid wheel-position format: {parts[1]}")
            
        truck_wheel = wheel_parts[0]
        position = wheel_parts[-1]
        
        # Return all possible metadata fields with empty strings as defaults
        return {
            'tire_number': tire_number,
            'truck_wheel': truck_wheel,
            'position': position,
            'pressure': 'Unknown',
            'truck_load': '',  # Empty for unknown
            'tire_size': '',   # Empty for unknown
            'tire_type': '',   # Empty for unknown
            'rim_type': '',    # Empty for unknown
            'wear_level': ''   # Empty for unknown
        }
    except Exception as e:
        raise ValueError(f"Error parsing filename {filename}: {str(e)}")
        
        # Split main components
        parts = base.split('_')
        if len(parts) < 4:
            raise ValueError(f"Invalid filename format: {filename}")
            
        tire_number = parts[0]
        
        # Parse wheel-position part
        wheel_parts = parts[1].split('-')
        if len(wheel_parts) < 3:
            raise ValueError(f"Invalid wheel-position format: {parts[1]}")
            
        truck_wheel = wheel_parts[0]
        position = wheel_parts[-1]
        
        # Get truck load without the .wav extension
        truck_load = parts[3].replace('.wav', '')
        
        return {
            'tire_number': tire_number,
            'truck_wheel': truck_wheel,
            'position': position,
            'truck_load': truck_load,
            'pressure': 'Unknown'  # Set pressure as Unknown as per new format
        }
    except Exception as e:
        raise ValueError(f"Error parsing filename {filename}: {str(e)}")

def load_audio(file_path: str) -> Tuple[int, np.ndarray]:
    """Load the WAV file and return samples and sample rate."""
    try:
        sample_rate, data = wavfile.read(file_path)
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
        samples = data.astype(np.float32)
        return sample_rate, samples
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")

def detect_hits(samples: np.ndarray, sample_rate: int, num_hits: int = 5) -> List[Tuple[int, int]]:
    """Detect hits in the audio signal."""
    # Processing logic remains the same as your original code
    processed_signal = np.abs(samples) ** 2
    window_size = int(sample_rate * 0.005)
    smoothed_signal = np.convolve(processed_signal, np.ones(window_size)/window_size, mode='same')

    peaks, _ = find_peaks(smoothed_signal, 
                         height=np.mean(smoothed_signal), 
                         distance=int(sample_rate * 0.2),
                         prominence=np.std(smoothed_signal))

    if len(peaks) < num_hits:
        peaks, _ = find_peaks(smoothed_signal, 
                            height=np.mean(smoothed_signal) * 0.8, 
                            distance=int(sample_rate * 0.1),
                            prominence=np.std(smoothed_signal) * 2)

    if len(peaks) > num_hits:
        peak_prominences = np.zeros_like(peaks, dtype=float)
        for i, peak in enumerate(peaks):
            left_base = max(0, peak - int(sample_rate * 0.05))
            right_base = min(len(smoothed_signal), peak + int(sample_rate * 0.05))
            peak_prominences[i] = smoothed_signal[peak] - np.min(smoothed_signal[left_base:right_base])
        top_peak_indices = np.argsort(peak_prominences)[-num_hits:]
        peaks = peaks[top_peak_indices]

    hit_segments = []
    context_width = int(sample_rate * 0.05)
    for peak in peaks:
        start = max(0, peak - context_width)
        end = min(len(samples), peak + context_width)
        hit_segments.append((start, end))

    hit_segments = sorted(hit_segments, key=lambda x: x[0])
    non_overlapping_segments = []
    for segment in hit_segments:
        if non_overlapping_segments and segment[0] < non_overlapping_segments[-1][1]:
            segment = (non_overlapping_segments[-1][1], segment[1])
        non_overlapping_segments.append(segment)

    return non_overlapping_segments

def process_hit(samples: np.ndarray, hit_segment: Tuple[int, int], desired_length: int = 1024) -> np.ndarray:
    """Process a single hit segment."""
    hit = samples[hit_segment[0]:hit_segment[1]]
    hit_corrected = hit - np.mean(hit)
    hit_intensity = hit_corrected ** 2
    hit_normalized = hit_intensity / np.sum(hit_intensity)
    
    if len(hit_normalized) >= desired_length:
        hit_processed = hit_normalized[:desired_length]
    else:
        padding = desired_length - len(hit_normalized)
        hit_processed = np.pad(hit_normalized, (0, padding), 'constant')
    
    hit_final = hit_processed / np.sum(hit_processed)
    return hit_final

def save_to_csv(file_info: Dict[str, str], segments: List[np.ndarray], 
                base_output_dir: str, hitting_position: str) -> str:
    """
    Save the processed segments to a CSV file in the appropriate directory structure.
    """
    # Get the appropriate truck wheel directory
    truck_wheel_dir = TRUCK_WHEEL_FOLDERS.get(file_info['truck_wheel'])
    if not truck_wheel_dir:
        raise ValueError(f"Invalid truck wheel type: {file_info['truck_wheel']}")
    
    # Get the appropriate hitting position directory
    hit_pos_subdir = HIT_POSITION_FOLDERS.get(hitting_position)
    if not hit_pos_subdir:
        raise ValueError(f"Invalid hitting position: {hitting_position}")
    
    # Create full directory path
    wheel_dir = os.path.join(base_output_dir, truck_wheel_dir)
    segments_dir = os.path.join(wheel_dir, hit_pos_subdir)
    os.makedirs(segments_dir, exist_ok=True)
    
    # Prepare data for CSV
    data = []
    for idx, seg in enumerate(segments, start=1):
        row = {
            "Segment ID / Value index": f"signal segment {idx}",
            "Tire Number": file_info['tire_number'],
            "Pressure": file_info['pressure'],
            "Position": file_info['position'],
            "TireSize": file_info['tire_size'],
            "Tire_Type": file_info['tire_type'],
            "Truck_Load": file_info['truck_load'],
            "Rim_Type": file_info['rim_type'],
            "Wear_Level": file_info['wear_level']
        }
        # Add signal values
        for i, value in enumerate(seg, start=1):
            row[f"Signal Value {i}"] = value
        data.append(row)

    # Define column order
    metadata_columns = [
        "Segment ID / Value index", "Tire Number", "Pressure", "Position",
        "TireSize", "Tire_Type", "Truck_Load", "Rim_Type", "Wear_Level"
    ]
    signal_columns = [f"Signal Value {i}" for i in range(1, len(segments[0]) + 1)]
    columns = metadata_columns + signal_columns
    
    df = pd.DataFrame(data, columns=columns)

    # Create filename with the new format - include tire number
    filename = (f"Strong {hitting_position} Time Domain segment {file_info['pressure']}-"
               f"TireNum-{file_info['tire_number']}-{file_info['position']}.csv")
    csv_path = os.path.join(segments_dir, filename)
    df.to_csv(csv_path, index=False)
    
    return csv_path

def process_file_with_metadata(file_path: str, file_info: Dict[str, str], 
                             hitting_position: str, base_output_dir: str) -> bool:
    """Process a single audio file with its metadata."""
    try:
        # Load and process audio
        sample_rate, samples = load_audio(file_path)
        hit_segments = detect_hits(samples, sample_rate)
        processed_hits = [process_hit(samples, segment) for segment in hit_segments]
        
        # Save to CSV
        csv_path = save_to_csv(file_info, processed_hits, base_output_dir, hitting_position)
        logging.info(f"Successfully processed {file_path} to {csv_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False

def main():
    print("Starting audio processing...")
    
    # Load metadata
    metadata_path = "IBS_TimeStamps_Metadata.csv"
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"Loaded metadata with {len(metadata)} entries")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Create base output directory
    os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)

    # Process files
    successful_files = 0
    failed_files = 0
    
    # Group files by hitting position
    grouped_files = metadata.groupby('HittingPosition')

    for hitting_position, group in grouped_files:
        print(f"\nProcessing {hitting_position} hits...")
        
        # Convert hitting position to match filename format
        hit_pos_suffix = 'Side' if hitting_position.lower() == 'side' else 'Tread'
        
        for _, row in tqdm(group.iterrows(), total=len(group), 
                          desc=f"Processing {hitting_position} hits"):
            try:
                filename = row['Filename']
                if not isinstance(filename, str):
                    continue
                    
                # Parse file information
                file_info = parse_filename(filename)
                
                # Construct correct filename by removing "Unknown_Empty" part
                base_parts = filename.split('_')
                # Take first two parts and position, then add hitting position
                if len(base_parts) >= 4:
                    clean_filename = f"{base_parts[0]}_{base_parts[1]}_{hit_pos_suffix}.wav"
                    file_path = os.path.join(FULL_SEGMENTS_FOLDER, clean_filename)
                    if not os.path.exists(file_path):
                        logging.warning(f"File not found: {file_path}")
                        failed_files += 1
                        continue

                # Process the file
                if process_file_with_metadata(file_path, file_info, hitting_position, BASE_OUTPUT_FOLDER):
                    successful_files += 1
                else:
                    failed_files += 1

            except Exception as e:
                logging.error(f"Error processing row: {e}")
                failed_files += 1
                continue

    # Print summary
    print("\n✅ Processing complete:")
    print(f"Successfully processed: {successful_files} files")
    print(f"Failed to process: {failed_files} files")
    
    # Print file counts by truck wheel and hitting position
    print("\nProcessed files by truck wheel type and hitting position:")
    for wheel_type, wheel_folder in TRUCK_WHEEL_FOLDERS.items():
        wheel_dir = os.path.join(BASE_OUTPUT_FOLDER, wheel_folder)
        if os.path.exists(wheel_dir):
            counts = count_files_in_directories(wheel_dir)
            print(f"\n{wheel_folder}:")
            for hit_pos, count in counts.items():
                print(f"  • {hit_pos}: {count} CSV files")

if __name__ == "__main__":
    main()