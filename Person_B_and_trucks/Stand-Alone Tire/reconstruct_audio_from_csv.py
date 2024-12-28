import os
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

# -----------------------------
# 📁 Constants and Configuration
# -----------------------------
INPUT_CSV_FOLDER = "/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Stand-Alone Tire/Processed_CSV_Files"
METADATA_FILE = "/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Stand-Alone Tire/Wavefile Data/metadata_uncleaned_wav.csv"
RECONSTRUCTED_AUDIO_FOLDER = "reconstructed_audio"
os.makedirs(RECONSTRUCTED_AUDIO_FOLDER, exist_ok=True)

# Sample rate and data configuration
SAMPLE_RATE = 2000  # Hz
METADATA_COLUMNS = 7  # Number of metadata columns to skip
EXPECTED_ROWS = 5
EXPECTED_COLS = 263  # Total columns including metadata

# -----------------------------
# 🔧 Helper Functions
# -----------------------------

def parse_csv_filename(csv_filename):
    """Extract tire number and pressure from the complex CSV filename format."""
    base = os.path.splitext(csv_filename)[0]
    match = re.match(r"(\d+)_(.*)", base)
    if not match:
        raise ValueError(f"Unexpected CSV filename format: {csv_filename}")
    tire_num, pressure_spec = match.groups()
    return tire_num, pressure_spec

def load_metadata(metadata_path):
    """Load and parse the metadata CSV file."""
    try:
        metadata = pd.read_csv(metadata_path)
        required_columns = {'Filename', 'Tire_Pressure', 'Start_Time', 'End_Time'}
        if not required_columns.issubset(metadata.columns):
            missing = required_columns - set(metadata.columns)
            raise ValueError(f"Metadata file is missing columns: {missing}")
        return metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {e}")

def get_expected_duration(metadata, tire_num, pressure_spec):
    """Get expected duration from metadata with complex pressure specifications."""
    matching_rows = metadata[
        (metadata['Filename'].str.startswith(f"{tire_num}_")) &
        (metadata['Tire_Pressure'] == pressure_spec)
    ]
    
    if matching_rows.empty:
        matching_rows = metadata[metadata['Filename'].str.startswith(f"{tire_num}_")]
        
        if matching_rows.empty:
            raise ValueError(f"No metadata entry found for Tire {tire_num} with Pressure {pressure_spec}")
            
        if len(matching_rows) > 1:
            try:
                current_pressure = int(re.search(r'-(\d+)-', pressure_spec).group(1))
                
                def extract_pressure(pressure_str):
                    match = re.search(r'-(\d+)-', pressure_str)
                    return int(match.group(1)) if match else None
                
                pressure_diffs = matching_rows['Tire_Pressure'].apply(
                    lambda x: abs(extract_pressure(x) - current_pressure) if extract_pressure(x) is not None else float('inf')
                )
                closest_match_idx = pressure_diffs.idxmin()
                matching_rows = matching_rows.loc[[closest_match_idx]]
            except (AttributeError, ValueError):
                matching_rows = matching_rows.iloc[[0]]
    
    row = matching_rows.iloc[0]
    return row['End_Time'] - row['Start_Time']

def reconstruct_signal(intensity_values):
    """Reconstruct an approximate audio signal from intensity values."""
    # Convert to float and handle any potential string values
    intensity_values = pd.to_numeric(intensity_values, errors='coerce')
    
    # Replace NaN values with 0
    intensity_values = np.nan_to_num(intensity_values, 0)
    
    # Ensure values are non-negative
    intensity_values = np.maximum(intensity_values, 0)
    
    # Apply square root to intensity values
    signal_approx = np.sqrt(intensity_values)
    
    # Normalize the signal
    if signal_approx.size > 0:
        max_val = np.max(np.abs(signal_approx))
        if max_val > 0:
            signal_approx = signal_approx / max_val
    
    # Convert to 16-bit integers
    signal_approx = (signal_approx * 32767).astype(np.int16)
    
    return signal_approx

def save_wav(filename, signal, sample_rate=SAMPLE_RATE):
    """Save a numpy array as a WAV file."""
    wavfile.write(filename, sample_rate, signal)

def combine_signals(signals):
    """Combine multiple audio signals into one by concatenation."""
    return np.concatenate(signals)

# -----------------------------
# 🚀 Main Processing Function
# -----------------------------

def reconstruct_audio_from_csv(selected_csvs=None):
    """Reconstruct audio segments from CSV files and verify durations."""
    try:
        metadata = load_metadata(METADATA_FILE)
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return
    
    csv_files = selected_csvs or [f for f in os.listdir(INPUT_CSV_FOLDER) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print("⚠️ No CSV files found to process.")
        return
    
    success_count = 0
    error_count = 0
    
    for csv_filename in tqdm(csv_files, desc="Reconstructing Audio Files"):
        try:
            tire_num, pressure_spec = parse_csv_filename(csv_filename)
            expected_duration = get_expected_duration(metadata, tire_num, pressure_spec)
            
            csv_path = os.path.join(INPUT_CSV_FOLDER, csv_filename)
            df = pd.read_csv(csv_path)
            
            if df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
                raise ValueError(f"Unexpected CSV dimensions: {df.shape}")
            
            segment_signals = []
            for idx, row in df.iterrows():
                # Skip metadata columns and process only numerical data
                intensity_values = row.iloc[METADATA_COLUMNS:].values
                signal = reconstruct_signal(intensity_values)
                segment_signals.append(signal)
                
                segment_filename = f"{tire_num}_{pressure_spec}_Segment{idx+1}.wav"
                segment_path = os.path.join(RECONSTRUCTED_AUDIO_FOLDER, segment_filename)
                save_wav(segment_path, signal)
            
            combined_signal = combine_signals(segment_signals)
            combined_filename = f"{tire_num}_{pressure_spec}_Combined.wav"
            combined_path = os.path.join(RECONSTRUCTED_AUDIO_FOLDER, combined_filename)
            save_wav(combined_path, combined_signal)
            
            actual_duration = len(combined_signal) / SAMPLE_RATE
            print(f"\n✅ Processed {csv_filename}")
            print(f"Expected Duration: {expected_duration:.3f}s")
            print(f"Actual Duration: {actual_duration:.3f}s")
            
            success_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing '{csv_filename}': {str(e)}")
            error_count += 1
            continue
    
    print(f"\n🎉 Processing complete!")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {error_count}")

# -----------------------------
# 🏁 Entry Point
# -----------------------------

if __name__ == "__main__":
    """
    Usage:
        - To process all CSV files in the 'processed_csv' folder:
            python reconstruct_audio.py
        - To process specific CSV files, modify the 'selected_csvs' list below.
    """
    
    # Example: To process specific CSV files, uncomment and modify the list below
    # selected_csvs = ['47_225-80R17.5-500-710R.csv', '39_11R22.5-600-903ZW.csv']
    # reconstruct_audio_from_csv(selected_csvs)
    
    # To process all CSV files, pass None or omit the argument
    reconstruct_audio_from_csv()