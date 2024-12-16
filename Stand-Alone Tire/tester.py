import os
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

# -----------------------------
# 📁 Constants and Configuration
# -----------------------------
INPUT_CSV_FOLDER = "processed_csv"       # Folder containing the CSV files
METADATA_FILE = "metadata.csv"           # Path to metadata.csv
RECONSTRUCTED_AUDIO_FOLDER = "reconstructed_audio"  # Output folder for WAV files
os.makedirs(RECONSTRUCTED_AUDIO_FOLDER, exist_ok=True)

# Sample rate as per the updated prompt
SAMPLE_RATE = 2000  # Hz

# -----------------------------
# 🔧 Helper Functions
# -----------------------------

def parse_csv_filename(csv_filename):
    """
    Extract tire number and pressure from the CSV filename.
    
    Example filename: "01_900.csv"
    - Tire number: "01"
    - Pressure: "900"
    
    Args:
        csv_filename (str): The name of the CSV file.
        
    Returns:
        tuple: (tire_num, pressure)
        
    Raises:
        ValueError: If the filename format is incorrect.
    """
    base = os.path.splitext(csv_filename)[0]  # e.g., "01_900"
    parts = base.split('_')
    if len(parts) != 2:
        raise ValueError(f"Unexpected CSV filename format: {csv_filename}")
    tire_num, pressure = parts
    return tire_num, pressure

def load_metadata(metadata_path):
    """
    Load and parse the metadata CSV file.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        
    Returns:
        pandas.DataFrame: Parsed metadata.
    """
    try:
        metadata = pd.read_csv(metadata_path)
        required_columns = {'Filename', 'Tire_Pressure', 'Start_Time', 'End_Time'}
        if not required_columns.issubset(metadata.columns):
            missing = required_columns - set(metadata.columns)
            raise ValueError(f"Metadata file is missing columns: {missing}")
        return metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {e}")

def reconstruct_signal(intensity_values):
    """
    Reconstruct an approximate audio signal from intensity values.
    
    Args:
        intensity_values (np.ndarray): Array of intensity values.
        
    Returns:
        np.ndarray: Reconstructed signal normalized to int16.
    """
    # Ensure intensity_values is a NumPy array of the correct type
    intensity_values = np.array(intensity_values, dtype=np.float64)
    
    # Apply square root to intensity values
    signal_approx = np.sqrt(intensity_values)
    
    return signal_approx

def save_wav(filename, signal, sample_rate=SAMPLE_RATE):
    """
    Save a numpy array as a WAV file.
    
    Args:
        filename (str): Path to save the WAV file.
        signal (np.ndarray): Audio signal as a numpy array.
        sample_rate (int): Sampling rate in Hz.
    """
    wavfile.write(filename, sample_rate, signal)

def combine_signals(signals):
    """
    Combine multiple audio signals into one by concatenation.
    
    Args:
        signals (list of np.ndarray): List of audio signals.
        
    Returns:
        np.ndarray: Combined audio signal.
    """
    return np.concatenate(signals)

def get_expected_duration(metadata, tire_num, pressure):
    """
    Retrieve the expected duration from metadata.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame.
        tire_num (str): Tire number.
        pressure (str): Tire pressure.
        
    Returns:
        float: Expected duration in seconds.
        
    Raises:
        ValueError: If metadata entry is not found.
    """
    # Construct the Filename as per metadata
    # Assuming Filename in metadata corresponds to CSV filename without pressure
    # Example: CSV '01_900.csv' corresponds to Filename '01_900' or similar
    # Adjust the pattern based on actual metadata
    # Here, assuming 'Filename' in metadata matches 'tire_num_pressure'
    matching_rows = metadata[
        (metadata['Tire_Pressure'].astype(str) == pressure) &
        (metadata['Filename'].str.startswith(tire_num + '_'))
    ]
    
    if matching_rows.empty:
        raise ValueError(f"No metadata entry found for Tire {tire_num} with Pressure {pressure}.")
    
    # If multiple entries, take the first one
    row = matching_rows.iloc[0]
    expected_duration = row['End_Time'] - row['Start_Time']
    return expected_duration

# -----------------------------
# 🚀 Main Processing Function
# -----------------------------

def reconstruct_audio_from_csv(selected_csvs=None):
    """
    Reconstruct audio segments from CSV files and verify durations.
    
    Args:
        selected_csvs (list of str, optional): List of CSV filenames to process.
                                               If None, process all CSVs in the folder.
    """
    # Load metadata
    try:
        metadata = load_metadata(METADATA_FILE)
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return
    
    # Get list of CSV files to process
    if selected_csvs is None:
        csv_files = [f for f in os.listdir(INPUT_CSV_FOLDER) if f.lower().endswith('.csv')]
    else:
        csv_files = selected_csvs
    
    if not csv_files:
        print("⚠️ No CSV files found to process.")
        return
    
    # Process each CSV file with a progress bar
    for csv_filename in tqdm(csv_files, desc="Reconstructing Audio Files"):
        try:
            tire_num, pressure = parse_csv_filename(csv_filename)
        except ValueError as ve:
            print(f"⚠️ Skipping '{csv_filename}': {ve}")
            continue
        
        # Get expected duration from metadata
        try:
            expected_duration = get_expected_duration(metadata, tire_num, pressure)
        except ValueError as ve:
            print(f"⚠️ Skipping '{csv_filename}': {ve}")
            continue
        
        # Load CSV data
        csv_path = os.path.join(INPUT_CSV_FOLDER, csv_filename)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️ Failed to read '{csv_filename}': {e}")
            continue
        
        # Check CSV structure
        if df.shape[0] != 5:
            print(f"⚠️ '{csv_filename}': Expected 5 segments, found {df.shape[0]}. Skipping.")
            continue
        if df.shape[1] != 513:
            print(f"⚠️ '{csv_filename}': Expected 513 columns, found {df.shape[1]}. Skipping.")
            continue
        
        # Reconstruct and save individual segments
        segment_signals = []
        for idx, row in df.iterrows():
            segment_id = row['Segment_ID']
            intensity_values = row[1:].values  # Exclude Segment_ID
            signal = reconstruct_signal(intensity_values)
            segment_signals.append(signal)
            
            # Save individual segment as WAV
            segment_filename = f"{tire_num}_{pressure}_Segment{segment_id}.wav"
            segment_path = os.path.join(RECONSTRUCTED_AUDIO_FOLDER, segment_filename)
            save_wav(segment_path, signal)
        
        # Combine segments into one audio
        combined_signal = combine_signals(segment_signals)
        combined_filename = f"{tire_num}_{pressure}_Combined.wav"
        combined_path = os.path.join(RECONSTRUCTED_AUDIO_FOLDER, combined_filename)
        save_wav(combined_path, combined_signal)
        
        # Calculate durations
        actual_duration = len(combined_signal) / SAMPLE_RATE  # in seconds
        expected_duration = round(expected_duration, 3)
        actual_duration = round(actual_duration, 3)
        
        # Display duration comparison
        print(f"\n📄 CSV: {csv_filename}")
        print(f"🔍 Expected Duration (End_Time - Start_Time): {expected_duration} seconds")
        print(f"🎧 Reconstructed Combined Duration: {actual_duration} seconds\n")
    
    print("✅ Audio reconstruction and verification complete.")

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
    # selected_csvs = ['01_900.csv', '02_900.csv', '03_900.csv']
    # reconstruct_audio_from_csv(selected_csvs)
    
    # To process all CSV files, pass None or omit the argument
    reconstruct_audio_from_csv()
