import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# 📁 Constants and Configuration
INPUT_FOLDER = "/Users/jothamwambi/Projects/tire_analysis/Stand-Alone Tire/downsampled_2k"
METADATA_FILE = "/Users/jothamwambi/Projects/tire_analysis/Stand-Alone Tire/metadata.csv"
FULL_SEGMENTS_OUTPUT_FOLDER = "Full_Segments"
os.makedirs(FULL_SEGMENTS_OUTPUT_FOLDER, exist_ok=True)

# 🔧 Helper Functions

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

def load_audio(file_path):
    """
    Load the WAV file using pydub and return raw samples as a NumPy array (mono).
    
    Args:
        file_path (str): Path to the WAV file.
        
    Returns:
        np.ndarray: Audio samples as a 1D NumPy array.
    """
    try:
        audio = AudioSegment.from_wav(file_path)
        # Ensure mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        # Export to raw data and convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        return samples, audio.frame_rate
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")

def extract_and_save_segments(metadata, input_folder, output_folder):
    """
    Extract full segments based on metadata and save them as WAV files.
    (Person A's Tread Strong 5 Hits (WAV File).)
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame.
        input_folder (str): Directory containing the WAV files.
        output_folder (str): Directory to save the extracted segments.
    """
    unique_entries = metadata.drop_duplicates(subset=['Filename', 'Tire_Pressure', 'Start_Time', 'End_Time'])
    
    for _, row in tqdm(unique_entries.iterrows(), total=unique_entries.shape[0], desc="Extracting Full Segments"):
        filename = row['Filename']
        pressure = row['Tire_Pressure']
        start_time = row['Start_Time']
        end_time = row['End_Time']
        
        file_path = os.path.join(input_folder, filename + ".wav")
        
        if not os.path.isfile(file_path):
            print(f"⚠️ File not found: {file_path}. Skipping.")
            continue
        
        try:
            samples, sample_rate = load_audio(file_path)
        except RuntimeError as e:
            print(f"⚠️ {e}. Skipping.")
            continue
        
        # Calculate sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Validate sample indices
        if start_sample < 0 or end_sample > len(samples):
            print(f"⚠️ Invalid start/end times for {filename}. Skipping.")
            continue
        if start_sample >= end_sample:
            print(f"⚠️ Start_Time >= End_Time for {filename}. Skipping.")
            continue
        
        # Extract segment
        segment = samples[start_sample:end_sample]
        
        # Normalize the segment to prevent clipping
        max_val = np.max(np.abs(segment))
        if max_val == 0:
            normalized_segment = segment
        else:
            normalized_segment = segment / max_val
        
        # Convert to int16
        segment_int16 = np.int16(normalized_segment * 32767)
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            segment_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=segment_int16.dtype.itemsize,
            channels=1
        )
        
        # Define output filename
        output_filename = f"{filename}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        # Export as WAV
        audio_segment.export(output_path, format="wav")

def main():
    # Load metadata
    try:
        metadata = load_metadata(METADATA_FILE)
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return
    
    # Extract and save full segments
    extract_and_save_segments(metadata, INPUT_FOLDER, FULL_SEGMENTS_OUTPUT_FOLDER)
    
    print("✅ Full segments extraction complete.")

# 🏁 Entry Point
if __name__ == "__main__":
    main()
