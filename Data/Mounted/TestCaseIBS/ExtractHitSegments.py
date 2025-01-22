import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
from typing import Tuple, Dict

# Constants and Configuration
BASE_FOLDER = "IBS@Kanada"
METADATA_FILE = "IBS_TimeStamps_Metadata.csv"
FULL_SEGMENTS_OUTPUT_FOLDER = "IBS@Kanada_Segments"
os.makedirs(FULL_SEGMENTS_OUTPUT_FOLDER, exist_ok=True)

def find_audio_file(filename: str, base_folder: str) -> str:
    """
    Search for an audio file in the base folder and its subdirectories.
    
    Args:
        filename (str): The filename to search for
        base_folder (str): The base directory to start the search
        
    Returns:
        str: Full path to the audio file if found, None otherwise
    """
    for root, _, files in os.walk(base_folder):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Valid configurations
VALID_TRUCK_WHEELS = {'6W'}  # Modified to only include 6W as per example
VALID_POSITIONS = {f'POS{i}' for i in range(1, 13)}  # POS1 to POS12

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse the truck tire filename into its components.
    
    Format: TireNumber_TruckWheel-Unknown-POS{i}_Unknown_Empty_H
    Example: 1906_6W-Unknown-POS6_Unknown_Empty_H
    
    Args:
        filename (str): The filename without extension
        
    Returns:
        dict: Dictionary containing parsed components
        
    Raises:
        ValueError: If filename format is invalid
    """
    try:
        # Remove _H suffix if present
        if filename.endswith('_H'):
            filename = filename[:-2]
            
        # Split by underscore
        parts = filename.split('_')
        if len(parts) < 4:
            raise ValueError(f"Not enough parts in filename: {filename}")
            
        tire_number = parts[0]
        
        # Handle the wheel-position part (e.g., "6W-Unknown-POS6")
        wheel_pos = parts[1]
        wheel_parts = wheel_pos.split('-')
        if len(wheel_parts) < 3:
            raise ValueError(f"Invalid wheel-position format: {wheel_pos}")
            
        truck_wheel = wheel_parts[0]
        if truck_wheel not in VALID_TRUCK_WHEELS:
            raise ValueError(f"Invalid truck wheel type: {truck_wheel}")
            
        position = wheel_parts[-1]
        if position not in VALID_POSITIONS:
            raise ValueError(f"Invalid position: {position}")
        
        # Get truck load status
        truck_load = parts[3]
        
        return {
            'tire_number': tire_number,
            'truck_wheel': truck_wheel,
            'position': position,
            'truck_load': truck_load,
            'hitting_position': None  # Will be filled from metadata
        }
    except Exception as e:
        raise ValueError(f"Error parsing filename {filename}: {str(e)}")

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load and parse the metadata CSV file.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        
    Returns:
        pandas.DataFrame: Parsed metadata.
    """
    try:
        # Read CSV with more robust error handling
        metadata = pd.read_csv(
            metadata_path,
            dtype={
                'Filename': str,
                'HittingPosition': str,
                'Start_Time': float,
                'End_Time': float,
                'Duration': float
            },
            na_values=['', 'NA', 'NaN', 'null'],
            keep_default_na=True
        )
        
        required_columns = {'Filename', 'HittingPosition', 'Start_Time', 'End_Time'}
        if not required_columns.issubset(metadata.columns):
            missing = required_columns - set(metadata.columns)
            raise ValueError(f"Metadata file is missing columns: {missing}")
        
        # Clean up the data
        metadata['Filename'] = metadata['Filename'].str.strip()
        metadata['HittingPosition'] = metadata['HittingPosition'].str.strip()
        
        # Identify rows with missing timestamps
        missing_timestamps = metadata[metadata[['Start_Time', 'End_Time']].isna().any(axis=1)]
        if not missing_timestamps.empty:
            print(f"ℹ️ Found {len(missing_timestamps)} rows with missing timestamps:")
            for idx, row in missing_timestamps.iterrows():
                print(f"   - File: {row['Filename']}")
        
        # Drop rows with empty timestamps and create a new DataFrame
        clean_metadata = metadata.dropna(subset=['Start_Time', 'End_Time']).copy()
        
        # Validate filenames
        valid_files = []
        for filename in clean_metadata['Filename']:
            try:
                parse_filename(filename)
                valid_files.append(True)
            except ValueError as e:
                print(f"Warning: {e}")
                valid_files.append(False)
        
        clean_metadata.loc[:, 'valid_filename'] = valid_files
        
        return clean_metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {e}")

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load the WAV file using pydub and return raw samples as a NumPy array (mono).
    
    Args:
        file_path (str): Path to the WAV file.
        
    Returns:
        tuple: (audio samples as np.ndarray, sample rate)
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

def extract_and_save_segments(metadata: pd.DataFrame, input_folder: str, output_folder: str):
    """
    Extract full segments based on metadata and save them as WAV files.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame.
        input_folder (str): Directory containing the WAV files.
        output_folder (str): Directory to save the extracted segments.
    """
    # Drop any rows with NaN values in critical columns
    critical_columns = ['Filename', 'HittingPosition', 'Start_Time', 'End_Time']
    clean_metadata = metadata.dropna(subset=critical_columns)
    
    successful_extractions = 0
    failed_extractions = 0

    for _, row in tqdm(clean_metadata.iterrows(), total=len(clean_metadata), desc="Extracting Segments"):
        try:
            filename = str(row['Filename']).strip()
            start_time = float(row['Start_Time'])
            end_time = float(row['End_Time'])
            hitting_position = str(row['HittingPosition']).strip()
            
            # Parse filename and validate format
            try:
                file_info = parse_filename(filename)
                file_info['hitting_position'] = hitting_position
            except ValueError as e:
                print(f"⚠️ {e}. Skipping.")
                failed_extractions += 1
                continue
            
            # Search for the file in subdirectories
            wav_filename = filename + ".wav"
            file_path = find_audio_file(wav_filename, input_folder)
            
            if file_path is None:
                print(f"⚠️ File not found in any subdirectory: {wav_filename}. Skipping.")
                failed_extractions += 1
                continue
            
            samples, sample_rate = load_audio(file_path)
            
            # Calculate sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Validate sample indices
            if start_sample < 0 or end_sample > len(samples):
                print(f"⚠️ Invalid start/end times for {filename}. Skipping.")
                failed_extractions += 1
                continue
            
            # Extract segment
            segment = samples[start_sample:end_sample]
            
            # Normalize the segment
            max_val = np.max(np.abs(segment))
            if max_val > 0:
                normalized_segment = segment / max_val
            else:
                normalized_segment = segment
            
            # Convert to int16
            segment_int16 = np.int16(normalized_segment * 32767)
            
            # Create AudioSegment
            audio_segment = AudioSegment(
                segment_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=segment_int16.dtype.itemsize,
                channels=1
            )
            
            # Create output filename with parsed information
            output_filename = (f"{file_info['tire_number']}_{file_info['truck_wheel']}-"
                             f"Unknown-{file_info['position']}_{hitting_position}.wav")
            output_path = os.path.join(output_folder, output_filename)
            
            # Export as WAV
            audio_segment.export(output_path, format="wav")
            successful_extractions += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {str(e)}. Skipping.")
            failed_extractions += 1
            continue
    
    print(f"\n✅ Processing complete:")
    print(f"   Successful extractions: {successful_extractions}")
    print(f"   Failed extractions: {failed_extractions}")

def main():
    print(f"🔍 Starting audio processing...")
    print(f"📁 Base folder: {BASE_FOLDER}")
    print(f"📝 Metadata file: {METADATA_FILE}")
    print(f"📂 Output folder: {FULL_SEGMENTS_OUTPUT_FOLDER}\n")

    # Load metadata
    try:
        metadata = load_metadata(METADATA_FILE)
        print(f"✅ Successfully loaded metadata with {len(metadata)} entries\n")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return

    # Extract and save segments
    extract_and_save_segments(metadata, BASE_FOLDER, FULL_SEGMENTS_OUTPUT_FOLDER)
    print("✅ Segments extraction complete.")

if __name__ == "__main__":
    main()