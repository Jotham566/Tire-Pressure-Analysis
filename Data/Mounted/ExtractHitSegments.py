import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
from typing import Tuple, Dict

# Constants and Configuration
INPUT_FOLDER = "abcd@10輪車-打音_downsampled_8k"
METADATA_FILE = "BestCase_Side_Strong_TimeStamps_MetadataV2.csv"
FULL_SEGMENTS_OUTPUT_FOLDER = "abcd@10輪車-打音_5Hits_SideStrong_BestCase"
os.makedirs(FULL_SEGMENTS_OUTPUT_FOLDER, exist_ok=True)

# Valid configurations
VALID_TRUCK_WHEELS = {'6W', '10W', '12W'}
VALID_PRESSURES = {500, 600, 700, 800, 850, 900}
VALID_POSITIONS = {f'POS{i}' for i in range(1, 13)}  # POS1 to POS12

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse the truck tire filename into its components.
    
    Format: TireNumber_TruckWheel-TireSize-POS{i}_Pressure_truckload_H
    Example: 300_10W-11R22.5-POS1_900_Empty_H
    
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
        
        # Handle the wheel-size-position part (e.g., "10W-11R22.5-POS1")
        wheel_size_pos = parts[1]
        wheel_parts = wheel_size_pos.split('-')
        if len(wheel_parts) < 3:
            raise ValueError(f"Invalid wheel-size-position format: {wheel_size_pos}")
            
        truck_wheel = wheel_parts[0]
        if truck_wheel not in VALID_TRUCK_WHEELS:
            raise ValueError(f"Invalid truck wheel type: {truck_wheel}")
            
        # Get position (last part)
        position = wheel_parts[-1]
        if position not in VALID_POSITIONS:
            raise ValueError(f"Invalid position: {position}")
            
        # Get tire size (everything between truck_wheel and position)
        tire_size = '-'.join(wheel_parts[1:-1])
        
        # Get pressure from the third part
        try:
            pressure = int(parts[2])
            if pressure not in VALID_PRESSURES:
                raise ValueError(f"Invalid pressure value: {pressure}")
        except ValueError:
            raise ValueError(f"Invalid pressure format: {parts[2]}")
        
        # Get truck load status
        truck_load = parts[3]
        
        return {
            'tire_number': tire_number,
            'truck_wheel': truck_wheel,
            'tire_size': tire_size,
            'position': position,
            'pressure': pressure,
            'truck_load': truck_load,
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
                'Tire_Pressure': 'Int64',  # Allows for NA values
                'Start_Time': float,
                'End_Time': float
            },
            na_values=['', 'NA', 'NaN', 'null'],  # Extended NA values
            keep_default_na=True
        )
        
        required_columns = {'Filename', 'Tire_Pressure', 'Start_Time', 'End_Time'}
        if not required_columns.issubset(metadata.columns):
            missing = required_columns - set(metadata.columns)
            raise ValueError(f"Metadata file is missing columns: {missing}")
        
        # Clean up the data
        metadata['Filename'] = metadata['Filename'].str.strip()
        
        # Identify rows with missing timestamps
        missing_timestamps = metadata[metadata[['Start_Time', 'End_Time']].isna().any(axis=1)]
        if not missing_timestamps.empty:
            print(f"ℹ️ Found {len(missing_timestamps)} rows with missing timestamps:")
            for idx, row in missing_timestamps.iterrows():
                print(f"   - File: {row['Filename']}")
                if pd.isna(row['Start_Time']):
                    print(f"     Missing Start_Time")
                if pd.isna(row['End_Time']):
                    print(f"     Missing End_Time")
                print()
        
        # Drop rows with empty timestamps and create a new DataFrame
        clean_metadata = metadata.dropna(subset=['Start_Time', 'End_Time']).copy()
        
        # Count dropped rows
        dropped_rows = len(metadata) - len(clean_metadata)
        if dropped_rows > 0:
            print(f"ℹ️ Total rows dropped due to missing timestamps: {dropped_rows}")
        
        # Validate remaining filenames but don't drop them
        valid_files = []
        for filename in clean_metadata['Filename']:
            try:
                parse_filename(filename)
                valid_files.append(True)
            except ValueError as e:
                print(f"Warning: {e}")
                valid_files.append(False)
        
        # Use loc to set values
        clean_metadata.loc[:, 'valid_filename'] = valid_files
        
        invalid_count = len(valid_files) - sum(valid_files)
        if invalid_count > 0:
            print(f"ℹ️ Found {invalid_count} files with invalid naming format")
        
        return clean_metadata
        
    except pd.errors.EmptyDataError:
        raise RuntimeError("Metadata file is empty")
    except pd.errors.ParserError as e:
        raise RuntimeError(f"Failed to parse metadata CSV: {e}")
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
    critical_columns = ['Filename', 'Tire_Pressure', 'Start_Time', 'End_Time']
    clean_metadata = metadata.dropna(subset=critical_columns)
    
    # Get unique entries
    unique_entries = clean_metadata.drop_duplicates(subset=critical_columns)
    
    successful_extractions = 0
    failed_extractions = 0

    for _, row in tqdm(unique_entries.iterrows(), total=unique_entries.shape[0], desc="Extracting Full Segments"):
        try:
            filename = str(row['Filename']).strip()  # Ensure string and remove whitespace
            start_time = float(row['Start_Time'])
            end_time = float(row['End_Time'])
            
            # Basic validation
            if not filename or pd.isna(start_time) or pd.isna(end_time):
                print(f"⚠️ Invalid row data: {row}. Skipping.")
                failed_extractions += 1
                continue
                
            # Parse filename and validate format
            try:
                file_info = parse_filename(filename)
            except ValueError as e:
                print(f"⚠️ {e}. Skipping.")
                failed_extractions += 1
                continue
            
            file_path = os.path.join(input_folder, filename + ".wav")
            
            if not os.path.isfile(file_path):
                print(f"⚠️ File not found: {file_path}. Skipping.")
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
            if start_sample >= end_sample:
                print(f"⚠️ Start_Time >= End_Time for {filename}. Skipping.")
                failed_extractions += 1
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
            
            # Create output filename with parsed information
            output_filename = (f"{file_info['tire_number']}_{file_info['truck_wheel']}-"
                             f"{file_info['tire_size']}-{file_info['position']}_"
                             f"{file_info['pressure']}_{file_info['truck_load']}.wav")
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
    # Load metadata
    try:
        metadata = load_metadata(METADATA_FILE)
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return

    # Extract and save full segments
    extract_and_save_segments(metadata, INPUT_FOLDER, FULL_SEGMENTS_OUTPUT_FOLDER)
    print("✅ Full segments extraction complete.")

if __name__ == "__main__":
    main()