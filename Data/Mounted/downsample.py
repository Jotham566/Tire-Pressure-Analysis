import os
import numpy as np
import soundfile as sf
import plotly.graph_objs as go
import plotly.express as px
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd

def custom_downsample(input_path, output_path, target_sample_rate=2000, averaging_window=24):
    """
    Downsample WAV file by averaging chunks of data points.
    
    Args:
    input_path (str): Path to the input WAV file
    output_path (str): Path to save the downsampled WAV file
    target_sample_rate (int): Desired sample rate (default 2000 Hz)
    averaging_window (int): Number of data points to average (default 24) Calculated by dividing the original sample rate by the target sample rate (e.g., 48000 / 2000 = 24)
    """
    # Read the audio file using soundfile for numpy array
    data, sample_rate = sf.read(input_path)
    
    # Handle stereo by taking first channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Create chunks for averaging
    # Ensure we have complete chunks by truncating if necessary
    num_chunks = len(data) // averaging_window
    truncated_data = data[:num_chunks * averaging_window]
    
    # Reshape and average
    reshaped_data = truncated_data.reshape(-1, averaging_window)
    downsampled_data = reshaped_data.mean(axis=1)
    
    # Save downsampled audio
    sf.write(output_path, downsampled_data, target_sample_rate, subtype='PCM_24')

def batch_downsample_wav_files(input_folder, output_folder, target_sample_rate=2000, averaging_window=24):
    """
    Batch downsample WAV files in the input folder.
    
    Args:
    input_folder (str): Path to the folder containing input WAV files
    output_folder (str): Path to the folder where downsampled files will be saved
    target_sample_rate (int): Desired sample rate for downsampling (default 2000 Hz)
    averaging_window (int): Number of data points to average (default 24)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all WAV files in the input folder
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
    
    # List to store downsampling results for visualization
    downsample_results = []
    
    # Process each WAV file with tqdm progress bar
    for file_name in tqdm(wav_files, desc="Downsampling WAV Files"):
        # Full input file path
        input_path = os.path.join(input_folder, file_name)
        
        # Full output file path
        output_path = os.path.join(output_folder, file_name)
        
        try:
            # Read original audio to get sample rate and length
            original_audio = AudioSegment.from_wav(input_path)
            original_sample_rate = original_audio.frame_rate
            original_length = len(original_audio)
            
            # Perform custom downsampling
            custom_downsample(
                input_path, 
                output_path, 
                target_sample_rate=target_sample_rate,
                averaging_window=averaging_window
            )
            
            # Read downsampled audio
            downsampled_audio = AudioSegment.from_wav(output_path)
            
            # Store results for visualization
            downsample_results.append({
                'filename': file_name,
                'original_sample_rate': original_sample_rate,
                'downsampled_sample_rate': target_sample_rate,
                'original_length_ms': original_length,
                'downsampled_length_ms': len(downsampled_audio)
            })
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    return downsample_results

# Main execution
def main():
    input_folder = "/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Data/Mounted/abcd@10輪車-打音" # Folder containing WAV files
    output_folder = "abcd@10輪車-打音_downsampled_8k"  # Folder to save downsampled files
    
    # Perform downsampling
    results = batch_downsample_wav_files(
        input_folder, 
        output_folder, 
        target_sample_rate=8000,  # 8 kHz
        averaging_window=6 # 48000 / 8000 = 6    
    )

# Run the main function
if __name__ == "__main__":
    main()