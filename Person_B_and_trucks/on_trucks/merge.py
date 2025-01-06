import os
import pandas as pd
from pathlib import Path

def merge_csv_files(folder_a, folder_b, output_folder):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get list of CSV files in folder A
    files_a = [f for f in os.listdir(folder_a) if f.endswith('.csv')]
    
    for file_a in files_a:
        # Construct full file paths
        path_a = os.path.join(folder_a, file_a)
        path_b = os.path.join(folder_b, file_a)
        
        # Check if corresponding file exists in folder B
        if not os.path.exists(path_b):
            print(f"Warning: No matching file found for {file_a} in Person B's folder")
            continue
        
        # Read both CSVs
        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        
        # Rename Person B's segments to continue the sequence
        segment_map = {
            'signal segment 1': 'signal segment 6',
            'signal segment 2': 'signal segment 7',
            'signal segment 3': 'signal segment 8',
            'signal segment 4': 'signal segment 9',
            'signal segment 5': 'signal segment 10'
        }
        
        # Replace segment names in Person B's data
        df_b['Segment ID / Value index'] = df_b['Segment ID / Value index'].replace(segment_map)
        
        # Concatenate the dataframes
        merged_df = pd.concat([df_a, df_b], ignore_index=True)
        
        # Save merged file
        output_path = os.path.join(output_folder, f'{file_a}')
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully merged {file_a}")

# Usage example
if __name__ == "__main__":
    # Define your folders here
    person_a_folder = "/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Person_A_and_trucks/Stand-Alone Tire/Processed_CSV_Files"
    person_b_folder = "/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Person_B_and_trucks/Stand-Alone Tire/Processed_CSV_Files"
    output_folder = "/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Merged"
    
    merge_csv_files(person_a_folder, person_b_folder, output_folder)