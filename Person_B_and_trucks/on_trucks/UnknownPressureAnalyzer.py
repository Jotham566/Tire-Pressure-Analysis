import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import yaml
from refactored_pw_calc import TireSoundProcessor, load_config

class UnknownPressureAnalyzer:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize analyzer with same configuration as main processor"""
        self.config = load_config(config_path)
        self.processor = TireSoundProcessor(self.config)
        self.intensity_thresholds = self.config['common']['intensity_thresholds']
        
    def process_unknown_files(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        """Process files with unknown pressure and compute pulse widths"""
        results = {str(threshold): [] for threshold in self.intensity_thresholds}
        processed_files = 0
        
        print(f"Processing files from directory: {input_dir}")
        
        for file_path in Path(input_dir).rglob('*.csv'):
            if file_path.name.startswith('~$'):  # Skip temporary files
                continue
                
            try:
                print(f"\nProcessing file: {file_path}")
                
                # Extract metadata from filename and path
                pos = self.extract_position(file_path.name)
                tire_num = self.extract_tire_number(file_path.name)
                hitting_type = self.extract_hitting_type(file_path)
                
                print(f"Extracted metadata - Position: {pos}, Tire Number: {tire_num}, Hitting Type: {hitting_type}")
                
                # Process file using existing methods
                step1, step2, _ = self.processor.process_file_step1_step2(file_path, mounted=True)
                
                # Calculate pulse widths for each threshold
                for threshold in self.intensity_thresholds:
                    first_increase_index = step2.apply(
                        lambda row: self.processor.get_first_increase_index(row, mounted=True),
                        axis=1
                    )
                    
                    point_exceeds_index = step2.apply(
                        lambda row: self.processor.get_point_exceeds_index(row, threshold),
                        axis=1
                    )
                    
                    pulse_width = point_exceeds_index - first_increase_index
                    median_pulse_width = pulse_width.median()
                    
                    results[str(threshold)].append({
                        'File_Name': file_path.name,
                        'Tire_Number': tire_num,
                        'Position': pos,
                        'Hitting_Type': hitting_type,
                        'Median_Pulse_Width': median_pulse_width
                    })
                
                processed_files += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print(f"\nProcessed {processed_files} files successfully")
        
        # Convert results to DataFrames and verify data
        result_dfs = {}
        for threshold, data in results.items():
            if not data:  # Check if we have any results
                print(f"No data collected for threshold {threshold}")
                continue
                
            df = pd.DataFrame(data)
            print(f"\nThreshold {threshold} DataFrame columns: {df.columns.tolist()}")
            print(f"Number of rows: {len(df)}")
            result_dfs[threshold] = df
            
        return result_dfs

    def visualize_results(self, results: Dict[str, pd.DataFrame], output_dir: str):
        """Create visualizations for each intensity threshold"""
        if not results:
            print("No results to visualize!")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # First, let's print what data we have
        print("\nPreparing to visualize results:")
        for threshold, df in results.items():
            print(f"\nThreshold {threshold}:")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Unique hitting types: {df['Hitting_Type'].unique().tolist()}")
        
        # Create subplot for each hitting type
        hitting_types = set()
        for df in results.values():
            hitting_types.update(df['Hitting_Type'].unique())
        
        for hitting_type in hitting_types:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Pulse Width Distribution by Position - {hitting_type} Hitting')
            
            for idx, threshold in enumerate(self.intensity_thresholds):
                if str(threshold) not in results:
                    continue
                    
                ax = axes[idx//2, idx%2]
                data = results[str(threshold)]
                data_filtered = data[data['Hitting_Type'] == hitting_type]
                
                if len(data_filtered) > 0:
                    sns.boxplot(
                        data=data_filtered,
                        x='Position',
                        y='Median_Pulse_Width',
                        ax=ax
                    )
                    
                ax.set_title(f'Intensity Threshold {threshold}')
                ax.set_ylabel('Median Pulse Width')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'pulse_width_distribution_{hitting_type}.png'))
            plt.close()
            
        # Save results to Excel
        with pd.ExcelWriter(os.path.join(output_dir, 'unknown_pressure_analysis.xlsx')) as writer:
            for threshold, df in results.items():
                df.to_excel(writer, sheet_name=f'Threshold_{threshold}', index=False)

    @staticmethod
    def extract_position(filename: str) -> str:
        """Extract position from filename"""
        import re
        match = re.search(r'POS(\d+)', filename)
        return match.group(1) if match else 'Unknown'
        
    @staticmethod
    def extract_tire_number(filename: str) -> str:
        """Extract tire number from filename"""
        import re
        match = re.search(r'TireNum-(\d+)', filename)
        return match.group(1) if match else 'Unknown'
        
    @staticmethod
    def extract_hitting_type(file_path: Path) -> str:
        """Extract hitting type from file path"""
        path_str = str(file_path).lower()
        if 'strong-side' in path_str:
            return 'Side'
        elif 'strong-tread' in path_str:
            return 'Tread'
        else:
            return 'Unknown'

def main():
    # Initialize analyzer
    analyzer = UnknownPressureAnalyzer()
    
    # Process files
    input_dir = '/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Data/Mounted/TestCaseIBS/Processed_CSV_Files(IBS@Kanada)'
    print("\nStarting analysis...")
    results = analyzer.process_unknown_files(input_dir)
    
    if not results:
        print("No results were generated. Check if input directory exists and contains CSV files.")
        return
        
    # Visualize results
    output_dir = "Unknown_Pressure_Analysis_Results"
    analyzer.visualize_results(results, output_dir)
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()