import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import yaml
import logging
from typing import Tuple

def load_config(config_file: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file, falling back to defaults if needed.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            pwc_config = config.get('pulse_width_calculator', {})
            
            return {
                'common': {
                    'intensity_thresholds': pwc_config.get('intensity_thresholds', [0.5, 0.7, 0.8, 0.9])
                },
                'mounted': pwc_config.get('mounted', {}),
                'standalone': pwc_config.get('standalone', {})
            }
    except Exception as e:
        print(f"Using default parameters ({str(e)})")
        return get_default_config()

def get_default_config() -> dict:
    """
    Returns default configuration values.
    """
    return {
        'common': {
            'intensity_thresholds': [0.5, 0.7, 0.8, 0.9]
        },
        'mounted': {
            'trim_signal': False,
            'trim_dims_after_rise': 32,
             'noise_threshold': 0.03,
            'baseline_window_size': 10,
            'std_dev_multiplier': 3.0,
            'min_threshold_percentage': 0.01,
            'sustained_rise_points': 3,
            'lookback_window_size': 3,
            'baseline_subtraction_method': 'adaptive',
            'fixed_baseline_value': 0.01,
            'baseline_computation': 'median',
            'sliding_window_min_size': 5,
            'sliding_window_max_size': 20,
            'quietness_metric': 'std',
            'volatility_threshold': 0.5,
            'high_volatility_percentile': 75,
            'low_volatility_percentile': 25,
            'max_threshold_factor': 0.3
        },
        'standalone': {
            'trim_signal': False,
            'trim_dims_after_rise': 32,
            'baseline_window_size': 10,
            'std_dev_multiplier': 3.0,
            'min_threshold_percentage': 0.01,
            'sustained_rise_points': 3,
            'lookback_window_size': 3,
            'baseline_subtraction_method': 'adaptive',
            'fixed_baseline_value': 0.01,
            'baseline_computation': 'median',
            'sliding_window_min_size': 5,
            'sliding_window_max_size': 20,
            'quietness_metric': 'std',
            'volatility_threshold': 0.5,
            'high_volatility_percentile': 75,
            'low_volatility_percentile': 25,
            'max_threshold_factor': 0.3
        }
    }

class TireSoundProcessor:
    def __init__(self, input_dir: str, standalone_dir: str, output_dir: str,
             # Common parameters
             intensity_thresholds: list = [0.5, 0.7, 0.8, 0.9],
             # Mounted tire parameters
             noise_threshold: float = 0.03,
             trim_signal: bool = False,
             trim_dims_after_rise: int = 32,
            baseline_window_size_mounted: int = 10,
            std_dev_multiplier_mounted: float = 3.0,
            min_threshold_percentage_mounted: float = 0.01,
            sustained_rise_points_mounted: int = 3,
            lookback_window_size_mounted: int = 3,
            baseline_subtraction_method_mounted: str = "adaptive",
            fixed_baseline_value_mounted: float = 0.01,
            baseline_computation_mounted: str = 'median',
            sliding_window_min_size_mounted: int = 5,
            sliding_window_max_size_mounted: int = 20,
            quietness_metric_mounted: str = 'std',
            volatility_threshold_mounted: float = 0.5,
            high_volatility_percentile_mounted: int = 75,
            low_volatility_percentile_mounted: int = 25,
            max_threshold_factor_mounted: float = 0.3,
             # Standalone parameters
             trim_signal_standalone: bool = False,
             trim_dims_after_rise_standalone: int = 32,
             baseline_window_size: int = 10,
             std_dev_multiplier: float = 3.0,
             min_threshold_percentage: float = 0.01,
             sustained_rise_points: int = 3,
             lookback_window_size: int = 3,
             baseline_subtraction_method: str = "adaptive",
             fixed_baseline_value: float = 0.01,
             baseline_computation: str = 'median',
             sliding_window_min_size: int = 5,
             sliding_window_max_size: int = 20,
             quietness_metric: str = 'std',
             volatility_threshold: float = 0.5,
             high_volatility_percentile: int = 75,
             low_volatility_percentile: int = 25,
             max_threshold_factor: float = 0.3
             ):
        
        # Common parameters
        self.intensity_thresholds = intensity_thresholds
        
        # Mounted tire parameters
        self.noise_threshold = noise_threshold
        self.trim_signal = trim_signal
        self.trim_dims_after_rise = trim_dims_after_rise
        self.baseline_window_size_mounted = baseline_window_size_mounted
        self.std_dev_multiplier_mounted = std_dev_multiplier_mounted
        self.min_threshold_percentage_mounted = min_threshold_percentage_mounted
        self.sustained_rise_points_mounted = sustained_rise_points_mounted
        self.lookback_window_size_mounted = lookback_window_size_mounted
        self.baseline_subtraction_method_mounted = baseline_subtraction_method_mounted.lower()
        self.fixed_baseline_value_mounted = fixed_baseline_value_mounted
        self.baseline_computation_mounted = baseline_computation_mounted
        self.sliding_window_min_size_mounted = sliding_window_min_size_mounted
        self.sliding_window_max_size_mounted = sliding_window_max_size_mounted
        self.quietness_metric_mounted = quietness_metric_mounted
        self.volatility_threshold_mounted = volatility_threshold_mounted
        self.high_volatility_percentile_mounted = high_volatility_percentile_mounted
        self.low_volatility_percentile_mounted = low_volatility_percentile_mounted
        self.max_threshold_factor_mounted = max_threshold_factor_mounted
        
        # Standalone parameters
        self.trim_signal_standalone = trim_signal_standalone
        self.trim_dims_after_rise_standalone = trim_dims_after_rise_standalone
        self.baseline_window_size = baseline_window_size
        self.std_dev_multiplier = std_dev_multiplier
        self.min_threshold_percentage = min_threshold_percentage
        self.sustained_rise_points = sustained_rise_points
        self.lookback_window_size = lookback_window_size
        self.baseline_subtraction_method = baseline_subtraction_method.lower()
        self.fixed_baseline_value = fixed_baseline_value
        self.baseline_computation = baseline_computation
        self.sliding_window_min_size = sliding_window_min_size
        self.sliding_window_max_size = sliding_window_max_size
        self.quietness_metric = quietness_metric
        self.volatility_threshold = volatility_threshold
        self.high_volatility_percentile = high_volatility_percentile
        self.low_volatility_percentile = low_volatility_percentile
        self.max_threshold_factor = max_threshold_factor

        # Directory paths
        self.input_dir = Path(input_dir)
        self.standalone_dir = Path(standalone_dir)
        self.output_dir = Path(output_dir)
        
        # Output directories
        self.processed_dir = self.output_dir / 'Processed_Mounted'
        self.processed_standalone_dir = self.output_dir / 'Processed_Standalone'

        # Create all output directories at initialization
        for directory in [self.processed_dir, self.processed_standalone_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename='processing.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()

    def extract_air_pressure(self, file_name: str) -> float:
        match = re.search(r'segment\s*(\d+)', file_name, re.IGNORECASE)
        return float(match.group(1)) if match else np.nan

    def extract_tire_position(self, file_name: str) -> str:
        match = re.search(r'POS-?(\d+)', file_name, re.IGNORECASE)
        return match.group(1) if match else 'Unknown'

    def extract_vehicle_type(self, file_path: Path) -> str:
        current_path = file_path.parent
        while current_path != self.input_dir.parent:
            dir_name = current_path.name.lower()
            if '6 wheels' in dir_name or '6w' in dir_name:
                return '6W'
            elif '10 wheels' in dir_name or '10w' in dir_name:
                return '10W'
            elif '12 wheels' in dir_name or '12w' in dir_name:
                return '12W'
            current_path = current_path.parent

        file_name = file_path.name.lower()
        if '6 wheels' in file_name or '6w' in file_name:
            return '6W'
        elif '10 wheels' in file_name or '10w' in file_name:
            return '10W'
        elif '12 wheels' in file_name or '12w' in file_name:
            return '12W'
        else:
            return 'Unknown'

    def _save_to_excel(self, file_path: Path, data_dict: dict, sheet_order: list = None):
        """
        Helper method to save multiple dataframes to Excel file.
        
        Args:
            file_path: Path to save Excel file
            data_dict: Dictionary of sheet_name: dataframe pairs
            sheet_order: Optional list to specify sheet order
        """
        try:
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                if sheet_order:
                    # Save sheets in specified order
                    for sheet_name in sheet_order:
                        if sheet_name in data_dict:
                            data_dict[sheet_name].to_excel(writer, sheet_name=sheet_name)
                else:
                    # Save sheets in arbitrary order
                    for sheet_name, df in data_dict.items():
                        df.to_excel(writer, sheet_name=sheet_name)
                        
            return True
        except Exception as e:
            self.logger.error(f"Error saving Excel file {file_path}: {str(e)}")
            return False

    def _process_mounted_file(self, file_path: Path, output_paths: dict):
        """
        Helper method to process a single mounted tire file.
        
        Args:
            file_path: Input CSV file path
            output_paths: Dictionary with 'full' output path
        """
        try:
            # Process base data
            step1_normalized, step2_cumulative = self.process_file_step1_step2(file_path)

            # Apply baseline subtraction
            if self.baseline_subtraction_method_mounted == "adaptive":
                self.logger.info("Applying adaptive baseline subtraction for mounted data")
                step1_normalized = self.apply_adaptive_baseline_subtraction(step1_normalized, mounted=True)
            elif self.baseline_subtraction_method_mounted == "fixed":
                self.logger.info(f"Applying fixed baseline subtraction: {self.fixed_baseline_value_mounted} for mounted data")
                step1_normalized = self.apply_fixed_baseline_subtraction(step1_normalized, mounted=True)
            else:  # "none" or any other value
                self.logger.info("No baseline subtraction applied for mounted data")

            # Normalize after baseline subtraction
            if self.baseline_subtraction_method_mounted in ["adaptive", "fixed"]:
                step1_normalized = step1_normalized.div(step1_normalized.sum(axis=1).replace(0, 1), axis=0)

            # Apply signal trimming if enabled
            if self.trim_signal:
                self.logger.info(f"Trimming signals to {self.trim_dims_after_rise} dims after rise point")
                step1_normalized = self.trim_signal_to_rise(step1_normalized)
                step1_normalized = step1_normalized.div(step1_normalized.sum(axis=1).replace(0, 1), axis=0)
                step2_cumulative = step1_normalized.cumsum(axis=1)

            # Prepare full data sheets
            full_data = {
                'Step1_Data': step1_normalized,
                'Step2_Sj': step2_cumulative
            }
            
            # Add Step3 sheets for full data
            for threshold in self.intensity_thresholds:
                step3_data = self.compute_step3_metrics(step2_cumulative, threshold, file_path)
                full_data[f'Step3_DataPts_{threshold}'] = step3_data


            # Save full files
            full_success = self._save_to_excel(output_paths['full'], full_data)
            
            if not full_success:
                raise Exception("Failed to save one or more output files")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def trim_signal_to_rise(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Trims signal data preserving original numbering and handling multiple segments.
        Finds earliest rise point across all segments and trims accordingly.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            
        Returns:
            pd.DataFrame: Trimmed signal data with preserved numbering
        """
        # Find rise points for all segments
        rise_points = {}
        earliest_rise = float('inf')
        
        # First pass: find all rise points and the earliest one
        for idx in df_signals.index:
            row_values = df_signals.loc[idx].values
            dynamic_threshold = self.calculate_dynamic_noise_threshold(pd.Series(row_values), mounted=True)
            rise_indices = np.where(row_values > dynamic_threshold)[0]
            
            if len(rise_indices) > 0:
                rise_point = rise_indices[0]
                rise_points[idx] = rise_point
                earliest_rise = min(earliest_rise, rise_point)
            else:
                self.logger.warning(f"No rise point found for segment {idx}")
                rise_points[idx] = None
        
        if earliest_rise == float('inf'):
            self.logger.warning("No valid rise points found in any segment")
            return df_signals
        
        # Calculate trimming boundaries
        trim_start = max(0, earliest_rise - 2)  # Keep 2 points before earliest rise
        
        # Create new DataFrame with preserved numbering
        df_trimmed = pd.DataFrame(index=df_signals.index)
        
        # Calculate the full range of columns needed
        max_endpoint = 0
        for idx, rise_point in rise_points.items():
            if rise_point is not None:
                endpoint = rise_point + self.trim_dims_after_rise
                max_endpoint = max(max_endpoint, endpoint)
        
        # Create columns with preserved numbering
        columns = [f'Signal_Value_{i+1}' for i in range(trim_start, max_endpoint)]
        df_trimmed = df_trimmed.reindex(columns=columns)
        
        # Second pass: process each segment
        for idx in df_signals.index:
            row_values = df_signals.loc[idx].values
            rise_point = rise_points[idx]
            
            if rise_point is not None:
                # Create the trimmed values array
                trimmed_values = np.zeros(len(columns))
                
                # Fill in values after this segment's rise point
                signal_end = min(rise_point + self.trim_dims_after_rise, len(row_values))
                values_to_keep = row_values[rise_point:signal_end]
                
                # Calculate where to place these values in the trimmed array
                start_idx = rise_point - trim_start
                end_idx = start_idx + len(values_to_keep)
                
                # Place the values
                trimmed_values[start_idx:end_idx] = values_to_keep
                df_trimmed.loc[idx] = trimmed_values
            else:
                # If no rise point found, fill with zeros
                df_trimmed.loc[idx] = np.zeros(len(columns))
        
        self.logger.info(f"Trimmed signals from {trim_start+1} to {max_endpoint}")
        self.logger.info(f"Earliest rise point: {earliest_rise+1}")
        
        return df_trimmed

    def find_adaptive_baseline_offset(self, row_values: np.ndarray, mounted:bool = False) -> float:
        """
        Finds a baseline offset using a sliding window approach.
        Scans subwindows near the start of the row to find the "quietest" segment.
        Returns the median of that chosen subwindow.
        """
        if mounted:
            n_samples = len(row_values)
            min_size = min(self.sliding_window_min_size_mounted, n_samples)
            max_size = min(self.sliding_window_max_size_mounted, n_samples)
        else:
            n_samples = len(row_values)
            min_size = min(self.sliding_window_min_size, n_samples)
            max_size = min(self.sliding_window_max_size, n_samples)


        best_metric = float('inf')
        best_offset = 0.0

        # Slide from index 0 up to max_size - min_size
        for start_idx in range(0, max_size - min_size + 1):
            end_idx = start_idx + min_size  # subwindow length = min_size
            subwindow = row_values[start_idx:end_idx]

            # Compute the baseline offset candidate (median or mean)
            if (mounted and self.baseline_computation_mounted.lower() == 'median') or (not mounted and self.baseline_computation.lower() == 'median'):
                offset_candidate = np.median(subwindow)
            else:
                offset_candidate = np.mean(subwindow)
            
            # Evaluate "quietness" based on self.quietness_metric
            if (mounted and self.quietness_metric_mounted.lower() == 'std') or (not mounted and self.quietness_metric.lower() == 'std'):
                metric_val = np.std(subwindow)
            elif (mounted and self.quietness_metric_mounted.lower() == 'mad') or (not mounted and self.quietness_metric.lower() == 'mad'):
                metric_val = np.median(np.abs(subwindow - np.median(subwindow)))
            else:
                # Example: a mix, or fallback to std
                metric_val = np.std(subwindow)

            # If this subwindow is quieter, update best
            if metric_val < best_metric:
                best_metric = metric_val
                best_offset = offset_candidate

        return best_offset

    def apply_fixed_baseline_subtraction(self, df_signals: pd.DataFrame, mounted: bool = False) -> pd.DataFrame:
        """
        Applies a fixed baseline subtraction to the signal data.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            
        Returns:
            pd.DataFrame: Signal data with fixed baseline subtracted and negative values clipped to zero
        """
        df_modified = df_signals.copy()
        
        # Subtract fixed value
        if mounted:
          df_modified = df_modified - self.fixed_baseline_value_mounted
        else:
          df_modified = df_modified - self.fixed_baseline_value
        
        # Zero-clip negative values
        df_modified = df_modified.clip(lower=0)
        
        return df_modified

    def apply_adaptive_baseline_subtraction(self, df_signals: pd.DataFrame, mounted: bool = False) -> pd.DataFrame:
        """
        For each row in df_signals, find a baseline offset using a sliding
        window "quietness" check. Subtract that offset and zero-clip negatives.
        """
        df_modified = df_signals.copy()
        
        for idx in df_modified.index:
            row_values = df_modified.loc[idx].values  # to numpy array
            offset = self.find_adaptive_baseline_offset(row_values, mounted)
            
            # Subtract offset
            row_values_sub = row_values - offset
            # Zero-clip negative
            row_values_sub = np.where(row_values_sub < 0, 0, row_values_sub)
            
            # Update the dataframe
            df_modified.loc[idx] = row_values_sub

        return df_modified

    def process_file_step1_step2(self, file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Processes a single CSV file by extracting metadata and signal values,
        normalizing (Step1_Data), and creating the cumulative sum (Step2_Sj).
        Considers configuration parameters for processing.
        """
        try:
            # Read the CSV file with the first row as header
            df = pd.read_csv(file_path)
            
            # Extract metadata columns
            metadata_columns = [
                'Segment ID / Value index', 'Tire Number', 'Pressure', 
                'TireSize', 'Tire_Type', 'Truck_Load'
            ]
            metadata = df[metadata_columns].copy()
            
            # Extract signal data (columns after metadata)
            signal_columns = [col for col in df.columns if col not in metadata_columns]
            df_signals = df[signal_columns].copy()
            
            # Convert signal values to numeric
            df_signals = df_signals.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Set index using Segment ID
            df_signals.index = metadata['Segment ID / Value index']
            
            # Apply baseline subtraction based on configuration
            if self.baseline_subtraction_method_mounted == "adaptive":
                self.logger.info("Applying adaptive baseline subtraction for mounted data")
                df_signals = self.apply_adaptive_baseline_subtraction(df_signals, mounted=True)
            elif self.baseline_subtraction_method_mounted == "fixed":
                self.logger.info(f"Applying fixed baseline subtraction: {self.fixed_baseline_value_mounted}")
                df_signals = self.apply_fixed_baseline_subtraction(df_signals, mounted=True)
            
            # Apply signal trimming if enabled in config
            if self.trim_signal:
                self.logger.info(f"Trimming signals to {self.trim_dims_after_rise} dimensions after rise point")
                df_signals = self.trim_signal_to_rise(df_signals)
            
            # Step1: Normalize so sum of each row is 1
            df_step1 = df_signals.div(df_signals.sum(axis=1).replace(0, 1), axis=0)
            
            # Step2: Cumulative Sum
            df_step2 = df_step1.cumsum(axis=1)
            
            # Set index names
            df_step1.index.name = 'Segment_ID'
            df_step2.index.name = 'Segment_ID'
            
            # Store metadata for later use
            metadata_dict = metadata.set_index('Segment ID / Value index').to_dict('index')
            
            return df_step1, df_step2, metadata_dict

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise Exception(f"Failed to process CSV: {e}")

    def get_first_increase_index(self, row: pd.Series, noise_threshold: float = None, mounted:bool = False) -> float:
      
      if mounted:
        dynamic_threshold = self.calculate_dynamic_noise_threshold(row, mounted=True)
        indices = np.where(row.values > dynamic_threshold)[0]
      else:
        indices = np.where(row.values > (noise_threshold if noise_threshold is not None else self.noise_threshold))[0]

      return indices[0] + 1 if indices.size > 0 else np.nan

    def get_point_exceeds_index(self, row: pd.Series, threshold: float) -> float:
        indices = np.where(row.values > threshold)[0]
        return indices[0] + 1 if indices.size > 0 else np.nan

    def get_cumulative_value_at_index(self, row: pd.Series, idx: float):
        if pd.isna(idx):
            return np.nan
        idx = int(idx) - 1
        if 0 <= idx < len(row):
            return row.iloc[idx]
        else:
            return np.nan

    def compute_step3_metrics(self, df_cumulative: pd.DataFrame, 
                         intensity_threshold: float, 
                         file_path: Path,
                         metadata_dict: dict) -> pd.DataFrame:
        """
        Compute Step3 metrics including metadata from the original file.
        """
        air_pressure = self.extract_air_pressure(file_path.name)
        tire_position = self.extract_tire_position(file_path.name)
        wheel_type = self.extract_vehicle_type(file_path)
        hitting_type = self.determine_hitting_type(file_path)

        first_increase_index = df_cumulative.apply(
            lambda row: self.get_first_increase_index(row, self.noise_threshold, mounted=True), 
            axis=1
        )
        point_exceeds_index = df_cumulative.apply(
            lambda row: self.get_point_exceeds_index(row, threshold=intensity_threshold), 
            axis=1
        )

        first_increase_cumulative_value = df_cumulative.apply(
            lambda row: self.get_cumulative_value_at_index(row, first_increase_index[row.name]), 
            axis=1
        )
        point_exceeds_cumulative_value = df_cumulative.apply(
            lambda row: self.get_cumulative_value_at_index(row, point_exceeds_index[row.name]), 
            axis=1
        )

        pulse_width = point_exceeds_index - first_increase_index

        # Create DataFrame with metrics and metadata
        step3_data_points = pd.DataFrame({
            'Intensity_Threshold': intensity_threshold,
            'First_Noticeable_Increase_Index': first_increase_index,
            'Point_Exceeds_Index': point_exceeds_index,
            'First_Noticeable_Increase_Cumulative_Value': first_increase_cumulative_value,
            'Point_Exceeds_Cumulative_Value': point_exceeds_cumulative_value,
            'Pulse_Width': pulse_width,
            'Air_Pressure': air_pressure,
            'Tire_Position': tire_position,
            'Wheel_Type': wheel_type,
            'Hitting_Type': hitting_type
        }, index=df_cumulative.index)

        # Add original metadata
        for idx in step3_data_points.index:
            if idx in metadata_dict:
                metadata = metadata_dict[idx]
                step3_data_points.loc[idx, 'Tire Number'] = metadata['Tire Number']
                step3_data_points.loc[idx, 'Pressure'] = metadata['Pressure']
                step3_data_points.loc[idx, 'TireSize'] = metadata['TireSize']
                step3_data_points.loc[idx, 'Tire_Type'] = metadata['Tire_Type']
                step3_data_points.loc[idx, 'Truck_Load'] = metadata['Truck_Load']

        return step3_data_points

    def calculate_median_pulse_width(self, excel_file_paths) -> pd.DataFrame:
        median_pulse_widths = []
        prefix = 'Step3_DataPts'

        for file_path in excel_file_paths:
            try:
                if file_path.name.startswith('~$'):
                    continue
                xls = pd.ExcelFile(file_path)

                for threshold in self.intensity_thresholds:
                    sheet_name = f'{prefix}_{threshold}'
                    if sheet_name not in xls.sheet_names:
                        reason = f'Sheet {sheet_name} not found'
                        self.add_exclusion_entry(file_path, reason)
                        self.logger.warning(f"{file_path.name}: {reason}")
                        continue

                    df_step3 = pd.read_excel(file_path, sheet_name=sheet_name, index_col='Segment_ID')

                    air_pressure = df_step3['Air_Pressure'].iloc[0] if len(df_step3) > 0 else np.nan
                    valid_pulse_widths = df_step3['Pulse_Width'].dropna()
                    count_valid = valid_pulse_widths.count()

                    if count_valid == 0:
                        reason = f'No valid Pulse_Width data for threshold {threshold}'
                        self.add_exclusion_entry(file_path, reason)
                        self.logger.warning(f"{file_path.name}: {reason}")
                        continue

                    pulse_widths = valid_pulse_widths.sample(n=min(10, len(valid_pulse_widths)), random_state=42)
                    median_pulse_width = pulse_widths.median()

                    hitting_type = self.determine_hitting_type(file_path)
                    tire_position = df_step3['Tire_Position'].iloc[0] if len(df_step3) > 0 else 'Unknown'
                    wheel_type = df_step3['Wheel_Type'].iloc[0] if len(df_step3) > 0 else 'Unknown'
                    tire_id = f"{wheel_type}_{hitting_type}_{tire_position}"
                    relative_path = file_path.relative_to(self.processed_dir)

                    median_pulse_widths.append({
                        'File_Name': str(relative_path),
                        'Intensity_Threshold': threshold,
                        'Median_Pulse_Width': median_pulse_width,
                        'Air_Pressure': air_pressure,
                        'Hitting_Type': hitting_type,
                        'Tire_Position': tire_position,
                        'Wheel_Type': wheel_type,
                        'Tire': tire_id
                    })

            except Exception as e:
                reason = f'Error processing file: {e}'
                self.add_exclusion_entry(file_path, reason)
                self.logger.error(f"{file_path.name}: {reason}")
                continue

        return pd.DataFrame(median_pulse_widths)

    def calculate_dynamic_noise_threshold(self, segment_values: pd.Series, mounted: bool = False) -> float:
        """
        Enhanced dynamic noise threshold calculation with configuration-based parameters.
        """
        try:
            signal_length = len(segment_values)
            
            if mounted:
                # Use mounted tire parameters
                if signal_length < self.baseline_window_size_mounted:
                    self.logger.warning("Signal too short for dynamic threshold (mounted)")
                    return self.noise_threshold
                    
                rolling_mean = segment_values.rolling(window=self.sliding_window_min_size_mounted, center=True).mean()
                rolling_std = segment_values.rolling(window=self.sliding_window_min_size_mounted, center=True).std()
                
            else:
                # Use standalone parameters
                if signal_length < self.baseline_window_size:
                    self.logger.warning("Signal too short for dynamic threshold (standalone)")
                    return self.noise_threshold
                    
                rolling_mean = segment_values.rolling(window=self.sliding_window_min_size, center=True).mean()
                rolling_std = segment_values.rolling(window=self.sliding_window_min_size, center=True).std()
            
            # Compute signal volatility
            mean_value = rolling_mean.mean()
            volatility = rolling_std.mean() / mean_value if mean_value != 0 else float('inf')
            
            # Get window sizes based on configuration
            if mounted:
                baseline_window = self.baseline_window_size_mounted
                min_window = self.sliding_window_min_size_mounted
                max_window = self.sliding_window_max_size_mounted
            else:
                baseline_window = self.baseline_window_size
                min_window = self.sliding_window_min_size
                max_window = self.sliding_window_max_size
            
            # Multi-window threshold candidates
            thresholds = []
            windows = [baseline_window]
            
            for window_size in windows:
                baseline = segment_values.iloc[:window_size]
                baseline_median = baseline.median()
                baseline_mad = np.median(np.abs(baseline - baseline_median))
                baseline_std = baseline.std()
                
                if mounted:
                    thresholds.extend([
                        baseline_median + (baseline_mad * 2.5),
                        baseline_median + (baseline_std * self.std_dev_multiplier_mounted)
                    ])
                else:
                    thresholds.extend([
                        baseline_median + (baseline_mad * 2.5),
                        baseline_median + (baseline_std * self.std_dev_multiplier)
                    ])
            
            # Signal range analysis with configuration parameters
            signal_range = segment_values.max() - segment_values.min()
            if mounted:
                min_range_threshold = signal_range * self.min_threshold_percentage_mounted
                volatility_threshold = self.volatility_threshold_mounted
                high_percentile = self.high_volatility_percentile_mounted
                low_percentile = self.low_volatility_percentile_mounted
                max_factor = self.max_threshold_factor_mounted
            else:
                min_range_threshold = signal_range * self.min_threshold_percentage
                volatility_threshold = self.volatility_threshold
                high_percentile = self.high_volatility_percentile
                low_percentile = self.low_volatility_percentile
                max_factor = self.max_threshold_factor
            
            # Volatility-based threshold selection
            base_threshold = np.percentile(
                thresholds, 
                high_percentile if volatility > volatility_threshold else low_percentile
            )
            
            # Combine with range-based threshold and apply maximum constraint
            dynamic_threshold = min(
                max(base_threshold, min_range_threshold),
                segment_values.max() * max_factor
            )
            
            return dynamic_threshold
                
        except Exception as e:
            self.logger.error(f"Error in dynamic threshold calculation: {str(e)}")
            return self.noise_threshold if mounted else self.noise_threshold
            
    def get_first_increase_index_standalone(self, row: pd.Series) -> float:
        """
        Enhanced first rise point detection optimized for baseline-subtracted data.
        Uses the new dynamic threshold calculation and adds validation steps.
        
        Args:
            row (pd.Series): A single segment's signal values
            
        Returns:
            float: The index of the first rise point (1-based indexing) or np.nan if no valid rise
        """
        try:
            # Calculate dynamic threshold
            dynamic_threshold = self.calculate_dynamic_noise_threshold(row)
            
            # Convert series to numpy array for faster computation
            values = row.values
            
            # 1. Find all threshold crossings
            above_threshold = values > dynamic_threshold
            crossings = np.where(above_threshold)[0]
            
            if len(crossings) == 0:
                return np.nan
                
            # 2. Analyze potential rise points
            for start_idx in crossings:
                # Skip if too close to end
                if start_idx + self.sustained_rise_points >= len(values):
                    continue
                    
                # Check for sustained rise
                sustained_segment = values[start_idx:start_idx + self.sustained_rise_points]
                if np.all(sustained_segment > dynamic_threshold):
                    # Look back for gradual rise start
                    if start_idx > 0:
                        lookback_start = max(0, start_idx - self.lookback_window_size)
                        lookback_values = values[lookback_start:start_idx + 1]
                        
                        # Check for monotonic increase in lookback window
                        differences = np.diff(lookback_values)
                        if np.all(differences >= 0):
                            # Found gradual rise start
                            return lookback_start + 1  # Convert to 1-based indexing
                    
                    # If no gradual rise found, use threshold crossing point
                    return start_idx + 1  # Convert to 1-based indexing
            
            return np.nan
            
        except Exception as e:
            self.logger.error(f"Error in rise point detection: {e}")
            return np.nan
    
    def process_standalone_file(self, file_path: Path):
        """
        Processes a single standalone tire CSV file with new metadata format.
        Incorporates adaptive baseline subtraction (if enabled).
        """
        try:
            # 1) Read CSV and extract metadata
            df = pd.read_csv(file_path)
            metadata_columns = ['Tire Number', 'Pressure', 'TireSize', 'Tire_Type', 'Wear', 'Rim']
            metadata = {col: df[col].iloc[0] for col in metadata_columns}
            
            # 2) Isolate signal data
            signal_data = df.iloc[:, 7:]  # skip the first 7 columns of metadata
            df_signals = signal_data.copy()
            df_signals.index = df['Segment ID / Value index']
            df_signals = df_signals.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Apply signal trimming if enabled
            if self.trim_signal_standalone:
                self.logger.info(f"Trimming signals to {self.trim_dims_after_rise_standalone} dims after rise point")
                df_signals = self.trim_signal_to_rise(df_signals)

            # 3) Step1: Normalize (sums to 1)
            step1_normalized = df_signals.div(df_signals.sum(axis=1).replace(0, 1), axis=0)

            # 4) Apply baseline subtraction based on selected method
            if self.baseline_subtraction_method == "adaptive":
                self.logger.info("Applying adaptive baseline subtraction")
                step1_normalized = self.apply_adaptive_baseline_subtraction(step1_normalized)
            elif self.baseline_subtraction_method == "fixed":
                self.logger.info(f"Applying fixed baseline subtraction: {self.fixed_baseline_value}")
                step1_normalized = self.apply_fixed_baseline_subtraction(step1_normalized)
            else:  # "none" or any other value
                self.logger.info("No baseline subtraction applied")

            # Normalize again after baseline subtraction if any
            if self.baseline_subtraction_method in ["adaptive", "fixed"]:
                step1_normalized = step1_normalized.div(step1_normalized.sum(axis=1).replace(0, 1), axis=0)

            # 5) Step2: Cumulative Sum
            step2_cumulative = step1_normalized.cumsum(axis=1)

            # Set index names
            step1_normalized.index.name = 'Segment_ID'
            step2_cumulative.index.name = 'Segment_ID'

            # 6) Build output file
            new_filename = f"{metadata['Tire Number']}_{metadata['TireSize']}-{metadata['Pressure']}-{metadata['Tire_Type']}"
            output_xlsx_path = (self.processed_standalone_dir / new_filename).with_suffix('.xlsx')
            output_xlsx_path.parent.mkdir(parents=True, exist_ok= True)

            with pd.ExcelWriter(output_xlsx_path, engine='xlsxwriter') as writer:
                step1_normalized.to_excel(writer, sheet_name='Step1_Data')
                step2_cumulative.to_excel(writer, sheet_name='Step2_Sj')

                # For each intensity threshold, compute Step3
                for threshold in self.intensity_thresholds:
                    step3_data = self.compute_standalone_step3_metrics(step2_cumulative, threshold, metadata)
                    step3_data.to_excel(writer, sheet_name=f'Step3_DataPts_{threshold}')

            return output_xlsx_path

        except Exception as e:
            raise Exception(f"Failed to process standalone file {file_path.name}: {str(e)}")

    def compute_standalone_step3_metrics(self, df_cumulative: pd.DataFrame, 
                                      intensity_threshold: float,
                                      metadata: dict) -> pd.DataFrame:
        """
        Modified compute_step3_metrics for standalone processing using dynamic thresholds.
        """
        # Calculate indices using dynamic thresholds for first increase
        first_increase_index = df_cumulative.apply(
            lambda row: self.get_first_increase_index_standalone(row), axis=1
        )
        
        # Use regular threshold for point_exceeds_index since this is intensity-based
        point_exceeds_index = df_cumulative.apply(
            lambda row: self.get_point_exceeds_index(row, threshold=intensity_threshold), axis=1
        )

        # Get cumulative values
        first_increase_cumulative_value = df_cumulative.apply(
            lambda row: self.get_cumulative_value_at_index(row, first_increase_index[row.name]), axis=1
        )
        point_exceeds_cumulative_value = df_cumulative.apply(
            lambda row: self.get_cumulative_value_at_index(row, point_exceeds_index[row.name]), axis=1
        )

        # Calculate pulse width
        pulse_width = point_exceeds_index - first_increase_index

        # Create DataFrame with all metadata columns
        step3_data_points = pd.DataFrame({
            'Intensity_Threshold': intensity_threshold,
            'First_Noticeable_Increase_Index': first_increase_index,
            'Point_Exceeds_Index': point_exceeds_index,
            'First_Noticeable_Increase_Cumulative_Value': first_increase_cumulative_value,
            'Point_Exceeds_Cumulative_Value': point_exceeds_cumulative_value,
            'Pulse_Width': pulse_width,
            'Tire_Number': metadata['Tire Number'],
            'Pressure': metadata['Pressure'],
            'TireSize': metadata['TireSize'],
            'Tire_Type': metadata['Tire_Type'],
            'Wear': metadata['Wear'],
            'Rim': metadata['Rim']
        }, index=df_cumulative.index)

        return step3_data_points

    def calculate_standalone_median_pulse_width(self, excel_file_paths) -> pd.DataFrame:
        """
        Calculates the median pulse width for standalone tire data including all metadata.
        """
        median_pulse_widths = []
        total_files = len(excel_file_paths)
        
        self.logger.info(f"Starting median calculation for {total_files} files")
        
        for file_path in excel_file_paths:
            try:
                if file_path.name.startswith('~$'):
                    continue

                if not file_path.exists():
                    self.logger.error(f"File does not exist: {file_path}")
                    continue
                    
                xls = pd.ExcelFile(file_path)

                for threshold in self.intensity_thresholds:
                    sheet_name = f'Step3_DataPts_{threshold}'
                    
                    if sheet_name not in xls.sheet_names:
                        continue

                    df_step3 = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Validate required columns
                    required_columns = ['Pulse_Width', 'Tire_Number', 'Pressure', 'TireSize', 'Tire_Type', 'Wear', 'Rim']
                    if not all(col in df_step3.columns for col in required_columns):
                        self.logger.error(f"Missing required columns in {file_path.name}")
                        continue

                    # Get valid pulse widths
                    valid_pulse_widths = df_step3['Pulse_Width'].dropna()
                    if len(valid_pulse_widths) == 0:
                        continue

                    # Calculate median
                    sample_size = min(5, len(valid_pulse_widths))
                    median_pulse_width = valid_pulse_widths.sample(n=sample_size, random_state=42).median()
                    if pd.isna(median_pulse_width):
                        continue

                    # Extract metadata and add to results
                    try:
                        metadata = {
                            'Tire_Number': str(df_step3['Tire_Number'].iloc[0]),
                            'Pressure': float(df_step3['Pressure'].iloc[0]),
                            'TireSize': str(df_step3['TireSize'].iloc[0]),
                            'Tire_Type': str(df_step3['Tire_Type'].iloc[0]),
                            'Wear': str(df_step3['Wear'].iloc[0]),
                            'Rim': str(df_step3['Rim'].iloc[0])
                        }

                        median_pulse_widths.append({
                            'File_Name': str(file_path.relative_to(self.processed_standalone_dir)),
                            'Intensity_Threshold': threshold,
                            'Median_Pulse_Width': median_pulse_width,
                            **metadata
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error extracting metadata from {file_path.name}: {str(e)}")
                        continue

            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue

        if not median_pulse_widths:
            self.logger.warning("No valid median pulse widths were calculated")
            return pd.DataFrame(columns=['File_Name', 'Intensity_Threshold', 'Median_Pulse_Width',
                                    'Tire_Number', 'Pressure', 'TireSize', 'Tire_Type', 'Wear', 'Rim'])
        
        return pd.DataFrame(median_pulse_widths)

    def save_processed_data_step1_step2(self, writer: pd.ExcelWriter,
                                      step1: pd.DataFrame, step2: pd.DataFrame):
        """
        Saves Step1_Data and Step2_Sj into the Excel file.
        """
        step1.to_excel(writer, sheet_name='Step1_Data')
        step2.to_excel(writer, sheet_name='Step2_Sj')

    def save_processed_data_step3(self, writer: pd.ExcelWriter,
                                step3: pd.DataFrame, intensity_threshold: float):
        """
        Saves Step3_DataPts sheet.
        """
        prefix = 'Step3_DataPts'
        sheet_name = f'{prefix}_{intensity_threshold}'
        step3.to_excel(writer, sheet_name=sheet_name)

    def process_standalone_files(self):
        """
        Process all standalone tire files.
        """
        self.logger.info(f"Processing standalone files from: {self.standalone_dir}")
        
        # Get all CSV files in the standalone directory
        csv_files = list(self.standalone_dir.glob('*.csv'))
        if not csv_files:
            self.logger.warning("No CSV files found in standalone directory")
            return
        
        processed_files = []
        for file_path in tqdm(csv_files, desc="Processing standalone files"):
            try:
                if file_path.name.startswith('~$'):
                    continue
                    
                output_path = self.process_standalone_file(file_path)
                if output_path and output_path.exists():
                    processed_files.append(output_path)
                    self.logger.info(f"Successfully processed: {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing '{file_path.name}': {str(e)}")
                continue

        # Calculate and save median pulse widths
        if processed_files:
            df_median_pulse_widths = self.calculate_standalone_median_pulse_width(processed_files)
            
            if not df_median_pulse_widths.empty:
                output_path = self.processed_standalone_dir / 'Median_Pulse_Widths_StandAlone.xlsx'
                try:
                    df_median_pulse_widths.to_excel(output_path, index=False)
                    self.logger.info(f"Saved median pulse widths to: {output_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save median pulse widths: {str(e)}")
        else:
            self.logger.warning("No files were successfully processed")

    def determine_hitting_type(self, file_path: Path) -> str:
        """
        Determines the hitting type (Side or Tread) based on the file path or name.
        """
        file_name_lower = file_path.name.lower()
        if 'side' in file_name_lower:
            return 'Side'
        elif 'tread' in file_name_lower:
            return 'Tread'
        else:
            return 'Unknown'

    def add_exclusion_entry(self, file_path: Path, reason: str):
        """
        Adds an entry to the exclusions list with the reason for exclusion.
        """
        if not hasattr(self, 'exclusions'):
            self.exclusions = []
        relative_path = file_path.relative_to(self.output_dir)
        self.exclusions.append({
            'File_Name': str(relative_path),
            'Reason': reason
        })

    def save_exclusions(self):
        """
        Saves the exclusions to an Excel file in the output directory.
        """
        if hasattr(self, 'exclusions') and self.exclusions:
            df_exclusions = pd.DataFrame(self.exclusions)
            exclusions_output = self.output_dir / 'Excluded_Files.xlsx'
            df_exclusions.to_excel(exclusions_output, index=False)
            self.logger.info(f"Exclusions saved to {exclusions_output}")
            print(f"Exclusions saved to {exclusions_output}")

    def traverse_and_process(self):
        """
        Process both mounted and standalone tire data.
        """
         # Process mounted tire data
        self.logger.info(f"Processing mounted tire data from: {self.input_dir}")
        
        csv_files = list(Path(self.input_dir).rglob('*.csv'))
        self.logger.info(f"Found {len(csv_files)} CSV files")

        for file_path in tqdm(csv_files, desc="Processing mounted tire files"):
            if file_path.name.startswith('~$'):
                continue
                
            relative_path = file_path.relative_to(self.input_dir)
            output_paths = {
                'full': (self.processed_dir / relative_path.parent / file_path.stem).with_suffix('.xlsx'),
            }

            # Create output directories
            for path in output_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Process base data with metadata
                step1_normalized, step2_cumulative, metadata_dict = self.process_file_step1_step2(file_path)
                
                # [Rest of the processing code remains the same until saving]

                # Save full data results
                with pd.ExcelWriter(output_paths['full'], engine='xlsxwriter') as writer:
                    self.save_processed_data_step1_step2(writer, step1_normalized, step2_cumulative)
                    for threshold in self.intensity_thresholds:
                        step3_data = self.compute_step3_metrics(
                            step2_cumulative, threshold, file_path, metadata_dict
                        )
                        self.save_processed_data_step3(writer, step3_data, threshold)

            except Exception as e:
                self.logger.error(f"Error processing '{file_path.name}': {str(e)}")
                continue

        # Calculate median pulse widths for mounted data
        self._save_mounted_median_pulse_widths()
        
        # Process standalone data
        self.logger.info("Processing standalone tire data")
        self.process_standalone_files()

        # Save exclusions
        self.save_exclusions()
        self.logger.info("Processing completed successfully")

    def _save_mounted_median_pulse_widths(self):
        """
        Calculate and save median pulse widths for mounted tire data.
        """
        # For full data
        excel_files = [f for f in self.processed_dir.rglob('*.xlsx')
                    if not f.name.startswith('~$') and f.name != 'Median_Pulse_Widths.xlsx']
        if excel_files:
            df_median = self.calculate_median_pulse_width(excel_files)
            output_path = self.processed_dir / 'Median_Pulse_Widths.xlsx'
            df_median.to_excel(output_path, index=False)
            self.logger.info(f"Saved mounted tire median pulse widths to: {output_path}")


if __name__ == "__main__":
    # Define directories
    INPUT_DIRECTORY = '/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Data/Mounted/Processed_CSV_Files'
    STANDALONE_DIRECTORY = '/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Merged'
    OUTPUT_DIRECTORY = '.'

    # Load configuration with new structure
    config = load_config('config.yaml')
    
    # Initialize processor with separated configurations
    processor = TireSoundProcessor(
        input_dir=INPUT_DIRECTORY,
        standalone_dir=STANDALONE_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        # Common parameters
        intensity_thresholds=config['common']['intensity_thresholds'],
        # Mounted tire parameters
        noise_threshold=config['mounted'].get('noise_threshold', 0.03),
        trim_signal=config['mounted'].get('trim_signal', False),
        trim_dims_after_rise=config['mounted'].get('trim_dims_after_rise', 32),
        baseline_window_size_mounted = config['mounted'].get('baseline_window_size', 10),
        std_dev_multiplier_mounted = config['mounted'].get('std_dev_multiplier', 3.0),
        min_threshold_percentage_mounted = config['mounted'].get('min_threshold_percentage', 0.01),
        sustained_rise_points_mounted = config['mounted'].get('sustained_rise_points', 3),
        lookback_window_size_mounted = config['mounted'].get('lookback_window_size', 3),
        baseline_subtraction_method_mounted = config['mounted'].get('baseline_subtraction_method', 'adaptive'),
        fixed_baseline_value_mounted = config['mounted'].get('fixed_baseline_value', 0.01),
        baseline_computation_mounted = config['mounted'].get('baseline_computation', 'median'),
        sliding_window_min_size_mounted = config['mounted'].get('sliding_window_min_size', 5),
        sliding_window_max_size_mounted = config['mounted'].get('sliding_window_max_size', 20),
        quietness_metric_mounted = config['mounted'].get('quietness_metric', 'std'),
        volatility_threshold_mounted = config['mounted'].get('volatility_threshold', 0.5),
        high_volatility_percentile_mounted = config['mounted'].get('high_volatility_percentile', 75),
        low_volatility_percentile_mounted = config['mounted'].get('low_volatility_percentile', 25),
        max_threshold_factor_mounted = config['mounted'].get('max_threshold_factor', 0.3),
        # Standalone parameters
        trim_signal_standalone=config['standalone'].get('trim_signal', False),
        trim_dims_after_rise_standalone=config['standalone'].get('trim_dims_after_rise', 32),
        baseline_window_size=config['standalone'].get('baseline_window_size', 10),
        std_dev_multiplier=config['standalone'].get('std_dev_multiplier', 3.0),
        min_threshold_percentage=config['standalone'].get('min_threshold_percentage', 0.01),
        sustained_rise_points=config['standalone'].get('sustained_rise_points', 3),
        lookback_window_size=config['standalone'].get('lookback_window_size', 3),
        baseline_subtraction_method=config['standalone'].get('baseline_subtraction_method', 'adaptive'),
        fixed_baseline_value=config['standalone'].get('fixed_baseline_value', 0.01),
        baseline_computation=config['standalone'].get('baseline_computation', 'median'),
        sliding_window_min_size=config['standalone'].get('sliding_window_min_size', 5),
        sliding_window_max_size=config['standalone'].get('sliding_window_max_size', 20),
        quietness_metric=config['standalone'].get('quietness_metric', 'std'),
        volatility_threshold=config['standalone'].get('volatility_threshold', 0.5),
        high_volatility_percentile=config['standalone'].get('high_volatility_percentile', 75),
        low_volatility_percentile=config['standalone'].get('low_volatility_percentile', 25),
        max_threshold_factor=config['standalone'].get('max_threshold_factor', 0.3)
    )

    # Start the processing
    processor.traverse_and_process()