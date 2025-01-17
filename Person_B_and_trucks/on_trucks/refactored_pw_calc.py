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
            
            loaded_config = {
                'directories': pwc_config.get('directories', {
                    'input_dir': '/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Data/Mounted/Processed_CSV_Files(abcd@10_SideStrong-Best)',
                    'standalone_dir': '/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Merged',
                    'output_dir': '.',
                    'output_mounted_subdir': 'Processed_Mounted(abcd@10_SideStrong-Best)',
                    'output_standalone_subdir': 'Processed_Standalone'
                }),
                'common': pwc_config.get('common', {}),
                'mounted': pwc_config.get('mounted', {}),
                'standalone': pwc_config.get('standalone', {})
            }
            
            # Validate configuration
            validate_config(loaded_config)
            return loaded_config
            
    except Exception as e:
        print(f"Using default parameters ({str(e)})")
        return get_default_config()

def get_default_config() -> dict:
    """
    Returns default configuration values aligned with config.yaml defaults.
    """
    return {
        'directories': {
            'input_dir': '.',
            'standalone_dir': '.',
            'output_dir': '.',
            'output_mounted_subdir': 'Processed_Mounted',
            'output_standalone_subdir': 'Processed_Standalone'

        },
        'common': {
            'intensity_thresholds': [0.5, 0.7, 0.8, 0.9]
        },
        'mounted': {
            'noise_threshold': 0.03,
            'noise_detection_method': 'dynamic',
            'trim_signal': False,
            'trim_dims_after_rise': 100,
            'baseline_window_size': 2,
            'std_dev_multiplier': 2,
            'min_threshold_percentage': 0.01,
            'sustained_rise_points': 4,
            'lookback_window_size': 2,
            'baseline_subtraction_method': 'fixed',
            'fixed_baseline_value': 0.003,
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
            'noise_threshold': 0.03,
            'noise_detection_method': 'dynamic',
            'trim_signal': False,
            'trim_dims_after_rise': 32,
            'baseline_window_size': 2,
            'std_dev_multiplier': 2,
            'min_threshold_percentage': 0.05,
            'sustained_rise_points': 4,
            'lookback_window_size': 2,
            'baseline_subtraction_method': 'none',
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

def validate_config(config: dict) -> None:
    """
    Validate configuration parameters.
    """
    for section in ['mounted', 'standalone']:
        cfg = config.get(section, {})
        
        # Validate numeric ranges
        if cfg.get('baseline_window_size', 0) <= 0:
            raise ValueError(f"{section}: baseline_window_size must be greater than 0")
        
        if not 0 <= cfg.get('volatility_threshold', 0) <= 1:
            raise ValueError(f"{section}: volatility_threshold must be between 0 and 1")
        
        if cfg.get('high_volatility_percentile', 0) <= cfg.get('low_volatility_percentile', 0):
            raise ValueError(f"{section}: high_volatility_percentile must be greater than low_volatility_percentile")
        
        if not 0 <= cfg.get('min_threshold_percentage', 0) <= 1:
            raise ValueError(f"{section}: min_threshold_percentage must be between 0 and 1")
        
        # Validate method choices
        valid_noise_methods = ['dynamic', 'fixed']
        if cfg.get('noise_detection_method') not in valid_noise_methods:
            raise ValueError(f"{section}: noise_detection_method must be one of {valid_noise_methods}")
        
        valid_baseline_methods = ['none', 'fixed', 'adaptive']
        if cfg.get('baseline_subtraction_method') not in valid_baseline_methods:
            raise ValueError(f"{section}: baseline_subtraction_method must be one of {valid_baseline_methods}")
        
        valid_baseline_comp = ['mean', 'median']
        if cfg.get('baseline_computation') not in valid_baseline_comp:
            raise ValueError(f"{section}: baseline_computation must be one of {valid_baseline_comp}")
        
        valid_quietness_metrics = ['std', 'mad', 'mix']
        if cfg.get('quietness_metric') not in valid_quietness_metrics:
            raise ValueError(f"{section}: quietness_metric must be one of {valid_quietness_metrics}")

class TireSoundProcessor:
    def __init__(self, config: dict):
        """
        Initialize with standardized configuration structure.
        All directories are managed through config.yaml for easier maintenance.
        """
        # Store configurations
        self.common_config = config.get('common', {})
        self.mounted_config = config.get('mounted', {})
        self.standalone_config = config.get('standalone', {})
        self.dir_config = config.get('directories', {})
        
        # Common parameters
        self.intensity_thresholds = self.common_config.get('intensity_thresholds', [0.5, 0.7, 0.8, 0.9])
        
        # Directory paths from config with defaults
        self.input_dir = Path(self.dir_config.get('input_dir', '.'))
        self.standalone_dir = Path(self.dir_config.get('standalone_dir', '.'))
        self.output_dir = Path(self.dir_config.get('output_dir', '.'))
        
        # Output subdirectories from config with defaults
        mounted_subdir = self.dir_config.get('output_mounted_subdir', 'Processed_Mounted')
        standalone_subdir = self.dir_config.get('output_standalone_subdir', 'Processed_Standalone')
        
        # Create full output directory paths
        self.processed_dir = self.output_dir / mounted_subdir
        self.processed_standalone_dir = self.output_dir / standalone_subdir
        
        # Create output directories if they don't exist
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
        
        # Log directory information
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Standalone directory: {self.standalone_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Processed mounted directory: {self.processed_dir}")
        self.logger.info(f"Processed standalone directory: {self.processed_standalone_dir}")

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
        """
        try:
            # Process base data with metadata
            step1_normalized, step2_cumulative, metadata_dict = self.process_file_step1_step2(file_path, mounted=True)
            
            # Prepare full data sheets
            full_data = {
                'Step1_Data': step1_normalized,
                'Step2_Sj': step2_cumulative
            }
            
            # Add Step3 sheets for full data
            for threshold in self.intensity_thresholds:
                step3_data = self.compute_step3_metrics(
                    step2_cumulative, threshold, file_path, metadata_dict
                )
                full_data[f'Step3_DataPts_{threshold}'] = step3_data

            # Save full files
            full_success = self._save_to_excel(output_paths['full'], full_data)
            
            if not full_success:
                raise Exception("Failed to save one or more output files")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def trim_signal_to_rise(self, df_signals: pd.DataFrame, mounted: bool = True) -> pd.DataFrame:
        """
        Trims signal data preserving original numbering and handling multiple segments.
        Finds earliest rise point across all segments and trims accordingly.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            mounted (bool): Whether processing mounted tire data (True) or standalone (False)
            
        Returns:
            pd.DataFrame: Trimmed signal data with preserved numbering
        """
        try:
            # Get configuration based on mounted/standalone
            config = self.mounted_config if mounted else self.standalone_config
            trim_dims = config['trim_dims_after_rise']
            
            # Find rise points for all segments
            rise_points = {}
            earliest_rise = float('inf')
            
            # First pass: find all rise points and the earliest one
            for idx in df_signals.index:
                row_values = df_signals.loc[idx].values
                dynamic_threshold = self.calculate_dynamic_noise_threshold(pd.Series(row_values), mounted)
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
                    endpoint = rise_point + trim_dims
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
                    signal_end = min(rise_point + trim_dims, len(row_values))
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
            
        except Exception as e:
            error_msg = f"Error in signal trimming: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

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
        Applies a fixed baseline subtraction to the signal data using configured values.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            mounted (bool): Whether processing mounted tire data (True) or standalone (False)
            
        Returns:
            pd.DataFrame: Signal data with fixed baseline subtracted and negative values clipped to zero
        """
        try:
            # Get configuration based on mounted/standalone
            config = self.mounted_config if mounted else self.standalone_config
            
            # Verify baseline subtraction method
            if config['baseline_subtraction_method'] != 'fixed':
                self.logger.warning(
                    f"Fixed baseline subtraction called but method is {config['baseline_subtraction_method']}"
                )
            
            # Create copy to avoid modifying original
            df_modified = df_signals.copy()
            
            # Subtract fixed baseline value
            baseline_value = config['fixed_baseline_value']
            df_modified = df_modified.subtract(baseline_value)
            
            # Zero-clip negative values
            df_modified = df_modified.clip(lower=0)
            
            return df_modified
            
        except Exception as e:
            self.logger.error(f"Error in fixed baseline subtraction: {str(e)}")
            return df_signals  # Return original signals on error

    def apply_adaptive_baseline_subtraction(self, df_signals: pd.DataFrame, mounted: bool = False) -> pd.DataFrame:
        """
        Applies adaptive baseline subtraction using sliding window analysis.
        For each row, finds optimal baseline offset using configured quietness metrics.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            mounted (bool): Whether processing mounted tire data (True) or standalone (False)
            
        Returns:
            pd.DataFrame: Signal data with adaptive baseline subtracted
        """
        try:
            # Get configuration based on mounted/standalone
            config = self.mounted_config if mounted else self.standalone_config
            
            # Verify baseline subtraction method
            if config['baseline_subtraction_method'] != 'adaptive':
                self.logger.warning(
                    f"Adaptive baseline subtraction called but method is {config['baseline_subtraction_method']}"
                )
            
            # Create copy to avoid modifying original
            df_modified = df_signals.copy()
            
            for idx in df_modified.index:
                try:
                    row_values = df_modified.loc[idx].values
                    offset = self._find_adaptive_baseline_offset(row_values, config)
                    
                    # Subtract offset and zero-clip
                    row_values_sub = np.maximum(row_values - offset, 0)
                    
                    # Update the dataframe
                    df_modified.loc[idx] = row_values_sub
                    
                except Exception as e:
                    self.logger.error(f"Error processing row {idx}: {str(e)}")
                    continue  # Keep original values for this row
            
            return df_modified
            
        except Exception as e:
            self.logger.error(f"Error in adaptive baseline subtraction: {str(e)}")
            return df_signals  # Return original signals on error

    def _find_adaptive_baseline_offset(self, row_values: np.ndarray, config: dict) -> float:
        """
        Helper method to find baseline offset using sliding window approach.
        
        Args:
            row_values (np.ndarray): Signal values for a single row
            config (dict): Configuration dictionary for processing
            
        Returns:
            float: Optimal baseline offset value
        """
        n_samples = len(row_values)
        min_size = min(config['sliding_window_min_size'], n_samples)
        max_size = min(config['sliding_window_max_size'], n_samples)
        
        best_metric = float('inf')
        best_offset = 0.0
        
        # Slide through possible windows
        for start_idx in range(0, max_size - min_size + 1):
            end_idx = start_idx + min_size
            subwindow = row_values[start_idx:end_idx]
            
            # Compute baseline offset candidate
            if config['baseline_computation'] == 'median':
                offset_candidate = np.median(subwindow)
            else:  # 'mean'
                offset_candidate = np.mean(subwindow)
            
            # Evaluate quietness based on configured metric
            if config['quietness_metric'] == 'std':
                metric_val = np.std(subwindow)
            elif config['quietness_metric'] == 'mad':
                metric_val = np.median(np.abs(subwindow - np.median(subwindow)))
            else:  # 'mix'
                std_val = np.std(subwindow)
                mad_val = np.median(np.abs(subwindow - np.median(subwindow)))
                metric_val = (std_val + mad_val) / 2
            
            # Update if this window is quieter
            if metric_val < best_metric:
                best_metric = metric_val
                best_offset = offset_candidate
        
        return best_offset

    def process_file_step1_step2(self, file_path: Path, mounted: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Processes a single CSV file by extracting metadata and signal values,
        normalizing (Step1_Data), and creating the cumulative sum (Step2_Sj).
        
        Args:
            file_path (Path): Path to the CSV file
            mounted (bool): Whether processing mounted tire data (True) or standalone (False)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, dict]: 
                - Normalized signal data (Step1)
                - Cumulative sum data (Step2)
                - Metadata dictionary
        """
        try:
            # Get configuration based on mounted/standalone
            config = self.mounted_config if mounted else self.standalone_config
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract metadata columns
            metadata_columns = [
                'Segment ID / Value index', 'Tire Number', 'Pressure', 
                'TireSize', 'Tire_Type', 'Truck_Load'
            ]
            metadata = df[metadata_columns].copy()
            
            # Extract signal data
            signal_columns = [col for col in df.columns if col not in metadata_columns]
            df_signals = df[signal_columns].copy()
            
            # Convert signal values to numeric and handle missing values
            df_signals = df_signals.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Set index using Segment ID
            df_signals.index = metadata['Segment ID / Value index']
            
            # Apply baseline subtraction based on configuration
            baseline_method = config['baseline_subtraction_method']
            if baseline_method == "adaptive":
                self.logger.info(f"Applying adaptive baseline subtraction for {'mounted' if mounted else 'standalone'} data")
                df_signals = self.apply_adaptive_baseline_subtraction(df_signals, mounted)
            elif baseline_method == "fixed":
                self.logger.info(f"Applying fixed baseline subtraction: {config['fixed_baseline_value']}")
                df_signals = self.apply_fixed_baseline_subtraction(df_signals, mounted)
            else:
                self.logger.info("No baseline subtraction applied")
            
            # Apply signal trimming if enabled
            if config['trim_signal']:
                self.logger.info(f"Trimming signals to {config['trim_dims_after_rise']} dimensions after rise point")
                df_signals = self.trim_signal_to_rise(df_signals, mounted)
            
            # Step1: Normalize so sum of each row is 1
            # Handle zero-sum rows to prevent division by zero
            row_sums = df_signals.sum(axis=1)
            row_sums = row_sums.where(row_sums != 0, 1)  # Replace 0 with 1 to avoid division by zero
            df_step1 = df_signals.div(row_sums, axis=0)
            
            # Step2: Cumulative Sum
            df_step2 = df_step1.cumsum(axis=1)
            
            # Set index names
            df_step1.index.name = 'Segment_ID'
            df_step2.index.name = 'Segment_ID'
            
            # Store metadata for later use
            metadata_dict = metadata.set_index('Segment ID / Value index').to_dict('index')
            
            # Validate outputs
            if df_step1.isnull().any().any() or df_step2.isnull().any().any():
                self.logger.warning(f"NaN values detected in processed data for file: {file_path}")
            
            return df_step1, df_step2, metadata_dict

        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def get_first_increase_index(self, row: pd.Series, mounted: bool = False) -> float:
        """
        Unified function for finding first rise point in signal for both mounted and standalone.
        """
        try:
            # Get configuration based on mounted/standalone
            config = self.mounted_config if mounted else self.standalone_config
            noise_method = config['noise_detection_method']
            
            # Get threshold based on method
            if noise_method == 'dynamic':
                threshold = self.calculate_dynamic_noise_threshold(row, mounted)
            else:  # 'fixed'
                threshold = config['noise_threshold']
            
            # Convert to numpy array for performance
            values = row.values
            
            # Find points above threshold
            above_threshold = values > threshold
            crossings = np.where(above_threshold)[0]
            
            if len(crossings) == 0:
                return np.nan
                
            # Get sustained rise parameters from config
            sustained_points = config['sustained_rise_points']
            lookback_window = config['lookback_window_size']
            
            # Check each potential rise point
            for start_idx in crossings:
                # Skip if too close to end
                if start_idx + sustained_points >= len(values):
                    continue
                    
                # Check for sustained rise
                sustained_segment = values[start_idx:start_idx + sustained_points]
                if np.all(sustained_segment > threshold):
                    # Look back for gradual rise start
                    if start_idx > 0:
                        lookback_start = max(0, start_idx - lookback_window)
                        lookback_values = values[lookback_start:start_idx + 1]
                        
                        # Check for monotonic increase
                        differences = np.diff(lookback_values)
                        if np.all(differences >= 0):
                            return lookback_start + 1  # 1-based indexing
                    
                    return start_idx + 1  # 1-based indexing
            
            return np.nan
            
        except Exception as e:
            self.logger.error(f"Error in rise point detection: {e}")
            return np.nan

    def get_point_exceeds_index(self, row: pd.Series, threshold: float) -> float:
        """
        Gets the index where signal exceeds the specified threshold.
        
        Args:
            row (pd.Series): Signal values
            threshold (float): Threshold value to check against
            
        Returns:
            float: Index (1-based) where threshold is exceeded, or np.nan if not found
        """
        try:
            indices = np.where(row.values > threshold)[0]
            return indices[0] + 1 if indices.size > 0 else np.nan
        except Exception as e:
            self.logger.error(f"Error in threshold detection: {str(e)}")
            return np.nan

    def get_cumulative_value_at_index(self, row: pd.Series, idx: float) -> float:
        """
        Gets the cumulative value at the specified index.
        
        Args:
            row (pd.Series): Cumulative signal values
            idx (float): Index to get value for (1-based)
            
        Returns:
            float: Cumulative value at index, or np.nan if invalid
        """
        try:
            if pd.isna(idx):
                return np.nan
            idx = int(idx) - 1  # Convert to 0-based index
            if 0 <= idx < len(row):
                return row.iloc[idx]
            return np.nan
        except Exception as e:
            self.logger.error(f"Error getting cumulative value: {str(e)}")
            return np.nan

    def compute_step3_metrics(self, df_cumulative: pd.DataFrame, 
                         intensity_threshold: float, 
                         file_path: Path,
                         metadata_dict: dict) -> pd.DataFrame:
        """
        Compute Step3 metrics including metadata from the original file.
        """
        try:
            air_pressure = self.extract_air_pressure(file_path.name)
            tire_position = self.extract_tire_position(file_path.name)
            wheel_type = self.extract_vehicle_type(file_path)
            hitting_type = self.determine_hitting_type(file_path)

            # Fixed: Remove noise_threshold parameter
            first_increase_index = df_cumulative.apply(
                lambda row: self.get_first_increase_index(row, mounted=True), 
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

            # Validate pulse widths
            invalid_pulse_widths = pulse_width[pulse_width < 0]
            if not invalid_pulse_widths.empty:
                self.logger.warning(
                    f"Invalid pulse widths detected for {len(invalid_pulse_widths)} segments. "
                    f"Setting to NaN."
                )
                pulse_width[pulse_width < 0] = np.nan

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
            
        except Exception as e:
            error_msg = f"Error computing Step3 metrics for threshold {intensity_threshold}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

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
        Enhanced dynamic noise threshold calculation with unified configuration structure.
        
        Args:
            segment_values (pd.Series): Signal values to analyze
            mounted (bool): Whether processing mounted tire data (True) or standalone (False)
        
        Returns:
            float: Calculated dynamic threshold value
        """
        try:
            # Get configuration based on mounted/standalone
            config = self.mounted_config if mounted else self.standalone_config
            signal_length = len(segment_values)
            
            # Validate signal length
            if signal_length < config['baseline_window_size']:
                self.logger.warning(
                    f"Signal too short for dynamic threshold ({'mounted' if mounted else 'standalone'})"
                )
                return config['noise_threshold']
            
            # Calculate rolling statistics
            window_size = config['sliding_window_min_size']
            rolling_mean = segment_values.rolling(window=window_size, center=True).mean()
            rolling_std = segment_values.rolling(window=window_size, center=True).std()
            
            # Compute signal volatility
            mean_value = rolling_mean.mean()
            volatility = rolling_std.mean() / mean_value if mean_value != 0 else float('inf')
            
            # Get baseline window parameters
            baseline_window = config['baseline_window_size']
            
            # Multi-window threshold candidates
            thresholds = []
            baseline = segment_values.iloc[:baseline_window]
            baseline_median = baseline.median()
            baseline_mad = np.median(np.abs(baseline - baseline_median))
            baseline_std = baseline.std()
            
            # Calculate threshold candidates
            thresholds.extend([
                baseline_median + (baseline_mad * 2.5),
                baseline_median + (baseline_std * config['std_dev_multiplier'])
            ])
            
            # Signal range analysis
            signal_range = segment_values.max() - segment_values.min()
            min_range_threshold = signal_range * config['min_threshold_percentage']
            
            # Volatility-based threshold selection
            base_threshold = np.percentile(
                thresholds,
                config['high_volatility_percentile'] if volatility > config['volatility_threshold'] 
                else config['low_volatility_percentile']
            )
            
            # Combine with range-based threshold and apply maximum constraint
            dynamic_threshold = min(
                max(base_threshold, min_range_threshold),
                segment_values.max() * config['max_threshold_factor']
            )
            
            return dynamic_threshold
                
        except Exception as e:
            self.logger.error(f"Error in dynamic threshold calculation: {str(e)}")
            return config['noise_threshold']  # Fallback to base noise threshold
    
    def process_standalone_file(self, file_path: Path) -> Path:
        """
        Processes a single standalone tire CSV file with enhanced error handling
        and configuration-based processing.
        
        Args:
            file_path (Path): Path to the standalone tire CSV file
            
        Returns:
            Path: Path to the processed Excel file
        """
        try:
            # Read CSV and extract metadata
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_metadata = ['Tire Number', 'Pressure', 'TireSize', 'Tire_Type', 'Wear', 'Rim']
            missing_columns = [col for col in required_metadata if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract metadata from first row
            metadata = {col: df[col].iloc[0] for col in required_metadata}
            
            # Isolate signal data (skip the metadata columns)
            signal_data = df.iloc[:, len(required_metadata) + 1:]  # +1 for Segment ID
            df_signals = signal_data.copy()
            df_signals.index = df['Segment ID / Value index']
            
            # Convert to numeric and handle missing values
            df_signals = df_signals.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Apply signal trimming if enabled
            if self.standalone_config['trim_signal']:
                self.logger.info(
                    f"Trimming signals to {self.standalone_config['trim_dims_after_rise']} dims after rise point"
                )
                df_signals = self.trim_signal_to_rise(df_signals, mounted=False)
            
            # Step1: Normalize (sums to 1)
            row_sums = df_signals.sum(axis=1)
            row_sums = row_sums.where(row_sums != 0, 1)  # Handle zero-sum rows
            step1_normalized = df_signals.div(row_sums, axis=0)
            
            # Apply baseline subtraction based on configuration
            baseline_method = self.standalone_config['baseline_subtraction_method']
            if baseline_method == "adaptive":
                self.logger.info("Applying adaptive baseline subtraction for standalone data")
                step1_normalized = self.apply_adaptive_baseline_subtraction(step1_normalized, mounted=False)
            elif baseline_method == "fixed":
                self.logger.info(f"Applying fixed baseline subtraction: {self.standalone_config['fixed_baseline_value']}")
                step1_normalized = self.apply_fixed_baseline_subtraction(step1_normalized, mounted=False)
            else:
                self.logger.info("No baseline subtraction applied")
            
            # Renormalize after baseline subtraction if applied
            if baseline_method in ["adaptive", "fixed"]:
                row_sums = step1_normalized.sum(axis=1)
                row_sums = row_sums.where(row_sums != 0, 1)
                step1_normalized = step1_normalized.div(row_sums, axis=0)
            
            # Step2: Cumulative Sum
            step2_cumulative = step1_normalized.cumsum(axis=1)
            
            # Set index names
            step1_normalized.index.name = 'Segment_ID'
            step2_cumulative.index.name = 'Segment_ID'
            
            # Generate output filename
            new_filename = (f"{metadata['Tire Number']}_{metadata['TireSize']}-"
                        f"{metadata['Pressure']}-{metadata['Tire_Type']}")
            output_xlsx_path = (self.processed_standalone_dir / new_filename).with_suffix('.xlsx')
            output_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to Excel with all sheets
            with pd.ExcelWriter(output_xlsx_path, engine='xlsxwriter') as writer:
                # Save Step1 and Step2 data
                step1_normalized.to_excel(writer, sheet_name='Step1_Data')
                step2_cumulative.to_excel(writer, sheet_name='Step2_Sj')
                
                # Compute and save Step3 data for each threshold
                for threshold in self.intensity_thresholds:
                    step3_data = self.compute_standalone_step3_metrics(
                        step2_cumulative, 
                        threshold, 
                        metadata
                    )
                    step3_data.to_excel(
                        writer, 
                        sheet_name=f'Step3_DataPts_{threshold}'
                    )
            
            # Validate output file
            if not output_xlsx_path.exists():
                raise FileNotFoundError(f"Failed to create output file: {output_xlsx_path}")
                
            self.logger.info(f"Successfully processed standalone file: {file_path.name}")
            return output_xlsx_path

        except Exception as e:
            error_msg = f"Failed to process standalone file {file_path.name}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def compute_standalone_step3_metrics(self, df_cumulative: pd.DataFrame, 
                                   intensity_threshold: float,
                                   metadata: dict) -> pd.DataFrame:
        """
        Computes Step3 metrics for standalone processing using unified configuration.
        
        Args:
            df_cumulative (pd.DataFrame): Cumulative sum data from Step2
            intensity_threshold (float): Current intensity threshold value
            metadata (dict): Metadata dictionary for the tire
            
        Returns:
            pd.DataFrame: Computed Step3 metrics
        """
        try:
            # Calculate first rise indices using unified method
            first_increase_index = df_cumulative.apply(
                lambda row: self.get_first_increase_index(row, mounted=False), 
                axis=1
            )
            
            # Calculate point exceeds indices
            point_exceeds_index = df_cumulative.apply(
                lambda row: self.get_point_exceeds_index(row, threshold=intensity_threshold), 
                axis=1
            )
            
            # Get cumulative values at rise and threshold points
            first_increase_cumulative_value = df_cumulative.apply(
                lambda row: self.get_cumulative_value_at_index(row, first_increase_index[row.name]), 
                axis=1
            )
            point_exceeds_cumulative_value = df_cumulative.apply(
                lambda row: self.get_cumulative_value_at_index(row, point_exceeds_index[row.name]), 
                axis=1
            )
            
            # Calculate pulse width
            pulse_width = point_exceeds_index - first_increase_index
            
            # Validate results
            invalid_pulse_widths = pulse_width[pulse_width < 0]
            if not invalid_pulse_widths.empty:
                self.logger.warning(
                    f"Invalid pulse widths detected for {len(invalid_pulse_widths)} segments. "
                    f"Setting to NaN."
                )
                pulse_width[pulse_width < 0] = np.nan
            
            # Create DataFrame with all metrics and metadata
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
            
            # Sort columns for consistency
            step3_data_points = step3_data_points.sort_index(axis=1)
            
            # Add statistics
            stats = {
                'Mean_Pulse_Width': pulse_width.mean(),
                'Median_Pulse_Width': pulse_width.median(),
                'Std_Pulse_Width': pulse_width.std(),
                'Valid_Measurements': pulse_width.notna().sum(),
                'Total_Measurements': len(pulse_width)
            }
            
            # Log statistics
            self.logger.info(
                f"Step3 metrics computed for threshold {intensity_threshold}:\n" +
                "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
            )
            
            return step3_data_points
            
        except Exception as e:
            error_msg = (f"Error computing Step3 metrics for threshold {intensity_threshold}: "
                        f"{str(e)}")
            self.logger.error(error_msg)
            raise Exception(error_msg)

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
    # Load and validate configuration
    config = load_config('config.yaml')
    
    # Initialize processor with complete configuration
    processor = TireSoundProcessor(config=config)
    
    # Start processing
    processor.traverse_and_process()