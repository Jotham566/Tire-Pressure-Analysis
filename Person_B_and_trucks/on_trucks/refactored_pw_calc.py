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
            'selected_dims_after_rise_point': 32,
            'noise_threshold': 0.03
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
             selected_dims_after_rise_point: int = 32,
             # Standalone parameters
             trim_signal: bool = False,
             trim_dims_after_rise: int = 32,
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
        self.selected_dims_after_rise_point = selected_dims_after_rise_point
        
        # Standalone parameters
        self.trim_signal = trim_signal
        self.trim_dims_after_rise = trim_dims_after_rise
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
        self.processed_trimmed_dir = self.output_dir / 'Processed_Mounted_Trimmed'
        self.processed_standalone_dir = self.output_dir / 'Processed_Standalone'

        # Create all output directories at initialization
        for directory in [self.processed_dir, self.processed_trimmed_dir, self.processed_standalone_dir]:
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
            output_paths: Dictionary with 'full' and 'trimmed' output paths
        """
        try:
            # Process base data
            step1_normalized, step2_cumulative = self.process_file_step1_step2(file_path)
            step2sj_trim = self.create_step2sj_trim(step1_normalized, step2_cumulative)

            # Prepare full data sheets
            full_data = {
                'Step1_Data': step1_normalized,
                'Step2_Sj': step2_cumulative
            }
            
            # Add Step3 sheets for full data
            for threshold in self.intensity_thresholds:
                step3_data = self.compute_step3_metrics(step2_cumulative, threshold, file_path)
                full_data[f'Step3_DataPts_{threshold}'] = step3_data

            # Prepare trimmed data sheets
            trimmed_data = {
                'Step1_Data': step1_normalized,
                'Step2_Sj': step2_cumulative,
                'Step2sj_Trim': step2sj_trim
            }
            
            # Add Step3 sheets for trimmed data
            for threshold in self.intensity_thresholds:
                step3_trim_data = self.compute_step3_metrics(step2sj_trim, threshold, file_path)
                trimmed_data[f'Step3_Trim_DataPts_{threshold}'] = step3_trim_data

            # Save both files
            full_success = self._save_to_excel(output_paths['full'], full_data)
            trim_success = self._save_to_excel(output_paths['trimmed'], trimmed_data)
            
            if not (full_success and trim_success):
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
            dynamic_threshold = self.calculate_dynamic_noise_threshold(pd.Series(row_values))
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

    def find_adaptive_baseline_offset(self, row_values: np.ndarray) -> float:
        """
        Finds a baseline offset using a sliding window approach.
        Scans subwindows near the start of the row to find the "quietest" segment.
        Returns the median of that chosen subwindow.
        """

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
            if self.baseline_computation.lower() == 'median':
                offset_candidate = np.median(subwindow)
            else:
                offset_candidate = np.mean(subwindow)
            
            # Evaluate "quietness" based on self.quietness_metric
            if self.quietness_metric.lower() == 'std':
                metric_val = np.std(subwindow)
            elif self.quietness_metric.lower() == 'mean':
                metric_val = np.mean(subwindow)
            else:
                # Example: a mix, or fallback to std
                metric_val = np.std(subwindow)

            # If this subwindow is quieter, update best
            if metric_val < best_metric:
                best_metric = metric_val
                best_offset = offset_candidate

        return best_offset

    def apply_fixed_baseline_subtraction(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a fixed baseline subtraction to the signal data.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            
        Returns:
            pd.DataFrame: Signal data with fixed baseline subtracted and negative values clipped to zero
        """
        df_modified = df_signals.copy()
        
        # Subtract fixed value
        df_modified = df_modified - self.fixed_baseline_value
        
        # Zero-clip negative values
        df_modified = df_modified.clip(lower=0)
        
        return df_modified

    def apply_adaptive_baseline_subtraction(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in df_signals, find a baseline offset using a sliding
        window "quietness" check. Subtract that offset and zero-clip negatives.
        """
        df_modified = df_signals.copy()
        
        for idx in df_modified.index:
            row_values = df_modified.loc[idx].values  # to numpy array
            offset = self.find_adaptive_baseline_offset(row_values)
            
            # Subtract offset
            row_values_sub = row_values - offset
            # Zero-clip negative
            row_values_sub = np.where(row_values_sub < 0, 0, row_values_sub)
            
            # Update the dataframe
            df_modified.loc[idx] = row_values_sub

        return df_modified

    def process_file_step1_step2(self, file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes a single CSV file by selecting the first 256 signal values,
        normalizing (Step1_Data), and creating the cumulative sum (Step2_Sj).
        Step2_Sj will have values before noise threshold set to zero.
        """
        try:
            df = pd.read_csv(file_path, header=None, skiprows=1, dtype=str)
        except Exception as e:
            raise Exception(f"Failed to read CSV: {e}")

        df.rename(columns={0: 'Segment_ID'}, inplace=True)
        if df.shape[1] < 257:
            raise ValueError(f"Expected at least 257 columns, found {df.shape[1]}")

        signal_columns = [f'Signal_Value_{i}' for i in range(1, 257)]
        df_selected = df[['Segment_ID'] + list(range(1, 257))].copy()
        df_selected.columns = ['Segment_ID'] + signal_columns
        df_selected[signal_columns] = df_selected[signal_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_selected.set_index('Segment_ID', inplace=True)

        # Step1: Normalize so sum of each row is 1 (keep original)
        df_step1 = df_selected.div(df_selected.sum(axis=1).replace(0, 1), axis=0)

        # Step2: Cumulative Sum (Step2_Sj)
        df_step2 = df_step1.cumsum(axis=1)

        return df_step1, df_step2

    def create_step2sj_trim(self, step1_data: pd.DataFrame, step2_data: pd.DataFrame) -> pd.DataFrame:
        # Identify first rise point index for each segment
        first_increase_index = step2_data.apply(lambda row: self.get_first_increase_index(row, self.noise_threshold), axis=1)

        # Drop segments with no rise point if any
        valid_fii = first_increase_index.dropna()
        if valid_fii.empty:
            return pd.DataFrame()

        # Determine the earliest first rise point (minimum fii)
        min_fii = int(valid_fii.min())

        # Determine the maximum endpoint for uniform length
        max_endpoint = 0
        for idx, fii_val in first_increase_index.items():
            if pd.isna(fii_val):
                fii_val = min_fii
            fii_val = int(fii_val)
            endpoint = fii_val - 1 + self.selected_dims_after_rise_point
            if endpoint > max_endpoint:
                max_endpoint = endpoint

        trimmed_data_list = []
        for idx, row in step1_data.iterrows():
            fii = first_increase_index.loc[idx]
            if pd.isna(fii):
                fii = min_fii
            fii = int(fii)

            start_idx = fii - 1
            end_idx = start_idx + self.selected_dims_after_rise_point
            row_values = row.values

            if start_idx >= len(row_values):
                trimmed_segment = np.array([])
            else:
                trimmed_segment = row_values[start_idx:end_idx]

            seg_sum = trimmed_segment.sum()
            if seg_sum == 0:
                normalized_segment = trimmed_segment
            else:
                normalized_segment = trimmed_segment / seg_sum

            leading_zeros_count = (fii - min_fii)
            if leading_zeros_count < 0:
                leading_zeros_count = 0

            leading_zeros = np.zeros(leading_zeros_count)

            segment_length = len(normalized_segment)
            total_length = (max_endpoint - min_fii + 1)
            trailing_zeros_count = total_length - (leading_zeros_count + segment_length)
            if trailing_zeros_count < 0:
                trailing_zeros_count = 0
            trailing_zeros = np.zeros(trailing_zeros_count)

            full_row_values = np.concatenate([leading_zeros, normalized_segment, trailing_zeros])

            columns = [f"Signal_Value_{col_idx}" for col_idx in range(min_fii, max_endpoint+1)]

            df_segment = pd.DataFrame([full_row_values], index=[idx], columns=columns)
            trimmed_data_list.append(df_segment)

        df_trimmed = pd.concat(trimmed_data_list)
        df_trimmed.index.name = 'Segment_ID'

        df_step2sj_trim = df_trimmed.cumsum(axis=1)
        df_step2sj_trim.index.name = 'Segment_ID'
        return df_step2sj_trim

    def get_first_increase_index(self, row: pd.Series, noise_threshold: float) -> float:
        indices = np.where(row.values > noise_threshold)[0]
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

    def compute_step3_metrics(self, df_cumulative: pd.DataFrame, intensity_threshold: float, file_path: Path) -> pd.DataFrame:
        air_pressure = self.extract_air_pressure(file_path.name)
        tire_position = self.extract_tire_position(file_path.name)
        wheel_type = self.extract_vehicle_type(file_path)
        hitting_type = self.determine_hitting_type(file_path)

        first_increase_index = df_cumulative.apply(lambda row: self.get_first_increase_index(row, self.noise_threshold), axis=1)
        point_exceeds_index = df_cumulative.apply(lambda row: self.get_point_exceeds_index(row, threshold=intensity_threshold), axis=1)

        def positional_to_original_index(pos_idx, columns):
            if pd.isna(pos_idx):
                return np.nan
            col_name = columns[int(pos_idx)-1]
            return int(col_name.split('_')[-1])

        first_increase_index_original = first_increase_index.apply(lambda x: positional_to_original_index(x, df_cumulative.columns))
        point_exceeds_index_original = point_exceeds_index.apply(lambda x: positional_to_original_index(x, df_cumulative.columns))

        first_increase_cumulative_value = df_cumulative.apply(
            lambda row: self.get_cumulative_value_at_index(row, first_increase_index[row.name]), axis=1)
        point_exceeds_cumulative_value = df_cumulative.apply(
            lambda row: self.get_cumulative_value_at_index(row, point_exceeds_index[row.name]), axis=1)

        pulse_width_original = point_exceeds_index_original - first_increase_index_original

        step3_data_points = pd.DataFrame({
            'Intensity_Threshold': intensity_threshold,
            'First_Noticeable_Increase_Index': first_increase_index_original,
            'Point_Exceeds_Index': point_exceeds_index_original,
            'First_Noticeable_Increase_Cumulative_Value': first_increase_cumulative_value,
            'Point_Exceeds_Cumulative_Value': point_exceeds_cumulative_value,
            'Pulse_Width': pulse_width_original,
            'Air_Pressure': air_pressure,
            'Tire_Position': tire_position,
            'Wheel_Type': wheel_type,
            'Hitting_Type': hitting_type
        }, index=df_cumulative.index)

        step3_data_points.reset_index(inplace=True)
        step3_data_points.set_index('Segment_ID', inplace=True)
        return step3_data_points

    def calculate_median_pulse_width(self, excel_file_paths, trimmed=False) -> pd.DataFrame:
        median_pulse_widths = []
        prefix = 'Step3_Trim_DataPts' if trimmed else 'Step3_DataPts'

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
                    relative_path = file_path.relative_to(self.trimmed_dir(file_path) if trimmed else self.processed_dir)

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

    def calculate_dynamic_noise_threshold(self, segment_values: pd.Series) -> float:
        """
        Enhanced dynamic noise threshold calculation with volatility-based adjustment.
        """
        try:
            signal_length = len(segment_values)
            if signal_length < self.baseline_window_size:
                self.logger.warning("Signal too short for dynamic threshold")
                return self.noise_threshold
                
            # Calculate volatility metrics
            rolling_mean = segment_values.rolling(window=5, center=True).mean()
            rolling_std = segment_values.rolling(window=5, center=True).std()
            
            # Compute signal volatility
            mean_value = rolling_mean.mean()
            volatility = rolling_std.mean() / mean_value if mean_value != 0 else float('inf')
            
            # Multi-window threshold candidates
            thresholds = []
            windows = [self.baseline_window_size]
            
            for window_size in windows:
                baseline = segment_values.iloc[:window_size]
                baseline_median = baseline.median()
                baseline_mad = np.median(np.abs(baseline - baseline_median))
                baseline_std = baseline.std()
                
                thresholds.extend([
                    baseline_median + (baseline_mad * 2.5),
                    baseline_median + (baseline_std * self.std_dev_multiplier)
                ])
            
            # Signal range analysis
            signal_range = segment_values.max() - segment_values.min()
            min_range_threshold = signal_range * self.min_threshold_percentage
            
            # Volatility-based threshold selection
            base_threshold = np.percentile(
                thresholds, 
                self.high_volatility_percentile if volatility > self.volatility_threshold else self.low_volatility_percentile
            )
            
            # Combine with range-based threshold and apply maximum constraint
            dynamic_threshold = min(
                max(base_threshold, min_range_threshold),
                segment_values.max() * self.max_threshold_factor
            )
            
            return dynamic_threshold
            
        except Exception as e:
            self.logger.error(f"Error in dynamic threshold calculation: {str(e)}")
            return self.noise_threshold
            
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
            if self.trim_signal:
                self.logger.info(f"Trimming signals to {self.trim_dims_after_rise} dims after rise point")
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
            output_xlsx_path.parent.mkdir(parents=True, exist_ok=True)

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
                                step3: pd.DataFrame, intensity_threshold: float, trimmed: bool = False):
        """
        Saves Step3_DataPts or Step3_Trim_DataPts sheet.
        """
        prefix = 'Step3_Trim_DataPts' if trimmed else 'Step3_DataPts'
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

    def trimmed_dir(self, file_path: Path) -> Path:
        """
        Helper method to determine the appropriate directory for relative paths
        """
        return self.processed_trimmed_dir if 'Processed_Mounted_Trimmed' in file_path.parts else self.processed_dir

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
                'trimmed': (self.processed_trimmed_dir / relative_path.parent / file_path.stem).with_suffix('.xlsx')
            }

            # Create output directories
            for path in output_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Process base data
                step1_normalized, step2_cumulative = self.process_file_step1_step2(file_path)
                step2sj_trim = self.create_step2sj_trim(step1_normalized, step2_cumulative)

                # Save full data results
                with pd.ExcelWriter(output_paths['full'], engine='xlsxwriter') as writer:
                    self.save_processed_data_step1_step2(writer, step1_normalized, step2_cumulative)
                    for threshold in self.intensity_thresholds:
                        step3_data = self.compute_step3_metrics(step2_cumulative, threshold, file_path)
                        self.save_processed_data_step3(writer, step3_data, threshold)

                # Save trimmed results
                with pd.ExcelWriter(output_paths['trimmed'], engine='xlsxwriter') as writer:
                    step1_normalized.to_excel(writer, sheet_name='Step1_Data')
                    step2_cumulative.to_excel(writer, sheet_name='Step2_Sj')
                    step2sj_trim.to_excel(writer, sheet_name='Step2sj_Trim')
                    
                    for threshold in self.intensity_thresholds:
                        step3_trim_data = self.compute_step3_metrics(step2sj_trim, threshold, file_path)
                        self.save_processed_data_step3(writer, step3_trim_data, threshold, trimmed=True)

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
        
        # For trimmed data
        excel_files_trim = [f for f in self.processed_trimmed_dir.rglob('*.xlsx')
                        if not f.name.startswith('~$') and f.name != 'Median_Pulse_Widths_Trim.xlsx']
        if excel_files_trim:
            df_median_trim = self.calculate_median_pulse_width(excel_files_trim, trimmed=True)
            output_path = self.processed_trimmed_dir / 'Median_Pulse_Widths_Trim.xlsx'
            df_median_trim.to_excel(output_path, index=False)
            self.logger.info(f"Saved mounted tire trimmed median pulse widths to: {output_path}")

if __name__ == "__main__":
    # Define directories
    INPUT_DIRECTORY = '/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Data/On_trucks_tires CSV data'
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
        noise_threshold=config['mounted']['noise_threshold'],
        selected_dims_after_rise_point=config['mounted']['selected_dims_after_rise_point'],
        # Standalone parameters
        trim_signal=config['standalone']['trim_signal'],
        trim_dims_after_rise=config['standalone']['trim_dims_after_rise'],
        baseline_window_size=config['standalone']['baseline_window_size'],
        std_dev_multiplier=config['standalone']['std_dev_multiplier'],
        min_threshold_percentage=config['standalone']['min_threshold_percentage'],
        sustained_rise_points=config['standalone']['sustained_rise_points'],
        lookback_window_size=config['standalone']['lookback_window_size'],
        baseline_subtraction_method=config['standalone']['baseline_subtraction_method'],
        fixed_baseline_value=config['standalone']['fixed_baseline_value'],
        baseline_computation=config['standalone']['baseline_computation'],
        sliding_window_min_size=config['standalone']['sliding_window_min_size'],
        sliding_window_max_size=config['standalone']['sliding_window_max_size'],
        quietness_metric=config['standalone']['quietness_metric'],
        volatility_threshold=config['standalone']['volatility_threshold'],
        high_volatility_percentile=config['standalone']['high_volatility_percentile'],
        low_volatility_percentile=config['standalone']['low_volatility_percentile'],
        max_threshold_factor=config['standalone']['max_threshold_factor']
    )

    # Start the processing
    processor.traverse_and_process()