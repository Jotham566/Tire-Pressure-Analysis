import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import yaml
import logging
import pywt
from typing import Union, Tuple

class TireSoundProcessor:
    def __init__(self, input_dir: str, standalone_dir: str, output_dir: str,
                 noise_threshold: float = 0.03,
                 intensity_thresholds: list = [0.5, 0.7, 0.8, 0.9],
                 selected_dims_after_rise_point: int = 32,
                 baseline_window_size: int = 10,
                 std_dev_multiplier: float = 3.0,
                 min_threshold_percentage: float = 0.01,
                 sustained_rise_points: int = 3,
                 lookback_window_size: int = 3,
                 use_adaptive_baseline_subtraction: bool = True,
                 baseline_computation: str = 'median',
                 sliding_window_min_size: int = 5,
                 sliding_window_max_size: int = 20,
                 quietness_metric: str = 'std',
                 volatility_threshold: float = 0.5,
                 high_volatility_percentile: int = 75,
                 low_volatility_percentile: int = 25,
                 max_threshold_factor: float = 0.3,
                 baseline_subtraction_method: str = "adaptive",
                 fixed_baseline_value: float = 0.01,
                 trim_signal: bool = False,
                 trim_dims_after_rise: int = 32
                 ):
        
        # Parameters for wavform processing 
        self.input_dir = Path(input_dir)
        self.standalone_dir = Path(standalone_dir)
        self.output_dir = Path(output_dir)
        self.noise_threshold = noise_threshold
        self.intensity_thresholds = intensity_thresholds
        self.selected_dims_after_rise_point = selected_dims_after_rise_point

        # Parameters for Dynamic Noise Threshold for Standalone Data
        self.baseline_window_size = baseline_window_size
        self.std_dev_multiplier = std_dev_multiplier
        self.min_threshold_percentage = min_threshold_percentage
        self.sustained_rise_points = sustained_rise_points
        self.lookback_window_size = lookback_window_size
        #self.zero_before_threshold = zero_before_threshold

        # Baseline parameters
        self.baseline_subtraction_method = baseline_subtraction_method.lower()
        self.fixed_baseline_value = fixed_baseline_value

        # Parameters for Adaptive Baseline Subtraction
        self.use_adaptive_baseline_subtraction = use_adaptive_baseline_subtraction
        self.baseline_computation = baseline_computation
        self.sliding_window_min_size = sliding_window_min_size
        self.sliding_window_max_size = sliding_window_max_size
        self.quietness_metric = quietness_metric

        # Parameters for Volatility-based Threshold Selection
        self.volatility_threshold = volatility_threshold
        self.high_volatility_percentile = high_volatility_percentile
        self.low_volatility_percentile = low_volatility_percentile
        self.max_threshold_factor = max_threshold_factor

        # Parameters for standalone signal trimming
        self.trim_signal = trim_signal
        self.trim_dims_after_rise = trim_dims_after_rise

        # Directories for different data types
        self.processed_dir = self.output_dir / 'Processed'
        self.processed_trimmed_dir = self.output_dir / 'Processed_Trimmed'
        self.processed_standalone_dir = self.output_dir / 'Processed_Stand_Alone'

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
        Includes detailed logging for parameter tuning.
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
            
            self.logger.info(f"Signal volatility: {volatility:.4f}")
            
            # Multi-window threshold candidates
            thresholds = []
            windows = [self.baseline_window_size]
            
            for window_size in windows:
                baseline = segment_values.iloc[:window_size]
                
                # Calculate robust statistics
                baseline_median = baseline.median()
                baseline_mad = np.median(np.abs(baseline - baseline_median))
                baseline_std = baseline.std()
                
                # Store different threshold candidates
                mad_threshold = baseline_median + (baseline_mad * 2.5)
                std_threshold = baseline_median + (baseline_std * self.std_dev_multiplier)
                
                thresholds.extend([mad_threshold, std_threshold])
                
                self.logger.debug(f"""
                    Window size: {window_size}
                    MAD threshold: {mad_threshold:.4f}
                    STD threshold: {std_threshold:.4f}
                """)
            
            # Signal range analysis
            signal_range = segment_values.max() - segment_values.min()
            min_range_threshold = signal_range * self.min_threshold_percentage
            
            # Volatility-based threshold selection
            if volatility > self.volatility_threshold:
                # High volatility: use higher percentile
                base_threshold = np.percentile(thresholds, self.high_volatility_percentile)
                self.logger.info(f"High volatility detected, using {self.high_volatility_percentile}th percentile")
            else:
                # Low volatility: use lower percentile
                base_threshold = np.percentile(thresholds, self.low_volatility_percentile)
                self.logger.info(f"Low volatility detected, using {self.low_volatility_percentile}th percentile")
            
            # Combine with range-based threshold
            dynamic_threshold = max(base_threshold, min_range_threshold)
            
            # Apply maximum threshold constraint
            max_allowed = segment_values.max() * self.max_threshold_factor
            dynamic_threshold = min(dynamic_threshold, max_allowed)
            
            self.logger.info(f"""
                Final threshold calculation:
                Base threshold: {base_threshold:.4f}
                Range-based minimum: {min_range_threshold:.4f}
                Maximum allowed: {max_allowed:.4f}
                Final dynamic threshold: {dynamic_threshold:.4f}
            """)
            
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

    def extract_standalone_metadata(self, file_name: str) -> tuple:
        """
        Extracts metadata from standalone tire file names (e.g., 01_900.csv)
        Returns: (tire_number, air_pressure)
        """
        match = re.match(r'(\d+)_(\d+)\.csv', file_name)
        if match:
            tire_number = match.group(1)
            air_pressure = float(match.group(2))
            return tire_number, air_pressure
        return None, np.nan
    
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

            # Apply signal trimming if enabled
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

            # Set index names
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
        Enhanced with detailed debugging and validation.
        """
        median_pulse_widths = []
        total_files = len(excel_file_paths)
        
        self.logger.info(f"Starting median calculation with {total_files} files")
        print(f"\nProcessing {total_files} files for median calculation...")

        for file_path in excel_file_paths:
            try:
                if file_path.name.startswith('~$'):
                    continue
                    
                self.logger.info(f"\nProcessing file: {file_path.name}")
                print(f"Processing: {file_path.name}")
                
                # Verify file exists and is readable
                if not file_path.exists():
                    self.logger.error(f"File does not exist: {file_path}")
                    continue
                    
                xls = pd.ExcelFile(file_path)
                
                # Debug: Print available sheets
                self.logger.info(f"Available sheets in {file_path.name}: {xls.sheet_names}")

                for threshold in self.intensity_thresholds:
                    sheet_name = f'Step3_DataPts_{threshold}'
                    
                    if sheet_name not in xls.sheet_names:
                        self.logger.warning(f"Sheet {sheet_name} not found in {file_path.name}")
                        continue

                    # Read the sheet
                    df_step3 = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Debug: Print shape and columns
                    self.logger.info(f"Sheet {sheet_name} shape: {df_step3.shape}")
                    self.logger.info(f"Columns: {df_step3.columns.tolist()}")

                    # Ensure required columns exist
                    required_columns = ['Pulse_Width', 'Tire_Number', 'Pressure', 'TireSize', 'Tire_Type', 'Wear', 'Rim']
                    missing_columns = [col for col in required_columns if col not in df_step3.columns]
                    
                    if missing_columns:
                        self.logger.error(f"Missing columns in {file_path.name}: {missing_columns}")
                        continue

                    # Get valid pulse widths
                    valid_pulse_widths = df_step3['Pulse_Width'].dropna()
                    self.logger.info(f"Found {len(valid_pulse_widths)} valid pulse widths")
                    
                    if len(valid_pulse_widths) == 0:
                        self.logger.warning(f"No valid pulse widths in {sheet_name}")
                        continue

                    # Sample pulse widths
                    sample_size = min(5, len(valid_pulse_widths))
                    pulse_widths = valid_pulse_widths.sample(n=sample_size, random_state=42)
                    median_pulse_width = pulse_widths.median()

                    if pd.isna(median_pulse_width):
                        self.logger.warning(f"Calculated median is NaN")
                        continue

                    # Debug: Print median value
                    self.logger.info(f"Calculated median pulse width: {median_pulse_width}")

                    # Get metadata from first row
                    try:
                        metadata = {
                            'Tire_Number': str(df_step3['Tire_Number'].iloc[0]),
                            'Pressure': float(df_step3['Pressure'].iloc[0]),
                            'TireSize': str(df_step3['TireSize'].iloc[0]),
                            'Tire_Type': str(df_step3['Tire_Type'].iloc[0]),
                            'Wear': str(df_step3['Wear'].iloc[0]),
                            'Rim': str(df_step3['Rim'].iloc[0])
                        }
                        
                        # Debug: Print metadata
                        self.logger.info(f"Extracted metadata: {metadata}")

                        median_pulse_widths.append({
                            'File_Name': str(file_path.relative_to(self.processed_standalone_dir)),
                            'Intensity_Threshold': threshold,
                            'Median_Pulse_Width': median_pulse_width,
                            **metadata
                        })
                        
                        self.logger.info(f"Successfully added data point for {file_path.name}, threshold {threshold}")
                        
                    except Exception as e:
                        self.logger.error(f"Error extracting metadata: {str(e)}")
                        continue

            except Exception as e:
                self.logger.error(f"Error processing file '{file_path.name}': {str(e)}")
                continue

        # Final validation
        self.logger.info(f"\nTotal data points collected: {len(median_pulse_widths)}")
        print(f"\nTotal data points collected: {len(median_pulse_widths)}")

        if not median_pulse_widths:
            self.logger.warning("No valid median pulse widths were calculated!")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['File_Name', 'Intensity_Threshold', 'Median_Pulse_Width',
                                    'Tire_Number', 'Pressure', 'TireSize', 'Tire_Type', 'Wear', 'Rim'])
        
        # Create DataFrame and verify
        df_result = pd.DataFrame(median_pulse_widths)
        self.logger.info(f"Final DataFrame shape: {df_result.shape}")
        self.logger.info(f"Columns in final DataFrame: {df_result.columns.tolist()}")
        
        return df_result

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
        Process all standalone tire files with enhanced debugging
        """
        self.logger.info(f"Processing standalone files from: {self.standalone_dir}")
        print(f"Processing standalone files from: {self.standalone_dir}")

        # Get all CSV files in the standalone directory
        csv_files = list(self.standalone_dir.glob('*.csv'))
        total_files = len(csv_files)
        
        if total_files == 0:
            self.logger.warning("No CSV files found in standalone directory!")
            print("No CSV files found in standalone directory!")
            return
            
        self.logger.info(f"Total standalone CSV files found: {total_files}")
        print(f"Total standalone CSV files found: {total_files}")

        processed_files = []
        for file_path in tqdm(csv_files, desc="Processing standalone files", unit="file"):
            try:
                if file_path.name.startswith('~$'):
                    continue
                    
                output_path = self.process_standalone_file(file_path)
                if output_path and output_path.exists():
                    processed_files.append(output_path)
                    self.logger.info(f"Successfully processed: {file_path.name} -> {output_path}")
                else:
                    self.logger.warning(f"Output file not created for: {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing standalone file '{file_path.name}': {str(e)}")
                continue

        # Calculate and save median pulse widths for standalone files
        if processed_files:
            self.logger.info(f"Calculating median pulse widths for {len(processed_files)} processed files")
            print(f"\nCalculating median pulse widths for {len(processed_files)} files...")
            
            df_median_pulse_widths = self.calculate_standalone_median_pulse_width(processed_files)
            
            # Debug: Print DataFrame info
            self.logger.info(f"DataFrame info: {df_median_pulse_widths.info()}")
            print(f"\nDataFrame shape: {df_median_pulse_widths.shape}")
            
            if not df_median_pulse_widths.empty:
                median_pulse_widths_output = self.processed_standalone_dir / 'Median_Pulse_Widths_StandAlone.xlsx'
                
                # Add a try-except block for saving
                try:
                    df_median_pulse_widths.to_excel(median_pulse_widths_output, index=False)
                    self.logger.info(f"Successfully saved median pulse widths to: {median_pulse_widths_output}")
                    print(f"\nSuccessfully saved median pulse widths to: {median_pulse_widths_output}")
                except Exception as e:
                    self.logger.error(f"Error saving median pulse widths: {str(e)}")
                    print(f"\nError saving median pulse widths: {str(e)}")
            else:
                self.logger.warning("DataFrame is empty - no median pulse widths to save")
                print("\nDataFrame is empty - no median pulse widths to save")
        else:
            self.logger.warning("No files were successfully processed - skipping median calculation")
            print("\nNo files were successfully processed - skipping median calculation")

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
        return self.processed_trimmed_dir if 'Processed_Trimmed' in file_path.parts else self.processed_dir

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
        Process both original and standalone data
        """
        # Process original data
        self.logger.info(f"Starting processing original data.\nInput Directory: {self.input_dir}\nOutput Directory: {self.output_dir}\n")
        print(f"Starting processing original data...\nInput Directory: {self.input_dir}\nOutput Directory: {self.output_dir}\n")

        # Collect all CSV file paths for original data
        csv_file_paths = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    input_file_path = Path(root) / file
                    csv_file_paths.append(input_file_path)

        total_files = len(csv_file_paths)
        self.logger.info(f"Total original CSV files found: {total_files}\n")
        print(f"Total original CSV files found: {total_files}\n")

        for file_path in tqdm(csv_file_paths, desc="Processing original CSV files", unit="file"):
            relative_path = file_path.relative_to(self.input_dir)
            
            # Output paths for full and trimmed scenarios
            output_xlsx_full_path = (self.processed_dir / relative_path.parent / file_path.stem).with_suffix('.xlsx')
            output_xlsx_trimmed_path = (self.processed_trimmed_dir / relative_path.parent / file_path.stem).with_suffix('.xlsx')

            # Ensure the output directories exist
            output_xlsx_full_path.parent.mkdir(parents=True, exist_ok=True)
            output_xlsx_trimmed_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Process Step1, Step2
                step1_normalized, step2_cumulative = self.process_file_step1_step2(file_path)

                # Create Step2sj_Trim
                step2sj_trim = self.create_step2sj_trim(step1_normalized, step2_cumulative)

                # Write Full Data Results (Processed)
                with pd.ExcelWriter(output_xlsx_full_path, engine='xlsxwriter') as writer_full:
                    # Save Step1 and Step2
                    self.save_processed_data_step1_step2(writer_full, step1_normalized, step2_cumulative)
                    # For each intensity threshold, compute and save step3 data
                    for threshold in self.intensity_thresholds:
                        step3_data_points = self.compute_step3_metrics(step2_cumulative, threshold, file_path)
                        self.save_processed_data_step3(writer_full, step3_data_points, threshold)

                # Write Trimmed Results (Processed_Trimmed)
                with pd.ExcelWriter(output_xlsx_trimmed_path, engine='xlsxwriter') as writer_trim:
                    # Save Step1, Step2_Sj, Step2sj_Trim
                    step1_normalized.to_excel(writer_trim, sheet_name='Step1_Data')
                    step2_cumulative.to_excel(writer_trim, sheet_name='Step2_Sj')
                    step2sj_trim.to_excel(writer_trim, sheet_name='Step2sj_Trim')

                    # For each intensity threshold, compute and save step3_Trim
                    for threshold in self.intensity_thresholds:
                        step3_trim_data_points = self.compute_step3_metrics(step2sj_trim, threshold, file_path)
                        # Rename sheet as Step3_Trim_DataPts_xxx
                        self.save_processed_data_step3(writer_trim, step3_trim_data_points, threshold, trimmed=True)

            except Exception as e:
                self.logger.error(f"Error processing file '{file_path.name}': {e}\n")
                print(f"Error processing file '{file_path.name}': {e}\n")

        # Calculate median pulse widths for original data
        excel_file_paths_full = list(self.processed_dir.rglob('*.xlsx'))
        excel_file_paths_full = [f for f in excel_file_paths_full if f.name != 'Median_Pulse_Widths.xlsx' and not f.name.startswith('~$')]
        df_median_pulse_widths = self.calculate_median_pulse_width(excel_file_paths_full)
        median_pulse_widths_output = self.processed_dir / 'Median_Pulse_Widths.xlsx'
        df_median_pulse_widths.to_excel(median_pulse_widths_output, index=False)
        
        # Calculate median pulse widths for trimmed data
        excel_file_paths_trimmed = list(self.processed_trimmed_dir.rglob('*.xlsx'))
        excel_file_paths_trimmed = [f for f in excel_file_paths_trimmed if f.name != 'Median_Pulse_Widths_Trim.xlsx' and not f.name.startswith('~$')]
        df_median_pulse_widths_trim = self.calculate_median_pulse_width(excel_file_paths_trimmed, trimmed=True)
        median_pulse_widths_trim_output = self.processed_trimmed_dir / 'Median_Pulse_Widths_Trim.xlsx'
        df_median_pulse_widths_trim.to_excel(median_pulse_widths_trim_output, index=False)

        # Process standalone data
        self.logger.info("Starting processing of standalone data...")
        print("\nStarting processing of standalone data...")
        self.process_standalone_files()

        # Save any exclusions
        self.save_exclusions()

        self.logger.info("All processing completed successfully.")
        print("\nAll processing completed successfully.")

if __name__ == "__main__":
    # Define directories
    INPUT_DIRECTORY = 'Attenuation Analysis on TD data'
    STANDALONE_DIRECTORY = '/Users/jothamwambi/Projects/tire_pressure_analysis/Pulse_Width_Analysis/Stand-Alone Tire/Processed_CSV_Files'
    OUTPUT_DIRECTORY = '.'

    # Load configuration
    config_file = 'config.yaml'
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        noise_threshold = config.get('pulse_width_calculator', {}).get('noise_threshold', 0.03)
        intensity_thresholds = config.get('pulse_width_calculator', {}).get('intensity_thresholds', [0.5, 0.7, 0.8, 0.9])
        selected_dims_after_rise_point = config.get('pulse_width_calculator', {}).get('selected_dims_after_rise_point', 32)
        baseline_window_size = config.get('pulse_width_calculator', {}).get('baseline_window_size', 10)
        std_dev_multiplier = config.get('pulse_width_calculator', {}).get('std_dev_multiplier', 3.0)
        min_threshold_percentage = config.get('pulse_width_calculator', {}).get('min_threshold_percentage', 0.01)
        sustained_rise_points = config.get('pulse_width_calculator', {}).get('sustained_rise_points', 3)
        lookback_window_size = config.get('pulse_width_calculator', {}).get('lookback_window_size', 3)
        use_adaptive_baseline_subtraction = config['pulse_width_calculator'].get('use_adaptive_baseline_subtraction', True)
        baseline_computation = config['pulse_width_calculator'].get('baseline_computation', 'median')
        sliding_window_min_size = config['pulse_width_calculator'].get('sliding_window_min_size', 5)
        sliding_window_max_size = config['pulse_width_calculator'].get('sliding_window_max_size', 20)
        quietness_metric = config['pulse_width_calculator'].get('quietness_metric', 'std')
        volatility_threshold = config['pulse_width_calculator'].get('volatility_threshold', 0.5)
        high_volatility_percentile = config['pulse_width_calculator'].get('high_volatility_percentile', 75)
        low_volatility_percentile = config['pulse_width_calculator'].get('low_volatility_percentile', 25)
        max_threshold_factor = config['pulse_width_calculator'].get('max_threshold_factor', 0.3)
        baseline_subtraction_method = config['pulse_width_calculator'].get('baseline_subtraction_method', 'adaptive')
        fixed_baseline_value = config['pulse_width_calculator'].get('fixed_baseline_value', 0.01)
        trim_signal = config['pulse_width_calculator'].get('trim_signal', False)
        trim_dims_after_rise = config['pulse_width_calculator'].get('trim_dims_after_rise', 32)


    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found. Using default parameters.")
        noise_threshold = 0.03
        intensity_thresholds = [0.5, 0.7, 0.8, 0.9]
        selected_dims_after_rise_point = 32
        baseline_window_size = 10
        std_dev_multiplier = 3.0
        min_threshold_percentage = 0.01
        sustained_rise_points = 3
        lookback_window_size = 3
        use_adaptive_baseline_subtraction = True
        baseline_computation = 'median'
        sliding_window_min_size = 5
        sliding_window_max_size = 20
        quietness_metric = 'std'
        volatility_threshold = 0.5
        high_volatility_percentile = 75
        low_volatility_percentile = 25
        max_threshold_factor = 0.3
        baseline_subtraction_method = 'adaptive'
        fixed_baseline_value = 0.01
        trim_signal = False
        trim_dims_after_rise = 32

    except Exception as e:
        print(f"Error reading configuration file '{config_file}': {e}")
        print("Using default parameters.")
        noise_threshold = 0.03
        intensity_thresholds = [0.5, 0.7, 0.8, 0.9]
        selected_dims_after_rise_point = 32

    # Initialize and run the processor
    processor = TireSoundProcessor(
        input_dir=INPUT_DIRECTORY,
        standalone_dir=STANDALONE_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        noise_threshold=noise_threshold,
        intensity_thresholds=intensity_thresholds,
        selected_dims_after_rise_point=selected_dims_after_rise_point,
        baseline_window_size=baseline_window_size,
        std_dev_multiplier=std_dev_multiplier,
        min_threshold_percentage=min_threshold_percentage,
        sustained_rise_points=sustained_rise_points,
        lookback_window_size=lookback_window_size,
        use_adaptive_baseline_subtraction=use_adaptive_baseline_subtraction,
        baseline_computation=baseline_computation,
        sliding_window_min_size=sliding_window_min_size,
        sliding_window_max_size=sliding_window_max_size,
        quietness_metric=quietness_metric,
        volatility_threshold=volatility_threshold,
        high_volatility_percentile=high_volatility_percentile,
        low_volatility_percentile=low_volatility_percentile,
        max_threshold_factor=max_threshold_factor,
        baseline_subtraction_method=baseline_subtraction_method,
        fixed_baseline_value=fixed_baseline_value,
        trim_signal=trim_signal,
        trim_dims_after_rise=trim_dims_after_rise
    )

    # Start the processing
    processor.traverse_and_process()