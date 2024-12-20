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
                 zero_before_threshold: bool = False):
        
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
        self.zero_before_threshold = zero_before_threshold

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

    def apply_noise_floor_threshold(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Sets all values to zero BEFORE the first crossing of the noise threshold.
        
        Args:
            data (pd.DataFrame): Input signal data
            threshold (float): Noise threshold
            
        Returns:
            pd.DataFrame: Data with values before first threshold crossing set to zero
        """
        thresholded_data = data.copy()
        
        for idx in thresholded_data.index:
            row_values = thresholded_data.loc[idx]
            # Find first index where value exceeds threshold
            threshold_crossing = np.where(row_values > threshold)[0]
            if len(threshold_crossing) > 0:
                first_crossing = threshold_crossing[0]
                # Zero out everything before the threshold crossing
                if first_crossing > 0:  # Only if there are values before crossing
                    thresholded_data.loc[idx, thresholded_data.columns[:first_crossing]] = 0
                    
        return thresholded_data

    def apply_first_rise_threshold(self, data: pd.DataFrame, first_rise_indices: pd.Series) -> pd.DataFrame:
        """
        Sets all values to zero BEFORE the first rise point for each segment.
        
        Args:
            data (pd.DataFrame): Input signal data
            first_rise_indices (pd.Series): Series containing first rise index for each segment
            
        Returns:
            pd.DataFrame: Data with values before first rise point set to zero
        """
        thresholded_data = data.copy()
        
        for idx in thresholded_data.index:
            first_rise = first_rise_indices.get(idx)
            if pd.notna(first_rise):
                # Convert to 0-based index and ensure it's an integer
                first_rise_idx = int(first_rise) - 1
                # Zero out everything before the first rise point
                if first_rise_idx > 0:  # Only if there are values before first rise
                    thresholded_data.loc[idx, thresholded_data.columns[:first_rise_idx]] = 0
                    
        return thresholded_data

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

        # Step2: Apply noise threshold to normalized data before cumsum
        if self.zero_before_threshold:
            df_step2_base = self.apply_noise_floor_threshold(df_step1, self.noise_threshold)
            df_step2 = df_step2_base.cumsum(axis=1)
        else:
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
        Calculate dynamic noise threshold with enhanced validation and parameters.
        Uses configured parameters from config.yaml for calculations.
        """
        try:
            # Get baseline using configured window size
            baseline = segment_values.iloc[:self.baseline_window_size]
            
            # Calculate baseline statistics
            baseline_mean = baseline.mean()
            baseline_std = baseline.std()
            
            # Calculate signal statistics
            signal_range = segment_values.max() - segment_values.min()
            
            # Make threshold more sensitive for initial detection
            statistical_threshold = baseline_mean + (self.std_dev_multiplier * baseline_std * 0.5)  # Reduced multiplier
            minimum_threshold = signal_range * self.min_threshold_percentage
            
            # Calculate moving average for validation
            window_size = 3
            moving_avg = pd.Series(segment_values).rolling(window=window_size, center=True).mean()
            
            # If moving average shows significant trend, lower threshold further
            if not moving_avg.empty and moving_avg.std() > baseline_std:
                statistical_threshold *= 0.5
            
            # Use the larger of the thresholds, but ensure it's not too high
            dynamic_threshold = min(
                max(statistical_threshold, minimum_threshold),
                baseline_mean + (signal_range * 0.15)  # Cap at 15% of range
            )
            
            return dynamic_threshold
            
        except Exception as e:
            self.logger.warning(f"Error calculating dynamic threshold: {e}. Falling back to default threshold.")
            return self.noise_threshold

    def get_first_increase_index_standalone(self, row: pd.Series) -> float:
        """
        Enhanced first rise point detection that handles both gradual and sudden rises in signal.
        
        This method implements a multi-stage approach to detect the true start of a pulse:
        1. First checks for gradual rises using slope analysis
        2. Then looks for sustained rises above the threshold
        3. Finally falls back to a stricter threshold if needed
        
        The method is particularly careful to detect soft initial rises that precede 
        sudden increases in the signal.
        
        Args:
            row (pd.Series): A single segment's signal values
            
        Returns:
            float: The index of the first rise point (1-based indexing) or np.nan if no valid rise found
        """
        try:
            # Step 1: Calculate the dynamic noise threshold for this segment
            dynamic_threshold = self.calculate_dynamic_noise_threshold(row)
            
            # Get all points that exceed the threshold
            indices = np.where(row.values > dynamic_threshold)[0]
            
            if indices.size > 0:
                # Convert series to numpy array for faster computation
                values = row.values
                
                # Step 2: Primary analysis - Check each threshold crossing point
                for i in range(len(indices) - (self.sustained_rise_points - 1)):
                    first_rise_idx = indices[i]
                    
                    # Step 2a: Gradual Rise Detection
                    # Look backwards from threshold crossing to find potential gradual rise start
                    if first_rise_idx > 1:
                        # Define window to look back for gradual rise
                        lookback_window = self.lookback_window_size 
                        start_idx = max(0, first_rise_idx - lookback_window)
                        
                        # Calculate slopes in the lookback window
                        # Positive slopes indicate increasing values
                        slopes = np.diff(values[start_idx:first_rise_idx + 1])
                        
                        # Check if we have a consistent increase (all slopes positive)
                        if np.all(slopes > 0):
                            # Verify this is actually part of the main pulse by checking
                            # if subsequent points maintain the rise
                            next_points = values[first_rise_idx:first_rise_idx + self.sustained_rise_points]
                            if np.all(next_points > dynamic_threshold):
                                # Found a valid gradual rise - return its starting point
                                return start_idx + 1  # Convert to 1-based indexing
                    
                    # Step 2b: Sustained Rise Detection
                    # Check if we have the required number of consecutive points above threshold
                    is_sustained = True
                    for j in range(1, self.sustained_rise_points):
                        if indices[i+j] != indices[i] + j:
                            is_sustained = False
                            break
                    
                    if is_sustained:
                        # Additional check for any rise before this sustained rise
                        if first_rise_idx > 1:
                            # Look at two points before the sustained rise
                            pre_rise_values = values[max(0, first_rise_idx-2):first_rise_idx]
                            # If values were already increasing, start from earlier point
                            if np.all(np.diff(pre_rise_values) > 0):
                                return first_rise_idx - 1
                        
                        return first_rise_idx + 1  # Convert to 1-based indexing
                
                # Step 3: Fallback Analysis - Stricter Threshold
                # If no clear rise found, use a stricter threshold as last resort
                stricter_threshold = dynamic_threshold * 1.2
                stricter_indices = np.where(values > stricter_threshold)[0]
                
                if stricter_indices.size > 0:
                    first_strict_idx = stricter_indices[0]
                    # Still check for gradual rise before strict threshold crossing
                    if first_strict_idx > 1:
                        pre_strict_values = values[max(0, first_strict_idx-2):first_strict_idx]
                        if np.all(np.diff(pre_strict_values) > 0):
                            return first_strict_idx - 1
                    return first_strict_idx + 1
                    
            # If no valid rise point found, return NaN
            return np.nan
                
        except Exception as e:
            self.logger.error(f"Error in dynamic threshold detection for segment {row.name}: {e}")
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
        """
        try:
            # Read the CSV file with metadata columns
            df = pd.read_csv(file_path)
            metadata_columns = ['Tire Number', 'Pressure', 'TireSize', 'Ver', 'Wear', 'Rim']
            
            # Extract metadata from the first row
            metadata = {col: df[col].iloc[0] for col in metadata_columns}
            
            # Process signal values (columns after metadata)
            signal_data = df.iloc[:, 7:]  # Skip the first 7 columns (metadata)
            df_signals = signal_data.copy()
            df_signals.index = df['Segment ID / Value index']
            
            # Convert to numeric and normalize
            df_signals = df_signals.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Step1: Normalize (keep original)
            step1_normalized = df_signals.div(df_signals.sum(axis=1).replace(0, 1), axis=0)
            
            if self.zero_before_threshold:
                # Calculate first rise points
                first_rise_indices = df_signals.apply(
                    lambda row: self.get_first_increase_index_standalone(row), axis=1
                )
            
                # Step2: Apply first rise point thresholding before cumsum
                step2_base = self.apply_first_rise_threshold(step1_normalized, first_rise_indices)
                step2_cumulative = step2_base.cumsum(axis=1)
            else:
                step2_cumulative = step1_normalized.cumsum(axis=1)
            
            # Set index names
            step1_normalized.index.name = 'Segment_ID'
            step2_cumulative.index.name = 'Segment_ID'

            # CSV Filename format
            new_filename = f"{metadata['Tire Number']}_{metadata['TireSize']}-{metadata['Pressure']}-{metadata['Ver']}"

            # Create output path
            output_xlsx_path = (self.processed_standalone_dir / new_filename).with_suffix('.xlsx')
            output_xlsx_path.parent.mkdir(parents=True, exist_ok=True)

            # Write Results
            with pd.ExcelWriter(output_xlsx_path, engine='xlsxwriter') as writer:
                # Save Step1 and Step2
                step1_normalized.to_excel(writer, sheet_name='Step1_Data')
                step2_cumulative.to_excel(writer, sheet_name='Step2_Sj')
                
                # For each intensity threshold, compute and save step3 data
                for threshold in self.intensity_thresholds:
                    step3_data = self.compute_standalone_step3_metrics(
                        step2_cumulative, threshold, metadata
                    )
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
            'Ver': metadata['Ver'],
            'Wear': metadata['Wear'],
            'Rim': metadata['Rim']
        }, index=df_cumulative.index)

        return step3_data_points

    def calculate_standalone_median_pulse_width(self, excel_file_paths) -> pd.DataFrame:
        """
        Calculates the median pulse width for standalone tire data including all metadata.
        """
        median_pulse_widths = []

        for file_path in excel_file_paths:
            try:
                if file_path.name.startswith('~$'):
                    continue
                xls = pd.ExcelFile(file_path)

                for threshold in self.intensity_thresholds:
                    sheet_name = f'Step3_DataPts_{threshold}'
                    if sheet_name not in xls.sheet_names:
                        reason = f'Sheet {sheet_name} not found'
                        self.add_exclusion_entry(file_path, reason)
                        self.logger.warning(f"{file_path.name}: {reason}")
                        continue

                    df_step3 = pd.read_excel(file_path, sheet_name=sheet_name, index_col='Segment_ID')
                    
                    valid_pulse_widths = df_step3['Pulse_Width'].dropna()
                    if len(valid_pulse_widths) == 0:
                        continue

                    pulse_widths = valid_pulse_widths.sample(n=min(5, len(valid_pulse_widths)), random_state=42)
                    median_pulse_width = pulse_widths.median()

                    # Get all metadata from first row
                    metadata = {
                        'Tire_Number': df_step3['Tire_Number'].iloc[0],
                        'Pressure': df_step3['Pressure'].iloc[0],
                        'TireSize': df_step3['TireSize'].iloc[0],
                        'Ver': df_step3['Ver'].iloc[0],
                        'Wear': df_step3['Wear'].iloc[0],
                        'Rim': df_step3['Rim'].iloc[0]
                    }

                    median_pulse_widths.append({
                        'File_Name': str(file_path.relative_to(self.processed_standalone_dir)),
                        'Intensity_Threshold': threshold,
                        'Median_Pulse_Width': median_pulse_width,
                        **metadata
                    })

            except Exception as e:
                self.logger.error(f"Error processing file '{file_path.name}': {e}")
                continue

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
        Process all standalone tire files
        """
        self.logger.info(f"Processing standalone files from: {self.standalone_dir}")
        print(f"Processing standalone files from: {self.standalone_dir}")

        # Get all CSV files in the standalone directory
        csv_files = list(self.standalone_dir.glob('*.csv'))
        total_files = len(csv_files)
        
        self.logger.info(f"Total standalone CSV files found: {total_files}")
        print(f"Total standalone CSV files found: {total_files}")

        processed_files = []
        for file_path in tqdm(csv_files, desc="Processing standalone files", unit="file"):
            try:
                output_path = self.process_standalone_file(file_path)
                processed_files.append(output_path)
            except Exception as e:
                self.logger.error(f"Error processing standalone file '{file_path.name}': {e}")
                print(f"Error processing standalone file '{file_path.name}': {e}")
                continue

        # Calculate and save median pulse widths for standalone files
        if processed_files:
            df_median_pulse_widths = self.calculate_standalone_median_pulse_width(processed_files)
            median_pulse_widths_output = self.processed_standalone_dir / 'Median_Pulse_Widths_StandAlone.xlsx'
            df_median_pulse_widths.to_excel(median_pulse_widths_output, index=False)
            self.logger.info(f"Standalone median pulse widths saved to {median_pulse_widths_output}")
            print(f"Standalone median pulse widths saved to {median_pulse_widths_output}")

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
        zero_before_threshold = config.get('pulse_width_calculator', {}).get('zero_before_threshold', False)

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
        zero_before_threshold = False

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
        zero_before_threshold=zero_before_threshold
    )

    # Start the processing
    processor.traverse_and_process()