"""
Tire Sound Data Processing System
--------------------------------
This module provides functionality for processing and analyzing tire sound data,
including both mounted and standalone tire configurations. It handles data preprocessing,
signal analysis, and metrics calculation for tire pressure analysis.

Main components:
- Configuration management
- Signal processing
- Baseline subtraction
- Metrics calculation
- File handling
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union, Any


class ProcessingError(Exception):
    """Custom exception for processing-related errors."""
    pass

def load_config(config_file: str = 'config.yaml') -> Dict:
    """
    Load and validate configuration from YAML file.
    
    This function reads the configuration file and ensures all required parameters
    are present with valid values. If the config file is missing or invalid,
    it falls back to default configuration values.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Returns:
        Dict: Validated configuration dictionary with all required parameters
        
    Raises:
        ValueError: If configuration validation fails
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
            
            validate_config(loaded_config)
            return loaded_config
            
    except Exception as e:
        print(f"Using default parameters ({str(e)})")
        return get_default_config()

def get_default_config() -> Dict:
    """
    Provide default configuration values for the system.
    
    This function defines and returns the default configuration parameters
    when a configuration file is not available or is invalid.
    
    Returns:
        Dict: Default configuration dictionary with all required parameters
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

def validate_config(config: Dict) -> None:
    """
    Validate configuration parameters against required constraints.
    
    This function checks all configuration parameters to ensure they meet
    the required constraints and have valid values. It validates numerical
    ranges and ensures method choices are valid.
    
    Args:
        config (Dict): Configuration dictionary to validate
        
    Raises:
        ValueError: If any configuration parameter is invalid
    """
    for section in ['mounted', 'standalone']:
        cfg = config.get(section, {})
        
        # Validate numeric ranges
        _validate_numeric_ranges(section, cfg)
        
        # Validate method choices
        _validate_method_choices(section, cfg)

def _validate_numeric_ranges(section: str, cfg: Dict) -> None:
    """
    Validate numerical parameters in configuration.
    
    Args:
        section (str): Configuration section name ('mounted' or 'standalone')
        cfg (Dict): Configuration dictionary for the section
        
    Raises:
        ValueError: If any numerical parameter is outside its valid range
    """
    if cfg.get('baseline_window_size', 0) <= 0:
        raise ValueError(f"{section}: baseline_window_size must be greater than 0")
    
    if not 0 <= cfg.get('volatility_threshold', 0) <= 1:
        raise ValueError(f"{section}: volatility_threshold must be between 0 and 1")
    
    if cfg.get('high_volatility_percentile', 0) <= cfg.get('low_volatility_percentile', 0):
        raise ValueError(f"{section}: high_volatility_percentile must be greater than low_volatility_percentile")
    
    if not 0 <= cfg.get('min_threshold_percentage', 0) <= 1:
        raise ValueError(f"{section}: min_threshold_percentage must be between 0 and 1")

def _validate_method_choices(section: str, cfg: Dict) -> None:
    """
    Validate method choice parameters in configuration.
    
    Args:
        section (str): Configuration section name ('mounted' or 'standalone')
        cfg (Dict): Configuration dictionary for the section
        
    Raises:
        ValueError: If any method choice is invalid
    """
    validations = {
        'noise_detection_method': ['dynamic', 'fixed'],
        'baseline_subtraction_method': ['none', 'fixed', 'adaptive'],
        'baseline_computation': ['mean', 'median'],
        'quietness_metric': ['std', 'mad', 'mix']
    }
    
    for param, valid_choices in validations.items():
        if cfg.get(param) not in valid_choices:
            raise ValueError(f"{section}: {param} must be one of {valid_choices}")
        

class TireSoundProcessor:
    """
    Core processor for tire sound data analysis.
    
    This class handles the processing and analysis of tire sound data for both mounted
    and standalone tire configurations. It implements a complete pipeline from data
    loading through signal processing to metrics calculation.
    
    The processing pipeline includes:
    1. Signal preprocessing and normalization
    2. Baseline subtraction (fixed or adaptive)
    3. Rise point detection
    4. Pulse width calculation
    5. Metrics computation and aggregation
    
    Attributes:
        common_config (Dict): Common configuration parameters
        mounted_config (Dict): Configuration for mounted tire processing
        standalone_config (Dict): Configuration for standalone tire processing
        dir_config (Dict): Directory paths configuration
        input_dir (Path): Input directory for mounted tire data
        standalone_dir (Path): Input directory for standalone tire data
        output_dir (Path): Base output directory
        processed_dir (Path): Output directory for processed mounted data
        processed_standalone_dir (Path): Output directory for processed standalone data
        intensity_thresholds (List[float]): Thresholds for pulse width calculation
        logger (logging.Logger): Logger instance for the processor
    """


    # Section 1: Initialization and Configuration
    def __init__(self, config: Dict) -> None:
        """
        Initialize the TireSoundProcessor with configuration parameters.
        
        Sets up the processing environment, including directory structures,
        configuration parameters, and logging system.
        
        Args:
            config (Dict): Complete configuration dictionary containing:
                - directories: Directory paths configuration
                - common: Common processing parameters
                - mounted: Mounted tire processing parameters
                - standalone: Standalone tire processing parameters
                
        Raises:
            ValueError: If required configuration parameters are missing
            FileExistsError: If unable to create required directories
        """
        # Initialize configurations
        self._initialize_configs(config)
        
        # Set up directory structure
        self._setup_directories()
        
        # Configure logging
        self._setup_logging()
        
        # Log configuration summary
        self._log_configuration()

    def _initialize_configs(self, config: Dict) -> None:
        """
        Initialize configuration parameters from input dictionary.
        
        Args:
            config (Dict): Input configuration dictionary
            
        Raises:
            ValueError: If required configuration sections are missing
        """
        # Store configurations with defaults
        self.common_config = config.get('common', {})
        self.mounted_config = config.get('mounted', {})
        self.standalone_config = config.get('standalone', {})
        self.dir_config = config.get('directories', {})
        
        # Common parameters
        self.intensity_thresholds = self.common_config.get(
            'intensity_thresholds', 
            [0.5, 0.7, 0.8, 0.9]
        )

    def _setup_directories(self) -> None:
        """
        Set up directory structure for data processing.
        
        Creates necessary directories and validates their existence.
        
        Raises:
            FileExistsError: If unable to create required directories
        """
        # Directory paths from config with defaults
        self.input_dir = Path(self.dir_config.get('input_dir', '.'))
        self.standalone_dir = Path(self.dir_config.get('standalone_dir', '.'))
        self.output_dir = Path(self.dir_config.get('output_dir', '.'))
        
        # Output subdirectories from config
        mounted_subdir = self.dir_config.get('output_mounted_subdir', 'Processed_Mounted')
        standalone_subdir = self.dir_config.get('output_standalone_subdir', 'Processed_Standalone')
        
        # Create full output directory paths
        self.processed_dir = self.output_dir / mounted_subdir
        self.processed_standalone_dir = self.output_dir / standalone_subdir
        
        # Create output directories if they don't exist
        for directory in [self.processed_dir, self.processed_standalone_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileExistsError(f"Failed to create directory {directory}: {str(e)}")

    def _setup_logging(self) -> None:
        """
        Configure logging system for the processor.
        
        Sets up logging with appropriate format and file handler.
        """
        logging.basicConfig(
            filename='processing.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()

    def _log_configuration(self) -> None:
        """
        Log current configuration and directory setup.
        
        Provides a summary of the current processing configuration
        and directory structure for debugging purposes.
        """
        self.logger.info("=== Processing Configuration ===")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Standalone directory: {self.standalone_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Processed mounted directory: {self.processed_dir}")
        self.logger.info(f"Processed standalone directory: {self.processed_standalone_dir}")
        self.logger.info(f"Intensity thresholds: {self.intensity_thresholds}")
        self.logger.info("============================")


    # Section 2: Main Processing Pipeline
    def process_all_files(self) -> None:
        """
        Main entry point for processing all tire data files.
        
        This method orchestrates the complete processing pipeline for both mounted
        and standalone tire data. The process includes:
        1. Processing mounted tire data
        2. Processing standalone tire data
        3. Saving exclusion reports
        
        Any failures in individual file processing are logged and tracked
        but don't stop the overall processing.
        """
        try:
            # Process mounted tire data
            self.logger.info("Starting mounted tire data processing")
            self.process_mounted_files()
            
            # Process standalone tire data
            self.logger.info("Starting standalone tire data processing")
            self.process_standalone_files()
            
            # Save exclusions report
            self.save_exclusions()
            self.logger.info("Processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in main processing pipeline: {str(e)}")
            raise RuntimeError("Processing pipeline failed") from e

    def process_mounted_files(self) -> None:
        """
        Process all mounted tire data files.
        
        Handles the processing of mounted tire CSV files, including:
        1. File discovery and validation
        2. Individual file processing
        3. Median pulse width calculation
        
        Files that fail processing are logged and tracked in exclusions.
        """
        self.logger.info(f"Discovering mounted tire files in: {self.input_dir}")
        
        # Find all CSV files recursively
        csv_files = list(self.input_dir.rglob('*.csv'))
        self.logger.info(f"Found {len(csv_files)} CSV files")
        
        # Process each file
        processed_files = []
        for file_path in tqdm(csv_files, desc="Processing mounted tire files"):
            try:
                if file_path.name.startswith('~$'):
                    continue
                    
                output_path = self._get_output_path(file_path)
                success = self.process_single_mounted_file(file_path, output_path)
                
                if success:
                    processed_files.append(output_path)
                    
            except Exception as e:
                self.logger.error(f"Error processing '{file_path.name}': {str(e)}")
                continue

        # Calculate and save median pulse widths
        if processed_files:
            self._save_mounted_summary(processed_files)
        else:
            self.logger.warning("No mounted files were successfully processed")

    def process_standalone_files(self) -> None:
        """
        Process all standalone tire data files.
        
        Handles the processing of standalone tire CSV files, including:
        1. File discovery and validation
        2. Individual file processing
        3. Median pulse width calculation
        
        Files that fail processing are logged and tracked in exclusions.
        """
        self.logger.info(f"Discovering standalone files in: {self.standalone_dir}")
        
        # Find all CSV files
        csv_files = list(self.standalone_dir.glob('*.csv'))
        if not csv_files:
            self.logger.warning("No CSV files found in standalone directory")
            return
        
        # Process each file
        processed_files = []
        for file_path in tqdm(csv_files, desc="Processing standalone files"):
            try:
                if file_path.name.startswith('~$'):
                    continue
                    
                output_path = self.process_single_standalone_file(file_path)
                if output_path and output_path.exists():
                    processed_files.append(output_path)
                    self.logger.info(f"Successfully processed: {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing '{file_path.name}': {str(e)}")
                continue

        # Calculate and save summary statistics
        self._save_standalone_summary(processed_files)

    def process_single_mounted_file(self, file_path: Path, output_path: Path) -> bool:
        """
        Process a single mounted tire data file.
        
        Performs the complete processing pipeline for one mounted tire file:
        1. Signal preprocessing and normalization
        2. Baseline subtraction
        3. Metrics calculation
        4. Results saving
        
        Args:
            file_path (Path): Path to the input CSV file
            output_path (Path): Path for saving processed results
            
        Returns:
            bool: True if processing was successful, False otherwise
            
        Raises:
            ValueError: If file format is invalid
            ProcessingError: If processing steps fail
        """
        try:
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process base data with metadata
            step1_normalized, step2_cumulative, metadata_dict = self.normalize_signal(
                file_path, mounted=True
            )
            
            # Prepare results data structure
            results_data = {
                'Step1_Data': step1_normalized,
                'Step2_Sj': step2_cumulative
            }
            
            # Add metrics for each threshold
            for threshold in self.intensity_thresholds:
                metrics = self.compute_pulse_metrics(
                    step2_cumulative, threshold, file_path, metadata_dict
                )
                results_data[f'Step3_DataPts_{threshold}'] = metrics

            # Save results
            success = self.save_to_excel(output_path, results_data)
            
            if not success:
                raise ProcessingError("Failed to save output files")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing mounted file {file_path}: {str(e)}")
            return False

    def process_single_standalone_file(self, file_path: Path) -> Optional[Path]:
        """
        Process a single standalone tire data file.
        
        Performs the complete processing pipeline for one standalone tire file:
        1. Metadata extraction and validation
        2. Signal preprocessing
        3. Metrics calculation
        4. Results saving
        
        Args:
            file_path (Path): Path to the input CSV file
            
        Returns:
            Optional[Path]: Path to the output file if successful, None otherwise
            
        Raises:
            ValueError: If required metadata columns are missing
            ProcessingError: If processing steps fail
        """
        try:
            # Extract and validate metadata
            metadata = self.extract_metadata(file_path, standalone=True)
            
            # Generate output filename
            output_path = self._generate_standalone_output_path(metadata)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process signal data
            signal_data = self._process_standalone_signal(file_path, metadata)
            
            # Save results
            self._save_standalone_results(output_path, signal_data, metadata)
            
            return output_path
            
        except Exception as e:
            error_msg = f"Failed to process standalone file {file_path.name}: {str(e)}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg) from e

    def _get_output_path(self, input_path: Path) -> Path:
        """
        Generate output path for processed file.
        
        Args:
            input_path (Path): Input file path
            
        Returns:
            Path: Generated output path
        """
        relative_path = input_path.relative_to(self.input_dir)
        return self.processed_dir / relative_path.parent / input_path.stem.with_suffix('.xlsx')

    def _process_standalone_signal(self, file_path: Path, metadata: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Process signal data from standalone tire file.
        
        Args:
            file_path (Path): Input file path
            metadata (Dict[str, Any]): File metadata
            
        Returns:
            Dict[str, pd.DataFrame]: Processed signal data
        """
        try:
            # Read and process signal data
            df = pd.read_csv(file_path)
            signal_data = df.iloc[:, len(metadata) + 1:]  # Skip metadata columns
            df_signals = signal_data.copy()
            df_signals.index = df['Segment ID / Value index']
            
            # Convert to numeric and handle missing values
            df_signals = df_signals.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Process signals
            step1_normalized, step2_cumulative = self.normalize_signal(df_signals, mounted=False)
            
            # Prepare results
            results = {
                'Step1_Data': step1_normalized,
                'Step2_Sj': step2_cumulative
            }
            
            # Add metrics for each threshold
            for threshold in self.intensity_thresholds:
                metrics = self.compute_standalone_metrics(
                    step2_cumulative, threshold, metadata
                )
                results[f'Step3_DataPts_{threshold}'] = metrics
                
            return results
            
        except Exception as e:
            raise ProcessingError(f"Signal processing failed: {str(e)}")

    def _save_standalone_results(self, output_path: Path, 
                            signal_data: Dict[str, pd.DataFrame],
                            metadata: Dict[str, Any]) -> None:
        """
        Save processed standalone results.
        
        Args:
            output_path (Path): Output file path
            signal_data (Dict[str, pd.DataFrame]): Processed signal data
            metadata (Dict[str, Any]): File metadata
        """
        try:
            sheet_order = [
                'Step1_Data',
                'Step2_Sj',
                *[f'Step3_DataPts_{threshold}' 
                for threshold in self.intensity_thresholds]
            ]
            
            success = self.save_to_excel(output_path, signal_data, sheet_order)
            
            if not success:
                raise ProcessingError("Failed to save standalone results")
                
        except Exception as e:
            raise ProcessingError(f"Failed to save results: {str(e)}")

    def _generate_standalone_output_path(self, metadata: Dict[str, Any]) -> Path:
        """
        Generate output path for standalone file.
        
        Args:
            metadata (Dict[str, Any]): File metadata
            
        Returns:
            Path: Generated output path
        """
        filename = self.generate_output_filename(metadata, mounted=False)
        return self.processed_standalone_dir / f"{filename}.xlsx"


    # Section 3: Signal Processing
    def normalize_signal(self, df_signals: pd.DataFrame, mounted: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize signal data and compute cumulative sums.
        
        Performs the following steps:
        1. Baseline subtraction (if configured)
        2. Signal trimming (if enabled)
        3. Normalization to unit sum
        4. Cumulative sum calculation
        
        Args:
            df_signals (pd.DataFrame): Raw signal data
            mounted (bool): Whether processing mounted tire data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - Normalized signal data
                - Cumulative sum data
                
        Raises:
            ValueError: If signal data is invalid
        """
        try:
            config = self.mounted_config if mounted else self.standalone_config
            
            # Apply baseline subtraction if configured
            df_processed = self._apply_baseline_subtraction(df_signals, config)
            
            # Apply signal trimming if enabled
            if config['trim_signal']:
                self.logger.info(f"Trimming signals to {config['trim_dims_after_rise']} dimensions")
                df_processed = self.trim_signal_at_rise(df_processed, mounted)
            
            # Normalize to unit sum
            row_sums = df_processed.sum(axis=1)
            row_sums = row_sums.where(row_sums != 0, 1)  # Handle zero sums
            df_normalized = df_processed.div(row_sums, axis=0)
            
            # Calculate cumulative sum
            df_cumulative = df_normalized.cumsum(axis=1)
            
            return df_normalized, df_cumulative
            
        except Exception as e:
            raise ValueError(f"Signal normalization failed: {str(e)}") from e

    def detect_rise_point(self, row: pd.Series, mounted: bool = False) -> float:
        """
        Detect the first significant rise point in a signal.
        
        Uses configured method (dynamic or fixed threshold) to identify
        the point where the signal starts rising significantly.
        
        Args:
            row (pd.Series): Signal values
            mounted (bool): Whether processing mounted tire data
            
        Returns:
            float: Index of rise point (1-based) or NaN if not found
        """
        try:
            config = self.mounted_config if mounted else self.standalone_config
            noise_method = config['noise_detection_method']
            
            # Get appropriate threshold
            threshold = (self.calculate_dynamic_threshold(row, mounted) 
                       if noise_method == 'dynamic' 
                       else config['noise_threshold'])
            
            # Find points above threshold
            values = row.values
            crossings = np.where(values > threshold)[0]
            
            if len(crossings) == 0:
                return np.nan
            
            return self._validate_rise_point(
                values, crossings, threshold, config
            )
            
        except Exception as e:
            self.logger.error(f"Rise point detection failed: {e}")
            return np.nan

    def _validate_rise_point(self, values: np.ndarray, 
                           crossings: np.ndarray,
                           threshold: float, 
                           config: Dict) -> float:
        """
        Validate potential rise points against sustained rise criteria.
        
        Args:
            values (np.ndarray): Signal values
            crossings (np.ndarray): Indices where threshold is crossed
            threshold (float): Detection threshold
            config (Dict): Configuration parameters
            
        Returns:
            float: Validated rise point index (1-based) or NaN
        """
        sustained_points = config['sustained_rise_points']
        lookback_window = config['lookback_window_size']
        
        for start_idx in crossings:
            # Skip if too close to end
            if start_idx + sustained_points >= len(values):
                continue
            
            # Check for sustained rise
            if self._check_sustained_rise(
                values, start_idx, threshold, sustained_points, lookback_window
            ):
                return start_idx + 1  # 1-based indexing
        
        return np.nan

    def _check_sustained_rise(self, values: np.ndarray, 
                            start_idx: int,
                            threshold: float,
                            sustained_points: int,
                            lookback_window: int) -> bool:
        """
        Check if a potential rise point meets sustained rise criteria.
        
        Args:
            values (np.ndarray): Signal values
            start_idx (int): Starting index to check
            threshold (float): Detection threshold
            sustained_points (int): Required number of sustained points
            lookback_window (int): Window size for gradual rise check
            
        Returns:
            bool: True if criteria are met
        """
        # Check sustained rise
        sustained_segment = values[start_idx:start_idx + sustained_points]
        if not np.all(sustained_segment > threshold):
            return False
            
        # Check for gradual rise if not at start
        if start_idx > 0:
            lookback_start = max(0, start_idx - lookback_window)
            lookback_values = values[lookback_start:start_idx + 1]
            
            # Check for monotonic increase
            differences = np.diff(lookback_values)
            if np.all(differences >= 0):
                return True
        
        return True

    def calculate_dynamic_threshold(self, segment_values: pd.Series, 
                                 mounted: bool = False) -> float:
        """
        Calculate dynamic noise threshold based on signal characteristics.
        
        Uses multiple methods to determine an appropriate threshold:
        1. Basic statistics (mean, std, median)
        2. Signal volatility analysis
        3. Range-based thresholding
        
        Args:
            segment_values (pd.Series): Signal values
            mounted (bool): Whether processing mounted tire data
            
        Returns:
            float: Calculated threshold value
        """
        try:
            config = self.mounted_config if mounted else self.standalone_config
            
            # Validate signal length
            if len(segment_values) < config['baseline_window_size']:
                return config['noise_threshold']
            
            # Calculate basic statistics
            threshold_candidates = self._calculate_threshold_candidates(
                segment_values, config
            )
            
            # Analyze signal characteristics
            volatility = self._calculate_signal_volatility(segment_values, config)
            
            # Select appropriate threshold
            return self._select_final_threshold(
                threshold_candidates, volatility, segment_values, config
            )
            
        except Exception as e:
            self.logger.error(f"Dynamic threshold calculation failed: {e}")
            return config['noise_threshold']

    def trim_signal_at_rise(self, df_signals: pd.DataFrame, 
                          mounted: bool = True) -> pd.DataFrame:
        """
        Trim signal data relative to detected rise points.
        
        Trims each signal segment based on its rise point while maintaining
        alignment across segments. Handles cases where rise points vary
        across segments.
        
        Args:
            df_signals (pd.DataFrame): Signal data to trim
            mounted (bool): Whether processing mounted tire data
            
        Returns:
            pd.DataFrame: Trimmed signal data
        """
        try:
            config = self.mounted_config if mounted else self.standalone_config
            trim_dims = config['trim_dims_after_rise']
            
            # Find rise points for all segments
            rise_points = self._find_all_rise_points(df_signals, mounted)
            
            if not rise_points:
                return df_signals
                
            # Create trimmed DataFrame
            return self._create_trimmed_dataframe(
                df_signals, rise_points, trim_dims
            )
            
        except Exception as e:
            self.logger.error(f"Signal trimming failed: {e}")
            return df_signals
    

    # Section 4: Baseline Processing
    def apply_baseline_subtraction(self, df_signals: pd.DataFrame, mounted: bool = False) -> pd.DataFrame:
        """
        Apply the configured baseline subtraction method to signal data.
        
        Acts as a dispatcher for different baseline subtraction methods based on
        configuration settings (fixed, adaptive, or none).
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            mounted (bool): Whether processing mounted tire data
            
        Returns:
            pd.DataFrame: Baseline-corrected signal data
            
        Raises:
            ValueError: If baseline subtraction method is invalid
        """
        try:
            config = self.mounted_config if mounted else self.standalone_config
            method = config['baseline_subtraction_method']
            
            if method == 'fixed':
                return self.apply_fixed_baseline(df_signals, mounted)
            elif method == 'adaptive':
                return self.apply_adaptive_baseline(df_signals, mounted)
            elif method == 'none':
                self.logger.info("No baseline subtraction applied")
                return df_signals.copy()
            else:
                raise ValueError(f"Invalid baseline subtraction method: {method}")
                
        except Exception as e:
            self.logger.error(f"Baseline subtraction failed: {str(e)}")
            return df_signals

    def apply_fixed_baseline(self, df_signals: pd.DataFrame, mounted: bool = False) -> pd.DataFrame:
        """
        Apply fixed-value baseline subtraction to signal data.
        
        Subtracts a fixed baseline value from all signals and clips negative
        values to zero or a small positive number.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            mounted (bool): Whether processing mounted tire data
            
        Returns:
            pd.DataFrame: Signal data with fixed baseline subtracted
        """
        try:
            config = self.mounted_config if mounted else self.standalone_config
            baseline_value = config['fixed_baseline_value']
            
            # Create working copy
            df_modified = df_signals.copy()
            
            # Subtract baseline
            df_modified = df_modified.subtract(baseline_value)
            
            # Clip negative values to small positive number
            EPSILON = 1e-10  # Small positive number to avoid exact zeros
            df_modified = df_modified.clip(lower=EPSILON)
            
            return df_modified
            
        except Exception as e:
            self.logger.error(f"Fixed baseline subtraction failed: {str(e)}")
            return df_signals

    def apply_adaptive_baseline(self, df_signals: pd.DataFrame, mounted: bool = False) -> pd.DataFrame:
        """
        Apply adaptive baseline subtraction to signal data.
        
        Uses sliding window analysis to determine and subtract an optimal
        baseline value for each signal segment independently.
        
        Args:
            df_signals (pd.DataFrame): Input signal data
            mounted (bool): Whether processing mounted tire data
            
        Returns:
            pd.DataFrame: Signal data with adaptive baseline subtracted
        """
        try:
            config = self.mounted_config if mounted else self.standalone_config
            df_modified = df_signals.copy()
            
            # Process each segment independently
            for idx in df_modified.index:
                try:
                    row_values = df_modified.loc[idx].values
                    baseline = self.find_adaptive_baseline(row_values, config)
                    
                    # Subtract baseline and clip to small positive number
                    EPSILON = 1e-10
                    row_corrected = np.maximum(row_values - baseline, EPSILON)
                    
                    df_modified.loc[idx] = row_corrected
                    
                except Exception as e:
                    self.logger.error(f"Error processing row {idx}: {str(e)}")
                    continue
            
            return df_modified
            
        except Exception as e:
            self.logger.error(f"Adaptive baseline subtraction failed: {str(e)}")
            return df_signals

    def find_adaptive_baseline(self, row_values: np.ndarray, config: Dict) -> float:
        """
        Find optimal baseline value using sliding window analysis.
        
        Analyzes the signal to find the "quietest" segment and uses its
        statistics to determine an appropriate baseline value.
        
        Args:
            row_values (np.ndarray): Signal values for one segment
            config (Dict): Configuration parameters
            
        Returns:
            float: Optimal baseline value
            
        Raises:
            ValueError: If signal is too short for analysis
        """
        # Get window parameters
        min_size = min(config['sliding_window_min_size'], len(row_values))
        max_size = min(config['sliding_window_max_size'], len(row_values))
        
        if min_size <= 0:
            raise ValueError("Signal too short for adaptive baseline")
            
        return self._analyze_signal_windows(
            row_values, min_size, max_size, config
        )

    def _analyze_signal_windows(self, values: np.ndarray, 
                              min_size: int, 
                              max_size: int,
                              config: Dict) -> float:
        """
        Analyze signal windows to find optimal baseline value.
        
        Args:
            values (np.ndarray): Signal values
            min_size (int): Minimum window size
            max_size (int): Maximum window size
            config (Dict): Configuration parameters
            
        Returns:
            float: Optimal baseline value
        """
        best_metric = float('inf')
        best_baseline = 0.0
        
        # Slide through possible windows
        for start_idx in range(max_size - min_size + 1):
            window = values[start_idx:start_idx + min_size]
            
            # Calculate candidate baseline
            baseline = self._compute_baseline_candidate(window, config)
            
            # Evaluate window quietness
            metric = self._evaluate_window_quietness(window, config)
            
            # Update if better
            if metric < best_metric:
                best_metric = metric
                best_baseline = baseline
        
        return best_baseline

    def _compute_baseline_candidate(self, window: np.ndarray, config: Dict) -> float:
        """
        Compute baseline value for a window using configured method.
        
        Args:
            window (np.ndarray): Signal window values
            config (Dict): Configuration parameters
            
        Returns:
            float: Baseline value for window
        """
        if config['baseline_computation'] == 'median':
            return np.median(window)
        else:  # 'mean'
            return np.mean(window)

    def _evaluate_window_quietness(self, window: np.ndarray, config: Dict) -> float:
        """
        Evaluate how "quiet" a signal window is using configured metric.
        
        Args:
            window (np.ndarray): Signal window values
            config (Dict): Configuration parameters
            
        Returns:
            float: Quietness metric value (lower is better)
        """
        if config['quietness_metric'] == 'std':
            return np.std(window)
        elif config['quietness_metric'] == 'mad':
            return np.median(np.abs(window - np.median(window)))
        else:  # 'mix'
            std_val = np.std(window)
            mad_val = np.median(np.abs(window - np.median(window)))
            return (std_val + mad_val) / 2
    

    # Section 5: Metrics Calculation
    def compute_pulse_metrics(self, df_cumulative: pd.DataFrame, 
                            intensity_threshold: float, 
                            file_path: Path,
                            metadata_dict: dict) -> pd.DataFrame:
        """
        Compute comprehensive pulse metrics from cumulative signal data.
        
        Calculates various metrics including:
        - First noticeable increase point
        - Point exceeding threshold
        - Pulse width
        - Cumulative values at key points
        
        Args:
            df_cumulative (pd.DataFrame): Cumulative sum data
            intensity_threshold (float): Threshold for pulse detection
            file_path (Path): Source file path for metadata
            metadata_dict (dict): Additional metadata information
            
        Returns:
            pd.DataFrame: Computed metrics for all segments
            
        Raises:
            ValueError: If computation fails for any required metric
        """
        try:
            # Extract tire identification data
            tire_info = self._extract_tire_info(file_path)
            
            # Calculate key indices
            indices = self._calculate_key_indices(
                df_cumulative, intensity_threshold
            )
            
            # Calculate cumulative values
            cumulative_values = self._calculate_cumulative_values(
                df_cumulative, indices
            )
            
            # Calculate pulse widths
            pulse_metrics = self._calculate_pulse_widths(indices)
            
            # Combine metrics into DataFrame
            metrics_df = self._combine_metrics(
                tire_info, indices, cumulative_values, 
                pulse_metrics, intensity_threshold
            )
            
            # Add metadata
            self._add_metadata(metrics_df, metadata_dict)
            
            return metrics_df
            
        except Exception as e:
            raise ValueError(f"Metrics computation failed: {str(e)}")

    def _extract_tire_info(self, file_path: Path) -> Dict[str, str]:
        """
        Extract tire identification information from file path.
        
        Args:
            file_path (Path): Path to source file
            
        Returns:
            Dict[str, str]: Tire identification information
        """
        return {
            'air_pressure': self.extract_air_pressure(file_path.name),
            'tire_position': self.extract_tire_position(file_path.name),
            'wheel_type': self.extract_vehicle_type(file_path),
            'hitting_type': self.determine_hitting_type(file_path)
        }

    def _calculate_key_indices(self, df_cumulative: pd.DataFrame, 
                             threshold: float) -> Dict[str, pd.Series]:
        """
        Calculate key indices in signal data.
        
        Args:
            df_cumulative (pd.DataFrame): Cumulative sum data
            threshold (float): Detection threshold
            
        Returns:
            Dict[str, pd.Series]: Key signal indices
        """
        return {
            'first_increase': df_cumulative.apply(
                lambda row: self.detect_rise_point(row, mounted=True),
                axis=1
            ),
            'threshold_exceed': df_cumulative.apply(
                lambda row: self.find_threshold_crossing(row, threshold),
                axis=1
            )
        }

    def find_threshold_crossing(self, row: pd.Series, threshold: float) -> float:
        """
        Find point where signal first exceeds threshold.
        
        Args:
            row (pd.Series): Signal values
            threshold (float): Threshold value
            
        Returns:
            float: Index where threshold is exceeded (1-based) or NaN
        """
        try:
            indices = np.where(row.values > threshold)[0]
            return indices[0] + 1 if indices.size > 0 else np.nan
        except Exception as e:
            self.logger.error(f"Threshold crossing detection failed: {str(e)}")
            return np.nan

    def _calculate_cumulative_values(self, df_cumulative: pd.DataFrame, 
                                   indices: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate cumulative values at key indices.
        
        Args:
            df_cumulative (pd.DataFrame): Cumulative sum data
            indices (Dict[str, pd.Series]): Key signal indices
            
        Returns:
            Dict[str, pd.Series]: Cumulative values at key points
        """
        return {
            'first_increase_value': df_cumulative.apply(
                lambda row: self.get_value_at_index(
                    row, indices['first_increase'][row.name]
                ),
                axis=1
            ),
            'threshold_value': df_cumulative.apply(
                lambda row: self.get_value_at_index(
                    row, indices['threshold_exceed'][row.name]
                ),
                axis=1
            )
        }

    def get_value_at_index(self, row: pd.Series, idx: float) -> float:
        """
        Get value at specified index, handling edge cases.
        
        Args:
            row (pd.Series): Signal values
            idx (float): Index to get value for (1-based)
            
        Returns:
            float: Value at index or NaN if invalid
        """
        try:
            if pd.isna(idx):
                return np.nan
                
            idx = int(idx) - 1  # Convert to 0-based
            return row.iloc[idx] if 0 <= idx < len(row) else np.nan
            
        except Exception as e:
            self.logger.error(f"Value extraction failed: {str(e)}")
            return np.nan

    def _calculate_pulse_widths(self, indices: Dict[str, pd.Series]) -> pd.Series:
        """
        Calculate pulse widths from key indices.
        
        Args:
            indices (Dict[str, pd.Series]): Key signal indices
            
        Returns:
            pd.Series: Calculated pulse widths
        """
        pulse_width = indices['threshold_exceed'] - indices['first_increase']
        
        # Validate pulse widths
        invalid_widths = pulse_width[pulse_width < 0]
        if not invalid_widths.empty:
            self.logger.warning(
                f"Invalid pulse widths detected for {len(invalid_widths)} segments. "
                f"Setting to NaN."
            )
            pulse_width[pulse_width < 0] = np.nan
            
        return pulse_width

    def calculate_median_statistics(self, df_metrics: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate median statistics from pulse metrics.
        
        Args:
            df_metrics (pd.DataFrame): Pulse metrics data
            
        Returns:
            Dict[str, float]: Statistical summary
        """
        valid_widths = df_metrics['Pulse_Width'].dropna()
        
        if valid_widths.empty:
            return {
                'median_width': np.nan,
                'mean_width': np.nan,
                'std_width': np.nan,
                'valid_count': 0,
                'total_count': len(df_metrics)
            }
            
        return {
            'median_width': valid_widths.median(),
            'mean_width': valid_widths.mean(),
            'std_width': valid_widths.std(),
            'valid_count': len(valid_widths),
            'total_count': len(df_metrics)
        }

    def _combine_metrics(self, tire_info: Dict[str, str],
                        indices: Dict[str, pd.Series],
                        cumulative_values: Dict[str, pd.Series],
                        pulse_metrics: pd.Series,
                        intensity_threshold: float) -> pd.DataFrame:
        """
        Combine all metrics into a single DataFrame.
        
        Args:
            tire_info (Dict[str, str]): Tire identification info
            indices (Dict[str, pd.Series]): Key indices
            cumulative_values (Dict[str, pd.Series]): Cumulative values
            pulse_metrics (pd.Series): Pulse width metrics
            intensity_threshold (float): Used threshold value
            
        Returns:
            pd.DataFrame: Combined metrics
        """
        metrics_df = pd.DataFrame({
            'Intensity_Threshold': intensity_threshold,
            'First_Noticeable_Increase_Index': indices['first_increase'],
            'Point_Exceeds_Index': indices['threshold_exceed'],
            'First_Noticeable_Increase_Cumulative_Value': cumulative_values['first_increase_value'],
            'Point_Exceeds_Cumulative_Value': cumulative_values['threshold_value'],
            'Pulse_Width': pulse_metrics,
            'Air_Pressure': tire_info['air_pressure'],
            'Tire_Position': tire_info['tire_position'],
            'Wheel_Type': tire_info['wheel_type'],
            'Hitting_Type': tire_info['hitting_type']
        })
        
        return metrics_df.sort_index(axis=1)


    # Section 6: Metadata Extraction
    def extract_metadata(self, file_path: Path, standalone: bool = False) -> Dict[str, Any]:
        """
        Extract metadata information from file.
        
        Handles both mounted and standalone tire data formats, extracting
        relevant metadata fields according to the file type.
        
        Args:
            file_path (Path): Path to data file
            standalone (bool): Whether file is standalone format
            
        Returns:
            Dict[str, Any]: Extracted metadata
            
        Raises:
            ValueError: If required metadata fields are missing
        """
        try:
            df = pd.read_csv(file_path)
            
            if standalone:
                return self._extract_standalone_metadata(df)
            else:
                return self._extract_mounted_metadata(df)
                
        except Exception as e:
            raise ValueError(f"Metadata extraction failed: {str(e)}")

    def _extract_standalone_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract metadata from standalone tire data format.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Extracted metadata
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = [
            'Tire Number', 'Pressure', 'TireSize', 
            'Tire_Type', 'Wear', 'Rim'
        ]
        
        missing = [field for field in required_fields if field not in df.columns]
        if missing:
            raise ValueError(f"Missing required metadata fields: {missing}")
            
        return {field: df[field].iloc[0] for field in required_fields}

    def _extract_mounted_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract metadata from mounted tire data format.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Extracted metadata
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = [
            'Segment ID / Value index', 'Tire Number', 'Pressure',
            'TireSize', 'Tire_Type', 'Truck_Load'
        ]
        
        missing = [field for field in required_fields if field not in df.columns]
        if missing:
            raise ValueError(f"Missing required metadata fields: {missing}")
            
        metadata = df[required_fields].copy()
        return metadata.set_index('Segment ID / Value index').to_dict('index')

    def extract_air_pressure(self, file_name: str) -> float:
        """
        Extract air pressure value from file name.
        
        Searches for segment number in file name which corresponds
        to air pressure value.
        
        Args:
            file_name (str): Name of data file
            
        Returns:
            float: Extracted air pressure value or NaN if not found
        """
        try:
            match = re.search(r'segment\s*(\d+)', file_name, re.IGNORECASE)
            return float(match.group(1)) if match else np.nan
            
        except Exception as e:
            self.logger.warning(f"Air pressure extraction failed: {str(e)}")
            return np.nan

    def extract_tire_position(self, file_name: str) -> str:
        """
        Extract tire position from file name.
        
        Searches for position identifier in file name.
        
        Args:
            file_name (str): Name of data file
            
        Returns:
            str: Extracted position or 'Unknown' if not found
        """
        try:
            match = re.search(r'POS-?(\d+)', file_name, re.IGNORECASE)
            return match.group(1) if match else 'Unknown'
            
        except Exception as e:
            self.logger.warning(f"Tire position extraction failed: {str(e)}")
            return 'Unknown'

    def extract_vehicle_type(self, file_path: Path) -> str:
        """
        Extract vehicle type from file path.
        
        Searches directory hierarchy and file name for vehicle type indicators.
        
        Args:
            file_path (Path): Path to data file
            
        Returns:
            str: Extracted vehicle type or 'Unknown' if not found
        """
        try:
            # Check directory hierarchy
            current_path = file_path.parent
            while current_path != self.input_dir.parent:
                type_code = self._check_vehicle_type(current_path.name)
                if type_code:
                    return type_code
                current_path = current_path.parent
            
            # Check filename
            return self._check_vehicle_type(file_path.name) or 'Unknown'
            
        except Exception as e:
            self.logger.warning(f"Vehicle type extraction failed: {str(e)}")
            return 'Unknown'

    def _check_vehicle_type(self, name: str) -> Optional[str]:
        """
        Check string for vehicle type indicators.
        
        Args:
            name (str): String to check
            
        Returns:
            Optional[str]: Vehicle type code if found, None otherwise
        """
        name = name.lower()
        
        if '6 wheels' in name or '6w' in name:
            return '6W'
        elif '10 wheels' in name or '10w' in name:
            return '10W'
        elif '12 wheels' in name or '12w' in name:
            return '12W'
            
        return None

    def determine_hitting_type(self, file_path: Path) -> str:
        """
        Determine hitting type from file name.
        
        Checks file name for indicators of hitting type (Side/Tread).
        
        Args:
            file_path (Path): Path to data file
            
        Returns:
            str: Determined hitting type or 'Unknown' if not found
        """
        try:
            file_name = file_path.name.lower()
            
            if 'side' in file_name:
                return 'Side'
            elif 'tread' in file_name:
                return 'Tread'
            
            return 'Unknown'
            
        except Exception as e:
            self.logger.warning(f"Hitting type determination failed: {str(e)}")
            return 'Unknown'

    def generate_output_filename(self, metadata: Dict[str, Any], 
                               mounted: bool = False) -> str:
        """
        Generate standardized output filename from metadata.
        
        Args:
            metadata (Dict[str, Any]): File metadata
            mounted (bool): Whether file is mounted format
            
        Returns:
            str: Generated filename
            
        Raises:
            ValueError: If required metadata fields are missing
        """
        try:
            if mounted:
                return (f"{metadata['Tire Number']}_{metadata['TireSize']}-"
                       f"{metadata['Pressure']}-{metadata['Tire_Type']}")
            else:
                return (f"{metadata['Tire_Number']}_{metadata['TireSize']}-"
                       f"{metadata['Pressure']}-{metadata['Tire_Type']}")
                       
        except KeyError as e:
            raise ValueError(f"Missing required metadata field: {e}")


    # Section 7: File Handling Operations
    def save_to_excel(self, file_path: Path, data_dict: Dict[str, pd.DataFrame], 
                     sheet_order: Optional[List[str]] = None) -> bool:
        """
        Save multiple DataFrames to an Excel file.
        
        Creates an Excel workbook with multiple sheets, optionally in a
        specified order. Handles file creation and error checking.
        
        Args:
            file_path (Path): Path to save Excel file
            data_dict (Dict[str, pd.DataFrame]): Sheet name to DataFrame mapping
            sheet_order (Optional[List[str]]): Optional order of sheets
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                if sheet_order:
                    # Save sheets in specified order
                    for sheet_name in sheet_order:
                        if sheet_name in data_dict:
                            data_dict[sheet_name].to_excel(
                                writer, 
                                sheet_name=sheet_name, 
                                index=True
                            )
                else:
                    # Save sheets in arbitrary order
                    for sheet_name, df in data_dict.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=True)
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Excel file save failed - {file_path}: {str(e)}")
            return False

    def save_processing_results(self, file_path: Path, 
                              results: Dict[str, pd.DataFrame]) -> bool:
        """
        Save processing results with standardized sheet structure.
        
        Saves results with standard sheet ordering and validates the output.
        
        Args:
            file_path (Path): Output file path
            results (Dict[str, pd.DataFrame]): Processing results
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Define standard sheet order
            sheet_order = [
                'Step1_Data',
                'Step2_Sj',
                *[f'Step3_DataPts_{threshold}' 
                  for threshold in self.intensity_thresholds]
            ]
            
            # Validate results contain required sheets
            missing_sheets = [sheet for sheet in sheet_order 
                            if sheet not in results]
            if missing_sheets:
                raise ValueError(f"Missing required sheets: {missing_sheets}")
            
            return self.save_to_excel(file_path, results, sheet_order)
            
        except Exception as e:
            self.logger.error(f"Results save failed: {str(e)}")
            return False

    def save_median_pulse_widths(self, file_path: Path, 
                               median_data: pd.DataFrame) -> None:
        """
        Save median pulse width summary data.
        
        Args:
            file_path (Path): Output file path
            median_data (pd.DataFrame): Median pulse width data
            
        Raises:
            IOError: If save operation fails
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            median_data.to_excel(file_path, index=False)
            self.logger.info(f"Saved median pulse widths to: {file_path}")
            
        except Exception as e:
            raise IOError(f"Failed to save median pulse widths: {str(e)}")

    def _save_mounted_summary(self, processed_files: List[Path]) -> None:
        """
        Save summary statistics for mounted tire data.
        
        Args:
            processed_files (List[Path]): List of processed file paths
        """
        try:
            if processed_files:
                df_median = self.calculate_median_pulse_width(processed_files)
                output_path = self.processed_dir / 'Median_Pulse_Widths.xlsx'
                df_median.to_excel(output_path, index=False)
                self.logger.info(f"Saved mounted tire median pulse widths to: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save mounted summary: {str(e)}")

    def _save_standalone_summary(self, processed_files: List[Path]) -> None:
        """
        Save summary statistics for standalone tire data.
        
        Args:
            processed_files (List[Path]): List of processed file paths
        """
        try:
            if processed_files:
                df_median_pulse_widths = self.calculate_standalone_median_pulse_width(processed_files)
                
                if not df_median_pulse_widths.empty:
                    output_path = self.processed_standalone_dir / 'Median_Pulse_Widths_StandAlone.xlsx'
                    df_median_pulse_widths.to_excel(output_path, index=False)
                    self.logger.info(f"Saved standalone median pulse widths to: {output_path}")
            else:
                self.logger.warning("No standalone files were successfully processed")
                
        except Exception as e:
            self.logger.error(f"Failed to save standalone summary: {str(e)}")

    def save_exclusions(self) -> None:
        """
        Save processing exclusions report.
        
        Creates a report of files that were excluded from processing
        with reasons for exclusion.
        """
        if not hasattr(self, 'exclusions') or not self.exclusions:
            return
            
        try:
            df_exclusions = pd.DataFrame(self.exclusions)
            exclusions_path = self.output_dir / 'Excluded_Files.xlsx'
            
            df_exclusions.to_excel(exclusions_path, index=False)
            self.logger.info(f"Exclusions saved to {exclusions_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save exclusions report: {str(e)}")

    def add_exclusion_entry(self, file_path: Path, reason: str) -> None:
        """
        Add entry to exclusions tracking.
        
        Args:
            file_path (Path): Path to excluded file
            reason (str): Reason for exclusion
        """
        if not hasattr(self, 'exclusions'):
            self.exclusions = []
            
        try:
            relative_path = file_path.relative_to(self.output_dir)
            self.exclusions.append({
                'File_Name': str(relative_path),
                'Reason': reason,
                'Timestamp': pd.Timestamp.now()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to add exclusion entry: {str(e)}")

    def cleanup_temp_files(self) -> None:
        """
        Clean up temporary files created during processing.
        
        Removes temporary Excel files and other processing artifacts.
        """
        try:
            # Clean up temp Excel files
            for path in [self.processed_dir, self.processed_standalone_dir]:
                for temp_file in path.glob('~$*.xlsx'):
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to remove temp file {temp_file}: {str(e)}"
                        )
                        
            self.logger.info("Temporary files cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup operation failed: {str(e)}")

    def archive_old_results(self, archive_dir: Path) -> None:
        """
        Archive old processing results.
        
        Moves old result files to archive directory with timestamp.
        
        Args:
            archive_dir (Path): Directory for archived files
        """
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            archive_path = archive_dir / timestamp
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # Move old result files
            for path in [self.processed_dir, self.processed_standalone_dir]:
                if path.exists():
                    dest_path = archive_path / path.name
                    path.rename(dest_path)
                    self.logger.info(f"Archived {path} to {dest_path}")
                    
        except Exception as e:
            self.logger.error(f"Archiving operation failed: {str(e)}")

# Main execution
if __name__ == '__main__':
    config = load_config()
    processor = TireSoundProcessor(config=config)
    processor.process_all_files()
