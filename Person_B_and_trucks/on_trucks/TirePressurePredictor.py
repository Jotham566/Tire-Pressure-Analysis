import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class TirePressurePredictor:
    def __init__(self):
        """Initialize the pressure predictor with calibrated parameters"""
        # Base pressure ranges
        self.min_pressure = 750
        self.max_pressure = 900
        
        # Calibration parameters based on pulse width analysis
        self.pulse_width_ranges = {
            'Side': {
                0.5: {'min': 16, 'max': 89},
                0.7: {'min': 25, 'max': 94},
                0.8: {'min': 30, 'max': 106},
                0.9: {'min': 51, 'max': 111}
            },
            'Tread': {
                0.5: {'min': 21, 'max': 113},
                0.7: {'min': 37, 'max': 115},
                0.8: {'min': 58, 'max': 116},
                0.9: {'min': 62, 'max': 121}
            }
        }
        
        # Position-based pressure adjustments
        self.position_factors = {
            'Side': {
                '1': 0.98, '2': 1.0, '3': 1.02, '6': 1.05,
                '4': 1.0, '5': 1.0
            },
            'Tread': {
                '1': 1.02, '2': 0.98, '3': 1.0, '6': 1.08,
                '4': 1.0, '5': 1.01
            }
        }
        
    def predict_pressure(self, pulse_width: float, position: str, hitting_type: str, 
                        threshold: float) -> tuple:
        """
        Predict tire pressure using calibrated model
        Returns: (predicted_pressure, confidence_score)
        """
        # Get relevant pulse width range
        pw_range = self.pulse_width_ranges[hitting_type][threshold]
        
        # Normalize pulse width to 0-1 range for current configuration
        range_min = pw_range['min']
        range_max = pw_range['max']
        normalized_pw = (pulse_width - range_min) / (range_max - range_min)
        normalized_pw = np.clip(normalized_pw, 0, 1)
        
        # Convert to pressure range
        base_pressure = self.min_pressure + normalized_pw * (self.max_pressure - self.min_pressure)
        
        # Apply position and hitting type adjustments
        position_factor = self.position_factors[hitting_type].get(str(position), 1.0)
        adjusted_pressure = base_pressure * position_factor
        
        # Calculate confidence score
        confidence = self._calculate_confidence(pulse_width, threshold, hitting_type, position,
                                             range_min, range_max)
        
        # Ensure pressure is within valid range
        final_pressure = np.clip(adjusted_pressure, self.min_pressure, self.max_pressure)
        
        return final_pressure, confidence
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process multiple measurements"""
        predictions = []
        
        for _, row in df.iterrows():
            try:
                pressure, confidence = self.predict_pressure(
                    row['Median_Pulse_Width'],
                    str(row['Position']),
                    row['Hitting_Type'],
                    row['Intensity_Threshold']
                )
                
                predictions.append({
                    'File_Name': row['File_Name'],
                    'Position': row['Position'],
                    'Hitting_Type': row['Hitting_Type'],
                    'Intensity_Threshold': row['Intensity_Threshold'],
                    'Median_Pulse_Width': row['Median_Pulse_Width'],
                    'Predicted_Pressure': round(pressure, 1),
                    'Confidence_Score': round(confidence, 2)
                })
                
            except Exception as e:
                print(f"Error processing {row['File_Name']}: {str(e)}")
                continue
                
        return pd.DataFrame(predictions)
    
    def _calculate_confidence(self, pulse_width: float, threshold: float, 
                            hitting_type: str, position: str,
                            range_min: float, range_max: float) -> float:
        """Calculate confidence score based on multiple factors"""
        # Base confidence from threshold
        threshold_confidence = {0.5: 0.7, 0.7: 0.8, 0.8: 0.9, 0.9: 1.0}[threshold]
        
        # Confidence based on pulse width being within expected range
        range_factor = 1.0
        if pulse_width < range_min or pulse_width > range_max:
            range_factor = 0.8
        
        # Position reliability
        position_reliability = {
            'Side': {'1': 0.9, '2': 0.9, '3': 0.95, '6': 0.95},
            'Tread': {'1': 0.95, '2': 0.9, '3': 0.95, '6': 0.95,
                     '4': 0.85, '5': 0.85}
        }
        pos_factor = position_reliability[hitting_type].get(str(position), 0.8)
        
        # Combine factors
        confidence = threshold_confidence * range_factor * pos_factor
        return confidence

def main():
    results_path = Path("Unknown_Pressure_Analysis_Results/unknown_pressure_analysis.xlsx")
    predictor = TirePressurePredictor()
    predictions_all = []
    
    # Process each threshold
    for threshold in [0.5, 0.7, 0.8, 0.9]:
        df = pd.read_excel(results_path, sheet_name=f'Threshold_{threshold}')
        df['Intensity_Threshold'] = threshold
        predictions = predictor.predict_batch(df)
        predictions_all.append(predictions)
    
    # Combine predictions
    final_predictions = pd.concat(predictions_all, ignore_index=True)
    
    # Save predictions
    output_path = Path("Unknown_Pressure_Analysis_Results/pressure_predictions.xlsx")
    final_predictions.to_excel(output_path, index=False)
    
    # Create visualization for threshold 0.9 (most reliable)
    plt.figure(figsize=(15, 6))
    
    # Side hitting predictions
    plt.subplot(1, 2, 1)
    side_data = final_predictions[
        (final_predictions['Hitting_Type'] == 'Side') & 
        (final_predictions['Intensity_Threshold'] == 0.9)
    ]
    sns.boxplot(data=side_data, x='Position', y='Predicted_Pressure')
    plt.title('Predicted Pressures - Side Hitting (Threshold 0.9)')
    plt.ylim(740, 910)
    
    # Tread hitting predictions
    plt.subplot(1, 2, 2)
    tread_data = final_predictions[
        (final_predictions['Hitting_Type'] == 'Tread') & 
        (final_predictions['Intensity_Threshold'] == 0.9)
    ]
    sns.boxplot(data=tread_data, x='Position', y='Predicted_Pressure')
    plt.title('Predicted Pressures - Tread Hitting (Threshold 0.9)')
    plt.ylim(740, 910)
    
    plt.tight_layout()
    plt.savefig('Unknown_Pressure_Analysis_Results/pressure_predictions_boxplot.png')
    plt.close()
    
    # Print summary statistics
    print("\nPrediction Summary (Threshold 0.9):")
    print("\nSide Hitting:")
    print(side_data.groupby('Position')['Predicted_Pressure'].describe())
    print("\nTread Hitting:")
    print(tread_data.groupby('Position')['Predicted_Pressure'].describe())
    
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    main()