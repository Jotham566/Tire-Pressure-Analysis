# new_visualizer.py
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  # For creating subplots
from pathlib import Path
import statsmodels.api as sm  # For statistical modeling
import re  # Ensure regex is imported for color extraction
import yaml  # For reading configuration from YAML file
from plotly.colors import qualitative  # Import color palettes

def list_excel_files(processed_dir):
    """
    Lists all Excel files in the specified Processed directory.
    """
    excel_files = list(Path(processed_dir).rglob('*.xlsx'))
    return excel_files


def select_file(excel_files, processed_dir):
    """
    Prompts the user to select an Excel file from the list and validates its structure.

    Returns:
        Path or None: The selected Excel file path if valid, else None.
    """
    if not excel_files:
        print("No Excel files found in the 'Processed' directory.")
        return None

    total_files = len(excel_files)
    print(f"\nTotal Excel files found: {total_files}\n")

    print("Available Excel files for visualization:")
    for idx, file in enumerate(excel_files, start=1):
        relative_path = file.relative_to(processed_dir)
        print(f"{idx}. {relative_path}")

    try:
        choice = int(input("\nEnter the number of the Excel file you want to visualize (0 to exit): "))
        if choice == 0:
            return None
        elif 1 <= choice <= len(excel_files):
            selected_file = excel_files[choice - 1]
            # Validate the structure of the selected Excel file
            required_sheets = ['Step1_Data', 'Step2_Sj_Denoised']
            # Open the Excel file to check sheet names
            with pd.ExcelFile(selected_file) as xls:
                sheet_names = xls.sheet_names

            # Check for required sheets
            missing_sheets = [sheet for sheet in required_sheets if sheet not in sheet_names]
            if missing_sheets:
                print(f"The selected file is missing required sheets: {missing_sheets}. Please select a different file.")
                return None

            # List available intensity thresholds
            step3_sheets = [sheet for sheet in sheet_names if sheet.startswith('Step3_DataPts_')]
            step3_thresholds = [sheet.replace('Step3_DataPts_', '') for sheet in step3_sheets]
            if step3_thresholds:
                print(f"Available Intensity Thresholds in the selected file: {', '.join(step3_thresholds)}")
            else:
                print("No Step3_DataPts_{threshold} sheets found in the selected file.")

            return selected_file
        else:
            print("Invalid choice. Exiting visualization.")
            return None
    except ValueError:
        print("Invalid input. Exiting visualization.")
        return None
    except Exception as e:
        print(f"An error occurred while validating the Excel file: {e}")
        return None


def list_signal_segments(step2_df):
    """
    Lists all available Signal Segments from Step2_Cumulative_Sj.
    """
    segments = step2_df.index.tolist()
    print("\nAvailable Signal Segments:")
    for idx, seg in enumerate(segments, start=1):
        print(f"{idx}. {seg}")
    return segments


def select_segments(segments):
    """
    Prompts the user to select which Signal Segments to visualize.
    Supports individual numbers and ranges (e.g., 1,3,5-7).
    """
    while True:
        seg_input = input("\nEnter the numbers of the Signal Segments you want to visualize separated by commas (e.g., 1,3,5-7) or 'all' to visualize all: ")
        if seg_input.lower() == 'all':
            return segments
        else:
            try:
                seg_indices = []
                for part in seg_input.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = part.split('-')
                        seg_indices.extend(range(int(start), int(end) + 1))
                    elif part.isdigit():
                        seg_indices.append(int(part))
                
                selected_segments = [segments[idx - 1] for idx in seg_indices if 1 <= idx <= len(segments)]
                if not selected_segments:
                    print("No valid segments selected. Please try again.")
                    continue
                
                # Display selected segments for confirmation
                print("\nYou have selected the following segments:")
                for seg in selected_segments:
                    print(f"- {seg}")
                
                confirm = input("Confirm selection? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    return selected_segments
                else:
                    print("Selection canceled. Please select again.")
            except Exception as e:
                print(f"Error selecting segments: {e}. Please try again.")


def plot_cumulative_data(step2_df, intensity_thresholds, selected_segments, selected_file, processed_dir):
    """
    Plots the Step2_Sj_Denoised data with annotations from Step3_DataPts_{threshold}
    using Plotly. Creates subplots for each intensity threshold.
    Also provides an option to save the plot in the Visualization folder.

    Args:
        step2_df (pd.DataFrame): DataFrame from 'Step2_Sj_Denoised' sheet.
        intensity_thresholds (list): List of intensity thresholds to visualize.
        selected_segments (list): List of selected segment IDs to plot.
        selected_file (Path): Path to the selected Excel file.
        processed_dir (Path): Path to the 'Processed' directory.
    """
    num_thresholds = len(intensity_thresholds)
    if num_thresholds == 0:
        print("No intensity thresholds provided for visualization.")
        return

    # Create subplots: one row, multiple columns (one for each intensity threshold)
    fig = make_subplots(
        rows=1,
        cols=num_thresholds,
        subplot_titles=[f'Intensity Threshold: {th}' for th in intensity_thresholds],
        horizontal_spacing=0.05  # Adjust spacing as needed
    )

    # Dictionary to store step3 DataFrames per threshold
    step3_dfs = {}

    # Read Step3_DataPts_{threshold} sheets
    try:
        with pd.ExcelFile(selected_file) as xls:
            for idx, threshold in enumerate(intensity_thresholds, start=1):
                sheet_name = f'Step3_DataPts_{threshold}'
                if sheet_name not in xls.sheet_names:
                    print(f"Sheet '{sheet_name}' not found in the Excel file.")
                    continue
                df_step3 = pd.read_excel(xls, sheet_name=sheet_name, index_col='Segment_ID')
                step3_dfs[threshold] = df_step3
    except Exception as e:
        print(f"Error reading Step3_DataPts sheets: {e}")
        return

    # Iterate over each intensity threshold and create corresponding subplot
    for idx, threshold in enumerate(intensity_thresholds, start=1):
        df_step3 = step3_dfs.get(threshold)
        if df_step3 is None:
            print(f"No data available for intensity threshold {threshold}. Skipping subplot.")
            continue

        for segment in selected_segments:
            if segment not in step2_df.index:
                print(f"Segment '{segment}' not found in Step2_Sj_Denoised.")
                continue

            cumulative_values = step2_df.loc[segment].values
            x_values = list(range(1, len(cumulative_values) + 1))  # 1 to 256

            # Add line for the segment with a unique legend group
            fig.add_trace(go.Scatter(
                x=x_values,
                y=cumulative_values,
                mode='lines',
                name=f'Segment {segment}',
                line=dict(width=2),
                legendgroup=segment,
                customdata=[[re.search(r'\d+$', segment).group()]] * len(x_values),  # Extract only the number at the end
                hovertemplate='Signal Segment: %{customdata[0]}<br>Signal Value: %{x}<br>Cumulative Sum: %{y:.4f}<extra></extra>',
                showlegend=(idx == 1)  # Show legend only once
            ), row=1, col=idx)

            # Get annotation points from Step3_DataPts_{threshold}
            if segment in df_step3.index:
                # Retrieve cumulative sum values and indices
                first_increase_val = df_step3.loc[segment, 'First_Noticeable_Increase_Cumulative_Value']
                point_exceeds_val = df_step3.loc[segment, 'Point_Exceeds_Cumulative_Value'] if 'Point_Exceeds_Cumulative_Value' in df_step3.columns else np.nan

                # Retrieve x positions (indices)
                x_first_increase = df_step3.loc[segment, 'First_Noticeable_Increase_Index']
                x_point_exceeds = df_step3.loc[segment, 'Point_Exceeds_Index'] if 'Point_Exceeds_Index' in df_step3.columns else np.nan

                # Add marker and annotation for First Increase
                if not pd.isna(x_first_increase) and not pd.isna(first_increase_val):
                    fig.add_trace(go.Scatter(
                        x=[x_first_increase],
                        y=[first_increase_val],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='circle'),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=1, col=idx)

                    fig.add_annotation(
                        x=x_first_increase,
                        y=first_increase_val,
                        text=f"First Increase<br>({x_first_increase}, {first_increase_val:.4f})",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40,
                        font=dict(color='red'),
                        arrowcolor='red',
                        xref=f'x{idx}',
                        yref=f'y{idx}',
                        row=1,
                        col=idx
                    )

                # Add marker and annotation for Point Exceeds
                if not pd.isna(x_point_exceeds) and not pd.isna(point_exceeds_val):
                    fig.add_trace(go.Scatter(
                        x=[x_point_exceeds],
                        y=[point_exceeds_val],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='circle'),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=1, col=idx)

                    fig.add_annotation(
                        x=x_point_exceeds,
                        y=point_exceeds_val,
                        text=f"Exceeds {threshold}<br>({x_point_exceeds}, {point_exceeds_val:.4f})",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=40,
                        font=dict(color='green'),
                        arrowcolor='green',
                        xref=f'x{idx}',
                        yref=f'y{idx}',
                        row=1,
                        col=idx
                    )

        # Update axes titles for each subplot
        fig.update_xaxes(title_text='Signal Value', row=1, col=idx)
        fig.update_yaxes(title_text='Denoised Cumulative Sum (Sj)', row=1, col=idx)

    # Update layout for better aesthetics
    
    fig.update_layout(
        title=f'Denoised Cumulative Sum of Signal Values ({selected_segments})<br><sup>File: {selected_file.relative_to(processed_dir)}</sup>',
        legend_title="Signal Segments",
        hovermode='x unified',
        template='plotly_white',
        width=500 * num_thresholds + 100,  # Adjust width based on number of subplots
        height=600
    )

    fig.show()

    # Option to save the plot
    save_plot = input("\nWould you like to save this plot? (yes/no): ").strip().lower()
    if save_plot in ['yes', 'y']:
        # Define the base Visualization directory
        visualization_dir = Path.cwd() / 'Visualization'

        # Get the relative path of the Excel file with respect to the Processed directory
        relative_path = selected_file.relative_to(processed_dir).with_suffix('')  # Remove .xlsx extension

        # Define the save directory within Visualization
        save_dir = visualization_dir / 'Cumulative_Sum_Plots' / relative_path.parent

        # Create the save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Define the save file name with the same name as the Excel file, changing extension to .html and .png
        save_file_html = save_dir / (selected_file.stem + '.html')
        save_file_png = save_dir / (selected_file.stem + '.png')

        try:
            # Save as HTML
            fig.write_html(save_file_html)
            # Save as PNG
            fig.write_image(save_file_png)
            print(f"Plot saved as HTML to {save_file_html}")
            print(f"Plot saved as PNG to {save_file_png}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("Plot not saved.")


def plot_aggregate_median_pulse_width_vs_pressure(processed_dir, hitting_type_filter, intensity_threshold_filter):
    """
    Plots Median Pulse Width vs Air Pressure aggregated by Tire Type and Hitting Type on subplots.
    Filters based on the provided hitting_type_filter and intensity_threshold_filter.
    After generating the plot, asks if you'd like to save it.
    
    Args:
        processed_dir (Path): Path to the 'Processed' directory.
        hitting_type_filter (list): List containing 'Tread', 'Side', or both.
        intensity_threshold_filter (list): List of intensity thresholds to visualize.
    """
    median_pulse_widths_file = processed_dir / 'Median_Pulse_Widths.xlsx'
    if not median_pulse_widths_file.exists():
        print(f"Median pulse widths file not found at {median_pulse_widths_file}. Please ensure it exists.")
        return

    try:
        df = pd.read_excel(median_pulse_widths_file)
    except Exception as e:
        print(f"Error reading Median_Pulse_Widths.xlsx: {e}")
        return

    # Ensure no missing values in required columns
    df = df.dropna(subset=['Median_Pulse_Width', 'Air_Pressure', 'Intensity_Threshold'])

    # Extract Tire_Type from the 'Tire' column (e.g., '6W', '10W', '12W')
    df['Tire_Type'] = df['Tire'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'Unknown')

    # Filter based on hitting_type_filter
    if hitting_type_filter and 'Both' not in hitting_type_filter:
        df = df[df['Hitting_Type'].isin(hitting_type_filter)]
    elif 'Both' in hitting_type_filter or not hitting_type_filter:
        pass  # Include all hitting types
    else:
        print("Invalid hitting_type_filter. Must be 'Tread', 'Side', or both.")
        return

    # Get unique Tire_Types
    tire_types = df['Tire_Type'].unique()

    # Prompt user to select a Tire_Type
    print("\nAvailable Tire Types:")
    for idx, tire_type in enumerate(tire_types, start=1):
        print(f"{idx}. {tire_type}")

    try:
        choice = int(input("\nEnter the number of the Tire Type you want to visualize (0 to exit): "))
        if choice == 0:
            print("No Tire Type selected.")
            return
        elif 1 <= choice <= len(tire_types):
            selected_tire_type = tire_types[choice - 1]
        else:
            print("Invalid choice.")
            return
    except ValueError:
        print("Invalid input.")
        return

    # Filter data for the selected Tire_Type
    df_selected = df[df['Tire_Type'] == selected_tire_type]

    if df_selected.empty:
        print(f"No data available for Tire Type '{selected_tire_type}'.")
        return

    # Get unique Hitting_Types
    hitting_types = df_selected['Hitting_Type'].unique()

    # Create subplots: rows=Hitting_Types, columns=Intensity Thresholds
    fig = make_subplots(
        rows=len(hitting_types),
        cols=len(intensity_threshold_filter),
        subplot_titles=[
            f"{ht} - Threshold {th}" for ht in hitting_types for th in intensity_threshold_filter
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    # Prepare min and max pressure values for x-axis range
    min_pressure = df_selected['Air_Pressure'].min()
    max_pressure = df_selected['Air_Pressure'].max()
    # Set buffer values if needed
    min_range = max(0, min_pressure - 50)
    max_range = max_pressure + 50

    # Prepare tick values based on data points
    tick_values = sorted(df_selected['Air_Pressure'].unique())
    # Optionally, extend ticks to include standard values
    standard_ticks = [500, 600, 700, 800, 850, 900, 950]
    for tick in standard_ticks:
        if tick not in tick_values and min_range <= tick <= max_range:
            tick_values.append(tick)
    tick_values = sorted(set(tick_values))

    # Iterate through Hitting_Types and Intensity Thresholds
    for row_idx, hitting_type in enumerate(hitting_types, start=1):
        for col_idx, threshold in enumerate(intensity_threshold_filter, start=1):
            subset = df_selected[
                (df_selected['Hitting_Type'] == hitting_type) &
                (df_selected['Intensity_Threshold'] == threshold)
            ]
            if subset.empty:
                continue

            # Scatter plot with detailed hover info
            fig.add_trace(
                go.Scatter(
                    x=subset['Air_Pressure'],
                    y=subset['Median_Pulse_Width'],
                    mode='markers',
                    name=f'{hitting_type} - Threshold {threshold}',
                    marker=dict(symbol='circle'),
                    hovertemplate=(
                        'Tire: %{customdata[0]}<br>'
                        'Tire Type: %{customdata[1]}<br>'
                        'Tire Position: %{customdata[2]}<br>'
                        'File Name: %{customdata[3]}<br>'
                        'Air Pressure: %{x}<br>'
                        'Median Pulse Width: %{y}<extra></extra>'
                    ),
                    customdata=np.stack((subset['Tire'], subset['Tire_Type'], subset['Tire_Position'], subset['File_Name']), axis=-1),
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx
            )

            # Fit a regression line
            if len(subset) >= 2:
                X = subset['Air_Pressure']
                Y = subset['Median_Pulse_Width']
                X_const = sm.add_constant(X)
                model = sm.OLS(Y, X_const)
                results = model.fit()
                Y_pred = results.predict(X_const)

                # Add regression line
                fig.add_trace(
                    go.Scatter(
                        x=X,
                        y=Y_pred,
                        mode='lines',
                        name=f'{hitting_type} Fit Line {threshold}',
                        line=dict(dash='dash'),
                        hoverinfo='skip',
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx
                )

                # Add R-squared annotation
                r_squared = results.rsquared
                fig.add_annotation(
                    x=0.95, y=0.05,
                    xref='x domain',
                    yref='y domain',
                    text=f'R²={r_squared:.2f}',
                    showarrow=False,
                    xanchor='right',
                    yanchor='bottom',
                    font=dict(size=12),
                    row=row_idx,
                    col=col_idx
                )

            # Update axes titles
            if row_idx == len(hitting_types):
                fig.update_xaxes(title_text='Air Pressure', row=row_idx, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text='Median Pulse Width', row=row_idx, col=col_idx)

            # Set x-axis range and ticks for each subplot
            fig.update_xaxes(
                range=[min_range, max_range],
                tickmode='array',
                tickvals=tick_values,
                row=row_idx,
                col=col_idx
            )

    # Update layout with dynamic title and adjusted dimensions
    fig.update_layout(
        title_text=(
            f'Median Pulse Width vs Air Pressure Aggregated by wheels: {selected_tire_type}<br>'
            f'<sup>Hitting Type: {", ".join(hitting_type_filter)}, '
            f'Intensity Thresholds: {", ".join(map(str, intensity_threshold_filter))}</sup>'
        ),
        width=500 * len(intensity_threshold_filter),
        height=500 * len(hitting_types),
        template='plotly_white'
    )

    fig.show()

    # Option to save the plot
    save_plot = input("\nWould you like to save this plot? (yes/no): ").strip().lower()
    if save_plot in ['yes', 'y']:
        # Define the base Visualization directory
        visualization_dir = Path.cwd() / 'Visualization'

        # Define the save directory within Visualization
        save_dir = visualization_dir / 'Median_Pulse_Width_vs_Pressure' / 'Aggregated' / selected_tire_type

        # Create the save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Define the save file name
        save_file_html = save_dir / f'{selected_tire_type}_Aggregated_Median_Pulse_Width_vs_Pressure.html'
        save_file_png = save_dir / f'{selected_tire_type}_Aggregated_Median_Pulse_Width_vs_Pressure.png'

        try:
            # Save as HTML
            fig.write_html(save_file_html)
            # Save as PNG
            fig.write_image(save_file_png)
            print(f"Plot saved to {save_dir}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("Plot not saved.")


def plot_aggregate_by_hitting_type(processed_dir, hitting_type_filter, intensity_threshold_filter):
    """
    Plots Median Pulse Width vs Air Pressure aggregated by Hitting Type (Tread or Side),
    creating subplots for each Intensity Threshold based on the configuration.
    Filters based on the provided hitting_type_filter and intensity_threshold_filter.
    
    Args:
        processed_dir (Path): Path to the 'Processed' directory.
        hitting_type_filter (list): List containing 'Tread', 'Side', or both.
        intensity_threshold_filter (list): List of intensity thresholds to visualize.
    """
    median_pulse_widths_file = processed_dir / 'Median_Pulse_Widths.xlsx'
    if not median_pulse_widths_file.exists():
        print(f"Median pulse widths file not found at {median_pulse_widths_file}. Please ensure it exists.")
        return

    try:
        df = pd.read_excel(median_pulse_widths_file)
    except Exception as e:
        print(f"Error reading Median_Pulse_Widths.xlsx: {e}")
        return

    # Ensure no missing values in required columns
    df = df.dropna(subset=['Median_Pulse_Width', 'Air_Pressure', 'Intensity_Threshold'])

    # Extract Tire_Type from the 'Tire' column
    df['Tire_Type'] = df['Tire'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'Unknown')

    # Filter based on hitting_type_filter
    if hitting_type_filter and 'Both' not in hitting_type_filter:
        df = df[df['Hitting_Type'].isin(hitting_type_filter)]
    elif 'Both' in hitting_type_filter or not hitting_type_filter:
        pass  # Include all hitting types
    else:
        print("Invalid hitting_type_filter. Must be 'Tread', 'Side', or both.")
        return

    # Get unique Hitting_Types
    hitting_types = df['Hitting_Type'].unique()

    # Create subplots: rows = number of Hitting_Types, columns = number of Intensity Thresholds
    fig = make_subplots(
        rows=len(hitting_types),
        cols=len(intensity_threshold_filter),
        subplot_titles=[
            f"{ht} - Threshold {th}" for ht in hitting_types for th in intensity_threshold_filter
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    # Prepare min and max pressure values for x-axis range
    min_pressure = df['Air_Pressure'].min()
    max_pressure = df['Air_Pressure'].max()
    # Set buffer values if needed
    min_range = max(0, min_pressure - 50)
    max_range = max_pressure + 50

    # Prepare tick values based on data points
    tick_values = sorted(df['Air_Pressure'].unique())
    # Optionally, extend ticks to include standard values
    standard_ticks = [500, 600, 700, 800, 850, 900, 950]
    for tick in standard_ticks:
        if tick not in tick_values and min_range <= tick <= max_range:
            tick_values.append(tick)
    tick_values = sorted(set(tick_values))

    # Iterate over Hitting_Types and Intensity Thresholds
    for row_idx, hitting_type in enumerate(hitting_types, start=1):
        for col_idx, threshold in enumerate(intensity_threshold_filter, start=1):
            subset = df[
                (df['Hitting_Type'] == hitting_type) &
                (df['Intensity_Threshold'] == threshold)
            ]
            if subset.empty:
                continue

            # Scatter plot with detailed hover info
            fig.add_trace(
                go.Scatter(
                    x=subset['Air_Pressure'],
                    y=subset['Median_Pulse_Width'],
                    mode='markers',
                    name=f'{hitting_type} Data - Threshold {threshold}',
                    marker=dict(symbol='circle'),
                    hovertemplate=(
                        'Tire: %{customdata[0]}<br>'
                        'Tire Type: %{customdata[1]}<br>'
                        'Tire Position: %{customdata[2]}<br>'
                        'File Name: %{customdata[3]}<br>'
                        'Air Pressure: %{x}<br>'
                        'Median Pulse Width: %{y}<extra></extra>'
                    ),
                    customdata=np.stack((
                        subset['Tire'],
                        subset['Tire_Type'],
                        subset['Tire_Position'],
                        subset['File_Name']
                    ), axis=-1),
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx
            )

            # Fit a regression line
            if len(subset) >= 2:
                X = subset['Air_Pressure']
                Y = subset['Median_Pulse_Width']
                X_const = sm.add_constant(X)
                model = sm.OLS(Y, X_const)
                results = model.fit()
                Y_pred = results.predict(X_const)

                # Add regression line
                fig.add_trace(
                    go.Scatter(
                        x=X,
                        y=Y_pred,
                        mode='lines',
                        name=f'{hitting_type} Fit Line - Threshold {threshold}',
                        line=dict(dash='dash'),
                        hoverinfo='skip',
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx
                )

                # Add R-squared annotation
                r_squared = results.rsquared
                fig.add_annotation(
                    x=0.95, y=0.05,
                    xref='x domain',
                    yref='y domain',
                    text=f'R²={r_squared:.2f}',
                    showarrow=False,
                    xanchor='right',
                    yanchor='bottom',
                    font=dict(size=12),
                    row=row_idx,
                    col=col_idx
                )

            # Update axes titles
            if row_idx == len(hitting_types):
                fig.update_xaxes(title_text='Air Pressure', row=row_idx, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text='Median Pulse Width', row=row_idx, col=col_idx)

            # Set x-axis range and ticks for each subplot
            fig.update_xaxes(
                range=[min_range, max_range],
                tickmode='array',
                tickvals=tick_values,
                row=row_idx,
                col=col_idx
            )

    # Update layout with dynamic title
    # Helper function to format hitting_type_filter for the title
    def format_hitting_type_filter(hitting_type_filter):
        if not hitting_type_filter:
            return "Both Tread and Side"
        elif len(hitting_type_filter) == 2:
            return "Both Tread and Side"
        else:
            return hitting_type_filter[0]

    hitting_type_str = format_hitting_type_filter(hitting_type_filter)
    fig.update_layout(
        title_text=(
            f'Pulse Width vs Air Pressure for {hitting_type_str} hitting<br>'
            f'<sup>Hitting Type: {hitting_type_str}, '
            f'Intensity Thresholds: {", ".join(map(str, intensity_threshold_filter))}</sup>'
        ),
        width=500 * len(intensity_threshold_filter),
        height=500 * len(hitting_types),
        template='plotly_white'
    )

    # Fix y-axis to start at 0 for all subplots
    for row in range(1, len(hitting_types) + 1):
        for col in range(1, len(intensity_threshold_filter) + 1):
            fig.update_yaxes(rangemode='tozero', row=row, col=col)

    fig.show()

    # Option to save the plot
    save_plot = input("\nWould you like to save this plot? (yes/no): ").strip().lower()
    if save_plot in ['yes', 'y']:
        # Define the base Visualization directory
        visualization_dir = Path.cwd() / 'Visualization'

        # Define the save directory within Visualization
        save_dir = visualization_dir / 'Median_Pulse_Width_vs_Pressure' / 'Aggregated_By_Hitting_Type'

        # Create the save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Define the save file name
        save_file_html = save_dir / 'Aggregated_By_Hitting_Type_Median_Pulse_Width_vs_Pressure.html'
        save_file_png = save_dir / 'Aggregated_By_Hitting_Type_Median_Pulse_Width_vs_Pressure.png'

        try:
            # Save as HTML
            fig.write_html(save_file_html)
            # Save as PNG
            fig.write_image(save_file_png)
            print(f"Plot saved to {save_dir}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("Plot not saved.")


def plot_wheel_type_scatter_fit(processed_dir, hitting_type_filter, intensity_threshold_filter):
    """
    Plots scatter data points and regression lines for each Wheel_Type (6W, 10W, 12W).
    Creates subplots for each Intensity Threshold.
    Filters based on the provided hitting_type_filter and intensity_threshold_filter.

    Features:
    - Scatter points representing actual data with distinct colors for each Wheel_Type.
    - Fitted linear regression lines for each Wheel_Type.
    - R² annotations for each regression line integrated into the trace.
    - Y-axis fixed to start at 0 for consistency.

    Args:
        processed_dir (Path): Path to the 'Processed' directory.
        hitting_type_filter (list): List containing 'Tread', 'Side', or both.
        intensity_threshold_filter (list): List of intensity thresholds to visualize.
    """
    median_pulse_widths_file = processed_dir / 'Median_Pulse_Widths.xlsx'
    if not median_pulse_widths_file.exists():
        print(f"Median pulse widths file not found at {median_pulse_widths_file}. Please ensure it exists.")
        return

    try:
        df = pd.read_excel(median_pulse_widths_file)
    except Exception as e:
        print(f"Error reading Median_Pulse_Widths.xlsx: {e}")
        return

    # Ensure no missing values in required columns
    required_columns = ['Median_Pulse_Width', 'Air_Pressure', 'Wheel_Type', 'Hitting_Type', 'Intensity_Threshold']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return

    df = df.dropna(subset=required_columns)

    # Filter based on hitting_type_filter
    if hitting_type_filter and 'Both' not in hitting_type_filter:
        df = df[df['Hitting_Type'].isin(hitting_type_filter)]
    elif 'Both' in hitting_type_filter or not hitting_type_filter:
        pass  # Include all hitting types
    else:
        print("Invalid hitting_type_filter. Must be 'Tread', 'Side', or both.")
        return

    # Filter for Wheel_Type in ['6W', '10W', '12W']
    wheel_types = ['6W', '10W', '12W']
    df = df[df['Wheel_Type'].isin(wheel_types)]

    if df.empty:
        print("No data available for Wheel_Types: 6W, 10W, 12W.")
        return

    # Create subplots: one row, multiple columns (one for each Intensity Threshold)
    fig = make_subplots(
        rows=1,
        cols=len(intensity_threshold_filter),
        subplot_titles=[f'Intensity Threshold: {th}' for th in intensity_threshold_filter],
        horizontal_spacing=0.05  # Adjust spacing as needed
    )

    # Define colors for each Wheel_Type
    wheel_colors = {
        '6W': 'blue',
        '10W': 'orange',
        '12W': 'green'
    }

    # Prepare min and max pressure values for x-axis range
    min_pressure = df['Air_Pressure'].min()
    max_pressure = df['Air_Pressure'].max()
    # Set buffer values if needed
    min_range = max(0, min_pressure - 50)
    max_range = max_pressure + 50

    # Prepare tick values based on data points
    tick_values = sorted(df['Air_Pressure'].unique())
    # Optionally, extend ticks to include standard values
    standard_ticks = [500, 600, 700, 800, 850, 900, 950]
    for tick in standard_ticks:
        if tick not in tick_values and min_range <= tick <= max_range:
            tick_values.append(tick)
    tick_values = sorted(set(tick_values))

    # Helper function to format hitting_type_filter for the title
    def format_hitting_type_filter(hitting_type_filter):
        if not hitting_type_filter:
            return "Both Tread and Side"
        elif len(hitting_type_filter) == 2:
            return "Both Tread and Side"
        else:
            return hitting_type_filter[0]

    # Iterate over each intensity threshold and create corresponding subplot
    for idx, threshold in enumerate(intensity_threshold_filter, start=1):
        df_threshold = df[df['Intensity_Threshold'] == threshold]
        if df_threshold.empty:
            print(f"No data available for intensity threshold {threshold}. Skipping subplot.")
            continue

        for wheel in wheel_types:
            df_wheel = df_threshold[df_threshold['Wheel_Type'] == wheel]
            if df_wheel.empty:
                print(f"No data available for Wheel_Type '{wheel}' at threshold {threshold}.")
                continue

            # Scatter Plot
            fig.add_trace(go.Scatter(
                x=df_wheel['Air_Pressure'],
                y=df_wheel['Median_Pulse_Width'],
                mode='markers',
                name=f'{wheel} Data',
                marker=dict(
                    color=wheel_colors.get(wheel, 'black'),
                    size=8,
                    opacity=0.7
                ),
                legendgroup=wheel,  # Group by wheel type
                showlegend=(idx == 1),  # Show legend only on first subplot
                hovertemplate=(
                    f'Wheel_Type: {wheel}<br>'
                    'Tire: %{customdata[0]}<br>'
                    'Air Pressure: %{x}<br>'
                    'Median Pulse Width: %{y:.2f}<extra></extra>'
                ),
                customdata=df_wheel[['Tire']].values  # Add Tire column to customdata
            ), row=1, col=idx)

            # Fit a linear regression line
            X = df_wheel['Air_Pressure']
            Y = df_wheel['Median_Pulse_Width']
            if len(X) >= 2:
                X_const = sm.add_constant(X)
                model = sm.OLS(Y, X_const)
                results = model.fit()
                Y_pred = results.predict(X_const)

                # Add regression line
                fig.add_trace(go.Scatter(
                    x=X,
                    y=Y_pred,
                    mode='lines',
                    name=f'{wheel} Fit Line',
                    line=dict(color=wheel_colors.get(wheel, 'black'), dash='dash'),
                    legendgroup=wheel,
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=idx)

                # Add R-squared annotation
                fig.add_annotation(
                    x=0.95, y=0.05, xref='x domain', yref='y domain',
                    text=f'R²={results.rsquared:.2f}', showarrow=False,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=12, color=wheel_colors.get(wheel, 'black')),
                    row=1, col=idx
                )

            else:
                print(f"Not enough data points to fit regression for Wheel_Type '{wheel}' at threshold {threshold}.")

        # Update axes titles
        fig.update_xaxes(title_text='Air Pressure', row=1, col=idx)
        if idx == 1:
            fig.update_yaxes(title_text='Median Pulse Width', row=1, col=idx)

        # Set x-axis range and ticks for each subplot
        fig.update_xaxes(
            range=[min_range, max_range],
            tickmode='array',
            tickvals=tick_values,
            row=1,
            col=idx
        )

        # Fix y-axis to start at 0
        fig.update_yaxes(rangemode='tozero', row=1, col=idx)

    # Update layout with dynamic title
    hitting_type_str = format_hitting_type_filter(hitting_type_filter)
    fig.update_layout(
        title=(
            f'Median Pulse Width vs Air Pressure by Wheel Type<br>'
            f'<sup>Hitting Type: {hitting_type_str}, '
            f'Intensity Thresholds: {", ".join(map(str, intensity_threshold_filter))}</sup>'
        ),
        legend_title="Wheel Types",
        hovermode='closest',
        template='plotly_white',
        width=500 * len(intensity_threshold_filter),
        height=600
    )

    fig.show()

    # Option to save the plot
    save_plot = input("\nWould you like to save this plot? (yes/no): ").strip().lower()
    if save_plot in ['yes', 'y']:
        # Define the base Visualization directory
        visualization_dir = Path.cwd() / 'Visualization'

        # Define the save directory within Visualization
        save_dir = visualization_dir / 'Wheel_Type_Scatter_Fit'

        # Create the save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Define the save file name
        save_file_html = save_dir / 'Wheel_Type_Median_Pulse_Width_vs_Air_Pressure.html'
        save_file_png = save_dir / 'Wheel_Type_Median_Pulse_Width_vs_Air_Pressure.png'

        try:
            # Save as HTML
            fig.write_html(save_file_html)
            print(f"Plot saved as HTML to {save_file_html}")

            # Save as PNG
            fig.write_image(save_file_png)
            print(f"Plot saved as PNG to {save_file_png}")

        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("Plot not saved.")


def plot_median_and_categories(processed_dir, hitting_type_filter, intensity_threshold_filter):
    """
    Plots Median Pulse Width vs Air Pressure with data points differentiated by Wheel Type (6W, 10W, 12W).
    Includes trend lines for specific tire position pairs for each Wheel Type.
    Trend lines are included in the legend and are unselected by default.
    Differentiates trend lines for each pair and wheel type with unique colors and dash styles.

    Creates subplots for each Intensity Threshold.

    Features:
    - Scatter points with interactive hover information.
    - Legends to differentiate between 6W, 10W, and 12W data points.
    - Trend lines for tire position pairs with unique styles per wheel type.
    - Y-axis fixed to start at 0 for consistency.

    Args:
        processed_dir (Path): Path to the 'Processed' directory.
        hitting_type_filter (list): List containing 'Tread', 'Side', or both.
        intensity_threshold_filter (list): List of intensity thresholds to visualize.
    """
    median_pulse_widths_file = processed_dir / 'Median_Pulse_Widths.xlsx'
    if not median_pulse_widths_file.exists():
        print(f"Median pulse widths file not found at {median_pulse_widths_file}. Please ensure it exists.")
        return

    try:
        df = pd.read_excel(median_pulse_widths_file)
    except Exception as e:
        print(f"Error reading Median_Pulse_Widths.xlsx: {e}")
        return

    # Ensure no missing values in required columns
    required_columns = ['Median_Pulse_Width', 'Air_Pressure', 'Wheel_Type', 'Hitting_Type', 'Intensity_Threshold', 'Tire_Position']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return

    df = df.dropna(subset=required_columns)

    # Clean data
    df['Wheel_Type'] = df['Wheel_Type'].astype(str).str.strip()
    df['Hitting_Type'] = df['Hitting_Type'].astype(str).str.strip()
    # Ensure Tire_Position is integer
    df['Tire_Position'] = df['Tire_Position'].astype(int)

    # Define colors for Wheel Types (for data points)
    wheel_type_colors = {
        '6W': 'blue',
        '10W': 'orange',
        '12W': 'green'
    }

    # Define tire position pairs
    tire_position_pairs = {
        '6W': [[1, 2], [3, 6], [4, 5]],
        '10W': [[1, 2], [3, 6], [4, 5], [7, 10], [8, 9]],
        '12W': [[1, 2], [3, 4], [5, 8], [6, 7], [9, 12], [10, 11]]
    }

    # Define color palettes for trend lines per wheel type
    wheel_trend_colors = {
        '6W': qualitative.Plotly,
        '10W': qualitative.Set1,
        '12W': qualitative.Dark2  # Changed from Set3 to Dark2
    }

    # Define dash styles
    dash_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    # Create subplots: one row, multiple columns (one for each Intensity Threshold)
    fig = make_subplots(
        rows=1,
        cols=len(intensity_threshold_filter),
        subplot_titles=[f'Intensity Threshold: {th}' for th in intensity_threshold_filter],
        horizontal_spacing=0.05  # Adjust spacing as needed
    )

    # Helper function to format hitting_type_filter for the title
    def format_hitting_type_filter(hitting_type_filter):
        if not hitting_type_filter:
            return "Both Tread and Side"
        elif len(hitting_type_filter) == 2:
            return "Both Tread and Side"
        else:
            return hitting_type_filter[0]

    # Iterate over each intensity threshold and create corresponding subplot
    for col_idx, threshold in enumerate(intensity_threshold_filter, start=1):
        df_threshold = df[df['Intensity_Threshold'] == threshold]
        if df_threshold.empty:
            print(f"No data available for intensity threshold {threshold}. Skipping subplot.")
            continue

        # Filter based on hitting_type_filter
        if hitting_type_filter and 'Both' not in hitting_type_filter:
            df_filtered = df_threshold[df_threshold['Hitting_Type'].isin(hitting_type_filter)]
        else:
            df_filtered = df_threshold  # Include all hitting types

        if df_filtered.empty:
            print(f"No data available for Hitting Types {hitting_type_filter} at threshold {threshold}.")
            continue

        # Plot data points for each Wheel_Type
        for wheel_type in ['6W', '10W', '12W']:
            df_wheel = df_filtered[df_filtered['Wheel_Type'] == wheel_type]
            if df_wheel.empty:
                print(f"No data available for Wheel_Type '{wheel_type}' at threshold {threshold}.")
                continue

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df_wheel['Air_Pressure'],
                    y=df_wheel['Median_Pulse_Width'],
                    mode='markers',
                    name=f'{wheel_type} Data',
                    marker=dict(
                        color=wheel_type_colors[wheel_type],
                        size=8,
                        opacity=0.7
                    ),
                    showlegend=(col_idx == 1),
                    legendgroup=wheel_type,
                    hovertemplate=(
                        'Wheel Type: %{customdata[0]}<br>'
                        'Tire Position: %{customdata[1]}<br>'
                        'Air Pressure: %{x}<br>'
                        'Median Pulse Width: %{y:.2f}<extra></extra>'
                    ),
                    customdata=np.stack((df_wheel['Wheel_Type'], df_wheel['Tire_Position']), axis=-1)
                ),
                row=1,
                col=col_idx
            )

            # Get tire position pairs for this wheel type
            position_pairs = tire_position_pairs.get(wheel_type, [])
            # Get color palette for this wheel type's trend lines
            trend_colors = wheel_trend_colors[wheel_type]
            num_colors = len(trend_colors)

            # For each tire position pair, fit a regression line
            for idx_pair, pair in enumerate(position_pairs):
                df_pair = df_wheel[df_wheel['Tire_Position'].isin(pair)]
                if len(df_pair) < 2:
                    continue  # Not enough data points to fit regression
                X = df_pair['Air_Pressure']
                Y = df_pair['Median_Pulse_Width']
                X_const = sm.add_constant(X)
                model = sm.OLS(Y, X_const)
                results = model.fit()
                Y_pred = results.predict(X_const)

                # Generate a name for the trend line, e.g., '6W Trend Positions 1 & 2'
                pair_name = f'{wheel_type} Trend Positions {pair[0]} & {pair[1]}'

                # Assign a unique color and dash style for this trend line
                trend_color = trend_colors[idx_pair % num_colors]
                dash_style = dash_styles[idx_pair % len(dash_styles)]

                # Plot trend line
                fig.add_trace(
                    go.Scatter(
                        x=X,
                        y=Y_pred,
                        mode='lines',
                        line=dict(
                            color=trend_color,
                            width=2,
                            dash=dash_style
                        ),
                        hoverinfo='skip',
                        showlegend=(col_idx == 1),
                        legendgroup=f"{wheel_type}_trend_{idx_pair}",
                        name=pair_name,
                        visible='legendonly',  # Make trend lines unselected by default
                        opacity=0.7  # Adjust opacity as needed
                    ),
                    row=1,
                    col=col_idx
                )

        # Update axes titles
        fig.update_xaxes(title_text='Air Pressure', row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text='Median Pulse Width', row=1, col=col_idx)

        # Prepare min and max pressure values for x-axis range
        min_pressure = df_filtered['Air_Pressure'].min()
        max_pressure = df_filtered['Air_Pressure'].max()
        # Set buffer values if needed
        min_range = max(0, min_pressure - 50)
        max_range = max_pressure + 50

        # Prepare tick values based on data points
        tick_values = sorted(df_filtered['Air_Pressure'].unique())
        # Optionally, extend ticks to include standard values
        standard_ticks = [500, 600, 700, 800, 850, 900, 950]
        for tick in standard_ticks:
            if tick not in tick_values and min_range <= tick <= max_range:
                tick_values.append(tick)
        tick_values = sorted(set(tick_values))

        # Set x-axis range and ticks for each subplot
        fig.update_xaxes(
            range=[min_range, max_range],
            tickmode='array',
            tickvals=tick_values,
            row=1,
            col=col_idx
        )

        # Fix y-axis to start at 0
        fig.update_yaxes(rangemode='tozero', row=1, col=col_idx)

    # Update layout with dynamic title
    hitting_type_str = format_hitting_type_filter(hitting_type_filter)
    fig.update_layout(
        title=(
            f'Median Pulse Width vs Air Pressure with Tire Pairing Trend Lines<br>'
            f'<sup>Hitting Type: {hitting_type_str}, '
            f'Intensity Thresholds: {", ".join(map(str, intensity_threshold_filter))}</sup>'
        ),
        legend_title="Wheel Types and Trend Lines",
        hovermode='closest',
        template='plotly_white',
        width=500 * len(intensity_threshold_filter),
        height=600
    )

    fig.show()

    # Option to save the plot
    save_plot = input("\nWould you like to save this plot? (yes/no): ").strip().lower()
    if save_plot in ['yes', 'y']:
        # Define the base Visualization directory
        visualization_dir = Path.cwd() / 'Visualization'

        # Define the save directory within Visualization
        save_dir = visualization_dir / 'Median_Pulse_Width_with_Tire_Position_Trends'

        # Create the save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Define the save file name
        save_file_html = save_dir / 'Median_Pulse_Width_vs_Air_Pressure_with_Tire_Position_Trends.html'
        save_file_png = save_dir / 'Median_Pulse_Width_vs_Air_Pressure_with_Tire_Position_Trends.png'

        try:
            # Save as HTML
            fig.write_html(save_file_html)
            print(f"Plot saved as HTML to {save_file_html}")

            # Save as PNG
            fig.write_image(save_file_png)
            print(f"Plot saved as PNG to {save_file_png}")

        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("Plot not saved.")


def main():
    """
    Main function to handle visualization.
    """
    print("\n--- Tire Sound Data Visualizer ---\n")

    # Load configuration from config.yaml
    config_file = 'config.yaml'
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            # Set default values if keys are missing
            hitting_type = config.get('new_visualizer', {}).get('hitting_type', 'Both')
            # Convert to list for consistency
            if hitting_type.lower() == 'both':
                hitting_type_filter = ['Tread', 'Side']
            elif hitting_type.lower() in ['tread', 'side']:
                hitting_type_filter = [hitting_type.capitalize()]
            else:
                print("Invalid hitting_type in config.yaml. Must be 'Tread', 'Side', or 'Both'. Using 'Both' as default.")
                hitting_type_filter = ['Tread', 'Side']
            # Read intensity_thresholds from config.yaml
            intensity_thresholds = config.get('new_main', {}).get('intensity_thresholds', [0.5, 0.7, 0.8, 0.9])

            # Ensure intensity_thresholds is a list for consistency
            if isinstance(intensity_thresholds, (list, tuple)):
                intensity_threshold_filter = intensity_thresholds
            else:
                intensity_threshold_filter = [intensity_thresholds]
                
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found. Using default hitting_type='Both' and intensity_thresholds=[0.5, 0.7, 0.8, 0.9].")
        hitting_type_filter = ['Tread', 'Side']
        intensity_threshold_filter = [0.5, 0.7, 0.8, 0.9]
    except Exception as e:
        print(f"Error reading configuration file '{config_file}': {e}")
        print("Using default hitting_type='Both' and intensity_thresholds=[0.5, 0.7, 0.8, 0.9].")
        hitting_type_filter = ['Tread', 'Side']
        intensity_threshold_filter = [0.5, 0.7, 0.8, 0.9]

    # Hard-coded path to the 'Processed' directory
    processed_dir = Path.cwd() / 'Processed'
    if not processed_dir.exists():
        print(f"The 'Processed' directory was not found at {processed_dir}. Please ensure it exists.")
        return

    while True:
        print("\nSelect Visualization Type:")
        print("1. Visualize Individual Cumulative & Denoised Segments")
        print("2. Visualize Pulse Width vs Air Pressure Aggregated by  Wheels")
        print("3. Visualize Pulse Width vs Air Pressure for selected Hitting Type")
        print("4. Visualize Median Pulse Width vs Air Pressure by Wheel Type") 
        print("5. Visualize Median Pulse Width vs Air Pressure with Tire Pairing Trend Lines")
        print("0. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            excel_files = list_excel_files(processed_dir)
            selected_file = select_file(excel_files, processed_dir)
            if not selected_file:
                print("No file selected for visualization.")
                continue

            try:
                # Read the necessary sheets with correct sheet names
                step2_cumulative = pd.read_excel(selected_file, sheet_name='Step2_Sj_Denoised', index_col='Segment_ID')
                # Removed reading 'Step3_Data_Points' as 'plot_cumulative_data' handles multiple thresholds internally
            except Exception as e:
                print(f"Error reading Excel sheets: {e}")
                continue

            segments = list_signal_segments(step2_cumulative)
            if not segments:
                print("No signal segments found.")
                continue

            selected_segments = select_segments(segments)
            if not selected_segments:
                print("No segments selected for visualization.")
                continue

            # Updated method call to pass 'intensity_threshold_filter' (a list)
            plot_cumulative_data(step2_cumulative, intensity_threshold_filter, selected_segments, selected_file, processed_dir)

        elif choice == '2':
            # Visualization 2: plot_aggregate_median_pulse_width_vs_pressure
            plot_aggregate_median_pulse_width_vs_pressure(processed_dir, hitting_type_filter, intensity_threshold_filter)

        elif choice == '3':
            # Visualization 3: plot_aggregate_by_hitting_type
            plot_aggregate_by_hitting_type(processed_dir, hitting_type_filter, intensity_threshold_filter)

        elif choice == '4':  # Existing option
            # Visualization 4: plot_wheel_type_scatter_fit
            plot_wheel_type_scatter_fit(processed_dir, hitting_type_filter, intensity_threshold_filter)

        elif choice == '5':  # **Handle new option**
            # Visualization 5: plot_median_and_categories
            plot_median_and_categories(processed_dir, hitting_type_filter, intensity_threshold_filter)

        elif choice == '0':
            print("Exiting visualization.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
