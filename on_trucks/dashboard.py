# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml 
import sys
import re
import statsmodels.api as sm
import plotly.express as px

# Import custom modules from new_visualizer.py
import new_visualizer

# Ensuring new_visualizer.py is in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data # Caching the data to avoid re-running the function on every page refresh
def load_config():
    config_file = 'config.yaml'
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            hitting_type = config.get('new_visualizer', {}).get('hitting_type', 'Both')
            if hitting_type.lower() == 'both':
                hitting_type_filter = ['Tread', 'Side']
            elif hitting_type.lower() in ['tread', 'side']:
                hitting_type_filter = [hitting_type.capitalize()]
            else:
                st.warning("Invalid hitting_type. Using Both.")
                hitting_type_filter = ['Tread', 'Side']

            intensity_thresholds = config.get('pulse_width_calculator', {}).get('intensity_thresholds', [0.5, 0.7, 0.8, 0.9])
            if not isinstance(intensity_thresholds, list):
                intensity_thresholds = [intensity_thresholds]

            return hitting_type_filter, intensity_thresholds
    except FileNotFoundError:
        st.error("config.yaml not found. Using defaults.")
        return ['Tread', 'Side'], [0.5, 0.7, 0.8, 0.9]
    except Exception as e:
        st.error(f"Error reading config: {e}")
        return ['Tread', 'Side'], [0.5, 0.7, 0.8, 0.9]

def main():
    st.set_page_config(
        page_title="Tire Sound Data Dashboard",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Basic styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f2f6;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'><i class='fas fa-tachometer-alt'></i> Tire Sound Data Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.sidebar.title("Navigation")

    # Data type selection
    data_type = st.sidebar.radio(
        "Select Data Type",
        ["Truck-full", "Truck-trim", "St-Alone"]
    )

    menu = ["Home", "Tire Analysis"]
    choice = st.sidebar.radio("Go to", menu)

    hitting_type_filter, intensity_threshold_filter = load_config()

    processed_dir = Path.cwd() / 'Processed'
    processed_trimmed_dir = Path.cwd() / 'Processed_Trimmed'

    if data_type == "Truck-full":
        data_dir = Path.cwd() / 'Processed'
        median_pulse_widths_file = data_dir / 'Median_Pulse_Widths.xlsx'
        step3_prefix = 'Step3_DataPts'
        step2sj_name = 'Step2_Sj'
        selected_dims_after_rise_point = 32
    elif data_type == "Truck-trim":
        data_dir = Path.cwd() / 'Processed_Trimmed'
        median_pulse_widths_file = data_dir / 'Median_Pulse_Widths_Trim.xlsx'
        step3_prefix = 'Step3_Trim_DataPts'
        step2sj_name = 'Step2sj_Trim'
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            selected_dims_after_rise_point = config.get('pulse_width_calculator', {}).get('selected_dims_after_rise_point', 32)
        except:
            selected_dims_after_rise_point = 32
    else:  # St-Alone
        data_dir = Path.cwd() / 'Processed_Stand_Alone'
        median_pulse_widths_file = data_dir / 'Median_Pulse_Widths_StandAlone.xlsx'
        step3_prefix = 'Step3_DataPts'
        step2sj_name = 'Step2_Sj'
        selected_dims_after_rise_point = 32

    if choice == "Home":
        display_home_page()
    elif choice == "Tire Analysis":
        if data_type == "St-Alone":
            plot_standalone_analysis(data_dir, intensity_threshold_filter, median_pulse_widths_file, 
                                step3_prefix, step2sj_name)
        else:
            plot_median_pulse_width_by_tire_pairing(
                data_dir, 
                hitting_type_filter, 
                intensity_threshold_filter, 
                median_pulse_widths_file, 
                step3_prefix, 
                step2sj_name, 
                data_type == "Truck-trim", 
                selected_dims_after_rise_point
            )

def display_home_page():
    st.markdown("<h2>Overview</h2>", unsafe_allow_html=True)
    st.write("Welcome to the Tire Sound Data Dashboard.")

    col1, col2 = st.columns(2)

    with col1:
        # Processing steps
        st.markdown("<h2 class='section-title'>Processing Steps</h2>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-title'>Step 1: Select and Normalize</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul class='content-list'>
            <li>Select first 256 signal points.</li>
            <li>Normalize Σ256ai = 1.</li>
            <li>Save results to <b>Step1_Data</b>.</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("<h3 class='subsection-title'>Step 2: Cumulative Summation</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul class='content-list'>
            <li>Apply a noise threshold of 0.03.</li>
            <li>Set values below threshold to 0.</li>
            <li>Save results to <b>Step2_Sj</b> (and Step2sj_Trim if trimming).</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("<h3 class='subsection-title'>Step 3: Data Points per Intensity Threshold</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul class='content-list'>
            <li>Calculate first pulse rise point.</li>
            <li>Calculate pulse width.</li>
            <li>Save results to <b>Step3_DataPts_{X}</b> or <b>Step3_Trim_DataPts_{X}</b> if trimming.</li>
        </ul>
        """, unsafe_allow_html=True)

    with col2:
        # Visualizations
        st.markdown("<h2 class='section-title'>Visualizations</h2>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-title'>Single Segments</h3>", unsafe_allow_html=True)
        st.markdown("<ul class='content-list'><li>View individual cumulative data for each segment.</li></ul>", unsafe_allow_html=True)

        st.markdown("<h3 class='subsection-title'>Median Pulse Width by Wheels</h3>", unsafe_allow_html=True)
        st.markdown("<ul class='content-list'><li>Median pulse width of each file’s segments.</li></ul>", unsafe_allow_html=True)

        st.markdown("<h3 class='subsection-title'>Median Pulse Width by Hitting Type</h3>", unsafe_allow_html=True)
        st.markdown("<ul class='content-list'><li>Median pulse width segregated by Tread or Side.</li></ul>", unsafe_allow_html=True)

        st.markdown("<h3 class='subsection-title'>Median Pulse Width by Tire Pairing</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul class='content-list'>
            <li>Trend lines for specified tire pairings per wheel type.</li>
        </ul>
        """, unsafe_allow_html=True)

def plot_median_pulse_width_by_tire_pairing(data_dir, hitting_type_filter, intensity_threshold_filter, median_pulse_widths_file, step3_prefix, step2sj_name, trim_option, selected_dims_after_rise_point):
    st.markdown("<h2 style='color: #1f77b4;'>Median Pulse Width vs Air Pressure by Tire Pairing</h2>", unsafe_allow_html=True)
    if not median_pulse_widths_file.exists():
        st.error(f"Median pulse widths file not found: {median_pulse_widths_file}")
        return

    try:
        df = pd.read_excel(median_pulse_widths_file)
    except Exception as e:
        st.error(f"Error reading median pulse widths: {e}")
        return

    required_columns = ['Median_Pulse_Width', 'Air_Pressure', 'Wheel_Type', 'Hitting_Type', 'Intensity_Threshold', 'Tire_Position']
    if any(col not in df.columns for col in required_columns):
        st.error("Median pulse width file missing required columns.")
        return

    df = df.dropna(subset=required_columns)
    df['Wheel_Type'] = df['Wheel_Type'].str.strip()
    df['Hitting_Type'] = df['Hitting_Type'].str.strip()
    df['Tire_Position'] = df['Tire_Position'].astype(int)

    tire_position_pairs = {
        '6W': [[1, 2], [3, 6], [4, 5]],
        '10W': [[1, 2], [3, 6], [4, 5], [7, 10], [8, 9]],
        '12W': [[1, 2], [3, 4], [5, 8], [6, 7], [9, 12], [10, 11]]
    }

    available_wheel_types = df['Wheel_Type'].unique()
    selected_wheel_type = st.selectbox("Select Wheel Type:", available_wheel_types, index=available_wheel_types.tolist().index('10W') if '10W' in available_wheel_types else 0)
    df_wheel_type = df[df['Wheel_Type'] == selected_wheel_type]
    if df_wheel_type.empty:
        st.warning("No data for this wheel type.")
        return

    hitting_types = df_wheel_type['Hitting_Type'].unique()
    selected_hitting_type = st.selectbox("Select Hitting Type:", hitting_types)
    df_filtered = df_wheel_type[df_wheel_type['Hitting_Type'] == selected_hitting_type]
    if df_filtered.empty:
        st.warning("No data for the selected hitting type.")
        return

    position_pairs = tire_position_pairs.get(selected_wheel_type, [])
    if not position_pairs:
        st.warning("No tire pairs defined for this wheel type.")
        return

    tabs = st.tabs([f"Tire Positions {p[0]} & {p[1]}" for p in position_pairs])

    wheel_type_directories = {
        '6W': '6 wheels',
        '10W': '10 wheels',
        '12W': '12 wheels'
    }
    hitting_type_directories = {
        'Side': 'Strong-Side Time-Domain Segments',
        'Tread': 'Strong-Tread Time-Domain Segments'
    }

    wheel_directory = wheel_type_directories.get(selected_wheel_type)
    hitting_directory = hitting_type_directories.get(selected_hitting_type)
    if not wheel_directory or not hitting_directory:
        st.error("Invalid directories.")
        return

    @st.cache_data
    def get_available_pressures_and_segments(data_dir, wheel_directory, hitting_directory, pair, selected_hitting_type):
        directory = data_dir / wheel_directory / hitting_directory
        files = list(directory.glob("*.xlsx"))
        available_pressures = set()
        segment_numbers = set()
        pattern = rf"Strong {re.escape(selected_hitting_type)} Time Domain segment (\d+)-POS-(\d+)\s*\.xlsx"

        for file_path in files:
            file_name = file_path.name
            match = re.match(pattern, file_name)
            if match:
                pressure = int(match.group(1))
                pos = int(match.group(2))
                if pos in pair:
                    available_pressures.add(pressure)
                    try:
                        df_waveform = pd.read_excel(file_path, sheet_name='Step1_Data', index_col='Segment_ID')
                    except:
                        continue
                    for seg_id in df_waveform.index:
                        seg_match = re.match(r'signal segment (\d+)', seg_id, re.IGNORECASE)
                        if seg_match:
                            segment_number = int(seg_match.group(1))
                            segment_numbers.add(segment_number)
        return sorted(available_pressures), sorted(segment_numbers)

    for tab, pair in zip(tabs, position_pairs):
        with tab:
            st.markdown(f"### Tire Positions {pair[0]} & {pair[1]}")

            # Plot median pulse widths
            fig = make_subplots(rows=1, cols=4, subplot_titles=[f"Intensity Threshold {thr}" for thr in intensity_threshold_filter])
            for i, intensity_threshold in enumerate(intensity_threshold_filter, start=1):
                df_intensity = df_filtered[(df_filtered['Intensity_Threshold'] == intensity_threshold) & (df_filtered['Tire_Position'].isin(pair))]
                if df_intensity.empty:
                    continue

                x_min = df_filtered['Air_Pressure'].min() - 50
                x_max = df_filtered['Air_Pressure'].max() + 50
                y_min = 0
                y_max = df_filtered['Median_Pulse_Width'].max() + 10

                position_colors = ['blue', 'orange']
                for pos, color in zip(pair, position_colors):
                    df_pos = df_intensity[df_intensity['Tire_Position'] == pos]
                    if df_pos.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=df_pos['Air_Pressure'],
                            y=df_pos['Median_Pulse_Width'],
                            mode='markers',
                            name=f'Pos {pos}',
                            marker=dict(color=color, size=8, opacity=0.7),
                            legendgroup=f'Pos {pos}',
                            showlegend=(i == 1)
                        ),
                        row=1, col=i
                    )

                if len(df_intensity) >= 2:
                    X = df_intensity['Air_Pressure']
                    Y = df_intensity['Median_Pulse_Width']
                    X_const = sm.add_constant(X)
                    model = sm.OLS(Y, X_const)
                    results = model.fit()
                    Y_pred = results.predict(X_const)
                    fig.add_trace(
                        go.Scatter(
                            x=X, y=Y_pred,
                            mode='lines',
                            line=dict(color='black', width=2, dash='dash'),
                            name='Trend Line',
                            legendgroup='Trend Line',
                            showlegend=(i == 1)
                        ),
                        row=1, col=i
                    )
                    fig.add_annotation(
                        x=0.95, y=0.05,
                        xref='x domain', yref='y domain',
                        text=f'R²={results.rsquared:.2f}',
                        showarrow=False,
                        xanchor='right', yanchor='bottom',
                        font=dict(size=12),
                        row=1, col=i
                    )

                fig.update_xaxes(title_text='Air Pressure', range=[x_min, x_max], row=1, col=i)
                fig.update_yaxes(title_text='Median Pulse Width', range=[y_min, y_max], row=1, col=i)

            fig.update_layout(
                title=(
                    f'Median Pulse Width vs Air Pressure for Tire Positions {pair[0]} & {pair[1]}<br>'
                    f'<span style="font-size:12px;">Wheel Type: {selected_wheel_type}, Hitting Type: {selected_hitting_type}</span>'
                ),
                template='plotly_white',
                height=600,
                showlegend=True,
                legend_title="Tire Positions"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Waveforms and cumulative segments visualization
            with st.expander(f"View Waveforms and Cumulative Segments for Tire Positions {pair[0]} & {pair[1]}"):
                st.markdown("#### Waveform Visualization")
                available_pressures, segment_numbers = get_available_pressures_and_segments(
                    data_dir, wheel_directory, hitting_directory, pair, selected_hitting_type
                )
                if not available_pressures:
                    st.warning("No available pressures.")
                    continue
                if not segment_numbers:
                    st.warning("No segments found.")
                    continue

                selected_pressure = st.selectbox("Select Pressure Level:", available_pressures, key=f"press_{pair}")
                selected_segment = st.slider(
                    "Select Signal Segment:",
                    min_value=min(segment_numbers),
                    max_value=max(segment_numbers),
                    value=min(segment_numbers),
                    key=f"seg_{pair}"
                )

                if selected_pressure and selected_segment:
                    # Waveform visualization
                    fig_waveform = make_subplots(
                        rows=1,
                        cols=len(pair),
                        subplot_titles=[f"Tire Position {p}" for p in pair]
                    )
                    waveforms_loaded = False

                    file_pattern = f"Strong {selected_hitting_type} Time Domain segment {selected_pressure}-POS-{{pos}}*.xlsx"
                    for idx_p, pos in enumerate(pair):
                        files = list((data_dir / wheel_directory / hitting_directory).glob(file_pattern.format(pos=pos)))
                        if not files:
                            st.warning(f"No file found for Pressure {selected_pressure}, POS {pos}")
                            continue
                        file_path = files[0]
                        try:
                            df_waveform = pd.read_excel(file_path, sheet_name='Step1_Data', index_col='Segment_ID')
                        except Exception as e:
                            st.error(f"Error reading waveform: {e}")
                            continue

                        seg_id = f"signal segment {selected_segment}"
                        if seg_id not in df_waveform.index:
                            st.warning(f"{seg_id} not found in {file_path.name}")
                            continue

                        signal_data = df_waveform.loc[seg_id].values
                        waveforms_loaded = True

                        fig_waveform.add_trace(
                            go.Scatter(
                                y=signal_data,
                                mode='lines',
                                name=f"Pressure {selected_pressure}, Segment {selected_segment}",
                                legendgroup=f"Pressure {selected_pressure}",
                                hovertemplate='Index: %{x}<br>Value: %{y:.4f}<extra></extra>'
                            ),
                            row=1, col=idx_p+1
                        )

                        fig_waveform.update_yaxes(title_text='Normalized Signal Value', row=1, col=idx_p+1)
                        fig_waveform.update_xaxes(title_text='Signal Value', row=1, col=idx_p+1)

                        # Annotate first rise and max trim if Trim = Yes
                        if trim_option:
                            threshold_to_check = intensity_threshold_filter[0]
                            sheet_name_3 = f"{step3_prefix}_{threshold_to_check}"
                            try:
                                df_step3_check = pd.read_excel(file_path, sheet_name=sheet_name_3, index_col='Segment_ID')
                            except:
                                df_step3_check = pd.DataFrame()

                            if seg_id in df_step3_check.index:
                                fii = df_step3_check.loc[seg_id, 'First_Noticeable_Increase_Index']
                                if not pd.isna(fii):
                                    # Annotate first noticeable increase
                                    fig_waveform.add_shape(
                                        type='line',
                                        x0=fii, x1=fii, y0=0, y1=max(signal_data),
                                        line=dict(color='red', dash='dot'),
                                        row=1, col=idx_p+1
                                    )
                                    fig_waveform.add_annotation(
                                        x=fii, y=max(signal_data),
                                        text=f"Rise at {int(fii)}",
                                        showarrow=True, arrowhead=2,
                                        font=dict(color='red'),
                                        arrowcolor='red',
                                        row=1, col=idx_p+1
                                    )
                                    # Annotate max trimmed point
                                    max_trim_index = fii + selected_dims_after_rise_point
                                    if max_trim_index > len(signal_data):
                                        max_trim_index = len(signal_data)

                                    fig_waveform.add_shape(
                                        type='line',
                                        x0=max_trim_index, x1=max_trim_index, y0=0, y1=max(signal_data),
                                        line=dict(color='green', dash='dot'),
                                        row=1, col=idx_p+1
                                    )
                                    fig_waveform.add_annotation(
                                        x=max_trim_index, y=max(signal_data),
                                        text=f"Trim at {int(max_trim_index)}",
                                        showarrow=True, arrowhead=2,
                                        font=dict(color='green'),
                                        arrowcolor='green',
                                        row=1, col=idx_p+1
                                    )

                    if waveforms_loaded:
                        fig_waveform.update_layout(
                            title=(
                                f"Waveforms for Tire Positions {pair[0]} & {pair[1]}<br>"
                                f"<span style='font-size:12px;'>Hitting Type: {selected_hitting_type}</span>"
                            ),
                            template='plotly_white',
                            height=500,
                            showlegend=True
                        )
                        st.plotly_chart(fig_waveform, use_container_width=True)
                    else:
                        st.info("No waveforms loaded.")

                    st.markdown("#### Cumulative Segment Visualization")
                    tabs_cumulative = st.tabs([f"Intensity Threshold {th}" for th in intensity_threshold_filter])

                    for tab_cum, threshold in zip(tabs_cumulative, intensity_threshold_filter):
                        with tab_cum:
                            st.write(f"**Intensity Threshold: {threshold}**")
                            fig_cumulative = make_subplots(
                                rows=1,
                                cols=len(pair),
                                subplot_titles=[f"Tire Position {p}" for p in pair]
                            )
                            data_loaded = False

                            file_pattern_cum = f"Strong {selected_hitting_type} Time Domain segment {selected_pressure}-POS-{{pos}}*.xlsx"
                            for idx_p, pos in enumerate(pair):
                                files_cum = list((data_dir / wheel_directory / hitting_directory).glob(file_pattern_cum.format(pos=pos)))
                                if not files_cum:
                                    continue
                                file_path_cum = files_cum[0]

                                try:
                                    df_cumulative = pd.read_excel(file_path_cum, sheet_name=step2sj_name, index_col='Segment_ID')
                                    df_step3_cum = pd.read_excel(file_path_cum, sheet_name=f"{step3_prefix}_{threshold}", index_col='Segment_ID')
                                except:
                                    continue

                                seg_id = f"signal segment {selected_segment}"
                                if seg_id not in df_cumulative.index:
                                    continue

                                cum_values = df_cumulative.loc[seg_id].values

                                # Determine x-values based on trim option
                                if trim_option:
                                    # Use actual indices from column names for trimmed data
                                    x_vals = [int(col.split('_')[-1]) for col in df_cumulative.columns]
                                else:
                                    # Use sequential indices for full data
                                    x_vals = list(range(1, len(cum_values) + 1))

                                fig_cumulative.add_trace(
                                    go.Scatter(
                                        x=x_vals,
                                        y=cum_values,
                                        mode='lines',
                                        name=f'Segment {selected_segment}',
                                        line=dict(width=2),
                                        legendgroup=seg_id,
                                        hovertemplate='Segment: %{text}<br>Signal Value: %{x}<br>Cumulative: %{y:.4f}<extra></extra>',
                                        text=[seg_id]*len(x_vals),
                                        showlegend=(idx_p == 0)
                                    ),
                                    row=1, col=idx_p+1
                                )
                                data_loaded = True

                                # Add annotations for key points
                                if seg_id in df_step3_cum.index:
                                    fi_val = df_step3_cum.loc[seg_id, 'First_Noticeable_Increase_Cumulative_Value']
                                    pe_val = df_step3_cum.loc[seg_id, 'Point_Exceeds_Cumulative_Value']
                                    x_fi = df_step3_cum.loc[seg_id, 'First_Noticeable_Increase_Index']
                                    x_pe = df_step3_cum.loc[seg_id, 'Point_Exceeds_Index']

                                    # First increase point
                                    if not pd.isna(x_fi) and not pd.isna(fi_val):
                                        fig_cumulative.add_trace(
                                            go.Scatter(
                                                x=[x_fi],
                                                y=[fi_val],
                                                mode='markers',
                                                marker=dict(color='red', size=10),
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ),
                                            row=1, col=idx_p+1
                                        )
                                        fig_cumulative.add_annotation(
                                            x=x_fi,
                                            y=fi_val,
                                            text=f"First Increase<br>({int(x_fi)}, {fi_val:.4f})",
                                            showarrow=True,
                                            arrowhead=2,
                                            ax=0,
                                            ay=-40,
                                            font=dict(color='red'),
                                            arrowcolor='red',
                                            xref=f'x{idx_p+1}',
                                            yref=f'y{idx_p+1}'
                                        )

                                    # Exceeds threshold point
                                    if not pd.isna(x_pe) and not pd.isna(pe_val):
                                        fig_cumulative.add_trace(
                                            go.Scatter(
                                                x=[x_pe],
                                                y=[pe_val],
                                                mode='markers',
                                                marker=dict(color='green', size=10),
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ),
                                            row=1, col=idx_p+1
                                        )
                                        fig_cumulative.add_annotation(
                                            x=x_pe,
                                            y=pe_val,
                                            text=f"Exceeds {threshold}<br>({int(x_pe)}, {pe_val:.4f})",
                                            showarrow=True,
                                            arrowhead=2,
                                            ax=0,
                                            ay=40,
                                            font=dict(color='green'),
                                            arrowcolor='green',
                                            xref=f'x{idx_p+1}',
                                            yref=f'y{idx_p+1}'
                                        )

                                fig_cumulative.update_xaxes(title_text='Signal Value', row=1, col=idx_p+1)
                                fig_cumulative.update_yaxes(title_text='Denoised Cumulative Sum (Sj)', row=1, col=idx_p+1)

                            if data_loaded:
                                fig_cumulative.update_layout(
                                    title=(
                                        f"Cumulative Segments for Positions {pair[0]} & {pair[1]}<br>"
                                        f"<span style='font-size:12px;'>Threshold: {threshold}</span>"
                                    ),
                                    template='plotly_white',
                                    height=500,
                                    showlegend=True
                                )
                                st.plotly_chart(fig_cumulative, use_container_width=True)
                            else:
                                st.info("No cumulative segments loaded.")
                else:
                    st.info("Please select a pressure level and a segment to proceed.")

def plot_standalone_analysis(data_dir, intensity_threshold_filter, median_pulse_widths_file, 
                           step3_prefix, step2sj_name):
    """
    Enhanced standalone analysis plot with comprehensive filtering and trend lines
    """
    st.markdown("<h2 style='color: #1f77b4;'>Stand-Alone Tire Analysis</h2>", unsafe_allow_html=True)
    
    if not median_pulse_widths_file.exists():
        st.error(f"Median pulse widths file not found: {median_pulse_widths_file}")
        return

    try:
        df = pd.read_excel(median_pulse_widths_file)
    except Exception as e:
        st.error(f"Error reading median pulse widths: {e}")
        return

    # Create filtering section with columns
    st.markdown("### Data Filtering")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    # Primary grouping selection
    group_options = {
        'Tire Size': 'TireSize',
        'Wear': 'Wear',
        'Rim': 'Rim',
        'Tire Type': 'Tire_Type'
    }
    
    with filter_col1:
        selected_group = st.selectbox(
            "Group by:", 
            options=list(group_options.keys()),
            index=0  # Default to Tire Size
        )
        
        # Add checkbox for trend lines
        show_trend_lines = st.checkbox("Show Trend Lines", value=False)
        
    # Initialize filter variables
    tire_size_filter = None
    wear_filter = None
    rim_filter = None
    tire_type_filter = None
        
    with filter_col2:
        if selected_group != 'Tire Size':
            tire_sizes = sorted(df['TireSize'].unique())
            tire_size_filter = st.selectbox('Filter by Tire Size:', ['All'] + tire_sizes)
        
        if selected_group != 'Wear':
            wear_conditions = sorted(df['Wear'].unique())
            wear_filter = st.selectbox('Filter by Wear:', ['All'] + wear_conditions)
            
    with filter_col3:
        if selected_group != 'Rim':
            rim_types = sorted(df['Rim'].unique())
            rim_filter = st.selectbox('Filter by Rim:', ['All'] + rim_types)
            
    with filter_col4:
        if selected_group != 'Tire Type':
            tire_types = sorted(df['Tire_Type'].unique())
            tire_type_filter = st.selectbox('Filter by Tire Type:', ['All'] + tire_types)

    # Apply all filters
    df_filtered = df.copy()
    
    if tire_size_filter and tire_size_filter != 'All':
        df_filtered = df_filtered[df_filtered['TireSize'] == tire_size_filter]
    
    if wear_filter and wear_filter != 'All':
        df_filtered = df_filtered[df_filtered['Wear'] == wear_filter]
        
    if rim_filter and rim_filter != 'All':
        df_filtered = df_filtered[df_filtered['Rim'] == rim_filter]
        
    if tire_type_filter and tire_type_filter != 'All':
        df_filtered = df_filtered[df_filtered['Tire_Type'] == tire_type_filter]

    # Get the column name for grouping
    group_column = group_options[selected_group]
    
    # Get unique values for the selected grouping
    unique_values = sorted(df_filtered[group_column].unique())
    
    if not unique_values:
        st.warning("No data available for the selected filters.")
        return

    # Update title to reflect all active filters
    active_filters = []
    if tire_size_filter and tire_size_filter != 'All':
        active_filters.append(f"Tire Size: {tire_size_filter}")
    if wear_filter and wear_filter != 'All':
        active_filters.append(f"Wear: {wear_filter}")
    if rim_filter and rim_filter != 'All':
        active_filters.append(f"Rim: {rim_filter}")
    if tire_type_filter and tire_type_filter != 'All':
        active_filters.append(f"Tire Type: {tire_type_filter}")

    # Calculate global y-axis range
    y_min = df_filtered['Median_Pulse_Width'].min()
    y_max = df_filtered['Median_Pulse_Width'].max()
    y_padding = (y_max - y_min) * 0.05
    y_min = y_min - y_padding
    y_max = y_max + y_padding

    # Define a fixed color palette for better distinction
    def get_distinct_colors(n):
        colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # yellow-green
            '#17becf'   # cyan
        ]
        return colors[:n]

    # Get colors for unique values
    colors = get_distinct_colors(len(unique_values))
    color_map = dict(zip(unique_values, colors))

    # Plot subplots
    fig = make_subplots(rows=1, cols=4, 
                       subplot_titles=[f"Intensity Threshold {thr}" for thr in intensity_threshold_filter],
                       horizontal_spacing=0.05)
    
    # Plot points for each threshold
    for i, intensity_threshold in enumerate(intensity_threshold_filter, start=1):
        df_intensity = df_filtered[df_filtered['Intensity_Threshold'] == intensity_threshold]
        if df_intensity.empty:
            continue

        for value in unique_values:
            df_group = df_intensity[df_intensity[group_column] == value]
            if df_group.empty:
                continue
                
            hover_text = [
                f'{selected_group}: {value}<br>'
                f'Tire Number: {tire_num}<br>'
                f'Tire Type: {tire_type}<br>'
                f'Air Pressure: {pressure}<br>'
                f'Median Pulse Width: {pulse_width:.1f}'
                for tire_num, tire_type, pressure, pulse_width in 
                zip(df_group['Tire_Number'], df_group['Tire_Type'], df_group['Pressure'], df_group['Median_Pulse_Width'])
            ]
                
            # Add scatter plot
            scatter_trace = go.Scatter(
                x=df_group['Pressure'],
                y=df_group['Median_Pulse_Width'],
                mode='markers',
                name=f'{value}',
                marker=dict(
                    color=color_map[value],
                    size=8,
                    opacity=0.9
                ),
                showlegend=(i == 1),
                legendgroup=str(value),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text
            )
            fig.add_trace(scatter_trace, row=1, col=i)

            # Add trend line if enabled
            if show_trend_lines and len(df_group) >= 2:
                X = df_group['Pressure']
                Y = df_group['Median_Pulse_Width']
                X_const = sm.add_constant(X)
                model = sm.OLS(Y, X_const)
                results = model.fit()
                Y_pred = results.predict(X_const)

                # Add trend line and R² annotation as a single trace
                trend_trace = go.Scatter(
                    x=list(X) + [None] + [X.iloc[-1]],  # Add None to create a break in the line
                    y=list(Y_pred) + [None] + [Y_pred.iloc[-1]],
                    mode='lines+text',
                    name=f'Trend: {value}',
                    text=[''] * len(X) + [''] + [f'R² = {results.rsquared:.3f}'],
                    textposition='bottom right',
                    textfont=dict(color=color_map[value]),
                    line=dict(
                        color=color_map[value],
                        width=2,
                        dash='dash'
                    ),
                    showlegend=(i == 1),
                    legendgroup=str(value),
                    hovertemplate='R² = ' + f'{results.rsquared:.3f}<extra></extra>'
                )
                fig.add_trace(trend_trace, row=1, col=i)
                
                # Calculate and add point P (700 kPa)
                if min(X) <= 700 <= max(X):
                    # Create prediction point ensuring correct shape
                    X_700 = np.array([700]).reshape(-1, 1)
                    X_700_const = sm.add_constant(X_700, has_constant='add')
                    Y_700 = results.predict(X_700_const)
                    
                    # Add point P marker with diamond shape and larger size
                    point_p = go.Scatter(
                        x=[700],
                        y=[Y_700[0]],
                        mode='markers+text',
                        marker=dict(
                            symbol='diamond',
                            size=10,
                            color=color_map[value],
                            line=dict(color='red', width=2),
                            opacity=0.5 
                        ),
                        text=['P'],
                        textposition='bottom left',
                        textfont=dict(
                            color=color_map[value],
                            size=10,
                            family='Arial'
                        ),
                        showlegend=False,
                        legendgroup=str(value),  # Link to the same legend group as trend line
                        hovertemplate=f'Point P<br>Pressure: 700 kPa<br>Pulse Width: %{{y:.1f}}<extra></extra>'
                    )
                    fig.add_trace(point_p, row=1, col=i)

        fig.update_xaxes(title_text='Air Pressure', row=1, col=i)
        fig.update_yaxes(title_text='Median Pulse Width', range=[y_min, y_max], row=1, col=i)

    title_text = 'Median Pulse Width vs Air Pressure'
    if active_filters:
        filters_text = ' | '.join(active_filters)
        title_text += f'<br><span style="font-size:12px;">Filters: {filters_text}</span>'
    title_text += f'<br><span style="font-size:12px;">Grouped by {selected_group}</span>'

    fig.update_layout(
        title=title_text,
        template='plotly_white',
        height=600,
        margin=dict(r=80, t=100), 
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


    # Waveform and cumulative visualization
    with st.expander("View Waveforms and Cumulative Segments"):
        # Get available files
        excel_files = list(data_dir.glob('*.xlsx'))
        excel_files = [f for f in excel_files if f.name != 'Median_Pulse_Widths_StandAlone.xlsx']
        
        if not excel_files:
            st.warning("No data files found.")
            return

        # Read metadata from the first row of each file to create selection options
        file_metadata = {}
        for file in excel_files:
            try:
                df_metadata = pd.read_excel(file, sheet_name='Step3_DataPts_0.5', nrows=1)
                metadata = {
                    'TireSize': df_metadata['TireSize'].iloc[0],
                    'Pressure': df_metadata['Pressure'].iloc[0],
                    'Tire_Number': df_metadata['Tire_Number'].iloc[0],
                    'Tire_Type': df_metadata['Tire_Type'].iloc[0],
                    'Wear': df_metadata['Wear'].iloc[0],
                    'Rim': df_metadata['Rim'].iloc[0]
                }
                file_metadata[file] = metadata
            except Exception as e:
                continue

        if not file_metadata:
            st.error("No valid files found.")
            return

        # Create two columns for the selection dropdowns
        col1, col2 = st.columns(2)
        
        # Get unique pressures and sort them
        available_pressures = sorted(set(meta['Pressure'] for meta in file_metadata.values()))
        
        # Pressure selection in first column
        with col1:
            selected_pressure = st.selectbox("Select Pressure Level:", available_pressures)
        
        # Filter files for selected pressure
        pressure_files = {f: meta for f, meta in file_metadata.items() 
                        if meta['Pressure'] == selected_pressure}
        
        # Tire selection in second column
        with col2:
            available_tires = sorted(set(meta['Tire_Number'] for meta in pressure_files.values()))
            selected_tire = st.selectbox("Select Tire:", available_tires)

        # Get selected file
        selected_file = next((f for f, meta in pressure_files.items() 
                            if meta['Tire_Number'] == selected_tire), None)

        if selected_file is None:
            st.error("Selected file not found.")
            return

        try:
            df_step1 = pd.read_excel(selected_file, sheet_name='Step1_Data')
            # Extract segment numbers and create slider
            segment_numbers = []
            for seg_id in df_step1['Segment_ID']:
                match = re.match(r'signal segment (\d+)', seg_id, re.IGNORECASE)
                if match:
                    segment_numbers.append(int(match.group(1)))
            
            if segment_numbers:
                selected_segment_num = st.slider("Select Signal Segment:", 
                                            min_value=min(segment_numbers),
                                            max_value=max(segment_numbers),
                                            value=min(segment_numbers))
                selected_segment = f"signal segment {selected_segment_num}"
            else:
                st.error("No valid segments found.")
                return
                
        except Exception as e:
            st.error(f"Error reading segments: {e}")
            return

        if selected_segment:
            try:
                # Waveform visualization
                st.markdown("#### Waveform")
                df_waveform = pd.read_excel(selected_file, sheet_name='Step1_Data', index_col='Segment_ID')
                waveform_data = df_waveform.loc[selected_segment].values
                
                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(
                    y=waveform_data,
                    mode='lines',
                    name='Waveform',
                    line=dict(width=2)
                ))

                # Read rise point from Step3_DataPts_0.5
                df_step3_base = pd.read_excel(selected_file, sheet_name='Step3_DataPts_0.5', index_col='Segment_ID')
                
                if selected_segment in df_step3_base.index:
                    # Add rise point annotation
                    rise_index = df_step3_base.loc[selected_segment, 'First_Noticeable_Increase_Index']
                    if not pd.isna(rise_index):
                        rise_index = int(rise_index)
                        # Add vertical line at rise point
                        fig_wave.add_shape(
                            type='line',
                            x0=rise_index, x1=rise_index,
                            y0=min(waveform_data), y1=max(waveform_data),
                            line=dict(color='red', dash='dot'),
                        )
                        # Add annotation for rise point
                        fig_wave.add_annotation(
                            x=rise_index,
                            y=max(waveform_data),
                            text=f"Rise at {rise_index}",
                            showarrow=True,
                            arrowhead=2, 
                            font=dict(color='red'),
                            arrowcolor='red'
                        )

                        # Add percentile points using Step3 data sheets
                        thresholds = [0.5, 0.7, 0.8, 0.9]
                        colors = ['blue', 'green', 'purple', 'orange']
                        
                        for threshold, color in zip(thresholds, colors):
                            sheet_name = f'Step3_DataPts_{threshold}'
                            try:
                                df_step3 = pd.read_excel(selected_file, sheet_name=sheet_name, index_col='Segment_ID')
                                if selected_segment in df_step3.index:
                                    percentile_index = df_step3.loc[selected_segment, 'Point_Exceeds_Index']
                                    
                                    if not pd.isna(percentile_index):
                                        percentile_index = int(percentile_index)
                                        # Add vertical line at percentile point
                                        fig_wave.add_shape(
                                            type='line',
                                            x0=percentile_index, x1=percentile_index,
                                            y0=min(waveform_data), y1=max(waveform_data),
                                            line=dict(color=color, dash='dot'),
                                        )
                                        # Add annotation for percentile point
                                        fig_wave.add_annotation(
                                            x=percentile_index,
                                            y=max(waveform_data) * (0.9 - (thresholds.index(threshold) * 0.15)),  # Stagger annotations
                                            text=f"{int(threshold*100)}th point<br>at {percentile_index}",
                                            showarrow=True,
                                            arrowhead=2,
                                            font=dict(color=color),
                                            arrowcolor=color
                                        )
                            except Exception as e:
                                st.warning(f"Could not read data for {int(threshold*100)}th percentile: {e}")

                fig_wave.update_layout(
                    title=f"Waveform for Tire {selected_tire}, Pressure {selected_pressure}, {selected_segment}",
                    xaxis_title="Signal Value",
                    yaxis_title="Normalized Signal Value",
                    template='plotly_white',
                    height=500
                )
                st.plotly_chart(fig_wave, use_container_width=True)

                # Cumulative visualization
                st.markdown("#### Cumulative Segments")
                tabs = st.tabs([f"Intensity Threshold {th}" for th in intensity_threshold_filter])
                
                for tab, threshold in zip(tabs, intensity_threshold_filter):
                    with tab:
                        df_cumulative = pd.read_excel(selected_file, sheet_name=step2sj_name, index_col='Segment_ID')
                        df_step3 = pd.read_excel(selected_file, sheet_name=f'{step3_prefix}_{threshold}', index_col='Segment_ID')
                        
                        cum_values = df_cumulative.loc[selected_segment].values
                        x_values = list(range(1, len(cum_values) + 1))
                        
                        fig_cum = go.Figure()
                        fig_cum.add_trace(go.Scatter(
                            x=x_values,
                            y=cum_values,
                            mode='lines',
                            name='Cumulative Sum',
                            line=dict(width=2),
                            hovertemplate='Signal Value: %{x}<br>Cumulative: %{y:.4f}<extra></extra>'
                        ))

                        # Add markers and annotations for key points
                        if selected_segment in df_step3.index:
                            first_increase_idx = df_step3.loc[selected_segment, 'First_Noticeable_Increase_Index']
                            first_increase_val = df_step3.loc[selected_segment, 'First_Noticeable_Increase_Cumulative_Value']
                            point_exceeds_idx = df_step3.loc[selected_segment, 'Point_Exceeds_Index']
                            point_exceeds_val = df_step3.loc[selected_segment, 'Point_Exceeds_Cumulative_Value']

                            # First increase point
                            if not pd.isna(first_increase_idx) and not pd.isna(first_increase_val):
                                fig_cum.add_trace(go.Scatter(
                                    x=[first_increase_idx],
                                    y=[first_increase_val],
                                    mode='markers',
                                    marker=dict(color='red', size=10),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig_cum.add_annotation(
                                    x=first_increase_idx,
                                    y=first_increase_val,
                                    text=f"First Increase<br>({int(first_increase_idx)}, {first_increase_val:.4f})",
                                    showarrow=True,
                                    arrowhead=2,
                                    ax=0,
                                    ay=-40,
                                    font=dict(color='red'),
                                    arrowcolor='red'
                                )

                            # Exceeds threshold point
                            if not pd.isna(point_exceeds_idx) and not pd.isna(point_exceeds_val):
                                fig_cum.add_trace(go.Scatter(
                                    x=[point_exceeds_idx],
                                    y=[point_exceeds_val],
                                    mode='markers',
                                    marker=dict(color='green', size=10),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig_cum.add_annotation(
                                    x=point_exceeds_idx,
                                    y=point_exceeds_val,
                                    text=f"Exceeds {threshold}<br>({int(point_exceeds_idx)}, {point_exceeds_val:.4f})",
                                    showarrow=True,
                                    arrowhead=2,
                                    ax=0,
                                    ay=40,
                                    font=dict(color='green'),
                                    arrowcolor='green'
                                )

                        fig_cum.update_layout(
                            title=f"Cumulative Sum for Tire {selected_tire}, Pressure {selected_pressure}, {selected_segment}",
                            xaxis_title="Signal Value",
                            yaxis_title="Denoised Cumulative Sum (Sj)",
                            template='plotly_white',
                            height=500
                        )
                        st.plotly_chart(fig_cum, use_container_width=True)

            except Exception as e:
                st.error(f"Error displaying visualizations: {e}")

if __name__ == '__main__':
    main()
