# src/myace/plotting.py
import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path

def _check_plotly():
    """Checks if plotly is installed and returns key functions."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError:
        raise ImportError(
            "Plotting functionality requires plotly. "
            "Please install it using 'pip install myace[plotting]' or 'pip install plotly'."
        )

def calculate_fit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various error metrics from a pacemaker prediction DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing prediction results, typically
                           from a 'train_pred.pckl.gzip' file. It must contain
                           'energy_corrected_per_atom', 'energy_pred', 'forces',
                           and 'forces_pred' columns.

    Returns:
        pd.DataFrame: A new DataFrame with added metric columns.
    """
    df_out = df.copy()
    
    # Calculate energy metrics
    if 'energy_pred_per_atom' not in df_out.columns and 'energy_pred' in df_out.columns and 'NUMBER_OF_ATOMS' in df_out.columns:
        df_out['energy_pred_per_atom'] = df_out['energy_pred'] / df_out['NUMBER_OF_ATOMS']
    
    if 'energy_pred_per_atom' in df_out.columns and 'energy_corrected_per_atom' in df_out.columns:
        df_out['energy_abs_error'] = np.abs(df_out['energy_pred_per_atom'] - df_out['energy_corrected_per_atom'])

    # Calculate force metrics
    def get_force_metrics_from_row(row):
        f_dft, f_ace = row.get('forces'), row.get('forces_pred')
        if not (isinstance(f_dft, np.ndarray) and isinstance(f_ace, np.ndarray) and f_dft.shape == f_ace.shape):
            return 0.0, 0.0, 0.0
        dft_norms = np.linalg.norm(f_dft, axis=1) if f_dft.size > 0 else np.array([0.0])
        ace_norms = np.linalg.norm(f_ace, axis=1) if f_ace.size > 0 else np.array([0.0])
        error_norms = np.linalg.norm(f_ace - f_dft, axis=1) if f_dft.size > 0 else np.array([0.0])
        return np.max(dft_norms), np.max(ace_norms), np.max(error_norms)

    force_metrics = df_out.apply(lambda row: pd.Series(get_force_metrics_from_row(row)), axis=1)
    force_metrics.columns = ['dft_max_force_norm', 'ace_max_force_norm', 'force_max_error']
    
    return pd.concat([df_out, force_metrics], axis=1)


def plot_fit_analysis(df_metrics: pd.DataFrame, output_filename: Union[str, Path] = "fit_analysis.html"):
    """
    Generates a comprehensive 3-panel plot for analyzing fitting results.

    Args:
        df_metrics (pd.DataFrame): A DataFrame that has been processed by
                                   `calculate_fit_metrics`.
        output_filename (Union[str, Path], optional): The name of the output HTML file.
                                                       Defaults to "fit_analysis.html".
    """
    go, make_subplots = _check_plotly()

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Energy Correlation", "Max Force Correlation", "Energy vs. Max Force Error")
    )

    # Figure 1: Energy Correlation
    if 'energy_corrected_per_atom' in df_metrics.columns and 'energy_pred_per_atom' in df_metrics.columns:
        fig.add_trace(go.Scatter(
            x=df_metrics['energy_corrected_per_atom'], y=df_metrics['energy_pred_per_atom'],
            mode='markers', marker=dict(size=5, color='royalblue'),
            hovertext=[f"Name: {name}<br>Index: {idx}<br>E_Err: {err*1000:.1f} meV/at" 
                       for name, idx, err in zip(df_metrics.get('name', ''), df_metrics.index, df_metrics.get('energy_abs_error', 0))],
            hoverinfo='text'
        ), row=1, col=1)

    # Figure 2: Max Force Correlation
    if 'dft_max_force_norm' in df_metrics.columns and 'ace_max_force_norm' in df_metrics.columns:
        fig.add_trace(go.Scatter(
            x=df_metrics['dft_max_force_norm'], y=df_metrics['ace_max_force_norm'],
            mode='markers', marker=dict(size=5, color='royalblue'),
            hovertext=[f"Name: {name}<br>Index: {idx}<br>F_Err_max: {err:.3f}"
                       for name, idx, err in zip(
                           df_metrics.get('name', ''), df_metrics.index, df_metrics.get('force_max_error', 0))],
            hoverinfo='text'
        ), row=2, col=1)

    # Figure 3: Energy vs. Force Error
    if 'energy_corrected_per_atom' in df_metrics.columns and 'force_max_error' in df_metrics.columns:
        fig.add_trace(go.Scatter(
            x=df_metrics['energy_corrected_per_atom'], y=df_metrics['force_max_error'],
            mode='markers', marker=dict(size=5, color='royalblue'),
            hovertext=[f"Name: {name}<br>Index: {idx}<br>Max_F_Err: {err:.3f} eV/Å" 
                       for name, idx, err in zip(df_metrics.get('name', ''), df_metrics.index, df_metrics.get('force_max_error', 0))],
            hoverinfo='text'
        ), row=3, col=1)

    # --- Layout Updates ---
    if 'energy_corrected_per_atom' in df_metrics.columns:
        fig.update_xaxes(title_text="E(DFT), eV/at", row=1, col=1)
        fig.update_yaxes(title_text="E(ACE), eV/at", row=1, col=1)
        min_e, max_e = df_metrics['energy_corrected_per_atom'].min(), df_metrics['energy_corrected_per_atom'].max()
        fig.add_shape(type="line", x0=min_e, y0=min_e, x1=max_e, y1=max_e, line=dict(color="grey", width=1, dash="dash"), row=1, col=1)

    if 'dft_max_force_norm' in df_metrics.columns and 'ace_max_force_norm' in df_metrics.columns:
        fig.update_xaxes(title_text="Max Force Norm (DFT), eV/Å", row=2, col=1)
        fig.update_yaxes(title_text="Max Force Norm (ACE), eV/Å", row=2, col=1)
        all_max_forces = pd.concat([df_metrics['dft_max_force_norm'], df_metrics['ace_max_force_norm']])
        min_f_max, max_f_max = all_max_forces.min(), all_max_forces.max()
        fig.add_shape(type="line", x0=min_f_max, y0=min_f_max, x1=max_f_max, y1=max_f_max, line=dict(color="grey", width=1, dash="dash"), row=2, col=1)

    if 'energy_corrected_per_atom' in df_metrics.columns and 'force_max_error' in df_metrics.columns:
        fig.update_xaxes(title_text="E(DFT), eV/at", row=3, col=1)
        fig.update_yaxes(title_text="Max Force Norm Error, eV/Å", type="log", row=3, col=1)

    fig.update_layout(
        title_text="Comprehensive Per-Structure Error Analysis",
        showlegend=False, width=800, height=1500
    )

    fig.write_html(output_filename)
    print(f"Comprehensive analysis plot saved to {output_filename}")