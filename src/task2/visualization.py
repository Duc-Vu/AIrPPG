"""Visualization functions for rPPG signals and results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_rgb_signals(
    rgb_signals: np.ndarray,
    fs: float,
    roi_names: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot RGB signals for each ROI.
    
    Args:
        rgb_signals: Array of shape (N, 3, 3) with RGB values (N frames, 3 ROIs, 3 channels).
        fs: Sampling frequency in Hz.
        roi_names: List of ROI names. If None, uses default names.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    if roi_names is None:
        roi_names = ['forehead', 'left_cheek', 'right_cheek']
    
    n_frames, n_rois, _ = rgb_signals.shape
    time = np.arange(n_frames) / fs
    
    fig, axes = plt.subplots(n_rois, 1, figsize=(12, 3 * n_rois), sharex=True)
    if n_rois == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'blue']
    
    for roi_idx, ax in enumerate(axes):
        for channel_idx, color in enumerate(colors):
            ax.plot(time, rgb_signals[:, roi_idx, channel_idx], 
                   color=color, label=f'{color.capitalize()} channel', alpha=0.7)
        ax.set_ylabel('Intensity')
        ax.set_title(f'{roi_names[roi_idx].replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rppg_signals(
    rppg_signals: dict[str, np.ndarray],
    fs: float,
    roi_names: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot rPPG signals from different baseline methods.
    
    Args:
        rppg_signals: Dictionary with method names as keys and signal arrays as values.
                      Each array can be (N,) for single ROI or (N, 3) for multi-ROI.
        fs: Sampling frequency in Hz.
        roi_names: List of ROI names. If None, uses default names.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    if roi_names is None:
        roi_names = ['forehead', 'left_cheek', 'right_cheek']
    
    # Determine if multi-ROI
    first_signal = next(iter(rppg_signals.values()))
    is_multi_roi = first_signal.ndim == 2
    
    n_rois = first_signal.shape[1] if is_multi_roi else 1
    n_methods = len(rppg_signals)
    
    if is_multi_roi:
        fig, axes = plt.subplots(n_rois, n_methods, figsize=(4 * n_methods, 3 * n_rois), sharex=True)
        if n_rois == 1:
            axes = axes.reshape(1, -1)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
    else:
        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 3))
        axes = axes.reshape(1, -1)
    
    time = np.arange(first_signal.shape[0]) / fs
    
    for method_idx, (method_name, signal) in enumerate(rppg_signals.items()):
        if is_multi_roi:
            for roi_idx in range(n_rois):
                ax = axes[roi_idx, method_idx]
                ax.plot(time, signal[:, roi_idx], linewidth=1)
                ax.set_ylabel('Amplitude')
                if roi_idx == 0:
                    ax.set_title(f'{method_name.upper()}')
                if method_idx == 0:
                    ax.set_ylabel(f'{roi_names[roi_idx].replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        else:
            ax = axes[0, method_idx]
            ax.plot(time, signal, linewidth=1)
            ax.set_title(f'{method_name.upper()}')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
    
    axes[-1, -1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_power_spectrum(
    rppg_signal: np.ndarray,
    fs: float,
    hr_bpm: float | None = None,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot power spectrum of rPPG signal.
    
    Args:
        rppg_signal: rPPG signal array.
        fs: Sampling frequency in Hz.
        hr_bpm: Optional heart rate in BPM to highlight.
        min_hr: Minimum heart rate to display in BPM.
        max_hr: Maximum heart rate to display in BPM.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    from scipy import signal
    
    # Compute power spectrum
    n = len(rppg_signal)
    if n < 10:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'Signal too short', ha='center', va='center')
        return fig
    
    freqs, psd = signal.welch(rppg_signal, fs=fs, nperseg=min(256, n // 4))
    
    # Convert to BPM
    freqs_bpm = freqs * 60
    
    # Filter to HR range
    mask = (freqs_bpm >= min_hr) & (freqs_bpm <= max_hr)
    freqs_bpm = freqs_bpm[mask]
    psd = psd[mask]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs_bpm, psd, linewidth=1.5)
    ax.set_xlabel('Frequency (BPM)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Power Spectrum')
    ax.grid(True, alpha=0.3)
    
    if hr_bpm is not None:
        ax.axvline(hr_bpm, color='red', linestyle='--', linewidth=2, label=f'HR: {hr_bpm:.1f} BPM')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_hr_comparison(
    predicted_hr: np.ndarray,
    ground_truth_hr: np.ndarray,
    time: np.ndarray,
    method_name: str = "Predicted",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot predicted vs ground truth heart rate over time.
    
    Args:
        predicted_hr: Predicted heart rate values in BPM.
        ground_truth_hr: Ground truth heart rate values in BPM.
        time: Time array in seconds.
        method_name: Name of the prediction method.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(time, ground_truth_hr, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(time, predicted_hr, 'r--', linewidth=2, label=method_name, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heart Rate (BPM)')
    ax.set_title('Heart Rate Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_scatter_comparison(
    predicted_hr: np.ndarray,
    ground_truth_hr: np.ndarray,
    method_name: str = "Predicted",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot scatter plot of predicted vs ground truth heart rate.
    
    Args:
        predicted_hr: Predicted heart rate values in BPM.
        ground_truth_hr: Ground truth heart rate values in BPM.
        method_name: Name of the prediction method.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(predicted_hr) & ~np.isnan(ground_truth_hr)
    pred_valid = predicted_hr[valid_mask]
    gt_valid = ground_truth_hr[valid_mask]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(gt_valid, pred_valid, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(np.min(gt_valid), np.min(pred_valid))
    max_val = max(np.max(gt_valid), np.max(pred_valid))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Ground Truth HR (BPM)')
    ax.set_ylabel(f'{method_name} HR (BPM)')
    ax.set_title('Scatter Plot: Predicted vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    from scipy import stats
    r, _ = stats.pearsonr(gt_valid, pred_valid)
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_table(
    metrics_dict: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot metrics as a table.
    
    Args:
        metrics_dict: Dictionary with method names as keys and metric dicts as values.
                      Each metric dict should have keys like 'mae', 'rmse', 'pearson_r', etc.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    methods = list(metrics_dict.keys())
    metric_names = list(metrics_dict[methods[0]].keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for method in methods:
        row = [method]
        for metric in metric_names:
            value = metrics_dict[method][metric]
            if isinstance(value, float):
                row.append(f'{value:.3f}')
            else:
                row.append(str(value))
        table_data.append(row)
    
    # Add header
    table_data = [['Method'] + metric_names] + table_data
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(len(metric_names) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Baseline Method Comparison', pad=20)
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_preprocessing_pipeline(
    original_signal: np.ndarray,
    normalized_signal: np.ndarray,
    detrended_signal: np.ndarray,
    filtered_signal: np.ndarray,
    fs: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot signal at each preprocessing stage.
    
    Args:
        original_signal: Original signal.
        normalized_signal: Normalized signal.
        detrended_signal: Detrended signal.
        filtered_signal: Bandpass filtered signal.
        fs: Sampling frequency in Hz.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    n_samples = len(original_signal)
    time = np.arange(n_samples) / fs
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(time, original_signal, linewidth=1)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, normalized_signal, linewidth=1)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Normalized Signal')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, detrended_signal, linewidth=1)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Detrended Signal')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(time, filtered_signal, linewidth=1)
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Bandpass Filtered Signal (0.7-4.0 Hz)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
