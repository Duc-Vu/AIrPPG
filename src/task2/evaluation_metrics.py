"""Evaluation metrics for rPPG heart rate estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def compute_mae(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        predicted: Predicted heart rate values in BPM.
        ground_truth: Ground truth heart rate values in BPM.
        
    Returns:
        MAE in BPM.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(predicted) & ~np.isnan(ground_truth)
    if not np.any(valid_mask):
        return np.nan
    
    pred_valid = predicted[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    return float(np.mean(np.abs(pred_valid - gt_valid)))


def compute_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Root Mean Square Error.
    
    Args:
        predicted: Predicted heart rate values in BPM.
        ground_truth: Ground truth heart rate values in BPM.
        
    Returns:
        RMSE in BPM.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(predicted) & ~np.isnan(ground_truth)
    if not np.any(valid_mask):
        return np.nan
    
    pred_valid = predicted[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    return float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2)))


def compute_pearson_correlation(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Pearson correlation coefficient.
    
    Args:
        predicted: Predicted heart rate values in BPM.
        ground_truth: Ground truth heart rate values in BPM.
        
    Returns:
        Pearson correlation coefficient (r).
    """
    # Remove NaN values
    valid_mask = ~np.isnan(predicted) & ~np.isnan(ground_truth)
    if not np.any(valid_mask) or len(predicted[valid_mask]) < 2:
        return np.nan
    
    pred_valid = predicted[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    r, _ = stats.pearsonr(pred_valid, gt_valid)
    return float(r)


def compute_snr(
    rppg_signal: np.ndarray,
    fs: float,
    hr_bpm: float,
    bandwidth_hz: float = 0.2,
) -> float:
    """Compute Signal-to-Noise Ratio for rPPG signal.
    
    SNR is defined as the power in the HR frequency band divided by
    the power in the surrounding noise bands.
    
    Args:
        rppg_signal: rPPG signal array.
        fs: Sampling frequency in Hz.
        hr_bpm: Heart rate in BPM.
        bandwidth_hz: Bandwidth around HR frequency in Hz.
        
    Returns:
        SNR in dB.
    """
    from .heart_rate_estimation import compute_power_spectrum
    
    freqs, psd = compute_power_spectrum(rppg_signal, fs)
    
    if len(freqs) == 0:
        return -np.inf
    
    hr_freq = hr_bpm / 60.0
    
    # Define signal band (HR ± bandwidth)
    signal_mask = (freqs >= hr_freq - bandwidth_hz) & (freqs <= hr_freq + bandwidth_hz)
    signal_power = np.sum(psd[signal_mask])
    
    # Define noise bands (exclude signal band)
    noise_mask = ~signal_mask
    noise_power = np.sum(psd[noise_mask])
    
    if noise_power == 0 or signal_power == 0:
        return -np.inf
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


def compute_mean_absolute_percentage_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Compute Mean Absolute Percentage Error.
    
    Args:
        predicted: Predicted heart rate values in BPM.
        ground_truth: Ground truth heart rate values in BPM.
        epsilon: Small value to avoid division by zero.
        
    Returns:
        MAPE as a percentage.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(predicted) & ~np.isnan(ground_truth)
    if not np.any(valid_mask):
        return np.nan
    
    pred_valid = predicted[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    return float(np.mean(np.abs((pred_valid - gt_valid) / (gt_valid + epsilon))) * 100)


def compute_all_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    rppg_signal: np.ndarray | None = None,
    fs: float | None = None,
    bandwidth_hz: float = 0.2,
) -> dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        predicted: Predicted heart rate values in BPM.
        ground_truth: Ground truth heart rate values in BPM.
        rppg_signal: Optional rPPG signal for SNR computation.
        fs: Optional sampling frequency for SNR computation.
        bandwidth_hz: Bandwidth for SNR computation.
        
    Returns:
        Dictionary with metrics: mae, rmse, pearson_r, mape, snr (if provided).
    """
    metrics = {
        'mae': compute_mae(predicted, ground_truth),
        'rmse': compute_rmse(predicted, ground_truth),
        'pearson_r': compute_pearson_correlation(predicted, ground_truth),
        'mape': compute_mean_absolute_percentage_error(predicted, ground_truth),
    }
    
    # Add SNR if rPPG signal and sampling rate are provided
    if rppg_signal is not None and fs is not None:
        # Use mean predicted HR for SNR computation
        mean_hr = np.nanmean(predicted)
        if not np.isnan(mean_hr):
            metrics['snr'] = compute_snr(rppg_signal, fs, mean_hr, bandwidth_hz)
    
    return metrics


def compute_metrics_per_window(
    predicted_hr: np.ndarray,
    ground_truth_hr: np.ndarray,
    window_size: int = 30,
) -> dict[str, Any]:
    """Compute metrics for each window.
    
    Args:
        predicted_hr: Predicted heart rate values in BPM.
        ground_truth_hr: Ground truth heart rate values in BPM.
        window_size: Number of samples per window.
        
    Returns:
        Dictionary with per-window metrics and overall metrics.
    """
    n_samples = len(predicted_hr)
    n_windows = n_samples // window_size
    
    if n_windows == 0:
        return {
            'per_window_mae': [],
            'per_window_rmse': [],
            'per_window_pearson': [],
            'overall_mae': compute_mae(predicted_hr, ground_truth_hr),
            'overall_rmse': compute_rmse(predicted_hr, ground_truth_hr),
            'overall_pearson': compute_pearson_correlation(predicted_hr, ground_truth_hr),
        }
    
    per_window_mae = []
    per_window_rmse = []
    per_window_pearson = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        
        pred_window = predicted_hr[start:end]
        gt_window = ground_truth_hr[start:end]
        
        per_window_mae.append(compute_mae(pred_window, gt_window))
        per_window_rmse.append(compute_rmse(pred_window, gt_window))
        per_window_pearson.append(compute_pearson_correlation(pred_window, gt_window))
    
    return {
        'per_window_mae': per_window_mae,
        'per_window_rmse': per_window_rmse,
        'per_window_pearson': per_window_pearson,
        'mean_mae': np.mean(per_window_mae),
        'std_mae': np.std(per_window_mae),
        'mean_rmse': np.mean(per_window_rmse),
        'std_rmse': np.std(per_window_rmse),
        'mean_pearson': np.mean(per_window_pearson),
        'std_pearson': np.std(per_window_pearson),
        'overall_mae': compute_mae(predicted_hr, ground_truth_hr),
        'overall_rmse': compute_rmse(predicted_hr, ground_truth_hr),
        'overall_pearson': compute_pearson_correlation(predicted_hr, ground_truth_hr),
    }
