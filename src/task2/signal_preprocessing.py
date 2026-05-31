"""Signal preprocessing functions for rPPG."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def normalize_signal(signal: np.ndarray, method: Literal["zscore", "minmax"] = "zscore") -> np.ndarray:
    """Normalize a signal.
    
    Args:
        signal: Input signal array.
        method: Normalization method - "zscore" (zero mean, unit variance) or "minmax" (scale to [0, 1]).
        
    Returns:
        Normalized signal.
    """
    if method == "zscore":
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean
    elif method == "minmax":
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val > min_val:
            return (signal - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(signal)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def detrend_signal(signal: np.ndarray, method: Literal["linear", "constant"] = "linear") -> np.ndarray:
    """Remove trend from signal.
    
    Args:
        signal: Input signal array.
        method: Detrending method - "linear" (remove linear trend) or "constant" (remove mean).
        
    Returns:
        Detrended signal.
    """
    if method == "linear":
        return signal.detrend(signal, type='linear')
    elif method == "constant":
        return signal - np.mean(signal)
    else:
        raise ValueError(f"Unknown detrending method: {method}")


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.7,
    highcut: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """Apply bandpass filter to signal.
    
    Typical heart rate range is 0.7-4.0 Hz (42-240 BPM).
    
    Args:
        signal: Input signal array.
        fs: Sampling frequency in Hz.
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        order: Filter order.
        
    Returns:
        Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if high >= 1.0:
        high = 0.99
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal)
    
    return filtered


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    normalize: bool = True,
    detrend: bool = True,
    bandpass: bool = True,
    lowcut: float = 0.7,
    highcut: float = 4.0,
) -> np.ndarray:
    """Apply full preprocessing pipeline to signal.
    
    Args:
        signal: Input signal array.
        fs: Sampling frequency in Hz.
        normalize: Whether to normalize the signal.
        detrend: Whether to detrend the signal.
        bandpass: Whether to apply bandpass filter.
        lowcut: Low cutoff frequency for bandpass filter in Hz.
        highcut: High cutoff frequency for bandpass filter in Hz.
        
    Returns:
        Preprocessed signal.
    """
    processed = signal.copy()
    
    if detrend:
        processed = detrend_signal(processed, method="linear")
    
    if bandpass:
        processed = bandpass_filter(processed, fs, lowcut=lowcut, highcut=highcut)
    
    if normalize:
        processed = normalize_signal(processed, method="zscore")
    
    return processed


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing to signal.
    
    Args:
        signal: Input signal array.
        window_size: Size of the moving average window.
        
    Returns:
        Smoothed signal.
    """
    if window_size >= len(signal):
        return np.full_like(signal, np.mean(signal))
    
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')


def interpolate_signal(
    signal: np.ndarray,
    original_indices: np.ndarray,
    target_indices: np.ndarray,
    kind: str = 'linear',
) -> np.ndarray:
    """Interpolate signal from original indices to target indices.
    
    Useful for aligning signals with different sampling rates or filling gaps.
    
    Args:
        signal: Input signal array.
        original_indices: Original time indices.
        target_indices: Target time indices.
        kind: Interpolation method ('linear', 'cubic', etc.).
        
    Returns:
        Interpolated signal at target indices.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(signal)
    if not np.any(valid_mask):
        return np.full(len(target_indices), np.nan)
    
    valid_signal = signal[valid_mask]
    valid_indices = original_indices[valid_mask]
    
    # Create interpolation function
    f = interp1d(valid_indices, valid_signal, kind=kind, bounds_error=False, fill_value='extrapolate')
    
    # Interpolate to target indices
    return f(target_indices)
