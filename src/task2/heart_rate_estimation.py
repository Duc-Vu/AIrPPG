"""Heart rate estimation from rPPG signals using frequency analysis."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal


def estimate_heart_rate_from_spectrum(
    rppg_signal: np.ndarray,
    fs: float,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
    method: Literal["peak", "weighted"] = "peak",
) -> float:
    """Estimate heart rate from rPPG signal using frequency spectrum.
    
    Args:
        rppg_signal: rPPG signal array.
        fs: Sampling frequency in Hz.
        min_hr: Minimum heart rate in BPM.
        max_hr: Maximum heart rate in BPM.
        method: Method for HR estimation - 'peak' (dominant frequency) or 'weighted' (weighted average).
        
    Returns:
        Estimated heart rate in BPM.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(rppg_signal)
    if not np.any(valid_mask):
        return np.nan
    
    signal_clean = rppg_signal[valid_mask]
    
    # Compute power spectral density
    n = len(signal_clean)
    if n < 10:
        return np.nan
    
    # Use Welch's method for better spectral estimation
    freqs, psd = signal.welch(signal_clean, fs=fs, nperseg=min(256, n // 4))
    
    # Convert frequency range to HR range
    min_freq = min_hr / 60.0
    max_freq = max_hr / 60.0
    
    # Find indices in valid frequency range
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(freq_mask):
        return np.nan
    
    freqs_valid = freqs[freq_mask]
    psd_valid = psd[freq_mask]
    
    if len(psd_valid) == 0:
        return np.nan
    
    if method == "peak":
        # Find dominant frequency
        peak_idx = np.argmax(psd_valid)
        dominant_freq = freqs_valid[peak_idx]
        hr = dominant_freq * 60.0
    elif method == "weighted":
        # Weighted average by power
        weighted_freq = np.sum(freqs_valid * psd_valid) / np.sum(psd_valid)
        hr = weighted_freq * 60.0
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return hr


def estimate_heart_rate_windowed(
    rppg_signal: np.ndarray,
    fs: float,
    window_size_sec: float = 8.0,
    overlap_sec: float = 1.0,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate heart rate in sliding windows.
    
    Args:
        rppg_signal: rPPG signal array.
        fs: Sampling frequency in Hz.
        window_size_sec: Window size in seconds.
        overlap_sec: Overlap between windows in seconds.
        min_hr: Minimum heart rate in BPM.
        max_hr: Maximum heart rate in BPM.
        
    Returns:
        Tuple of (hr_values, time_centers):
        - hr_values: Array of HR estimates in BPM for each window.
        - time_centers: Array of time centers for each window in seconds.
    """
    n_samples = len(rppg_signal)
    window_size = int(window_size_sec * fs)
    overlap = int(overlap_sec * fs)
    step = window_size - overlap
    
    if window_size >= n_samples:
        # Single window
        hr = estimate_heart_rate_from_spectrum(rppg_signal, fs, min_hr, max_hr)
        return np.array([hr]), np.array([window_size_sec * fs / 2 / fs])
    
    hr_values = []
    time_centers = []
    
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        window_signal = rppg_signal[start:end]
        
        hr = estimate_heart_rate_from_spectrum(window_signal, fs, min_hr, max_hr)
        hr_values.append(hr)
        
        time_center = (start + end) / 2 / fs
        time_centers.append(time_center)
    
    return np.array(hr_values), np.array(time_centers)


def compute_power_spectrum(
    rppg_signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum of rPPG signal.
    
    Args:
        rppg_signal: rPPG signal array.
        fs: Sampling frequency in Hz.
        
    Returns:
        Tuple of (frequencies, power):
        - frequencies: Frequency array in Hz.
        - power: Power spectral density.
    """
    valid_mask = ~np.isnan(rppg_signal)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    signal_clean = rppg_signal[valid_mask]
    n = len(signal_clean)
    
    if n < 10:
        return np.array([]), np.array([])
    
    # Use Welch's method
    freqs, psd = signal.welch(signal_clean, fs=fs, nperseg=min(256, n // 4))
    
    return freqs, psd


def find_snr(
    rppg_signal: np.ndarray,
    fs: float,
    hr_bpm: float,
    bandwidth_hz: float = 0.2,
) -> float:
    """Compute signal-to-noise ratio for HR estimation.
    
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
