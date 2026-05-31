"""Baseline rPPG methods: Green channel, CHROM, and POS."""

from __future__ import annotations

import numpy as np


def green_channel_method(rgb_signals: np.ndarray) -> np.ndarray:
    """Green channel baseline method.
    
    Simply uses the green channel as the rPPG signal.
    
    Args:
        rgb_signals: Array of shape (N, 3) with RGB values over time.
                    Or shape (N, 3, 3) for multi-ROI (N frames, 3 ROIs, 3 channels).
        
    Returns:
        rPPG signal from green channel. Shape depends on input:
        - (N,) if input is (N, 3)
        - (N, 3) if input is (N, 3, 3) - one signal per ROI
    """
    if rgb_signals.ndim == 2:
        # Single ROI: shape (N, 3) -> return green channel (index 1)
        return rgb_signals[:, 1]
    elif rgb_signals.ndim == 3:
        # Multi-ROI: shape (N, 3, 3) -> return green channel for each ROI
        return rgb_signals[:, :, 1]
    else:
        raise ValueError(f"Unexpected input shape: {rgb_signals.shape}")


def chrom_method(rgb_signals: np.ndarray, fs: float) -> np.ndarray:
    """CHROM (CHRominance) method for rPPG.
    
    Based on: "Improved Pulse Rate Detection from Motion Corrupted Data
    Using Adaptive Chrominance Features" by de Haan and Jeanne.
    
    Args:
        rgb_signals: Array of shape (N, 3) with RGB values over time.
                    Or shape (N, 3, 3) for multi-ROI.
        fs: Sampling frequency in Hz.
        
    Returns:
        rPPG signal. Shape depends on input:
        - (N,) if input is (N, 3)
        - (N, 3) if input is (N, 3, 3) - one signal per ROI
    """
    if rgb_signals.ndim == 2:
        # Single ROI
        return _chrom_single(rgb_signals, fs)
    elif rgb_signals.ndim == 3:
        # Multi-ROI: apply CHROM to each ROI
        n_frames, n_rois, _ = rgb_signals.shape
        results = np.zeros((n_frames, n_rois))
        for roi_idx in range(n_rois):
            results[:, roi_idx] = _chrom_single(rgb_signals[:, roi_idx, :], fs)
        return results
    else:
        raise ValueError(f"Unexpected input shape: {rgb_signals.shape}")


def _chrom_single(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM method for a single ROI.
    
    Args:
        rgb: Array of shape (N, 3) with RGB values.
        fs: Sampling frequency in Hz.
        
    Returns:
        rPPG signal of shape (N,).
    """
    # Normalize RGB signals
    r_norm = rgb[:, 0] / np.mean(rgb[:, 0])
    g_norm = rgb[:, 1] / np.mean(rgb[:, 1])
    b_norm = rgb[:, 2] / np.mean(rgb[:, 2])
    
    # Compute chrominance signals
    xs = 3 * r_norm - 2 * g_norm
    ys = 1.5 * r_norm + g_norm - 1.5 * b_norm
    
    # Standard deviation for normalization
    sigma_x = np.std(xs)
    sigma_y = np.std(ys)
    
    if sigma_x == 0 or sigma_y == 0:
        return np.zeros_like(xs)
    
    # Normalize
    xs = xs / sigma_x
    ys = ys / sigma_y
    
    # Bandpass filter (0.7 - 4.0 Hz for heart rate)
    from scipy import signal
    nyquist = 0.5 * fs
    low = 0.7 / nyquist
    high = 4.0 / nyquist
    if high >= 1.0:
        high = 0.99
    b, a = signal.butter(4, [low, high], btype='band')
    xs_filtered = signal.filtfilt(b, a, xs)
    ys_filtered = signal.filtfilt(b, a, ys)
    
    # Combine signals
    s = xs_filtered - (sigma_x / sigma_y) * ys_filtered
    
    return s


def pos_method(rgb_signals: np.ndarray, fs: float, window_size: int = 32) -> np.ndarray:
    """POS (Plane-Orthogonal-to-Skin) method for rPPG.
    
    Based on: "Algorithmic Principles of Remote PPG" by de Haan and van Leest.
    
    Args:
        rgb_signals: Array of shape (N, 3) with RGB values over time.
                    Or shape (N, 3, 3) for multi-ROI.
        fs: Sampling frequency in Hz.
        window_size: Window size in frames for temporal filtering (default 32).
        
    Returns:
        rPPG signal. Shape depends on input:
        - (N,) if input is (N, 3)
        - (N, 3) if input is (N, 3, 3) - one signal per ROI
    """
    if rgb_signals.ndim == 2:
        # Single ROI
        return _pos_single(rgb_signals, fs, window_size)
    elif rgb_signals.ndim == 3:
        # Multi-ROI: apply POS to each ROI
        n_frames, n_rois, _ = rgb_signals.shape
        results = np.zeros((n_frames, n_rois))
        for roi_idx in range(n_rois):
            results[:, roi_idx] = _pos_single(rgb_signals[:, roi_idx, :], fs, window_size)
        return results
    else:
        raise ValueError(f"Unexpected input shape: {rgb_signals.shape}")


def _pos_single(rgb: np.ndarray, fs: float, window_size: int) -> np.ndarray:
    """POS method for a single ROI.
    
    Args:
        rgb: Array of shape (N, 3) with RGB values.
        fs: Sampling frequency in Hz.
        window_size: Window size in frames.
        
    Returns:
        rPPG signal of shape (N,).
    """
    n_frames = rgb.shape[0]
    
    # Normalize RGB
    c = rgb.T  # Shape (3, N)
    c_mean = np.mean(c, axis=1, keepdims=True)
    c_norm = c / c_mean
    
    # Initialize output
    s = np.zeros(n_frames)
    
    # Process in windows
    for t in range(window_size, n_frames):
        # Get window
        window = c_norm[:, t-window_size:t]
        
        # Compute temporal means
        mean_t = np.mean(window, axis=1)
        
        # Subtract mean
        window_centered = window - mean_t[:, np.newaxis]
        
        # Compute covariance
        cov = np.cov(window_centered)
        
        # Projection matrix (orthogonal to skin tone)
        # Skin tone direction is approximately (1, 1, 1)
        skin_dir = np.array([1, 1, 1]) / np.sqrt(3)
        
        # Project onto plane orthogonal to skin direction
        # Using SVD to find the plane
        try:
            u, s_vals, vh = np.linalg.svd(cov)
            # The first principal component is typically the motion direction
            # We want the component orthogonal to skin tone
            # Simple approach: project onto direction orthogonal to (1,1,1)
            projection = np.eye(3) - np.outer(skin_dir, skin_dir)
            projected = projection @ window_centered
            
            # Take the first component (dominant pulse direction)
            s[t] = projected[0, -1]  # Use last sample in window
        except np.linalg.LinAlgError:
            s[t] = 0
    
    # Bandpass filter
    from scipy import signal
    nyquist = 0.5 * fs
    low = 0.7 / nyquist
    high = 4.0 / nyquist
    if high >= 1.0:
        high = 0.99
    b, a = signal.butter(4, [low, high], btype='band')
    s_filtered = signal.filtfilt(b, a, s)
    
    return s_filtered


def apply_baseline_method(
    rgb_signals: np.ndarray,
    method: str,
    fs: float,
    **kwargs
) -> np.ndarray:
    """Apply a baseline method to RGB signals.
    
    Args:
        rgb_signals: Array of shape (N, 3) or (N, 3, 3) with RGB values.
        method: Baseline method name - 'green', 'chrom', or 'pos'.
        fs: Sampling frequency in Hz.
        **kwargs: Additional arguments for the method.
        
    Returns:
        rPPG signal.
    """
    method = method.lower()
    
    if method == 'green':
        return green_channel_method(rgb_signals)
    elif method == 'chrom':
        return chrom_method(rgb_signals, fs, **kwargs)
    elif method == 'pos':
        return pos_method(rgb_signals, fs, **kwargs)
    else:
        raise ValueError(f"Unknown baseline method: {method}. Choose from 'green', 'chrom', 'pos'.")
