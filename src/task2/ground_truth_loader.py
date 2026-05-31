"""Ground truth loading functions for UBFC dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_ground_truth_ubfc(
    ground_truth_path: Path,
    dataset_split: str = "DATASET_2",
) -> dict[str, Any]:
    """Load ground truth heart rate data from UBFC dataset.
    
    UBFC dataset has different formats for DATASET_1 and DATASET_2:
    - DATASET_1: Uses .xmp files with pulse wave data
    - DATASET_2: Uses .txt files with time-stamped heart rate values
    
    Args:
        ground_truth_path: Path to ground truth file (.xmp or .txt).
        dataset_split: Dataset split - 'DATASET_1' or 'DATASET_2'.
        
    Returns:
        Dictionary with ground truth data:
        - 'times': Time array in seconds.
        - 'hr_values': Heart rate values in BPM.
        - 'pulse_signal': Optional pulse wave signal (for DATASET_1).
    """
    if dataset_split == "DATASET_1":
        return _load_ground_truth_dataset1(ground_truth_path)
    elif dataset_split == "DATASET_2":
        return _load_ground_truth_dataset2(ground_truth_path)
    else:
        raise ValueError(f"Unknown dataset split: {dataset_split}")


def _load_ground_truth_dataset1(xmp_path: Path) -> dict[str, Any]:
    """Load ground truth from DATASET_1 .xmp file.
    
    DATASET_1 contains pulse wave data sampled at specific rate.
    """
    # DATASET_1 .xmp files contain pulse wave data
    # Format is typically: timestamp, pulse_value
    # This is a simplified loader - actual format may vary
    
    try:
        data = np.loadtxt(xmp_path, delimiter=',')
        if data.ndim == 1:
            # Single column - assume it's pulse values
            pulse_signal = data
            times = np.arange(len(pulse_signal)) / 30.0  # Assume 30 Hz sampling
            hr_values = None
        elif data.shape[1] >= 2:
            # Two columns: timestamp, pulse_value
            times = data[:, 0]
            pulse_signal = data[:, 1]
            # Estimate HR from pulse signal
            hr_values = _estimate_hr_from_pulse(pulse_signal, times)
        else:
            raise ValueError(f"Unexpected data shape in {xmp_path}")
        
        return {
            'times': times,
            'hr_values': hr_values,
            'pulse_signal': pulse_signal,
        }
    except Exception as e:
        raise ValueError(f"Failed to load DATASET_1 ground truth from {xmp_path}: {e}")


def _load_ground_truth_dataset2(txt_path: Path) -> dict[str, Any]:
    """Load ground truth from DATASET_2 .txt file.
    
    DATASET_2 contains time-stamped heart rate values.
    Format: time_seconds, heart_rate_bpm
    """
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1:
            # Single value - constant HR
            hr_values = np.array([data[0]])
            times = np.array([0.0])
        elif data.shape[1] >= 2:
            times = data[:, 0]
            hr_values = data[:, 1]
        else:
            raise ValueError(f"Unexpected data shape in {txt_path}")
        
        return {
            'times': times,
            'hr_values': hr_values,
            'pulse_signal': None,
        }
    except Exception as e:
        raise ValueError(f"Failed to load DATASET_2 ground truth from {txt_path}: {e}")


def _estimate_hr_from_pulse(pulse_signal: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Estimate heart rate from pulse signal using peak detection.
    
    Args:
        pulse_signal: Pulse wave signal.
        times: Time array in seconds.
        
    Returns:
        Heart rate values in BPM (single value or array).
    """
    from scipy import signal
    
    # Find peaks
    peaks, _ = signal.find_peaks(pulse_signal, distance=int(0.5 * (times[1] - times[0]) * len(times)))
    
    if len(peaks) < 2:
        return np.array([np.nan])
    
    # Compute inter-beat intervals
    peak_times = times[peaks]
    ibi = np.diff(peak_times)
    
    # Convert to BPM
    hr_values = 60.0 / ibi
    
    # Return mean HR
    return np.array([np.mean(hr_values)])


def align_ground_truth_with_signal(
    ground_truth: dict[str, Any],
    signal_times: np.ndarray,
    method: str = "nearest",
) -> np.ndarray:
    """Align ground truth HR values with signal time points.
    
    Args:
        ground_truth: Dictionary with 'times' and 'hr_values'.
        signal_times: Time points for the rPPG signal.
        method: Interpolation method - 'nearest', 'linear', or 'constant'.
        
    Returns:
        Aligned HR values at signal time points.
    """
    gt_times = ground_truth['times']
    gt_hr = ground_truth['hr_values']
    
    if gt_hr is None or len(gt_hr) == 0:
        return np.full(len(signal_times), np.nan)
    
    if method == "nearest":
        # Find nearest ground truth time for each signal time
        aligned_hr = np.zeros(len(signal_times))
        for i, t in enumerate(signal_times):
            idx = np.argmin(np.abs(gt_times - t))
            aligned_hr[i] = gt_hr[idx]
        return aligned_hr
    elif method == "linear":
        # Linear interpolation
        from scipy.interpolate import interp1d
        f = interp1d(gt_times, gt_hr, kind='linear', bounds_error=False, fill_value='extrapolate')
        return f(signal_times)
    elif method == "constant":
        # Use constant value (mean)
        mean_hr = np.nanmean(gt_hr)
        return np.full(len(signal_times), mean_hr)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def find_ground_truth_file(video_path: Path, dataset_root: Path) -> Path | None:
    """Find ground truth file for a given video.
    
    Args:
        video_path: Path to video file.
        dataset_root: Root directory of dataset.
        
    Returns:
        Path to ground truth file, or None if not found.
    """
    video_dir = video_path.parent
    
    # Try common ground truth file names
    gt_candidates = [
        video_dir / "ground_truth.txt",
        video_dir / "gtdump.xmp",
        video_dir / "gt.txt",
        video_dir / "hr.txt",
    ]
    
    for candidate in gt_candidates:
        if candidate.exists():
            return candidate
    
    return None


def get_dataset_split_from_path(video_path: Path, dataset_root: Path) -> str:
    """Determine dataset split from video path.
    
    Args:
        video_path: Path to video file.
        dataset_root: Root directory of dataset.
        
    Returns:
        Dataset split name ('DATASET_1' or 'DATASET_2').
    """
    try:
        relative = video_path.relative_to(dataset_root)
        parts = relative.parts
        if parts[0].startswith("DATASET_"):
            return parts[0]
    except ValueError:
        pass
    
    # Default to DATASET_2
    return "DATASET_2"
