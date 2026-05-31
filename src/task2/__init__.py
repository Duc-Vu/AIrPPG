"""Task 2: Signal Extraction and Baseline Methods for rPPG"""

from .signal_extraction import extract_rgb_signals_from_video
from .signal_preprocessing import normalize_signal, detrend_signal, bandpass_filter
from .baseline_methods import green_channel_method, chrom_method, pos_method
from .heart_rate_estimation import estimate_heart_rate_from_spectrum
from .ground_truth_loader import load_ground_truth_ubfc
from .evaluation_metrics import compute_mae, compute_rmse, compute_pearson_correlation, compute_snr

__all__ = [
    "extract_rgb_signals_from_video",
    "normalize_signal",
    "detrend_signal",
    "bandpass_filter",
    "green_channel_method",
    "chrom_method",
    "pos_method",
    "estimate_heart_rate_from_spectrum",
    "load_ground_truth_ubfc",
    "compute_mae",
    "compute_rmse",
    "compute_pearson_correlation",
    "compute_snr",
]
