"""Metric helpers for rPPG heart-rate evaluation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from airppg.signal import estimate_snr_db


def align_ground_truth(gt: dict[str, np.ndarray], target_timestamps: np.ndarray) -> np.ndarray:
    target = np.asarray(target_timestamps, dtype=np.float64)
    gt_ts = np.asarray(gt["timestamps"], dtype=np.float64)
    gt_hr = np.asarray(gt["heart_rate_bpm"], dtype=np.float64)
    finite = np.isfinite(gt_ts) & np.isfinite(gt_hr)
    if finite.sum() == 0:
        return np.full(target.shape, np.nan, dtype=np.float64)
    if finite.sum() == 1:
        return np.full(target.shape, gt_hr[finite][0], dtype=np.float64)
    order = np.argsort(gt_ts[finite])
    interpolator = interp1d(
        gt_ts[finite][order],
        gt_hr[finite][order],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interpolator(target)


def evaluate_method(
    hr_values: np.ndarray,
    hr_timestamps: np.ndarray,
    gt: dict[str, np.ndarray],
    rppg_signal: np.ndarray,
    fps: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
) -> dict[str, float | int]:
    gt_aligned = align_ground_truth(gt, hr_timestamps)
    finite = np.isfinite(hr_values) & np.isfinite(gt_aligned)
    if finite.sum() < 1:
        return {
            "mae_bpm": np.nan,
            "rmse_bpm": np.nan,
            "pearson_r": np.nan,
            "bias_bpm": np.nan,
            "snr_db": np.nan,
            "n_windows": 0,
        }
    pred = hr_values[finite].astype(np.float64)
    ref = gt_aligned[finite].astype(np.float64)
    corr = pearsonr(pred, ref)[0] if finite.sum() >= 2 and np.nanstd(pred) > 0 and np.nanstd(ref) > 0 else np.nan
    return {
        "mae_bpm": float(np.mean(np.abs(pred - ref))),
        "rmse_bpm": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "pearson_r": float(corr) if np.isfinite(corr) else np.nan,
        "bias_bpm": float(np.mean(pred - ref)),
        "snr_db": estimate_snr_db(rppg_signal, fps, float(np.nanmean(ref)), hr_min_bpm, hr_max_bpm),
        "n_windows": int(finite.sum()),
    }
