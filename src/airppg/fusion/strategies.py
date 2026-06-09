"""Fusion strategy helpers."""

from __future__ import annotations

import numpy as np

from airppg.signal import estimate_hr_trace, zscore


def make_windows(
    n_frames: int, fps: float, window_sec: float, step_sec: float
) -> list[tuple[int, int]]:
    if n_frames <= 0:
        return []
    window_size = min(max(3, int(round(window_sec * fps))), n_frames)
    step_size = max(1, int(round(step_sec * fps)))
    return [
        (start, start + window_size)
        for start in range(0, n_frames - window_size + 1, step_size)
    ]


def estimate_single_roi_hr(
    rppg_roi_signals: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
    window_sec: float,
    step_sec: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
) -> tuple[np.ndarray, np.ndarray]:
    signals = np.asarray(rppg_roi_signals, dtype=np.float64)
    n_methods, _, n_rois = signals.shape
    hr_values: np.ndarray | None = None
    centers: np.ndarray | None = None
    for m_idx in range(n_methods):
        for r_idx in range(n_rois):
            roi_hr, roi_centers = estimate_hr_trace(
                signals[m_idx, :, r_idx],
                timestamps,
                fps,
                window_sec,
                step_sec,
                hr_min_bpm,
                hr_max_bpm,
            )
            if hr_values is None:
                hr_values = np.full((roi_hr.size, n_methods, n_rois), np.nan, dtype=np.float32)
                centers = roi_centers
            elif centers is not None and not np.allclose(
                roi_centers, centers, atol=1.0 / fps, rtol=0.0
            ):
                raise ValueError("Single-ROI HR centers do not match across methods/ROIs.")
            hr_values[:, m_idx, r_idx] = roi_hr
    if hr_values is None or centers is None:
        return np.empty((0, n_methods, n_rois), dtype=np.float32), np.array([], dtype=np.float32)
    return hr_values, centers.astype(np.float32)


def average_fuse_signals(rppg_roi_signals: np.ndarray) -> np.ndarray:
    signals = np.asarray(rppg_roi_signals, dtype=np.float64)
    fused = np.nanmean(signals, axis=2).T
    for m_idx in range(fused.shape[1]):
        fused[:, m_idx] = zscore(fused[:, m_idx])
    return fused.astype(np.float32)


def estimate_fused_hr(
    fused_signals: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
    window_sec: float,
    step_sec: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
) -> tuple[np.ndarray, np.ndarray]:
    signals = np.asarray(fused_signals, dtype=np.float64)
    hr_values: np.ndarray | None = None
    centers: np.ndarray | None = None
    for m_idx in range(signals.shape[1]):
        method_hr, method_centers = estimate_hr_trace(
            signals[:, m_idx], timestamps, fps, window_sec, step_sec, hr_min_bpm, hr_max_bpm
        )
        if hr_values is None:
            hr_values = np.full((method_hr.size, signals.shape[1]), np.nan, dtype=np.float32)
            centers = method_centers
        elif centers is not None and not np.allclose(
            method_centers, centers, atol=1.0 / fps, rtol=0.0
        ):
            raise ValueError("Fused HR centers do not match across methods.")
        hr_values[:, m_idx] = method_hr
    if hr_values is None or centers is None:
        return np.empty((0, signals.shape[1]), dtype=np.float32), np.array([], dtype=np.float32)
    return hr_values, centers.astype(np.float32)


def quality_weighted_fuse_signals(
    rppg_roi_signals: np.ndarray,
    roi_weights: np.ndarray,
    windows: list[tuple[int, int]],
    average_fused_signals: np.ndarray,
) -> np.ndarray:
    signals = np.asarray(rppg_roi_signals, dtype=np.float64)
    weights = np.asarray(roi_weights, dtype=np.float64)
    n_methods, n_frames, _ = signals.shape
    accum = np.zeros((n_frames, n_methods), dtype=np.float64)
    coverage = np.zeros((n_frames, n_methods), dtype=np.float64)
    for w_idx, (start, end) in enumerate(windows):
        for m_idx in range(n_methods):
            segment = signals[m_idx, start:end]
            weighted = np.nansum(segment * weights[w_idx, m_idx][None, :], axis=1)
            accum[start:end, m_idx] += weighted
            coverage[start:end, m_idx] += 1.0
    fused = np.asarray(average_fused_signals, dtype=np.float64).copy()
    covered = coverage > 0
    fused[covered] = accum[covered] / coverage[covered]
    for m_idx in range(fused.shape[1]):
        fused[:, m_idx] = zscore(fused[:, m_idx])
    return fused.astype(np.float32)
