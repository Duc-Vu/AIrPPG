"""Green, CHROM, and POS rPPG baselines."""

from __future__ import annotations

import numpy as np

from airppg.schemas import METHOD_NAMES
from airppg.signal import fill_nan_1d, preprocess_trace, zscore


def combine_roi_traces(per_roi: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        combined = np.nanmean(np.asarray(per_roi, dtype=np.float64), axis=1)
    return zscore(combined)


def green_baseline(mean_rgb: np.ndarray, fps: float, hr_min_bpm: float = 42.0, hr_max_bpm: float = 210.0) -> tuple[np.ndarray, np.ndarray]:
    per_roi = np.stack(
        [preprocess_trace(mean_rgb[:, roi_idx, 1], fps, hr_min_bpm, hr_max_bpm) for roi_idx in range(mean_rgb.shape[1])],
        axis=1,
    )
    return combine_roi_traces(per_roi), per_roi


def chrom_single(rgb: np.ndarray, fps: float, hr_min_bpm: float = 42.0, hr_max_bpm: float = 210.0) -> np.ndarray:
    rgb_clean = np.column_stack([fill_nan_1d(rgb[:, channel]) for channel in range(3)])
    n_frames = rgb_clean.shape[0]
    window_len = max(3, int(round(1.6 * fps)))
    if n_frames < window_len:
        return preprocess_trace(np.zeros(n_frames), fps, hr_min_bpm, hr_max_bpm)
    signal_acc = np.zeros(n_frames, dtype=np.float64)
    weight_acc = np.zeros(n_frames, dtype=np.float64)
    window = np.hanning(window_len)
    for start in range(n_frames - window_len + 1):
        segment = rgb_clean[start : start + window_len]
        mean_rgb_window = np.mean(segment, axis=0)
        normalized = segment / np.where(mean_rgb_window == 0, 1.0, mean_rgb_window)
        xs = 3.0 * normalized[:, 0] - 2.0 * normalized[:, 1]
        ys = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
        std_y = np.std(ys)
        alpha = 0.0 if std_y <= 0 else np.std(xs) / std_y
        signal_acc[start : start + window_len] += (xs - alpha * ys) * window
        weight_acc[start : start + window_len] += window
    raw = signal_acc / np.where(weight_acc == 0, 1.0, weight_acc)
    return preprocess_trace(raw, fps, hr_min_bpm, hr_max_bpm)


def pos_single(rgb: np.ndarray, fps: float, hr_min_bpm: float = 42.0, hr_max_bpm: float = 210.0) -> np.ndarray:
    rgb_clean = np.column_stack([fill_nan_1d(rgb[:, channel]) for channel in range(3)])
    n_frames = rgb_clean.shape[0]
    window_len = max(3, int(round(1.6 * fps)))
    if n_frames < window_len:
        return preprocess_trace(np.zeros(n_frames), fps, hr_min_bpm, hr_max_bpm)
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)
    signal_acc = np.zeros(n_frames, dtype=np.float64)
    weight_acc = np.zeros(n_frames, dtype=np.float64)
    window = np.hanning(window_len)
    for start in range(n_frames - window_len + 1):
        segment = rgb_clean[start : start + window_len]
        mean_rgb_window = np.mean(segment, axis=0)
        normalized = (segment / np.where(mean_rgb_window == 0, 1.0, mean_rgb_window)).T
        projected = projection @ normalized
        std_1 = np.std(projected[1])
        alpha = 0.0 if std_1 <= 0 else np.std(projected[0]) / std_1
        pulse = projected[0] + alpha * projected[1]
        signal_acc[start : start + window_len] += pulse * window
        weight_acc[start : start + window_len] += window
    raw = signal_acc / np.where(weight_acc == 0, 1.0, weight_acc)
    return preprocess_trace(raw, fps, hr_min_bpm, hr_max_bpm)


def method_per_roi(method: str, mean_rgb: np.ndarray, fps: float, hr_min_bpm: float = 42.0, hr_max_bpm: float = 210.0) -> tuple[np.ndarray, np.ndarray]:
    if method == "green":
        return green_baseline(mean_rgb, fps, hr_min_bpm, hr_max_bpm)
    worker = chrom_single if method == "chrom" else pos_single
    per_roi = np.stack(
        [worker(mean_rgb[:, roi_idx, :], fps, hr_min_bpm, hr_max_bpm) for roi_idx in range(mean_rgb.shape[1])],
        axis=1,
    )
    return combine_roi_traces(per_roi), per_roi


def run_baselines(mean_rgb: np.ndarray, fps: float, hr_min_bpm: float = 42.0, hr_max_bpm: float = 210.0) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for method in METHOD_NAMES:
        signal, per_roi = method_per_roi(method, mean_rgb, fps, hr_min_bpm, hr_max_bpm)
        results[method] = {"signal": signal, "per_roi": per_roi}
    return results
