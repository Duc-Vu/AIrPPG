"""ROI quality features and weights for fusion."""

from __future__ import annotations

import numpy as np

from airppg.signal import fill_nan_1d

FEATURE_NAMES = (
    "valid_ratio",
    "signal_std",
    "dominant_power_ratio",
    "snr_like_db",
    "inter_roi_disagreement_bpm",
)


def estimate_fps(timestamps: np.ndarray) -> float:
    t = np.asarray(timestamps, dtype=np.float64)
    diffs = np.diff(t[np.isfinite(t)])
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("Cannot estimate FPS from timestamps with no positive finite differences.")
    return float(1.0 / np.median(diffs))


def _hr_band_power(
    segment: np.ndarray, fps: float, hr_min_bpm: float, hr_max_bpm: float
) -> tuple[np.ndarray, np.ndarray]:
    x = fill_nan_1d(segment)
    if x.size < 3:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fps)
    band = (freqs >= hr_min_bpm / 60.0) & (freqs <= hr_max_bpm / 60.0)
    return freqs[band], power[band]


def compute_quality_features(
    rppg_roi_signals: np.ndarray,
    valid: np.ndarray,
    windows: list[tuple[int, int]],
    fps: float,
    single_roi_hr: np.ndarray,
    hr_min_bpm: float,
    hr_max_bpm: float,
) -> np.ndarray:
    signals = np.asarray(rppg_roi_signals, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)
    n_windows = len(windows)
    n_methods, _, n_rois = signals.shape
    features = np.full((n_windows, n_methods, n_rois, len(FEATURE_NAMES)), np.nan, dtype=np.float32)

    for w_idx, (start, end) in enumerate(windows):
        valid_ratio = float(np.mean(valid_mask[start:end])) if end > start else 0.0
        for m_idx in range(n_methods):
            finite_roi_hr = single_roi_hr[w_idx, m_idx]
            finite_roi_hr = finite_roi_hr[np.isfinite(finite_roi_hr)]
            median_roi_hr = float(np.median(finite_roi_hr)) if finite_roi_hr.size else np.nan
            for r_idx in range(n_rois):
                segment = signals[m_idx, start:end, r_idx]
                features[w_idx, m_idx, r_idx, 0] = valid_ratio
                features[w_idx, m_idx, r_idx, 1] = np.nanstd(segment)
                freqs, power = _hr_band_power(segment, fps, hr_min_bpm, hr_max_bpm)
                if power.size:
                    peak_idx = int(np.argmax(power))
                    median_power = float(np.median(power))
                    features[w_idx, m_idx, r_idx, 2] = float(
                        power[peak_idx] / (median_power + 1e-12)
                    )
                    peak_hz = float(freqs[peak_idx])
                    near_peak = (freqs >= peak_hz - 0.1) & (freqs <= peak_hz + 0.1)
                    signal_power = float(power[near_peak].sum())
                    noise_power = float(power[~near_peak].sum())
                    features[w_idx, m_idx, r_idx, 3] = (
                        10.0 * np.log10(signal_power / (noise_power + 1e-12))
                        if signal_power > 0
                        else np.nan
                    )
                roi_hr = float(single_roi_hr[w_idx, m_idx, r_idx])
                features[w_idx, m_idx, r_idx, 4] = (
                    abs(roi_hr - median_roi_hr)
                    if np.isfinite(roi_hr) and np.isfinite(median_roi_hr)
                    else np.nan
                )
    return features


def _minmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    lo = float(np.min(x[finite]))
    hi = float(np.max(x[finite]))
    if hi > lo:
        out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def compute_roi_weights(features: np.ndarray) -> np.ndarray:
    f = np.asarray(features, dtype=np.float64)
    valid_ratio = np.clip(f[..., 0], 0.0, 1.0)
    snr_norm = np.apply_along_axis(_minmax, 2, f[..., 3])
    clarity_norm = np.apply_along_axis(_minmax, 2, f[..., 2])
    std_norm = np.apply_along_axis(_minmax, 2, f[..., 1])
    disagreement_norm = np.apply_along_axis(_minmax, 2, f[..., 4])
    scores = (
        0.35 * snr_norm
        + 0.25 * clarity_norm
        + 0.20 * valid_ratio
        + 0.10 * std_norm
        - 0.10 * disagreement_norm
    )
    scores[(valid_ratio < 0.8) | ~np.isfinite(scores)] = 0.0
    scores = np.clip(scores, 0.0, None)
    weights = np.zeros_like(scores, dtype=np.float64)

    for index in np.ndindex(scores.shape[0], scores.shape[1]):
        row = scores[index]
        total = float(row.sum())
        if total > 0.0:
            weights[index] = row / total
            continue
        eligible = np.isfinite(valid_ratio[index]) & (valid_ratio[index] >= 0.8)
        if eligible.any():
            weights[index][eligible] = 1.0 / float(eligible.sum())
        else:
            weights[index] = 1.0 / float(row.size)
    return weights.astype(np.float32)
