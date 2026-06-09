"""Signal preprocessing and spectral heart-rate helpers."""

from __future__ import annotations

import numpy as np
import scipy.signal as sp_signal


def fill_nan_1d(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x


def zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, np.nan, dtype=np.float64)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std <= 0 or not np.isfinite(std):
        return x - mean
    return (x - mean) / std


def preprocess_trace(values: np.ndarray, fps: float, hr_min_bpm: float, hr_max_bpm: float) -> np.ndarray:
    x = fill_nan_1d(values)
    if x.size < 3:
        return zscore(x)
    x = sp_signal.detrend(x, type="linear")
    nyquist = 0.5 * float(fps)
    if nyquist > 0:
        low = max(0.001, (hr_min_bpm / 60.0) / nyquist)
        high = min(0.999, (hr_max_bpm / 60.0) / nyquist)
        if low < high:
            b, a = sp_signal.butter(4, [low, high], btype="bandpass")
            padlen = 3 * max(len(a), len(b))
            if x.size > padlen:
                x = sp_signal.filtfilt(b, a, x)
    return zscore(x)


def estimate_hr_trace(
    rppg_signal: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
    window_sec: float,
    step_sec: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = fill_nan_1d(rppg_signal)
    t = np.asarray(timestamps, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    window_size = min(max(3, int(round(window_sec * fps))), n)
    step_size = max(1, int(round(step_sec * fps)))
    min_freq = hr_min_bpm / 60.0
    max_freq = hr_max_bpm / 60.0
    window = np.hanning(window_size)
    hr_values: list[float] = []
    centers: list[float] = []
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        segment = x[start:end] - np.mean(x[start:end])
        freqs = np.fft.rfftfreq(window_size, 1.0 / fps)
        power = np.abs(np.fft.rfft(segment * window)) ** 2
        band = (freqs >= min_freq) & (freqs <= max_freq)
        hr_values.append(float(freqs[band][np.argmax(power[band])] * 60.0) if band.any() else np.nan)
        centers.append(float(np.nanmean(t[start:end])))
    return np.asarray(hr_values, dtype=np.float32), np.asarray(centers, dtype=np.float32)


def compute_spectrum(rppg_signal: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    x = fill_nan_1d(rppg_signal)
    if x.size < 3:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = sp_signal.detrend(x - np.mean(x), type="linear")
    freqs_bpm = np.fft.rfftfreq(x.size, 1.0 / fps) * 60.0
    magnitude = np.abs(np.fft.rfft(x))
    max_magnitude = magnitude.max() if magnitude.size else 0.0
    if max_magnitude > 0:
        magnitude = magnitude / max_magnitude
    return freqs_bpm, magnitude


def estimate_snr_db(
    rppg_signal: np.ndarray,
    fps: float,
    reference_hr_bpm: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
    band_width_hz: float = 0.1,
) -> float:
    if not np.isfinite(reference_hr_bpm):
        return float("nan")
    x = fill_nan_1d(rppg_signal)
    if x.size < 3:
        return float("nan")
    x = sp_signal.detrend(x - np.mean(x), type="linear")
    freqs = np.fft.rfftfreq(x.size, 1.0 / fps)
    power = np.abs(np.fft.rfft(x)) ** 2
    reference_hz = reference_hr_bpm / 60.0
    total = (freqs >= hr_min_bpm / 60.0) & (freqs <= hr_max_bpm / 60.0)
    fundamental = (freqs >= reference_hz - band_width_hz) & (freqs <= reference_hz + band_width_hz)
    harmonic = (freqs >= 2.0 * reference_hz - band_width_hz) & (freqs <= 2.0 * reference_hz + band_width_hz)
    signal_power = power[total & (fundamental | harmonic)].sum()
    noise_power = power[total & ~(fundamental | harmonic)].sum()
    if signal_power <= 0 or noise_power <= 0:
        return float("nan")
    return float(10.0 * np.log10(signal_power / noise_power))
