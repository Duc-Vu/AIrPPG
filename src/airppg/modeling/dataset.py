"""Task 4 dataset builder: read Task 2/3 artifacts → window-level feature dataset.

Design
------
Each *window* in hr_timestamps corresponds to a sliding window over the rPPG
signal.  Rather than feeding ~860 raw frames into a CNN (variable length, too
large), we compute the **FFT power spectrum inside the HR band** for that
window's signal segment.  This gives a fixed-size feature vector of
``n_fft_bins`` values regardless of FPS differences.

Window geometry (identical to Task 2/3):
    window_size = min(max(3, round(WINDOW_SEC * fps)), N)
    step_size   = max(1, round(STEP_SEC * fps))
    starts      = range(0, N - window_size + 1, step_size)

No data leakage: normalization is fit on train split only, then applied to
val and test.  Invalid windows are recorded with a reason, not silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from airppg.paths import load_json, resolve_relative
from airppg.modeling.config import ModelingConfig
from airppg.signal import fill_nan_1d

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
MIN_VALID_RATIO = 0.5   # window must have at least 50 % valid frames
MIN_SIGNAL_STD = 1e-8   # window signal must not be flat


# ── dataclass for one window ───────────────────────────────────────────────────
@dataclass
class WindowSample:
    sample_id: str
    split: str
    window_idx: int          # index into hr_timestamps
    hr_timestamp: float      # centre time (seconds)
    gt_hr: float             # ground-truth HR bpm
    features: np.ndarray     # shape (C, n_bins) float32, already normalised
    valid: bool
    invalid_reason: str      # empty string when valid


# ── helpers ────────────────────────────────────────────────────────────────────

def _estimate_fps(timestamps: np.ndarray) -> float:
    """Median FPS from frame timestamps."""
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size < 2:
        return 30.0
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 30.0
    return float(1.0 / np.median(diffs))


def _compute_spectrum_in_band(
    signal: np.ndarray,
    fps: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
    n_bins: int,
) -> np.ndarray:
    """FFT power spectrum of ``signal``, resampled to ``n_bins`` within HR band.

    Returns float32 array of shape (n_bins,).
    All-NaN input returns zeros.
    """
    x = fill_nan_1d(np.asarray(signal, dtype=np.float64))
    N = x.size
    if N < 4:
        return np.zeros(n_bins, dtype=np.float32)

    # zero-mean
    x = x - np.mean(x)

    # FFT
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)   # in Hz
    power = np.abs(np.fft.rfft(x)) ** 2

    # keep HR band
    min_hz = hr_min_bpm / 60.0
    max_hz = hr_max_bpm / 60.0
    band_mask = (freqs >= min_hz) & (freqs <= max_hz)
    band_power = power[band_mask]
    band_freqs = freqs[band_mask]

    if band_power.size == 0:
        return np.zeros(n_bins, dtype=np.float32)

    # resample / interpolate to fixed n_bins grid
    target_freqs = np.linspace(band_freqs[0], band_freqs[-1], n_bins)
    resampled = np.interp(target_freqs, band_freqs, band_power)

    # normalise to [0,1] per window (keeps shape info)
    peak = resampled.max()
    if peak > 0:
        resampled = resampled / peak
    return resampled.astype(np.float32)


def _extract_window_features(
    signals: dict[str, np.ndarray],
    use_fusion: bool,
    fps: float,
    window_start: int,
    window_size: int,
    cfg: ModelingConfig,
) -> tuple[np.ndarray, bool, str]:
    """Extract feature array for one window.

    Returns (features, valid, reason).
    features shape: (C, n_fft_bins) where C depends on input_strategy.
    """
    strategy = cfg.input_strategy
    n_bins = cfg.n_fft_bins
    hr_min = cfg.hr_min_bpm
    hr_max = cfg.hr_max_bpm
    sl = slice(window_start, window_start + window_size)

    # ── validity check ─────────────────────────────────────────────────────────
    if "valid" in signals:
        valid_arr = np.asarray(signals["valid"], dtype=bool)
        seg_valid = valid_arr[sl]
        valid_ratio = seg_valid.mean() if seg_valid.size > 0 else 0.0
        if valid_ratio < MIN_VALID_RATIO:
            return (
                np.zeros((1, n_bins), dtype=np.float32),
                False,
                f"valid_ratio={valid_ratio:.2f}<{MIN_VALID_RATIO}",
            )

    # ── choose signal source ───────────────────────────────────────────────────
    if strategy == "pos_spectrum":
        # rppg_signals shape (N, 3): columns = green, chrom, pos
        if "rppg_signals" not in signals:
            return np.zeros((1, n_bins), dtype=np.float32), False, "missing rppg_signals"
        rppg = np.asarray(signals["rppg_signals"], dtype=np.float32)
        seg = rppg[sl, 2]   # POS column
        if np.std(seg[np.isfinite(seg)]) < MIN_SIGNAL_STD:
            return np.zeros((1, n_bins), dtype=np.float32), False, "signal_std<threshold (pos)"
        spec = _compute_spectrum_in_band(seg, fps, hr_min, hr_max, n_bins)
        features = spec[np.newaxis, :]   # (1, n_bins)

    elif strategy == "fused_spectrum":
        # average_fused_rppg_signals shape (N, 3): columns = green, chrom, pos
        key = "average_fused_rppg_signals"
        if not use_fusion or key not in signals:
            # fallback to rppg_signals POS
            if "rppg_signals" not in signals:
                return np.zeros((1, n_bins), dtype=np.float32), False, "missing fused/rppg_signals"
            rppg = np.asarray(signals["rppg_signals"], dtype=np.float32)
            seg = rppg[sl, 2]
        else:
            fused = np.asarray(signals[key], dtype=np.float32)
            seg = fused[sl, 2]   # POS column of average-fused
        if np.std(seg[np.isfinite(seg)]) < MIN_SIGNAL_STD:
            return np.zeros((1, n_bins), dtype=np.float32), False, "signal_std<threshold (fused)"
        spec = _compute_spectrum_in_band(seg, fps, hr_min, hr_max, n_bins)
        features = spec[np.newaxis, :]

    elif strategy == "multi_channel_spectrum":
        # use average_fused_rppg_signals all 3 methods → (3, n_bins)
        key = "average_fused_rppg_signals"
        if use_fusion and key in signals:
            fused = np.asarray(signals[key], dtype=np.float32)
        elif "rppg_signals" in signals:
            fused = np.asarray(signals["rppg_signals"], dtype=np.float32)
        else:
            return np.zeros((3, n_bins), dtype=np.float32), False, "missing signals"
        specs = []
        for ch in range(min(3, fused.shape[1])):
            seg = fused[sl, ch]
            if np.std(seg[np.isfinite(seg)]) < MIN_SIGNAL_STD:
                specs.append(np.zeros(n_bins, dtype=np.float32))
            else:
                specs.append(_compute_spectrum_in_band(seg, fps, hr_min, hr_max, n_bins))
        while len(specs) < 3:
            specs.append(np.zeros(n_bins, dtype=np.float32))
        features = np.stack(specs, axis=0)   # (3, n_bins)

    else:
        raise ValueError(f"Unknown input_strategy: {strategy!r}")

    # check for NaN in features
    if not np.isfinite(features).all():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features.astype(np.float32), True, ""


# ── load one sample ────────────────────────────────────────────────────────────

def _load_sample_signals(
    item: dict[str, Any],
    fusion_manifest_root: Path,
    signal_extraction_manifest_root: Path,
) -> tuple[dict[str, np.ndarray], bool]:
    """Load and merge npz arrays from both signal extraction and fusion outputs.

    Loads signals.npz first (provides rppg_signals, rppg_roi_signals, etc.),
    then overlays with fusion_signals.npz (adds average_fused_rppg_signals,
    weighted_fused_rppg_signals, etc.).  This ensures all keys are available
    regardless of which input strategy is configured.

    Returns (arrays, use_fusion) where use_fusion=True when fusion npz was found.
    """
    arrays: dict[str, np.ndarray] = {}
    use_fusion = False

    # 1) Load signal extraction npz (rppg_signals, rppg_roi_signals, ...)
    sig_npz_rel = item.get("outputs_signal", {}).get("signals_npz")
    if sig_npz_rel:
        p = resolve_relative(sig_npz_rel, signal_extraction_manifest_root)
        if p.exists():
            with np.load(p, allow_pickle=False) as d:
                arrays.update({k: d[k] for k in d.files})

    # 2) Overlay fusion npz (adds fused signals; timestamps/valid from signals.npz kept)
    fusion_npz_rel = item.get("outputs_fusion", {}).get("fusion_npz")
    if fusion_npz_rel:
        p = resolve_relative(fusion_npz_rel, fusion_manifest_root)
        if p.exists():
            with np.load(p, allow_pickle=False) as d:
                for k in d.files:
                    # Keep signal extraction's timestamps / gt to avoid conflicts
                    if k in ("timestamps", "valid", "hr_timestamps", "gt_aligned_to_hr"):
                        if k not in arrays:
                            arrays[k] = d[k]
                    else:
                        arrays[k] = d[k]
            use_fusion = True

    if not arrays:
        raise FileNotFoundError(f"No npz found for sample {item['sample_id']!r}")

    return arrays, use_fusion




# ── build window list for one sample ──────────────────────────────────────────

def _build_windows_for_sample(
    item: dict[str, Any],
    signals: dict[str, np.ndarray],
    use_fusion: bool,
    cfg: ModelingConfig,
) -> list[WindowSample]:
    sample_id: str = item["sample_id"]
    split: str = item["split"]

    timestamps = np.asarray(signals["timestamps"], dtype=np.float64)
    N = timestamps.size
    fps = _estimate_fps(timestamps)

    # window geometry — identical to Task 2/3
    window_size = min(max(3, int(round(cfg.window_sec * fps))), N)
    step_size = max(1, int(round(cfg.step_sec * fps)))

    # hr_timestamps and gt_aligned_to_hr come pre-computed in the npz
    hr_timestamps = np.asarray(signals["hr_timestamps"], dtype=np.float32)
    gt_aligned = np.asarray(signals["gt_aligned_to_hr"], dtype=np.float32)

    starts = list(range(0, N - window_size + 1, step_size))
    # Align with stored hr_timestamps count
    n_windows = min(len(starts), len(hr_timestamps))

    windows: list[WindowSample] = []
    for w_idx in range(n_windows):
        start = starts[w_idx]
        gt_hr = float(gt_aligned[w_idx])

        # skip windows with NaN ground truth
        if not np.isfinite(gt_hr):
            windows.append(
                WindowSample(
                    sample_id=sample_id,
                    split=split,
                    window_idx=w_idx,
                    hr_timestamp=float(hr_timestamps[w_idx]),
                    gt_hr=gt_hr,
                    features=np.zeros((1, cfg.n_fft_bins), dtype=np.float32),
                    valid=False,
                    invalid_reason="gt_hr is NaN",
                )
            )
            continue

        features, valid, reason = _extract_window_features(
            signals, use_fusion, fps, start, window_size, cfg
        )
        windows.append(
            WindowSample(
                sample_id=sample_id,
                split=split,
                window_idx=w_idx,
                hr_timestamp=float(hr_timestamps[w_idx]),
                gt_hr=gt_hr,
                features=features,
                valid=valid,
                invalid_reason=reason,
            )
        )

    if len(windows) == 0:
        logger.warning("Sample %s produced 0 windows", sample_id)
    return windows


# ── public API ─────────────────────────────────────────────────────────────────

def load_all_windows(
    cfg: ModelingConfig,
    fusion_manifest: dict[str, Any],
    signal_extraction_manifest: dict[str, Any],
) -> dict[str, list[WindowSample]]:
    """Load windows for all processed samples, grouped by split.

    Returns dict with keys "train", "val", "test".
    """
    fusion_root = Path(cfg.fusion_manifest_path).parent
    sig_root = Path(cfg.signal_extraction_manifest_path).parent

    # Build lookup: sample_id → fusion outputs path
    fusion_items: dict[str, dict] = {
        item["sample_id"]: item
        for item in fusion_manifest.get("items", [])
        if item.get("status") == "processed"
    }
    # Build lookup: sample_id → signal extraction outputs path
    sig_items: dict[str, dict] = {
        item["sample_id"]: item
        for item in signal_extraction_manifest.get("items", [])
        if item.get("status") == "processed"
    }

    # Merge: signal_extraction is authoritative for split; fusion gives extra arrays
    all_sample_ids = sorted(set(sig_items) | set(fusion_items))
    if cfg.batch_limit is not None:
        all_sample_ids = all_sample_ids[: cfg.batch_limit]

    split_windows: dict[str, list[WindowSample]] = {"train": [], "val": [], "test": []}

    for sid in all_sample_ids:
        # Build a unified item dict
        sig_item = sig_items.get(sid)
        fusion_item = fusion_items.get(sid)

        if sig_item is None and fusion_item is None:
            continue

        base_item = sig_item or fusion_item
        split = str(base_item["split"])

        # Assemble unified outputs
        merged: dict[str, Any] = {
            "sample_id": sid,
            "split": split,
            "outputs_signal": (sig_item or {}).get("outputs", {}),
            "outputs_fusion": (fusion_item or {}).get("outputs", {}),
        }

        try:
            signals, use_fusion = _load_sample_signals(merged, fusion_root, sig_root)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", sid, exc)
            continue

        windows = _build_windows_for_sample(merged, signals, use_fusion, cfg)
        split_windows.setdefault(split, []).extend(windows)

    return split_windows


def fit_normalization(
    train_windows: list[WindowSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin mean and std from valid train windows only.

    Returns (mean, std) arrays of shape (C, n_bins) flattened to 1-D for storage.
    """
    valid = [w for w in train_windows if w.valid]
    if not valid:
        # fallback
        n = train_windows[0].features.size if train_windows else 1
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)

    stack = np.stack([w.features.ravel() for w in valid], axis=0)  # (N, C*bins)
    mean = stack.mean(axis=0).astype(np.float32)
    std = stack.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0   # avoid divide-by-zero for constant bins
    return mean, std


def apply_normalization(
    windows: list[WindowSample],
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """Normalize features in-place using provided mean/std."""
    shape = windows[0].features.shape if windows else None
    for w in windows:
        if w.valid and shape is not None:
            flat = w.features.ravel()
            norm_flat = (flat - mean) / std
            w.features = norm_flat.reshape(shape).astype(np.float32)


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """PyTorch Dataset wrapping a list of WindowSample (valid only)."""

    def __init__(self, windows: list[WindowSample], valid_only: bool = True) -> None:
        self.samples = [w for w in windows if w.valid] if valid_only else windows
        self.valid_only = valid_only

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.samples[idx]
        x = torch.as_tensor(w.features, dtype=torch.float32)   # (C, n_bins)
        y = torch.tensor([w.gt_hr], dtype=torch.float32)       # (1,)
        return x, y

    @property
    def n_channels(self) -> int:
        if not self.samples:
            return 1
        return self.samples[0].features.shape[0]

    @property
    def n_bins(self) -> int:
        if not self.samples:
            return 64
        return self.samples[0].features.shape[-1]
