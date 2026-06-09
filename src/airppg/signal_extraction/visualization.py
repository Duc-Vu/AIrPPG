"""Signal extraction visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from airppg.metrics import align_ground_truth
from airppg.schemas import METHOD_NAMES
from airppg.signal import compute_spectrum


def build_visualization(
    sample_id: str,
    rgb_result: dict[str, np.ndarray],
    baselines: dict[str, dict[str, np.ndarray]],
    hr_results: dict[str, np.ndarray],
    hr_timestamps: np.ndarray,
    gt: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float | int]],
    fps: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
    output_path: Path | None = None,
) -> plt.Figure:
    timestamps = rgb_result["timestamps"]
    mean_rgb = rgb_result["mean_rgb"]
    gt_aligned = align_ground_truth(gt, hr_timestamps)

    fig, axes = plt.subplots(5, 1, figsize=(13, 20))
    fig.suptitle(f"Signal extraction rPPG evaluation: {sample_id}", fontsize=16)

    axes[0].plot(timestamps, mean_rgb[:, 0, 0], color="tab:red", label="forehead R", alpha=0.8)
    axes[0].plot(timestamps, mean_rgb[:, 0, 1], color="tab:green", label="forehead G", alpha=0.8)
    axes[0].plot(timestamps, mean_rgb[:, 0, 2], color="tab:blue", label="forehead B", alpha=0.8)
    axes[0].set_title("Mean RGB trace from forehead ROI")
    axes[0].set_ylabel("Pixel value")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    if timestamps.size:
        plot_start = float(np.nanmin(timestamps))
        plot_end = plot_start + min(15.0, float(np.nanmax(timestamps) - plot_start))
        window_mask = (timestamps >= plot_start) & (timestamps <= plot_end)
    else:
        window_mask = np.array([], dtype=bool)
    for method in METHOD_NAMES:
        axes[1].plot(timestamps[window_mask], baselines[method]["signal"][window_mask], label=method.upper(), alpha=0.85)
    axes[1].set_title("rPPG signals over the first analysis window")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    for method in METHOD_NAMES:
        freqs_bpm, magnitude = compute_spectrum(baselines[method]["signal"], fps)
        band = (freqs_bpm >= hr_min_bpm) & (freqs_bpm <= hr_max_bpm)
        axes[2].plot(freqs_bpm[band], magnitude[band], label=method.upper(), alpha=0.85)
    gt_mean = float(np.nanmean(gt["heart_rate_bpm"]))
    if np.isfinite(gt_mean):
        axes[2].axvline(gt_mean, color="black", linestyle="--", linewidth=1.5, label=f"GT mean {gt_mean:.1f} BPM")
    axes[2].set_title("Normalized frequency spectrum")
    axes[2].set_xlabel("Frequency (BPM)")
    axes[2].set_ylabel("Normalized magnitude")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(hr_timestamps, gt_aligned, color="black", linestyle="--", linewidth=2.0, label="Ground truth")
    for method in METHOD_NAMES:
        axes[3].plot(hr_timestamps, hr_results[method], label=method.upper(), alpha=0.85)
    axes[3].set_title("Heart-rate estimates vs ground truth")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Heart rate (BPM)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    metric_names = ("mae_bpm", "rmse_bpm", "snr_db")
    x = np.arange(len(metric_names))
    width = 0.25
    for offset, method in zip((-width, 0.0, width), METHOD_NAMES, strict=True):
        axes[4].bar(x + offset, [metrics[method][name] for name in metric_names], width=width, label=method.upper())
    axes[4].set_title("Evaluation metrics")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(("MAE (BPM)", "RMSE (BPM)", "SNR (dB)"))
    axes[4].legend(loc="upper right")
    axes[4].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig
