"""Visualization helpers for fusion experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_fusion_visualization(
    sample_id: str,
    method_names: np.ndarray,
    roi_names: np.ndarray,
    hr_timestamps: np.ndarray,
    gt_aligned_to_hr: np.ndarray,
    baseline_hr: np.ndarray,
    average_fusion_hr: np.ndarray,
    weighted_fusion_hr: np.ndarray,
    roi_weights: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    method_idx = 0
    method = str(method_names[method_idx]) if len(method_names) else "method"
    roi_labels = [str(name) for name in roi_names.tolist()]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    axes[0].plot(
        hr_timestamps,
        gt_aligned_to_hr,
        label="Ground truth",
        color="black",
        linewidth=2.0,
        zorder=4,
    )
    axes[0].plot(
        hr_timestamps,
        baseline_hr[:, method_idx],
        label="Signal baseline",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=3,
    )
    axes[0].plot(
        hr_timestamps,
        average_fusion_hr[:, method_idx],
        label="Average fusion",
        linestyle=":",
        linewidth=2.2,
        alpha=0.9,
        zorder=2,
    )
    axes[0].plot(
        hr_timestamps,
        weighted_fusion_hr[:, method_idx],
        label="Quality-weighted fusion",
        linestyle="-",
        linewidth=1.4,
        alpha=0.65,
        zorder=1,
    )
    axes[0].set_title(f"{sample_id} - {method} HR traces")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Heart rate (BPM)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    if roi_weights.size:
        for roi_idx, roi_name in enumerate(roi_labels):
            axes[1].plot(
                hr_timestamps,
                roi_weights[:, method_idx, roi_idx],
                label=roi_name,
            )
        axes[1].legend(loc="best")
    axes[1].set_title(f"ROI weights for {method}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Weight")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
