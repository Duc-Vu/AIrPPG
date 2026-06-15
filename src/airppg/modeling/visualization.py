"""Task 4 visualization helpers.

Three plot types:
  1. plot_training_curves   — epoch-level train/val loss and val MAE
  2. plot_prediction_vs_gt  — per-sample HR over time (model + baselines)
  3. plot_comparison_bar    — bar chart comparing MAE/RMSE/Pearson across methods
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── style helpers ──────────────────────────────────────────────────────────────

_MODEL_COLORS = {
    "tiny_cnn": "#e63946",
    "small_tcn": "#457b9d",
    "pos_spectrum": "#e63946",
    "fused_spectrum": "#2a9d8f",
    "multi_channel_spectrum": "#f4a261",
}

_BASELINE_STYLES = {
    "task2_green":   {"color": "#adb5bd", "ls": "--", "lw": 1.4},
    "task2_chrom":   {"color": "#6c757d", "ls": "-.", "lw": 1.2},
    "task2_pos":     {"color": "#495057", "ls": "--", "lw": 1.6},
    "average_fusion":           {"color": "#1d3557", "ls": ":", "lw": 1.8},
    "quality_weighted_fusion":  {"color": "#457b9d", "ls": ":", "lw": 1.8},
    "signal_extraction_baseline": {"color": "#6c757d", "ls": "-.", "lw": 1.2},
}


# ── 1. Training curves ────────────────────────────────────────────────────────

def plot_training_curves(
    history: list[dict],
    run_name: str,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot train_loss, val_loss, and val_mae_bpm vs epoch."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    val_mae = [r["val_mae_bpm"] for r in history]
    best_epoch = int(np.argmin(val_mae)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    fig.suptitle(f"Training curves — {run_name}", fontsize=13, fontweight="bold")

    ax1.plot(epochs, train_loss, label="Train loss", color="#e63946", lw=1.8)
    ax1.plot(epochs, val_loss, label="Val loss", color="#457b9d", lw=1.8, ls="--")
    ax1.axvline(best_epoch, color="gray", ls=":", lw=1.2, label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Smooth L1 loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_mae, color="#f4a261", lw=1.8, label="Val MAE (bpm)")
    ax2.axvline(best_epoch, color="gray", ls=":", lw=1.2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (bpm)")
    ax2.set_title("Val MAE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
    return fig


# ── 2. Prediction vs ground truth ─────────────────────────────────────────────

def plot_prediction_vs_gt(
    hr_timestamps: np.ndarray,
    gt_hr: np.ndarray,
    predicted_hr: np.ndarray,
    baseline_hr: dict[str, np.ndarray] | None,
    sample_id: str,
    model_name: str,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """HR over time: model prediction + optional baselines vs ground truth.

    Parameters
    ----------
    baseline_hr: dict {label: (W,) array} for baseline curves, or None
    """
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

    ax.plot(
        hr_timestamps, gt_hr,
        label="Ground truth", color="black", lw=2.2, zorder=5,
    )
    ax.plot(
        hr_timestamps, predicted_hr,
        label=model_name, color="#e63946", lw=2.0, ls="-", zorder=4, alpha=0.9,
    )

    if baseline_hr:
        for label, hr_arr in baseline_hr.items():
            style = _BASELINE_STYLES.get(label, {"color": "#adb5bd", "ls": "--", "lw": 1.3})
            ax.plot(hr_timestamps, hr_arr, label=label, **style, alpha=0.75, zorder=3)

    ax.set_title(f"{sample_id} — HR prediction", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
    return fig


# ── 3. Comparison bar chart ────────────────────────────────────────────────────

def plot_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str = "mae_bpm",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart comparing a metric across all methods/strategies.

    Parameters
    ----------
    comparison_df: DataFrame with columns [source, method, strategy, mae_bpm, ...]
    metric:        column to plot ("mae_bpm", "rmse_bpm", "pearson_r")
    """
    if comparison_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    df = comparison_df.copy()
    df["label"] = df["strategy"].fillna(df["method"])
    # sort by metric (ascending for error metrics, descending for pearson)
    ascending = metric != "pearson_r"
    df = df.sort_values(metric, ascending=not ascending, na_position="last")

    colors = []
    for _, row in df.iterrows():
        src = str(row.get("source", ""))
        if src == "model":
            colors.append("#e63946")
        elif src == "task3_fusion":
            colors.append("#457b9d")
        else:
            colors.append("#adb5bd")

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.55 + 1.5)), constrained_layout=True)
    bars = ax.barh(df["label"], df[metric], color=colors, edgecolor="white", height=0.6)

    # value labels on bars
    for bar, val in zip(bars, df[metric]):
        if np.isfinite(val):
            ax.text(
                bar.get_width() + 0.01 * ax.get_xlim()[1],
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left", fontsize=8,
            )

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Comparison: {metric.replace('_', ' ').title()}", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

    # legend patches
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(color="#e63946", label="Model (Task 4)"),
        Patch(color="#457b9d", label="Task 3 Fusion"),
        Patch(color="#adb5bd", label="Task 2 Signal"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    return fig


# ── 4. Per-sample visualization (combined) ────────────────────────────────────

def build_sample_visualization(
    hr_timestamps: np.ndarray,
    gt_hr: np.ndarray,
    predicted_hr: np.ndarray,
    baseline_hr: dict[str, np.ndarray] | None,
    sample_id: str,
    model_name: str,
    metrics: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """2-panel figure: HR trace (top) + error per window (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(f"{sample_id} — {model_name}", fontsize=12, fontweight="bold")

    # top: HR trace
    ax1.plot(hr_timestamps, gt_hr, label="Ground truth", color="black", lw=2.2, zorder=5)
    ax1.plot(hr_timestamps, predicted_hr, label=model_name, color="#e63946", lw=2.0, alpha=0.9, zorder=4)
    if baseline_hr:
        for label, arr in baseline_hr.items():
            style = _BASELINE_STYLES.get(label, {"color": "#adb5bd", "ls": "--", "lw": 1.3})
            ax1.plot(hr_timestamps, arr, label=label, **style, alpha=0.65, zorder=3)
    ax1.set_ylabel("Heart rate (bpm)")
    ax1.set_title(
        f"MAE={metrics.get('mae_bpm', float('nan')):.2f} bpm  "
        f"RMSE={metrics.get('rmse_bpm', float('nan')):.2f} bpm  "
        f"Pearson r={metrics.get('pearson_r', float('nan')):.3f}"
    )
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # bottom: absolute error
    errors = np.asarray(predicted_hr, dtype=np.float64) - np.asarray(gt_hr, dtype=np.float64)
    ax2.bar(hr_timestamps, np.abs(errors), color="#e63946", alpha=0.7, width=0.9, label="|Error|")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|Error| (bpm)")
    ax2.set_title("Absolute error per window")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
    return fig
