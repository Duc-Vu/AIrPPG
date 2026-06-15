"""Task 4 evaluation metrics: per-sample and aggregate.

Reuses ``airppg.metrics.evaluate_method`` where possible.
Adds helpers for model-vs-baseline comparison tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ── core metric computation ────────────────────────────────────────────────────

def compute_hr_metrics(
    predicted_hr: np.ndarray,
    gt_hr: np.ndarray,
    valid_windows: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute MAE, RMSE, Pearson r, bias for HR estimates vs ground truth.

    Parameters
    ----------
    predicted_hr:   (W,) array of predicted HR bpm
    gt_hr:          (W,) array of ground truth HR bpm
    valid_windows:  (W,) bool; if None, use windows where both are finite
    """
    pred = np.asarray(predicted_hr, dtype=np.float64)
    gt = np.asarray(gt_hr, dtype=np.float64)

    finite = np.isfinite(pred) & np.isfinite(gt)
    if valid_windows is not None:
        finite = finite & np.asarray(valid_windows, dtype=bool)

    n = int(finite.sum())
    if n < 1:
        return {
            "mae_bpm": float("nan"),
            "rmse_bpm": float("nan"),
            "pearson_r": float("nan"),
            "bias_bpm": float("nan"),
            "n_windows": 0,
        }

    p = pred[finite]
    g = gt[finite]
    errors = p - g
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    pearson = float("nan")
    if n >= 2 and np.std(p) > 0 and np.std(g) > 0:
        r, _ = pearsonr(p, g)
        pearson = float(r) if np.isfinite(r) else float("nan")

    return {
        "mae_bpm": mae,
        "rmse_bpm": rmse,
        "pearson_r": pearson,
        "bias_bpm": bias,
        "n_windows": n,
    }


# ── aggregate across samples ───────────────────────────────────────────────────

def aggregate_model_metric_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mean / std of per-sample model metrics, grouped by (split, model_name, strategy)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (str(row["split"]), str(row["model_name"]), str(row.get("strategy", "model")))
        groups.setdefault(key, []).append(row)

    metric_names = ("mae_bpm", "rmse_bpm", "pearson_r", "bias_bpm")
    aggregate: list[dict[str, Any]] = []
    for (split, model_name, strategy), group in sorted(groups.items()):
        out: dict[str, Any] = {
            "split": split,
            "model_name": model_name,
            "strategy": strategy,
        }
        for m in metric_names:
            vals = np.asarray([r.get(m, float("nan")) for r in group], dtype=np.float64)
            out[f"{m}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            out[f"{m}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan")
        out["n_windows"] = int(sum(r.get("n_windows", 0) for r in group))
        aggregate.append(out)
    return aggregate


# ── baseline loading ───────────────────────────────────────────────────────────

def load_fusion_baseline_summary(
    fusion_aggregate_csv: str | Path,
    split: str = "test",
) -> pd.DataFrame:
    """Load Task 3 fusion aggregate CSV and filter to the given split.

    Returns a DataFrame with columns: method, strategy, mae_bpm_mean, rmse_bpm_mean,
    pearson_r_mean, bias_bpm_mean.
    """
    p = Path(fusion_aggregate_csv)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[df["split"] == split].copy()
    return df


def load_signal_extraction_baseline_summary(
    signal_extraction_metrics_csv: str | Path,
    split: str = "test",
) -> pd.DataFrame:
    """Load Task 2 per-sample metrics CSV and aggregate to method-level mean.

    Returns a DataFrame with columns: method, strategy, mae_bpm_mean, rmse_bpm_mean,
    pearson_r_mean, bias_bpm_mean.
    """
    p = Path(signal_extraction_metrics_csv)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "split" in df.columns:
        df = df[df["split"] == split]
    if df.empty:
        return df

    agg_rows: list[dict] = []
    for method, grp in df.groupby("method"):
        agg_rows.append({
            "method": method,
            "strategy": f"task2_{method}",
            "mae_bpm_mean": grp["mae_bpm"].mean(),
            "rmse_bpm_mean": grp["rmse_bpm"].mean(),
            "pearson_r_mean": grp["pearson_r"].mean(),
            "bias_bpm_mean": grp["bias_bpm"].mean(),
        })
    return pd.DataFrame(agg_rows)


# ── comparison table builder ───────────────────────────────────────────────────

def build_comparison_table(
    model_aggregate_rows: list[dict[str, Any]],
    fusion_aggregate_csv: str | Path,
    signal_metrics_summary_csv: str | Path,
    split: str = "test",
) -> pd.DataFrame:
    """Unified comparison table: model vs Task 2/3 baselines.

    Returns a single DataFrame with columns:
        source, method/model, strategy, mae_bpm, rmse_bpm, pearson_r, bias_bpm
    """
    rows: list[dict] = []

    # ── model rows ────────────────────────────────────────────────────────────
    for r in model_aggregate_rows:
        if r.get("split") == split:
            rows.append({
                "source": "model",
                "method": r.get("model_name", "?"),
                "strategy": r.get("strategy", "model"),
                "mae_bpm": r.get("mae_bpm_mean", float("nan")),
                "rmse_bpm": r.get("rmse_bpm_mean", float("nan")),
                "pearson_r": r.get("pearson_r_mean", float("nan")),
                "bias_bpm": r.get("bias_bpm_mean", float("nan")),
            })

    # ── Task 3 fusion baselines ────────────────────────────────────────────────
    fusion_df = load_fusion_baseline_summary(fusion_aggregate_csv, split=split)
    if not fusion_df.empty:
        want_strategies = {
            "single_forehead", "single_left_cheek", "single_right_cheek",
            "average_fusion", "quality_weighted_fusion",
        }
        for _, row in fusion_df.iterrows():
            if str(row.get("strategy", "")) in want_strategies:
                rows.append({
                    "source": "task3_fusion",
                    "method": str(row.get("method", "?")),
                    "strategy": str(row.get("strategy", "?")),
                    "mae_bpm": row.get("mae_bpm_mean", float("nan")),
                    "rmse_bpm": row.get("rmse_bpm_mean", float("nan")),
                    "pearson_r": row.get("pearson_r_mean", float("nan")),
                    "bias_bpm": row.get("bias_bpm_mean", float("nan")),
                })

    # ── Task 2 signal extraction baselines ────────────────────────────────────
    sig_df = load_signal_extraction_baseline_summary(signal_metrics_summary_csv, split=split)
    if not sig_df.empty:
        for _, row in sig_df.iterrows():
            rows.append({
                "source": "task2_signal",
                "method": str(row.get("method", "?")),
                "strategy": str(row.get("strategy", "?")),
                "mae_bpm": row.get("mae_bpm_mean", float("nan")),
                "rmse_bpm": row.get("rmse_bpm_mean", float("nan")),
                "pearson_r": row.get("pearson_r_mean", float("nan")),
                "bias_bpm": row.get("bias_bpm_mean", float("nan")),
            })

    return pd.DataFrame(rows)
