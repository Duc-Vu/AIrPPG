"""Metric table helpers for fusion outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from airppg.metrics import evaluate_method

FUSION_STRATEGY_NAMES = (
    "single_forehead",
    "single_left_cheek",
    "single_right_cheek",
    "average_fusion",
    "quality_weighted_fusion",
    "signal_extraction_baseline",
)

METRIC_FIELDNAMES = (
    "sample_id",
    "split",
    "method",
    "strategy",
    "mae_bpm",
    "rmse_bpm",
    "pearson_r",
    "bias_bpm",
    "snr_db",
    "n_windows",
)


def _gt_from_aligned(
    hr_timestamps: np.ndarray, gt_aligned_to_hr: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "timestamps": np.asarray(hr_timestamps, dtype=np.float64),
        "heart_rate_bpm": np.asarray(gt_aligned_to_hr, dtype=np.float64),
    }


def build_metric_rows(
    sample_id: str,
    split: str,
    method_names: np.ndarray,
    roi_names: np.ndarray,
    hr_timestamps: np.ndarray,
    gt_aligned_to_hr: np.ndarray,
    single_roi_hr: np.ndarray,
    average_fusion_hr: np.ndarray,
    weighted_fusion_hr: np.ndarray,
    baseline_hr: np.ndarray,
    rppg_roi_signals: np.ndarray,
    average_fused_rppg_signals: np.ndarray,
    weighted_fused_rppg_signals: np.ndarray,
    baseline_rppg_signals: np.ndarray,
    fps: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
) -> list[dict[str, Any]]:
    gt = _gt_from_aligned(hr_timestamps, gt_aligned_to_hr)
    rows: list[dict[str, Any]] = []
    method_text = [str(name) for name in method_names.tolist()]
    roi_text = [str(name) for name in roi_names.tolist()]
    for m_idx, method in enumerate(method_text):
        for r_idx, roi in enumerate(roi_text):
            strategy = f"single_{roi}"
            metrics = evaluate_method(
                single_roi_hr[:, m_idx, r_idx],
                hr_timestamps,
                gt,
                rppg_roi_signals[m_idx, :, r_idx],
                fps,
                hr_min_bpm,
                hr_max_bpm,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "method": method,
                    "strategy": strategy,
                    **metrics,
                }
            )
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "method": method,
                "strategy": "average_fusion",
                **evaluate_method(
                    average_fusion_hr[:, m_idx],
                    hr_timestamps,
                    gt,
                    average_fused_rppg_signals[:, m_idx],
                    fps,
                    hr_min_bpm,
                    hr_max_bpm,
                ),
            }
        )
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "method": method,
                "strategy": "quality_weighted_fusion",
                **evaluate_method(
                    weighted_fusion_hr[:, m_idx],
                    hr_timestamps,
                    gt,
                    weighted_fused_rppg_signals[:, m_idx],
                    fps,
                    hr_min_bpm,
                    hr_max_bpm,
                ),
            }
        )
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "method": method,
                "strategy": "signal_extraction_baseline",
                **evaluate_method(
                    baseline_hr[:, m_idx],
                    hr_timestamps,
                    gt,
                    baseline_rppg_signals[:, m_idx],
                    fps,
                    hr_min_bpm,
                    hr_max_bpm,
                ),
            }
        )
    return rows


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["split"]), str(row["method"]), str(row["strategy"]))
        groups.setdefault(key, []).append(row)
    aggregate: list[dict[str, Any]] = []
    metric_names = ("mae_bpm", "rmse_bpm", "pearson_r", "bias_bpm", "snr_db")
    for (split, method, strategy), group_rows in sorted(groups.items()):
        out: dict[str, Any] = {"split": split, "method": method, "strategy": strategy}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group_rows], dtype=np.float64)
            out[f"{metric}_mean"] = (
                float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            )
            out[f"{metric}_std"] = (
                float(np.nanstd(values)) if np.isfinite(values).any() else np.nan
            )
        out["n_windows"] = int(np.sum([int(row["n_windows"]) for row in group_rows]))
        aggregate.append(out)
    return aggregate
