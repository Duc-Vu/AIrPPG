"""Task 4 — Full modeling pipeline script.

Usage examples (from repo root):
    uv run python scripts/run_modeling.py
    uv run python scripts/run_modeling.py --strategy fused_spectrum --model small_tcn
    uv run python scripts/run_modeling.py --strategy multi_channel_spectrum --n_fft_bins 128
    uv run python scripts/run_modeling.py --strategy pos_spectrum --batch_limit 5  # quick test

Outputs written to outputs/modeling/  (schema_version = modeling_v1).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── repo root resolution (same pattern as Task 1-3) ───────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from airppg.modeling.config import MODELING_SCHEMA_VERSION, ModelingConfig
from airppg.modeling.dataset import (
    WindowDataset,
    apply_normalization,
    fit_normalization,
    load_all_windows,
)
from airppg.modeling.metrics import (
    aggregate_model_metric_rows,
    build_comparison_table,
    compute_hr_metrics,
)
from airppg.modeling.models import build_model
from airppg.modeling.train import save_config, train
from airppg.modeling.visualization import (
    build_sample_visualization,
    plot_comparison_bar,
    plot_training_curves,
)
from airppg.paths import load_json, write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_modeling")


# ── argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIrPPG Task 4 — Lightweight model pipeline")
    p.add_argument(
        "--strategy",
        choices=["pos_spectrum", "fused_spectrum", "multi_channel_spectrum"],
        default="pos_spectrum",
        help="Input feature strategy (default: pos_spectrum)",
    )
    p.add_argument(
        "--model",
        choices=["tiny_cnn", "small_tcn"],
        default="tiny_cnn",
        help="Model architecture (default: tiny_cnn)",
    )
    p.add_argument("--n_fft_bins", type=int, default=64, help="FFT bins in HR band (default: 64)")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Checkpoint sub-dir name. Defaults to '{model}_{strategy}'",
    )
    p.add_argument("--batch_limit", type=int, default=None, help="Limit samples for quick test")
    p.add_argument("--no_figures", action="store_true", help="Skip saving per-sample figures")
    return p.parse_args()


# ── inference on one sample ────────────────────────────────────────────────────

@torch.no_grad()
def _infer_sample(
    model: torch.nn.Module,
    windows: list,   # list[WindowSample] for one sample
    device: torch.device,
) -> np.ndarray:
    """Run model inference on all windows of one sample.

    Returns predicted_hr array of shape (W,).  NaN for invalid windows.
    """
    predicted = np.full(len(windows), float("nan"), dtype=np.float32)
    model.eval()
    for idx, w in enumerate(windows):
        if not w.valid:
            continue
        x = torch.as_tensor(w.features, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x)  # (1, 1)
        predicted[idx] = float(pred.squeeze().item())
    return predicted


# ── save per-sample npz ────────────────────────────────────────────────────────

def _save_predictions_npz(
    path: Path,
    sample_id: str,
    split: str,
    model_name: str,
    input_strategy: str,
    hr_timestamps: np.ndarray,
    gt_aligned_to_hr: np.ndarray,
    predicted_hr: np.ndarray,
    baseline_hr: np.ndarray | None,
    baseline_names: list[str],
    valid_windows: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "schema_version": np.array(MODELING_SCHEMA_VERSION),
        "sample_id": np.array(sample_id),
        "split": np.array(split),
        "model_name": np.array(model_name),
        "hr_timestamps": hr_timestamps.astype(np.float32),
        "gt_aligned_to_hr": gt_aligned_to_hr.astype(np.float32),
        "predicted_hr": predicted_hr.astype(np.float32),
        "input_strategy_names": np.array([input_strategy]),
        "valid_windows": valid_windows.astype(bool),
    }
    if baseline_hr is not None and len(baseline_hr) > 0:
        arrays["baseline_hr"] = baseline_hr.astype(np.float32)
        arrays["baseline_names"] = np.array(baseline_names)
    np.savez_compressed(path, **arrays)


# ── main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    run_name = args.run_name or f"{args.model}_{args.strategy}"
    cfg = ModelingConfig(
        input_strategy=args.strategy,
        model_type=args.model,
        n_fft_bins=args.n_fft_bins,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        run_name=run_name,
        batch_limit=args.batch_limit,
        save_figures=not args.no_figures,
    )

    # ── change to repo root so relative paths work ─────────────────────────────
    import os
    os.chdir(_REPO_ROOT)

    logger.info("=== Task 4 Modeling Pipeline ===")
    logger.info("Strategy: %s | Model: %s | Run: %s", cfg.input_strategy, cfg.model_type, cfg.run_name)

    # ── load manifests ─────────────────────────────────────────────────────────
    logger.info("Loading manifests …")
    fusion_manifest = load_json(cfg.fusion_manifest_path)
    sig_manifest = load_json(cfg.signal_extraction_manifest_path)

    # ── build window datasets ──────────────────────────────────────────────────
    logger.info("Building window datasets …")
    t0 = time.perf_counter()
    split_windows = load_all_windows(cfg, fusion_manifest, sig_manifest)
    logger.info(
        "Windows loaded in %.1fs: train=%d val=%d test=%d",
        time.perf_counter() - t0,
        len(split_windows.get("train", [])),
        len(split_windows.get("val", [])),
        len(split_windows.get("test", [])),
    )

    train_windows = split_windows.get("train", [])
    val_windows = split_windows.get("val", [])
    test_windows = split_windows.get("test", [])

    # ── fit normalization on train ─────────────────────────────────────────────
    logger.info("Fitting normalization on train split …")
    norm_mean, norm_std = fit_normalization(train_windows)
    apply_normalization(train_windows, norm_mean, norm_std)
    apply_normalization(val_windows, norm_mean, norm_std)
    apply_normalization(test_windows, norm_mean, norm_std)

    # store in cfg for config.json
    cfg.norm_mean = norm_mean.tolist()
    cfg.norm_std = norm_std.tolist()

    # ── build Datasets ─────────────────────────────────────────────────────────
    train_ds = WindowDataset(train_windows, valid_only=True)
    val_ds = WindowDataset(val_windows, valid_only=True)
    logger.info("Effective train=%d val=%d (valid windows)", len(train_ds), len(val_ds))

    if len(train_ds) == 0:
        logger.error("No valid train windows! Aborting.")
        sys.exit(1)

    # ── build model ────────────────────────────────────────────────────────────
    cfg.n_input_channels = train_ds.n_channels
    cfg.n_input_bins = train_ds.n_bins
    model = build_model(cfg.model_type, cfg.n_input_channels, cfg.n_input_bins)
    cfg.param_count = model.param_count()  # type: ignore[attr-defined]
    logger.info(
        "Model: %s | channels=%d | bins=%d | params=%d",
        cfg.model_type, cfg.n_input_channels, cfg.n_input_bins, cfg.param_count,
    )

    # ── train ──────────────────────────────────────────────────────────────────
    logger.info("Training …")
    model, history = train(model, train_ds, val_ds, cfg)

    # ── save config ────────────────────────────────────────────────────────────
    save_config(cfg, cfg.checkpoint_dir / "config.json")

    # ── plot training curves ───────────────────────────────────────────────────
    if cfg.save_figures and history:
        tc_path = cfg.modeling_output_root / "checkpoints" / cfg.run_name / "training_curves.png"
        fig = plot_training_curves(history, cfg.run_name, output_path=tc_path)
        plt.close(fig)
        logger.info("Training curves saved to %s", tc_path)

    # ── inference + per-sample metrics ────────────────────────────────────────
    device = torch.device("cpu")
    all_metric_rows: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []

    for split_name, windows in [
        ("train", train_windows),
        ("val", val_windows),
        ("test", test_windows),
    ]:
        # group windows by sample_id
        sample_groups: dict[str, list] = {}
        for w in windows:
            sample_groups.setdefault(w.sample_id, []).append(w)

        for sample_id, s_windows in sample_groups.items():
            logger.debug("Inference: %s/%s (%d windows)", split_name, sample_id, len(s_windows))

            hr_timestamps = np.array([w.hr_timestamp for w in s_windows], dtype=np.float32)
            gt_aligned = np.array([w.gt_hr for w in s_windows], dtype=np.float32)
            valid_windows_arr = np.array([w.valid for w in s_windows], dtype=bool)
            invalid_reasons = [w.invalid_reason for w in s_windows]

            # infer
            t_infer = time.perf_counter()
            predicted_hr = _infer_sample(model, s_windows, device)
            inference_ms = (time.perf_counter() - t_infer) * 1000.0

            # compute metrics
            metrics = compute_hr_metrics(predicted_hr, gt_aligned, valid_windows_arr)
            metrics["inference_ms_total"] = round(inference_ms, 2)
            metrics["inference_ms_per_window"] = (
                round(inference_ms / len(s_windows), 3) if s_windows else float("nan")
            )

            # baseline HR arrays (from fusion npz if available)
            baseline_hr_dict: dict[str, np.ndarray] = {}
            baseline_np: np.ndarray | None = None
            baseline_names: list[str] = []

            # try to load baseline from fusion for this sample
            fusion_items_map = {
                item["sample_id"]: item
                for item in fusion_manifest.get("items", [])
                if item.get("status") == "processed"
            }
            if sample_id in fusion_items_map:
                fusion_item = fusion_items_map[sample_id]
                fusion_root = Path(cfg.fusion_manifest_path).parent
                npz_path = fusion_root / fusion_item["outputs"]["fusion_npz"]
                if npz_path.exists():
                    with np.load(npz_path, allow_pickle=False) as d:
                        # average_fusion POS column
                        if "average_fusion_hr" in d.files:
                            af = d["average_fusion_hr"][:, 2]  # POS
                            baseline_hr_dict["average_fusion"] = af
                        if "weighted_fusion_hr" in d.files:
                            wf = d["weighted_fusion_hr"][:, 2]
                            baseline_hr_dict["quality_weighted_fusion"] = wf
                        if "baseline_hr" in d.files:
                            bhr = d["baseline_hr"]  # (W, 3) green/chrom/pos
                            for mi, mname in enumerate(["task2_green", "task2_chrom", "task2_pos"]):
                                baseline_hr_dict[mname] = bhr[:, mi]

            if baseline_hr_dict:
                baseline_names = list(baseline_hr_dict.keys())
                baseline_np = np.stack(list(baseline_hr_dict.values()), axis=1)  # (W, B)

            # output paths
            out_dir = cfg.modeling_output_root / split_name / sample_id
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path_out = out_dir / "model_predictions.npz"
            metrics_json_path = out_dir / "model_metrics.json"
            metrics_csv_path = out_dir / "model_metrics.csv"
            vis_path = out_dir / "model_visualization.png"

            # save npz
            _save_predictions_npz(
                npz_path_out,
                sample_id=sample_id,
                split=split_name,
                model_name=cfg.run_name,
                input_strategy=cfg.input_strategy,
                hr_timestamps=hr_timestamps,
                gt_aligned_to_hr=gt_aligned,
                predicted_hr=predicted_hr,
                baseline_hr=baseline_np,
                baseline_names=baseline_names,
                valid_windows=valid_windows_arr,
            )

            # save metrics JSON
            metrics_full = {
                "schema_version": MODELING_SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": split_name,
                "model_name": cfg.run_name,
                "input_strategy": cfg.input_strategy,
                "model_type": cfg.model_type,
                "param_count": cfg.param_count,
                "n_windows_total": len(s_windows),
                "n_windows_valid": int(valid_windows_arr.sum()),
                "n_windows_invalid": int((~valid_windows_arr).sum()),
                "invalid_reasons": [r for r in invalid_reasons if r],
                **metrics,
            }
            write_json(metrics_json_path, metrics_full)

            # save metrics CSV (one row)
            with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["sample_id", "split", "model_name", "strategy",
                                "mae_bpm", "rmse_bpm", "pearson_r", "bias_bpm", "n_windows"],
                )
                writer.writeheader()
                writer.writerow({
                    "sample_id": sample_id,
                    "split": split_name,
                    "model_name": cfg.run_name,
                    "strategy": cfg.input_strategy,
                    "mae_bpm": metrics.get("mae_bpm"),
                    "rmse_bpm": metrics.get("rmse_bpm"),
                    "pearson_r": metrics.get("pearson_r"),
                    "bias_bpm": metrics.get("bias_bpm"),
                    "n_windows": metrics.get("n_windows"),
                })

            # per-sample visualization
            if cfg.save_figures:
                fig = build_sample_visualization(
                    hr_timestamps=hr_timestamps,
                    gt_hr=gt_aligned,
                    predicted_hr=predicted_hr,
                    baseline_hr=baseline_hr_dict if baseline_hr_dict else None,
                    sample_id=sample_id,
                    model_name=cfg.run_name,
                    metrics=metrics,
                    output_path=vis_path,
                )
                plt.close(fig)

            # collect for summary
            metric_row = {
                "sample_id": sample_id,
                "split": split_name,
                "model_name": cfg.run_name,
                "strategy": cfg.input_strategy,
                **{k: v for k, v in metrics.items()
                   if k in ("mae_bpm", "rmse_bpm", "pearson_r", "bias_bpm", "n_windows")},
            }
            all_metric_rows.append(metric_row)

            # manifest item
            manifest_items.append({
                "sample_id": sample_id,
                "split": split_name,
                "status": "processed",
                "model_name": cfg.run_name,
                "n_windows_total": len(s_windows),
                "n_windows_valid": int(valid_windows_arr.sum()),
                "outputs": {
                    "predictions_npz": (npz_path_out.relative_to(cfg.modeling_output_root)).as_posix(),
                    "metrics_json": (metrics_json_path.relative_to(cfg.modeling_output_root)).as_posix(),
                    "metrics_csv": (metrics_csv_path.relative_to(cfg.modeling_output_root)).as_posix(),
                    "visualization_png": (vis_path.relative_to(cfg.modeling_output_root)).as_posix(),
                },
            })

    # ── aggregate summary ─────────────────────────────────────────────────────
    aggregate_rows = aggregate_model_metric_rows(all_metric_rows)

    # save modeling_metrics_summary.csv
    summary_csv = cfg.modeling_output_root / "modeling_metrics_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_metric_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_metric_rows)

    # save modeling_aggregate_summary.csv
    agg_csv = cfg.modeling_output_root / "modeling_aggregate_summary.csv"
    if aggregate_rows:
        with agg_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    # ── comparison table + bar charts ─────────────────────────────────────────
    fusion_agg_csv = Path("outputs/fusion/fusion_aggregate_summary.csv")
    sig_metrics_csv = Path("outputs/signal_extraction/signal_extraction_metrics_summary.csv")

    comparison_df = build_comparison_table(
        aggregate_rows,
        fusion_agg_csv,
        sig_metrics_csv,
        split="test",
    )
    if not comparison_df.empty:
        comp_csv = cfg.modeling_output_root / "comparison_table_test.csv"
        comparison_df.to_csv(comp_csv, index=False)
        logger.info("Comparison table saved to %s", comp_csv)

        if cfg.save_figures:
            assets_dir = Path("report/assets")
            assets_dir.mkdir(parents=True, exist_ok=True)
            for metric in ["mae_bpm", "rmse_bpm", "pearson_r"]:
                fig = plot_comparison_bar(
                    comparison_df, metric=metric,
                    title=f"Test {metric.replace('_', ' ').upper()} — Model vs Baselines ({cfg.run_name})",
                    output_path=assets_dir / f"task4_{cfg.run_name}_{metric}.png",
                )
                plt.close(fig)

    # ── modeling manifest ─────────────────────────────────────────────────────
    counts = {
        "total": len(manifest_items),
        "processed": sum(1 for i in manifest_items if i["status"] == "processed"),
        "failed": sum(1 for i in manifest_items if i["status"] != "processed"),
        "train": sum(1 for i in manifest_items if i["split"] == "train"),
        "val": sum(1 for i in manifest_items if i["split"] == "val"),
        "test": sum(1 for i in manifest_items if i["split"] == "test"),
    }
    manifest = {
        "schema_version": MODELING_SCHEMA_VERSION,
        "fusion_manifest": str(cfg.fusion_manifest_path),
        "output_root_hint": str(cfg.modeling_output_root),
        "config": cfg.to_dict(),
        "counts": counts,
        "items": manifest_items,
    }
    manifest_path = cfg.modeling_output_root / "modeling_manifest.json"
    write_json(manifest_path, manifest)
    logger.info("Manifest saved to %s", manifest_path)

    # ── final summary ─────────────────────────────────────────────────────────
    test_rows = [r for r in all_metric_rows if r["split"] == "test"]
    if test_rows:
        test_mae_vals = [r["mae_bpm"] for r in test_rows if r["mae_bpm"] is not None]
        test_rmse_vals = [r["rmse_bpm"] for r in test_rows if r["rmse_bpm"] is not None]
        logger.info(
            "=== TEST RESULTS ===  MAE=%.2f±%.2f bpm  RMSE=%.2f±%.2f bpm  (n_samples=%d)",
            np.nanmean(test_mae_vals), np.nanstd(test_mae_vals),
            np.nanmean(test_rmse_vals), np.nanstd(test_rmse_vals),
            len(test_rows),
        )
    logger.info("=== Done. Outputs in %s ===", cfg.modeling_output_root)


if __name__ == "__main__":
    main()
