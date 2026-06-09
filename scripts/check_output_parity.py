from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def compare_array(label: str, old: np.ndarray, new: np.ndarray) -> bool:
    old = np.asarray(old)
    new = np.asarray(new)
    same_shape = old.shape == new.shape
    same_dtype = str(old.dtype) == str(new.dtype)
    if old.dtype.kind in "SUO" or new.dtype.kind in "SUO":
        same_values = same_shape and np.array_equal(old.astype(str), new.astype(str))
        diff: str | float = "n/a"
    elif old.dtype.kind == "b" or new.dtype.kind == "b" or old.dtype == np.uint8 or new.dtype == np.uint8:
        same_values = same_shape and np.array_equal(old.astype(np.uint8), new.astype(np.uint8))
        diff = 0 if same_values else ("n/a" if not same_shape else int(np.count_nonzero(old.astype(np.uint8) != new.astype(np.uint8))))
    else:
        same_values = same_shape and np.allclose(old, new, equal_nan=True)
        diff = "n/a" if not same_shape else (float(np.nanmax(np.abs(old.astype(np.float32) - new.astype(np.float32)))) if old.size else 0.0)
    print(f"{label}: same_shape={same_shape} same_dtype={same_dtype} same_values={same_values} diff={diff}")
    return bool(same_shape and same_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one overlapping legacy/new preprocessing and signal extraction sample.")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--old-pre", type=Path, default=Path("outputs/task1_preprocessing"))
    parser.add_argument("--new-pre", type=Path, default=Path("outputs/preprocessing"))
    parser.add_argument("--old-sig", type=Path, default=Path("outputs/task2_signals"))
    parser.add_argument("--new-sig", type=Path, default=Path("outputs/signal_extraction"))
    args = parser.parse_args()

    old_pre_manifest = load_json(args.old_pre / "task1_dataset_manifest.json")
    new_pre_manifest = load_json(args.new_pre / "preprocessing_dataset_manifest.json")
    old_pre_items = {item["sample_id"]: item for item in old_pre_manifest["items"]}
    new_pre_items = {item["sample_id"]: item for item in new_pre_manifest["items"]}
    sample_id = args.sample_id or sorted(set(old_pre_items) & set(new_pre_items))[0]
    print("preprocessing counts old/new:", old_pre_manifest["counts"], new_pre_manifest["counts"])
    print("sample:", sample_id)

    old_pre_item = old_pre_items[sample_id]
    new_pre_item = new_pre_items[sample_id]
    old_roi = load_npz(args.old_pre / old_pre_item["roi_npz"])
    new_roi = load_npz(args.new_pre / new_pre_item["roi_npz"])
    ok = True
    for label, old_key, new_key in (
        ("pre.frame_indices", "frame_indices", "frame_indices"),
        ("pre.valid", "valid", "valid"),
        ("pre.roi_names", "roi_names", "roi_names"),
        ("pre.landmarks", "landmarks_px", "landmarks"),
        ("pre.roi_polygons", "roi_polygons_px", "roi_polygons"),
        ("pre.roi_masks", "roi_masks", "roi_masks"),
    ):
        ok &= compare_array(label, old_roi[old_key], new_roi[new_key])

    old_gt = load_npz(args.old_pre / old_pre_item["ground_truth_npz"])
    new_gt = load_npz(args.new_pre / new_pre_item["ground_truth_npz"])
    for key in ("timestamps", "heart_rate_bpm", "ppg_signal", "ppg_timestamps", "spo2_percent"):
        ok &= compare_array(f"gt.{key}", old_gt[key], new_gt[key])

    old_sig_manifest = load_json(args.old_sig / "task2_dataset_manifest.json")
    new_sig_manifest = load_json(args.new_sig / "signal_extraction_dataset_manifest.json")
    old_sig_item = {item["sample_id"]: item for item in old_sig_manifest["items"]}[sample_id]
    new_sig_item = {item["sample_id"]: item for item in new_sig_manifest["items"]}[sample_id]
    print("signal counts old/new:", old_sig_manifest["counts"], new_sig_manifest["counts"])
    old_sig = load_npz(args.old_sig / old_sig_item["outputs"]["signals_npz"])
    new_sig = load_npz(args.new_sig / new_sig_item["outputs"]["signals_npz"])
    for key in (
        "frame_indices",
        "timestamps",
        "valid",
        "roi_names",
        "method_names",
        "mean_rgb",
        "rppg_signals",
        "rppg_roi_signals",
        "hr_timestamps",
        "hr_estimates",
        "gt_timestamps",
        "gt_heart_rate_bpm",
        "gt_aligned_to_hr",
        "ppg_timestamps",
        "ppg_signal",
        "spo2_percent",
    ):
        ok &= compare_array(f"sig.{key}", old_sig[key], new_sig[key])

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
