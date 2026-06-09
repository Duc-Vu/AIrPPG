"""UBFC ground-truth parsers and normalized writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from airppg.preprocessing.dataset import source_dataset_name, source_relpath
from airppg.schemas import PREPROCESSING_GT_SCHEMA_VERSION


def find_ground_truth_file(video_path: Path) -> Path | None:
    for filename in ("ground_truth.txt", "gtdump.xmp"):
        candidate = video_path.parent / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _as_float32_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"Ground-truth field {name!r} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"Ground-truth field {name!r} contains NaN or infinite values")
    return array


def load_dataset_1_ground_truth(gt_path: Path) -> dict[str, Any]:
    table = np.loadtxt(gt_path, delimiter=",", dtype=np.float32)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 4:
        raise ValueError(f"DATASET_1 ground truth must have at least 4 columns: {gt_path}")
    timestamps = _as_float32_1d(table[:, 0] / 1000.0, "timestamps")
    heart_rate_bpm = _as_float32_1d(table[:, 1], "heart_rate_bpm")
    spo2_percent = _as_float32_1d(table[:, 2], "spo2_percent")
    ppg_signal = _as_float32_1d(table[:, 3], "ppg_signal")
    return {
        "source_format": "ubfc_dataset_1_gtdump_xmp",
        "arrays": {
            "timestamps": timestamps,
            "heart_rate_bpm": heart_rate_bpm,
            "ppg_signal": ppg_signal,
            "ppg_timestamps": timestamps,
            "spo2_percent": spo2_percent,
        },
        "missing_fields": [],
        "notes": ["DATASET_1 gtdump.xmp columns are timestamp_ms, heart_rate_bpm, spo2_percent, ppg_signal."],
    }


def parse_ground_truth_line(line: str, gt_path: Path, field_name: str) -> np.ndarray:
    values = np.fromstring(line.strip(), sep=" ", dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"DATASET_2 ground-truth line for {field_name!r} is empty: {gt_path}")
    if not np.isfinite(values).all():
        raise ValueError(f"DATASET_2 ground-truth line for {field_name!r} contains NaN or infinite values: {gt_path}")
    return values.astype(np.float32, copy=False)


def load_dataset_2_ground_truth(gt_path: Path) -> dict[str, Any]:
    non_empty_lines = [line for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(non_empty_lines) != 3:
        raise ValueError(f"DATASET_2 ground_truth.txt must contain exactly 3 non-empty lines: {gt_path}")
    ppg_signal = parse_ground_truth_line(non_empty_lines[0], gt_path, "ppg_signal")
    heart_rate_bpm = parse_ground_truth_line(non_empty_lines[1], gt_path, "heart_rate_bpm")
    timestamps = parse_ground_truth_line(non_empty_lines[2], gt_path, "timestamps")
    if not (ppg_signal.size == heart_rate_bpm.size == timestamps.size):
        raise ValueError(
            "DATASET_2 ground-truth lengths differ "
            f"for {gt_path}: ppg={ppg_signal.size}, hr={heart_rate_bpm.size}, timestamps={timestamps.size}"
        )
    return {
        "source_format": "ubfc_dataset_2_ground_truth_txt",
        "arrays": {
            "timestamps": timestamps,
            "heart_rate_bpm": heart_rate_bpm,
            "ppg_signal": ppg_signal,
            "ppg_timestamps": timestamps,
            "spo2_percent": np.full(timestamps.shape, np.nan, dtype=np.float32),
        },
        "missing_fields": ["spo2_percent"],
        "notes": ["DATASET_2 ground_truth.txt lines are ppg_signal, heart_rate_bpm, timestamps_seconds; SpO2 is unavailable and stored as NaN."],
    }


def load_and_normalize_ground_truth(gt_path: Path, dataset_source: str) -> dict[str, Any]:
    if dataset_source == "DATASET_1" or gt_path.name == "gtdump.xmp":
        loaded = load_dataset_1_ground_truth(gt_path)
    elif dataset_source == "DATASET_2" or gt_path.name == "ground_truth.txt":
        loaded = load_dataset_2_ground_truth(gt_path)
    else:
        raise ValueError(f"Unsupported dataset source for ground truth: {dataset_source} ({gt_path})")
    arrays = {name: value.astype(np.float32, copy=False) for name, value in loaded["arrays"].items()}
    shape = arrays["timestamps"].shape
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"Ground-truth field {name!r} shape {array.shape} does not match timestamps shape {shape}: {gt_path}")
    return {
        "arrays": arrays,
        "metadata": {
            "available": True,
            "schema_version": PREPROCESSING_GT_SCHEMA_VERSION,
            "source_format": loaded["source_format"],
            "units": {
                "timestamps": "seconds",
                "heart_rate_bpm": "beats_per_minute",
                "ppg_signal": "arbitrary_unit",
                "ppg_timestamps": "seconds",
                "spo2_percent": "percent",
            },
            "sample_counts": {name: int(array.size) for name, array in arrays.items()},
            "missing_fields": loaded["missing_fields"],
            "notes": loaded["notes"],
        },
    }


def save_normalized_ground_truth(video_path: Path, dataset_root: Path, dataset_root_hint: str, output_dir: Path) -> dict[str, Any]:
    gt_path = find_ground_truth_file(video_path)
    if gt_path is None:
        return {
            "available": False,
            "schema_version": PREPROCESSING_GT_SCHEMA_VERSION,
            "source_relpath": None,
            "source_format": None,
            "units": {},
            "sample_counts": {},
            "missing_fields": [],
            "notes": ["No UBFC ground-truth file was found next to the video."],
        }
    normalized = load_and_normalize_ground_truth(gt_path, source_dataset_name(video_path, dataset_root))
    arrays = normalized["arrays"]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "ground_truth.npz",
        timestamps=arrays["timestamps"],
        heart_rate_bpm=arrays["heart_rate_bpm"],
        ppg_signal=arrays["ppg_signal"],
        ppg_timestamps=arrays["ppg_timestamps"],
        spo2_percent=arrays["spo2_percent"],
    )
    metadata = dict(normalized["metadata"])
    metadata["source_relpath"] = source_relpath(gt_path, dataset_root, dataset_root_hint)
    return metadata
