"""Signal extraction artifact loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from airppg.paths import load_json, resolve_relative
from airppg.schemas import ROI_NAMES


def load_preprocessing_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest.get("items"), list):
        raise KeyError("preprocessing manifest must contain an 'items' list.")
    return manifest


def select_preprocessing_items(manifest: dict[str, Any], process_all: bool = True, batch_limit: int | None = None) -> list[dict[str, Any]]:
    items = [item for item in manifest["items"] if item.get("status") == "processed"]
    if not process_all:
        raise ValueError("Signal extraction is configured to process all preprocessing outputs. Set process_all=True.")
    return items[:batch_limit] if batch_limit is not None else items


def load_preprocessing_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    required = {"frame_indices", "roi_masks", "valid"}
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"preprocessing ROI npz is missing required arrays: {sorted(missing)}")
    return arrays


def get_roi_names(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> np.ndarray:
    if "roi_names" in arrays:
        return arrays["roi_names"].astype(object)
    return np.array(metadata.get("roi_names", ROI_NAMES), dtype=object)


def load_ground_truth_npz(path: str | Path) -> dict[str, np.ndarray]:
    required = ("timestamps", "heart_rate_bpm", "ppg_signal", "ppg_timestamps", "spo2_percent")
    with np.load(path, allow_pickle=False) as data:
        missing = set(required).difference(data.files)
        if missing:
            raise KeyError(f"Ground truth npz is missing required arrays: {sorted(missing)}")
        return {key: np.atleast_1d(np.asarray(data[key]).squeeze()).astype(np.float64) for key in required}


def resolve_preprocessing_output_path(path_text: str | Path, preprocessing_manifest_path: str | Path) -> Path:
    return resolve_relative(path_text, Path(preprocessing_manifest_path).parent)
