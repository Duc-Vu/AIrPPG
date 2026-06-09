"""Load Signal extraction artifacts for fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from airppg.paths import load_json, resolve_relative

REQUIRED_SIGNAL_KEYS = {
    "timestamps",
    "valid",
    "roi_names",
    "method_names",
    "rppg_roi_signals",
    "hr_timestamps",
    "gt_aligned_to_hr",
    "hr_estimates",
    "mean_rgb",
}


def load_signal_extraction_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest.get("items"), list):
        raise KeyError("signal extraction manifest must contain an 'items' list.")
    return manifest


def select_signal_extraction_items(
    manifest: dict[str, Any], batch_limit: int | None = None
) -> list[dict[str, Any]]:
    items = [item for item in manifest["items"] if item.get("status") == "processed"]
    return items[:batch_limit] if batch_limit is not None else items


def resolve_signal_output_path(
    path_text: str | Path, signal_extraction_manifest_path: str | Path
) -> Path:
    return resolve_relative(path_text, Path(signal_extraction_manifest_path).parent)


def load_signal_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_SIGNAL_KEYS.difference(data.files)
        if missing:
            raise KeyError(f"Signal extraction npz is missing required arrays: {sorted(missing)}")
        return {key: data[key] for key in data.files}
