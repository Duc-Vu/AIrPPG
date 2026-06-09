"""Preprocessing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from airppg.schemas import PREPROCESSING_SCHEMA_VERSION


@dataclass(slots=True)
class PreprocessingConfig:
    target_fps: float | None = None
    resize_width: int | None = 640
    max_frames: int | None = None
    save_overlay_video: bool = False
    overlay_sample_count: int = 8
    min_roi_area_px: float = 100.0
    schema_version: str = PREPROCESSING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "target_fps": self.target_fps,
            "resize_width": self.resize_width,
            "max_frames": self.max_frames,
            "save_overlay_video": self.save_overlay_video,
            "overlay_sample_count": self.overlay_sample_count,
            "min_roi_area_px": self.min_roi_area_px,
            "schema_version": self.schema_version,
        }


@dataclass(slots=True)
class PreprocessingBatchConfig:
    dataset_root: Path = Path("datasets/UBFC_DATASET")
    output_root: Path = Path("outputs/preprocessing")
    process_all_videos: bool = True
    batch_max_videos: int | None = None
    skip_existing: bool = True
    split_ratios: dict[str, float] | None = None
    split_seed: int = 42

    def ratios(self) -> dict[str, float]:
        return dict(self.split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1})

    def to_dict(self) -> dict[str, object]:
        return {
            "process_all_videos": self.process_all_videos,
            "batch_max_videos": self.batch_max_videos,
            "skip_existing": self.skip_existing,
        }
