"""Fusion experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from airppg.schemas import FUSION_SCHEMA_VERSION


@dataclass(slots=True)
class FusionConfig:
    signal_extraction_manifest_path: Path = Path(
        "outputs/signal_extraction/signal_extraction_dataset_manifest.json"
    )
    fusion_output_root: Path = Path("outputs/fusion")
    batch_limit: int | None = None
    window_sec: float = 30.0
    step_sec: float = 1.0
    hr_min_bpm: float = 42.0
    hr_max_bpm: float = 210.0
    save_figures: bool = True
    show_sample_visualization: bool = True
    progress: bool = True
    schema_version: str = FUSION_SCHEMA_VERSION

    def to_manifest_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signal_extraction_manifest_path": self.signal_extraction_manifest_path.as_posix(),
            "fusion_output_root": self.fusion_output_root.as_posix(),
            "batch_limit": self.batch_limit,
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "hr_min_bpm": self.hr_min_bpm,
            "hr_max_bpm": self.hr_max_bpm,
            "save_figures": self.save_figures,
        }
