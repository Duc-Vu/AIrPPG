"""Signal extraction configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from airppg.schemas import SIGNAL_EXTRACTION_SCHEMA_VERSION


@dataclass(slots=True)
class SignalExtractionConfig:
    preprocessing_manifest_path: Path = Path("outputs/preprocessing/preprocessing_dataset_manifest.json")
    signal_extraction_output_root: Path = Path("outputs/signal_extraction")
    process_all_preprocessing_outputs: bool = True
    batch_limit: int | None = None
    max_frames_per_sample: int | None = None
    window_sec: float = 30.0
    step_sec: float = 1.0
    hr_min_bpm: float = 42.0
    hr_max_bpm: float = 210.0
    save_figures: bool = True
    show_sample_visualization: bool = True
    progress: bool = True
    schema_version: str = SIGNAL_EXTRACTION_SCHEMA_VERSION

    def to_manifest_config(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "preprocessing_manifest_path": self.preprocessing_manifest_path.as_posix(),
            "signal_extraction_output_root": self.signal_extraction_output_root.as_posix(),
            "process_all_preprocessing_outputs": self.process_all_preprocessing_outputs,
            "batch_limit": self.batch_limit,
            "max_frames_per_sample": self.max_frames_per_sample,
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "hr_min_bpm": self.hr_min_bpm,
            "hr_max_bpm": self.hr_max_bpm,
            "save_figures": self.save_figures,
        }
