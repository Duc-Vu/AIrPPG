"""Task 4 modeling configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MODELING_SCHEMA_VERSION = "modeling_v1"


@dataclass(slots=True)
class ModelingConfig:
    """All hyper-parameters and paths for Task 4.

    Design notes
    ------------
    * ``input_strategy`` controls what signal is fed into the model:
        - ``"pos_spectrum"``          : FFT power spectrum of POS (average-ROI) signal.
        - ``"fused_spectrum"``        : FFT power spectrum of average-fused POS signal (Task 3).
        - ``"multi_channel_spectrum"``: stack spectra of all 3 methods (green/chrom/pos) of
                                        average-fused signals → shape (3, n_fft_bins).
    * ``n_fft_bins`` is the number of frequency bins kept inside the HR band
      [hr_min_bpm, hr_max_bpm]. Fixed at build-time so the model input size is constant
      regardless of FPS differences across samples.
    * Normalization is fit on train, applied to val/test, and stored in config.json.
    """

    # ── paths ─────────────────────────────────────────────────────────────────
    fusion_manifest_path: Path = Path("outputs/fusion/fusion_manifest.json")
    signal_extraction_manifest_path: Path = Path(
        "outputs/signal_extraction/signal_extraction_dataset_manifest.json"
    )
    modeling_output_root: Path = Path("outputs/modeling")

    # ── run identity ──────────────────────────────────────────────────────────
    run_name: str = "tiny_cnn_pos_spectrum"
    schema_version: str = MODELING_SCHEMA_VERSION

    # ── signal window geometry (must match Task 2/3) ──────────────────────────
    window_sec: float = 30.0
    step_sec: float = 1.0
    hr_min_bpm: float = 42.0
    hr_max_bpm: float = 210.0

    # ── feature engineering ───────────────────────────────────────────────────
    input_strategy: Literal[
        "pos_spectrum", "fused_spectrum", "multi_channel_spectrum"
    ] = "pos_spectrum"
    n_fft_bins: int = 64  # bins inside [hr_min_bpm, hr_max_bpm]

    # ── model architecture ────────────────────────────────────────────────────
    model_type: Literal["tiny_cnn", "small_tcn"] = "tiny_cnn"

    # ── training hyper-parameters ─────────────────────────────────────────────
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 30          # early stopping patience (val_mae_bpm)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    # ── misc ──────────────────────────────────────────────────────────────────
    batch_limit: int | None = None   # limit samples for quick tests
    save_figures: bool = True
    progress: bool = True

    # ── normalization params (filled after fitting on train) ──────────────────
    norm_mean: list[float] = field(default_factory=list)
    norm_std: list[float] = field(default_factory=list)

    # ── derived (read-only, set externally after build) ───────────────────────
    n_input_channels: int = 1
    n_input_bins: int = 64
    param_count: int = 0

    @property
    def checkpoint_dir(self) -> Path:
        return self.modeling_output_root / "checkpoints" / self.run_name

    def to_dict(self) -> dict:
        """Serializable dict for config.json."""
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "fusion_manifest_path": self.fusion_manifest_path.as_posix(),
            "signal_extraction_manifest_path": self.signal_extraction_manifest_path.as_posix(),
            "modeling_output_root": self.modeling_output_root.as_posix(),
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "hr_min_bpm": self.hr_min_bpm,
            "hr_max_bpm": self.hr_max_bpm,
            "input_strategy": self.input_strategy,
            "n_fft_bins": self.n_fft_bins,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "n_input_channels": self.n_input_channels,
            "n_input_bins": self.n_input_bins,
            "param_count": self.param_count,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
        }
