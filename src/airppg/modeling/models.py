"""Task 4 model architectures: TinyCNN1D and SmallTCN.

Both models:
  - Input shape: (batch, channels, n_bins)
  - Output shape: (batch, 1)  — scalar HR bpm (regression head)
  - Expose ``param_count()`` method
  - Are intentionally small (~20-40 K parameters) to avoid overfitting on the
    ~1 500 training windows available.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── TinyCNN1D ─────────────────────────────────────────────────────────────────

class TinyCNN1D(nn.Module):
    """Three-block 1-D CNN for HR regression from spectral features.

    Architecture (default for n_bins=64, channels=1):
        Conv1d(C→16, k=5, pad=2) → BN → ReLU
        Conv1d(16→32, k=5, pad=2) → BN → ReLU → MaxPool1d(2)
        Conv1d(32→32, k=3, pad=1) → BN → ReLU → AdaptiveAvgPool1d(8)
        Flatten(32*8=256) → Linear(256→64) → ReLU → Dropout(0.3)
        Linear(64→1)

    Param count ≈ 25 K.
    """

    def __init__(self, n_channels: int = 1, n_bins: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_bins = n_bins

        self.features = nn.Sequential(
            # block 1
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # block 2
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # block 3
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
        )

        flat_size = 32 * 8  # 256

        self.head = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, n_bins)
        h = self.features(x)          # (B, 32, 8)
        h = h.flatten(start_dim=1)    # (B, 256)
        return self.head(h)           # (B, 1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── SmallTCN ──────────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    """Single dilated residual block for 1-D temporal convolution."""

    def __init__(
        self,
        n_channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation   # causal padding

        self.conv1 = nn.Conv1d(
            n_channels, n_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=pad,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.conv2 = nn.Conv1d(
            n_channels, n_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=pad,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        out = self.conv1(x)
        # trim to original length (causal padding over-extends right end)
        out = out[..., : x.size(-1)]
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[..., : x.size(-1)]
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        return self.relu(out + x)   # residual connection


class SmallTCN(nn.Module):
    """Small Temporal Convolutional Network with dilated causal conv blocks.

    Architecture (default for n_bins=64, channels=1):
        Input projection: Conv1d(C→32)
        4 TCN residual blocks with dilation 1, 2, 4, 8
        AdaptiveAvgPool1d(1) → Flatten → Linear(32→1)

    Param count ≈ 30 K.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_bins: int = 64,
        hidden: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_bins = n_bins

        # input projection to hidden dim
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[_TCNBlock(hidden, dilation=d, dropout=dropout) for d in dilations]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, n_bins)
        h = self.input_proj(x)    # (B, hidden, n_bins)
        h = self.blocks(h)        # (B, hidden, n_bins)
        h = self.pool(h)          # (B, hidden, 1)
        h = h.squeeze(-1)         # (B, hidden)
        return self.head(h)       # (B, 1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── factory ────────────────────────────────────────────────────────────────────

def build_model(model_type: str, n_channels: int, n_bins: int) -> nn.Module:
    """Instantiate a model by name."""
    if model_type == "tiny_cnn":
        return TinyCNN1D(n_channels=n_channels, n_bins=n_bins)
    if model_type == "small_tcn":
        return SmallTCN(n_channels=n_channels, n_bins=n_bins)
    raise ValueError(f"Unknown model_type: {model_type!r}. Choose 'tiny_cnn' or 'small_tcn'.")
