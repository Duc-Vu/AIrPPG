"""Task 4 training loop with early stopping.

Protocol
--------
* Train on ``train`` split, select best checkpoint by ``val_mae_bpm``.
* ``test`` split is NEVER used during training or model selection.
* Loss: Smooth L1 (Huber) which is less sensitive to outlier windows than MSE.
* Saves:
    checkpoints/<run_name>/best_model.pt
    checkpoints/<run_name>/training_history.csv
    checkpoints/<run_name>/config.json
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from airppg.modeling.config import ModelingConfig
from airppg.modeling.dataset import WindowDataset

logger = logging.getLogger(__name__)


# ── seeding ────────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── one epoch helpers ──────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)           # (B, 1)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n if n > 0 else float("nan")


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_pred: list[float] = []
    all_gt: list[float] = []
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_pred.extend(pred.squeeze(-1).cpu().tolist())
        all_gt.extend(y.squeeze(-1).cpu().tolist())
    if n == 0:
        return {"loss": float("nan"), "mae_bpm": float("nan"), "rmse_bpm": float("nan")}
    preds = np.array(all_pred, dtype=np.float64)
    gts = np.array(all_gt, dtype=np.float64)
    errors = preds - gts
    return {
        "loss": total_loss / n,
        "mae_bpm": float(np.mean(np.abs(errors))),
        "rmse_bpm": float(np.sqrt(np.mean(errors ** 2))),
    }


# ── main train function ────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    cfg: ModelingConfig,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train model with early stopping; return (best_model, history).

    Parameters
    ----------
    model:     initialised but untrained model
    train_ds:  WindowDataset for train split (valid windows only)
    val_ds:    WindowDataset for val split
    cfg:       ModelingConfig with all hyper-parameters

    Returns
    -------
    model:    model loaded with best weights
    history:  list of per-epoch dicts
    """
    _set_seed(cfg.seed)
    device = torch.device("cpu")
    model = model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    history: list[dict[str, Any]] = []

    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_model.pt"

    logger.info(
        "Training %s | strategy=%s | train=%d val=%d | max_epochs=%d patience=%d",
        cfg.model_type, cfg.input_strategy,
        len(train_ds), len(val_ds),
        cfg.max_epochs, cfg.patience,
    )

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = _eval_epoch(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0

        val_mae = val_metrics["mae_bpm"]
        scheduler.step(val_mae)

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_mae_bpm": round(val_mae, 4),
            "val_rmse_bpm": round(val_metrics["rmse_bpm"], 4),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 2),
        }
        history.append(row)

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if cfg.progress and (epoch % 10 == 0 or epoch == 1 or improved):
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_mae=%.2f bpm  val_rmse=%.2f bpm  %s",
                epoch, cfg.max_epochs,
                train_loss, val_mae, val_metrics["rmse_bpm"],
                "✓ best" if improved else f"(patience {patience_counter}/{cfg.patience})",
            )

        if patience_counter >= cfg.patience:
            logger.info("Early stopping at epoch %d (best val_mae=%.2f bpm)", epoch, best_val_mae)
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        torch.save(model.state_dict(), best_model_path)

    # save training history CSV
    history_path = ckpt_dir / "training_history.csv"
    _save_history_csv(history, history_path)
    logger.info("Training history saved to %s", history_path)

    return model, history


def _save_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_config(cfg: ModelingConfig, path: Path) -> None:
    """Persist ModelingConfig as JSON (includes normalization params)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Config saved to %s", path)
