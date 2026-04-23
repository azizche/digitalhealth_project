"""
src/trainer.py — Training and validation loop for the SED pipeline.

Loss: FocalLoss (multiclass) with per-class alpha weights applied per frame.
  FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
  4 classes: 0=bg, 1=b, 2=mb, 3=h.
  Background is down-weighted (class_weight[0] < 1) since it dominates frame counts.

Labels are (B, T) int64 tensors — 0 for background, 1/2/3 for event classes.
All frames carry a valid training signal.

Shape alignment: model output T may differ from label T by ±1 frame due to
CNN integer-stride arithmetic; the shorter dimension is used.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import Config
from src.model import BowelSoundSEDModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal loss (multiclass)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification.
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        pos_weight: Per-class weight for positive examples, shape (C,).
                    Analogous to BCEWithLogitsLoss(pos_weight=...).
        gamma:      Focusing exponent. 0 = standard weighted BCE. 2 is typical.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C) raw (pre-sigmoid) scores.
            targets: (N, C) float32 binary targets in {0.0, 1.0}.
        Returns:
            Scalar mean focal loss.
        """
        probs = torch.sigmoid(logits)                              # (N, C)
        # p_t: probability of the true label per element
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)   # (N, C)
        focal_factor = (1.0 - p_t) ** self.gamma                   # (N, C)

        # Numerically stable per-element BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )                                                           # (N, C)

        if self.pos_weight is not None:
            # α for positives = pos_weight[c], α for negatives = 1
            alpha = self.pos_weight * targets + (1.0 - targets)     # (N, C)
        else:
            alpha = torch.ones_like(targets)

        loss = alpha * focal_factor * bce                           # (N, C)
        return loss.mean()


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def _align_logits_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim the longer of logits/labels to match the shorter time dimension."""
    T = min(logits.size(1), labels.size(1))
    return logits[:, :T, :], labels[:, :T, :]


def _log_pred_counts(
    preds_np: np.ndarray,
    labels_np: np.ndarray,
    class_names: list[str],
    prefix: str,
) -> None:
    """Log per-class predicted frame count vs ground-truth for multi-label sigmoid."""
    n_total = max(preds_np.shape[0], 1)
    parts = [
        f"{name}: {int(preds_np[:, i].sum()):,} pred / {int(labels_np[:, i].sum()):,} gt"
        for i, name in enumerate(class_names)
    ]
    logger.info("%s  [frames=%d]  %s", prefix, n_total, "  |  ".join(parts))


def train_one_epoch(
    model: BowelSoundSEDModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    class_names: list[str],
    log_every: int = 10,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    """
    Run one full training epoch with binary focal loss.

    Args:
        model: BowelSoundSEDModel.
        loader: Training DataLoader; each batch has "input_values" and "labels".
        optimizer: AdamW optimizer.
        scheduler: Linear warmup scheduler.
        criterion: FocalLoss with pos_weight.
        device: CUDA or CPU.
        epoch: Current epoch number (for logging only).
        log_every: Log a progress line every N batches.
        gradient_accumulation_steps: Accumulate gradients over N batches.

    Returns:
        Dict with "loss" (mean per-frame focal loss) and "frame_macro_f1".
    """
    model.train()
    total_loss = 0.0
    total_frames = 0
    optimizer.zero_grad()

    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for step, batch in enumerate(loader):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)       # (B, T, C) float32

        logits = model(input_values)              # (B, T', C)
        logits, labels = _align_logits_labels(logits, labels)

        B, T, C = logits.shape
        loss = criterion(logits.reshape(-1, C), labels.reshape(-1, C))

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            batch_loss = loss.item() * (
                gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1
            )
            total_loss += batch_loss * B * T
            total_frames += B * T

            preds = (logits.sigmoid() > 0.5).cpu().numpy().reshape(-1, C)  # (N, C) bool
            all_labels.append(labels.cpu().numpy().reshape(-1, C))          # (N, C)
            all_preds.append(preds)

        if log_every > 0 and (step + 1) % log_every == 0:
            logger.info(
                "  Epoch %d | step %d/%d | loss %.4f",
                epoch, step + 1, len(loader), batch_loss,
            )

    mean_loss = total_loss / max(total_frames, 1)

    labels_np = np.concatenate(all_labels, axis=0)  # (N, C)
    preds_np = np.concatenate(all_preds, axis=0)    # (N, C)
    f1s = [
        f1_score(labels_np[:, c], preds_np[:, c], zero_division=0)
        for c in range(len(class_names))
    ]
    macro_f1 = float(np.mean(f1s))

    _log_pred_counts(preds_np.astype(int), labels_np.astype(int), class_names, prefix=f"[train epoch {epoch}]")

    return {"loss": mean_loss, "frame_macro_f1": macro_f1}


def validate(
    model: BowelSoundSEDModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    epoch: int = 0,
) -> dict[str, float]:
    """
    Run a validation pass and compute per-class binary frame-level metrics.

    Returns:
        Dict with keys: loss, frame_macro_f1, f1_<class_name> × num_classes.
    """
    model.eval()
    total_loss = 0.0
    total_frames = 0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)       # (B, T, C) float32

            logits = model(input_values)              # (B, T', C)
            logits, labels = _align_logits_labels(logits, labels)

            B, T, C = logits.shape
            loss = criterion(logits.reshape(-1, C), labels.reshape(-1, C))
            total_loss += loss.item() * B * T
            total_frames += B * T

            preds = (logits.sigmoid() > 0.5).cpu().numpy().reshape(-1, C)  # (N, C) bool
            all_labels.append(labels.cpu().numpy().reshape(-1, C))          # (N, C)
            all_preds.append(preds)

    mean_loss = total_loss / max(total_frames, 1)

    labels_np = np.concatenate(all_labels, axis=0)  # (N, C)
    preds_np = np.concatenate(all_preds, axis=0)    # (N, C)
    f1s = [
        f1_score(labels_np[:, c], preds_np[:, c], zero_division=0)
        for c in range(len(class_names))
    ]
    macro_f1 = float(np.mean(f1s))

    _log_pred_counts(preds_np.astype(int), labels_np.astype(int), class_names, prefix="[val]")

    result: dict[str, float] = {"loss": mean_loss, "frame_macro_f1": macro_f1}
    for i, cls in enumerate(class_names):
        result[f"f1_{cls}"] = float(f1s[i])

    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: BowelSoundSEDModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    output_dir: Path,
    filename: str = "best_model.pt",
) -> None:
    """Save model checkpoint with training state and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info(
        "Checkpoint saved → %s  (val frame_macro_f1=%.4f)",
        path, metrics.get("frame_macro_f1", 0.0),
    )


def load_checkpoint(
    model: BowelSoundSEDModel,
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Load a checkpoint into `model` (and optionally `optimizer`).

    Returns:
        The raw checkpoint dict.
    """
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        "Loaded checkpoint from %s (epoch %d)",
        checkpoint_path, ckpt.get("epoch", -1),
    )
    return ckpt


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when val frame macro F1 does not improve for `patience` epochs.
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = -float("inf")
        self._counter = 0

    @property
    def best(self) -> float:
        return self._best

    def step(self, metric: float) -> bool:
        """
        Update state with the latest metric.

        Returns:
            True if training should stop.
        """
        if metric > self._best + self.min_delta:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d / %d epochs (best=%.4f)",
                self._counter, self.patience, self._best,
            )
        return self._counter >= self.patience


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(
    model: BowelSoundSEDModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> dict[str, list]:
    """
    Full SED training loop with early stopping.

    Args:
        model: BowelSoundSEDModel to train.
        train_loader: DataLoader for the training chunks.
        val_loader: DataLoader for the validation chunks.
        cfg: Full Config.
        device: CUDA or CPU.

    Returns:
        History dict with per-epoch metric lists.
    """
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = cfg.data.class_names

    pos_weight = torch.tensor(cfg.training.pos_weight, dtype=torch.float32).to(device)
    criterion = FocalLoss(pos_weight=pos_weight, gamma=cfg.training.focal_gamma)
    logger.info(
        "FocalLoss(pos_weight=%s, gamma=%.1f)",
        cfg.training.pos_weight, cfg.training.focal_gamma,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = len(train_loader) * cfg.training.num_epochs
    warmup_steps = int(cfg.training.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    early_stopping = EarlyStopping(patience=cfg.training.early_stopping_patience)

    history: dict[str, list] = defaultdict(list)
    csv_path = output_dir / "metrics.csv"
    csv_headers = [
        "epoch", "train_loss", "train_f1",
        "val_loss", "val_frame_macro_f1",
    ] + [f"val_f1_{c}" for c in class_names]

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_headers).writeheader()

    best_val_f1 = -float("inf")

    for epoch in range(cfg.training.num_epochs):
        logger.info("=== Epoch %d / %d ===", epoch + 1, cfg.training.num_epochs)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            epoch=epoch + 1,
            class_names=class_names,
            log_every=cfg.training.log_every_n_steps,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        )
        val_metrics = validate(model, val_loader, criterion, device, class_names, epoch=epoch + 1)

        logger.info(
            "  train loss=%.4f  F1=%.4f | val loss=%.4f  F1=%.4f",
            train_metrics["loss"], train_metrics["frame_macro_f1"],
            val_metrics["loss"], val_metrics["frame_macro_f1"],
        )
        for cls in class_names:
            logger.info("    val_f1_%s = %.4f", cls, val_metrics.get(f"f1_{cls}", 0.0))

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["frame_macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_frame_macro_f1"].append(val_metrics["frame_macro_f1"])
        for cls in class_names:
            history[f"val_f1_{cls}"].append(val_metrics.get(f"f1_{cls}", 0.0))

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["loss"], 6),
            "train_f1": round(train_metrics["frame_macro_f1"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_frame_macro_f1": round(val_metrics["frame_macro_f1"], 6),
        }
        for cls in class_names:
            row[f"val_f1_{cls}"] = round(val_metrics.get(f"f1_{cls}", 0.0), 6)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_headers).writerow(row)

        current_f1 = val_metrics["frame_macro_f1"]
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, output_dir, "best_model.pt")

        save_checkpoint(model, optimizer, epoch + 1, val_metrics, output_dir, "last_model.pt")

        if early_stopping.step(current_f1):
            logger.info("Early stopping triggered after epoch %d.", epoch + 1)
            break

    logger.info(
        "Training complete. Best val frame macro F1: %.4f — checkpoint: %s",
        early_stopping.best, output_dir / "best_model.pt",
    )

    with open(output_dir / "history.json", "w") as f:
        json.dump(dict(history), f, indent=2)

    return dict(history)
