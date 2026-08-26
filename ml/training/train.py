"""
ml/training/train.py
====================
SIH 2026 — PS 26143: Sentinel-1 SAR Oil-Spill Detection
Complete training loop for the U-Net segmentation model.

Loss function
-------------
Combined loss = BCEWithLogitsLoss  +  Dice Loss
- BCEWithLogitsLoss: pixel-wise binary cross-entropy (numerically stable).
- Dice Loss: directly optimises overlap ratio — important for class-imbalanced
  SAR segmentation where oil-spill pixels are rare (many background pixels).

Usage (Google Colab / terminal)
--------------------------------
# Minimal (uses defaults):
    python ml/training/train.py \\
        --data-dir /content/sih26143/prototype

# Full example:
    python ml/training/train.py \\
        --data-dir    /content/sih26143/prototype \\
        --ckpt-dir    ml/training/checkpoints \\
        --epochs      50 \\
        --batch-size  4 \\
        --lr          1e-4 \\
        --image-size  512 \\
        --patience    10 \\
        --seed        42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Allow running from Colab as:  python ml/training/train.py
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.training.dataset import SAROilSpillDataset
from ml.training.unet    import UNet


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN (slight speed penalty — acceptable for SIH prototype)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice Loss for binary segmentation.

    Applies sigmoid to logits before computing the Dice coefficient so that
    the gradient flows correctly through the probability values.

    smooth: Laplace smoothing constant — prevents division by zero on empty
            masks (oil-free scenes from Part II of the dataset).

    Returns scalar loss in [0, 1] where 0 = perfect overlap.
    """
    probs   = torch.sigmoid(logits)            # convert logits → probabilities
    probs   = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    dice_coeff   = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1.0 - dice_coeff


class CombinedLoss(nn.Module):
    """
    BCE-with-Logits  +  Soft Dice Loss, weighted equally.

    BCE captures pixel-level accuracy.
    Dice captures region-level overlap (class imbalance robust).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, bce_loss, dice_loss_value) for logging.
        """
        bce_l  = self.bce(logits, targets)
        dice_l = dice_loss(logits, targets)
        total  = self.bce_weight * bce_l + self.dice_weight * dice_l
        return total, bce_l, dice_l


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dice_iou(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute Dice score and IoU from raw logits and binary targets.

    1. Apply sigmoid to convert logits → probabilities.
    2. Threshold at 0.5 to produce binary predictions.
    3. Compute Dice and IoU over the entire batch.

    Returns (dice, iou) in [0, 1].
    """
    probs   = torch.sigmoid(logits)
    preds   = (probs > threshold).float()
    targets = targets.float()

    preds_f   = preds.view(-1)
    targets_f = targets.view(-1)

    intersection = (preds_f * targets_f).sum().item()
    union        = (preds_f + targets_f).sum().item() - intersection
    total_pred   = preds_f.sum().item()
    total_tgt    = targets_f.sum().item()

    dice = (2.0 * intersection + 1e-6) / (total_pred + total_tgt + 1e-6)
    iou  = (intersection + 1e-6) / (union + 1e-6)

    return dice, iou


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> dict:
    """
    Run one full training epoch and return a metrics dict.
    """
    model.train()
    total_loss = dice_sum = iou_sum = bce_sum = dice_loss_sum = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)  # (B, 2, H, W)
        masks  = masks.to(device,  non_blocking=True)  # (B, 1, H, W)

        # Forward pass — model outputs raw logits
        logits = model(images)

        # Compute combined loss
        loss, bce_l, dice_l = criterion(logits, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — helps stabilise early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics (detached from computation graph)
        with torch.no_grad():
            dice, iou = compute_dice_iou(logits.detach(), masks)

        total_loss    += loss.item()
        bce_sum       += bce_l.item()
        dice_loss_sum += dice_l.item()
        dice_sum      += dice
        iou_sum       += iou

        # Print progress every 10 batches or on the last batch
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{n_batches}]"
                f"  loss={loss.item():.4f}"
                f"  bce={bce_l.item():.4f}"
                f"  dice_loss={dice_l.item():.4f}"
                f"  dice={dice:.4f}"
                f"  iou={iou:.4f}"
            )

    return {
        "loss":      total_loss    / n_batches,
        "bce":       bce_sum       / n_batches,
        "dice_loss": dice_loss_sum / n_batches,
        "dice":      dice_sum      / n_batches,
        "iou":       iou_sum       / n_batches,
    }


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: CombinedLoss,
    device:    torch.device,
) -> dict:
    """
    Run validation and return a metrics dict.
    Model is in eval() mode — BatchNorm uses running statistics.
    """
    model.eval()
    total_loss = dice_sum = iou_sum = 0.0
    n_batches  = len(loader)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        loss, _, _ = criterion(logits, masks)

        dice, iou = compute_dice_iou(logits, masks)

        total_loss += loss.item()
        dice_sum   += dice
        iou_sum    += iou

    return {
        "loss": total_loss / n_batches,
        "dice": dice_sum   / n_batches,
        "iou":  iou_sum    / n_batches,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """
    Full training pipeline:
    1. Set up device, seed, directories.
    2. Build datasets and data loaders.
    3. Build model, optimiser, loss.
    4. Train with early stopping.
    5. Save best checkpoint.
    """
    set_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SIH 2026 — PS 26143: SAR Oil-Spill U-Net Training")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Seed      : {args.seed}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print(f"  Image size: {args.image_size}")
    print(f"  Patience  : {args.patience}")
    print(f"{'='*60}\n")

    # ── Directories ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_unet.pth"

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Loading datasets ...")
    train_ds = SAROilSpillDataset(
        image_dir  = data_dir / "train" / "images",
        mask_dir   = data_dir / "train" / "masks",
        image_size = args.image_size,
        augment    = True,   # flip augmentation for training
    )
    val_ds = SAROilSpillDataset(
        image_dir  = data_dir / "val" / "images",
        mask_dir   = data_dir / "val" / "masks",
        image_size = args.image_size,
        augment    = False,  # no augmentation for validation
    )

    # ── DataLoaders ──────────────────────────────────────────────────────────
    # num_workers=2 is a safe default in Colab.
    # pin_memory=True accelerates host-to-GPU transfers when using CUDA.
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = device.type == "cuda",
        drop_last   = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = device.type == "cuda",
    )

    print(f"\n  Train samples : {len(train_ds)}")
    print(f"  Val samples   : {len(val_ds)}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = UNet(
        in_channels   = 2,
        out_channels  = 1,
        base_features = args.base_features,
    ).to(device)
    print(f"  Model parameters: {model.count_parameters():,}\n")

    # ── Optimiser + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # ReduceLROnPlateau: halve LR if val Dice doesn't improve for 5 epochs.
    # NOTE: verbose=True was removed in PyTorch >= 2.2 — LR change is logged
    # manually below by comparing LR before and after scheduler.step().
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_dice  = 0.0
    epochs_no_impv = 0
    history        = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch}/{args.epochs}")
        print(f"{'─'*60}")

        # --- Train ---
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # --- Validate ---
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  [Epoch {epoch}] "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"train_dice={train_metrics['dice']:.4f}  "
              f"train_iou={train_metrics['iou']:.4f}")
        print(f"  [Epoch {epoch}] "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_dice={val_metrics['dice']:.4f}  "
              f"val_iou={val_metrics['iou']:.4f}  "
              f"lr={current_lr:.2e}  "
              f"time={elapsed:.1f}s")

        # Step the LR scheduler based on validation Dice.
        # Capture LR before and after so we can log any reduction manually
        # (verbose=True was removed from ReduceLROnPlateau in PyTorch >= 2.2).
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["dice"])
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(
                f"  *** LR reduced: {lr_before:.2e} → {lr_after:.2e} "
                f"(val_dice has not improved for 5 epochs) ***"
            )

        # Record history
        history.append({
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou":  train_metrics["iou"],
            "val_loss":   val_metrics["loss"],
            "val_dice":   val_metrics["dice"],
            "val_iou":    val_metrics["iou"],
        })

        # --- Save best checkpoint ---
        if val_metrics["dice"] > best_val_dice:
            best_val_dice  = val_metrics["dice"]
            epochs_no_impv = 0

            torch.save(
                {
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_dice":       best_val_dice,
                    "val_iou":        val_metrics["iou"],
                    "args":           vars(args),
                },
                best_ckpt,
            )
            print(f"  ✓ New best model saved  →  val_dice={best_val_dice:.4f}  ({best_ckpt})")
        else:
            epochs_no_impv += 1
            print(f"  No improvement ({epochs_no_impv}/{args.patience})")

        # --- Early stopping ---
        if epochs_no_impv >= args.patience:
            print(f"\n  Early stopping triggered after {epoch} epochs "
                  f"(no val_dice improvement for {args.patience} epochs).")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val Dice : {best_val_dice:.4f}")
    print(f"  Best model    : {best_ckpt}")
    print(f"{'='*60}\n")

    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train U-Net for SAR oil-spill segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/content/sih26143/prototype",
        help="Root of the prototype dataset (contains train/ val/ test/).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="ml/training/checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=4,
                        help="Conservative default — increase if you have ≥16 GB VRAM.")
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--image-size",    type=int,   default=512)
    parser.add_argument("--base-features", type=int,   default=64,
                        help="U-Net base feature count. Use 32 to reduce VRAM.")
    parser.add_argument("--patience",      type=int,   default=10,
                        help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--seed",          type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    history = train(args)
