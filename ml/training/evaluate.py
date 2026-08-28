"""
ml/training/evaluate.py
========================
SIH 2026 — PS 26143: Sentinel-1 SAR Oil-Spill Detection
Test-set evaluation and visual predictions.

Loads the best saved checkpoint and evaluates on the 13-image test set.
Computes per-image and aggregate Dice / IoU.
Displays prediction panels: SAR channel 0 | SAR channel 1 | GT mask | Pred mask | Overlay.

Usage
-----
python ml/training/evaluate.py \\
    --data-dir  /content/sih26143/prototype \\
    --ckpt      ml/training/checkpoints/best_unet.pth \\
    --n-display 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.training.dataset import SAROilSpillDataset
from ml.training.unet    import UNet
from ml.training.train   import compute_dice_iou

import warnings
import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> Tuple[UNet, dict]:
    """
    Load a saved UNet checkpoint.
    Returns (model, checkpoint_dict).
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first to generate the checkpoint."
        )

    # weights_only=False required: checkpoint dict contains non-tensor args.
    # PyTorch >= 2.4 warns if this is not set explicitly.
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Recreate the model with the same arguments used during training
    saved_args   = ckpt.get("args", {})
    base_features = saved_args.get("base_features", 64)

    model = UNet(in_channels=1, out_channels=1, base_features=base_features)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"Checkpoint loaded from: {ckpt_path}")
    print(f"  Trained for {ckpt.get('epoch', '?')} epochs")
    print(f"  Best val Dice : {ckpt.get('val_dice', '?'):.4f}")
    print(f"  Best val IoU  : {ckpt.get('val_iou',  '?'):.4f}")

    return model, ckpt


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test_set(
    model:      UNet,
    test_ds:    SAROilSpillDataset,
    device:     torch.device,
    threshold:  float = 0.5,
    batch_size: int   = 1,
) -> dict:
    """
    Evaluate on every image in test_ds.
    Returns a dict with per-image metrics and aggregate stats.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model.eval()

    per_image_metrics = []
    all_dice = []
    all_iou  = []

    print(f"\nEvaluating on {len(test_ds)} test images …")
    print(f"{'─'*55}")
    print(f"  {'Filename':30s}  {'Dice':>8}  {'IoU':>8}")
    print(f"{'─'*55}")

    for idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        dice, iou = compute_dice_iou(logits, masks, threshold=threshold)

        fname = test_ds.get_filename(idx)
        per_image_metrics.append({"filename": fname, "dice": dice, "iou": iou})
        all_dice.append(dice)
        all_iou.append(iou)

        print(f"  {fname:30s}  {dice:8.4f}  {iou:8.4f}")

    mean_dice = float(np.mean(all_dice))
    mean_iou  = float(np.mean(all_iou))
    std_dice  = float(np.std(all_dice))
    std_iou   = float(np.std(all_iou))

    print(f"{'─'*55}")
    print(f"  {'MEAN':30s}  {mean_dice:8.4f}  {mean_iou:8.4f}")
    print(f"  {'STD':30s}  {std_dice:8.4f}  {std_iou:8.4f}")
    print(f"{'─'*55}")

    return {
        "per_image":  per_image_metrics,
        "mean_dice":  mean_dice,
        "mean_iou":   mean_iou,
        "std_dice":   std_dice,
        "std_iou":    std_iou,
        "threshold":  threshold,
        "n_images":   len(test_ds),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualise_predictions(
    model:     UNet,
    test_ds:   SAROilSpillDataset,
    device:    torch.device,
    n:         int   = 5,
    threshold: float = 0.5,
) -> None:
    """
    Display n prediction panels.
    Each panel shows 5 sub-plots:
        [0] SAR Channel 0 (VV)
        [1] SAR Channel 1 (VH)
        [2] Ground-truth mask
        [3] Predicted binary mask
        [4] Overlay (pred mask on top of SAR ch0)
    """
    n = min(n, len(test_ds))
    model.eval()

    for idx in range(n):
        image_t, mask_t = test_ds[idx]
        fname = test_ds.get_filename(idx)

        # Add batch dimension for model
        logits  = model(image_t.unsqueeze(0).to(device))   # (1, 1, H, W)
        prob    = torch.sigmoid(logits).squeeze().cpu().numpy()   # (H, W)
        pred    = (prob > threshold).astype(np.float32)          # (H, W) binary

        # Tensors → numpy for plotting
        img_np  = image_t.numpy()   # (2, H, W)
        mask_np = mask_t.squeeze().numpy()  # (H, W)

        ch0 = img_np[0]  # VV polarisation
        ch1 = img_np[1]  # VH polarisation

        # Build overlay: SAR ch0 as greyscale + predicted mask as red tint
        rgb_base = np.stack([ch0, ch0, ch0], axis=-1)            # (H, W, 3)
        overlay  = rgb_base.copy()
        overlay[pred > 0.5, 0] = 1.0   # highlight oil-spill in red
        overlay[pred > 0.5, 1] = 0.2
        overlay[pred > 0.5, 2] = 0.2

        # Plot
        fig = plt.figure(figsize=(20, 4))
        gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

        axes_data = [
            (ch0,      "gray",    f"SAR Ch-0 (VV)\n{fname}"),
            (ch1,      "gray",    "SAR Ch-1 (VH)"),
            (mask_np,  "binary",  "Ground-truth Mask"),
            (pred,     "binary",  f"Predicted Mask\n(thresh={threshold})"),
            (overlay,  None,      "Overlay\n(red = predicted oil)"),
        ]

        for col, (data, cmap, title) in enumerate(axes_data):
            ax = fig.add_subplot(gs[0, col])
            if cmap is None:
                ax.imshow(data, vmin=0, vmax=1)
            else:
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.suptitle(
            f"Test sample {idx + 1}/{n}",
            fontsize=11, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net on the test set and visualise predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="/content/sih26143/prototype",
        help="Root of the prototype dataset.",
    )
    parser.add_argument(
        "--ckpt",
        default="ml/training/checkpoints/best_unet.pth",
        help="Path to the best model checkpoint.",
    )
    parser.add_argument("--image-size",    type=int,   default=512)
    parser.add_argument("--batch-size",    type=int,   default=1)
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Sigmoid probability threshold for binary prediction.")
    parser.add_argument("--n-display",     type=int,   default=5,
                        help="Number of test images to display.")
    parser.add_argument("--base-features", type=int,   default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test set
    test_ds = SAROilSpillDataset(
        image_dir  = Path(args.data_dir) / "test" / "images",
        mask_dir   = Path(args.data_dir) / "test" / "masks",
        image_size = args.image_size,
        augment    = False,
    )

    # Load model
    model, ckpt = load_model(Path(args.ckpt), device)

    # Quantitative evaluation
    results = evaluate_test_set(
        model, test_ds, device,
        threshold  = args.threshold,
        batch_size = args.batch_size,
    )

    # Visual predictions
    visualise_predictions(
        model, test_ds, device,
        n         = args.n_display,
        threshold = args.threshold,
    )

    print(f"\nFinal Test Results:")
    print(f"  Mean Dice : {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
    print(f"  Mean IoU  : {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
