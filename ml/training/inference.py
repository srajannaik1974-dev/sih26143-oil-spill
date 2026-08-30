"""
ml/training/inference.py
========================
SIH 2026 — PS 26143: Sentinel-1 SAR Oil-Spill Detection
Inference module — predict oil-spill masks from new Sentinel-1 TIFF files.

This module is intentionally separated from training so that it can be
imported by a future backend API or Colab demo without pulling in
training dependencies.

Public API
----------
    from ml.training.inference import OilSpillPredictor

    predictor = OilSpillPredictor(ckpt_path="ml/training/checkpoints/best_unet.pth")
    binary_mask, prob_map = predictor.predict("/path/to/new_sar_image.tif")
    predictor.visualise("/path/to/new_sar_image.tif")

Input requirements
------------------
- A Sentinel-1 SAR TIFF file.
- Must contain exactly 1 band (VV channel), shape (1, H, W).
- dtype: float32, values typically in the range -48 to +11 dB.
- The model was trained on 512×512 patches — the predictor automatically
  resizes the input before inference and returns the mask at the original
  image resolution.

Output
------
binary_mask : np.ndarray, shape (H, W), dtype uint8, values {0, 1}
    1 = oil spill predicted, 0 = background.

prob_map : np.ndarray, shape (H, W), dtype float32, values [0, 1]
    Sigmoid probability map (confidence) — useful for uncertainty analysis.
"""

from __future__ import annotations

import sys
import warnings
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.training.dataset import normalise_sar_channel
from ml.training.unet    import UNet


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class OilSpillPredictor:
    """
    Loads a trained U-Net checkpoint and predicts oil-spill masks from
    new Sentinel-1 SAR TIFF images.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the best model checkpoint (best_unet.pth).
    image_size : int
        Size used during training (default 512). The input will be resized
        to this before inference and the output will be upsampled back to
        the original resolution.
    threshold : float
        Sigmoid probability threshold for binary classification.
        Default: 0.5.
    device : str or torch.device or None
        "cuda", "cpu", or None for auto-detection.
    """

    def __init__(
        self,
        ckpt_path:  Union[str, Path],
        image_size: int   = 512,
        threshold:  float = 0.5,
        device:     Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.image_size = image_size
        self.threshold  = threshold

        # Device auto-selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model(Path(ckpt_path))
        print(f"[OilSpillPredictor] Ready on device={self.device}, "
              f"threshold={self.threshold}")

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_model(self, ckpt_path: Path) -> UNet:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Train the model first with ml/training/train.py"
            )
        # weights_only=False is required because the checkpoint dict contains
        # non-tensor Python objects (the args namespace). PyTorch >= 2.4 will
        # raise a FutureWarning if this is not set explicitly.
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        saved_args    = ckpt.get("args", {})
        base_features = saved_args.get("base_features", 64)

        model = UNet(in_channels=1, out_channels=1, base_features=base_features)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        return model

    def _load_tiff(self, tiff_path: Path) -> Tuple[np.ndarray, tuple]:
        """
        Load a 1-channel SAR TIFF.
        Returns (array, original_shape) where array has shape (1, H, W).
        """
        with rasterio.open(str(tiff_path)) as src:
            arr = src.read().astype(np.float32)  # (C, H, W)

        if arr.ndim != 3 or arr.shape[0] != 1:
            raise ValueError(
                f"Expected a 1-channel TIFF, got shape {arr.shape}: {tiff_path}\n"
                "This model only accepts Sentinel-1 VV SAR imagery."
            )
        return arr, arr.shape[1:]  # (1, H, W), (H, W)

    def _preprocess(self, arr: np.ndarray) -> torch.Tensor:
        """
        Normalise and resize a (1, H, W) array to (1, 1, size, size) tensor.
        """
        # Normalise each channel independently
        norm = arr.copy()
        norm[0] = normalise_sar_channel(norm[0])

        # Convert to tensor and add batch dimension
        t = torch.from_numpy(norm).unsqueeze(0)  # (1, 1, H, W)

        # Resize to training size
        t = torch.nn.functional.interpolate(
            t, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        return t  # (1, 1, size, size)

    def _postprocess(
        self,
        logits:        torch.Tensor,
        original_shape: tuple,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert logits → probability map and binary mask at original resolution.

        1. Apply sigmoid to get probabilities (0–1).
        2. Upsample to original image resolution using bilinear interpolation.
        3. Threshold to produce binary mask.

        Returns (binary_mask, prob_map) both as numpy arrays of shape (H, W).
        """
        # Probability map at training resolution
        prob = torch.sigmoid(logits)  # (1, 1, size, size)

        # Upsample back to original resolution
        prob_full = torch.nn.functional.interpolate(
            prob,
            size=original_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()  # (H, W) float32

        # Binary mask
        binary_mask = (prob_full > self.threshold).astype(np.uint8)

        return binary_mask, prob_full.astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        tiff_path: str | Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict oil-spill mask for a new Sentinel-1 TIFF image.

        Parameters
        ----------
        tiff_path : str or Path
            Path to a 1-channel Sentinel-1 SAR TIFF file.

        Returns
        -------
        binary_mask : np.ndarray  shape (H, W)  dtype uint8  values {0, 1}
            1 = oil spill predicted.
        prob_map : np.ndarray  shape (H, W)  dtype float32  values [0, 1]
            Sigmoid probability (confidence map).
        """
        tiff_path = Path(tiff_path)
        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF not found: {tiff_path}")

        # Load
        arr, original_shape = self._load_tiff(tiff_path)

        # Preprocess
        tensor = self._preprocess(arr).to(self.device)  # (1, 1, size, size)

        # Inference
        logits = self.model(tensor)  # (1, 1, size, size) raw logits

        # Postprocess
        binary_mask, prob_map = self._postprocess(logits, original_shape)

        spill_pixels = int(binary_mask.sum())
        total_pixels = binary_mask.size
        print(f"[predict] {tiff_path.name}  "
              f"oil-spill pixels={spill_pixels}  "
              f"coverage={100*spill_pixels/total_pixels:.2f}%")

        return binary_mask, prob_map

    def get_spill_location(
        self,
        tiff_path: str | Path,
        binary_mask: np.ndarray,
        prob_map: np.ndarray,
        original_filename: str = None,
    ) -> dict:
        """
        Calculate geographic location, area and confidence
        of the detected oil spill.
        """
        tiff_path = Path(tiff_path)

        # Synthetic timestamp for SIH prototype only.
        # Original Sentinel-1 acquisition time is unavailable
        # in the processed TIFF dataset.
        synthetic_date = None
        synthetic_timestamp = None
        
        filename_to_parse = original_filename if original_filename else tiff_path.name
        match = re.search(r'(\d{4})_?(\d{2})_?(\d{2})', filename_to_parse)
        if match:
            year, month, day = match.groups()
            synthetic_date = f"{year}-{month}-{day}"
            synthetic_timestamp = f"{synthetic_date}T14:30:00Z"

        with rasterio.open(str(tiff_path)) as src:

            if src.crs is None:
                raise ValueError(
                    f"TIFF has no CRS/geospatial information: {tiff_path}"
                )

            rows, cols = np.where(binary_mask == 1)

            if len(rows) == 0:
                return {
                    "detected": False,
                    "latitude": None,
                    "longitude": None,
                    "area_km2": 0.0,
                    "area_percent": 0.0,
                    "confidence": 0.0,
                    "date": synthetic_date,
                    "timestamp": synthetic_timestamp,
                    "timestamp_type": "synthetic",
                }

            centroid_row = float(rows.mean())
            centroid_col = float(cols.mean())

            x, y = src.xy(centroid_row, centroid_col)

            from rasterio.warp import transform

            longitude, latitude = transform(
                src.crs,
                "EPSG:4326",
                [x],
                [y],
            )

            pixel_width = abs(src.transform.a)
            pixel_height = abs(src.transform.e)
            pixel_area_m2 = pixel_width * pixel_height

            spill_pixels = len(rows)

            area_m2 = spill_pixels * pixel_area_m2
            area_km2 = area_m2 / 1_000_000

            area_percent = (
                100.0 * spill_pixels / binary_mask.size
            )

            confidence = float(
                prob_map[binary_mask == 1].mean()
            )

            return {
                "detected": True,
                "latitude": float(latitude[0]),
                "longitude": float(longitude[0]),
                "area_km2": float(area_km2),
                "area_percent": float(area_percent),
                "confidence": confidence,
                "date": synthetic_date,
                "timestamp": synthetic_timestamp,
                "timestamp_type": "synthetic",
            }

    def visualise(
        self,
        tiff_path: str | Path,
        figsize:   tuple = (20, 5),
    ) -> None:
        """
        Run inference and display a 3-panel figure:
            [0] SAR Ch-0 (VV)
            [1] Confidence / probability map
            [2] Binary prediction + colour overlay
        """
        tiff_path = Path(tiff_path)
        arr, original_shape = self._load_tiff(tiff_path)

        binary_mask, prob_map = self.predict(tiff_path)

        # Normalised channel for display only (do NOT use for model input again)
        ch0_disp = normalise_sar_channel(arr[0])

        # Overlay: ch0 greyscale + red for predicted oil
        overlay = np.stack([ch0_disp, ch0_disp, ch0_disp], axis=-1)
        overlay[binary_mask == 1, 0] = 1.0
        overlay[binary_mask == 1, 1] = 0.15
        overlay[binary_mask == 1, 2] = 0.15

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(ch0_disp, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"SAR Ch-0 (VV)\n{tiff_path.name}", fontsize=9)

        im = axes[1].imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[1].set_title(f"Probability Map\n(threshold={self.threshold})", fontsize=9)
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(overlay, vmin=0, vmax=1)
        axes[2].set_title("Predicted Oil Spill\n(red overlay)", fontsize=9)

        for ax in axes:
            ax.axis("off")

        spill_pct = 100 * binary_mask.sum() / binary_mask.size
        fig.suptitle(
            f"Oil-Spill Prediction  |  Coverage: {spill_pct:.2f}%",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# CLI — quick inference on a single file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run oil-spill inference on a single Sentinel-1 TIFF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tiff_path",
        help="Path to the input SAR TIFF file.",
    )
    parser.add_argument(
        "--ckpt",
        default="ml/training/checkpoints/best_unet.pth",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--image-size", type=int,   default=512)
    parser.add_argument("--save-mask",  type=str,   default=None,
                        help="If provided, save the binary mask as a .npy file.")
    args = parser.parse_args()

    predictor = OilSpillPredictor(
        ckpt_path  = args.ckpt,
        image_size = args.image_size,
        threshold  = args.threshold,
    )

    binary_mask, prob_map = predictor.predict(args.tiff_path)
    predictor.visualise(args.tiff_path)

    if args.save_mask:
        np.save(args.save_mask, binary_mask)
        print(f"Binary mask saved to: {args.save_mask}")
