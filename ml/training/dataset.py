"""
ml/training/dataset.py
======================
SIH 2026 — PS 26143: Sentinel-1 SAR Oil-Spill Detection
PyTorch Dataset for 1-channel SAR images + binary segmentation masks.

Key design decisions
--------------------
- Reads TIFF files via rasterio (suppresses NotGeoreferencedWarning).
- Expects images of shape (1, H, W) — single SAR polarisation channel.
- Normalises each channel independently with percentile clipping so that
  the highly variable SAR dB range is mapped to [0, 1] reliably.
- Resizes both image and mask to a configurable target size (default 512x512).
- Keeps masks binary (0 or 1) — does NOT apply sigmoid here.
- Returns torch.Tensor with dtypes float32 (image) and float32 (mask).
- Verifies image/mask filename correspondence at construction time.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Suppress the rasterio "no CRS" warning that fires on these non-georeferenced
# Sentinel-1 patches — they are expected to lack spatial metadata.
import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

try:
    from torchvision.transforms.functional import resize as tv_resize
    from torchvision.transforms import InterpolationMode
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Fixed percentile bounds derived from the Sentinel-1 Sigma0 dB range
# described in the Zenodo metadata (~-48 to +11 dB).
# Using 2nd / 98th percentile clipping rather than hard min/max prevents
# a single bright/dark outlier from collapsing the entire dynamic range.
_SAR_LOW_PERCENTILE  = 2.0
_SAR_HIGH_PERCENTILE = 98.0


def normalise_sar_channel(arr: np.ndarray) -> np.ndarray:
    """
    Map a 2-D float32 SAR channel (H, W) to [0, 1] using percentile clipping.

    Steps:
    1. Compute the 2nd and 98th percentile of non-NaN values.
    2. Clip the array to [p2, p98].
    3. Min-max scale to [0, 1].

    This is robust to outlier pixels (e.g. ships, corner reflectors) and
    handles the wide dB dynamic range of Sentinel-1 SAR imagery.
    """
    arr = arr.astype(np.float32)
    lo = np.nanpercentile(arr, _SAR_LOW_PERCENTILE)
    hi = np.nanpercentile(arr, _SAR_HIGH_PERCENTILE)
    if hi - lo < 1e-6:
        # Degenerate case (constant image) — return zeros
        return np.zeros_like(arr)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# File-pair discovery
# ---------------------------------------------------------------------------

def _find_pairs(
    image_dir: Path,
    mask_dir:  Path,
) -> List[Tuple[Path, Path]]:
    """
    Return sorted list of (image_path, mask_path) tuples.

    Matching strategy:
    - Sort both directories alphabetically.
    - Require the same number of files.
    - Match by position (same sorted index) — which is valid because the
      Zenodo dataset uses the same numeric stem (e.g. 0001.tif) in both
      image and mask directories.
    - Also assert that stems match to catch any accidental mismatch.

    Raises ValueError if the counts differ or any stem pair doesn't match.
    """
    image_files = sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.tiff"))
    mask_files  = sorted(mask_dir.glob("*.tif"))  + sorted(mask_dir.glob("*.tiff"))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No TIFF images found in {image_dir}")
    if len(mask_files) == 0:
        raise FileNotFoundError(f"No TIFF masks found in {mask_dir}")
    if len(image_files) != len(mask_files):
        raise ValueError(
            f"Image/mask count mismatch: {len(image_files)} images vs "
            f"{len(mask_files)} masks in\n  {image_dir}\n  {mask_dir}"
        )

    pairs = []
    for img_path, msk_path in zip(image_files, mask_files):
        if img_path.stem != msk_path.stem:
            raise ValueError(
                f"Filename mismatch: image '{img_path.name}' paired with "
                f"mask '{msk_path.name}'. Stems must match."
            )
        pairs.append((img_path, msk_path))

    return pairs


# ---------------------------------------------------------------------------
# Resize helper (no torchvision dependency required)
# ---------------------------------------------------------------------------

def _resize_image(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resize a (C, H, W) float tensor to (C, size, size) using bilinear
    interpolation via torch.nn.functional.interpolate.
    """
    # interpolate expects (N, C, H, W)
    t = tensor.unsqueeze(0)  # (1, C, H, W)
    t = torch.nn.functional.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    )
    return t.squeeze(0)  # (C, size, size)


def _resize_mask(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resize a (1, H, W) mask tensor to (1, size, size) using nearest-neighbour
    interpolation to preserve binary values exactly (no interpolation artefacts).
    """
    t = tensor.unsqueeze(0)  # (1, 1, H, W)
    t = torch.nn.functional.interpolate(
        t, size=(size, size), mode="nearest"
    )
    return t.squeeze(0)  # (1, size, size)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SAROilSpillDataset(Dataset):
    """
    PyTorch Dataset for 1-channel Sentinel-1 SAR oil-spill segmentation.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing TIFF images with shape (1, H, W).
    mask_dir : str or Path
        Directory containing TIFF masks with shape (1, H, W), values 0 or 1.
    image_size : int
        Both image and mask will be resized to (image_size, image_size).
        Default: 512.
    augment : bool
        If True, apply random horizontal and vertical flips during training.
        Default: False.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir:  str | Path,
        image_size: int = 512,
        augment:    bool = False,
    ) -> None:
        self.image_dir  = Path(image_dir)
        self.mask_dir   = Path(mask_dir)
        self.image_size = image_size
        self.augment    = augment

        # Validate and build paired file list at construction time so that
        # any missing / mismatched files are caught before training starts.
        self.pairs = _find_pairs(self.image_dir, self.mask_dir)

        print(
            f"[SAROilSpillDataset] {len(self.pairs)} pairs loaded from "
            f"{self.image_dir.parent.name}/"
            f"{self.image_dir.name} + {self.mask_dir.name}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.pairs[idx]

        # ── Load image ──────────────────────────────────────────────────────
        # rasterio.read() returns (C, H, W) — exactly what we need.
        with rasterio.open(str(img_path)) as src:
            image = src.read().astype(np.float32)  # (1, H, W) float32

        if image.shape[0] != 1:
            raise ValueError(
                f"Expected 1-channel SAR image, got {image.shape[0]} channels: "
                f"{img_path}"
            )

        # ── Normalise each SAR channel independently ─────────────────────
        # Channel 0: VV polarisation
        image[0] = normalise_sar_channel(image[0])

        # ── Load mask ───────────────────────────────────────────────────────
        with rasterio.open(str(msk_path)) as src:
            mask = src.read().astype(np.float32)  # (1, H, W) float32, values 0/1

        # Binarise — clip to [0, 1] as a safety measure
        mask = np.clip(mask, 0.0, 1.0)

        # ── Convert to tensors ──────────────────────────────────────────────
        image_t = torch.from_numpy(image)  # (1, H, W)
        mask_t  = torch.from_numpy(mask)   # (1, H, W)

        # ── Resize to target size ───────────────────────────────────────────
        image_t = _resize_image(image_t, self.image_size)  # (1, 512, 512)
        mask_t  = _resize_mask(mask_t,  self.image_size)   # (1, 512, 512)

        # ── Augmentation (training only) ────────────────────────────────────
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image_t = torch.flip(image_t, dims=[2])
                mask_t  = torch.flip(mask_t,  dims=[2])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image_t = torch.flip(image_t, dims=[1])
                mask_t  = torch.flip(mask_t,  dims=[1])

        return image_t, mask_t

    def get_filename(self, idx: int) -> str:
        """Return the image filename for a given index (useful for evaluation)."""
        return self.pairs[idx][0].name
