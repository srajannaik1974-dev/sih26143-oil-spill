"""
dataset_inspector.py
====================
Modular, Colab-compatible Sentinel-1 SAR oil-spill dataset inspector.

Usage
-----
from ml.preprocessing.dataset_inspector import DatasetInspector

inspector = DatasetInspector("/path/to/ml/dataset")
inspector.run()

All public methods can also be called individually:
    inspector.detect_structure()
    inspector.list_files()
    inspector.pair_images_and_masks()
    inspector.inspect_samples(n=4)
    inspector.print_summary()
"""

import os
import sys
import glob
import json
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Optional heavy imports — rasterio is preferred but we fall back gracefully
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn(
        "rasterio not found. GeoTIFF support will be limited. "
        "Install with: pip install rasterio",
        stacklevel=2,
    )

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All extensions considered as potential image or mask files
RASTER_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg",
    ".npy", ".npz", ".img", ".hdr",
}

# Common keyword patterns that suggest a file is a MASK / LABEL / ANNOTATION
MASK_KEYWORDS = {
    "mask", "label", "annotation", "gt", "ground_truth",
    "groundtruth", "seg", "segmentation", "class",
}

IMAGE_KEYWORDS = {
    "image", "img", "sar", "sentinel", "patch", "scene", "input",
}

SPLIT_DIRS = {"train", "val", "validation", "test", "eval"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _guess_role(path: Path) -> str:
    """Return 'mask', 'image', or 'unknown' based on path heuristics."""
    parts = {p.lower() for p in path.parts}
    stem = path.stem.lower()

    # Check parent directories first
    for kw in MASK_KEYWORDS:
        if kw in parts:
            return "mask"
    for kw in IMAGE_KEYWORDS:
        if kw in parts:
            return "image"

    # Check filename stem
    for kw in MASK_KEYWORDS:
        if kw in stem:
            return "mask"
    for kw in IMAGE_KEYWORDS:
        if kw in stem:
            return "image"

    return "unknown"


def _load_array(path: Path) -> Optional[np.ndarray]:
    """
    Load a raster file as a NumPy array.
    Returns None if loading fails.
    Shape: (H, W) for single-band or (H, W, C) for multi-band.
    """
    ext = path.suffix.lower()

    # ---- NumPy native ----
    if ext == ".npy":
        try:
            return np.load(str(path), allow_pickle=False)
        except Exception:
            return None

    if ext == ".npz":
        try:
            data = np.load(str(path), allow_pickle=False)
            key = list(data.keys())[0]
            return data[key]
        except Exception:
            return None

    # ---- GeoTIFF via rasterio ----
    if ext in {".tif", ".tiff", ".img", ".hdr"} and RASTERIO_AVAILABLE:
        try:
            with rasterio.open(str(path)) as src:
                arr = src.read()  # (C, H, W)
                if arr.shape[0] == 1:
                    return arr[0]  # (H, W)
                return np.moveaxis(arr, 0, -1)  # (H, W, C)
        except Exception:
            pass  # fall through to PIL

    # ---- PIL fallback ----
    if PIL_AVAILABLE:
        try:
            img = PILImage.open(str(path))
            return np.array(img)
        except Exception:
            return None

    return None


def _array_info(arr: np.ndarray) -> Dict:
    """Return a dict of shape / dtype / channel info."""
    if arr is None:
        return {}
    shape = arr.shape
    if arr.ndim == 2:
        h, w, c = shape[0], shape[1], 1
    elif arr.ndim == 3:
        h, w, c = shape[0], shape[1], shape[2]
    else:
        h, w, c = None, None, None

    info = {
        "height": h,
        "width": w,
        "channels": c,
        "dtype": str(arr.dtype),
        "shape": shape,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }
    return info


def _normalise_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Safely normalise any dtype array to float32 [0, 1] for matplotlib display.
    Handles SAR float data, uint16, int32, etc.
    """
    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    if hi == lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _display_2d(arr: np.ndarray) -> np.ndarray:
    """Return a 2-D (H, W) float32 array suitable for imshow (grayscale)."""
    if arr.ndim == 3:
        arr = arr[..., 0]  # show first channel for multi-band
    return _normalise_for_display(arr)


def _display_rgb(arr: np.ndarray) -> np.ndarray:
    """Return a (H, W, 3) float32 array for RGB imshow."""
    if arr.ndim == 2:
        disp = _normalise_for_display(arr)
        return np.stack([disp, disp, disp], axis=-1)
    if arr.shape[-1] >= 3:
        return _normalise_for_display(arr[..., :3])
    # 2-channel or single-channel stored as 3D
    disp = _normalise_for_display(arr[..., 0])
    return np.stack([disp, disp, disp], axis=-1)


# ---------------------------------------------------------------------------
# Main inspector class
# ---------------------------------------------------------------------------

class DatasetInspector:
    """
    Inspects an on-disk Sentinel-1 SAR oil-spill dataset without any
    assumptions about channel count, dtype, or mask encoding.

    Parameters
    ----------
    dataset_root : str or Path
        Root directory that contains the raw dataset (i.e. `ml/dataset/`).
    verbose : bool
        Print progress messages.
    """

    def __init__(self, dataset_root: str | Path, verbose: bool = True):
        self.root = Path(dataset_root).expanduser().resolve()
        self.verbose = verbose

        # Results populated by each inspection step
        self.all_files: List[Path] = []
        self.image_files: List[Path] = []
        self.mask_files: List[Path] = []
        self.unknown_files: List[Path] = []
        self.structure: Dict = {}

        # Paired results
        self.pairs: List[Tuple[Path, Optional[Path]]] = []  # (img, mask_or_None)
        self.unmatched_masks: List[Path] = []

        # Per-file metadata
        self.file_metadata: Dict[str, Dict] = {}

        # Aggregate summary
        self.summary: Dict = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, n_samples: int = 4) -> None:
        """Run the full inspection pipeline."""
        self._check_root()
        self.detect_structure()
        self.list_files()
        self.pair_images_and_masks()
        self.inspect_samples(n=n_samples)
        self.report_missing_pairs()
        self.print_summary()

    def detect_structure(self) -> Dict:
        """
        Walk the dataset root and record the directory tree.
        Returns a nested dict representing the folder hierarchy.
        """
        self._check_root()
        self._log("── Detecting dataset directory structure …")

        tree = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel = Path(dirpath).relative_to(self.root)
            node = tree
            for part in rel.parts:
                node = node.setdefault(part, {})
            node["__files__"] = filenames
            dirnames[:] = sorted(dirnames)  # deterministic traversal

        self.structure = tree
        self._print_tree(self.root, prefix="")
        return tree

    def list_files(self) -> Dict[str, List[Path]]:
        """
        Enumerate ALL raster files, classify them as image / mask / unknown.
        Returns a dict with keys 'images', 'masks', 'unknown'.
        """
        self._check_root()
        self._log("\n── Listing all raster files …")

        all_rasters = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in RASTER_EXTENSIONS
        )
        self.all_files = all_rasters

        images, masks, unknown = [], [], []
        for p in all_rasters:
            role = _guess_role(p)
            if role == "image":
                images.append(p)
            elif role == "mask":
                masks.append(p)
            else:
                unknown.append(p)

        self.image_files = images
        self.mask_files = masks
        self.unknown_files = unknown

        self._log(f"   Total raster files : {len(all_rasters)}")
        self._log(f"   Image files        : {len(images)}")
        self._log(f"   Mask / label files : {len(masks)}")
        self._log(f"   Unclassified files : {len(unknown)}")

        if unknown:
            self._log("   Unclassified files (first 10):")
            for p in unknown[:10]:
                self._log(f"     {p.relative_to(self.root)}")

        return {"images": images, "masks": masks, "unknown": unknown}

    def pair_images_and_masks(self) -> List[Tuple[Path, Optional[Path]]]:
        """
        Match each image file with its corresponding mask by filename stem.
        Handles mismatched prefixes (e.g. 'image_001' ↔ 'mask_001').

        Returns list of (image_path, mask_path_or_None).
        """
        self._check_root()
        self._log("\n── Pairing images with masks …")

        # Build a lookup: cleaned stem → mask path
        # "cleaned stem" strips known mask/image prefixes so '0001' matches
        def _clean_stem(p: Path) -> str:
            s = p.stem.lower()
            for kw in MASK_KEYWORDS | IMAGE_KEYWORDS:
                s = s.replace(kw, "")
            return s.strip("_- ")

        mask_by_stem: Dict[str, Path] = {}
        for m in self.mask_files:
            mask_by_stem[_clean_stem(m)] = m

        # Also map exact stems for datasets with identical names
        mask_by_exact: Dict[str, Path] = {m.stem: m for m in self.mask_files}

        pairs = []
        unmatched_masks = set(self.mask_files)

        for img in self.image_files:
            # Try exact match first
            mask = mask_by_exact.get(img.stem)
            if mask is None:
                mask = mask_by_stem.get(_clean_stem(img))
            if mask is not None:
                unmatched_masks.discard(mask)
            pairs.append((img, mask))

        # Images in unknown category may also have corresponding masks
        for unk in self.unknown_files:
            mask = mask_by_exact.get(unk.stem)
            if mask is None:
                mask = mask_by_stem.get(_clean_stem(unk))
            if mask is not None:
                unmatched_masks.discard(mask)
                # Treat these as images
                pairs.append((unk, mask))

        self.pairs = pairs
        self.unmatched_masks = list(unmatched_masks)

        matched = sum(1 for _, m in pairs if m is not None)
        self._log(f"   Total image entries   : {len(pairs)}")
        self._log(f"   Matched pairs         : {matched}")
        self._log(f"   Images without mask   : {len(pairs) - matched}")
        self._log(f"   Masks without image   : {len(unmatched_masks)}")

        return pairs

    def inspect_file(self, path: Path) -> Dict:
        """
        Load a single raster file and return its metadata dict.
        Caches results to avoid redundant I/O.
        """
        key = str(path)
        if key in self.file_metadata:
            return self.file_metadata[key]

        meta = {"path": str(path), "relative_path": str(path.relative_to(self.root))}
        arr = _load_array(path)
        if arr is None:
            meta["error"] = "Could not load file"
        else:
            meta.update(_array_info(arr))
        self.file_metadata[key] = meta
        return meta

    def inspect_samples(self, n: int = 4) -> None:
        """
        Load and display the first *n* image–mask pairs.
        For each pair shows: raw image | normalised image | mask | overlay.
        """
        self._check_root()
        self._log(f"\n── Inspecting up to {n} sample pairs …")

        if not self.pairs:
            self.pair_images_and_masks()

        sample_pairs = [p for p in self.pairs if p[1] is not None][:n]
        image_only = [p for p in self.pairs if p[1] is None][:max(0, n - len(sample_pairs))]
        samples = sample_pairs + image_only

        if not samples:
            self._log("   No files found to display.")
            return

        for idx, (img_path, mask_path) in enumerate(samples):
            self._log(f"\n   Sample {idx + 1}: {img_path.relative_to(self.root)}")

            img_arr = _load_array(img_path)
            mask_arr = _load_array(mask_path) if mask_path else None

            if img_arr is None:
                self._log(f"   ⚠ Could not load image: {img_path.name}")
                continue

            img_meta = _array_info(img_arr)
            self._log(f"     Image  shape={img_meta['shape']}  "
                      f"dtype={img_meta['dtype']}  "
                      f"min={img_meta['min']:.4g}  max={img_meta['max']:.4g}")

            if mask_arr is not None:
                mask_meta = _array_info(mask_arr)
                unique_vals = np.unique(mask_arr).tolist()
                self._log(f"     Mask   shape={mask_meta['shape']}  "
                          f"dtype={mask_meta['dtype']}  "
                          f"unique_values={unique_vals[:20]}")

            n_cols = 4 if mask_arr is not None else 2
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            if n_cols == 1:
                axes = [axes]

            # --- Panel 0: raw / first-channel image ---
            disp_img = _display_2d(img_arr)
            axes[0].imshow(disp_img, cmap="gray")
            ch_label = (f"{img_meta['channels']}ch" if img_meta['channels'] != 1
                        else "1ch")
            axes[0].set_title(
                f"Image (ch0)\n{img_meta['shape']} | {img_meta['dtype']} | {ch_label}"
            )
            axes[0].axis("off")

            # --- Panel 1: colourised image (all channels if ≥2) ---
            if img_arr.ndim == 3 and img_arr.shape[-1] >= 2:
                rgb = _display_rgb(img_arr)
                axes[1].imshow(rgb)
                axes[1].set_title(
                    f"Image (ch 0-2 as RGB)\n{img_meta['dtype']}"
                )
            else:
                axes[1].imshow(disp_img, cmap="viridis")
                axes[1].set_title(
                    f"Image (viridis)\n{img_meta['dtype']}"
                )
            axes[1].axis("off")

            if mask_arr is not None:
                # --- Panel 2: mask ---
                unique_vals = np.unique(mask_arr)
                axes[2].imshow(mask_arr, cmap="tab10",
                               vmin=0, vmax=max(9, len(unique_vals)))
                axes[2].set_title(
                    f"Mask\n{mask_meta['shape']} | dtype={mask_meta['dtype']}\n"
                    f"unique={unique_vals[:8].tolist()}"
                )
                axes[2].axis("off")

                # --- Panel 3: overlay ---
                overlay_img = _display_rgb(img_arr)
                overlay = np.zeros((*overlay_img.shape[:2], 4), dtype=np.float32)
                for i, val in enumerate(unique_vals):
                    if val == 0:
                        continue  # skip background
                    colour = plt.cm.tab10(i / max(len(unique_vals), 1))
                    mask_region = (mask_arr == val)
                    for c in range(3):
                        overlay[mask_region, c] = colour[c]
                    overlay[mask_region, 3] = 0.5  # alpha

                axes[3].imshow(overlay_img)
                axes[3].imshow(overlay)
                axes[3].set_title("Image + Mask Overlay")
                axes[3].axis("off")

            fig.suptitle(
                f"Sample {idx + 1}: {img_path.name}",
                fontsize=13, fontweight="bold", y=1.01
            )
            plt.tight_layout()
            plt.show()
            print()

    def report_missing_pairs(self) -> None:
        """Print a report of images without masks and masks without images."""
        self._log("\n── Missing / unmatched pair report …")

        no_mask = [(img, mask) for img, mask in self.pairs if mask is None]
        if no_mask:
            self._log(f"   Images without a matching mask ({len(no_mask)}):")
            for img, _ in no_mask[:20]:
                self._log(f"     {img.relative_to(self.root)}")
            if len(no_mask) > 20:
                self._log(f"     … and {len(no_mask) - 20} more")
        else:
            self._log("   ✓ All images have a corresponding mask.")

        if self.unmatched_masks:
            self._log(f"\n   Masks without a matching image ({len(self.unmatched_masks)}):")
            for m in self.unmatched_masks[:20]:
                self._log(f"     {m.relative_to(self.root)}")
        else:
            self._log("   ✓ All masks have a corresponding image.")

    def print_summary(self) -> None:
        """Collect per-file metadata and print a human-readable dataset summary."""
        self._log("\n" + "=" * 60)
        self._log("  DATASET SUMMARY")
        self._log("=" * 60)

        # Collect metadata for all images
        shapes, dtypes, channels = [], [], []
        mask_dtypes, mask_unique = [], []

        iterator = (tqdm(self.pairs, desc="Inspecting files")
                    if TQDM_AVAILABLE else self.pairs)

        for img_path, mask_path in iterator:
            img_arr = _load_array(img_path)
            if img_arr is not None:
                info = _array_info(img_arr)
                shapes.append(info["shape"])
                dtypes.append(info["dtype"])
                channels.append(info["channels"])

            if mask_path is not None:
                m_arr = _load_array(mask_path)
                if m_arr is not None:
                    mask_dtypes.append(str(m_arr.dtype))
                    for v in np.unique(m_arr).tolist():
                        if v not in mask_unique:
                            mask_unique.append(v)

        # Aggregate
        unique_shapes = list({str(s): s for s in shapes}.values())
        unique_dtypes = list(set(dtypes))
        unique_channels = sorted(set(channels))
        unique_mask_dtypes = list(set(mask_dtypes))
        mask_unique.sort()

        self.summary = {
            "dataset_root": str(self.root),
            "total_raster_files": len(self.all_files),
            "image_files": len(self.image_files),
            "mask_files": len(self.mask_files),
            "unclassified_files": len(self.unknown_files),
            "matched_pairs": sum(1 for _, m in self.pairs if m is not None),
            "images_without_mask": sum(1 for _, m in self.pairs if m is None),
            "masks_without_image": len(self.unmatched_masks),
            "image_shapes_found": [list(s) for s in unique_shapes],
            "image_dtypes": unique_dtypes,
            "image_channels": unique_channels,
            "mask_dtypes": unique_mask_dtypes,
            "mask_unique_values": mask_unique[:50],
        }

        lines = [
            f"  Dataset root        : {self.root}",
            f"  Total raster files  : {len(self.all_files)}",
            f"  Image files         : {len(self.image_files)}",
            f"  Mask / label files  : {len(self.mask_files)}",
            f"  Unclassified files  : {len(self.unknown_files)}",
            "",
            f"  Matched pairs       : {self.summary['matched_pairs']}",
            f"  Images w/o mask     : {self.summary['images_without_mask']}",
            f"  Masks w/o image     : {self.summary['masks_without_image']}",
            "",
            f"  Image dtypes found  : {unique_dtypes}",
            f"  Image channels      : {unique_channels}",
            f"  Image shapes        : {[list(s) for s in unique_shapes[:5]]}{'…' if len(unique_shapes) > 5 else ''}",
            "",
            f"  Mask dtypes found   : {unique_mask_dtypes}",
            f"  Mask unique values  : {mask_unique[:20]}{'…' if len(mask_unique) > 20 else ''}",
        ]
        for line in lines:
            print(line)

        print("=" * 60)

    def save_summary_json(self, output_path: str | Path = "dataset_summary.json") -> Path:
        """Save the summary dict to a JSON file."""
        if not self.summary:
            self.print_summary()
        out = Path(output_path)
        with open(out, "w") as f:
            json.dump(self.summary, f, indent=2)
        self._log(f"\n   Summary saved to: {out.resolve()}")
        return out.resolve()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _check_root(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(
                f"\n[DatasetInspector] Dataset root not found:\n  {self.root}\n\n"
                "Please follow the instructions in ml/dataset/README.md to place the "
                "Sentinel-1 dataset before running the inspector."
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _print_tree(self, path: Path, prefix: str = "", _root: bool = True) -> None:
        """Recursively print directory tree."""
        if _root:
            print(f"\n{path.name}/")
            prefix = ""
        try:
            children = sorted(path.iterdir())
        except PermissionError:
            return
        dirs = [c for c in children if c.is_dir()]
        files = [c for c in children if c.is_file()
                 and c.suffix.lower() in RASTER_EXTENSIONS]
        all_items = dirs + files
        for i, item in enumerate(all_items):
            connector = "└── " if i == len(all_items) - 1 else "├── "
            if item.is_dir():
                print(f"{prefix}{connector}{item.name}/")
                extension = "    " if i == len(all_items) - 1 else "│   "
                self._print_tree(item, prefix + extension, _root=False)
            else:
                print(f"{prefix}{connector}{item.name}")
