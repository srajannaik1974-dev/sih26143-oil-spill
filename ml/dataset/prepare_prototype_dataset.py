"""
prepare_prototype_dataset.py
============================
SIH 2026 — PS 26143: Sentinel-1 SAR Oil-Spill Dataset
Prototype Subset Preparation Utility

PURPOSE
-------
Select a reproducible, leakage-free prototype subset from the full Zenodo
dataset and copy it into ml/dataset/prototype/ with canonical train/val/test
splits.

USAGE
-----
# From project root:
python ml/dataset/prepare_prototype_dataset.py

# Or with custom paths / counts:
python ml/dataset/prepare_prototype_dataset.py \\
    --raw-dir    ml/dataset/raw \\
    --out-dir    ml/dataset/prototype \\
    --n-train    400 \\
    --n-val       75 \\
    --n-test      75 \\
    --seed        42

RULES
-----
- Does NOT download anything automatically.
- Does NOT modify original files.
- Does NOT put the same file in more than one split (zero data leakage).
- Uses a fixed random seed for full reproducibility.
- Copies files (not symlinks) so the prototype is self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration defaults  (override via CLI args)
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR   = Path(__file__).parent / "raw"
DEFAULT_OUT_DIR   = Path(__file__).parent / "prototype"
DEFAULT_N_TRAIN   = 400
DEFAULT_N_VAL     = 75
DEFAULT_N_TEST    = 75
DEFAULT_SEED      = 42

# Extensions we consider raster files
RASTER_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz"}

# Keywords that identify mask/label folders or filenames
MASK_KEYWORDS  = {"mask", "label", "annotation", "gt", "ground_truth",
                  "groundtruth", "seg", "segmentation"}
IMAGE_KEYWORDS = {"image", "img", "sar", "sentinel", "patch", "scene",
                  "no_oil", "lookalike", "oil_spill"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _guess_role(path: Path) -> str:
    """Return 'mask', 'image', or 'unknown' for a raster file."""
    parts_lower = {p.lower() for p in path.parts}
    stem_lower  = path.stem.lower()

    for kw in MASK_KEYWORDS:
        if kw in parts_lower or kw in stem_lower:
            return "mask"
    for kw in IMAGE_KEYWORDS:
        if kw in parts_lower or kw in stem_lower:
            return "image"
    return "unknown"


def _clean_stem(path: Path) -> str:
    """
    Strip known image/mask prefixes from a stem to produce a matchable key.
    E.g. 'oil_spill_0001' -> '0001', 'mask_0001' -> '0001'
    """
    s = path.stem.lower()
    for kw in MASK_KEYWORDS | IMAGE_KEYWORDS:
        s = s.replace(kw, "")
    return s.strip("_-. ")


def _category_from_path(path: Path) -> str:
    """
    Infer semantic category from the directory name:
      oil_spill / lookalike / no_oil / unknown
    """
    parts_lower = " ".join(p.lower() for p in path.parts)
    if "no_oil" in parts_lower or "no oil" in parts_lower:
        return "no_oil"
    if "lookalike" in parts_lower or "look_alike" in parts_lower:
        return "lookalike"
    if "oil_spill" in parts_lower or "oil spill" in parts_lower:
        return "oil_spill"
    return "unknown"


# ---------------------------------------------------------------------------
# Step 1 - Scan raw directory
# ---------------------------------------------------------------------------

def scan_raw_directory(raw_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Walk raw_dir and classify every raster file as image / mask / unknown.
    Returns (images, masks, unknowns).
    """
    _log(f"\n{'--'*30}")
    _log(f"  Scanning raw directory: {raw_dir}")
    _log(f"{'--'*30}")

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"\n[STOP] Raw dataset directory not found:\n  {raw_dir}\n\n"
            "Please follow the download instructions in ml/dataset/README.md\n"
            "and extract the Zenodo archives into ml/dataset/raw/\n"
            "before running this script."
        )

    all_rasters = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in RASTER_EXT
    )

    images, masks, unknown = [], [], []
    for p in all_rasters:
        role = _guess_role(p)
        if role == "image":
            images.append(p)
        elif role == "mask":
            masks.append(p)
        else:
            unknown.append(p)

    _log(f"  Total raster files : {len(all_rasters)}")
    _log(f"  Image files        : {len(images)}")
    _log(f"  Mask files         : {len(masks)}")
    _log(f"  Unclassified files : {len(unknown)}")

    if unknown:
        _log(f"\n  WARNING: Unclassified files (first 10) - check directory names:")
        for p in unknown[:10]:
            _log(f"    {p.relative_to(raw_dir)}")

    return images, masks, unknown


# ---------------------------------------------------------------------------
# Step 2 - Pair images with masks
# ---------------------------------------------------------------------------

def build_pairs(
    images: List[Path],
    masks: List[Path],
    raw_dir: Path,
) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    """
    Match each image to its mask by (a) exact stem, then (b) cleaned stem.
    Returns:
        valid_pairs    - list of (image_path, mask_path) tuples
        images_no_mask - images that could not be matched
        orphan_masks   - masks that could not be matched
    """
    _log(f"\n  Pairing images with masks ...")

    mask_by_exact: Dict[str, Path] = {m.stem: m for m in masks}
    mask_by_clean: Dict[str, Path] = {_clean_stem(m): m for m in masks}

    valid_pairs: List[Tuple[Path, Path]] = []
    images_no_mask: List[Path] = []
    matched_masks = set()

    for img in images:
        mask = mask_by_exact.get(img.stem)
        if mask is None:
            mask = mask_by_clean.get(_clean_stem(img))
        if mask is not None:
            valid_pairs.append((img, mask))
            matched_masks.add(mask)
        else:
            images_no_mask.append(img)

    orphan_masks = [m for m in masks if m not in matched_masks]

    _log(f"  Valid pairs            : {len(valid_pairs)}")
    _log(f"  Images without mask    : {len(images_no_mask)}")
    _log(f"  Masks without image    : {len(orphan_masks)}")

    if images_no_mask:
        _log(f"\n  WARNING: Images without a matching mask (first 10):")
        for p in images_no_mask[:10]:
            _log(f"    {p.relative_to(raw_dir)}")

    if orphan_masks:
        _log(f"\n  WARNING: Orphan masks without a matching image (first 10):")
        for p in orphan_masks[:10]:
            _log(f"    {p.relative_to(raw_dir)}")

    return valid_pairs, images_no_mask, orphan_masks


# ---------------------------------------------------------------------------
# Step 3 - Validate pairs
# ---------------------------------------------------------------------------

def validate_pairs(
    pairs: List[Tuple[Path, Path]],
    raw_dir: Path,
) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    """
    Basic integrity checks:
    - Both files exist on disk.
    - No duplicate image stems within the pool.
    Returns (valid_pairs, list_of_warnings).
    """
    _log(f"\n  Validating {len(pairs)} pairs ...")

    warnings: List[str] = []
    valid: List[Tuple[Path, Path]] = []
    seen_stems: Dict[str, Path] = {}

    for img, mask in pairs:
        issues = []
        if not img.exists():
            issues.append(f"image missing on disk: {img}")
        if not mask.exists():
            issues.append(f"mask missing on disk: {mask}")

        stem = img.stem
        if stem in seen_stems:
            issues.append(
                f"duplicate image stem '{stem}': "
                f"{img.relative_to(raw_dir)} vs "
                f"{seen_stems[stem].relative_to(raw_dir)}"
            )
        else:
            seen_stems[stem] = img

        if issues:
            for issue in issues:
                warnings.append(f"  SKIP: {issue}")
        else:
            valid.append((img, mask))

    for w in warnings:
        _log(w)

    _log(f"  Valid after checks     : {len(valid)}")
    _log(f"  Skipped (issues)       : {len(pairs) - len(valid)}")
    return valid, warnings


# ---------------------------------------------------------------------------
# Step 4 - Split
# ---------------------------------------------------------------------------

def split_pairs(
    pairs: List[Tuple[Path, Path]],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Randomly sample and split pairs into train / val / test with ZERO leakage.
    Each image appears in exactly one split.

    Strategy:
    - Group pairs by semantic category (oil_spill / lookalike / no_oil).
    - Shuffle each category independently with the same seed.
    - Draw test first, then val, then train (test is truly unseen).
    - Verify no overlaps exist (safety assertion).
    """
    _log(f"\n  Splitting with seed={seed} ...")

    total_needed = n_train + n_val + n_test
    if len(pairs) < total_needed:
        raise ValueError(
            f"Not enough valid pairs ({len(pairs)}) for the requested split "
            f"({n_train} train + {n_val} val + {n_test} test = {total_needed}).\n"
            "Reduce --n-train / --n-val / --n-test, or download more data."
        )

    # Group by category
    by_category: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for img, mask in pairs:
        cat = _category_from_path(img)
        by_category[cat].append((img, mask))

    _log(f"\n  Category breakdown (all valid pairs):")
    for cat, items in sorted(by_category.items()):
        _log(f"    {cat:15s}: {len(items)} pairs")

    # Shuffle each category independently
    rng = random.Random(seed)
    for cat in by_category:
        rng.shuffle(by_category[cat])

    # Flatten in deterministic category order
    flat = []
    for cat in sorted(by_category.keys()):
        flat.extend(by_category[cat])

    # Global shuffle for final mixing
    rng2 = random.Random(seed + 1)
    rng2.shuffle(flat)

    # Draw splits: test first (completely unseen), val next, train last
    test_pairs  = flat[:n_test]
    val_pairs   = flat[n_test : n_test + n_val]
    train_pairs = flat[n_test + n_val : n_test + n_val + n_train]

    # Safety: verify zero overlap
    test_stems  = {img.stem for img, _ in test_pairs}
    val_stems   = {img.stem for img, _ in val_pairs}
    train_stems = {img.stem for img, _ in train_pairs}

    overlap_tv = train_stems & val_stems
    overlap_tt = train_stems & test_stems
    overlap_vt = val_stems & test_stems

    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"Data leakage detected!\n"
            f"  train & val  : {len(overlap_tv)}\n"
            f"  train & test : {len(overlap_tt)}\n"
            f"  val & test   : {len(overlap_vt)}\n"
            "This is a bug - please report it."
        )

    splits = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
    }

    _log(f"\n  Split summary:")
    for split_name, split_list in splits.items():
        cats = defaultdict(int)
        for img, _ in split_list:
            cats[_category_from_path(img)] += 1
        cat_str = "  ".join(f"{k}={v}" for k, v in sorted(cats.items()))
        _log(f"    {split_name:5s}: {len(split_list):4d} pairs  [{cat_str}]")

    _log(f"\n  Zero data leakage: VERIFIED")

    return splits


# ---------------------------------------------------------------------------
# Step 5 - Copy to prototype directory
# ---------------------------------------------------------------------------

def copy_splits(
    splits: Dict[str, List[Tuple[Path, Path]]],
    out_dir: Path,
) -> None:
    """
    Copy selected pairs into out_dir/train|val|test/images|masks/.
    Original files are NEVER modified.
    """
    _log(f"\n  Copying files to: {out_dir}")

    for split_name, pairs in splits.items():
        img_dir  = out_dir / split_name / "images"
        mask_dir = out_dir / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        _log(f"\n    [{split_name}] Copying {len(pairs)} pairs ...")
        for i, (img, mask) in enumerate(pairs, 1):
            shutil.copy2(img,  img_dir  / img.name)
            shutil.copy2(mask, mask_dir / mask.name)
            if i % 50 == 0 or i == len(pairs):
                _log(f"      {i}/{len(pairs)} copied ...")

    _log(f"\n  All splits written to {out_dir}")


# ---------------------------------------------------------------------------
# Step 6 - Summary JSON
# ---------------------------------------------------------------------------

def write_summary(
    splits: Dict[str, List[Tuple[Path, Path]]],
    out_dir: Path,
    raw_dir: Path,
    warnings: List[str],
    args: argparse.Namespace,
) -> Path:
    """Write dataset_summary.json into out_dir."""

    def _pair_record(img: Path, mask: Path) -> dict:
        return {
            "image":    img.name,
            "mask":     mask.name,
            "category": _category_from_path(img),
        }

    totals: Dict[str, int] = defaultdict(int)
    for pairs in splits.values():
        for img, _ in pairs:
            totals[_category_from_path(img)] += 1

    summary = {
        "raw_dir":          str(raw_dir),
        "out_dir":          str(out_dir),
        "seed":             args.seed,
        "n_train":          args.n_train,
        "n_val":            args.n_val,
        "n_test":           args.n_test,
        "zero_leakage":     True,
        "category_totals":  dict(sorted(totals.items())),
        "splits": {
            name: {
                "count": len(pairs),
                "files": [_pair_record(img, mask) for img, mask in pairs],
            }
            for name, pairs in splits.items()
        },
        "warnings": warnings,
    }

    out_path = out_dir / "dataset_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"\n  Summary JSON written to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a prototype subset of the Sentinel-1 oil-spill dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Directory containing extracted Zenodo archives (ml/dataset/raw/).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory for the prototype subset (ml/dataset/prototype/).",
    )
    parser.add_argument(
        "--n-train", type=int, default=DEFAULT_N_TRAIN,
        help="Number of training pairs to select.",
    )
    parser.add_argument(
        "--n-val", type=int, default=DEFAULT_N_VAL,
        help="Number of validation pairs to select.",
    )
    parser.add_argument(
        "--n-test", type=int, default=DEFAULT_N_TEST,
        help="Number of test pairs to select (completely unseen).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan the splits but do not copy any files.",
    )
    args = parser.parse_args()

    _log("=" * 60)
    _log("  SIH 2026 - PS 26143: Prototype Dataset Preparation")
    _log("=" * 60)
    _log(f"  Raw dir   : {args.raw_dir}")
    _log(f"  Output    : {args.out_dir}")
    _log(f"  Train     : {args.n_train}")
    _log(f"  Val       : {args.n_val}")
    _log(f"  Test      : {args.n_test}")
    _log(f"  Seed      : {args.seed}")
    _log(f"  Dry run   : {args.dry_run}")

    # Step 1: Scan
    images, masks, unknowns = scan_raw_directory(args.raw_dir)

    if not images:
        _log("\n[STOP] No image files found in the raw directory.")
        _log("Please extract the Zenodo archives into ml/dataset/raw/")
        _log("and re-run this script.")
        sys.exit(1)

    # Step 2: Pair
    pairs, images_no_mask, orphan_masks = build_pairs(images, masks, args.raw_dir)

    if not pairs:
        _log("\n[STOP] No valid image-mask pairs found.")
        _log("Check that image and mask files share the same numeric stem (e.g. 0001).")
        sys.exit(1)

    # Step 3: Validate
    valid_pairs, warnings = validate_pairs(pairs, args.raw_dir)

    # Step 4: Split
    splits = split_pairs(valid_pairs, args.n_train, args.n_val, args.n_test, args.seed)

    # Step 5: Copy
    if args.dry_run:
        _log("\n  DRY RUN - no files were copied.")
    else:
        copy_splits(splits, args.out_dir)

    # Step 6: Summary
    if not args.dry_run:
        write_summary(splits, args.out_dir, args.raw_dir, warnings, args)

    # Final report
    _log("\n" + "=" * 60)
    _log("  PROTOTYPE DATASET READY" if not args.dry_run else "  DRY RUN COMPLETE")
    _log("=" * 60)
    _log(f"  Train  : {len(splits['train']):4d} pairs  ->  {args.out_dir}/train/")
    _log(f"  Val    : {len(splits['val']):4d} pairs  ->  {args.out_dir}/val/")
    _log(f"  Test   : {len(splits['test']):4d} pairs  ->  {args.out_dir}/test/")
    _log(f"  Total  : {sum(len(v) for v in splits.values())} pairs selected")
    _log(f"\n  Zero data leakage  : VERIFIED")
    _log(f"  Originals modified : NO")
    _log(f"  Random seed        : {args.seed} (reproducible)")
    _log("=" * 60)


if __name__ == "__main__":
    main()
