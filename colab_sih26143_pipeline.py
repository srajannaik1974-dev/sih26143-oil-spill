"""
colab_sih26143_pipeline.py
==========================
SIH 2026 — Problem Statement 26143
Member 1: Sentinel-1 SAR Oil-Spill Detection — Complete Colab Pipeline

HOW TO USE
----------
1. Upload this file to Google Colab (File → Upload notebook, or copy sections into cells).
2. Run each PHASE block IN ORDER.
3. Read the printed output after each phase before proceeding.
4. STOP after PHASE 1 to read the environment report.
5. STOP after PHASE 2 to confirm your dataset strategy.
6. Only proceed to PHASE 3+ once images are in place.

PHASES
------
 0  Imports & Drive mount
 1  Environment Inspection   ← START HERE, read all output
 2  Dataset Strategy         ← configure USER CONFIG block, then re-run
 3  Dataset Verification     ← after images are placed/downloaded
 4  Split + Copy to Drive    ← permanent storage 70/15/15
 5  Training                 ← GPU recommended
 6  Save Checkpoint to Drive ← permanent
 7  Inference on Test Image  ← held-out image only
 8  Geolocation + Area       ← lat/lon from TIFF metadata; honest NULL if not georeferenced
 9  Member 2 Output          ← final structured dict
10  API Verification + pytest
"""

# ============================================================
# PHASE 0 — COLAB SETUP  (run once)
# ============================================================

import os, sys, shutil, subprocess, warnings, json, math, random, time
from pathlib import Path
from datetime import datetime, timezone

# Mount Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# Suppress rasterio CRS warnings
warnings.filterwarnings("ignore", message=".*crs.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*CRS.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")

print("Phase 0 complete — Drive mounted.")


# ============================================================
# PHASE 1 — ENVIRONMENT INSPECTION
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 1: ENVIRONMENT INSPECTION")
print("=" * 65)

import torch
print(f"\n[GPU]")
print(f"  torch version       : {torch.__version__}")
print(f"  CUDA available      : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU name            : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  VRAM total          : {props.total_memory / 1e9:.1f} GB")

# Existing masks
MASKS_TEMP = Path("/content/masks_temp/Mask_oil")
print(f"\n[EXISTING MASKS]  {MASKS_TEMP}")
if MASKS_TEMP.exists():
    mask_files = sorted(MASKS_TEMP.glob("*.tif")) + sorted(MASKS_TEMP.glob("*.tiff"))
    print(f"  Total masks found   : {len(mask_files)}")
    if mask_files:
        ids = [f.stem for f in mask_files]
        print(f"  ID range            : {ids[0]} -> {ids[-1]}")
        print(f"  First 10 IDs        : {ids[:10]}")
        print(f"  Last  10 IDs        : {ids[-10:]}")
        try:
            import rasterio
            with rasterio.open(str(mask_files[0])) as src:
                import numpy as np
                arr = src.read().astype(np.float32)
                print(f"  Sample mask shape   : bands={src.count}, H={src.height}, W={src.width}")
                print(f"  Sample mask dtype   : {src.dtypes[0]}")
                print(f"  Sample mask values  : {sorted(set(arr.flatten().tolist()))[:8]}")
        except Exception as e:
            print(f"  Could not read sample mask: {e}")
else:
    print("  NOT FOUND — /content/masks_temp/ does not exist!")

# Existing SAR images
print(f"\n[EXISTING SAR IMAGES]")
for search_dir in [Path("/content"), Path("/content/drive/MyDrive/SIH26143")]:
    if search_dir.exists():
        tifs = [f for f in search_dir.rglob("*.tif")
                if "mask" not in str(f).lower() and "masks_temp" not in str(f)]
        print(f"  {search_dir}: {len(tifs)} non-mask TIFFs")
        for f in tifs[:5]:
            print(f"    {f}")
    else:
        print(f"  {search_dir}: does not exist")

# Drive space
print(f"\n[GOOGLE DRIVE SPACE]")
try:
    stat = shutil.disk_usage("/content/drive/MyDrive")
    print(f"  Total : {stat.total/1e9:.1f} GB")
    print(f"  Used  : {stat.used/1e9:.1f} GB")
    print(f"  Free  : {stat.free/1e9:.1f} GB")
except Exception as e:
    print(f"  Error: {e}")

# Drive dataset structure
DRIVE_ROOT = Path("/content/drive/MyDrive/SIH26143")
PROTO_ROOT = DRIVE_ROOT / "prototype"
CKPT_DRIVE = DRIVE_ROOT / "checkpoints"
print(f"\n[DRIVE DATASET STRUCTURE]")
for split in ["train", "val", "test"]:
    for kind in ["images", "masks"]:
        d = PROTO_ROOT / split / kind
        if d.exists():
            n = len(list(d.glob("*.tif"))) + len(list(d.glob("*.tiff")))
            print(f"  {split}/{kind}: {n} files")
        else:
            print(f"  {split}/{kind}: not found")

# Existing checkpoints
print(f"\n[EXISTING CHECKPOINTS]")
for cp in [
    CKPT_DRIVE / "best_unet.pth",
    Path("ml/training/checkpoints/best_unet.pth"),
    Path("/content/sih26143-oil-spill/ml/training/checkpoints/best_unet.pth"),
]:
    if cp.exists():
        print(f"  FOUND {cp}  ({cp.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  missing: {cp}")

# Python packages
print(f"\n[KEY PACKAGES]")
for pkg in ["torch", "numpy", "rasterio", "PIL", "fastapi", "uvicorn",
            "matplotlib", "scipy", "pyproj", "sklearn"]:
    try:
        mod = __import__(pkg if pkg != "PIL" else "PIL")
        ver = getattr(mod, "__version__", "?")
        print(f"  {pkg:15s}: {ver}")
    except ImportError:
        print(f"  {pkg:15s}: NOT INSTALLED")

# Project root
print(f"\n[PROJECT ROOT]")
PROJECT_ROOT = None
for cand in [
    Path("/content/sih26143-oil-spill"),
    Path("/content/drive/MyDrive/sih26143-oil-spill"),
    Path("."),
]:
    if (cand / "ml" / "training" / "train.py").exists():
        PROJECT_ROOT = cand.resolve()
        print(f"  Found: {PROJECT_ROOT}")
        break
if PROJECT_ROOT is None:
    print("  NOT FOUND — clone the repo first")

# Zenodo record metadata (no download)
print(f"\n[ZENODO RECORD CHECK — no download]")
try:
    import urllib.request as _ur, json as _js
    ZENODO_ID = "4748119"
    with _ur.urlopen(f"https://zenodo.org/api/records/{ZENODO_ID}", timeout=10) as r:
        rec = _js.loads(r.read())
    print(f"  Title  : {rec.get('metadata',{}).get('title','?')}")
    for f in rec.get("files", []):
        print(f"  File   : {f['key']:<60s}  {f.get('size',0)/1e6:8.1f} MB")
except Exception as e:
    print(f"  Could not reach Zenodo: {e}")

print("\n" + "=" * 65)
print("  PHASE 1 COMPLETE. Read all output above before Phase 2.")
print("=" * 65)


# ============================================================
# PHASE 2 — DATASET STRATEGY CONFIGURATION
# ============================================================
# Edit the USER CONFIG block below, then re-run this phase.
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 2: DATASET STRATEGY")
print("=" * 65)

# ───────────────────── USER CONFIG ───────────────────────────
# Set exactly ONE of the three options below (others = None).

# Option A: SAR images already exist in Drive
EXISTING_IMAGES_DIR = None
# e.g. "/content/drive/MyDrive/SIH26143/images_raw"

# Option B: Download from Kaggle (need kaggle.json at ~/.kaggle/kaggle.json)
KAGGLE_DATASET = None
# e.g. "sudhanshu2198/oil-spill-detection"

# Option C: Images placed manually at a local path
MANUAL_IMAGES_DIR = None
# e.g. "/content/images_raw"

# How many pairs to use (max)
MAX_PAIRS  = 200
SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
# ─────────────────────────────────────────────────────────────

def _count_tifs(d):
    if d is None or not Path(d).exists():
        return 0
    return len(list(Path(d).glob("*.tif"))) + len(list(Path(d).glob("*.tiff")))

IMAGES_RAW_DIR = None
strategy       = None

if EXISTING_IMAGES_DIR and _count_tifs(EXISTING_IMAGES_DIR) > 0:
    strategy       = "A_existing"
    IMAGES_RAW_DIR = Path(EXISTING_IMAGES_DIR)
    print(f"  Strategy A — existing images: {IMAGES_RAW_DIR}")
    print(f"  Count: {_count_tifs(IMAGES_RAW_DIR)}")

elif MANUAL_IMAGES_DIR and _count_tifs(MANUAL_IMAGES_DIR) > 0:
    strategy       = "C_manual"
    IMAGES_RAW_DIR = Path(MANUAL_IMAGES_DIR)
    print(f"  Strategy C — manual images: {IMAGES_RAW_DIR}")
    print(f"  Count: {_count_tifs(IMAGES_RAW_DIR)}")

elif KAGGLE_DATASET:
    strategy       = "B_kaggle"
    IMAGES_RAW_DIR = Path("/content/kaggle_images")
    print(f"  Strategy B — Kaggle: {KAGGLE_DATASET}")
    print(f"  WARNING: Verify Kaggle dataset has 2-channel VV+VH SAR TIFFs before training.")

else:
    print("  NO STRATEGY CONFIGURED.\n")
    print("  Instructions:")
    print("  1. Put your SAR images somewhere and set EXISTING_IMAGES_DIR or MANUAL_IMAGES_DIR")
    print("  2. OR set KAGGLE_DATASET to a Kaggle slug")
    print()
    print("  FASTEST for SIH demo:")
    print("  - If Zenodo record (Phase 1) shows individual TIFFs: run try_zenodo_direct() below")
    print("  - Otherwise: download images manually via browser from Zenodo,")
    print("    upload to Colab, set MANUAL_IMAGES_DIR = '/content/images_raw'")
    print()
    print("  STOP: configure strategy above and re-run Phase 2.")

def try_zenodo_direct(record_id, target_dir, max_files=MAX_PAIRS):
    """Download individual TIFFs from Zenodo if they exist as separate files (not in archive)."""
    import urllib.request as _ur, json as _js
    with _ur.urlopen(f"https://zenodo.org/api/records/{record_id}", timeout=15) as r:
        rec = _js.loads(r.read())
    tifs = [f for f in rec.get("files", [])
            if f["key"].lower().endswith((".tif", ".tiff"))
            and "mask" not in f["key"].lower()]
    print(f"  Individual image TIFFs in record: {len(tifs)}")
    if not tifs:
        print("  None found — record contains only archive files.")
        return 0
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for finfo in tifs[:max_files]:
        dest = Path(target_dir) / finfo["key"]
        if not dest.exists():
            print(f"  Downloading {finfo['key']} ...", end=" ", flush=True)
            _ur.urlretrieve(finfo["links"]["self"], str(dest))
            print("ok")
        downloaded += 1
    return downloaded

print(f"\n  Strategy: {strategy or 'NOT SET'}")
print("=" * 65)


# ============================================================
# PHASE 3 — DATASET VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 3: DATASET VERIFICATION")
print("=" * 65)

if strategy is None or IMAGES_RAW_DIR is None:
    raise SystemExit("Phase 2 not configured. Set strategy and re-run.")

# Kaggle download if needed
if strategy == "B_kaggle":
    print(f"  Downloading from Kaggle: {KAGGLE_DATASET} ...")
    IMAGES_RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(IMAGES_RAW_DIR), "--unzip"],
        check=True
    )

import rasterio
import numpy as np

all_masks  = {f.stem: f for f in sorted(MASKS_TEMP.glob("*.tif")) + sorted(MASKS_TEMP.glob("*.tiff"))}
all_images = {f.stem: f for f in sorted(IMAGES_RAW_DIR.glob("*.tif")) + sorted(IMAGES_RAW_DIR.glob("*.tiff"))
              if "mask" not in str(f).lower()}

common_ids = sorted(set(all_masks) & set(all_images))
print(f"  Masks available : {len(all_masks)}")
print(f"  Images available: {len(all_images)}")
print(f"  Matched pairs   : {len(common_ids)}")

if not common_ids:
    print("\n  NO MATCHING PAIRS. Image stems must match mask stems (e.g. 00642).")
    raise SystemExit("No matched pairs.")

# Validate up to 50 pairs for shape/channels
print(f"\n  Validating pairs ...")
valid_pairs   = []
invalid_pairs = []

for stem in common_ids:
    errors = []
    try:
        with rasterio.open(str(all_images[stem])) as src:
            if src.count != 2:
                errors.append(f"image has {src.count} channels (expected 2)")
        with rasterio.open(str(all_masks[stem])) as src:
            if src.count != 1:
                errors.append(f"mask has {src.count} channels (expected 1)")
            arr = src.read().astype(np.float32)
            uniq = set(arr.flatten())
            # Accept 0/1 or 0/255 (will be normalised)
            if not uniq.issubset({0.0, 1.0}) and max(uniq) > 1.0:
                errors.append(f"mask values unexpected: {sorted(uniq)[:5]}")
    except Exception as e:
        errors.append(str(e))

    if errors:
        invalid_pairs.append((stem, errors))
        if len(invalid_pairs) <= 5:
            print(f"  INVALID {stem}: {errors}")
    else:
        valid_pairs.append(stem)

print(f"\n  Valid   : {len(valid_pairs)}")
print(f"  Invalid : {len(invalid_pairs)}")

# Limit to MAX_PAIRS
random.seed(SEED)
if len(valid_pairs) > MAX_PAIRS:
    random.shuffle(valid_pairs)
    valid_pairs = sorted(valid_pairs[:MAX_PAIRS])
    print(f"  Limited to {len(valid_pairs)} pairs (MAX_PAIRS={MAX_PAIRS})")

# Sample stats
s = valid_pairs[0]
with rasterio.open(str(all_images[s])) as src:
    a = src.read().astype(np.float32)
    print(f"\n  Sample image {s}: shape={a.shape}, ch0=[{a[0].min():.1f},{a[0].max():.1f}], ch1=[{a[1].min():.1f},{a[1].max():.1f}]")
with rasterio.open(str(all_masks[s])) as src:
    m = src.read().astype(np.float32)
    print(f"  Sample mask  {s}: shape={m.shape}, spill%={100*(m>0).sum()/m.size:.2f}%")

print(f"\n  Final verified pairs: {len(valid_pairs)}")
print("=" * 65)


# ============================================================
# PHASE 4 — SPLIT + COPY TO GOOGLE DRIVE
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 4: SPLIT + COPY TO DRIVE")
print("=" * 65)

random.seed(SEED)
shuffled = valid_pairs.copy()
random.shuffle(shuffled)
n       = len(shuffled)
n_train = math.floor(n * TRAIN_FRAC)
n_val   = math.floor(n * VAL_FRAC)
n_test  = n - n_train - n_val
train_ids = sorted(shuffled[:n_train])
val_ids   = sorted(shuffled[n_train:n_train+n_val])
test_ids  = sorted(shuffled[n_train+n_val:])

print(f"  Train : {len(train_ids)}")
print(f"  Val   : {len(val_ids)}")
print(f"  Test  : {len(test_ids)}")
assert not (set(train_ids) & set(val_ids) & set(test_ids))

DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
manifest = {"seed": SEED, "train": train_ids, "val": val_ids, "test": test_ids,
            "created": datetime.now(timezone.utc).isoformat()}
with open(DRIVE_ROOT / "split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  Split manifest saved.")

def copy_split(ids, split_name):
    img_dir = PROTO_ROOT / split_name / "images"
    msk_dir = PROTO_ROOT / split_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    for stem in ids:
        src_i = all_images[stem]
        dst_i = img_dir / src_i.name
        if not dst_i.exists():
            shutil.copy2(src_i, dst_i)
        src_m = all_masks[stem]
        dst_m = msk_dir / src_m.name
        if not dst_m.exists():
            shutil.copy2(src_m, dst_m)
    n_imgs = len(list(img_dir.glob("*.tif"))) + len(list(img_dir.glob("*.tiff")))
    n_msks = len(list(msk_dir.glob("*.tif"))) + len(list(msk_dir.glob("*.tiff")))
    print(f"  {split_name}: {n_imgs} images, {n_msks} masks -> {PROTO_ROOT}/{split_name}/")

copy_split(train_ids, "train")
copy_split(val_ids,   "val")
copy_split(test_ids,  "test")

# Verify
total = 0
for split in ["train", "val", "test"]:
    for kind in ["images", "masks"]:
        d = PROTO_ROOT / split / kind
        n_f = len(list(d.glob("*.tif"))) + len(list(d.glob("*.tiff")))
        total += n_f
print(f"\n  Total files in Drive prototype/: {total}")
print(f"  PROTO_ROOT exists: {PROTO_ROOT.exists()}")
print("=" * 65)


# ============================================================
# PHASE 5 — TRAINING
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 5: TRAINING")
print("=" * 65)

if PROJECT_ROOT is None:
    raise RuntimeError("PROJECT_ROOT not found. Clone the repo first:\n"
                       "  !git clone https://github.com/yourname/sih26143-oil-spill /content/sih26143-oil-spill")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.train import train as run_train
import argparse

# VRAM-aware defaults
_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
_batch   = 4 if _vram_gb >= 8 else 2
_feats   = 64 if _vram_gb >= 8 else 32
print(f"  GPU VRAM: {_vram_gb:.1f} GB -> batch_size={_batch}, base_features={_feats}")

LOCAL_CKPT_DIR = PROJECT_ROOT / "ml" / "training" / "checkpoints"
args = argparse.Namespace(
    data_dir      = str(PROTO_ROOT),
    ckpt_dir      = str(LOCAL_CKPT_DIR),
    epochs        = 50,
    batch_size    = _batch,
    lr            = 1e-4,
    image_size    = 512,
    base_features = _feats,
    patience      = 10,
    seed          = SEED,
)
print(f"  data_dir      : {args.data_dir}")
print(f"  ckpt_dir      : {args.ckpt_dir}")
print(f"  epochs        : {args.epochs}")
print(f"  image_size    : {args.image_size}")

t0 = time.time()
print(f"\n  Started: {datetime.now(timezone.utc).isoformat()}")
history = run_train(args)
elapsed = time.time() - t0
print(f"\n  Finished in {elapsed/60:.1f} min")

best_epoch = max(history, key=lambda x: x["val_dice"]) if history else {}
print(f"\n  Best epoch   : {best_epoch.get('epoch', 'N/A')}")
print(f"  Best val Dice: {best_epoch.get('val_dice', 0):.4f}")
print(f"  Best val IoU : {best_epoch.get('val_iou', 0):.4f}")

LOCAL_CKPT = LOCAL_CKPT_DIR / "best_unet.pth"
print(f"\n  Checkpoint: {LOCAL_CKPT}  exists={LOCAL_CKPT.exists()}")
if LOCAL_CKPT.exists():
    print(f"  Size: {LOCAL_CKPT.stat().st_size/1e6:.1f} MB")

print("=" * 65)


# ============================================================
# PHASE 6 — SAVE CHECKPOINT TO DRIVE
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 6: CHECKPOINT -> DRIVE")
print("=" * 65)

CKPT_DRIVE.mkdir(parents=True, exist_ok=True)
DRIVE_CKPT = CKPT_DRIVE / "best_unet.pth"

if not LOCAL_CKPT.exists():
    raise FileNotFoundError(f"Local checkpoint missing: {LOCAL_CKPT}")

shutil.copy2(LOCAL_CKPT, DRIVE_CKPT)
print(f"  Copied to: {DRIVE_CKPT}")
print(f"  Size: {DRIVE_CKPT.stat().st_size/1e6:.1f} MB")
print(f"  Exists: {DRIVE_CKPT.exists()}")

if history:
    with open(DRIVE_ROOT / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to Drive.")

print("=" * 65)


# ============================================================
# PHASE 7 — INFERENCE ON TEST IMAGE
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 7: INFERENCE ON TEST IMAGE")
print("=" * 65)

from ml.training.inference import OilSpillPredictor
import numpy as np

# Prefer Drive checkpoint for permanence
ckpt_to_use = DRIVE_CKPT if DRIVE_CKPT.exists() else LOCAL_CKPT
print(f"  Loading: {ckpt_to_use}")

predictor = OilSpillPredictor(
    ckpt_path  = ckpt_to_use,
    image_size = args.image_size,
    threshold  = 0.5,
)

# Use the first held-out test image
test_img_dir = PROTO_ROOT / "test" / "images"
test_imgs    = sorted(test_img_dir.glob("*.tif")) + sorted(test_img_dir.glob("*.tiff"))
if not test_imgs:
    raise FileNotFoundError(f"No test images in {test_img_dir}")

test_img_path = test_imgs[0]
test_stem     = test_img_path.stem
print(f"  Test image: {test_img_path.name}")

DETECTION_TIMESTAMP = datetime.now(timezone.utc)
binary_mask, prob_map = predictor.predict(test_img_path)

spill_pixels = int(binary_mask.sum())
total_pixels = int(binary_mask.size)
h, w         = binary_mask.shape

print(f"\n  Detection timestamp: {DETECTION_TIMESTAMP.isoformat()}")
print(f"  Binary mask shape  : {binary_mask.shape}")
print(f"  Oil spill detected : {spill_pixels > 0}")
print(f"  Spill pixels       : {spill_pixels} / {total_pixels}")
print(f"  Coverage           : {100*spill_pixels/total_pixels:.4f}%")
print(f"  Mean confidence    : {float(prob_map.mean()):.4f}")
print(f"  Max  confidence    : {float(prob_map.max()):.4f}")

# Acquisition timestamp from TIFF tags (if present)
ACQUISITION_TIMESTAMP = None
try:
    with rasterio.open(str(test_img_path)) as src:
        tags = src.tags()
        for key in ["ACQUISITION_DATE", "DATE_ACQUIRED", "datetime", "TIFFTAG_DATETIME"]:
            if key in tags:
                try:
                    acq = datetime.strptime(tags[key], "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    ACQUISITION_TIMESTAMP = acq
                    print(f"  Acquisition timestamp (TIFF tag): {ACQUISITION_TIMESTAMP.isoformat()}")
                    break
                except ValueError:
                    pass
except Exception:
    pass

if ACQUISITION_TIMESTAMP is None:
    print("  No acquisition timestamp in TIFF metadata (normal for cropped patches).")

print("=" * 65)


# ============================================================
# PHASE 8 — GEOLOCATION + AREA
# ============================================================
# Honest NULL if TIFF is not georeferenced. NEVER invents coordinates.
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 8: GEOLOCATION + AREA")
print("=" * 65)

latitude    = None
longitude   = None
area_km2    = None
georef_note = "not_checked"

try:
    with rasterio.open(str(test_img_path)) as src:
        transform = src.transform
        crs       = src.crs

    print(f"  CRS       : {crs}")
    print(f"  Transform : {transform}")
    is_identity = (transform.a == 1.0 and transform.b == 0.0 and transform.c == 0.0
                   and transform.d == 0.0 and transform.e == 1.0 and transform.f == 0.0)

    if crs is not None and not is_identity:
        # Georeferenced — extract centroid
        rows, cols = np.where(binary_mask > 0)
        if len(rows) == 0:
            print("  No spill pixels — centroid not applicable.")
            georef_note = "georeferenced_no_spill"
        else:
            c_row = float(rows.mean())
            c_col = float(cols.mean())
            x, y  = rasterio.transform.xy(transform, c_row, c_col)
            print(f"  Centroid (CRS): x={x:.3f}, y={y:.3f}")

            # Reproject to WGS84
            try:
                from pyproj import Transformer
                t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon_r, lat_r = t.transform(x, y)
                latitude  = round(float(lat_r), 6)
                longitude = round(float(lon_r), 6)
                georef_note = "georeferenced_wgs84"
            except ImportError:
                if "4326" in str(crs) or "WGS 84" in str(crs):
                    latitude  = round(float(y), 6)
                    longitude = round(float(x), 6)
                    georef_note = "georeferenced_wgs84_nopyproj"
                else:
                    georef_note = "crs_not_wgs84_pyproj_missing"

            print(f"  latitude  : {latitude}")
            print(f"  longitude : {longitude}")

            # Area
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            if crs.is_geographic:
                lat_rad  = math.radians(latitude or 0)
                px_area  = (res_x * 111_319.5 * math.cos(lat_rad)) * (res_y * 111_132.0)
            else:
                px_area = res_x * res_y   # square meters
            area_km2 = round(spill_pixels * px_area / 1e6, 4)
            print(f"  area_km2  : {area_km2}")

    else:
        # NOT georeferenced — common for cropped Sentinel-1 patches
        print()
        print("  WARNING: TIFF IS NOT GEOREFERENCED.")
        print("  The Sentinel-1 patches in this dataset are pre-cropped sub-images")
        print("  that typically lack CRS/transform metadata.")
        print("  latitude, longitude, and area_km2 will be NULL (honest).")
        print()
        print("  To obtain real coordinates you would need:")
        print("  1. The original full Sentinel-1 scene from ESA Copernicus Hub.")
        print("  2. The crop bounding box metadata from the Zenodo dataset description.")
        print("  3. OR the accompanying CSV/JSON if the dataset authors provided it.")
        print()
        print("  For the SIH demo: report NULL and document this limitation.")
        georef_note = "not_georeferenced"
        latitude = longitude = area_km2 = None

except Exception as e:
    print(f"  Error reading geospatial metadata: {e}")
    georef_note = f"error:{e}"

print(f"\n  latitude  = {latitude}")
print(f"  longitude = {longitude}")
print(f"  area_km2  = {area_km2}")
print(f"  note      = {georef_note}")
print("=" * 65)


# ============================================================
# PHASE 9 — MEMBER 2 STRUCTURED OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 9: MEMBER 2 STRUCTURED OUTPUT")
print("=" * 65)

# Confidence = mean probability over spill pixels (or max if no spill)
spill_vals = prob_map[binary_mask > 0]
confidence = round(float(spill_vals.mean()), 4) if len(spill_vals) > 0 else round(float(prob_map.max()), 4)

# Core output — all values from actual inference, NEVER invented
member2_output = {
    "spill_id":            test_stem,
    "latitude":            latitude,               # None if not georeferenced
    "longitude":           longitude,              # None if not georeferenced
    "detection_timestamp": DETECTION_TIMESTAMP.isoformat(),  # UTC, timezone-aware
    "area_km2":            area_km2,               # None if not georeferenced
    "confidence":          confidence,
}

# Extended metadata (for internal use / debugging)
member2_full = {
    **member2_output,
    "_meta": {
        "oil_spill_detected":    bool(spill_pixels > 0),
        "spill_pixels":          spill_pixels,
        "total_pixels":          total_pixels,
        "coverage_pct":          round(100.0 * spill_pixels / total_pixels, 4),
        "mean_confidence":       round(float(prob_map.mean()), 4),
        "max_confidence":        round(float(prob_map.max()), 4),
        "image_shape_hw":        list(binary_mask.shape),
        "georef_note":           georef_note,
        "acquisition_timestamp": ACQUISITION_TIMESTAMP.isoformat() if ACQUISITION_TIMESTAMP else None,
        "test_image_path":       str(test_img_path),
        "checkpoint_used":       str(ckpt_to_use),
        "model_architecture":    "UNet_in2_out1_512",
        "training_best_epoch":   best_epoch.get("epoch"),
        "training_val_dice":     best_epoch.get("val_dice"),
        "training_val_iou":      best_epoch.get("val_iou"),
    }
}

print("\n  ── CORE OUTPUT (for Member 2) ──")
print(json.dumps(member2_output, indent=4, default=str))
print("\n  ── FULL OUTPUT (with metadata) ──")
print(json.dumps(member2_full, indent=4, default=str))

# Save to Drive
out_path = DRIVE_ROOT / f"member2_output_{test_stem}.json"
with open(out_path, "w") as f:
    json.dump(member2_full, f, indent=2, default=str)
print(f"\n  Saved to Drive: {out_path}")
print("=" * 65)


# ============================================================
# PHASE 10 — API VERIFICATION + PYTEST
# ============================================================

print("\n" + "=" * 65)
print("  PHASE 10: API VERIFICATION + PYTEST")
print("=" * 65)

# 10a — pytest
print("\n  [10a] Running pytest ...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
    cwd=str(PROJECT_ROOT),
    capture_output=True,
    text=True,
)
print(result.stdout[-2000:])  # last 2000 chars
if result.returncode != 0 and result.stderr:
    print("  STDERR:", result.stderr[:500])
print(f"  pytest exit code: {result.returncode} ({'PASS' if result.returncode == 0 else 'FAIL'})")

# 10b — start uvicorn
print("\n  [10b] Starting uvicorn on port 9000 ...")
uvicorn_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app",
     "--host", "0.0.0.0", "--port", "9000", "--log-level", "error"],
    cwd=str(PROJECT_ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(4)

import urllib.request as _ur, urllib.error as _ue

# 10c — GET /api/ml/status
print("\n  [10c] GET /api/ml/status ...")
try:
    with _ur.urlopen("http://localhost:9000/api/ml/status", timeout=8) as r:
        status_data = json.loads(r.read())
    print(f"  model_ready  : {status_data.get('model_ready')}")
    print(f"  message      : {status_data.get('message')}")
    print(f"  model_path   : {status_data.get('model_path')}")
    if not status_data.get("model_ready"):
        print("  NOTE: model_ready=False means the API server started without the checkpoint.")
        print(f"  Copy {LOCAL_CKPT} to the path shown above and restart uvicorn.")
except Exception as e:
    print(f"  FAILED: {e}")

# 10d — POST /api/ml/predict
print("\n  [10d] POST /api/ml/predict ...")
try:
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()
    boundary = "----SIHBoundary7654"
    file_name = test_img_path.name
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
        f"Content-Type: image/tiff\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = _ur.Request(
        "http://localhost:9000/api/ml/predict",
        data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with _ur.urlopen(req, timeout=60) as r:
        predict_data = json.loads(r.read())
    pred  = predict_data.get("prediction", {})
    stats = pred.get("stats", {})
    print(f"  success            : {predict_data.get('success')}")
    print(f"  oil_spill_detected : {pred.get('oil_spill_detected')}")
    print(f"  spill_pixels       : {stats.get('spill_pixels')}")
    print(f"  coverage_pct       : {stats.get('spill_coverage_pct')}")
    print(f"  mean_confidence    : {stats.get('mean_confidence')}")
    print(f"  binary_mask_png    : <base64, {len(pred.get('binary_mask_png',''))} chars>")
    print("  POST /api/ml/predict: SUCCESS")
except _ue.HTTPError as e:
    print(f"  HTTP {e.code}: {e.read().decode()[:400]}")
except Exception as e:
    print(f"  FAILED: {e}")
finally:
    uvicorn_proc.terminate()
    time.sleep(1)

print("\n" + "=" * 65)
print("  PHASE 10 COMPLETE.")
print("=" * 65)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("  FINAL REPORT — SIH 2026 PS-26143 Member 1")
print("=" * 65)

def _safe(expr, default="N/A"):
    try:
        return eval(expr)
    except Exception:
        return default

print(f"""
DATA
────
  Source            : {IMAGES_RAW_DIR}
  Total pairs       : {len(valid_pairs) if 'valid_pairs' in dir() else 'N/A'}
  Train / Val / Test: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}
  Image shape       : (2, H, W) float32 — VV + VH SAR
  Mask  shape       : (1, H, W) float32 — binary
  Drive storage     : {PROTO_ROOT}

MODEL
─────
  Architecture      : U-Net (2-channel SAR -> 1-channel logits)
  in_channels       : 2
  image_size        : 512
  Best epoch        : {best_epoch.get('epoch', 'N/A')}
  Best val Dice     : {best_epoch.get('val_dice', 0):.4f}
  Best val IoU      : {best_epoch.get('val_iou', 0):.4f}
  Checkpoint local  : {LOCAL_CKPT}
  Checkpoint Drive  : {DRIVE_CKPT}

INFERENCE
─────────
  Test image ID     : {test_stem}
  Oil spill detected: {spill_pixels > 0}
  Spill pixels      : {spill_pixels} / {total_pixels}
  Coverage          : {100*spill_pixels/total_pixels:.4f}%
  Confidence        : {confidence}
  Latitude          : {latitude}
  Longitude         : {longitude}
  area_km2          : {area_km2}
  Detection ts (UTC): {DETECTION_TIMESTAMP.isoformat()}
  Georef note       : {georef_note}

MEMBER 2 OUTPUT
───────────────
  spill_id           : {member2_output.get('spill_id')}
  latitude           : {member2_output.get('latitude')}
  longitude          : {member2_output.get('longitude')}
  detection_timestamp: {member2_output.get('detection_timestamp')}
  area_km2           : {member2_output.get('area_km2')}
  confidence         : {member2_output.get('confidence')}

PERMANENCE
──────────
  prototype/ exists  : {PROTO_ROOT.exists() if 'PROTO_ROOT' in dir() else 'N/A'}
  checkpoint exists  : {DRIVE_CKPT.exists() if 'DRIVE_CKPT' in dir() else 'N/A'}
""")
print("=" * 65)
print("  Pipeline complete.")
print("=" * 65)
