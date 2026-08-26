# Dataset Directory — Placement & Download Guide

> **STORAGE RULE:** Do NOT download files automatically.
> Do NOT extract files larger than 1 GB without user intent.
> Only use this directory for files you have manually downloaded.

---

## Required Directory Layout

```
ml/dataset/
├── raw/                          ← extract all Zenodo archives here
│   ├── 01_Train_Val_Oil_Spill_images/    (from Part I)
│   ├── 01_Train_Val_Oil_Spill_mask/      (from Part I)
│   ├── 01_Train_Val_No_Oil_Images/       (from Part II)
│   ├── 01_Train_Val_No_Oil_mask/         (from Part II)
│   ├── 01_Train_Val_Lookalike_images/    (from Part II)
│   ├── 01_Train_Val_Lookalike_mask/      (from Part II)
│   └── <Part III folders>/               (from Part III)
├── prototype/                    ← created automatically by prepare_prototype_dataset.py
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── prepare_prototype_dataset.py  ← run this after extraction
└── README.md                     ← this file
```

---

## Step-by-Step Download Instructions

### Files to Download (Manual — browser or wget)

Each file is a **7-Zip archive** (`.7z`). You need **7-Zip** or **p7zip** to extract.

#### Part I — Oil Spill Images + Masks (1 200 pairs)
**Zenodo:** https://zenodo.org/records/8346860

| File | Contents | Approx. Size |
|---|---|---|
| `01_Train_Val_Oil_Spill_images.7z` | 1 200 SAR images (2048×2048×2 float32 TIFF) | ~9–12 GB compressed |
| `01_Train_Val_Oil_Spill_mask.7z` | 1 200 binary masks (2048×2048 uint8 TIFF) | ~0.5–1 GB compressed |

**Direct download links:**
```
https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z?download=1
https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z?download=1
```

---

#### Part II — No-Oil + Lookalike Images + Masks (685 + 685 pairs)
**Zenodo:** https://zenodo.org/records/8253899

| File | Contents | Approx. Size |
|---|---|---|
| `01_Train_Val_No_Oil_Images.7z` | 685 oil-free SAR images | ~5–7 GB compressed |
| `01_Train_Val_No_Oil_mask.7z` | 685 no-oil masks (all zeros) | ~0.3 GB compressed |
| `01_Train_Val_Lookalike_images.7z` | 685 look-alike SAR images | ~5–7 GB compressed |
| `01_Train_Val_Lookalike_mask.7z` | 685 look-alike masks (all zeros) | ~0.3 GB compressed |

**Direct download links:**
```
https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_Images.7z?download=1
https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_mask.7z?download=1
https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_images.7z?download=1
https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_mask.7z?download=1
```

---

#### Part III — Test Images (completely unseen)
**Zenodo:** https://zenodo.org/records/13761290
*(Visit the page manually to see the file list — the page was unreachable at time of writing.)*

---

## Minimum Download for Prototype

**You do NOT need all three parts for the initial prototype.**

For 400 train + 75 val + 75 test = **550 samples**, you only need:

| Priority | File | Why |
|---|---|---|
| **Required** | `01_Train_Val_Oil_Spill_images.7z` (Part I) | Primary oil-spill images |
| **Required** | `01_Train_Val_Oil_Spill_mask.7z` (Part I) | Corresponding masks |
| Optional | Part II No-Oil / Lookalike files | Adds negative examples |
| Optional | Part III | Held-out test set |

**Minimum storage needed (oil-spill only):**
- Compressed: ~10–13 GB
- After extraction: ~40–50 GB (each 2048×2048×2 float32 TIFF ≈ 32 MB uncompressed)

**Recommended free space before starting: ≥ 60 GB**

---

## Extraction Instructions

### Windows (7-Zip GUI)
1. Right-click the `.7z` file → **7-Zip** → **Extract Here**
2. Move the extracted folder into `ml/dataset/raw/`

### Windows / Linux (command line)
```bash
# Install 7-Zip (Ubuntu/Debian)
sudo apt install p7zip-full

# Extract
7z x 01_Train_Val_Oil_Spill_images.7z -o"ml/dataset/raw/"
7z x 01_Train_Val_Oil_Spill_mask.7z   -o"ml/dataset/raw/"
```

### Do NOT rename any extracted folders.

---

## After Extraction — Run the Prototype Preparation Script

```bash
# From project root (dry run first to verify):
python ml/dataset/prepare_prototype_dataset.py --dry-run

# Then run for real:
python ml/dataset/prepare_prototype_dataset.py
```

This will create `ml/dataset/prototype/` with clean train/val/test splits.
No original files are modified.

---

## Image Format Details

| Property | Value |
|---|---|
| Image shape | 2048 × 2048 × 2 (two SAR polarisation channels) |
| Image dtype | float32 (Sigma0 in dB) |
| Mask shape | 2048 × 2048 |
| Mask dtype | uint8 |
| Mask values | 0 = background, 1 = foreground |
| Mask semantics | Oil spill = 1 only for Part I; Parts II & III masks are all 0 |
| File format | GeoTIFF (`.tif`) |

---

## Git — This Directory is Ignored

Raw data and prototype images are listed in `.gitignore`.
Only this `README.md` and `prepare_prototype_dataset.py` are committed.
