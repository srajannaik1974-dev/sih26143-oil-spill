# ML Module — SIH 2026 Problem Statement 26143
## Satellite Imagery + Oil Spill AI Detection

> **Scope (Member 1):** Satellite imagery ingestion, preprocessing, and AI-based oil spill detection.
> AIS correlation, vessel attribution, backend, and frontend are **out of scope** for this module.

---

## Directory Structure

```
ml/
├── dataset/              # Raw / processed dataset storage (NOT committed to git)
│   └── README.md         # Instructions for placing the dataset
├── preprocessing/        # Dataset preprocessing utilities
├── training/             # Model training scripts (future)
├── inference/            # Inference / prediction scripts (future)
├── notebooks/            # Jupyter / Colab notebooks
│   └── 01_dataset_inspection.ipynb   ← START HERE
└── README.md             # This file
```

---

## Dataset

The Sentinel-1 SAR oil-spill dataset is provided through **Zenodo**.

- Download it manually from Zenodo and place it inside `ml/dataset/`.
- See [`ml/dataset/README.md`](dataset/README.md) for exact placement instructions.

---

## Quick Start

### Step 1 — Place the dataset
Follow the instructions in `ml/dataset/README.md`.

### Step 2 — Run the inspection notebook
Open and run `ml/notebooks/01_dataset_inspection.ipynb` in:
- **Google Colab** (recommended) — upload or mount your Drive
- **JupyterLab / VS Code** — run locally

The notebook will:
- Auto-detect the dataset layout
- List all image and mask files
- Report image dimensions, channels, and dtypes
- Display sample images + mask overlays
- Summarise any missing / unmatched pairs

### Step 3 — (Future) Preprocessing → Training → Inference
Scripts in `preprocessing/`, `training/`, and `inference/` will be populated in later phases.

---

## Dependencies

```
numpy
rasterio
matplotlib
Pillow
tqdm
```

Install with:
```bash
pip install numpy rasterio matplotlib Pillow tqdm
```

---

## Notes

- Sentinel-1 images are **SAR (Synthetic Aperture Radar)**, not RGB.
- Do **not** assume 3 channels — inspect the actual data first.
- Do **not** assume binary masks — unique mask values are reported during inspection.
