# SIH 2026 — Problem Statement 26143
## Leveraging Satellite Imagery to Detect Oil Spills at Sea

> **Member 1 scope:** Satellite Imagery + AI Oil-Spill Detection

---

## Project Structure

```
sih26143-oil-spill/
├── api/                   ← REST API (FastAPI)
│   ├── main.py            ← Server entry point
│   ├── config.py          ← Environment-variable settings
│   ├── dependencies.py    ← Lazy model singleton
│   ├── routers/ml.py      ← POST /api/ml/predict
│   ├── services/          ← Prediction business logic
│   ├── schemas/           ← Pydantic models
│   └── README.md          ← Full API documentation
├── ml/                    ← Machine learning pipeline
│   ├── training/          ← U-Net, dataset, train, evaluate, inference
│   ├── preprocessing/     ← Dataset inspector
│   ├── dataset/           ← Prototype split utilities
│   └── notebooks/         ← Colab notebooks
├── tests/                 ← API unit tests (no model/dataset required)
├── requirements.txt
├── .env.example           ← Copy to .env and configure
└── README.md
```

---

## Quick Start — API

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set OIL_SPILL_MODEL_PATH to your trained checkpoint

# 3. Start server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 4. Test prediction
curl -X POST http://localhost:8000/api/ml/predict \
  -F "file=@/path/to/sentinel1_scene.tif"

# 5. Check model status
curl http://localhost:8000/api/ml/status
```

📖 See [`api/README.md`](api/README.md) for full endpoint documentation.

---

## Quick Start — Training

```bash
# Train U-Net on the Sentinel-1 prototype dataset (Colab recommended)
python ml/training/train.py \
    --data-dir /content/sih26143/prototype \
    --epochs 50 --batch-size 4

# Checkpoint saved to: ml/training/checkpoints/best_unet.pth
# Set OIL_SPILL_MODEL_PATH=ml/training/checkpoints/best_unet.pth in .env
```

📖 See [`ml/training/README.md`](ml/training/README.md) for training details.

---

## Running Tests

```bash
# All tests — no trained model or dataset required
pytest tests/ -v
```

---

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `OIL_SPILL_MODEL_PATH` | Path to trained `.pth` checkpoint |
| `OIL_SPILL_THRESHOLD` | Binary classification threshold (default `0.5`) |
| `MAX_UPLOAD_MB` | Max TIFF upload size (default `200` MB) |

See [`.env.example`](.env.example) for the full list.