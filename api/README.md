# SAR Oil-Spill Detection API

> SIH 2026 — Problem Statement 26143  
> Member 1 scope: Satellite Imagery + AI Detection

REST API built with **FastAPI** that exposes the trained U-Net segmentation model for oil-spill detection in Sentinel-1 SAR imagery.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OIL_SPILL_MODEL_PATH to your trained checkpoint
```

### 3. Start the server

```bash
# Development (auto-reload on file changes)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Open auto-generated docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OIL_SPILL_MODEL_PATH` | `ml/training/checkpoints/best_unet.pth` | **Required for predictions.** Path to the trained U-Net checkpoint. |
| `OIL_SPILL_IMAGE_SIZE` | `512` | Image size the model was trained on. Must match training config. |
| `OIL_SPILL_THRESHOLD` | `0.5` | Sigmoid probability threshold for binary classification. |
| `MAX_UPLOAD_MB` | `200` | Maximum allowed upload size in MB. |

> **The server starts and responds to all routes even when `OIL_SPILL_MODEL_PATH` does not exist.**  
> `POST /api/ml/predict` returns HTTP 503 with a clear message until the model is trained.

---

## Endpoints

### `GET /api/ml/status`

Check whether the model is loaded and ready.

```bash
curl http://localhost:8000/api/ml/status
```

**Response — model ready:**
```json
{
  "model_ready": true,
  "message": "Model is loaded and ready.",
  "model_path": "ml/training/checkpoints/best_unet.pth",
  "image_size": 512,
  "threshold": 0.5
}
```

**Response — model not trained yet:**
```json
{
  "model_ready": false,
  "message": "Model checkpoint not found at 'ml/training/checkpoints/best_unet.pth'. Train the model first with: python ml/training/train.py",
  "model_path": "ml/training/checkpoints/best_unet.pth",
  "hint": "Train the model first with:  python ml/training/train.py  then restart the server, or set OIL_SPILL_MODEL_PATH to your checkpoint."
}
```

---

### `POST /api/ml/predict`

Predict oil spills in a Sentinel-1 SAR TIFF image.

**Request format:** `multipart/form-data`  
**Field name:** `file`  
**Accepted types:** `.tif`, `.tiff`  
**Input requirements:** 2-channel float32 TIFF (VV + VH polarisation)

#### Example — curl

```bash
curl -X POST http://localhost:8000/api/ml/predict \
  -F "file=@/path/to/sentinel1_scene.tif"
```

#### Example — Python

```python
import requests

with open("/path/to/sentinel1_scene.tif", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/ml/predict",
        files={"file": ("scene.tif", f, "image/tiff")},
    )
print(resp.json())
```

#### Example — JavaScript (fetch)

```javascript
const form = new FormData();
form.append("file", tiffFile);

const resp = await fetch("http://localhost:8000/api/ml/predict", {
  method: "POST",
  body: form,
});
const data = await resp.json();

// Render the binary mask:
document.getElementById("mask").src =
  `data:image/png;base64,${data.prediction.binary_mask_png}`;
```

---

#### Successful Response (HTTP 200)

```json
{
  "success": true,
  "filename": "sentinel1_scene.tif",
  "prediction": {
    "oil_spill_detected": true,
    "stats": {
      "image_height": 2048,
      "image_width": 2048,
      "total_pixels": 4194304,
      "spill_pixels": 83621,
      "spill_coverage_pct": 1.9937,
      "mean_confidence": 0.134,
      "max_confidence": 0.987,
      "threshold_used": 0.5
    },
    "binary_mask_png": "<base64-encoded PNG string>",
    "prob_map_png": "<base64-encoded PNG string>"
  }
}
```

**Render the masks in HTML:**
```html
<!-- Binary oil-spill mask (white = oil, black = background) -->
<img src="data:image/png;base64,{binary_mask_png}">

<!-- Probability heatmap (blue = low, red = high) -->
<img src="data:image/png;base64,{prob_map_png}">
```

---

#### Error Responses

| HTTP | `error.code` | Cause |
|---|---|---|
| `422` | `MISSING_FILENAME` | No file attached to the request |
| `422` | `EMPTY_FILE` | Uploaded file has 0 bytes |
| `413` | `FILE_TOO_LARGE` | File exceeds `MAX_UPLOAD_MB` |
| `415` | `UNSUPPORTED_FILE_TYPE` | Not a `.tif` / `.tiff` file |
| `400` | `INVALID_TIFF` | TIFF is not 2-channel, or is corrupt |
| `500` | `INFERENCE_ERROR` | Unexpected model inference failure |
| `503` | `MODEL_NOT_AVAILABLE` | Checkpoint not found / model not trained yet |

**Example 503 response (model not trained):**
```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_AVAILABLE",
    "message": "Model checkpoint not found at 'ml/training/checkpoints/best_unet.pth'. Train the model first with: python ml/training/train.py then restart the API server, or set OIL_SPILL_MODEL_PATH to the correct checkpoint path."
  }
}
```

---

## Running Tests

```bash
# All tests (no model, no dataset required)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=api --cov-report=term-missing
```

---

## Architecture

```
api/
├── main.py            FastAPI app, CORS, health endpoints
├── config.py          Environment variable settings
├── dependencies.py    Lazy OilSpillPredictor singleton
├── routers/
│   └── ml.py          POST /api/ml/predict  +  GET /api/ml/status
├── services/
│   └── prediction.py  Business logic (temp files, inference call, PNG encoding)
└── schemas/
    └── prediction.py  Pydantic request/response models
```

**The API never duplicates ML logic.** All inference is delegated to:
```python
from ml.training.inference import OilSpillPredictor
```

---

## Training the Model First

The API requires a trained checkpoint before predictions are possible.

```bash
# Train on the prototype dataset (run in Colab or locally with GPU)
python ml/training/train.py \
    --data-dir /content/sih26143/prototype \
    --epochs 50

# Checkpoint saved to:
#   ml/training/checkpoints/best_unet.pth

# Restart the API server to pick up the new checkpoint
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
