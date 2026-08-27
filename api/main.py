"""api/main.py
==============
SIH 2026 — PS 26143: SAR Oil-Spill Detection API
FastAPI application entry point.

Start the server
----------------
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Auto-generated docs
-------------------
    http://localhost:8000/docs      Swagger UI
    http://localhost:8000/redoc     ReDoc
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Ensure project root is importable (for ml.training.* imports) ─────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SIH 2026 PS-26143: SAR Oil-Spill Detection API",
    description = (
        "REST API for detecting oil spills in Sentinel-1 SAR satellite imagery "
        "using a trained U-Net segmentation model.\n\n"
        "**Note:** POST `/api/ml/predict` requires a trained model checkpoint. "
        "Use GET `/api/ml/status` to check model availability."
    ),
    version     = "0.1.0",
    contact     = {
        "name": "SIH 2026 Team — Member 1 (Satellite Imagery + AI Detection)",
    },
    license_info = {"name": "MIT"},
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Permissive for development (all origins, all methods).
# Tighten this to specific frontend origins before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # TODO: restrict to frontend origin in production
    allow_credentials = False,   # Must be False when allow_origins=["*"]
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from api.routers.ml import router as ml_router  # noqa: E402

app.include_router(ml_router)


# ── Root health-check ─────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root() -> dict:
    """Basic health-check endpoint. Returns API name and version."""
    return {
        "api":     "SIH 2026 PS-26143 SAR Oil-Spill Detection",
        "version": "0.1.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health() -> dict:
    """Liveness probe — always returns 200 if the server is running."""
    return {"status": "ok"}
