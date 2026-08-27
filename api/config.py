"""api/config.py
================
SIH 2026 — PS 26143: SAR Oil-Spill Detection API
Centralised configuration loaded from environment variables / .env file.

All settings have safe defaults so the server can start without a .env file.
The only value that MUST be set to run predictions is OIL_SPILL_MODEL_PATH.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if it exists (no-op otherwise — pure env vars work too)
load_dotenv()


class Settings:
    """
    Application settings read from environment variables.
    Instantiated once at module import time and re-used everywhere.
    """

    # ── Model ──────────────────────────────────────────────────────────────────
    # Path to the trained U-Net checkpoint (.pth file).
    # If the file does not exist at request time, the /predict endpoint
    # returns HTTP 503 with a clear message instead of crashing.
    model_path: Path = Path(
        os.environ.get("OIL_SPILL_MODEL_PATH", "ml/training/checkpoints/best_unet.pth")
    )

    # Image size the model was trained on (must match training config).
    image_size: int = int(os.environ.get("OIL_SPILL_IMAGE_SIZE", "512"))

    # Sigmoid probability threshold for binary oil-spill classification.
    threshold: float = float(os.environ.get("OIL_SPILL_THRESHOLD", "0.5"))

    # ── Upload ─────────────────────────────────────────────────────────────────
    # Maximum allowed upload size in bytes.
    max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024

    # Allowed TIFF file extensions (case-insensitive).
    allowed_extensions: frozenset[str] = frozenset({".tif", ".tiff"})


# Single shared instance — import this everywhere.
settings = Settings()
