"""api/dependencies.py
======================
SIH 2026 — PS 26143: SAR Oil-Spill Detection API
FastAPI dependency: lazy singleton OilSpillPredictor.

Why lazy?
---------
OilSpillPredictor.__init__ raises FileNotFoundError when the checkpoint is
missing. If we loaded the predictor at startup we would crash the server every
time the model has not yet been trained.

Instead we load it on first request and cache it. This means:
- The server starts and stays alive even without a trained model.
- POST /api/ml/predict returns HTTP 503 with a clear message.
- Once the model file appears (after training), the next request loads it
  without restarting the server.

Thread safety
-------------
FastAPI runs synchronous dependencies in a threadpool. The double-checked lock
pattern (read → check → write under lock) prevents two simultaneous first
requests from each constructing a predictor.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure the project root is importable so ml.training.inference works
# regardless of the working directory the server is started from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.config import settings

# Imported here so we can patch it easily in tests without importing torch
try:
    from ml.training.inference import OilSpillPredictor as _OilSpillPredictor
    _PREDICTOR_CLASS_AVAILABLE = True
except ImportError as exc:
    _PREDICTOR_CLASS_AVAILABLE = False
    logger.warning(
        "Could not import OilSpillPredictor (%s). "
        "POST /api/ml/predict will return 503 until the ml package is available.",
        exc,
    )
    _OilSpillPredictor = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Lazy singleton state
# ---------------------------------------------------------------------------

_predictor_lock:    threading.Lock           = threading.Lock()
_predictor_instance: Optional[object]        = None   # OilSpillPredictor | None
_predictor_load_error: Optional[str]         = None   # error message if load failed


def get_predictor() -> Optional[object]:
    """
    FastAPI dependency that returns the shared OilSpillPredictor instance,
    or None if the model checkpoint is not available.

    On first call:
    - Checks if OIL_SPILL_MODEL_PATH exists.
    - If yes: constructs and caches OilSpillPredictor.
    - If no: caches None and records the reason.

    On subsequent calls: returns the cached result immediately (no I/O).

    To reload after training: restart the server (or call _reset_predictor()
    from tests).
    """
    global _predictor_instance, _predictor_load_error

    # Fast path: already loaded or already determined unavailable
    if _predictor_instance is not None or _predictor_load_error is not None:
        return _predictor_instance

    # Slow path: first call — use lock to prevent duplicate construction
    with _predictor_lock:
        # Re-check inside lock (another thread may have beaten us here)
        if _predictor_instance is not None or _predictor_load_error is not None:
            return _predictor_instance

        if not _PREDICTOR_CLASS_AVAILABLE:
            _predictor_load_error = (
                "The ml package could not be imported. "
                "Ensure the project root is on PYTHONPATH and all ML dependencies are installed."
            )
            logger.error(_predictor_load_error)
            return None

        model_path = settings.model_path
        if not model_path.exists():
            _predictor_load_error = (
                f"Model checkpoint not found at '{model_path}'. "
                "Train the model first with:  python ml/training/train.py  "
                "then restart the API server, or set OIL_SPILL_MODEL_PATH "
                "to the correct checkpoint path."
            )
            logger.warning(_predictor_load_error)
            return None

        # Model file exists — try to load
        try:
            logger.info("Loading OilSpillPredictor from %s …", model_path)
            _predictor_instance = _OilSpillPredictor(
                ckpt_path  = model_path,
                image_size = settings.image_size,
                threshold  = settings.threshold,
            )
            logger.info("OilSpillPredictor loaded successfully.")
        except Exception as exc:
            _predictor_load_error = (
                f"Failed to load model checkpoint from '{model_path}': {exc}"
            )
            logger.error(_predictor_load_error)
            _predictor_instance = None

    return _predictor_instance


def get_predictor_error() -> Optional[str]:
    """Return the recorded load error message, or None if no error."""
    return _predictor_load_error


def _reset_predictor() -> None:
    """
    Reset the singleton — used in tests to simulate loading a fresh predictor
    or to force a reload after changing OIL_SPILL_MODEL_PATH.
    Do NOT call this in production code.
    """
    global _predictor_instance, _predictor_load_error
    with _predictor_lock:
        _predictor_instance  = None
        _predictor_load_error = None
