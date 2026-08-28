"""api/routers/ml.py
====================
SIH 2026 — PS 26143: SAR Oil-Spill Detection API
Router for all ML-related endpoints.

Endpoints
---------
POST /api/ml/predict
    Accept a Sentinel-1 SAR TIFF image via multipart/form-data.
    Returns oil-spill detection results including statistics and PNG masks.

GET  /api/ml/status
    Returns the current model availability status.
    Useful for frontend health-checks before enabling the upload UI.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.config      import settings
import api.dependencies as deps
from api.schemas.prediction import ErrorDetail, ErrorResponse, PredictResponse
from api.services.prediction import run_prediction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ML Prediction"])


# ---------------------------------------------------------------------------
# Helper: validate upload
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile) -> None:
    """
    Raise HTTPException for invalid uploads before reading the file content.

    Checks:
    1. A file was actually attached (filename is not empty).
    2. The file extension is .tif or .tiff (case-insensitive).
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="MISSING_FILENAME",
                    message="No file was provided in the request.",
                )
            ).model_dump(),
        )

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="UNSUPPORTED_FILE_TYPE",
                    message=(
                        f"Only Sentinel-1 SAR TIFF files are accepted "
                        f"(extensions: {', '.join(sorted(settings.allowed_extensions))}). "
                        f"Received: '{file.filename}'."
                    ),
                )
            ).model_dump(),
        )


# ---------------------------------------------------------------------------
# GET /api/ml/status
# ---------------------------------------------------------------------------

@router.get(
    "/status",
    summary="Model availability status",
    response_description="Current state of the oil-spill model.",
)
async def model_status() -> dict:
    """
    Returns whether the model is loaded and ready for inference.
    Use this endpoint to check model availability before uploading an image.
    """
    predictor = deps.get_predictor()
    error_msg = deps.get_predictor_error()

    if predictor is not None:
        return {
            "model_ready": True,
            "message": "Model is loaded and ready.",
            "model_path": str(settings.model_path),
            "image_size": settings.image_size,
            "threshold":  settings.threshold,
        }
    else:
        return {
            "model_ready": False,
            "message": error_msg or "Model is not available.",
            "model_path": str(settings.model_path),
            "hint": (
                "Train the model first with:  python ml/training/train.py  "
                "then restart the server, or set OIL_SPILL_MODEL_PATH to your checkpoint."
            ),
        }


# ---------------------------------------------------------------------------
# POST /api/ml/predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict oil spill from a Sentinel-1 SAR TIFF",
    responses={
        200: {"description": "Prediction successful."},
        400: {"model": ErrorResponse, "description": "Invalid or unsupported TIFF file."},
        413: {"model": ErrorResponse, "description": "Uploaded file exceeds the size limit."},
        415: {"model": ErrorResponse, "description": "Unsupported file type."},
        422: {"model": ErrorResponse, "description": "Validation error (missing file, etc.)."},
        500: {"model": ErrorResponse, "description": "Unexpected inference error."},
        503: {"model": ErrorResponse, "description": "Model not available — not yet trained."},
    },
)
async def predict(
    file: Annotated[
        UploadFile,
        File(description="A 1-channel Sentinel-1 SAR TIFF file (VV band)."),
    ],
) -> PredictResponse:
    """
    **POST /api/ml/predict**

    Upload a Sentinel-1 SAR TIFF image and receive oil-spill detection results.

    **Required file format:**
    - TIFF with exactly **2 bands** (VV polarisation channel 0, VH channel 1)
    - dtype: float32, values approximately -48 to +11 dB
    - Typical size: 2048 × 2048 pixels

    **Response includes:**
    - `oil_spill_detected` — boolean
    - `stats` — coverage percentage, pixel counts, confidence scores
    - `binary_mask_png` — base64 PNG of the binary oil-spill mask
    - `prob_map_png` — base64 PNG of the sigmoid probability heatmap

    **Render masks in HTML:**
    ```html
    <img src="data:image/png;base64,{binary_mask_png}">
    ```

    **Error responses:**
    - `503` — model not yet trained / checkpoint missing
    - `415` — wrong file type (not a TIFF)
    - `413` — file too large
    - `400` — TIFF has wrong channel count or is corrupted
    - `500` — unexpected inference failure
    """
    # ── 1. Validate filename / extension ────────────────────────────────────
    _validate_upload(file)

    # ── 2. Check model availability ──────────────────────────────────────────
    predictor = deps.get_predictor()
    if predictor is None:
        error_msg = deps.get_predictor_error() or "Model checkpoint not found."
        logger.warning("Prediction requested but model is unavailable: %s", error_msg)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="MODEL_NOT_AVAILABLE",
                    message=error_msg,
                )
            ).model_dump(),
        )

    # ── 3. Read file bytes + enforce size limit ───────────────────────────────
    file_bytes = await file.read()

    if len(file_bytes) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        actual_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="FILE_TOO_LARGE",
                    message=(
                        f"Uploaded file is {actual_mb:.1f} MB, "
                        f"but the limit is {limit_mb} MB. "
                        "Set MAX_UPLOAD_MB environment variable to increase the limit."
                    ),
                )
            ).model_dump(),
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="EMPTY_FILE",
                    message="The uploaded file is empty.",
                )
            ).model_dump(),
        )

    # ── 4. Run prediction (service handles temp file + cleanup) ───────────────
    try:
        result = run_prediction(
            file_bytes = file_bytes,
            filename   = file.filename,
            predictor  = predictor,
        )
    except ValueError as exc:
        # Wrong channel count or corrupt TIFF — user error
        logger.warning("Invalid TIFF from '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_TIFF",
                    message=str(exc),
                )
            ).model_dump(),
        )
    except RuntimeError as exc:
        # Unexpected inference failure — server error
        logger.error("Inference runtime error for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="INFERENCE_ERROR",
                    message=str(exc),
                )
            ).model_dump(),
        )

    # ── 5. Return structured response ─────────────────────────────────────────
    return PredictResponse(
        success    = True,
        filename   = file.filename,
        prediction = result,
    )
