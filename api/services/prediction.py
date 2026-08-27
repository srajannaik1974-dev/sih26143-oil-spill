"""api/services/prediction.py
==============================
SIH 2026 — PS 26143: SAR Oil-Spill Detection API
Business logic layer: runs inference and builds the API response.

Responsibilities
----------------
1. Save the uploaded bytes to a temporary file (deleted in finally).
2. Call OilSpillPredictor.predict() — never duplicates inference logic.
3. Compute statistics from the returned binary_mask and prob_map arrays.
4. PNG-encode both arrays to base64 strings suitable for JSON / <img> tags.
5. Return a PredictionResult pydantic model.

Why base64 PNG?
---------------
- A 2048×2048 float32 array = 16 MB of JSON numbers — impractical.
- A PNG-encoded binary mask is typically < 50 KB (1-bit lossless).
- The probability heatmap PNG is < 200 KB.
- Both are immediately renderable in any browser:
    <img src="data:image/png;base64,{value}">
- No second HTTP round-trip needed for the mask image.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

from api.schemas.prediction import PredictionResult, PredictionStats

if TYPE_CHECKING:
    from ml.training.inference import OilSpillPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PNG encoding helpers
# ---------------------------------------------------------------------------

def _mask_to_png_b64(binary_mask: np.ndarray) -> str:
    """
    Convert a (H, W) uint8 binary mask {0, 1} to a base64-encoded PNG string.
    Oil-spill pixels (1) → white (255).
    Background pixels (0) → black (0).
    """
    # Scale 0/1 → 0/255 for PNG encoding
    img_array = (binary_mask * 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_array, mode="L")  # grayscale
    buf       = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _prob_map_to_png_b64(prob_map: np.ndarray) -> str:
    """
    Convert a (H, W) float32 probability map [0, 1] to a base64-encoded PNG
    using the 'jet' colormap (blue=low, red=high) so it is visually meaningful.
    """
    import matplotlib
    matplotlib.use("Agg")            # non-interactive backend — safe in API context
    import matplotlib.pyplot as plt  # imported here so matplotlib is optional for tests

    # Map [0, 1] → RGBA via jet colormap
    cmap     = plt.cm.jet
    rgba_arr = cmap(prob_map)               # (H, W, 4) float64 in [0, 1]
    rgb_arr  = (rgba_arr[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)

    pil_img = Image.fromarray(rgb_arr, mode="RGB")
    buf     = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------

def run_prediction(
    file_bytes: bytes,
    filename:   str,
    predictor:  "OilSpillPredictor",
) -> PredictionResult:
    """
    Save file_bytes to a temporary TIFF, run inference, return PredictionResult.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded TIFF file.
    filename : str
        Original filename from the upload (used for the temp file suffix).
    predictor : OilSpillPredictor
        The loaded predictor instance from api/dependencies.py.

    Returns
    -------
    PredictionResult
        Structured prediction result with stats + base64 PNG masks.

    Raises
    ------
    ValueError
        If the TIFF does not have exactly 2 SAR channels.
    RuntimeError
        If inference fails for any other reason.
    """
    # Determine suffix — rasterio requires the correct extension to select driver
    suffix = Path(filename).suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        suffix = ".tif"

    tmp_path: Optional[Path] = None
    try:
        # ── 1. Write to temp file ────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        logger.info("Running inference on temp file: %s (%d bytes)", tmp_path, len(file_bytes))

        # ── 2. Run inference (calls existing OilSpillPredictor, no duplication) ─
        binary_mask, prob_map = predictor.predict(tmp_path)
        # binary_mask : np.ndarray (H, W) uint8   {0, 1}
        # prob_map    : np.ndarray (H, W) float32 [0.0, 1.0]

        # ── 3. Compute statistics ─────────────────────────────────────────────
        h, w          = binary_mask.shape
        total_pixels  = h * w
        spill_pixels  = int(binary_mask.sum())
        coverage_pct  = round(100.0 * spill_pixels / total_pixels, 4)
        mean_conf     = round(float(np.mean(prob_map)), 6)
        max_conf      = round(float(np.max(prob_map)),  6)

        stats = PredictionStats(
            image_height       = h,
            image_width        = w,
            total_pixels       = total_pixels,
            spill_pixels       = spill_pixels,
            spill_coverage_pct = coverage_pct,
            mean_confidence    = mean_conf,
            max_confidence     = max_conf,
            threshold_used     = predictor.threshold,
        )

        # ── 4. Encode masks as base64 PNGs ────────────────────────────────────
        binary_png = _mask_to_png_b64(binary_mask)
        prob_png   = _prob_map_to_png_b64(prob_map)

        logger.info(
            "Prediction complete: spill=%s, coverage=%.2f%%, pixels=%d/%d",
            spill_pixels > 0, coverage_pct, spill_pixels, total_pixels,
        )

        return PredictionResult(
            oil_spill_detected = spill_pixels > 0,
            stats              = stats,
            binary_mask_png    = binary_png,
            prob_map_png       = prob_png,
        )

    except (ValueError, FileNotFoundError):
        # Let the router convert these to 422 / 400 responses
        raise

    except Exception as exc:
        logger.exception("Unexpected inference error for file '%s'", filename)
        raise RuntimeError(f"Inference failed: {exc}") from exc

    finally:
        # ── 5. Always clean up the temp file ─────────────────────────────────
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
                logger.debug("Deleted temp file: %s", tmp_path)
            except OSError as cleanup_err:
                logger.warning("Could not delete temp file %s: %s", tmp_path, cleanup_err)

