"""api/schemas/prediction.py
============================
SIH 2026 — PS 26143: SAR Oil-Spill Detection API
Pydantic request/response models for the /predict endpoint.

Design notes
------------
- binary_mask_png  : base64-encoded PNG of the binary (0/1) mask.
                     Clients render as: <img src="data:image/png;base64,{value}">
- prob_map_png     : base64-encoded PNG of the probability heatmap (jet colormap).
- Raw numpy arrays are NOT included in the response — a 2048×2048 float32 array
  is 16 MB of JSON numbers, which is neither practical nor browser-friendly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionStats(BaseModel):
    """Numerical summary derived from binary_mask and prob_map."""

    image_height: int = Field(..., description="Height of the analysed SAR image (pixels).")
    image_width:  int = Field(..., description="Width of the analysed SAR image (pixels).")
    total_pixels: int = Field(..., description="Total pixel count (height × width).")

    spill_pixels: int   = Field(..., description="Pixels classified as oil spill.")
    spill_coverage_pct: float = Field(
        ..., description="Oil-spill coverage as a percentage of total image area."
    )

    mean_confidence: float = Field(
        ..., description="Mean sigmoid probability across all pixels (0–1)."
    )
    max_confidence:  float = Field(
        ..., description="Maximum sigmoid probability in the image (0–1)."
    )
    threshold_used:  float = Field(
        ..., description="Probability threshold used to binarise the mask."
    )


class PredictionResult(BaseModel):
    """Full prediction output for a single SAR image."""

    oil_spill_detected: bool = Field(
        ..., description="True if at least one pixel was classified as oil spill."
    )
    stats: PredictionStats

    # PNG masks encoded as base64 strings.
    # Frontend usage:  <img src="data:image/png;base64,{binary_mask_png}">
    binary_mask_png: str = Field(
        ...,
        description=(
            "Base64-encoded PNG of the binary oil-spill mask "
            "(white = oil spill, black = background). "
            "Render with: data:image/png;base64,<value>"
        ),
    )
    prob_map_png: str = Field(
        ...,
        description=(
            "Base64-encoded PNG of the sigmoid probability heatmap "
            "(jet colormap, blue = low, red = high). "
            "Render with: data:image/png;base64,<value>"
        ),
    )


class PredictResponse(BaseModel):
    """Top-level response envelope for POST /api/ml/predict."""

    success: bool = True
    filename: str = Field(..., description="Original uploaded filename.")
    prediction: PredictionResult


class ErrorDetail(BaseModel):
    """Structured error detail returned in non-2xx responses."""

    code: str   = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")


class ErrorResponse(BaseModel):
    """Top-level error envelope."""

    success: bool = False
    error: ErrorDetail
