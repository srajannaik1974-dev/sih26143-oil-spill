import os
import shutil
import tempfile
import rasterio

from fastapi import APIRouter, Depends, status, HTTPException, File, UploadFile
from backend.schemas.spill import SpillDetectionRequest, SpillDetectionResponse
from backend.schemas.drift import BacktrackRequest, BacktrackResponse
from backend.services.spill_service import BaseSpillService
from backend.services.drift_service import BaseDriftService
from backend.dependencies import get_spill_service, get_drift_service

router = APIRouter(prefix="/spill", tags=["Spill Detection & Backtracking"])


@router.post(
    "/detect",
    response_model=SpillDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Oil Spill from Satellite Imagery (JSON Metadata)",
    description="Accepts satellite imagery metadata and returns oil spill detection results with confidence and bounding polygon."
)
async def detect_spill(
    request: SpillDetectionRequest,
    spill_service: BaseSpillService = Depends(get_spill_service)
) -> SpillDetectionResponse:
    try:
        return await spill_service.detect_spill(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing spill detection: {str(e)}"
        )


@router.post(
    "/detect/upload",
    response_model=SpillDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Oil Spill from Uploaded Sentinel-1 GeoTIFF (.tif/.tiff)",
    description="Accepts a raw 1-channel Sentinel-1 SAR GeoTIFF image file via multipart/form-data upload and runs U-Net segmentation inference."
)
async def detect_spill_upload(
    file: UploadFile = File(...),
    spill_service: BaseSpillService = Depends(get_spill_service)
) -> SpillDetectionResponse:
    filename = file.filename or "uploaded_sar.tif"
    
    # 1. Extension Validation
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Only Sentinel-1 GeoTIFF (.tif, .tiff) files are accepted."
        )

    tmp_path = None
    try:
        # 2. Write upload stream to temporary file safely
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 3. Validate GeoTIFF via Rasterio
        try:
            with rasterio.open(tmp_path) as src:
                if src.count < 1:
                    raise ValueError("GeoTIFF dataset contains 0 bands.")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupt or invalid GeoTIFF file. Rasterio failed to read dataset: {str(exc)}"
            )

        # 4. Run inference via Spill Service
        return await spill_service.detect_spill_file(tmp_path, filename)

    finally:
        # 5. Clean up temporary file safely
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@router.post(
    "/backtrack",
    response_model=BacktrackResponse,
    status_code=status.HTTP_200_OK,
    summary="Backtrack Ocean Drift Trajectory",
    description="Accepts detected spill position and time, and computes backtrack trajectory and estimated origin area."
)
async def backtrack_spill(
    request: BacktrackRequest,
    drift_service: BaseDriftService = Depends(get_drift_service)
) -> BacktrackResponse:
    try:
        return await drift_service.backtrack(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating drift backtrack: {str(e)}"
        )
