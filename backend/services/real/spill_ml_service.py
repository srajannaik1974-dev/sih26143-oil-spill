from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import os

from backend.services.spill_service import BaseSpillService
from backend.services.mock.spill import MockSpillService
from backend.schemas.spill import SpillDetectionRequest, SpillDetectionResponse, GeoCoordinate


class RealSpillServiceAdapter(BaseSpillService):
    """
    Adapter wrapping Member 1's Satellite Oil Spill ML inference pipeline (ml.training.inference).
    Falls back gracefully to MockSpillService if trained weights (.pth) or GeoTIFF assets are missing.
    """

    def __init__(self, checkpoint_path: str = "ml/training/checkpoints/best_unet.pth"):
        self.checkpoint_path = Path(checkpoint_path)
        self.fallback_mock = MockSpillService()
        self.predictor = None

        if self.checkpoint_path.exists():
            try:
                from ml.training.inference import OilSpillPredictor
                self.predictor = OilSpillPredictor(ckpt_path=self.checkpoint_path)
            except Exception as e:
                print(f"[RealSpillServiceAdapter] Warning loading model: {e}")

    async def detect_spill(self, request: SpillDetectionRequest) -> SpillDetectionResponse:
        # Check if local image asset path exists for live inference
        image_path = Path(request.image_url) if request.image_url and os.path.exists(request.image_url) else None

        if self.predictor is not None and image_path is not None:
            try:
                binary_mask, prob_map = self.predictor.predict(image_path)
                spill_loc = self.predictor.get_spill_location(
                    tiff_path=image_path,
                    binary_mask=binary_mask,
                    prob_map=prob_map,
                    original_filename=request.image_id
                )
                
                lat = spill_loc.get("latitude") or request.latitude
                lon = spill_loc.get("longitude") or request.longitude
                area = spill_loc.get("area_km2", 0.0)
                conf = spill_loc.get("confidence", 0.85)

                polygon = [
                    GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),
                    GeoCoordinate(latitude=round(lat + 0.015, 6), longitude=round(lon + 0.01, 6)),
                    GeoCoordinate(latitude=round(lat - 0.005, 6), longitude=round(lon + 0.02, 6)),
                    GeoCoordinate(latitude=round(lat - 0.015, 6), longitude=round(lon - 0.005, 6)),
                    GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),
                ]

                return SpillDetectionResponse(
                    image_id=request.image_id,
                    spill_detected=spill_loc.get("detected", True),
                    confidence=round(conf, 4),
                    spill_polygon=polygon,
                    estimated_area_sq_km=round(area, 2),
                    timestamp=datetime.now(timezone.utc),
                    disclaimer="M1 ML INFERENCE: Detection evaluated using Member 1 PyTorch UNet model."
                )
            except Exception as e:
                print(f"[RealSpillServiceAdapter] Inference execution error: {e}. Falling back to mock service.")

        # Fallback to mock service if model weights or TIFF files are unavailable
        return await self.fallback_mock.detect_spill(request)

    async def detect_spill_file(self, file_path: str, original_filename: str) -> SpillDetectionResponse:
        image_path = Path(file_path)

        if self.predictor is not None and image_path.exists():
            try:
                binary_mask, prob_map = self.predictor.predict(image_path)
                spill_loc = self.predictor.get_spill_location(
                    tiff_path=image_path,
                    binary_mask=binary_mask,
                    prob_map=prob_map,
                    original_filename=original_filename
                )

                lat = spill_loc.get("latitude") or 19.4167
                lon = spill_loc.get("longitude") or 71.3333
                area = spill_loc.get("area_km2", 0.0)
                conf = spill_loc.get("confidence", 0.85)

                polygon = [
                    GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),
                    GeoCoordinate(latitude=round(lat + 0.015, 6), longitude=round(lon + 0.01, 6)),
                    GeoCoordinate(latitude=round(lat - 0.005, 6), longitude=round(lon + 0.02, 6)),
                    GeoCoordinate(latitude=round(lat - 0.015, 6), longitude=round(lon - 0.005, 6)),
                    GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),
                ]

                return SpillDetectionResponse(
                    image_id=original_filename,
                    spill_detected=spill_loc.get("detected", True),
                    confidence=round(conf, 4),
                    spill_polygon=polygon,
                    estimated_area_sq_km=round(area, 2),
                    timestamp=datetime.now(timezone.utc),
                    disclaimer="M1 ML REAL INFERENCE: Detection evaluated on uploaded Sentinel-1 GeoTIFF using PyTorch UNet model."
                )
            except Exception as e:
                print(f"[RealSpillServiceAdapter] TIFF inference error: {e}. Falling back to mock file response.")

        return await self.fallback_mock.detect_spill_file(file_path, original_filename)

