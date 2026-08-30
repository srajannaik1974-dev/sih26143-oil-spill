from datetime import datetime, timezone
from backend.services.spill_service import BaseSpillService
from backend.schemas.spill import SpillDetectionRequest, SpillDetectionResponse, GeoCoordinate


class MockSpillService(BaseSpillService):
    """
    Mock implementation of satellite oil spill detection service.
    Generates deterministic bounding polygons and mock confidence score.
    """

    async def detect_spill(self, request: SpillDetectionRequest) -> SpillDetectionResponse:
        lat = request.latitude
        lon = request.longitude
        
        # Create a mock spill polygon around the request coordinates (~ 0.02 deg offset)
        mock_polygon = [
            GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),
            GeoCoordinate(latitude=round(lat + 0.015, 6), longitude=round(lon + 0.01, 6)),
            GeoCoordinate(latitude=round(lat - 0.005, 6), longitude=round(lon + 0.02, 6)),
            GeoCoordinate(latitude=round(lat - 0.015, 6), longitude=round(lon - 0.005, 6)),
            GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),  # Close polygon
        ]
        
        return SpillDetectionResponse(
            image_id=request.image_id,
            spill_detected=True,
            confidence=0.91,
            spill_polygon=mock_polygon,
            estimated_area_sq_km=4.85,
            timestamp=datetime.now(timezone.utc)
        )

    async def detect_spill_file(self, file_path: str, original_filename: str) -> SpillDetectionResponse:
        lat = 19.4167
        lon = 71.3333
        mock_polygon = [
            GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),
            GeoCoordinate(latitude=round(lat + 0.015, 6), longitude=round(lon + 0.01, 6)),
            GeoCoordinate(latitude=round(lat - 0.005, 6), longitude=round(lon + 0.02, 6)),
            GeoCoordinate(latitude=round(lat - 0.015, 6), longitude=round(lon - 0.005, 6)),
            GeoCoordinate(latitude=round(lat + 0.01, 6), longitude=round(lon - 0.01, 6)),
        ]
        return SpillDetectionResponse(
            image_id=original_filename,
            spill_detected=True,
            confidence=0.88,
            spill_polygon=mock_polygon,
            estimated_area_sq_km=3.50,
            timestamp=datetime.now(timezone.utc),
            disclaimer="MOCK FILE DETECTION: Detection evaluated from uploaded file stub."
        )

