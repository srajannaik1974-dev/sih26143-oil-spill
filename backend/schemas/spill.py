from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class GeoCoordinate(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees (-180 to 180)")


class SpillDetectionRequest(BaseModel):
    image_id: str = Field(..., description="Unique identifier for the satellite image")
    image_url: Optional[str] = Field(None, description="Optional URL to the satellite image asset")
    timestamp: datetime = Field(..., description="ISO-8601 timestamp of image acquisition")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Center latitude of the target area")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Center longitude of the target area")


class SpillDetectionResponse(BaseModel):
    image_id: str = Field(..., description="Satellite image identifier from request")
    spill_detected: bool = Field(..., description="True if an oil spill is detected in the image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score between 0.0 and 1.0 (mock score)")
    spill_polygon: List[GeoCoordinate] = Field(default_factory=list, description="Polygon vertices bounding the oil spill area")
    estimated_area_sq_km: Optional[float] = Field(None, description="Estimated surface area of the spill in square kilometers")
    timestamp: datetime = Field(..., description="ISO-8601 timestamp of detection process")
    disclaimer: str = Field(
        "MOCK DETECTION RESULT: Confidence and polygon boundaries are simulated and must not be treated as real satellite ML inference.",
        description="Notice regarding mock data"
    )
