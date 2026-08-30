from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.schemas.spill import GeoCoordinate


class BacktrackRequest(BaseModel):
    spill_location: GeoCoordinate = Field(..., description="Geographic coordinate of detected oil spill")
    timestamp: datetime = Field(..., description="ISO-8601 timestamp when the spill was detected")
    drift_hours: float = Field(24.0, gt=0, le=168.0, description="Duration in hours to backtrack drift (default: 24h)")
    wind_vector_deg: Optional[float] = Field(None, ge=0.0, lt=360.0, description="Optional wind direction in degrees")
    current_vector_deg: Optional[float] = Field(None, ge=0.0, lt=360.0, description="Optional ocean current direction in degrees")


class TrajectoryPoint(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of position along backtrack trajectory")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    uncertainty_radius_km: float = Field(..., description="Uncertainty radius in km at this point in time")


class SourceArea(BaseModel):
    center: GeoCoordinate = Field(..., description="Estimated central coordinate of the discharge/spill source")
    radius_km: float = Field(..., description="Radius of estimated origin zone in km")
    boundary_polygon: List[GeoCoordinate] = Field(default_factory=list, description="Polygon enclosing the estimated origin zone")


class BacktrackResponse(BaseModel):
    spill_location: GeoCoordinate = Field(..., description="Original detection location")
    detection_timestamp: datetime = Field(..., description="Original detection timestamp")
    estimated_source_area: SourceArea = Field(..., description="Estimated spill origin region")
    trajectory: List[TrajectoryPoint] = Field(..., description="Step-by-step backtrack trajectory points from current to origin")
    timestamp: datetime = Field(..., description="ISO-8601 timestamp of analysis execution")
    disclaimer: str = Field(
        "MOCK DRIFT RESULT: Backtrack trajectory and source area are generated using simple geometric approximation and must not be treated as hydro-dynamic model output.",
        description="Notice regarding mock data"
    )
