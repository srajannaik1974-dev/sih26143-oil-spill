from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.schemas.spill import GeoCoordinate
from backend.schemas.ais import VesselAISData


class VesselRankRequest(BaseModel):
    spill_source: GeoCoordinate = Field(..., description="Estimated spill origin point")
    timestamp: datetime = Field(..., description="Estimated spill origin time")
    candidate_vessels: List[VesselAISData] = Field(..., description="Candidate vessels retrieved from AIS query")
    estimated_release_time: Optional[datetime] = Field(None, description="Optional estimated release time (defaults to timestamp if omitted)")
    search_radius_km: Optional[float] = Field(50.0, gt=0.0, description="Spatial search radius for candidate filtering in km")
    time_window_hours: Optional[float] = Field(24.0, gt=0.0, description="Temporal search window in hours")


class AttributionFactor(BaseModel):
    factor_name: str = Field(..., description="Name of attribution heuristic/factor (e.g. Distance Proximity, Velocity Anomaly)")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score contribution between 0.0 and 1.0")
    description: str = Field(..., description="Human readable explanation of factor score")


class RankedVessel(BaseModel):
    vessel: VesselAISData = Field(..., description="Vessel AIS details")
    rank: int = Field(..., ge=1, description="Attribution rank (1 = highest suspect)")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized suspect risk score between 0.0 and 1.0")
    attribution_factors: List[AttributionFactor] = Field(default_factory=list, description="Breakdown of factors contributing to risk score")
    classification: Optional[str] = Field(None, description="Standardized classification label from Member 4 attribution engine")
    explanation: Optional[str] = Field(None, description="Explainable textual summary of attribution features")
    final_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Raw weighted attribution score (0.0 to 100.0)")


class VesselRankResponse(BaseModel):
    ranked_vessels: List[RankedVessel] = Field(default_factory=list, description="List of vessels ordered by suspect risk rank")
    total_ranked: int = Field(..., description="Total candidate vessels ranked")
    ranked_at: datetime = Field(..., description="ISO-8601 timestamp of ranking computation")
    disclaimer: str = Field(
        "ATTRIBUTION RESULT: Vessel rankings and correlation scores calculated using Member 4 Vessel Attribution Engine.",
        description="Notice regarding attribution analysis data"
    )
