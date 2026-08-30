"""
Data models and schemas for the Vessel Attribution module.
Designed to interface cleanly with Member 3's AIS stream and Member 6's backend API.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class SpillOriginInput(BaseModel):
    """Input parameters representing estimated oil spill origin and release time."""
    latitude: float = Field(..., description="Latitude of spill origin in degrees (-90 to 90)", ge=-90.0, le=90.0)
    longitude: float = Field(..., description="Longitude of spill origin in degrees (-180 to 180)", ge=-180.0, le=180.0)
    estimated_release_time: datetime = Field(..., description="Estimated time of oil release (UTC)")
    max_search_radius_km: float = Field(default=50.0, description="Spatial search radius for candidate filtering in kilometers", gt=0.0)
    max_time_window_hours: float = Field(default=24.0, description="Temporal search window around release time in hours", gt=0.0)


class AISPosition(BaseModel):
    """Single AIS position report for a vessel."""
    timestamp: datetime = Field(..., description="Timestamp of the AIS report (UTC)")
    latitude: float = Field(..., description="Vessel latitude in degrees (-90 to 90)", ge=-90.0, le=90.0)
    longitude: float = Field(..., description="Vessel longitude in degrees (-180 to 180)", ge=-180.0, le=180.0)
    speed_knots: float = Field(default=0.0, description="Speed Over Ground (SOG) in knots", ge=0.0)
    heading_deg: Optional[float] = Field(default=None, description="True heading in degrees (0 to 360)", ge=0.0, le=360.0)
    course_over_ground: Optional[float] = Field(default=None, description="Course Over Ground (COG) in degrees (0 to 360)", ge=0.0, le=360.0)


class AISTrajectoryRecord(BaseModel):
    """AIS trajectory record for a vessel containing identity and time-series positions."""
    vessel_id: str = Field(..., description="Unique vessel identifier or UUID")
    mmsi: str = Field(..., description="Maritime Mobile Service Identity (MMSI) number")
    vessel_name: str = Field(default="Unknown Vessel", description="Vessel name")
    vessel_type: str = Field(default="Cargo/Tanker", description="Vessel category (e.g. Tanker, Cargo, Tug)")
    positions: List[AISPosition] = Field(..., description="Time-series AIS position reports ordered chronologically")


class FeatureScores(BaseModel):
    """Normalized feature scores (0.0 to 100.0) for explainable correlation scoring."""
    distance_score: float = Field(..., description="Distance score (30% weight): proximity of vessel to spill origin", ge=0.0, le=100.0)
    time_score: float = Field(..., description="Time score (25% weight): temporal proximity of vessel to release time", ge=0.0, le=100.0)
    trajectory_score: float = Field(..., description="Trajectory score (25% weight): path geometry and persistence near origin", ge=0.0, le=100.0)
    speed_heading_score: float = Field(..., description="Speed/Heading score (20% weight): speed changes and directional alignment", ge=0.0, le=100.0)


class CandidateVesselResult(BaseModel):
    """Attribution analysis output for a candidate vessel."""
    vessel_id: str = Field(..., description="Vessel identifier")
    mmsi: str = Field(..., description="MMSI number")
    vessel_name: str = Field(..., description="Vessel name")
    vessel_type: str = Field(..., description="Vessel type")
    distance_km: float = Field(..., description="Closest approach distance to spill origin in kilometers")
    time_difference_minutes: float = Field(..., description="Time difference in minutes between closest approach and spill release time")
    feature_scores: FeatureScores = Field(..., description="Breakdown of individual feature correlation scores")
    final_score: float = Field(..., description="Final weighted correlation score (0.0 to 100.0)", ge=0.0, le=100.0)
    rank: int = Field(..., description="Ranking position among candidate vessels (1-indexed)", ge=1)
    classification: str = Field(..., description="Standardized classification label ('Highest-correlated candidate vessel' or 'Potentially responsible vessel')")
    explanation: str = Field(..., description="Explainable textual summary of attribution correlation features")


class AttributionResponse(BaseModel):
    """Complete vessel attribution analysis response."""
    spill_origin: SpillOriginInput = Field(..., description="Spill origin input specifications")
    total_vessels_evaluated: int = Field(..., description="Total count of candidate vessels evaluated")
    candidate_vessels: List[CandidateVesselResult] = Field(..., description="List of evaluated vessels sorted by correlation rank (descending)")
