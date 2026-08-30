"""
Pydantic data models for oil-spill observations, environmental data,
trajectory points, and backtracking analysis results.
"""

from datetime import datetime
from typing import List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, field_validator


def validate_latitude(v: float) -> float:
    """Validates that latitude is within [-90, 90]."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("Latitude must be a numeric value.")
    if not (-90.0 <= float(v) <= 90.0):
        raise ValueError(f"Latitude must be within [-90, 90], got {v}")
    return float(v)


def validate_longitude(v: float) -> float:
    """Validates that longitude is within [-180, 180]."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("Longitude must be a numeric value.")
    if not (-180.0 <= float(v) <= 180.0):
        raise ValueError(f"Longitude must be within [-180, 180], got {v}")
    return float(v)


class SpillObservation(BaseModel):
    """
    Model representing an oil spill observation detected from SAR/satellite data.
    """
    model_config = ConfigDict(extra="forbid")

    spill_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    area_km2: Optional[float] = None
    confidence: Optional[float] = None

    @field_validator("latitude")
    @classmethod
    def check_latitude(cls, v: float) -> float:
        return validate_latitude(v)

    @field_validator("longitude")
    @classmethod
    def check_longitude(cls, v: float) -> float:
        return validate_longitude(v)


class EnvironmentalObservation(BaseModel):
    """
    Model representing environmental conditions (wind and current vectors)
    at a specific point and time.
    """
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    latitude: float
    longitude: float
    wind_speed_mps: float
    wind_direction_deg: float
    current_speed_mps: float
    current_direction_deg: float

    @field_validator("latitude")
    @classmethod
    def check_latitude(cls, v: float) -> float:
        return validate_latitude(v)

    @field_validator("longitude")
    @classmethod
    def check_longitude(cls, v: float) -> float:
        return validate_longitude(v)


class TrajectoryPoint(BaseModel):
    """
    Model representing a single state/point along a surface drift trajectory.
    """
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    latitude: float
    longitude: float
    wind_speed_mps: float
    wind_direction_deg: float
    current_speed_mps: float
    current_direction_deg: float

    @field_validator("latitude")
    @classmethod
    def check_latitude(cls, v: float) -> float:
        return validate_latitude(v)

    @field_validator("longitude")
    @classmethod
    def check_longitude(cls, v: float) -> float:
        return validate_longitude(v)


class BacktrackingResult(BaseModel):
    """
    Model representing the result of backward drift simulation and probable origin estimation.
    """
    model_config = ConfigDict(extra="forbid")

    spill_id: str
    detected_location: Tuple[float, float]  # (latitude, longitude)
    detected_time: datetime
    estimated_origin: Tuple[float, float]  # (latitude, longitude)
    estimated_release_time: datetime
    trajectory: List[TrajectoryPoint]
    uncertainty_radius_km: float
    confidence: float
    data_source: str
    model_description: str

    @field_validator("detected_location", "estimated_origin")
    @classmethod
    def check_location_tuple(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError("Coordinates must be a (latitude, longitude) tuple of length 2.")
        lat, lon = v
        validate_latitude(lat)
        validate_longitude(lon)
        return (float(lat), float(lon))
