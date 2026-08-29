"""Data schemas and transfer objects for the AIS / Vessel Tracking module."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class AISPoint:
    """Standardized single AIS observation record."""

    vessel_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed_knots: Optional[float] = None
    heading_deg: Optional[float] = None

    @property
    def timestamp_iso(self) -> str:
        """Return ISO 8601 UTC formatted string."""
        if self.timestamp.tzinfo is None:
            dt = self.timestamp.replace(tzinfo=timezone.utc)
        else:
            dt = self.timestamp.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict[str, Any]:
        """Convert point to dictionary representation."""
        data: Dict[str, Any] = {
            "vessel_id": self.vessel_id,
            "timestamp": self.timestamp_iso,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed_knots": self.speed_knots,
            "heading_deg": self.heading_deg,
        }
        return data


@dataclass
class VesselTrajectory:
    """Chronologically sorted sequence of AIS observations for a vessel."""

    vessel_id: str
    points: List[AISPoint] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def start_time(self) -> Optional[datetime]:
        return self.points[0].timestamp if self.points else None

    @property
    def end_time(self) -> Optional[datetime]:
        return self.points[-1].timestamp if self.points else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class CandidateVessel:
    """Candidate vessel matching spatial-temporal spill query criteria."""

    vessel_id: str
    closest_distance_km: float
    closest_timestamp: str
    latitude: float
    longitude: float
    speed_knots: Optional[float] = None
    heading_deg: Optional[float] = None
    time_difference_minutes: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate vessel to the standard output format."""
        return {
            "vessel_id": self.vessel_id,
            "closest_distance_km": round(self.closest_distance_km, 3),
            "closest_timestamp": self.closest_timestamp,
            "time_difference_minutes": round(self.time_difference_minutes, 2) if self.time_difference_minutes is not None else None,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed_knots": self.speed_knots,
            "heading_deg": self.heading_deg,
        }


@dataclass
class CandidateOutput:
    """Complete candidate vessels output for downstream attribution module."""

    spill_id: str
    candidates: List[CandidateVessel] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize output to the standard candidate response format."""
        return {
            "spill_id": self.spill_id,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class SpillQuery:
    """Input query parameters supplied by caller / backtracking module."""

    spill_id: str
    origin_lat: float
    origin_lon: float
    release_start: datetime
    release_end: datetime
    search_radius_km: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        start_iso = self.release_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = self.release_end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "spill_id": self.spill_id,
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "release_start": start_iso,
            "release_end": end_iso,
            "search_radius_km": self.search_radius_km,
        }

