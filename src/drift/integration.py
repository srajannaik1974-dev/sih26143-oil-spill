"""
Member 2 Integration Boundary Module.

Defines the clean input contract (DetectedSpillInput) received from Member 1 (SAR/Satellite Detection),
and the clean output contract (DriftOriginOutput) passed to Member 3 (AIS Vessel Attribution).

Data Ownership & Downstream Handoff Workflow:
1. Member 1: SAR image processing -> detects spill -> provides DetectedSpillInput (spill_id, lat, lon, timestamp, metadata).
2. Member 2: Receives DetectedSpillInput -> drift vector physics & backward simulation -> produces DriftOriginOutput.
3. Member 3: Receives DriftOriginOutput (probable_latitude, probable_longitude, estimated_release_time) -> correlates historical AIS tracks -> candidate vessels.
4. Member 4: Receives candidate vessels from Member 3 + origin context from Member 2 -> attribution scoring API.

Serialization Contract:
- All datetime fields serialize to standard ISO 8601 strings (e.g., '2026-08-27T12:00:00+00:00').
- Outputs are fully JSON-serializable using .to_dict() and .to_json().
- Status string values ('origin_estimated', 'insufficient_trajectory', 'environmental_data_unavailable')
  are strictly preserved for downstream conditional handling.

Environmental Data Source Note:
The pipeline uses SyntheticEnvironmentalProvider or FileEnvironmentalProvider. Real-world evaluation
requires live oceanographic/meteorological feeds.
"""

from datetime import datetime, timezone
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, field_validator

from .models import validate_latitude, validate_longitude, TrajectoryPoint
from .environment import EnvironmentalDataProvider
from .pipeline import DriftOriginPipeline


class DetectedSpillInput(BaseModel):
    """
    Upstream input model representing a detected oil spill event from Member 1.
    """
    model_config = ConfigDict(extra="forbid")

    spill_id: str
    latitude: float
    longitude: float
    detection_timestamp: datetime
    area_km2: Optional[float] = None
    confidence: Optional[float] = None

    @field_validator("spill_id")
    @classmethod
    def check_spill_id(cls, v: str) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("spill_id must be a non-empty string.")
        return v.strip()

    @field_validator("latitude")
    @classmethod
    def check_latitude(cls, v: float) -> float:
        return validate_latitude(v)

    @field_validator("longitude")
    @classmethod
    def check_longitude(cls, v: float) -> float:
        return validate_longitude(v)

    @field_validator("detection_timestamp")
    @classmethod
    def check_timestamp(cls, v: datetime) -> datetime:
        if v is None or v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("detection_timestamp must be timezone-aware (UTC preferred).")
        return v

    @field_validator("area_km2")
    @classmethod
    def check_area(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("area_km2 must be numeric.")
            if float(v) < 0:
                raise ValueError(f"area_km2 cannot be negative, got {v}")
            return float(v)
        return None

    @field_validator("confidence")
    @classmethod
    def check_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("confidence must be numeric.")
            val = float(v)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
            return val
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Converts input object to a dictionary with ISO 8601 timestamp string."""
        data = self.model_dump()
        data["detection_timestamp"] = self.detection_timestamp.isoformat()
        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes input object to a JSON-formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectedSpillInput":
        """Deserializes a dictionary into a DetectedSpillInput instance."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "DetectedSpillInput":
        """Deserializes a JSON string into a DetectedSpillInput instance."""
        return cls.model_validate_json(json_str)


class DriftOriginOutput(BaseModel):
    """
    Downstream output contract produced by Member 2 for Member 3 (AIS Vessel Attribution).
    """
    model_config = ConfigDict(extra="forbid")

    spill_id: str
    detected_latitude: float
    detected_longitude: float
    detection_timestamp: datetime
    probable_latitude: float
    probable_longitude: float
    estimated_release_time: datetime
    trajectory_points_used: int
    status: str
    message: str
    backward_trajectory: List[TrajectoryPoint]

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the output object to a clean Python dictionary with ISO 8601 formatted datetime strings.
        """
        data = self.model_dump()
        data["detection_timestamp"] = self.detection_timestamp.isoformat()
        data["estimated_release_time"] = self.estimated_release_time.isoformat()
        data["backward_trajectory"] = [
            {
                **pt,
                "timestamp": pt["timestamp"].isoformat() if isinstance(pt["timestamp"], datetime) else pt["timestamp"]
            }
            for pt in data["backward_trajectory"]
        ]
        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serializes the output object to a JSON-formatted string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftOriginOutput":
        """
        Deserializes a dictionary into a DriftOriginOutput instance.
        """
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "DriftOriginOutput":
        """
        Deserializes a JSON-formatted string into a DriftOriginOutput instance.
        """
        return cls.model_validate_json(json_str)


def process_detected_spill(
    detected_spill: DetectedSpillInput,
    duration_hours: float = 2.0,
    step_minutes: float = 30.0,
    windage_factor: float = 0.03,
    max_gap_minutes: Optional[float] = 60.0,
    env_provider: Optional[EnvironmentalDataProvider] = None,
) -> DriftOriginOutput:
    """
    Processes an upstream DetectedSpillInput event through the Member 2 drift/origin pipeline
    and returns a structured DriftOriginOutput for Member 3 vessel attribution.

    :param detected_spill: DetectedSpillInput instance containing spill detection details
    :param duration_hours: Backward tracking duration in hours (default: 2.0)
    :param step_minutes: Simulation step size in minutes (default: 30.0)
    :param windage_factor: Windage coefficient (default: 0.03)
    :param max_gap_minutes: Max environmental lookup gap in minutes (default: 60.0)
    :param env_provider: Optional EnvironmentalDataProvider override
    :return: DriftOriginOutput instance
    :raises ValueError: If detected_spill is invalid or parameter validation fails
    """
    if not isinstance(detected_spill, DetectedSpillInput):
        raise ValueError("detected_spill must be an instance of DetectedSpillInput.")

    pipeline = DriftOriginPipeline(env_provider=env_provider)
    pipeline_result = pipeline.analyze_spill(
        latitude=detected_spill.latitude,
        longitude=detected_spill.longitude,
        detection_timestamp=detected_spill.detection_timestamp,
        spill_id=detected_spill.spill_id,
        duration_hours=duration_hours,
        step_minutes=step_minutes,
        windage_factor=windage_factor,
        max_gap_minutes=max_gap_minutes,
    )

    return DriftOriginOutput(
        spill_id=pipeline_result.spill_id,
        detected_latitude=pipeline_result.detected_latitude,
        detected_longitude=pipeline_result.detected_longitude,
        detection_timestamp=pipeline_result.detection_timestamp,
        probable_latitude=pipeline_result.probable_latitude,
        probable_longitude=pipeline_result.probable_longitude,
        estimated_release_time=pipeline_result.estimated_release_time,
        trajectory_points_used=pipeline_result.trajectory_points_used,
        status=pipeline_result.status,
        message=pipeline_result.message,
        backward_trajectory=pipeline_result.backward_trajectory,
    )
