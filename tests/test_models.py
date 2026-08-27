"""
Unit tests for Pydantic data models (src/drift/models.py).
"""

from datetime import datetime, timezone
import json
import pytest
from src.drift.models import (
    SpillObservation,
    EnvironmentalObservation,
    TrajectoryPoint,
    BacktrackingResult,
)


def test_spill_observation_serialization():
    """Requirement 10 & 12: SpillObservation JSON serialization/deserialization."""
    now = datetime.now(timezone.utc)
    spill = SpillObservation(
        spill_id="SPILL_2026_001",
        latitude=18.92,
        longitude=72.83,
        timestamp=now,
        area_km2=4.5,
        confidence=0.92,
    )

    # Serialize to JSON string
    json_str = spill.model_dump_json()
    assert "SPILL_2026_001" in json_str
    assert "18.92" in json_str

    # Deserialize back to object
    deserialized = SpillObservation.model_validate_json(json_str)
    assert deserialized.spill_id == spill.spill_id
    assert deserialized.latitude == spill.latitude
    assert deserialized.longitude == spill.longitude
    assert deserialized.area_km2 == spill.area_km2
    assert deserialized.confidence == spill.confidence


def test_environmental_observation_serialization():
    """Requirement 11: EnvironmentalObservation serialization/deserialization."""
    now = datetime.now(timezone.utc)
    env = EnvironmentalObservation(
        timestamp=now,
        latitude=18.90,
        longitude=72.80,
        wind_speed_mps=8.5,
        wind_direction_deg=135.0,
        current_speed_mps=0.45,
        current_direction_deg=210.0,
    )

    json_str = env.model_dump_json()
    deserialized = EnvironmentalObservation.model_validate_json(json_str)

    assert deserialized.latitude == env.latitude
    assert deserialized.longitude == env.longitude
    assert deserialized.wind_speed_mps == env.wind_speed_mps
    assert deserialized.wind_direction_deg == env.wind_direction_deg
    assert deserialized.current_speed_mps == env.current_speed_mps
    assert deserialized.current_direction_deg == env.current_direction_deg


def test_trajectory_point_serialization():
    """Requirement 13: TrajectoryPoint serialization/deserialization."""
    now = datetime.now(timezone.utc)
    point = TrajectoryPoint(
        timestamp=now,
        latitude=18.91,
        longitude=72.82,
        wind_speed_mps=7.0,
        wind_direction_deg=90.0,
        current_speed_mps=0.3,
        current_direction_deg=180.0,
    )

    json_str = point.model_dump_json()
    deserialized = TrajectoryPoint.model_validate_json(json_str)

    assert deserialized.latitude == point.latitude
    assert deserialized.longitude == point.longitude
    assert deserialized.wind_speed_mps == point.wind_speed_mps


def test_backtracking_result_serialization():
    """Requirement 14: BacktrackingResult serialization/deserialization."""
    t0 = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 26, 18, 0, 0, tzinfo=timezone.utc)

    traj_point = TrajectoryPoint(
        timestamp=t0,
        latitude=18.90,
        longitude=72.80,
        wind_speed_mps=5.0,
        wind_direction_deg=45.0,
        current_speed_mps=0.2,
        current_direction_deg=90.0,
    )

    result = BacktrackingResult(
        spill_id="SPILL_2026_001",
        detected_location=(18.95, 72.85),
        detected_time=t1,
        estimated_origin=(18.90, 72.80),
        estimated_release_time=t0,
        trajectory=[traj_point],
        uncertainty_radius_km=2.5,
        confidence=0.88,
        data_source="Synthetic Environmental Demo Provider",
        model_description="Simplified Surface Drift Vector (3% Windage Rule)",
    )

    json_str = result.model_dump_json()
    deserialized = BacktrackingResult.model_validate_json(json_str)

    assert deserialized.spill_id == result.spill_id
    assert deserialized.detected_location == (18.95, 72.85)
    assert deserialized.estimated_origin == (18.90, 72.80)
    assert deserialized.uncertainty_radius_km == 2.5
    assert len(deserialized.trajectory) == 1
    assert deserialized.trajectory[0].latitude == 18.90


def test_invalid_coordinates_validation_in_models():
    """Verify that models reject out-of-bounds latitude and longitude."""
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError):
        SpillObservation(
            spill_id="FAIL_1",
            latitude=95.0,
            longitude=70.0,
            timestamp=now,
        )

    with pytest.raises(ValueError):
        SpillObservation(
            spill_id="FAIL_2",
            latitude=15.0,
            longitude=-185.0,
            timestamp=now,
        )

    with pytest.raises(ValueError):
        BacktrackingResult(
            spill_id="FAIL_3",
            detected_location=(95.0, 70.0),
            detected_time=now,
            estimated_origin=(15.0, 70.0),
            estimated_release_time=now,
            trajectory=[],
            uncertainty_radius_km=1.0,
            confidence=0.8,
            data_source="Test",
            model_description="Test",
        )
