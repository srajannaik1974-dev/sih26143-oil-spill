"""
Unit and integration tests for Member 2 Integration Boundary and Output Serialization (src/drift/integration.py).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import pytest

from src.drift.integration import DetectedSpillInput, DriftOriginOutput, process_detected_spill
from src.drift.environment import SyntheticEnvironmentalProvider, FileEnvironmentalProvider, EnvironmentalDataProvider


@pytest.fixture
def valid_spill_input():
    return DetectedSpillInput(
        spill_id="SPILL-2026-MEMBER1-001",
        latitude=12.3456,
        longitude=74.5678,
        detection_timestamp=datetime(2026, 8, 27, 11, 20, 0, tzinfo=timezone.utc),
        area_km2=2.35,
        confidence=0.91,
    )


@pytest.fixture
def demo_csv_path():
    return Path(__file__).parents[1] / "data" / "environment_demo.csv"


class MockFailingEnvProvider(EnvironmentalDataProvider):
    """Provider that returns empty list to simulate unavailable environmental data."""
    def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
        return []


def test_valid_detected_spill_input_accepted(valid_spill_input):
    """Requirement 1: Valid detected spill input is accepted."""
    assert valid_spill_input.spill_id == "SPILL-2026-MEMBER1-001"
    assert valid_spill_input.latitude == 12.3456
    assert valid_spill_input.longitude == 74.5678
    assert valid_spill_input.area_km2 == 2.35
    assert valid_spill_input.confidence == 0.91


def test_invalid_latitude_rejected():
    """Requirement 2: Invalid latitude is rejected."""
    with pytest.raises(ValueError):
        DetectedSpillInput(
            spill_id="FAIL_LAT",
            latitude=95.0,
            longitude=74.0,
            detection_timestamp=datetime.now(timezone.utc),
        )


def test_invalid_longitude_rejected():
    """Requirement 3: Invalid longitude is rejected."""
    with pytest.raises(ValueError):
        DetectedSpillInput(
            spill_id="FAIL_LON",
            latitude=12.0,
            longitude=-185.0,
            detection_timestamp=datetime.now(timezone.utc),
        )


def test_naive_timestamp_rejected():
    """Requirement 4: Naive timestamp is rejected."""
    with pytest.raises(ValueError):
        DetectedSpillInput(
            spill_id="FAIL_TIME",
            latitude=12.0,
            longitude=74.0,
            detection_timestamp=datetime(2026, 8, 27, 11, 20, 0),  # naive
        )


def test_empty_spill_id_rejected():
    """Requirement 5: Empty spill_id is rejected."""
    with pytest.raises(ValueError):
        DetectedSpillInput(
            spill_id="   ",
            latitude=12.0,
            longitude=74.0,
            detection_timestamp=datetime.now(timezone.utc),
        )


def test_optional_area_km2_validation():
    """Requirement 6: Optional area_km2 validation works."""
    with pytest.raises(ValueError):
        DetectedSpillInput(
            spill_id="FAIL_AREA",
            latitude=12.0,
            longitude=74.0,
            detection_timestamp=datetime.now(timezone.utc),
            area_km2=-1.5,
        )


def test_optional_confidence_validation():
    """Requirement 7: Optional confidence validation works."""
    with pytest.raises(ValueError):
        DetectedSpillInput(
            spill_id="FAIL_CONF_HIGH",
            latitude=12.0,
            longitude=74.0,
            detection_timestamp=datetime.now(timezone.utc),
            confidence=1.5,
        )
    with pytest.raises(ValueError):
        DetectedSpillInput(
            spill_id="FAIL_CONF_LOW",
            latitude=12.0,
            longitude=74.0,
            detection_timestamp=datetime.now(timezone.utc),
            confidence=-0.1,
        )


def test_process_detected_spill_end_to_end(valid_spill_input):
    """Requirements 8-16: Valid input reaches pipeline and produces complete DriftOriginOutput."""
    output = process_detected_spill(valid_spill_input, duration_hours=2.0, step_minutes=30.0)

    assert isinstance(output, DriftOriginOutput)
    assert output.spill_id == "SPILL-2026-MEMBER1-001"
    assert output.detected_latitude == 12.3456
    assert output.detected_longitude == 74.5678
    assert output.detection_timestamp == valid_spill_input.detection_timestamp
    assert isinstance(output.probable_latitude, float)
    assert isinstance(output.probable_longitude, float)
    assert output.estimated_release_time < valid_spill_input.detection_timestamp
    assert output.trajectory_points_used == 5
    assert output.status == "origin_estimated"
    assert len(output.backward_trajectory) == 5


# --- PHASE 10 SERIALIZATION & HANDOFF TESTS ---

def test_output_serialization_to_dict(valid_spill_input):
    """Requirement 1, 2, 3, 4, 5, 6: Output to_dict() serializes ISO timestamps and clean dict structure."""
    output = process_detected_spill(valid_spill_input, duration_hours=2.0, step_minutes=30.0)
    data = output.to_dict()

    assert isinstance(data, dict)
    assert data["spill_id"] == "SPILL-2026-MEMBER1-001"
    assert data["detected_latitude"] == 12.3456
    assert data["detected_longitude"] == 74.5678
    # Datetimes must be serialized to ISO 8601 strings
    assert isinstance(data["detection_timestamp"], str)
    assert isinstance(data["estimated_release_time"], str)
    assert "T" in data["detection_timestamp"]
    assert "T" in data["estimated_release_time"]
    assert data["status"] == "origin_estimated"
    assert data["trajectory_points_used"] == 5
    assert isinstance(data["backward_trajectory"], list)
    assert len(data["backward_trajectory"]) == 5
    assert isinstance(data["backward_trajectory"][0]["timestamp"], str)


def test_output_serialization_to_json_and_roundtrip(valid_spill_input):
    """Requirement 1 & 2: Output to_json() serializes to valid JSON string and roundtrips via from_json()."""
    output = process_detected_spill(valid_spill_input, duration_hours=2.0, step_minutes=30.0)
    json_str = output.to_json(indent=2)

    assert isinstance(json_str, str)
    parsed_json = json.loads(json_str)
    assert parsed_json["spill_id"] == "SPILL-2026-MEMBER1-001"
    assert parsed_json["status"] == "origin_estimated"

    # Roundtrip via from_json()
    reconstructed = DriftOriginOutput.from_json(json_str)
    assert reconstructed.spill_id == output.spill_id
    assert reconstructed.detected_latitude == output.detected_latitude
    assert reconstructed.probable_latitude == output.probable_latitude
    assert reconstructed.estimated_release_time == output.estimated_release_time
    assert reconstructed.estimated_release_time.tzinfo is not None


def test_insufficient_trajectory_status_preservation():
    """Requirement 8: insufficient_trajectory status is preserved in output."""
    spill = DetectedSpillInput(
        spill_id="SPILL-INSUFFICIENT",
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    # 0 hours duration results in 1 point -> status 'insufficient_trajectory'
    output = process_detected_spill(spill, duration_hours=0.0)
    assert output.status == "insufficient_trajectory"
    assert output.trajectory_points_used == 1

    d = output.to_dict()
    assert d["status"] == "insufficient_trajectory"


def test_environmental_data_unavailable_status_preservation():
    """Requirement 9: environmental_data_unavailable status is preserved in output."""
    spill = DetectedSpillInput(
        spill_id="SPILL-UNAVAILABLE",
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    failing_prov = MockFailingEnvProvider()
    output = process_detected_spill(spill, env_provider=failing_prov)

    assert output.status == "environmental_data_unavailable"
    assert output.trajectory_points_used == 0

    d = output.to_dict()
    assert d["status"] == "environmental_data_unavailable"


def test_file_provider_works_with_integration_contract(demo_csv_path):
    """Requirement 12: Existing FileEnvironmentalProvider works with process_detected_spill."""
    file_prov = FileEnvironmentalProvider(demo_csv_path)
    spill = DetectedSpillInput(
        spill_id="SPILL-FILE-DEMO",
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    output = process_detected_spill(spill, env_provider=file_prov)

    assert output.status == "origin_estimated"
    assert output.spill_id == "SPILL-FILE-DEMO"
    assert len(output.backward_trajectory) == 5
    assert output.to_json() is not None
