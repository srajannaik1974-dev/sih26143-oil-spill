"""
Unit tests for Environmental Data Layer (src/drift/environment.py).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import pytest

from src.drift.environment import (
    SyntheticEnvironmentalProvider,
    FileEnvironmentalProvider,
    RealEnvironmentalProvider,
    get_nearest_observation,
    EnvironmentalDataUnavailableError,
)
from src.drift.models import EnvironmentalObservation
from src.drift.simulator import ForwardSimulator
from src.drift.backtracker import BackwardSimulator
from src.drift.origin import estimate_probable_origin
from src.drift.pipeline import DriftOriginPipeline


@pytest.fixture
def provider():
    return SyntheticEnvironmentalProvider()


@pytest.fixture
def sample_times():
    start = datetime(2026, 8, 26, 6, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 26, 10, 0, 0, tzinfo=timezone.utc)
    return start, end


@pytest.fixture
def demo_csv_path():
    return Path(__file__).parents[1] / "data" / "environment_demo.csv"


def test_provider_returns_environmental_observations(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    assert len(observations) > 0
    for obs in observations:
        assert isinstance(obs, EnvironmentalObservation)


def test_deterministic_output(provider, sample_times):
    start, end = sample_times
    obs_run1 = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)
    obs_run2 = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    assert len(obs_run1) == len(obs_run2)
    for o1, o2 in zip(obs_run1, obs_run2):
        assert o1.timestamp == o2.timestamp
        assert o1.wind_speed_mps == o2.wind_speed_mps
        assert o1.wind_direction_deg == o2.wind_direction_deg
        assert o1.current_speed_mps == o2.current_speed_mps
        assert o1.current_direction_deg == o2.current_direction_deg


def test_chronological_ordering(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    for i in range(len(observations) - 1):
        assert observations[i].timestamp < observations[i + 1].timestamp


def test_timezone_aware_timestamps(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    for obs in observations:
        assert obs.timestamp.tzinfo is not None
        assert obs.timestamp.tzinfo.utcoffset(obs.timestamp) is not None


def test_valid_coordinates_in_generated_observations(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    for obs in observations:
        assert -90.0 <= obs.latitude <= 90.0
        assert -180.0 <= obs.longitude <= 180.0


def test_non_negative_wind_and_current_speeds(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    for obs in observations:
        assert obs.wind_speed_mps >= 0.0
        assert obs.current_speed_mps >= 0.0


def test_valid_directions(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    for obs in observations:
        assert 0.0 <= obs.wind_direction_deg < 360.0
        assert 0.0 <= obs.current_direction_deg < 360.0


def test_correct_observation_count(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)
    assert len(observations) == 9


def test_source_name_identifier(provider):
    assert provider.SOURCE_NAME == "synthetic_demo"


def test_invalid_time_range_rejection(provider, sample_times):
    start, end = sample_times
    with pytest.raises(ValueError):
        provider.get_observations(end, start, 18.92, 72.83, interval_minutes=30.0)


def test_invalid_coordinates_rejection(provider, sample_times):
    start, end = sample_times
    with pytest.raises(ValueError):
        provider.get_observations(start, end, 95.0, 72.83)
    with pytest.raises(ValueError):
        provider.get_observations(start, end, 18.92, 200.0)


def test_timezone_naive_input_rejection(provider):
    naive_start = datetime(2026, 8, 26, 6, 0, 0)
    aware_end = datetime(2026, 8, 26, 10, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        provider.get_observations(naive_start, aware_end, 18.92, 72.83)

    with pytest.raises(ValueError):
        provider.get_observations(aware_end, naive_start, 18.92, 72.83)


def test_nearest_timestamp_selection(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    target_time = datetime(2026, 8, 26, 6, 20, 0, tzinfo=timezone.utc)
    nearest = get_nearest_observation(observations, target_time)

    expected_time = datetime(2026, 8, 26, 6, 30, 0, tzinfo=timezone.utc)
    assert nearest.timestamp == expected_time


def test_maximum_time_gap_behavior(provider, sample_times):
    start, end = sample_times
    observations = provider.get_observations(start, end, 18.92, 72.83, interval_minutes=30.0)

    target_time = datetime(2026, 8, 26, 6, 15, 0, tzinfo=timezone.utc)

    obs = get_nearest_observation(observations, target_time, max_gap_minutes=20.0)
    assert obs is not None

    with pytest.raises(EnvironmentalDataUnavailableError):
        get_nearest_observation(observations, target_time, max_gap_minutes=10.0)


def test_missing_observation_handling():
    target_time = datetime(2026, 8, 26, 6, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(EnvironmentalDataUnavailableError):
        get_nearest_observation([], target_time)


def test_nearest_timestamp_rejects_naive_target():
    obs = EnvironmentalObservation(
        timestamp=datetime(2026, 8, 26, 6, 0, 0, tzinfo=timezone.utc),
        latitude=18.92,
        longitude=72.83,
        wind_speed_mps=5.0,
        wind_direction_deg=90.0,
        current_speed_mps=0.3,
        current_direction_deg=180.0,
    )
    naive_target = datetime(2026, 8, 26, 6, 0, 0)
    with pytest.raises(ValueError):
        get_nearest_observation([obs], naive_target)


# --- FILE ENVIRONMENTAL PROVIDER TESTS ---

def test_file_provider_loads_valid_csv(demo_csv_path):
    """Requirement 1 & 3: File provider loads valid CSV and parses UTC timestamps."""
    file_prov = FileEnvironmentalProvider(demo_csv_path)
    start = datetime(2026, 8, 27, 9, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 27, 11, 0, 0, tzinfo=timezone.utc)

    obs_list = file_prov.get_observations(start, end, 18.92, 72.83)
    assert len(obs_list) == 5
    for obs in obs_list:
        assert isinstance(obs, EnvironmentalObservation)
        assert obs.timestamp.tzinfo is timezone.utc


def test_file_provider_loads_valid_json():
    """Requirement 2: File provider loads valid JSON input."""
    json_data = [
        {
            "timestamp": "2026-08-27T10:00:00Z",
            "latitude": 18.92,
            "longitude": 72.83,
            "wind_speed_mps": 8.5,
            "wind_direction_deg": 150.0,
            "current_speed_mps": 0.4,
            "current_direction_deg": 210.0,
        },
        {
            "timestamp": "2026-08-27T10:30:00Z",
            "latitude": 18.92,
            "longitude": 72.83,
            "wind_speed_mps": 8.7,
            "wind_direction_deg": 152.0,
            "current_speed_mps": 0.42,
            "current_direction_deg": 211.0,
        },
    ]
    json_prov = FileEnvironmentalProvider(json_data)
    start = datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 27, 10, 30, 0, tzinfo=timezone.utc)

    obs = json_prov.get_observations(start, end, 18.92, 72.83)
    assert len(obs) == 2
    assert obs[0].wind_speed_mps == 8.5


def test_file_provider_rejects_invalid_timestamps():
    """Requirement 4: Invalid/naive timestamps are rejected."""
    # Naive timestamp string (no Z or offset)
    naive_json = [{
        "timestamp": "2026-08-27T10:00:00",
        "latitude": 18.92,
        "longitude": 72.83,
        "wind_speed_mps": 5.0,
        "wind_direction_deg": 90.0,
        "current_speed_mps": 0.3,
        "current_direction_deg": 180.0,
    }]
    with pytest.raises(ValueError):
        FileEnvironmentalProvider(naive_json)


def test_file_provider_rejects_invalid_coordinates():
    """Requirement 5: Invalid coordinates are rejected."""
    bad_coords_json = [{
        "timestamp": "2026-08-27T10:00:00Z",
        "latitude": 95.0,  # Out of bounds
        "longitude": 72.83,
        "wind_speed_mps": 5.0,
        "wind_direction_deg": 90.0,
        "current_speed_mps": 0.3,
        "current_direction_deg": 180.0,
    }]
    with pytest.raises(ValueError):
        FileEnvironmentalProvider(bad_coords_json)


def test_file_provider_rejects_invalid_speeds():
    """Requirement 6: Invalid wind/current speeds (negative) are rejected."""
    negative_wind_json = [{
        "timestamp": "2026-08-27T10:00:00Z",
        "latitude": 18.92,
        "longitude": 72.83,
        "wind_speed_mps": -5.0,
        "wind_direction_deg": 90.0,
        "current_speed_mps": 0.3,
        "current_direction_deg": 180.0,
    }]
    with pytest.raises(ValueError):
        FileEnvironmentalProvider(negative_wind_json)


def test_file_provider_rejects_missing_columns():
    """Requirement 7: Missing required fields are rejected."""
    missing_col_json = [{
        "timestamp": "2026-08-27T10:00:00Z",
        "latitude": 18.92,
        "longitude": 72.83,
        "wind_speed_mps": 5.0,
        # missing wind_direction_deg
        "current_speed_mps": 0.3,
        "current_direction_deg": 180.0,
    }]
    with pytest.raises(ValueError):
        FileEnvironmentalProvider(missing_col_json)


def test_forward_simulator_works_with_file_provider(demo_csv_path):
    """Requirement 10: ForwardSimulator works with FileEnvironmentalProvider without core changes."""
    file_prov = FileEnvironmentalProvider(demo_csv_path)
    sim = ForwardSimulator(env_provider=file_prov)

    start_time = datetime(2026, 8, 27, 9, 0, 0, tzinfo=timezone.utc)
    trajectory = sim.run(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=2.0,
        step_minutes=30.0,
    )
    assert len(trajectory) == 5
    assert trajectory[0].latitude == 18.92


def test_backward_simulator_works_with_file_provider(demo_csv_path):
    """Requirement 11: BackwardSimulator works with FileEnvironmentalProvider without core changes."""
    file_prov = FileEnvironmentalProvider(demo_csv_path)
    sim = BackwardSimulator(env_provider=file_prov)

    spill_time = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
    trajectory = sim.run(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
    )
    assert len(trajectory) == 5
    assert trajectory[0].timestamp == spill_time


def test_pipeline_works_with_file_provider(demo_csv_path):
    """Requirement 12: DriftOriginPipeline works with FileEnvironmentalProvider without core changes."""
    file_prov = FileEnvironmentalProvider(demo_csv_path)
    pipeline = DriftOriginPipeline(env_provider=file_prov)

    detection_time = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
    result = pipeline.analyze_spill(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        spill_id="SPILL-FILE-PROV-01",
        duration_hours=2.0,
        step_minutes=30.0,
    )

    assert result.status == "origin_estimated"
    assert result.spill_id == "SPILL-FILE-PROV-01"
    assert len(result.backward_trajectory) == 5


def test_real_provider_stub_behavior():
    """Verify placeholder RealEnvironmentalProvider raises NotImplementedError."""
    real_prov = RealEnvironmentalProvider(api_key="TEST_KEY")
    start = datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(NotImplementedError):
        real_prov.get_observations(start, end, 18.92, 72.83)
