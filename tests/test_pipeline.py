"""
Unit and integration tests for Member 2 Drift Origin Pipeline (src/drift/pipeline.py).
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.drift.pipeline import DriftOriginPipeline, run_drift_pipeline, DriftPipelineResult
from src.drift.environment import SyntheticEnvironmentalProvider, EnvironmentalDataProvider
from src.drift.models import EnvironmentalObservation


@pytest.fixture
def pipeline():
    return DriftOriginPipeline()


@pytest.fixture
def detection_time():
    return datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)


class MockFailingEnvProvider(EnvironmentalDataProvider):
    """Provider that returns empty list to simulate unavailable environmental data."""
    def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
        return []


def test_end_to_end_pipeline_success(pipeline, detection_time):
    """Requirements 1, 2, 3, 4, 5, 6, 7, 8, 9 & 18: Real end-to-end pipeline execution with SyntheticEnvironmentalProvider."""
    result = pipeline.analyze_spill(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        spill_id="SPILL-2026-TEST",
        duration_hours=2.0,
        step_minutes=30.0,
    )

    assert isinstance(result, DriftPipelineResult)
    assert result.spill_id == "SPILL-2026-TEST"
    assert result.status == "origin_estimated"
    assert result.detected_latitude == 18.92
    assert result.detected_longitude == 72.83
    assert result.detection_timestamp == detection_time
    assert result.trajectory_points_used == 5
    assert len(result.backward_trajectory) == 5

    # Origin candidate must differ from detection point for a moving fluid
    assert result.probable_latitude != result.detected_latitude or \
           result.probable_longitude != result.detected_longitude

    # Estimated release time must be 2 hours before detection time (10:00 UTC)
    expected_release_time = detection_time - timedelta(hours=2.0)
    assert result.estimated_release_time == expected_release_time


def test_pipeline_handles_insufficient_history(detection_time):
    """Requirement 10: Pipeline handles insufficient trajectory points."""
    # 0 hours duration means only 1 trajectory point at detection time
    result = run_drift_pipeline(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        duration_hours=0.0,
        step_minutes=30.0,
    )

    assert result.status == "insufficient_trajectory"
    assert result.trajectory_points_used == 1
    assert result.probable_latitude == 18.92
    assert result.probable_longitude == 72.83


def test_pipeline_handles_missing_environmental_data(detection_time):
    """Requirement 11: Pipeline handles missing environmental data gracefully."""
    failing_provider = MockFailingEnvProvider()
    result = run_drift_pipeline(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        env_provider=failing_provider,
    )

    assert result.status == "environmental_data_unavailable"
    assert result.trajectory_points_used == 0
    assert result.backward_trajectory == []


def test_pipeline_rejects_invalid_coordinates(pipeline, detection_time):
    """Requirement 12: Pipeline rejects invalid coordinates."""
    with pytest.raises(ValueError):
        pipeline.analyze_spill(95.0, 72.83, detection_time)
    with pytest.raises(ValueError):
        pipeline.analyze_spill(18.92, -200.0, detection_time)


def test_pipeline_rejects_naive_timestamps(pipeline):
    """Requirement 13: Pipeline rejects timezone-naive timestamps."""
    naive_time = datetime(2026, 8, 27, 12, 0, 0)
    with pytest.raises(ValueError):
        pipeline.analyze_spill(18.92, 72.83, naive_time)


def test_pipeline_respects_custom_duration(pipeline, detection_time):
    """Requirement 14: Pipeline respects custom duration."""
    # 4 hours duration with 30 min step -> 9 trajectory points
    result = pipeline.analyze_spill(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        duration_hours=4.0,
        step_minutes=30.0,
    )
    assert result.trajectory_points_used == 9
    assert result.estimated_release_time == detection_time - timedelta(hours=4.0)


def test_pipeline_respects_custom_step_size(pipeline, detection_time):
    """Requirement 15: Pipeline respects custom step size."""
    # 2 hours duration with 60 min step -> 3 trajectory points (0, 60, 120 min)
    result = pipeline.analyze_spill(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        duration_hours=2.0,
        step_minutes=60.0,
    )
    assert result.trajectory_points_used == 3


def test_pipeline_respects_custom_windage_factor(detection_time):
    """Requirement 16: Pipeline respects custom windage factor."""
    res_low = run_drift_pipeline(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        windage_factor=0.01,
    )
    res_high = run_drift_pipeline(
        latitude=18.92,
        longitude=72.83,
        detection_timestamp=detection_time,
        windage_factor=0.05,
    )

    assert res_low.probable_latitude != res_high.probable_latitude or \
           res_low.probable_longitude != res_high.probable_longitude
