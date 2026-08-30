"""
Unit tests for Probable Oil-Spill Origin Estimator (src/drift/origin.py).
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.drift.origin import (
    ProbableOriginEstimator,
    estimate_probable_origin,
    ProbableOriginResult,
    InsufficientTrajectoryError,
)
from src.drift.backtracker import simulate_backward_trajectory
from src.drift.environment import SyntheticEnvironmentalProvider
from src.drift.models import TrajectoryPoint


@pytest.fixture
def env_provider():
    return SyntheticEnvironmentalProvider()


@pytest.fixture
def spill_time():
    return datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_backward_trajectory(env_provider, spill_time):
    return simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )


def test_origin_estimator_accepts_valid_backward_trajectory(sample_backward_trajectory):
    """Requirement 1: Origin estimator accepts a valid backward trajectory."""
    estimator = ProbableOriginEstimator()
    result = estimator.estimate(sample_backward_trajectory, spill_id="SPILL-001")

    assert isinstance(result, ProbableOriginResult)
    assert result.spill_id == "SPILL-001"
    assert result.status == "origin_estimated"


def test_probable_origin_based_on_oldest_trajectory_point(sample_backward_trajectory):
    """Requirement 2: Probable origin is based on the oldest valid trajectory point."""
    result = estimate_probable_origin(sample_backward_trajectory)
    oldest_point = sample_backward_trajectory[-1]

    assert result.candidate_latitude == oldest_point.latitude
    assert result.candidate_longitude == oldest_point.longitude


def test_estimated_release_time_matches_oldest_timestamp(sample_backward_trajectory):
    """Requirement 3: Estimated release time matches the oldest valid trajectory timestamp."""
    result = estimate_probable_origin(sample_backward_trajectory)
    oldest_point = sample_backward_trajectory[-1]

    assert result.estimated_release_time == oldest_point.timestamp


def test_returned_coordinates_are_valid(sample_backward_trajectory):
    """Requirement 4: Returned coordinates are valid."""
    result = estimate_probable_origin(sample_backward_trajectory)
    assert -90.0 <= result.candidate_latitude <= 90.0
    assert -180.0 <= result.candidate_longitude <= 180.0


def test_timezone_aware_timestamps_preserved(sample_backward_trajectory):
    """Requirement 5: Timezone-aware timestamps are preserved."""
    result = estimate_probable_origin(sample_backward_trajectory)
    assert result.estimated_release_time.tzinfo is not None
    assert result.estimated_release_time.tzinfo.utcoffset(result.estimated_release_time) is not None


def test_empty_trajectory_handled_correctly():
    """Requirement 6: Empty trajectory raises InsufficientTrajectoryError."""
    estimator = ProbableOriginEstimator()
    with pytest.raises(InsufficientTrajectoryError):
        estimator.estimate([])
    with pytest.raises(InsufficientTrajectoryError):
        estimator.estimate(None)


def test_single_point_trajectory_handled_explicitly(spill_time):
    """Requirement 7 & 8: Single-point trajectory returns status 'insufficient_trajectory' without fabricating origin."""
    single_point = TrajectoryPoint(
        timestamp=spill_time,
        latitude=18.92,
        longitude=72.83,
        wind_speed_mps=5.0,
        wind_direction_deg=90.0,
        current_speed_mps=0.3,
        current_direction_deg=180.0,
    )
    result = estimate_probable_origin([single_point], spill_id="SPILL-SINGLE")

    assert result.status == "insufficient_trajectory"
    assert result.trajectory_points_used == 1
    assert result.candidate_latitude == 18.92
    assert result.candidate_longitude == 72.83
    assert result.estimated_release_time == spill_time


def test_result_contains_trajectory_points_used(sample_backward_trajectory):
    """Requirement 9: Result contains the number of trajectory points used."""
    result = estimate_probable_origin(sample_backward_trajectory)
    assert result.trajectory_points_used == len(sample_backward_trajectory)
    assert result.trajectory_points_used == 5
