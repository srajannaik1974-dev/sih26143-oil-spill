"""
Unit tests for Backward Oil-Spill Simulator (src/drift/backtracker.py).
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.drift.backtracker import BackwardSimulator, simulate_backward_trajectory
from src.drift.environment import SyntheticEnvironmentalProvider, EnvironmentalDataProvider
from src.drift.models import TrajectoryPoint, EnvironmentalObservation
from src.drift.geo_utils import distance_between_points


@pytest.fixture
def env_provider():
    return SyntheticEnvironmentalProvider()


@pytest.fixture
def spill_time():
    return datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)


class MockZeroVelocityProvider(EnvironmentalDataProvider):
    """Mock provider returning zero wind and zero current."""
    def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
        obs_list = []
        curr = start_time
        while curr <= end_time:
            obs_list.append(
                EnvironmentalObservation(
                    timestamp=curr,
                    latitude=latitude,
                    longitude=longitude,
                    wind_speed_mps=0.0,
                    wind_direction_deg=0.0,
                    current_speed_mps=0.0,
                    current_direction_deg=0.0,
                )
            )
            curr += timedelta(minutes=interval_minutes)
        return obs_list


class MockEastMovementProvider(EnvironmentalDataProvider):
    """Mock provider returning 1 m/s Eastward current (Forward = East, Backward = West)."""
    def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
        obs_list = []
        curr = start_time
        while curr <= end_time:
            obs_list.append(
                EnvironmentalObservation(
                    timestamp=curr,
                    latitude=latitude,
                    longitude=longitude,
                    wind_speed_mps=0.0,
                    wind_direction_deg=0.0,
                    current_speed_mps=1.0,
                    current_direction_deg=90.0,  # Forward East (90°)
                )
            )
            curr += timedelta(minutes=interval_minutes)
        return obs_list


class MockNorthMovementProvider(EnvironmentalDataProvider):
    """Mock provider returning 1 m/s Northward current (Forward = North, Backward = South)."""
    def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
        obs_list = []
        curr = start_time
        while curr <= end_time:
            obs_list.append(
                EnvironmentalObservation(
                    timestamp=curr,
                    latitude=latitude,
                    longitude=longitude,
                    wind_speed_mps=0.0,
                    wind_direction_deg=0.0,
                    current_speed_mps=1.0,
                    current_direction_deg=0.0,  # Forward North (0°)
                )
            )
            curr += timedelta(minutes=interval_minutes)
        return obs_list


def test_backtracker_produces_trajectory_points(env_provider, spill_time):
    """Requirement 1 & 19: Backtracker produces TrajectoryPoint objects."""
    sim = BackwardSimulator(env_provider)
    points = sim.run(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
    )
    assert len(points) > 0
    for p in points:
        assert isinstance(p, TrajectoryPoint)


def test_first_point_matches_spill_location(env_provider, spill_time):
    """Requirement 2: First point matches the known spill location."""
    points = simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=1.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    assert points[0].latitude == pytest.approx(18.92)
    assert points[0].longitude == pytest.approx(72.83)


def test_first_point_timestamp_matches_spill_timestamp(env_provider, spill_time):
    """Requirement 3: First point timestamp matches the spill timestamp."""
    points = simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=1.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    assert points[0].timestamp == spill_time


def test_timestamps_move_backward_correctly(env_provider, spill_time):
    """Requirement 4: Timestamps move backward in time (T_0, T_0 - step, T_0 - 2*step)."""
    points = simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    assert points[0].timestamp == spill_time
    assert points[1].timestamp == spill_time - timedelta(minutes=30)
    assert points[2].timestamp == spill_time - timedelta(minutes=60)
    assert points[3].timestamp == spill_time - timedelta(minutes=90)
    assert points[4].timestamp == spill_time - timedelta(minutes=120)

    for i in range(len(points) - 1):
        assert points[i].timestamp > points[i + 1].timestamp


def test_expected_number_of_trajectory_points(env_provider, spill_time):
    """Requirement 5: Expected number of trajectory points for duration/step."""
    # 2 hours duration with 30-minute step -> 5 points (0, -30, -60, -90, -120 minutes)
    points = simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    assert len(points) == 5


def test_eastward_forward_drift_produces_westward_backward_movement(spill_time):
    """Requirement 6: Pure eastward forward drift produces westward backward movement."""
    mock_env = MockEastMovementProvider()
    points = simulate_backward_trajectory(
        spill_latitude=0.0,
        spill_longitude=0.0,
        spill_timestamp=spill_time,
        duration_hours=1.0,  # 3600 seconds @ 1 m/s Westward backward = 3600m West
        step_minutes=60.0,
        env_provider=mock_env,
    )
    assert len(points) == 2
    p0, p1 = points[0], points[1]
    assert p1.latitude == pytest.approx(0.0, abs=1e-4)
    # Moving Westward means longitude decreases
    assert p1.longitude < p0.longitude

    dist = distance_between_points(p0.latitude, p0.longitude, p1.latitude, p1.longitude)
    assert dist == pytest.approx(3600.0, rel=1e-3)


def test_northward_forward_drift_produces_southward_backward_movement(spill_time):
    """Requirement 7: Pure northward forward drift produces southward backward movement."""
    mock_env = MockNorthMovementProvider()
    points = simulate_backward_trajectory(
        spill_latitude=0.0,
        spill_longitude=0.0,
        spill_timestamp=spill_time,
        duration_hours=1.0,  # 3600 seconds @ 1 m/s Southward backward = 3600m South
        step_minutes=60.0,
        env_provider=mock_env,
    )
    assert len(points) == 2
    p0, p1 = points[0], points[1]
    assert p1.longitude == pytest.approx(0.0, abs=1e-4)
    # Moving Southward means latitude decreases
    assert p1.latitude < p0.latitude

    dist = distance_between_points(p0.latitude, p0.longitude, p1.latitude, p1.longitude)
    assert dist == pytest.approx(3600.0, rel=1e-3)


def test_zero_velocity_keeps_coordinates_unchanged(spill_time):
    """Requirement 8: Zero velocity keeps coordinates unchanged while timestamps move backward."""
    mock_env = MockZeroVelocityProvider()
    points = simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=mock_env,
    )
    assert len(points) == 5
    for p in points:
        assert p.latitude == pytest.approx(18.92)
        assert p.longitude == pytest.approx(72.83)


def test_windage_factor_affects_backward_movement(env_provider, spill_time):
    """Requirement 9 & 10: Existing windage factor affects backward displacement magnitude."""
    pts_default = simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
        windage_factor=0.03,
    )

    pts_high_windage = simulate_backward_trajectory(
        spill_latitude=18.92,
        spill_longitude=72.83,
        spill_timestamp=spill_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
        windage_factor=0.06,
    )

    assert pts_default[-1].latitude != pts_high_windage[-1].latitude or \
           pts_default[-1].longitude != pts_high_windage[-1].longitude


def test_invalid_coordinates_rejected(env_provider, spill_time):
    """Requirement 11: Invalid coordinates are rejected."""
    with pytest.raises(ValueError):
        simulate_backward_trajectory(95.0, 72.83, spill_time, 1.0, 30.0, env_provider)
    with pytest.raises(ValueError):
        simulate_backward_trajectory(18.92, -200.0, spill_time, 1.0, 30.0, env_provider)


def test_naive_spill_timestamp_rejected(env_provider):
    """Requirement 12: Naive spill timestamp is rejected."""
    naive_time = datetime(2026, 8, 27, 12, 0, 0)
    with pytest.raises(ValueError):
        simulate_backward_trajectory(18.92, 72.83, naive_time, 1.0, 30.0, env_provider)


def test_invalid_duration_and_step_rejected(env_provider, spill_time):
    """Requirement 13 & 14: Negative duration and zero/negative step are rejected."""
    with pytest.raises(ValueError):
        simulate_backward_trajectory(18.92, 72.83, spill_time, -1.0, 30.0, env_provider)
    with pytest.raises(ValueError):
        simulate_backward_trajectory(18.92, 72.83, spill_time, 1.0, 0.0, env_provider)
    with pytest.raises(ValueError):
        simulate_backward_trajectory(18.92, 72.83, spill_time, 1.0, -15.0, env_provider)


def test_environmental_provider_and_nearest_obs_used(spill_time):
    """Requirement 15 & 16: Environmental provider and nearest observation are used."""
    class CustomSpyProvider(EnvironmentalDataProvider):
        def __init__(self):
            self.invoked = False
        def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
            self.invoked = True
            return SyntheticEnvironmentalProvider().get_observations(
                start_time, end_time, latitude, longitude, interval_minutes
            )

    spy = CustomSpyProvider()
    simulate_backward_trajectory(18.92, 72.83, spill_time, 1.0, 30.0, spy)
    assert spy.invoked is True


def test_missing_environmental_data_handled(spill_time):
    """Requirement 17 & 18: Missing environmental data handled without fabricated values."""
    class EmptyProvider(EnvironmentalDataProvider):
        def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
            return []

    empty_prov = EmptyProvider()
    points = simulate_backward_trajectory(18.92, 72.83, spill_time, 1.0, 30.0, empty_prov)
    assert len(points) == 0
