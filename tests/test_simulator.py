"""
Unit tests for Forward Oil-Spill Simulator (src/drift/simulator.py).
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.drift.simulator import ForwardSimulator, simulate_forward_trajectory
from src.drift.environment import SyntheticEnvironmentalProvider, EnvironmentalDataProvider
from src.drift.models import TrajectoryPoint, EnvironmentalObservation
from src.drift.geo_utils import distance_between_points


@pytest.fixture
def env_provider():
    return SyntheticEnvironmentalProvider()


@pytest.fixture
def start_time():
    return datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)


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
    """Mock provider returning 1 m/s Eastward current and 0 wind."""
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
                    current_direction_deg=90.0,  # 90° = East
                )
            )
            curr += timedelta(minutes=interval_minutes)
        return obs_list


class MockNorthMovementProvider(EnvironmentalDataProvider):
    """Mock provider returning 1 m/s Northward current and 0 wind."""
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
                    current_direction_deg=0.0,  # 0° = North
                )
            )
            curr += timedelta(minutes=interval_minutes)
        return obs_list


def test_simulator_produces_trajectory_points(env_provider, start_time):
    """Requirement 1: Simulator produces trajectory points."""
    sim = ForwardSimulator(env_provider)
    points = sim.run(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=2.0,
        step_minutes=30.0,
    )
    assert len(points) > 0
    for p in points:
        assert isinstance(p, TrajectoryPoint)


def test_starting_location_is_correct(env_provider, start_time):
    """Requirement 2: Starting location is correct."""
    points = simulate_forward_trajectory(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=1.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    assert points[0].latitude == pytest.approx(18.92)
    assert points[0].longitude == pytest.approx(72.83)


def test_starting_timestamp_is_correct(env_provider, start_time):
    """Requirement 3: Starting timestamp is correct."""
    points = simulate_forward_trajectory(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=1.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    assert points[0].timestamp == start_time


def test_timestamps_are_chronological(env_provider, start_time):
    """Requirement 4: Timestamps are chronological."""
    points = simulate_forward_trajectory(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    for i in range(len(points) - 1):
        assert points[i].timestamp < points[i + 1].timestamp


def test_correct_number_of_trajectory_points(env_provider, start_time):
    """Requirement 5: Correct number of trajectory points for known duration/step."""
    # 2 hours duration with 30-minute step -> 5 points (0, 30, 60, 90, 120 minutes)
    points = simulate_forward_trajectory(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
    )
    assert len(points) == 5


def test_eastward_movement_changes_longitude(start_time):
    """Requirement 6: Eastward movement changes longitude appropriately."""
    mock_env = MockEastMovementProvider()
    points = simulate_forward_trajectory(
        start_latitude=0.0,
        start_longitude=0.0,
        start_timestamp=start_time,
        duration_hours=1.0,  # 3600 seconds @ 1 m/s East = 3600 meters East
        step_minutes=60.0,
        env_provider=mock_env,
    )
    assert len(points) == 2
    p0, p1 = points[0], points[1]
    assert p1.latitude == pytest.approx(0.0, abs=1e-4)
    assert p1.longitude > p0.longitude
    
    # Check total distance travelled ~ 3600 meters
    dist = distance_between_points(p0.latitude, p0.longitude, p1.latitude, p1.longitude)
    assert dist == pytest.approx(3600.0, rel=1e-3)


def test_northward_movement_changes_latitude(start_time):
    """Requirement 7: Northward movement changes latitude appropriately."""
    mock_env = MockNorthMovementProvider()
    points = simulate_forward_trajectory(
        start_latitude=0.0,
        start_longitude=0.0,
        start_timestamp=start_time,
        duration_hours=1.0,  # 3600 seconds @ 1 m/s North = 3600 meters North
        step_minutes=60.0,
        env_provider=mock_env,
    )
    assert len(points) == 2
    p0, p1 = points[0], points[1]
    assert p1.longitude == pytest.approx(0.0, abs=1e-4)
    assert p1.latitude > p0.latitude
    
    dist = distance_between_points(p0.latitude, p0.longitude, p1.latitude, p1.longitude)
    assert dist == pytest.approx(3600.0, rel=1e-3)


def test_zero_velocity_keeps_location_unchanged(start_time):
    """Requirement 8: Zero velocity keeps location unchanged."""
    mock_env = MockZeroVelocityProvider()
    points = simulate_forward_trajectory(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=mock_env,
    )
    assert len(points) == 5
    for p in points:
        assert p.latitude == pytest.approx(18.92)
        assert p.longitude == pytest.approx(72.83)


def test_windage_factor_affects_movement(env_provider, start_time):
    """Requirement 9 & 10: Windage factor affects movement and custom factor works."""
    pts_default = simulate_forward_trajectory(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
        windage_factor=0.03,
    )

    pts_high_windage = simulate_forward_trajectory(
        start_latitude=18.92,
        start_longitude=72.83,
        start_timestamp=start_time,
        duration_hours=2.0,
        step_minutes=30.0,
        env_provider=env_provider,
        windage_factor=0.06,
    )

    # Higher windage factor must result in different/further displacement
    assert pts_default[-1].latitude != pts_high_windage[-1].latitude or \
           pts_default[-1].longitude != pts_high_windage[-1].longitude


def test_invalid_coordinates_rejected(env_provider, start_time):
    """Requirement 11: Invalid coordinates are rejected."""
    with pytest.raises(ValueError):
        simulate_forward_trajectory(95.0, 72.83, start_time, 1.0, 30.0, env_provider)
    with pytest.raises(ValueError):
        simulate_forward_trajectory(18.92, -200.0, start_time, 1.0, 30.0, env_provider)


def test_naive_timestamp_rejected(env_provider):
    """Requirement 12: Naive timestamp is rejected."""
    naive_time = datetime(2026, 8, 27, 10, 0, 0)
    with pytest.raises(ValueError):
        simulate_forward_trajectory(18.92, 72.83, naive_time, 1.0, 30.0, env_provider)


def test_invalid_duration_step_rejected(env_provider, start_time):
    """Requirement 13: Invalid simulation duration/step is rejected."""
    # Negative duration
    with pytest.raises(ValueError):
        simulate_forward_trajectory(18.92, 72.83, start_time, -1.0, 30.0, env_provider)
    # Zero or negative step
    with pytest.raises(ValueError):
        simulate_forward_trajectory(18.92, 72.83, start_time, 1.0, 0.0, env_provider)
    with pytest.raises(ValueError):
        simulate_forward_trajectory(18.92, 72.83, start_time, 1.0, -10.0, env_provider)


def test_environmental_provider_is_actually_used(start_time):
    """Requirement 14: Environmental provider is invoked."""
    class CustomSpyProvider(EnvironmentalDataProvider):
        def __init__(self):
            self.invoked = False
        def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
            self.invoked = True
            return SyntheticEnvironmentalProvider().get_observations(
                start_time, end_time, latitude, longitude, interval_minutes
            )

    spy = CustomSpyProvider()
    simulate_forward_trajectory(18.92, 72.83, start_time, 1.0, 30.0, spy)
    assert spy.invoked is True


def test_missing_environmental_data_handled(start_time):
    """Requirement 15: Missing environmental data is handled without fabricated values."""
    class EmptyProvider(EnvironmentalDataProvider):
        def get_observations(self, start_time, end_time, latitude, longitude, interval_minutes=30.0):
            return []

    empty_prov = EmptyProvider()
    # Empty provider results in 0 trajectory points (simulation halts without crashing or fabricating data)
    points = simulate_forward_trajectory(18.92, 72.83, start_time, 1.0, 30.0, empty_prov)
    assert len(points) == 0
