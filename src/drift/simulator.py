"""
Forward Oil-Spill Trajectory Simulator.

Simulates the forward motion of an oil spill over time based on environmental
wind and ocean current observations.

Scientific Scope & Model Assumptions:
- Simplified SIH Prototype forward surface drift model:
      oil_velocity = current_velocity + windage_factor * wind_velocity
- Default windage factor is 0.03 (3% rule of thumb).
- Uses spherical Earth geodesy (destination_point) for geographic position updates.
- Uses nearest-timestamp environmental lookup; NO interpolation is performed.
- Synthetic data provided by SyntheticEnvironmentalProvider is DEMO data for testing.
- This is NOT a production-grade oceanographic forecasting model.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .geo_utils import validate_coordinates, destination_point
from .models import TrajectoryPoint, EnvironmentalObservation
from .environment import (
    EnvironmentalDataProvider,
    EnvironmentalDataUnavailableError,
    get_nearest_observation,
)
from .physics import (
    wind_to_vector,
    current_to_vector,
    combine_drift_velocity,
    speed_and_bearing_from_vector,
)


class ForwardSimulator:
    """
    Simulator for computing forward surface oil-spill trajectory.
    """

    def __init__(
        self,
        env_provider: EnvironmentalDataProvider,
        windage_factor: float = 0.03,
        max_gap_minutes: Optional[float] = 60.0,
    ):
        """
        Initialize the forward simulator.

        :param env_provider: EnvironmentalDataProvider instance
        :param windage_factor: Windage factor for oil drift (default: 0.03)
        :param max_gap_minutes: Maximum allowable gap for environmental lookup in minutes
        """
        if env_provider is None or not isinstance(env_provider, EnvironmentalDataProvider):
            raise ValueError("env_provider must be an instance of EnvironmentalDataProvider.")

        if windage_factor is None or isinstance(windage_factor, bool) or not isinstance(windage_factor, (int, float)):
            raise ValueError("windage_factor must be a numeric value.")
        if float(windage_factor) < 0:
            raise ValueError(f"windage_factor cannot be negative, got {windage_factor}")

        self.env_provider = env_provider
        self.windage_factor = float(windage_factor)
        self.max_gap_minutes = max_gap_minutes

    def run(
        self,
        start_latitude: float,
        start_longitude: float,
        start_timestamp: datetime,
        duration_hours: float,
        step_minutes: float,
    ) -> List[TrajectoryPoint]:
        """
        Run forward simulation to compute trajectory points.

        :param start_latitude: Initial latitude in degrees [-90, 90]
        :param start_longitude: Initial longitude in degrees [-180, 180]
        :param start_timestamp: Inclusive start datetime (timezone-aware)
        :param duration_hours: Total simulation duration in hours (>= 0)
        :param step_minutes: Simulation step size in minutes (> 0)
        :return: Chronological list of TrajectoryPoint instances including start and end steps
        :raises ValueError: If input parameters or timestamps are invalid
        """
        return simulate_forward_trajectory(
            start_latitude=start_latitude,
            start_longitude=start_longitude,
            start_timestamp=start_timestamp,
            duration_hours=duration_hours,
            step_minutes=step_minutes,
            env_provider=self.env_provider,
            windage_factor=self.windage_factor,
            max_gap_minutes=self.max_gap_minutes,
        )


def simulate_forward_trajectory(
    start_latitude: float,
    start_longitude: float,
    start_timestamp: datetime,
    duration_hours: float,
    step_minutes: float,
    env_provider: EnvironmentalDataProvider,
    windage_factor: float = 0.03,
    max_gap_minutes: Optional[float] = 60.0,
) -> List[TrajectoryPoint]:
    """
    Simulates forward surface oil trajectory over a specified time duration and step interval.

    :param start_latitude: Initial latitude in degrees [-90, 90]
    :param start_longitude: Initial longitude in degrees [-180, 180]
    :param start_timestamp: Inclusive start datetime (must be timezone-aware)
    :param duration_hours: Simulation duration in hours (>= 0)
    :param step_minutes: Simulation step size in minutes (> 0)
    :param env_provider: EnvironmentalDataProvider instance
    :param windage_factor: Windage coefficient (default: 0.03)
    :param max_gap_minutes: Maximum allowable gap for environmental data lookup in minutes
    :return: Chronological list of TrajectoryPoint instances
    :raises ValueError: For invalid coordinates, naive timestamps, or invalid duration/step.
    """
    validate_coordinates(start_latitude, start_longitude)

    if start_timestamp is None or start_timestamp.tzinfo is None or start_timestamp.tzinfo.utcoffset(start_timestamp) is None:
        raise ValueError("start_timestamp must be timezone-aware.")

    if duration_hours is None or isinstance(duration_hours, bool) or not isinstance(duration_hours, (int, float)):
        raise ValueError("duration_hours must be a numeric value.")
    if float(duration_hours) < 0:
        raise ValueError(f"duration_hours cannot be negative, got {duration_hours}")

    if step_minutes is None or isinstance(step_minutes, bool) or not isinstance(step_minutes, (int, float)):
        raise ValueError("step_minutes must be a numeric value.")
    if float(step_minutes) <= 0:
        raise ValueError(f"step_minutes must be > 0, got {step_minutes}")

    if env_provider is None or not isinstance(env_provider, EnvironmentalDataProvider):
        raise ValueError("env_provider must be an instance of EnvironmentalDataProvider.")

    wf = float(windage_factor)
    if wf < 0:
        raise ValueError(f"windage_factor cannot be negative, got {windage_factor}")

    dur_hours = float(duration_hours)
    step_mins = float(step_minutes)
    end_timestamp = start_timestamp + timedelta(hours=dur_hours)

    # Retrieve environmental observations from provider covering the window
    # Buffer search window by step_minutes + max_gap to ensure complete coverage
    fetch_start = start_timestamp - timedelta(minutes=float(max_gap_minutes or 60.0))
    fetch_end = end_timestamp + timedelta(minutes=float(max_gap_minutes or 60.0))

    try:
        env_observations = env_provider.get_observations(
            start_time=fetch_start,
            end_time=fetch_end,
            latitude=start_latitude,
            longitude=start_longitude,
            interval_minutes=min(step_mins, 30.0),
        )
    except Exception as e:
        raise EnvironmentalDataUnavailableError(f"Failed to fetch environmental data: {str(e)}") from e

    trajectory: List[TrajectoryPoint] = []
    curr_lat = float(start_latitude)
    curr_lon = float(start_longitude)
    curr_time = start_timestamp
    step_seconds = step_mins * 60.0

    while True:
        # 1. Lookup nearest environmental observation
        try:
            obs = get_nearest_observation(
                observations=env_observations,
                target_time=curr_time,
                max_gap_minutes=max_gap_minutes,
            )
        except EnvironmentalDataUnavailableError:
            # Stop trajectory generation if environmental data is unavailable within max gap
            break

        # 2. Record current trajectory point
        point = TrajectoryPoint(
            timestamp=curr_time,
            latitude=round(curr_lat, 8),
            longitude=round(curr_lon, 8),
            wind_speed_mps=obs.wind_speed_mps,
            wind_direction_deg=obs.wind_direction_deg,
            current_speed_mps=obs.current_speed_mps,
            current_direction_deg=obs.current_direction_deg,
        )
        trajectory.append(point)

        # Check loop termination condition
        if curr_time >= end_timestamp:
            break

        # Adjust final step time delta if remaining time is less than step_seconds
        remaining_seconds = (end_timestamp - curr_time).total_seconds()
        actual_step_seconds = min(step_seconds, remaining_seconds)
        if actual_step_seconds <= 0:
            break

        # 3. Convert wind and current to 2D vectors
        wind_vec = wind_to_vector(obs.wind_speed_mps, obs.wind_direction_deg)
        curr_vec = current_to_vector(obs.current_speed_mps, obs.current_direction_deg)

        # 4. Calculate combined oil velocity vector
        oil_vec = combine_drift_velocity(wind_vec, curr_vec, windage_factor=wf)

        # 5. Extract speed and bearing
        speed_mps, bearing_deg = speed_and_bearing_from_vector(oil_vec)

        # 6. Calculate displacement distance
        distance_m = speed_mps * actual_step_seconds

        # 7. Update geographic position using destination_point
        if speed_mps > 0.0 and distance_m > 0.0:
            next_lat, next_lon = destination_point(
                latitude=curr_lat,
                longitude=curr_lon,
                bearing_deg=bearing_deg,
                distance_m=distance_m,
            )
            curr_lat, curr_lon = next_lat, next_lon
        # If velocity is zero, position remains unchanged

        # 8. Advance simulation timestamp
        curr_time += timedelta(seconds=actual_step_seconds)

    return trajectory
