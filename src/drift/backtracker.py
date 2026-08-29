"""
Backward Oil-Spill Trajectory Simulator.

Reconstructs the historical movement path of an oil spill by stepping backward in time
from a known detection location and timestamp using environmental observations and drift physics.

Scientific Scope & Model Assumptions:
- Simplified SIH Prototype backward surface drift model:
      backward_velocity = -1 * (current_velocity + windage_factor * wind_velocity)
- Default windage factor is 0.03 (3% rule of thumb).
- Moving backward steps in the opposite direction of the calculated forward oil velocity vector.
- Uses spherical Earth geodesy (destination_point) for backward geographic position updates.
- Uses nearest-timestamp environmental lookup; NO interpolation is performed.
- Synthetic data provided by SyntheticEnvironmentalProvider is DEMO data for testing.
- This is NOT a scientifically complete oceanographic hindcast model.
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


class BackwardSimulator:
    """
    Simulator for computing backward surface oil-spill trajectory.
    """

    def __init__(
        self,
        env_provider: EnvironmentalDataProvider,
        windage_factor: float = 0.03,
        max_gap_minutes: Optional[float] = 60.0,
    ):
        """
        Initialize the backward simulator.

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
        spill_latitude: float,
        spill_longitude: float,
        spill_timestamp: datetime,
        duration_hours: float,
        step_minutes: float,
    ) -> List[TrajectoryPoint]:
        """
        Run backward simulation to compute historical trajectory points.

        :param spill_latitude: Known spill latitude in degrees [-90, 90]
        :param spill_longitude: Known spill longitude in degrees [-180, 180]
        :param spill_timestamp: Known spill detection timestamp (timezone-aware)
        :param duration_hours: Total backward duration in hours (>= 0)
        :param step_minutes: Simulation step size in minutes (> 0)
        :return: Chronological backward list of TrajectoryPoint instances starting from spill_timestamp
        :raises ValueError: If input parameters or timestamps are invalid
        """
        return simulate_backward_trajectory(
            spill_latitude=spill_latitude,
            spill_longitude=spill_longitude,
            spill_timestamp=spill_timestamp,
            duration_hours=duration_hours,
            step_minutes=step_minutes,
            env_provider=self.env_provider,
            windage_factor=self.windage_factor,
            max_gap_minutes=self.max_gap_minutes,
        )


def simulate_backward_trajectory(
    spill_latitude: float,
    spill_longitude: float,
    spill_timestamp: datetime,
    duration_hours: float,
    step_minutes: float,
    env_provider: EnvironmentalDataProvider,
    windage_factor: float = 0.03,
    max_gap_minutes: Optional[float] = 60.0,
) -> List[TrajectoryPoint]:
    """
    Simulates backward surface oil trajectory stepping backward through time.

    Ordering:
        trajectory[0] = known spill detection position/time (T_0)
        trajectory[1] = position at T_0 - step
        trajectory[2] = position at T_0 - 2*step
        ...

    :param spill_latitude: Known spill latitude in degrees [-90, 90]
    :param spill_longitude: Known spill longitude in degrees [-180, 180]
    :param spill_timestamp: Known spill detection timestamp (must be timezone-aware)
    :param duration_hours: Backward duration in hours (>= 0)
    :param step_minutes: Simulation step size in minutes (> 0)
    :param env_provider: EnvironmentalDataProvider instance
    :param windage_factor: Windage coefficient (default: 0.03)
    :param max_gap_minutes: Maximum allowable gap for environmental lookup in minutes
    :return: List of TrajectoryPoint instances ordered from spill timestamp backward in time
    :raises ValueError: For invalid coordinates, naive timestamps, or invalid duration/step.
    """
    validate_coordinates(spill_latitude, spill_longitude)

    if spill_timestamp is None or spill_timestamp.tzinfo is None or spill_timestamp.tzinfo.utcoffset(spill_timestamp) is None:
        raise ValueError("spill_timestamp must be timezone-aware.")

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
    start_timestamp = spill_timestamp - timedelta(hours=dur_hours)

    # Retrieve environmental observations covering the backward search window
    buffer_mins = float(max_gap_minutes or 60.0)
    fetch_start = start_timestamp - timedelta(minutes=buffer_mins)
    fetch_end = spill_timestamp + timedelta(minutes=buffer_mins)

    try:
        env_observations = env_provider.get_observations(
            start_time=fetch_start,
            end_time=fetch_end,
            latitude=spill_latitude,
            longitude=spill_longitude,
            interval_minutes=min(step_mins, 30.0),
        )
    except Exception as e:
        raise EnvironmentalDataUnavailableError(f"Failed to fetch environmental data: {str(e)}") from e

    trajectory: List[TrajectoryPoint] = []
    curr_lat = float(spill_latitude)
    curr_lon = float(spill_longitude)
    curr_time = spill_timestamp
    step_seconds = step_mins * 60.0

    while True:
        # 1. Lookup nearest environmental observation for current backward timestamp
        try:
            obs = get_nearest_observation(
                observations=env_observations,
                target_time=curr_time,
                max_gap_minutes=max_gap_minutes,
            )
        except EnvironmentalDataUnavailableError:
            # Stop backward trajectory generation if environmental data is unavailable
            break

        # 2. Record current backward trajectory point
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
        if curr_time <= start_timestamp:
            break

        # Adjust final backward step time delta if remaining time is less than step_seconds
        remaining_seconds = (curr_time - start_timestamp).total_seconds()
        actual_step_seconds = min(step_seconds, remaining_seconds)
        if actual_step_seconds <= 0:
            break

        # 3. Convert wind and current to 2D vectors
        wind_vec = wind_to_vector(obs.wind_speed_mps, obs.wind_direction_deg)
        curr_vec = current_to_vector(obs.current_speed_mps, obs.current_direction_deg)

        # 4. Calculate forward oil velocity vector
        oil_vec = combine_drift_velocity(wind_vec, curr_vec, windage_factor=wf)

        # 5. Extract forward speed and bearing
        speed_mps, forward_bearing = speed_and_bearing_from_vector(oil_vec)

        # 6. Opposite bearing for backward movement (-V_oil)
        backward_bearing = (forward_bearing + 180.0) % 360.0

        # 7. Calculate backward displacement distance
        distance_m = speed_mps * actual_step_seconds

        # 8. Update geographic position moving backward
        if speed_mps > 0.0 and distance_m > 0.0:
            prev_lat, prev_lon = destination_point(
                latitude=curr_lat,
                longitude=curr_lon,
                bearing_deg=backward_bearing,
                distance_m=distance_m,
            )
            curr_lat, curr_lon = prev_lat, prev_lon
        # If velocity is zero, position remains unchanged

        # 9. Step backward in time
        curr_time -= timedelta(seconds=actual_step_seconds)

    return trajectory
