"""
Drift Vector Physics for Oil-Spill Surface Drift.

Provides vector conversions and simplified surface oil drift velocity calculation.

Scientific Scope & Model Assumptions:
- Simplified SIH Prototype surface drift model:
      oil_velocity = current_velocity + windage_factor * wind_velocity
- Default windage factor is 0.03 (3% rule of thumb in ocean surface drift modeling).
- Vector Convention:
      0┬░ = North, 90┬░ = East, 180┬░ = South, 270┬░ = West
- Direction convention:
      Refers to the direction the flow (wind/current) is MOVING TOWARD.
- This module computes VELOCITY ONLY (m/s). Geographic coordinate movement is handled separately.
"""

import math
from typing import Tuple
from pydantic import BaseModel, ConfigDict

from .geo_utils import normalize_bearing


class Vector2D(BaseModel):
    """
    2D velocity vector in Cartesian components (East-North framework).
    
    Units: meters per second (m/s).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    east_mps: float
    north_mps: float


def wind_to_vector(speed_mps: float, direction_deg: float) -> Vector2D:
    """
    Converts wind speed and direction into 2D velocity vector components (east, north).

    Convention: direction_deg is the direction the wind is MOVING TOWARD.
    0┬░ = North, 90┬░ = East, 180┬░ = South, 270┬░ = West.

    :param speed_mps: Wind speed in m/s (must be >= 0)
    :param direction_deg: Direction in degrees
    :return: Vector2D instance with east_mps and north_mps
    :raises ValueError: If speed_mps is negative or not numeric
    """
    if speed_mps is None or isinstance(speed_mps, bool) or not isinstance(speed_mps, (int, float)):
        raise ValueError("Speed must be a numeric value.")
    
    speed = float(speed_mps)
    if speed < 0:
        raise ValueError(f"Wind speed cannot be negative, got {speed_mps}")

    bearing_deg = normalize_bearing(direction_deg)
    bearing_rad = math.radians(bearing_deg)

    east_mps = speed * math.sin(bearing_rad)
    north_mps = speed * math.cos(bearing_rad)

    return Vector2D(east_mps=round(east_mps, 8), north_mps=round(north_mps, 8))


def current_to_vector(speed_mps: float, direction_deg: float) -> Vector2D:
    """
    Converts ocean current speed and direction into 2D velocity vector components (east, north).

    Convention: direction_deg is the direction the ocean current is MOVING TOWARD.
    0┬░ = North, 90┬░ = East, 180┬░ = South, 270┬░ = West.

    :param speed_mps: Current speed in m/s (must be >= 0)
    :param direction_deg: Direction in degrees
    :return: Vector2D instance with east_mps and north_mps
    :raises ValueError: If speed_mps is negative or not numeric
    """
    if speed_mps is None or isinstance(speed_mps, bool) or not isinstance(speed_mps, (int, float)):
        raise ValueError("Speed must be a numeric value.")

    speed = float(speed_mps)
    if speed < 0:
        raise ValueError(f"Current speed cannot be negative, got {speed_mps}")

    bearing_deg = normalize_bearing(direction_deg)
    bearing_rad = math.radians(bearing_deg)

    east_mps = speed * math.sin(bearing_rad)
    north_mps = speed * math.cos(bearing_rad)

    return Vector2D(east_mps=round(east_mps, 8), north_mps=round(north_mps, 8))


def combine_drift_velocity(
    wind_vector: Vector2D,
    current_vector: Vector2D,
    windage_factor: float = 0.03
) -> Vector2D:
    """
    Calculates combined surface oil drift velocity vector using the prototype equation:
    
        oil_velocity = current_velocity + windage_factor * wind_velocity

    :param wind_vector: Wind velocity vector in m/s
    :param current_vector: Ocean current velocity vector in m/s
    :param windage_factor: Windage coefficient (default: 0.03, must be >= 0)
    :return: Combined oil drift velocity Vector2D in m/s
    :raises ValueError: If windage_factor is negative or vectors are invalid
    """
    if not isinstance(wind_vector, Vector2D):
        raise ValueError("wind_vector must be an instance of Vector2D.")
    if not isinstance(current_vector, Vector2D):
        raise ValueError("current_vector must be an instance of Vector2D.")

    if windage_factor is None or isinstance(windage_factor, bool) or not isinstance(windage_factor, (int, float)):
        raise ValueError("windage_factor must be a numeric value.")

    wf = float(windage_factor)
    if wf < 0:
        raise ValueError(f"windage_factor cannot be negative, got {windage_factor}")

    east_mps = current_vector.east_mps + wf * wind_vector.east_mps
    north_mps = current_vector.north_mps + wf * wind_vector.north_mps

    return Vector2D(east_mps=round(east_mps, 8), north_mps=round(north_mps, 8))


def speed_and_bearing_from_vector(vector: Vector2D) -> Tuple[float, float]:
    """
    Converts 2D Cartesian velocity vector components (east, north) back to speed and bearing.

    :param vector: Vector2D instance
    :return: Tuple of (speed_mps, bearing_deg) where bearing is in [0, 360)
    :raises ValueError: If vector is invalid
    """
    if not isinstance(vector, Vector2D):
        raise ValueError("vector must be an instance of Vector2D.")

    east = vector.east_mps
    north = vector.north_mps

    speed = math.hypot(east, north)
    if speed == 0.0:
        return 0.0, 0.0

    bearing_rad = math.atan2(east, north)
    bearing_deg = math.degrees(bearing_rad) % 360.0

    return round(speed, 8), round(bearing_deg, 8)
