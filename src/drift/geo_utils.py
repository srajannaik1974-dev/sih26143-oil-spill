"""
Geographic coordinate utilities for spherical Earth calculations,
distance determination, bearing calculation, and destination point movement.
"""

import math
from typing import Tuple

# Standard mean radius of Earth in meters (IUGG recommended value)
EARTH_RADIUS_METERS: float = 6371000.0


def validate_coordinates(latitude: float, longitude: float) -> None:
    """
    Validates latitude and longitude geographic boundaries.

    :param latitude: Latitude in degrees (-90.0 to 90.0)
    :param longitude: Longitude in degrees (-180.0 to 180.0)
    :raises ValueError: If coordinates are out of valid bounds.
    """
    if latitude is None or isinstance(latitude, bool) or not isinstance(latitude, (int, float)):
        raise ValueError("Latitude must be a numeric value.")
    if longitude is None or isinstance(longitude, bool) or not isinstance(longitude, (int, float)):
        raise ValueError("Longitude must be a numeric value.")

    lat = float(latitude)
    lon = float(longitude)

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of bounds [-90, 90]: {latitude}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of bounds [-180, 180]: {longitude}")


def normalize_bearing(bearing_deg: float) -> float:
    """
    Normalizes bearing to range [0, 360).

    :param bearing_deg: Bearing in degrees.
    :return: Normalized bearing in degrees within [0, 360).
    """
    if bearing_deg is None or isinstance(bearing_deg, bool) or not isinstance(bearing_deg, (int, float)):
        raise ValueError("Bearing must be a numeric value.")
    b = float(bearing_deg) % 360.0
    if b < 0:
        b += 360.0
    return b


def destination_point(
    latitude: float,
    longitude: float,
    bearing_deg: float,
    distance_m: float
) -> Tuple[float, float]:
    """
    Calculates destination point given starting latitude, longitude, bearing, and distance.

    Uses spherical Earth geodesy (Great Circle direct formula).

    :param latitude: Starting latitude in degrees (-90 to 90)
    :param longitude: Starting longitude in degrees (-180 to 180)
    :param bearing_deg: Bearing in degrees (0 = North, 90 = East, etc.)
    :param distance_m: Distance to travel in meters
    :return: Tuple of (destination_latitude, destination_longitude) in degrees
    :raises ValueError: If coordinates are invalid or distance is negative.
    """
    validate_coordinates(latitude, longitude)

    if distance_m is None or isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)):
        raise ValueError("Distance must be a numeric value.")

    dist = float(distance_m)
    if dist < 0:
        raise ValueError(f"Distance cannot be negative, got {distance_m}")
    if dist == 0.0:
        return float(latitude), float(longitude)

    norm_bearing = normalize_bearing(bearing_deg)

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)
    bearing_rad = math.radians(norm_bearing)
    angular_distance = dist / EARTH_RADIUS_METERS

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance) +
        math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing_rad)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2)
    )

    dest_lat = math.degrees(lat2)
    dest_lon = math.degrees(lon2)

    # Wrap longitude to [-180, 180]
    dest_lon = (dest_lon + 180.0) % 360.0 - 180.0
    # Ensure -180 wrapping boundary edge case handling
    if dest_lon == -180.0 and longitude > 0:
        dest_lon = 180.0

    # Clamp latitude to valid [-90, 90] bounds
    dest_lat = max(-90.0, min(90.0, dest_lat))

    return dest_lat, dest_lon


def distance_between_points(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculates geographic distance in meters between two latitude/longitude points
    using the Haversine formula.

    :param lat1: Latitude of point 1 in degrees
    :param lon1: Longitude of point 1 in degrees
    :param lat2: Latitude of point 2 in degrees
    :param lon2: Longitude of point 2 in degrees
    :return: Distance in meters
    :raises ValueError: If any coordinate is invalid.
    """
    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    l1, o1 = float(lat1), float(lon1)
    l2, o2 = float(lat2), float(lon2)

    if l1 == l2 and o1 == o2:
        return 0.0

    phi1 = math.radians(l1)
    phi2 = math.radians(l2)
    delta_phi = math.radians(l2 - l1)
    delta_lambda = math.radians(o2 - o1)

    a = (
        math.sin(delta_phi / 2.0) ** 2 +
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    a = max(0.0, min(1.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_METERS * c


def bearing_between_points(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculates the initial bearing in degrees from point 1 to point 2.

    Convention:
    - 0┬░ = North
    - 90┬░ = East
    - 180┬░ = South
    - 270┬░ = West

    :param lat1: Latitude of point 1 in degrees
    :param lon1: Longitude of point 1 in degrees
    :param lat2: Latitude of point 2 in degrees
    :param lon2: Longitude of point 2 in degrees
    :return: Bearing in degrees normalized to [0, 360)
    :raises ValueError: If any coordinate is invalid.
    """
    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    l1, o1 = float(lat1), float(lon1)
    l2, o2 = float(lat2), float(lon2)

    if l1 == l2 and o1 == o2:
        return 0.0

    phi1 = math.radians(l1)
    phi2 = math.radians(l2)
    delta_lambda = math.radians(o2 - o1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2) -
        math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )

    initial_bearing = math.degrees(math.atan2(y, x))
    return normalize_bearing(initial_bearing)
