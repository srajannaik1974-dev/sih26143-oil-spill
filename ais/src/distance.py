"""Geographic distance calculations using the Haversine formula."""

import math

# Earth's mean radius in kilometers (IUGG standard / WGS84 mean)
EARTH_RADIUS_KM: float = 6371.0088


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth's surface.

    Uses the Haversine formula. Coordinates are specified in decimal degrees.

    Args:
        lat1: Latitude of the first point in decimal degrees [-90.0, 90.0].
        lon1: Longitude of the first point in decimal degrees [-180.0, 180.0].
        lat2: Latitude of the second point in decimal degrees [-90.0, 90.0].
        lon2: Longitude of the second point in decimal degrees [-180.0, 180.0].

    Returns:
        Great-circle distance in kilometers (float).

    Raises:
        ValueError: If any coordinate falls outside valid geographic boundaries.
    """
    if not (-90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0):
        raise ValueError(f"Latitude must be in range [-90, 90]. Got lat1={lat1}, lat2={lat2}")
    if not (-180.0 <= lon1 <= 180.0 and -180.0 <= lon2 <= 180.0):
        raise ValueError(f"Longitude must be in range [-180, 180]. Got lon1={lon1}, lon2={lon2}")

    # Convert decimal degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    sin_dphi_2 = math.sin(delta_phi / 2.0)
    sin_dlambda_2 = math.sin(delta_lambda / 2.0)
    
    a = sin_dphi_2 * sin_dphi_2 + math.cos(phi1) * math.cos(phi2) * sin_dlambda_2 * sin_dlambda_2

    # Clamp 'a' to [0.0, 1.0] to prevent math domain error due to floating point precision
    a = min(1.0, max(0.0, a))
    
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c

