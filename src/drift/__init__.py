"""
Oil-spill drift module containing data models and geographic coordinate utilities.
"""

from .models import (
    SpillObservation,
    EnvironmentalObservation,
    TrajectoryPoint,
    BacktrackingResult,
)
from .geo_utils import (
    destination_point,
    distance_between_points,
    bearing_between_points,
    validate_coordinates,
)

__all__ = [
    "SpillObservation",
    "EnvironmentalObservation",
    "TrajectoryPoint",
    "BacktrackingResult",
    "destination_point",
    "distance_between_points",
    "bearing_between_points",
    "validate_coordinates",
]
