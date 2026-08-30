"""
Unit tests for geographic coordinate utilities (src/drift/geo_utils.py).
"""

import math
import pytest
from src.drift.geo_utils import (
    validate_coordinates,
    destination_point,
    distance_between_points,
    bearing_between_points,
    normalize_bearing,
    EARTH_RADIUS_METERS,
)


def test_valid_coordinates_accepted():
    """Requirement 1: Valid coordinates accepted without raising exceptions."""
    validate_coordinates(0.0, 0.0)
    validate_coordinates(45.5, 90.2)
    validate_coordinates(-90.0, -180.0)
    validate_coordinates(90.0, 180.0)


def test_invalid_latitude_rejected():
    """Requirement 2: Invalid latitude rejected."""
    with pytest.raises(ValueError):
        validate_coordinates(90.1, 0.0)
    with pytest.raises(ValueError):
        validate_coordinates(-90.1, 0.0)
    with pytest.raises(ValueError):
        validate_coordinates(150.0, 45.0)


def test_invalid_longitude_rejected():
    """Requirement 3: Invalid longitude rejected."""
    with pytest.raises(ValueError):
        validate_coordinates(0.0, 180.1)
    with pytest.raises(ValueError):
        validate_coordinates(0.0, -180.1)
    with pytest.raises(ValueError):
        validate_coordinates(10.0, 200.0)


def test_zero_distance_returns_approximately_zero():
    """Requirement 4: Zero distance returns approximately zero."""
    dist = distance_between_points(15.0, 75.0, 15.0, 75.0)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_moving_east_produces_approximately_90_degree_bearing():
    """Requirement 5: Moving east produces approximately 90┬░ bearing."""
    bearing = bearing_between_points(0.0, 0.0, 0.0, 1.0)
    assert bearing == pytest.approx(90.0, abs=1e-3)


def test_moving_north_produces_approximately_0_degree_bearing():
    """Requirement 6: Moving north produces approximately 0┬░ bearing."""
    bearing = bearing_between_points(0.0, 0.0, 1.0, 0.0)
    assert bearing == pytest.approx(0.0, abs=1e-3)


def test_destination_point_zero_distance_returns_starting_point():
    """Requirement 7: destination_point with zero distance returns starting point."""
    lat, lon = destination_point(20.0, 80.0, 45.0, 0.0)
    assert lat == pytest.approx(20.0, abs=1e-6)
    assert lon == pytest.approx(80.0, abs=1e-6)


def test_destination_point_moves_east_correctly():
    """Requirement 8: destination_point correctly moves east for a known small distance."""
    start_lat, start_lon = 0.0, 0.0
    distance_m = 10000.0  # 10 km
    bearing_deg = 90.0   # East

    dest_lat, dest_lon = destination_point(start_lat, start_lon, bearing_deg, distance_m)

    # Latitude should remain ~0
    assert dest_lat == pytest.approx(0.0, abs=1e-4)

    # Expected longitude change: dist / (R * cos(lat) * pi/180)
    expected_lon_deg = math.degrees(distance_m / EARTH_RADIUS_METERS)
    assert dest_lon == pytest.approx(expected_lon_deg, abs=1e-4)

    # Verify calculated distance matches back
    calc_dist = distance_between_points(start_lat, start_lon, dest_lat, dest_lon)
    assert calc_dist == pytest.approx(distance_m, rel=1e-4)


def test_longitude_wrapping():
    """Requirement 9: Longitude wrapping across +180/-180 meridian works correctly."""
    # Start at 179.9┬░ E on the equator, move 25 km East (approx 0.225 degrees)
    start_lat, start_lon = 0.0, 179.9
    distance_m = 25000.0
    dest_lat, dest_lon = destination_point(start_lat, start_lon, 90.0, distance_m)

    # Should wrap into negative longitudes (around -179.875)
    assert -180.0 <= dest_lon <= 180.0
    assert dest_lon < 0.0


def test_bearing_normalization():
    """Test bearing normalization for angles outside [0, 360)."""
    assert normalize_bearing(450.0) == pytest.approx(90.0)
    assert normalize_bearing(-90.0) == pytest.approx(270.0)
    assert normalize_bearing(360.0) == pytest.approx(0.0)


def test_invalid_coordinates_in_geo_functions():
    """Verify that invalid coordinates are rejected across all geo functions."""
    with pytest.raises(ValueError):
        destination_point(95.0, 0.0, 0.0, 100.0)
    with pytest.raises(ValueError):
        destination_point(0.0, 0.0, 0.0, -100.0)
    with pytest.raises(ValueError):
        distance_between_points(0.0, 0.0, 200.0, 0.0)
    with pytest.raises(ValueError):
        bearing_between_points(-100.0, 0.0, 0.0, 0.0)
