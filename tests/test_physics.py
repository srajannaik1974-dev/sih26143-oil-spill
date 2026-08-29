"""
Unit tests for drift vector physics (src/drift/physics.py).
"""

import math
import pytest
from src.drift.physics import (
    Vector2D,
    wind_to_vector,
    current_to_vector,
    combine_drift_velocity,
    speed_and_bearing_from_vector,
)


def test_north_wind_vector():
    """Requirement 1: North wind (0°) produces positive north component, east ~ 0."""
    v = wind_to_vector(10.0, 0.0)
    assert v.north_mps == pytest.approx(10.0, abs=1e-5)
    assert v.east_mps == pytest.approx(0.0, abs=1e-5)


def test_east_wind_vector():
    """Requirement 2: East wind (90°) produces positive east component, north ~ 0."""
    v = wind_to_vector(10.0, 90.0)
    assert v.east_mps == pytest.approx(10.0, abs=1e-5)
    assert v.north_mps == pytest.approx(0.0, abs=1e-5)


def test_south_wind_vector():
    """Requirement 3: South wind (180°) produces negative north component."""
    v = wind_to_vector(10.0, 180.0)
    assert v.north_mps == pytest.approx(-10.0, abs=1e-5)
    assert v.east_mps == pytest.approx(0.0, abs=1e-5)


def test_west_wind_vector():
    """Requirement 4: West wind (270°) produces negative east component."""
    v = wind_to_vector(10.0, 270.0)
    assert v.east_mps == pytest.approx(-10.0, abs=1e-5)
    assert v.north_mps == pytest.approx(0.0, abs=1e-5)


def test_zero_speed_returns_zero_vector():
    """Requirement 5: Zero speed produces zero vector."""
    v_wind = wind_to_vector(0.0, 45.0)
    assert v_wind.east_mps == pytest.approx(0.0)
    assert v_wind.north_mps == pytest.approx(0.0)

    v_curr = current_to_vector(0.0, 180.0)
    assert v_curr.east_mps == pytest.approx(0.0)
    assert v_curr.north_mps == pytest.approx(0.0)


def test_direction_normalization():
    """Requirement 6: Direction normalization works for angles outside [0, 360)."""
    v1 = wind_to_vector(10.0, 450.0)  # 450° = 90° (East)
    assert v1.east_mps == pytest.approx(10.0, abs=1e-5)
    assert v1.north_mps == pytest.approx(0.0, abs=1e-5)

    v2 = wind_to_vector(10.0, -90.0)  # -90° = 270° (West)
    assert v2.east_mps == pytest.approx(-10.0, abs=1e-5)
    assert v2.north_mps == pytest.approx(0.0, abs=1e-5)


def test_negative_speed_rejected():
    """Requirement 7: Negative speed is rejected with ValueError."""
    with pytest.raises(ValueError):
        wind_to_vector(-5.0, 90.0)
    with pytest.raises(ValueError):
        current_to_vector(-1.0, 0.0)


def test_current_vector_convention():
    """Requirement 8: Current vector uses the same convention."""
    v_curr = current_to_vector(2.0, 90.0)
    assert v_curr.east_mps == pytest.approx(2.0, abs=1e-5)
    assert v_curr.north_mps == pytest.approx(0.0, abs=1e-5)


def test_pure_current_plus_zero_wind():
    """Requirement 9: Pure current + zero wind gives current velocity."""
    v_wind = wind_to_vector(0.0, 90.0)
    v_curr = current_to_vector(0.5, 0.0)  # 0.5 m/s North

    v_oil = combine_drift_velocity(v_wind, v_curr)
    assert v_oil.north_mps == pytest.approx(0.5, abs=1e-5)
    assert v_oil.east_mps == pytest.approx(0.0, abs=1e-5)


def test_zero_current_plus_wind():
    """Requirement 10: Zero current + wind gives windage-scaled velocity."""
    v_wind = wind_to_vector(10.0, 90.0)  # 10 m/s East
    v_curr = current_to_vector(0.0, 0.0)

    # 10 m/s * 0.03 = 0.3 m/s East
    v_oil = combine_drift_velocity(v_wind, v_curr, windage_factor=0.03)
    assert v_oil.east_mps == pytest.approx(0.3, abs=1e-5)
    assert v_oil.north_mps == pytest.approx(0.0, abs=1e-5)


def test_default_windage_factor():
    """Requirement 11: Default windage factor is 0.03."""
    v_wind = wind_to_vector(10.0, 0.0)
    v_curr = current_to_vector(0.0, 0.0)

    v_oil = combine_drift_velocity(v_wind, v_curr)  # default 0.03
    assert v_oil.north_mps == pytest.approx(0.3, abs=1e-5)


def test_custom_windage_factor():
    """Requirement 12: Custom windage factor works."""
    v_wind = wind_to_vector(10.0, 0.0)
    v_curr = current_to_vector(0.0, 0.0)

    v_oil = combine_drift_velocity(v_wind, v_curr, windage_factor=0.05)
    assert v_oil.north_mps == pytest.approx(0.5, abs=1e-5)


def test_negative_windage_factor_rejected():
    """Requirement 13: Negative windage factor is rejected."""
    v_wind = wind_to_vector(10.0, 0.0)
    v_curr = current_to_vector(0.5, 0.0)

    with pytest.raises(ValueError):
        combine_drift_velocity(v_wind, v_curr, windage_factor=-0.01)


def test_combined_drift_velocity_vector_addition():
    """Requirement 14: Combined drift velocity is vector addition (V_current + alpha * V_wind)."""
    # Wind: 10 m/s @ 90° (East) -> wind_vector = (10, 0)
    # Current: 0.5 m/s @ 0° (North) -> current_vector = (0, 0.5)
    # Windage: 0.03 -> alpha * wind_vector = (0.3, 0)
    # Combined: (0.3, 0.5)
    v_wind = wind_to_vector(10.0, 90.0)
    v_curr = current_to_vector(0.5, 0.0)

    v_oil = combine_drift_velocity(v_wind, v_curr, windage_factor=0.03)

    assert v_oil.east_mps == pytest.approx(0.3, abs=1e-5)
    assert v_oil.north_mps == pytest.approx(0.5, abs=1e-5)


def test_speed_and_bearing_from_vector_helper():
    """Requirement 15: Vector to speed/bearing helper works."""
    # Vector (0.3 East, 0.4 North) -> speed = sqrt(0.09 + 0.16) = 0.5 m/s
    v = Vector2D(east_mps=0.3, north_mps=0.4)
    speed, bearing = speed_and_bearing_from_vector(v)

    assert speed == pytest.approx(0.5, abs=1e-5)
    # bearing = atan2(0.3, 0.4) ~ 36.87°
    expected_bearing = math.degrees(math.atan2(0.3, 0.4))
    assert bearing == pytest.approx(expected_bearing, abs=1e-4)

    # Zero vector handling
    v_zero = Vector2D(east_mps=0.0, north_mps=0.0)
    sp_z, br_z = speed_and_bearing_from_vector(v_zero)
    assert sp_z == 0.0
    assert br_z == 0.0
