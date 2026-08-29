"""
tests/test_integration_ais.py
=============================
Integration tests for Member 2 (drift origin backtracking) to Member 3 (AIS Vessel Analysis) integration.
"""

from datetime import datetime, timezone
import pytest

from ais_adapter import run_ais_analysis


def test_m2_to_m3_integration_success():
    """Verify that M2 output values are successfully parsed and candidate vessels returned."""
    probable_lat = 29.004872
    probable_lon = -88.970347
    release_time = datetime(2018, 9, 26, 8, 30, tzinfo=timezone.utc)

    # Run analysis with default settings
    result = run_ais_analysis(
        probable_latitude=probable_lat,
        probable_longitude=probable_lon,
        estimated_release_time=release_time,
        search_radius_km=20.0,
        time_window_minutes=60.0,
    )

    # 1. Verify structure of output dict
    assert "origin" in result
    assert "search_parameters" in result
    assert "candidate_vessels" in result
    assert "metadata" in result

    # 2. Check correctness of passed coordinates and timestamps
    assert result["origin"]["latitude"] == probable_lat
    assert result["origin"]["longitude"] == probable_lon
    assert result["origin"]["estimated_release_time"] == release_time.isoformat()

    assert result["search_parameters"]["radius_km"] == 20.0
    assert result["search_parameters"]["time_window_minutes"] == 60.0

    # 3. Candidate vessels list is returned and has 3 vessels matching expected cases
    candidates = result["candidate_vessels"]
    assert len(candidates) == 3

    # Rank 1 should be Vessel A (OCEAN VOYAGER)
    assert candidates[0]["mmsi"] == "477123456"
    assert candidates[0]["vessel_name"] == "OCEAN VOYAGER"
    assert candidates[0]["vessel_type"] == "Tanker"
    assert candidates[0]["observations"] == 3
    assert candidates[0]["candidate_score"] > 90.0

    # Verification: distance & time difference
    assert candidates[0]["minimum_distance_km"] < 1.0
    assert candidates[0]["time_difference_minutes"] == 0.0


def test_radius_filtering():
    """Verify that narrowing search radius excludes more distant candidates."""
    probable_lat = 29.004872
    probable_lon = -88.970347
    release_time = datetime(2018, 9, 26, 8, 30, tzinfo=timezone.utc)

    # Broad search: 3 vessels found
    broad_res = run_ais_analysis(
        probable_latitude=probable_lat,
        probable_longitude=probable_lon,
        estimated_release_time=release_time,
        search_radius_km=20.0,
        time_window_minutes=60.0,
    )
    assert len(broad_res["candidate_vessels"]) == 3

    # Narrow search: 5.0 km radius (should exclude Vessel B/SEA RACER at ~10km)
    narrow_res = run_ais_analysis(
        probable_latitude=probable_lat,
        probable_longitude=probable_lon,
        estimated_release_time=release_time,
        search_radius_km=5.0,
        time_window_minutes=60.0,
    )
    # Should find only Vessel A (OCEAN VOYAGER) and Vessel E (CRUISING STAR)
    mmsis = [c["mmsi"] for c in narrow_res["candidate_vessels"]]
    assert len(mmsis) == 2
    assert "477123456" in mmsis  # OCEAN VOYAGER
    assert "477111222" in mmsis  # CRUISING STAR
    assert "477654321" not in mmsis  # SEA RACER excluded


def test_time_window_filtering():
    """Verify that narrowing time window excludes candidates outside temporal thresholds."""
    probable_lat = 29.004872
    probable_lon = -88.970347
    release_time = datetime(2018, 9, 26, 8, 30, tzinfo=timezone.utc)

    # Broad search: 3 vessels found
    broad_res = run_ais_analysis(
        probable_latitude=probable_lat,
        probable_longitude=probable_lon,
        estimated_release_time=release_time,
        search_radius_km=20.0,
        time_window_minutes=60.0,
    )
    assert len(broad_res["candidate_vessels"]) == 3

    # Narrow time window: 10 minutes (excludes Vessel E/CRUISING STAR at 15 min time diff)
    narrow_res = run_ais_analysis(
        probable_latitude=probable_lat,
        probable_longitude=probable_lon,
        estimated_release_time=release_time,
        search_radius_km=20.0,
        time_window_minutes=10.0,
    )
    mmsis = [c["mmsi"] for c in narrow_res["candidate_vessels"]]
    # OCEAN VOYAGER (10 min diff) and SEA RACER (5 min diff) remain, CRUISING STAR (15 min diff) excluded
    assert "477111222" not in mmsis


def test_no_candidates_returned():
    """Verify that if search parameters are empty or coordinates represent empty ocean, no candidates are returned."""
    # Coords located far away in South Pacific
    far_lat = -40.0
    far_lon = -120.0
    release_time = datetime(2018, 9, 26, 8, 30, tzinfo=timezone.utc)

    result = run_ais_analysis(
        probable_latitude=far_lat,
        probable_longitude=far_lon,
        estimated_release_time=release_time,
        search_radius_km=20.0,
        time_window_minutes=60.0,
    )
    assert len(result["candidate_vessels"]) == 0
