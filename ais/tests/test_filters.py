"""Unit tests for spatio-temporal filters and candidate vessel generation."""

import unittest
from datetime import datetime, timezone

from ais.filters import (
    filter_by_distance,
    filter_by_time_window,
    find_candidate_vessels,
    haversine_distance,
)
from ais.schemas import AISPoint


class TestFilters(unittest.TestCase):
    """Test suite for Haversine distance, time window, and radius filtering."""

    def test_haversine_distance_known_points(self):
        """Test Haversine distance against known geographic coordinates."""
        # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) ~343.5 km
        dist = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        self.assertAlmostEqual(dist, 343.5, delta=5.0)

        # Same point
        self.assertAlmostEqual(haversine_distance(12.85, 74.72, 12.85, 74.72), 0.0, places=5)

    def test_haversine_invalid_coordinates_raises_error(self):
        """Test out-of-bounds coordinates raise ValueError."""
        with self.assertRaises(ValueError):
            haversine_distance(95.0, 74.72, 12.85, 74.72)
        with self.assertRaises(ValueError):
            haversine_distance(12.85, -190.0, 12.85, 74.72)

    def test_filter_by_time_window(self):
        """Test filtering points by time window."""
        t1 = datetime(2026, 8, 20, 13, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 8, 20, 16, 0, 0, tzinfo=timezone.utc)

        pts = [
            AISPoint(vessel_id="V1", timestamp=t1, latitude=12.85, longitude=74.72),
            AISPoint(vessel_id="V1", timestamp=t2, latitude=12.85, longitude=74.72),
            AISPoint(vessel_id="V1", timestamp=t3, latitude=12.85, longitude=74.72),
        ]

        filtered = filter_by_time_window(pts, "2026-08-20T14:00:00Z", "2026-08-20T15:00:00Z")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].timestamp, t2)

    def test_filter_by_distance(self):
        """Test filtering points by geographic radius."""
        origin_lat, origin_lon = 12.8500, 74.7200
        pts = [
            AISPoint(vessel_id="V_NEAR", timestamp=datetime.now(timezone.utc), latitude=12.8520, longitude=74.7180), # ~0.31 km
            AISPoint(vessel_id="V_FAR", timestamp=datetime.now(timezone.utc), latitude=12.9800, longitude=74.7800),   # ~15.8 km
        ]

        matches = filter_by_distance(pts, origin_lat, origin_lon, search_radius_km=10.0)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].vessel_id, "V_NEAR")
        self.assertLess(matches[0][1], 1.0)

    def test_find_candidate_vessels_end_to_end(self):
        """Test candidate identification workflow."""
        origin_lat, origin_lon = 12.8500, 74.7200
        start = "2026-08-20T14:00:00Z"
        end = "2026-08-20T15:00:00Z"

        t_in = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)
        t_out = datetime(2026, 8, 20, 18, 0, 0, tzinfo=timezone.utc)

        pts = [
            # Vessel 1: In window, very close (~0.31 km)
            AISPoint(vessel_id="V1", timestamp=t_in, latitude=12.8520, longitude=74.7180, speed_knots=12.0, heading_deg=140.0),
            # Vessel 2: In window, outside radius (~15.8 km)
            AISPoint(vessel_id="V2", timestamp=t_in, latitude=12.9800, longitude=74.7800, speed_knots=15.0, heading_deg=200.0),
            # Vessel 3: Outside window (4 hours late), near origin
            AISPoint(vessel_id="V3", timestamp=t_out, latitude=12.8510, longitude=74.7210, speed_knots=11.0, heading_deg=180.0),
        ]

        result = find_candidate_vessels(
            trajectories=pts,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            release_start=start,
            release_end=end,
            search_radius_km=10.0,
            spill_id="SPILL_2026_01",
        )

        self.assertEqual(result.spill_id, "SPILL_2026_01")
        self.assertEqual(len(result.candidates), 1)

        c = result.candidates[0]
        self.assertEqual(c.vessel_id, "V1")
        self.assertAlmostEqual(c.closest_distance_km, 0.31, delta=0.05)
        self.assertEqual(c.closest_timestamp, "2026-08-20T14:30:00Z")
        self.assertEqual(c.latitude, 12.8520)
        self.assertEqual(c.longitude, 74.7180)
        self.assertEqual(c.speed_knots, 12.0)
        self.assertEqual(c.heading_deg, 140.0)

    def test_find_candidate_vessels_no_match(self):
        """Test no-candidate case returns empty list without error."""
        pts = [
            AISPoint(vessel_id="V1", timestamp=datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc), latitude=12.8520, longitude=74.7180)
        ]
        result = find_candidate_vessels(
            trajectories=pts,
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=0.01,  # Strict radius -> 0 matches
            spill_id="SPILL_EMPTY",
        )
        self.assertEqual(result.spill_id, "SPILL_EMPTY")
        self.assertEqual(len(result.candidates), 0)


if __name__ == "__main__":
    unittest.main()

