"""Unit tests for canonical AIS spatio-temporal filtering logic."""

import unittest
from datetime import datetime, timezone

from ais.cleaner import parse_utc_timestamp
from ais.filters import filter_by_distance, filter_by_time_window, haversine_distance
from ais.schemas import AISPoint


class TestFiltering(unittest.TestCase):
    """Test cases for spatial and temporal AIS filtering."""

    def test_1_nearby_vessel_retained(self):
        """A vessel within the radius and time window should be returned."""
        spill_time = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)
        pts = [
            AISPoint(
                vessel_id="123456789",
                timestamp=datetime(2026, 8, 20, 14, 29, 0, tzinfo=timezone.utc),
                latitude=12.8520,
                longitude=74.7180,
                speed_knots=12.0,
                heading_deg=140.0,
            )
        ]

        filtered = filter_by_time_window(pts, spill_time, spill_time)
        self.assertEqual(len(filtered), 0)

        filtered = filter_by_time_window(
            pts,
            "2026-08-20T14:00:00Z",
            "2026-08-20T15:00:00Z",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].vessel_id, "123456789")

        distance_matches = filter_by_distance(pts, 12.8500, 74.7200, 10.0)
        self.assertEqual(len(distance_matches), 1)
        self.assertLess(distance_matches[0][1], 1.0)

    def test_2_outside_radius_excluded(self):
        """A vessel 15 km away should NOT be returned when search radius is 10 km."""
        far_rec = AISPoint(
            vessel_id="111222333",
            timestamp=datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc),
            latitude=12.9800,
            longitude=74.7800,
            speed_knots=12.0,
            heading_deg=180.0,
        )
        filtered = filter_by_distance([far_rec], 12.8500, 74.7200, 10.0)
        self.assertEqual(len(filtered), 0)

    def test_3_outside_time_window_excluded(self):
        """A vessel near the spill but recorded outside the time window should not be returned."""
        late_rec = AISPoint(
            vessel_id="444333222",
            timestamp=datetime(2026, 8, 20, 18, 30, 0, tzinfo=timezone.utc),
            latitude=12.8510,
            longitude=74.7210,
            speed_knots=11.0,
            heading_deg=180.0,
        )
        early_rec = AISPoint(
            vessel_id="888777666",
            timestamp=datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc),
            latitude=12.8530,
            longitude=74.7220,
            speed_knots=11.0,
            heading_deg=180.0,
        )

        late_matches = filter_by_time_window([late_rec], "2026-08-20T14:00:00Z", "2026-08-20T15:00:00Z")
        early_matches = filter_by_time_window([early_rec], "2026-08-20T14:00:00Z", "2026-08-20T15:00:00Z")
        self.assertEqual(len(late_matches), 0)
        self.assertEqual(len(early_matches), 0)

    def test_5_invalid_coordinates_handled(self):
        """Coordinates exceeding valid bounds should be rejected."""
        with self.assertRaises(ValueError):
            haversine_distance(95.0, 74.72, 12.85, 74.72)

    def test_6_invalid_timestamp_rejected(self):
        """Invalid timestamp strings should be rejected during parsing."""
        with self.assertRaises(ValueError):
            parse_utc_timestamp("not_a_valid_date_or_time")


if __name__ == "__main__":
    unittest.main()

