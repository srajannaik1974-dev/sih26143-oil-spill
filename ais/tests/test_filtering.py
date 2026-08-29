"""Unit tests for spatio-temporal filtering logic (Tests 1, 2, 3, 5, 6)."""

import unittest
from datetime import datetime, timezone

from ais.src.config import AISConfig
from ais.src.filtering import apply_filters, filter_by_distance, filter_by_time
from ais.src.loader import AISRecord, SpillIncident, parse_utc_timestamp


class TestFiltering(unittest.TestCase):
    """Test cases for spatial and temporal AIS filtering."""

    def setUp(self):
        self.spill_time = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)
        self.spill = SpillIncident(
            latitude=12.8500,
            longitude=74.7200,
            timestamp=self.spill_time,
        )
        self.config = AISConfig(
            search_radius_km=10.0,
            time_window_minutes=30.0,
        )

    def test_1_nearby_vessel_retained(self):
        """Test 1: A vessel within the radius and time window should be returned."""
        # 0.3 km away, 1 min time difference
        rec = AISRecord(
            mmsi="123456789",
            timestamp=datetime(2026, 8, 20, 14, 29, 0, tzinfo=timezone.utc),
            latitude=12.8520,
            longitude=74.7180,
            ship_name="TEST OCEAN",
        )
        filtered = apply_filters([rec], self.spill, self.config)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].mmsi, "123456789")
        self.assertIsNotNone(filtered[0].distance_km)
        self.assertLess(filtered[0].distance_km, 1.0)
        self.assertEqual(filtered[0].time_difference_minutes, 1.0)

    def test_2_outside_radius_excluded(self):
        """Test 2: A vessel 15 km away should NOT be returned when search radius is 10 km."""
        # ~15.8 km away, exact time
        far_rec = AISRecord(
            mmsi="111222333",
            timestamp=self.spill_time,
            latitude=12.9800,
            longitude=74.7800,
            ship_name="TEST FAR",
        )
        filtered = apply_filters([far_rec], self.spill, self.config)
        self.assertEqual(len(filtered), 0)

    def test_3_outside_time_window_excluded(self):
        """Test 3: A vessel near the spill but recorded hours away should NOT be returned."""
        # 0.15 km away, but 4 hours later (18:30 UTC instead of 14:30 UTC)
        late_rec = AISRecord(
            mmsi="444333222",
            timestamp=datetime(2026, 8, 20, 18, 30, 0, tzinfo=timezone.utc),
            latitude=12.8510,
            longitude=74.7210,
            ship_name="TEST LATE",
        )
        # Early record: 4.5 hours before spill
        early_rec = AISRecord(
            mmsi="888777666",
            timestamp=datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc),
            latitude=12.8530,
            longitude=74.7220,
            ship_name="TEST EARLY",
        )
        filtered = apply_filters([late_rec, early_rec], self.spill, self.config)
        self.assertEqual(len(filtered), 0)

    def test_5_invalid_coordinates_handled(self):
        """Test 5: Coordinates exceeding valid bounds [-90, 90] should be caught/rejected."""
        with self.assertRaises(ValueError):
            filter_by_distance(
                [
                    AISRecord(
                        mmsi="999",
                        timestamp=self.spill_time,
                        latitude=95.0,  # Invalid
                        longitude=74.72,
                    )
                ],
                self.spill.latitude,
                self.spill.longitude,
                radius_km=10.0,
            )

    def test_6_invalid_timestamp_rejected(self):
        """Test 6: Invalid timestamp strings should be rejected during parsing."""
        with self.assertRaises(ValueError):
            parse_utc_timestamp("not_a_valid_date_or_time")


if __name__ == "__main__":
    unittest.main()

