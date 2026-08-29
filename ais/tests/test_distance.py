"""Unit tests for Haversine geographic distance calculations."""

import unittest

from ais.filters import haversine_distance


class TestDistance(unittest.TestCase):
    """Test suite for haversine_distance."""

    def test_same_point_returns_zero_distance(self):
        """Distance from a point to itself must be 0.0 km."""
        lat, lon = 12.8500, 74.7200
        dist = haversine_distance(lat, lon, lat, lon)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_known_geographic_distance(self):
        """Test distance calculation against known benchmark (London to Paris ~343.5 km)."""
        # London (51.5074 N, 0.1278 W) to Paris (48.8566 N, 2.3522 E)
        london_lat, london_lon = 51.5074, -0.1278
        paris_lat, paris_lon = 48.8566, 2.3522
        dist = haversine_distance(london_lat, london_lon, paris_lat, paris_lon)
        # Expected around 343 - 344 km
        self.assertGreater(dist, 340.0)
        self.assertLess(dist, 350.0)

    def test_short_distance_accuracy(self):
        """Test short nautical distance (~0.3 km) near spill origin."""
        # 12.8500, 74.7200 to 12.8520, 74.7180
        dist = haversine_distance(12.8500, 74.7200, 12.8520, 74.7180)
        self.assertGreater(dist, 0.2)
        self.assertLess(dist, 0.5)

    def test_invalid_latitude_raises_value_error(self):
        """Latitude outside [-90, 90] should raise ValueError."""
        with self.assertRaises(ValueError):
            haversine_distance(95.0, 74.72, 12.85, 74.72)
        with self.assertRaises(ValueError):
            haversine_distance(12.85, 74.72, -91.0, 74.72)

    def test_invalid_longitude_raises_value_error(self):
        """Longitude outside [-180, 180] should raise ValueError."""
        with self.assertRaises(ValueError):
            haversine_distance(12.85, 185.0, 12.85, 74.72)
        with self.assertRaises(ValueError):
            haversine_distance(12.85, 74.72, 12.85, -181.0)


if __name__ == "__main__":
    unittest.main()

