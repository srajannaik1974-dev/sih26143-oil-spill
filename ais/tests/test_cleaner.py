"""Unit tests for AIS cleaner module (coordinate validation, deduplication, UTC normalization)."""

import unittest
from datetime import datetime, timezone

from ais.cleaner import clean_ais_records, parse_utc_timestamp


class TestCleaner(unittest.TestCase):
    """Test suite for cleaning and validation operations."""

    def test_parse_utc_timestamp_iso_formats(self):
        """Test timestamp parser on ISO formats."""
        dt1 = parse_utc_timestamp("2026-08-20T14:30:00Z")
        self.assertEqual(dt1.tzinfo, timezone.utc)
        self.assertEqual(dt1.hour, 14)
        self.assertEqual(dt1.minute, 30)

        dt2 = parse_utc_timestamp("2026-08-20 14:30:00")
        self.assertEqual(dt2.tzinfo, timezone.utc)
        self.assertEqual(dt2.hour, 14)

    def test_parse_utc_timestamp_invalid_raises_error(self):
        """Test timestamp parser raises ValueError on bad format."""
        with self.assertRaises(ValueError):
            parse_utc_timestamp("invalid_date")
        with self.assertRaises(ValueError):
            parse_utc_timestamp("")

    def test_remove_invalid_coordinates(self):
        """Test that out-of-bounds coordinates are filtered out."""
        raw = [
            {"vessel_id": "V1", "timestamp": "2026-08-20T14:00:00Z", "latitude": 12.85, "longitude": 74.72},
            {"vessel_id": "V2", "timestamp": "2026-08-20T14:00:00Z", "latitude": 95.0, "longitude": 74.72},  # Bad Lat
            {"vessel_id": "V3", "timestamp": "2026-08-20T14:00:00Z", "latitude": 12.85, "longitude": -190.0}, # Bad Lon
            {"vessel_id": "V4", "timestamp": "2026-08-20T14:00:00Z", "latitude": "invalid", "longitude": 74.72}, # Non-numeric
        ]

        cleaned = clean_ais_records(raw)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].vessel_id, "V1")

    def test_remove_duplicate_records(self):
        """Test that duplicate records for same vessel and timestamp are deduplicated."""
        raw = [
            {"vessel_id": "V1", "timestamp": "2026-08-20T14:00:00Z", "latitude": 12.85, "longitude": 74.72, "speed_knots": 10.0},
            {"vessel_id": "V1", "timestamp": "2026-08-20T14:00:00Z", "latitude": 12.85, "longitude": 74.72, "speed_knots": 10.0}, # Duplicate
            {"vessel_id": "V1", "timestamp": "2026-08-20T14:10:00Z", "latitude": 12.86, "longitude": 74.73, "speed_knots": 10.5},
        ]

        cleaned = clean_ais_records(raw)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0].timestamp_iso, "2026-08-20T14:00:00Z")
        self.assertEqual(cleaned[1].timestamp_iso, "2026-08-20T14:10:00Z")

    def test_missing_speed_and_heading_handling(self):
        """Test that missing speed/heading are safely set to None without inventing values."""
        raw = [
            {
                "vessel_id": "V1",
                "timestamp": "2026-08-20T14:00:00Z",
                "latitude": 12.85,
                "longitude": 74.72,
                "speed_knots": "",
                "heading_deg": None,
            }
        ]

        cleaned = clean_ais_records(raw)
        self.assertEqual(len(cleaned), 1)
        self.assertIsNone(cleaned[0].speed_knots)
        self.assertIsNone(cleaned[0].heading_deg)

    def test_records_sorted_by_vessel_and_timestamp(self):
        """Test that output records are sorted by vessel_id and timestamp."""
        raw = [
            {"vessel_id": "V_B", "timestamp": "2026-08-20T14:30:00Z", "latitude": 12.85, "longitude": 74.72},
            {"vessel_id": "V_A", "timestamp": "2026-08-20T14:40:00Z", "latitude": 12.85, "longitude": 74.72},
            {"vessel_id": "V_A", "timestamp": "2026-08-20T14:20:00Z", "latitude": 12.85, "longitude": 74.72},
        ]

        cleaned = clean_ais_records(raw)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned[0].vessel_id, "V_A")
        self.assertEqual(cleaned[0].timestamp_iso, "2026-08-20T14:20:00Z")
        self.assertEqual(cleaned[1].vessel_id, "V_A")
        self.assertEqual(cleaned[1].timestamp_iso, "2026-08-20T14:40:00Z")
        self.assertEqual(cleaned[2].vessel_id, "V_B")


if __name__ == "__main__":
    unittest.main()

