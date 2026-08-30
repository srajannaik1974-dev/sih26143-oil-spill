"""Unit tests for AIS loader module (CSV & JSON ingestion and column validation)."""

import json
import tempfile
import unittest
from pathlib import Path

from ais.loader import load_ais_csv, load_ais_file, load_ais_json


class TestLoader(unittest.TestCase):
    """Test suite for data ingestion and column mapping."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_csv_canonical_columns(self):
        """Test loading well-formed CSV with exact canonical headers."""
        csv_file = self.temp_path / "valid.csv"
        csv_file.write_text(
            "vessel_id,timestamp,latitude,longitude,speed_knots,heading_deg\n"
            "V1,2026-08-20T14:30:00Z,12.8500,74.7200,12.5,180.0\n",
            encoding="utf-8",
        )

        records = load_ais_csv(csv_file)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["vessel_id"], "V1")
        self.assertEqual(records[0]["timestamp"], "2026-08-20T14:30:00Z")
        self.assertEqual(records[0]["latitude"], "12.8500")
        self.assertEqual(records[0]["longitude"], "74.7200")
        self.assertEqual(records[0]["speed_knots"], "12.5")
        self.assertEqual(records[0]["heading_deg"], "180.0")

    def test_load_csv_alias_columns_mapping(self):
        """Test loading CSV with real-world column aliases (mmsi, sog, cog, lat, lon)."""
        csv_file = self.temp_path / "aliases.csv"
        csv_file.write_text(
            "mmsi,datetime,lat,lon,sog,cog\n"
            "987654321,2026-08-20 14:30:00,12.8500,74.7200,14.2,210.0\n",
            encoding="utf-8",
        )

        records = load_ais_csv(csv_file)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["vessel_id"], "987654321")
        self.assertEqual(records[0]["latitude"], "12.8500")
        self.assertEqual(records[0]["longitude"], "74.7200")
        self.assertEqual(records[0]["speed_knots"], "14.2")
        self.assertEqual(records[0]["heading_deg"], "210.0")

    def test_load_csv_missing_required_columns_raises_error(self):
        """Test that missing required columns raise a clear ValueError."""
        csv_file = self.temp_path / "missing_cols.csv"
        csv_file.write_text(
            "vessel_id,latitude,longitude\n"
            "V1,12.8500,74.7200\n",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as ctx:
            load_ais_csv(csv_file)
        self.assertIn("Missing required columns in AIS file", str(ctx.exception))
        self.assertIn("timestamp", str(ctx.exception))

    def test_load_json_array_format(self):
        """Test loading JSON array of AIS records."""
        json_file = self.temp_path / "records.json"
        data = [
            {
                "vessel_id": "V_JSON",
                "timestamp": "2026-08-20T14:00:00Z",
                "latitude": 12.81,
                "longitude": 74.65,
                "speed_knots": 10.0,
                "heading_deg": 90.0,
            }
        ]
        json_file.write_text(json.dumps(data), encoding="utf-8")

        records = load_ais_file(json_file)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["vessel_id"], "V_JSON")
        self.assertEqual(records[0]["speed_knots"], 10.0)

    def test_load_json_wrapped_object_format(self):
        """Test loading JSON object with 'records' key."""
        json_file = self.temp_path / "wrapped.json"
        data = {
            "records": [
                {
                    "mmsi": "V_WRAP",
                    "timestamp": "2026-08-20T14:15:00Z",
                    "latitude": 12.82,
                    "longitude": 74.66,
                }
            ]
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        records = load_ais_json(json_file)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["vessel_id"], "V_WRAP")

    def test_unsupported_file_extension_raises_error(self):
        """Test that unknown extensions raise ValueError."""
        txt_file = self.temp_path / "data.txt"
        txt_file.write_text("hello", encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_ais_file(txt_file)
        self.assertIn("Unsupported file format", str(ctx.exception))

    def test_nonexistent_file_raises_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_ais_file(self.temp_path / "does_not_exist.csv")


if __name__ == "__main__":
    unittest.main()
