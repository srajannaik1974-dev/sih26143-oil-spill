"""Integration tests for end-to-end pipeline execution and output schema verification (Test 7)."""

import json
import tempfile
import unittest
from pathlib import Path

from ais.src.config import AISConfig
from ais.src.pipeline import run_ais_pipeline


class TestPipeline(unittest.TestCase):
    """End-to-end integration test suite."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.output_json = self.temp_path / "ais_result.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_end_to_end(self):
        """Execute pipeline on synthetic dataset and verify output content and schema."""
        cfg = AISConfig()
        result = run_ais_pipeline(
            spill_input_path=cfg.default_spill_path,
            ais_csv_path=cfg.default_ais_path,
            output_path=self.output_json,
            config=cfg,
            verbose=False,
        )

        # Check JSON output exists on disk
        self.assertTrue(self.output_json.exists())

        # Verify root fields
        self.assertIn("spill", result)
        self.assertIn("search_parameters", result)
        self.assertIn("candidate_vessels", result)

        # Verify spill schema uses estimated_time
        self.assertEqual(result["spill"]["latitude"], 12.85)
        self.assertEqual(result["spill"]["longitude"], 74.72)
        self.assertEqual(result["spill"]["estimated_time"], "2026-08-20T14:30:00Z")

        candidates = result["candidate_vessels"]
        self.assertEqual(len(candidates), 3)

        # Candidate #1: TEST OCEAN (closest distance ~0.31 km, highest score)
        c1 = candidates[0]
        self.assertEqual(c1["mmsi"], "123456789")
        self.assertEqual(c1["vessel_name"], "TEST OCEAN")
        self.assertEqual(c1["vessel_type"], "Tanker")
        self.assertAlmostEqual(c1["minimum_distance_km"], 0.31, delta=0.05)
        self.assertEqual(c1["observations"], 5)
        self.assertGreater(c1["candidate_score"], 85.0)

        # Verify ais_records trajectory presence and contents
        self.assertIn("ais_records", c1)
        self.assertEqual(len(c1["ais_records"]), 5)
        for ping in c1["ais_records"]:
            self.assertIn("timestamp", ping)
            self.assertIn("latitude", ping)
            self.assertIn("longitude", ping)
            self.assertIn("sog", ping)
            self.assertIn("cog", ping)
            self.assertIsInstance(ping["latitude"], float)
            self.assertIsInstance(ping["longitude"], float)

        # Verify chronological order
        timestamps = [ping["timestamp"] for ping in c1["ais_records"]]
        self.assertEqual(timestamps, sorted(timestamps))

        # Candidate #2: TEST CARRIER
        c2 = candidates[1]
        self.assertEqual(c2["mmsi"], "987654321")
        self.assertEqual(c2["vessel_name"], "TEST CARRIER")
        self.assertEqual(c2["vessel_type"], "Cargo")
        self.assertEqual(c2["observations"], 3)
        self.assertEqual(len(c2["ais_records"]), 3)

        # Candidate #3: TEST MARINE
        c3 = candidates[2]
        self.assertEqual(c3["mmsi"], "555555555")
        self.assertEqual(c3["vessel_name"], "TEST MARINE")
        self.assertEqual(c3["vessel_type"], "Fishing")
        self.assertEqual(c3["observations"], 2)
        self.assertEqual(len(c3["ais_records"]), 2)

    def test_7_no_candidates_returns_empty_list_without_crashing(self):
        """Test 7: If no vessels satisfy criteria, return empty candidate list without error."""
        # Restrict radius to 0.001 km so nothing matches
        strict_cfg = AISConfig(search_radius_km=0.001, time_window_minutes=1.0)
        result = run_ais_pipeline(
            spill_input_path=strict_cfg.default_spill_path,
            ais_csv_path=strict_cfg.default_ais_path,
            output_path=self.output_json,
            config=strict_cfg,
            verbose=False,
        )

        self.assertIn("candidate_vessels", result)
        self.assertEqual(result["candidate_vessels"], [])

        # Read back from JSON file
        with open(self.output_json, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json["candidate_vessels"], [])


if __name__ == "__main__":
    unittest.main()
