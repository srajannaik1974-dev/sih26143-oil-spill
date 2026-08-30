"""
ais/tests/test_drift_ais_integration.py
=======================================
Integration tests for verifying Drift/Backtracking to AIS candidate vessel connection.
"""

import unittest
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel

from ais import load_ais_data, clean_ais_data, get_candidate_vessels


class MockDriftOutput(BaseModel):
    """Mock DriftOriginOutput matching step 3 requirements."""
    probable_latitude: float
    probable_longitude: float
    estimated_release_time: datetime


class TestDriftAisIntegration(unittest.TestCase):
    """Validate that Drift/Backtracking outputs can be cleanly consumed by the AIS module."""

    def test_drift_to_ais_integration(self):
        # 1. Setup mock drift output matching Step 3
        drift_output = MockDriftOutput(
            probable_latitude=12.8500,
            probable_longitude=74.7200,
            estimated_release_time=datetime(2026, 8, 20, 14, 30, tzinfo=timezone.utc)
        )

        # 2. Define release window and search parameters
        release_start = "2026-08-20T14:00:00Z"
        release_end = "2026-08-20T15:00:00Z"
        search_radius_km = 10.0

        # 3. Load synthetic AIS data
        csv_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_ais.csv"
        raw_data = load_ais_data(csv_path)
        cleaned_points = clean_ais_data(raw_data)

        # 4. Pass drift output directly to AIS candidate-generation function
        result = get_candidate_vessels(
            ais_data=cleaned_points,
            origin_lat=drift_output.probable_latitude,
            origin_lon=drift_output.probable_longitude,
            release_start=release_start,
            release_end=release_end,
            search_radius_km=search_radius_km,
            spill_id="spill_sih_test",
        )

        # 5. Verify structure and values
        self.assertEqual(result["spill_id"], "spill_sih_test")
        self.assertEqual(result["origin"]["latitude"], 12.8500)
        self.assertEqual(result["origin"]["longitude"], 74.7200)
        self.assertEqual(result["search_radius_km"], 10.0)

        # 6. Verify candidates are returned
        candidates = result["candidate_vessels"]
        self.assertGreater(len(candidates), 0)

        # 7. Check candidate evidence structure (no attribution score, no guilt/ranking decision)
        for cand in candidates:
            # Must contain required fields
            self.assertIn("vessel_id", cand)
            self.assertIn("closest_distance_km", cand)
            self.assertIn("closest_timestamp", cand)
            self.assertIn("time_difference_minutes", cand)
            self.assertIn("latitude", cand)
            self.assertIn("longitude", cand)
            self.assertIn("speed_knots", cand)
            self.assertIn("heading_deg", cand)

            # Ensure no scoring or ranking attributes are present in AIS output
            self.assertNotIn("candidate_score", cand)
            self.assertNotIn("score", cand)
            self.assertNotIn("rank", cand)
            self.assertNotIn("attribution", cand)

            # Verify that time/distance filtering worked
            self.assertLessEqual(cand["closest_distance_km"], search_radius_km)
            
            # Timestamp check
            cand_ts = datetime.fromisoformat(cand["closest_timestamp"].replace("Z", "+00:00"))
            self.assertTrue(
                datetime.fromisoformat("2026-08-20T14:00:00+00:00")
                <= cand_ts
                <= datetime.fromisoformat("2026-08-20T15:00:00+00:00")
            )
