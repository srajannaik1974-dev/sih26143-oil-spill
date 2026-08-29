"""Canonical AIS candidate output contract tests."""

import unittest

from ais.filters import get_candidate_vessels
from ais.synthetic_generator import generate_synthetic_ais


class TestPipeline(unittest.TestCase):
    """End-to-end AIS candidate extraction using the canonical package."""

    def test_candidate_output_contract(self):
        """Synthetic AIS should return the expected canonical candidate evidence structure."""
        points = generate_synthetic_ais(
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
        )

        result = get_candidate_vessels(
            ais_data=points,
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
            spill_id="SPILL_2026_01",
        )

        self.assertIn("spill_id", result)
        self.assertIn("origin", result)
        self.assertIn("release_window", result)
        self.assertIn("search_radius_km", result)
        self.assertIn("candidate_vessels", result)

        self.assertEqual(result["spill_id"], "SPILL_2026_01")
        self.assertEqual(result["origin"]["latitude"], 12.85)
        self.assertEqual(result["origin"]["longitude"], 74.72)
        self.assertEqual(result["release_window"]["start"], "2026-08-20T14:00:00Z")
        self.assertEqual(result["release_window"]["end"], "2026-08-20T15:00:00Z")
        self.assertEqual(result["search_radius_km"], 10.0)

        candidates = result["candidate_vessels"]
        self.assertTrue(len(candidates) >= 1)

        first = candidates[0]
        self.assertIn("vessel_id", first)
        self.assertIn("closest_distance_km", first)
        self.assertIn("closest_timestamp", first)
        self.assertIn("time_difference_minutes", first)
        self.assertIn("latitude", first)
        self.assertIn("longitude", first)
        self.assertIn("speed_knots", first)
        self.assertIn("heading_deg", first)

        self.assertNotIn("candidate_score", first)
        self.assertNotIn("vessel_name", first)
        self.assertNotIn("vessel_type", first)

    def test_no_candidates_returns_empty_list_without_crashing(self):
        """Strict radius should return no candidate vessels without any attribution score."""
        points = generate_synthetic_ais(
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
        )

        result = get_candidate_vessels(
            ais_data=points,
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=0.001,
            spill_id="SPILL_EMPTY",
        )

        self.assertEqual(result["candidate_vessels"], [])
        for candidate in result["candidate_vessels"]:
            self.assertNotIn("candidate_score", candidate)


if __name__ == "__main__":
    unittest.main()
