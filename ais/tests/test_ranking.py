"""Canonical AIS candidate evidence and trajectory validation tests."""

import unittest
from datetime import datetime, timezone

from ais.filters import find_candidate_vessels, get_candidate_vessels
from ais.schemas import AISPoint
from ais.trajectory import build_trajectories, get_vessel_trajectory


class TestRanking(unittest.TestCase):
    """Tests for AIS trajectory grouping and candidate evidence extraction only."""

    def test_trajectory_grouping_and_lookup(self):
        """Records for the same vessel should be grouped and ordered chronologically."""
        points = [
            AISPoint(vessel_id="V1", timestamp=datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc), latitude=12.83, longitude=74.71, speed_knots=12.0, heading_deg=180.0),
            AISPoint(vessel_id="V2", timestamp=datetime(2026, 8, 20, 14, 10, 0, tzinfo=timezone.utc), latitude=12.90, longitude=74.80, speed_knots=15.0, heading_deg=200.0),
            AISPoint(vessel_id="V1", timestamp=datetime(2026, 8, 20, 14, 10, 0, tzinfo=timezone.utc), latitude=12.81, longitude=74.69, speed_knots=11.5, heading_deg=170.0),
            AISPoint(vessel_id="V1", timestamp=datetime(2026, 8, 20, 14, 20, 0, tzinfo=timezone.utc), latitude=12.82, longitude=74.70, speed_knots=11.8, heading_deg=175.0),
        ]

        trajectories = build_trajectories(points)
        self.assertIn("V1", trajectories)
        self.assertIn("V2", trajectories)

        traj = get_vessel_trajectory(trajectories, "V1")
        self.assertIsNotNone(traj)
        self.assertEqual(traj.point_count, 3)
        self.assertEqual(traj.points[0].timestamp_iso, "2026-08-20T14:10:00Z")

    def test_candidate_evidence_fields(self):
        """Candidate output should contain raw AIS evidence and no final attribution score."""
        points = [
            AISPoint(vessel_id="V001", timestamp=datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc), latitude=12.8520, longitude=74.7180, speed_knots=12.0, heading_deg=140.0),
            AISPoint(vessel_id="V002", timestamp=datetime(2026, 8, 20, 14, 40, 0, tzinfo=timezone.utc), latitude=12.9800, longitude=74.7800, speed_knots=15.0, heading_deg=200.0),
        ]

        result = get_candidate_vessels(
            ais_data=points,
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
            spill_id="SPILL_1",
        )

        self.assertEqual(result["spill_id"], "SPILL_1")
        self.assertTrue(len(result["candidate_vessels"]) >= 1)

        first = result["candidate_vessels"][0]
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

    def test_find_candidate_vessels_uses_evidence_not_attribution(self):
        """AIS candidate generation should produce evidence only and not final ranking metadata."""
        points = [
            AISPoint(vessel_id="V001", timestamp=datetime(2026, 8, 20, 14, 29, 0, tzinfo=timezone.utc), latitude=12.8520, longitude=74.7180, speed_knots=12.0, heading_deg=140.0),
            AISPoint(vessel_id="V003", timestamp=datetime(2026, 8, 20, 18, 0, 0, tzinfo=timezone.utc), latitude=12.8510, longitude=74.7210, speed_knots=11.0, heading_deg=180.0),
        ]

        result = find_candidate_vessels(
            trajectories=points,
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
            spill_id="SPILL_2",
        )

        self.assertEqual(result.spill_id, "SPILL_2")
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].vessel_id, "V001")
        self.assertNotIn("candidate_score", result.candidates[0].to_dict())


if __name__ == "__main__":
    unittest.main()
