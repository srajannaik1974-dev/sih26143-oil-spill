"""Unit tests for trajectory reconstruction and single vessel sequence lookup."""

import unittest
from datetime import datetime, timezone

from ais.schemas import AISPoint
from ais.trajectory import build_trajectories, get_vessel_trajectory


class TestTrajectory(unittest.TestCase):
    """Test suite for trajectory grouping and ordering."""

    def setUp(self):
        t1 = datetime(2026, 8, 20, 14, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 8, 20, 14, 20, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)

        # Intentionally provide unsorted points
        self.points = [
            AISPoint(vessel_id="V1", timestamp=t3, latitude=12.83, longitude=74.71, speed_knots=12.0),
            AISPoint(vessel_id="V2", timestamp=t1, latitude=12.90, longitude=74.80, speed_knots=15.0),
            AISPoint(vessel_id="V1", timestamp=t1, latitude=12.81, longitude=74.69, speed_knots=11.5),
            AISPoint(vessel_id="V1", timestamp=t2, latitude=12.82, longitude=74.70, speed_knots=11.8),
        ]

    def test_build_trajectories_grouping_and_sorting(self):
        """Test that points are grouped by vessel_id and sorted chronologically."""
        trajectories = build_trajectories(self.points)

        self.assertIn("V1", trajectories)
        self.assertIn("V2", trajectories)

        t_v1 = trajectories["V1"]
        self.assertEqual(t_v1.point_count, 3)
        self.assertEqual(t_v1.points[0].timestamp_iso, "2026-08-20T14:10:00Z")
        self.assertEqual(t_v1.points[1].timestamp_iso, "2026-08-20T14:20:00Z")
        self.assertEqual(t_v1.points[2].timestamp_iso, "2026-08-20T14:30:00Z")

        t_v2 = trajectories["V2"]
        self.assertEqual(t_v2.point_count, 1)

    def test_get_vessel_trajectory_lookup(self):
        """Test retrieving a specific vessel trajectory by ID."""
        trajectories = build_trajectories(self.points)

        traj = get_vessel_trajectory(trajectories, "V1")
        self.assertIsNotNone(traj)
        self.assertEqual(traj.vessel_id, "V1")
        self.assertEqual(traj.point_count, 3)

        missing = get_vessel_trajectory(trajectories, "NON_EXISTENT")
        self.assertIsNone(missing)


if __name__ == "__main__":
    unittest.main()

