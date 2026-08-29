"""Unit tests for vessel grouping, metric aggregation, candidate ranking, and trajectory extraction (Test 4)."""

import unittest
from datetime import datetime, timezone

from ais.src.config import AISConfig
from ais.src.loader import AISRecord
from ais.src.ranking import (
    CandidateVessel,
    compute_candidate_score,
    group_by_vessel,
    rank_candidates,
)


class TestRanking(unittest.TestCase):
    """Test cases for grouping and ranking candidate vessels."""

    def setUp(self):
        self.config = AISConfig(
            search_radius_km=10.0,
            time_window_minutes=30.0,
            top_n_candidates=5,
        )

    def test_4_multiple_records_grouped_per_mmsi(self):
        """Test 4: Multiple records for same MMSI grouped into one candidate with minimum distance."""
        records = [
            AISRecord(
                mmsi="123456789",
                timestamp=datetime(2026, 8, 20, 14, 28, 0, tzinfo=timezone.utc),
                latitude=12.8525,
                longitude=74.7175,
                ship_name="TEST OCEAN",
                ship_type="Tanker",
                sog=12.2,
                cog=180.0,
                heading=181.0,
                distance_km=0.45,
                time_difference_minutes=2.0,
            ),
            AISRecord(
                mmsi="123456789",
                timestamp=datetime(2026, 8, 20, 14, 29, 0, tzinfo=timezone.utc),
                latitude=12.8520,
                longitude=74.7180,
                ship_name="TEST OCEAN",
                ship_type="Tanker",
                sog=12.4,
                cog=180.0,
                heading=182.0,
                distance_km=0.32,  # Minimum distance
                time_difference_minutes=1.0,
            ),
            AISRecord(
                mmsi="123456789",
                timestamp=datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc),
                latitude=12.8515,
                longitude=74.7185,
                ship_name="TEST OCEAN",
                ship_type="Tanker",
                sog=12.5,
                cog=180.0,
                heading=182.0,
                distance_km=0.38,
                time_difference_minutes=0.0,
            ),
        ]

        candidates = group_by_vessel(records, self.config)
        self.assertEqual(len(candidates), 1)

        vessel = candidates[0]
        self.assertEqual(vessel.mmsi, "123456789")
        self.assertEqual(vessel.vessel_name, "TEST OCEAN")
        self.assertEqual(vessel.vessel_type, "Tanker")
        self.assertEqual(vessel.observations, 3)
        self.assertEqual(vessel.minimum_distance_km, 0.32)
        self.assertEqual(vessel.latitude, 12.8520)
        self.assertEqual(vessel.longitude, 74.7180)
        self.assertGreater(vessel.candidate_score, 80.0)

        # Verify ais_records trajectory list
        self.assertEqual(len(vessel.ais_records), 3)
        dict_rep = vessel.to_dict()
        self.assertEqual(dict_rep["vessel_name"], "TEST OCEAN")
        self.assertEqual(dict_rep["vessel_type"], "Tanker")
        self.assertIn("ais_records", dict_rep)
        self.assertEqual(len(dict_rep["ais_records"]), 3)

    def test_ais_records_chronological_ordering(self):
        """Ensure candidate ais_records are sorted chronologically regardless of input order."""
        # Unsorted inputs: 14:30, 14:10, 14:20
        t1 = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 8, 20, 14, 10, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 8, 20, 14, 20, 0, tzinfo=timezone.utc)

        records = [
            AISRecord(mmsi="123", timestamp=t1, latitude=12.83, longitude=74.71, distance_km=1.0, time_difference_minutes=0.0),
            AISRecord(mmsi="123", timestamp=t2, latitude=12.81, longitude=74.65, distance_km=3.0, time_difference_minutes=20.0),
            AISRecord(mmsi="123", timestamp=t3, latitude=12.82, longitude=74.68, distance_km=2.0, time_difference_minutes=10.0),
        ]

        candidates = group_by_vessel(records, self.config)
        self.assertEqual(len(candidates), 1)
        ais_recs = candidates[0].ais_records

        self.assertEqual(len(ais_recs), 3)
        self.assertEqual(ais_recs[0]["timestamp"], "2026-08-20T14:10:00Z")
        self.assertEqual(ais_recs[1]["timestamp"], "2026-08-20T14:20:00Z")
        self.assertEqual(ais_recs[2]["timestamp"], "2026-08-20T14:30:00Z")

    def test_ais_records_fields_presence_and_heading_handling(self):
        """Verify timestamp, latitude, longitude, sog, cog, heading presence and absence handling."""
        # Record with heading
        rec_with_heading = AISRecord(
            mmsi="101",
            timestamp=datetime(2026, 8, 20, 14, 0, 0, tzinfo=timezone.utc),
            latitude=12.8100,
            longitude=74.6500,
            sog=12.4,
            cog=178.2,
            heading=179.0,
            distance_km=1.0,
            time_difference_minutes=5.0,
        )
        # Record without heading (None)
        rec_without_heading = AISRecord(
            mmsi="102",
            timestamp=datetime(2026, 8, 20, 14, 0, 0, tzinfo=timezone.utc),
            latitude=12.8100,
            longitude=74.6500,
            sog=12.4,
            cog=178.2,
            heading=None,  # No heading
            distance_km=1.0,
            time_difference_minutes=5.0,
        )

        cand1 = group_by_vessel([rec_with_heading], self.config)[0]
        self.assertIn("heading", cand1.ais_records[0])
        self.assertEqual(cand1.ais_records[0]["heading"], 179.0)

        cand2 = group_by_vessel([rec_without_heading], self.config)[0]
        self.assertNotIn("heading", cand2.ais_records[0])

    def test_candidate_score_monotonicity(self):
        """A closer, timely vessel should strictly score higher than a distant, delayed vessel."""
        # Vessel A: 0.32 km, 1 min diff, 5 obs, Tanker
        score_a = compute_candidate_score(
            distance_km=0.32,
            time_diff_min=1.0,
            observations=5,
            ship_type="Tanker",
            config=self.config,
        )

        # Vessel B: 3.10 km, 5 min diff, 3 obs, Cargo
        score_b = compute_candidate_score(
            distance_km=3.10,
            time_diff_min=5.0,
            observations=3,
            ship_type="Cargo",
            config=self.config,
        )

        # Vessel C: 7.80 km, 12 min diff, 2 obs, Fishing
        score_c = compute_candidate_score(
            distance_km=7.80,
            time_diff_min=12.0,
            observations=2,
            ship_type="Fishing",
            config=self.config,
        )

        self.assertGreater(score_a, score_b)
        self.assertGreater(score_b, score_c)
        self.assertLessEqual(score_a, 100.0)
        self.assertGreaterEqual(score_c, 0.0)

    def test_rank_candidates_ordering_and_limit(self):
        """Candidate ranking respects top_n cutoff and score order."""
        vessels = [
            CandidateVessel(
                mmsi="1",
                vessel_name="V1",
                vessel_type=None,
                latitude=0,
                longitude=0,
                closest_record_time="",
                minimum_distance_km=5.0,
                time_difference_minutes=10.0,
                observations=1,
                candidate_score=40.0,
            ),
            CandidateVessel(
                mmsi="2",
                vessel_name="V2",
                vessel_type=None,
                latitude=0,
                longitude=0,
                closest_record_time="",
                minimum_distance_km=0.5,
                time_difference_minutes=1.0,
                observations=5,
                candidate_score=95.0,
            ),
            CandidateVessel(
                mmsi="3",
                vessel_name="V3",
                vessel_type=None,
                latitude=0,
                longitude=0,
                closest_record_time="",
                minimum_distance_km=2.0,
                time_difference_minutes=5.0,
                observations=3,
                candidate_score=75.0,
            ),
        ]

        ranked = rank_candidates(vessels, top_n=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].mmsi, "2")
        self.assertEqual(ranked[1].mmsi, "3")


if __name__ == "__main__":
    unittest.main()
