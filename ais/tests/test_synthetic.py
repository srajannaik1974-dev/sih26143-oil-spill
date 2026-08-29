"""Unit tests for synthetic AIS generator and demonstration scenarios."""

import tempfile
import unittest
from pathlib import Path

from ais.cleaner import clean_ais_records
from ais.filters import find_candidate_vessels
from ais.loader import load_ais_csv
from ais.synthetic_generator import generate_synthetic_ais, save_synthetic_ais_csv


class TestSynthetic(unittest.TestCase):
    """Test suite for synthetic dataset generator."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_synthetic_ais_structure(self):
        """Test generation of synthetic AIS observations."""
        points = generate_synthetic_ais(
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
        )

        self.assertGreater(len(points), 10)
        vessel_ids = {p.vessel_id for p in points}
        self.assertIn("VESSEL_001", vessel_ids)
        self.assertIn("VESSEL_002", vessel_ids)
        self.assertIn("VESSEL_003", vessel_ids)
        self.assertIn("VESSEL_004", vessel_ids)
        self.assertIn("VESSEL_005", vessel_ids)

    def test_save_and_load_synthetic_ais_csv(self):
        """Test saving synthetic AIS to CSV and reloading through loader & cleaner."""
        csv_file = self.temp_path / "synthetic.csv"
        points = generate_synthetic_ais()
        save_synthetic_ais_csv(csv_file, points)

        # Check file content has DEMO/SYNTHETIC label
        content = csv_file.read_text(encoding="utf-8")
        self.assertIn("DEMO/SYNTHETIC", content)

        # Load through standard loader
        raw_records = load_ais_csv(csv_file)
        self.assertEqual(len(raw_records), len(points))

        # Clean through standard cleaner
        cleaned = clean_ais_records(raw_records)
        self.assertEqual(len(cleaned), len(points))

    def test_synthetic_candidate_filtering_scenario(self):
        """Test that synthetic generator yields the expected candidate filtering outcome."""
        points = generate_synthetic_ais(
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
        )

        result = find_candidate_vessels(
            trajectories=points,
            origin_lat=12.8500,
            origin_lon=74.7200,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
            spill_id="DEMO_SPILL_001",
        )

        candidate_ids = [c.vessel_id for c in result.candidates]
        # In-window & inside 10 km: VESSEL_001 (~0.3 km), VESSEL_002 (~3.0 km), VESSEL_005 (~6.5 km)
        self.assertIn("VESSEL_001", candidate_ids)
        self.assertIn("VESSEL_002", candidate_ids)
        self.assertIn("VESSEL_005", candidate_ids)

        # Outside radius: VESSEL_003 (15.5 km)
        self.assertNotIn("VESSEL_003", candidate_ids)

        # Outside time window: VESSEL_004 (4 hours late)
        self.assertNotIn("VESSEL_004", candidate_ids)

        # Top candidate is VESSEL_001 with distance ~0.31 km
        self.assertEqual(result.candidates[0].vessel_id, "VESSEL_001")
        self.assertLess(result.candidates[0].closest_distance_km, 1.0)


if __name__ == "__main__":
    unittest.main()

