"""Contract tests for the public AIS API required by the SIH assignment."""

import tempfile
import unittest
from pathlib import Path

from ais import (
    clean_ais_data,
    filter_vessels_by_distance,
    filter_vessels_by_time,
    generate_synthetic_ais,
    get_candidate_vessels,
    load_ais_data,
)
from ais.schemas import AISPoint


class TestPublicApiContract(unittest.TestCase):
    """Validate the assignment-required public function names and output schema."""

    def test_loader_and_cleaner_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(
                "mmsi,BaseDateTime,LAT,LON,SOG,COG\n"
                "V100,2026-08-20T14:30:00Z,12.8500,74.7200,12.5,180.0\n",
                encoding="utf-8",
            )

            raw = load_ais_data(csv_path)
            self.assertEqual(raw[0]["vessel_id"], "V100")
            self.assertEqual(raw[0]["speed_knots"], "12.5")

            cleaned = clean_ais_data(raw)
            self.assertEqual(len(cleaned), 1)
            self.assertIsInstance(cleaned[0], AISPoint)
            self.assertEqual(cleaned[0].vessel_id, "V100")

    def test_public_filtering_api_and_candidate_output(self):
        points = [
            AISPoint(
                vessel_id="V001",
                timestamp="2026-08-20T14:30:00Z" if False else __import__("datetime").datetime(2026, 8, 20, 14, 30, tzinfo=__import__("datetime").timezone.utc),
                latitude=12.8520,
                longitude=74.7180,
                speed_knots=12.0,
                heading_deg=140.0,
            ),
            AISPoint(
                vessel_id="V002",
                timestamp=__import__("datetime").datetime(2026, 8, 20, 18, 0, tzinfo=__import__("datetime").timezone.utc),
                latitude=12.9800,
                longitude=74.7800,
                speed_knots=15.0,
                heading_deg=200.0,
            ),
        ]

        time_filtered = filter_vessels_by_time(points, "2026-08-20T14:00:00Z", "2026-08-20T15:00:00Z")
        self.assertEqual(len(time_filtered), 1)

        distance_filtered = filter_vessels_by_distance(points, 12.85, 74.72, 10.0)
        self.assertEqual(len(distance_filtered), 1)

        result = get_candidate_vessels(
            ais_data=points,
            origin_lat=12.85,
            origin_lon=74.72,
            release_start="2026-08-20T14:00:00Z",
            release_end="2026-08-20T15:00:00Z",
            search_radius_km=10.0,
        )

        self.assertIn("spill_id", result)
        self.assertIn("origin", result)
        self.assertIn("release_window", result)
        self.assertIn("search_radius_km", result)
        self.assertIn("candidate_vessels", result)
        self.assertEqual(result["candidate_vessels"][0]["vessel_id"], "V001")

    def test_synthetic_generator_accepts_assignment_names(self):
        points = generate_synthetic_ais(
            origin_lat=12.85,
            origin_lon=74.72,
            start_time="2026-08-20T14:00:00Z",
            end_time="2026-08-20T15:00:00Z",
            number_of_vessels=5,
            time_interval=5,
            search_area=10.0,
            vessel_speed=12.0,
            random_seed=42,
        )
        self.assertGreater(len(points), 0)
        self.assertTrue(all(hasattr(p, "vessel_id") for p in points))


if __name__ == "__main__":
    unittest.main()
