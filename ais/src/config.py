"""Configuration settings for AIS Candidate Identification Module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class AISConfig:
    """Configurable parameters for spatial-temporal AIS candidate searching and ranking."""

    # Search boundaries
    search_radius_km: float = 10.0
    time_window_minutes: float = 30.0
    top_n_candidates: int = 5

    # Scoring weights (sum to 1.0)
    distance_weight: float = 0.50
    time_weight: float = 0.35
    observations_weight: float = 0.10
    ship_type_weight: float = 0.05

    # Ship type risk multipliers (heuristic supporting feature)
    ship_type_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "tanker": 100.0,
            "cargo": 80.0,
            "carrier": 80.0,
            "container": 75.0,
            "tug": 50.0,
            "passenger": 40.0,
            "fishing": 35.0,
            "pleasure": 20.0,
            "other": 50.0,
        }
    )

    # Base paths relative to module root
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "output")

    @property
    def default_spill_path(self) -> Path:
        return self.data_dir / "mock_spill.json"

    @property
    def default_ais_path(self) -> Path:
        return self.data_dir / "synthetic_ais.csv"

    @property
    def default_output_path(self) -> Path:
        return self.output_dir / "ais_result.json"


# Default module-level configuration instance
DEFAULT_CONFIG = AISConfig()

