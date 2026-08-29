"""Compatibility facade for the legacy ais.src package.

The authoritative implementation for SIH 26143 AIS work lives under the canonical
ais package. This module remains only to avoid breaking older imports while the
project converges on the single-source AIS design.
"""

from ais.cleaner import clean_ais_data, clean_ais_records, parse_utc_timestamp
from ais.filters import (
    calculate_distance_km,
    filter_by_distance,
    filter_by_time_window,
    filter_vessels_by_distance,
    filter_vessels_by_time,
    find_candidate_vessels,
    get_candidate_vessels,
    haversine_distance,
)
from ais.loader import load_ais_csv, load_ais_data, load_ais_file, load_ais_json
from ais.schemas import AISPoint, CandidateOutput, CandidateVessel, SpillQuery, VesselTrajectory
from ais.synthetic_generator import generate_synthetic_ais, save_synthetic_ais_csv
from ais.trajectory import build_trajectories, get_vessel_trajectory

__version__ = "1.0.0"

__all__ = [
    "AISPoint",
    "CandidateOutput",
    "CandidateVessel",
    "SpillQuery",
    "VesselTrajectory",
    "build_trajectories",
    "calculate_distance_km",
    "clean_ais_data",
    "clean_ais_records",
    "filter_by_distance",
    "filter_by_time_window",
    "filter_vessels_by_distance",
    "filter_vessels_by_time",
    "find_candidate_vessels",
    "generate_synthetic_ais",
    "get_candidate_vessels",
    "get_vessel_trajectory",
    "haversine_distance",
    "load_ais_csv",
    "load_ais_data",
    "load_ais_file",
    "load_ais_json",
    "parse_utc_timestamp",
    "save_synthetic_ais_csv",
]

