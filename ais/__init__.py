"""Public API for the AIS / vessel-tracking module."""

from .cleaner import clean_ais_data, clean_ais_records, parse_utc_timestamp
from .filters import (
    calculate_distance_km,
    filter_by_distance,
    filter_by_time_window,
    filter_vessels_by_distance,
    filter_vessels_by_time,
    find_candidate_vessels,
    get_candidate_vessels,
    haversine_distance,
)
from .loader import load_ais_csv, load_ais_data, load_ais_file, load_ais_json
from .schemas import AISPoint, CandidateOutput, CandidateVessel, SpillQuery, VesselTrajectory
from .synthetic_generator import generate_synthetic_ais, save_synthetic_ais_csv
from .trajectory import build_trajectories, get_vessel_trajectory

__all__ = [
    "AISPoint",
    "CandidateOutput",
    "CandidateVessel",
    "SpillQuery",
    "VesselTrajectory",
    "clean_ais_data",
    "clean_ais_records",
    "calculate_distance_km",
    "filter_by_distance",
    "filter_by_time_window",
    "filter_vessels_by_distance",
    "filter_vessels_by_time",
    "find_candidate_vessels",
    "get_candidate_vessels",
    "generate_synthetic_ais",
    "get_vessel_trajectory",
    "haversine_distance",
    "load_ais_csv",
    "load_ais_data",
    "load_ais_file",
    "load_ais_json",
    "parse_utc_timestamp",
    "save_synthetic_ais_csv",
    "build_trajectories",
]
