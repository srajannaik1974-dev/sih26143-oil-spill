"""Vessel grouping, metric aggregation, and candidate scoring/ranking."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import AISConfig, DEFAULT_CONFIG
from .loader import AISRecord


@dataclass
class CandidateVessel:
    """Aggregated candidate vessel profile near the spill location."""

    mmsi: str
    vessel_name: Optional[str]
    vessel_type: Optional[str]
    latitude: float
    longitude: float
    closest_record_time: str
    minimum_distance_km: float
    time_difference_minutes: float
    observations: int
    candidate_score: float = 0.0
    imo: Optional[str] = None
    ais_records: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.ais_records is None:
            self.ais_records = []

    @property
    def name(self) -> Optional[str]:
        return self.vessel_name

    @property
    def ship_type(self) -> Optional[str]:
        return self.vessel_type

    @property
    def distance_km(self) -> float:
        return self.minimum_distance_km

    @property
    def timestamp(self) -> str:
        return self.closest_record_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate vessel to output dictionary format."""
        data: Dict[str, Any] = {
            "mmsi": self.mmsi,
            "vessel_name": self.vessel_name,
            "vessel_type": self.vessel_type,
        }
        if self.imo is not None:
            data["imo"] = self.imo

        data.update(
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "closest_record_time": self.closest_record_time,
                "minimum_distance_km": self.minimum_distance_km,
                "time_difference_minutes": self.time_difference_minutes,
                "observations": self.observations,
                "candidate_score": self.candidate_score,
                "ais_records": self.ais_records,
            }
        )
        return data


def compute_candidate_score(
    distance_km: float,
    time_diff_min: float,
    observations: int,
    ship_type: Optional[str],
    config: AISConfig,
) -> float:
    """Calculate a transparent, heuristic candidate proximity score between 0.0 and 100.0.

    The score reflects spatial proximity, temporal alignment, observation confidence,
    and ship type risk weighting.

    NOTE: This score is a heuristic candidate ranking score. It is NOT a mathematical
    probability and does NOT imply guilt or direct liability.

    Formulation:
      - S_dist = max(0, 1 - (distance / radius)) * 100
      - S_time = max(0, 1 - (time_diff / time_window)) * 100
      - S_obs  = min(100, (observations / 5.0) * 100)
      - S_type = lookup(ship_type) (default 50.0)
      - Score  = w_dist * S_dist + w_time * S_time + w_obs * S_obs + w_type * S_type

    Args:
        distance_km: Minimum distance from spill in km.
        time_diff_min: Absolute time difference from spill in minutes.
        observations: Number of recorded AIS positions within filter window.
        ship_type: Vessel classification string.
        config: AISConfig containing search parameters and weights.

    Returns:
        float: Candidate score clamped between 0.0 and 100.0, rounded to 1 decimal.
    """
    # 1. Distance component (closer -> higher)
    radius = max(config.search_radius_km, 0.001)
    s_dist = max(0.0, 1.0 - (distance_km / radius)) * 100.0

    # 2. Time difference component (closer to spill time -> higher)
    window = max(config.time_window_minutes, 0.001)
    s_time = max(0.0, 1.0 - (time_diff_min / window)) * 100.0

    # 3. Observation density / continuity component
    # Reaching 5 or more AIS pings provides full observation confidence
    s_obs = min(100.0, (observations / 5.0) * 100.0)

    # 4. Ship type supporting feature
    type_key = ship_type.lower() if ship_type else "other"
    s_type = config.ship_type_scores.get(type_key, config.ship_type_scores.get("other", 50.0))
    # Handle partial substring matches (e.g. "Oil Tanker" -> "tanker")
    for key, score_val in config.ship_type_scores.items():
        if key in type_key:
            s_type = score_val
            break

    total_score = (
        config.distance_weight * s_dist
        + config.time_weight * s_time
        + config.observations_weight * s_obs
        + config.ship_type_weight * s_type
    )

    clamped_score = max(0.0, min(100.0, total_score))
    return round(clamped_score, 1)


def group_by_vessel(
    filtered_records: List[AISRecord],
    config: Optional[AISConfig] = None,
) -> List[CandidateVessel]:
    """Group AIS records by MMSI, determine the closest observation point, and compute scores.

    Args:
        filtered_records: List of AIS records that have passed spatio-temporal filters.
        config: Optional configuration for scoring parameters.

    Returns:
        List of CandidateVessel objects (unsorted).
    """
    cfg = config or DEFAULT_CONFIG
    if not filtered_records:
        return []

    # Group records by MMSI
    grouped: Dict[str, List[AISRecord]] = defaultdict(list)
    for rec in filtered_records:
        grouped[rec.mmsi].append(rec)

    candidates: List[CandidateVessel] = []

    for mmsi, records in grouped.items():
        # Sort trajectory records chronologically
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        trajectory_records = [r.to_trajectory_dict() for r in sorted_records]

        # Identify the closest observation point to the spill
        # Secondary key is minimum time difference
        closest_rec = min(
            records,
            key=lambda r: (
                r.distance_km if r.distance_km is not None else float("inf"),
                r.time_difference_minutes if r.time_difference_minutes is not None else float("inf"),
            ),
        )

        # Merge metadata (find first non-null across records if available)
        ship_name = next((r.ship_name for r in records if r.ship_name), closest_rec.ship_name)
        imo = next((r.imo for r in records if r.imo), closest_rec.imo)
        ship_type = next((r.ship_type for r in records if r.ship_type), closest_rec.ship_type)

        dist_val = round(closest_rec.distance_km or 0.0, 2)
        time_diff_val = round(closest_rec.time_difference_minutes or 0.0, 1)
        obs_count = len(records)

        score = compute_candidate_score(
            distance_km=dist_val,
            time_diff_min=time_diff_val,
            observations=obs_count,
            ship_type=ship_type,
            config=cfg,
        )

        candidates.append(
            CandidateVessel(
                mmsi=mmsi,
                vessel_name=ship_name,
                vessel_type=ship_type,
                imo=imo,
                latitude=closest_rec.latitude,
                longitude=closest_rec.longitude,
                closest_record_time=closest_rec.timestamp_iso,
                minimum_distance_km=dist_val,
                time_difference_minutes=time_diff_val,
                observations=obs_count,
                candidate_score=score,
                ais_records=trajectory_records,
            )
        )

    return candidates


def rank_candidates(
    candidates: List[CandidateVessel],
    top_n: int = 5,
) -> List[CandidateVessel]:
    """Rank candidate vessels descending by candidate score, tie-breaking by distance and time.

    Args:
        candidates: List of CandidateVessel instances.
        top_n: Maximum number of candidate vessels to return.

    Returns:
        Top N candidate vessels ordered by likelihood/score.
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (-c.candidate_score, c.distance_km, c.time_difference_minutes),
    )
    return sorted_candidates[:top_n]

