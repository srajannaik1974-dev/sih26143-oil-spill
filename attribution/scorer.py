"""
Explainable correlation scoring model for Vessel Attribution.
Implements exact weighted scoring model:
- Distance Score: 30%
- Time Score: 25%
- Trajectory Score: 25%
- Speed/Heading Score: 20%
Outputs standardized classifications and human-readable explainability strings.
"""

import math
from typing import Dict, Any, Tuple
from .schemas import FeatureScores, CandidateVesselResult


def compute_distance_score(dist_km: float, max_radius_km: float = 50.0) -> float:
    """
    Compute normalized Distance Score (0-100) using exponential decay.
    0 km = 100.0; decay half-life at ~10 km.
    """
    if dist_km < 0:
        return 0.0
    # Half-distance scaling factor (10 km)
    scale = 10.0
    score = 100.0 * math.exp(-dist_km / scale)
    return max(0.0, min(100.0, round(score, 2)))


def compute_time_score(dt_minutes: float, max_window_hours: float = 24.0) -> float:
    """
    Compute normalized Time Score (0-100) using exponential decay.
    0 min = 100.0; decay half-life at 180 min (3 hours).
    """
    if dt_minutes < 0:
        return 0.0
    scale_minutes = 180.0  # 3 hours
    score = 100.0 * math.exp(-dt_minutes / scale_minutes)
    return max(0.0, min(100.0, round(score, 2)))


def compute_trajectory_score(
    min_dist_km: float,
    avg_dist_in_zone_km: float,
    dwell_count: int
) -> float:
    """
    Compute normalized Trajectory Score (0-100) based on spatial persistence and path proximity.
    """
    # Proximity component (0-70 points)
    proximity_score = 70.0 * math.exp(-avg_dist_in_zone_km / 12.0)
    
    # Persistence component (0-30 points) based on multiple pings near origin
    persistence_score = min(30.0, dwell_count * 7.5)

    total = proximity_score + persistence_score
    return max(0.0, min(100.0, round(total, 2)))


def compute_speed_heading_score(
    speed_knots: float,
    heading_diff_deg: float,
    vessel_type: str
) -> float:
    """
    Compute normalized Speed/Heading Score (0-20 points for speed anomaly + 0-20 for heading alignment, scaled to 0-100).
    Slower speeds (0.5 to 7.0 knots) near open-sea spill origin indicate potential dumping/cargo operation.
    Heading pointing towards/along spill vector increases correlation.
    """
    # Speed score: anomalous slow speed (0.5 - 6 knots) vs high speed transit
    if 0.5 <= speed_knots <= 6.0:
        speed_subscore = 100.0  # Slow speed anomaly near origin
    elif speed_knots < 0.5:
        speed_subscore = 70.0   # Stationary/anchored
    elif 6.0 < speed_knots <= 12.0:
        speed_subscore = 60.0   # Medium transit
    else:
        speed_subscore = 30.0   # High speed transit

    # Heading alignment score (0 deg diff = direct alignment = 100; 180 deg = opposite = 20)
    heading_subscore = max(20.0, 100.0 - (heading_diff_deg / 180.0) * 80.0)

    # Combine 50% speed anomaly + 50% heading alignment
    total = (0.5 * speed_subscore) + (0.5 * heading_subscore)
    return max(0.0, min(100.0, round(total, 2)))


def score_vessel_candidate(
    features: Dict[str, Any],
    max_search_radius_km: float = 50.0,
    max_time_window_hours: float = 24.0
) -> Tuple[FeatureScores, float, str]:
    """
    Score a single candidate vessel using the strict weighted correlation formula:
    - Distance: 30%
    - Time: 25%
    - Trajectory: 25%
    - Speed/Heading: 20%

    Returns (FeatureScores, final_score, raw_explanation_facts).
    """
    dist_km = features["min_distance_km"]
    dt_minutes = features["cpa_time_diff_minutes"]

    s_dist = compute_distance_score(dist_km, max_search_radius_km)
    s_time = compute_time_score(dt_minutes, max_time_window_hours)
    s_traj = compute_trajectory_score(
        dist_km, features["avg_dist_in_zone_km"], features["dwell_count"]
    )
    s_speed_heading = compute_speed_heading_score(
        features["speed_at_cpa_knots"],
        features["heading_diff_deg"],
        features["vessel_type"]
    )

    # Final weighted correlation score calculation (30%, 25%, 25%, 20%)
    final_score = (
        (0.30 * s_dist) +
        (0.25 * s_time) +
        (0.25 * s_traj) +
        (0.20 * s_speed_heading)
    )
    final_score = round(final_score, 2)

    feature_scores = FeatureScores(
        distance_score=s_dist,
        time_score=s_time,
        trajectory_score=s_traj,
        speed_heading_score=s_speed_heading
    )

    explanation_facts = (
        f"Passed within {dist_km:.2f} km of spill origin with closest approach "
        f"{dt_minutes:.1f} minutes from estimated release time. "
        f"Speed at CPA was {features['speed_at_cpa_knots']:.1f} knots. "
        f"Dwell count in zone: {features['dwell_count']} position reports. "
        f"Feature correlation sub-scores: Distance ({s_dist:.1f}/100 [30% weight]), "
        f"Time ({s_time:.1f}/100 [25% weight]), Trajectory ({s_traj:.1f}/100 [25% weight]), "
        f"Speed/Heading ({s_speed_heading:.1f}/100 [20% weight])."
    )

    return feature_scores, final_score, explanation_facts


def build_candidate_result(
    features: Dict[str, Any],
    feature_scores: FeatureScores,
    final_score: float,
    raw_explanation: str,
    rank: int
) -> CandidateVesselResult:
    """
    Build structured CandidateVesselResult enforcing correct classification terminology.
    Must use: 'Highest-correlated candidate vessel' (for Rank 1) or 'Potentially responsible vessel'.
    Must NOT state or claim that a vessel is definitely responsible.
    """
    if rank == 1:
        classification = "Highest-correlated candidate vessel"
    else:
        classification = "Potentially responsible vessel"

    explanation = (
        f"Classified as '{classification}' (Rank {rank}, Score {final_score:.2f}/100). "
        f"{raw_explanation} "
        "Note: This score indicates correlation based on AIS trajectory data and release window; "
        "it does not constitute definitive proof of responsibility."
    )

    return CandidateVesselResult(
        vessel_id=features["vessel_id"],
        mmsi=features["mmsi"],
        vessel_name=features["vessel_name"],
        vessel_type=features["vessel_type"],
        distance_km=round(features["min_distance_km"], 2),
        time_difference_minutes=round(features["cpa_time_diff_minutes"], 1),
        feature_scores=feature_scores,
        final_score=final_score,
        rank=rank,
        classification=classification,
        explanation=explanation
    )
