"""
Unit and integration tests for the Vessel Attribution module (Member 4).
"""

from datetime import datetime, timedelta, timezone
import pytest
from attribution.schemas import SpillOriginInput, AISTrajectoryRecord, AISPosition
from attribution.features import haversine_distance_km, extract_vessel_features
from attribution.scorer import (
    compute_distance_score,
    compute_time_score,
    compute_trajectory_score,
    compute_speed_heading_score,
    score_vessel_candidate,
    build_candidate_result
)
from attribution.service import VesselAttributionService
from attribution.mock_data import create_sample_spill_origin, generate_mock_vessel_trajectories


def test_haversine_distance():
    # Mumbai to Goa approximate distance (~400 km)
    dist = haversine_distance_km(18.96, 72.82, 15.49, 73.82)
    assert 370.0 < dist < 420.0

    # Same location distance == 0
    dist_zero = haversine_distance_km(19.4167, 71.3333, 19.4167, 71.3333)
    assert dist_zero == pytest.approx(0.0, abs=1e-5)


def test_distance_score_decay():
    score_0 = compute_distance_score(0.0)
    assert score_0 == 100.0

    score_10 = compute_distance_score(10.0)
    assert 30.0 < score_10 < 40.0  # e^-1 ~ 0.3678 -> ~36.8

    score_50 = compute_distance_score(50.0)
    assert score_50 < 2.0


def test_weighted_scoring_formula():
    # Manually test 30% / 25% / 25% / 20% weights
    features = {
        "vessel_id": "V1",
        "mmsi": "123456789",
        "vessel_name": "Test Vessel",
        "vessel_type": "Tanker",
        "min_distance_km": 0.0,  # s_dist = 100.0 (30% weight -> 30.0)
        "cpa_time_diff_minutes": 0.0,  # s_time = 100.0 (25% weight -> 25.0)
        "avg_dist_in_zone_km": 0.0,
        "dwell_count": 5,
        "speed_at_cpa_knots": 4.0,  # slow speed -> high speed subscore
        "heading_diff_deg": 0.0,  # direct heading alignment -> 100
    }
    feature_scores, final_score, _ = score_vessel_candidate(features)

    assert feature_scores.distance_score == 100.0
    assert feature_scores.time_score == 100.0

    # Check exact weighted calculation:
    expected_weighted = (
        (0.30 * feature_scores.distance_score) +
        (0.25 * feature_scores.time_score) +
        (0.25 * feature_scores.trajectory_score) +
        (0.20 * feature_scores.speed_heading_score)
    )
    assert final_score == pytest.approx(round(expected_weighted, 2), abs=0.01)


def test_candidate_filtering_spatial_out_of_bounds():
    spill = create_sample_spill_origin()
    # Create vessel trajectory way outside max_search_radius_km (e.g. 100 km away)
    far_pos = AISPosition(
        timestamp=spill.estimated_release_time,
        latitude=spill.latitude + 1.0,
        longitude=spill.longitude + 1.0,
        speed_knots=10.0
    )
    far_vessel = AISTrajectoryRecord(
        vessel_id="V_FAR",
        mmsi="999999999",
        vessel_name="Far Vessel",
        vessel_type="Cargo",
        positions=[far_pos]
    )

    extracted = extract_vessel_features(spill, far_vessel)
    assert extracted is None  # Must be filtered out


def test_terminology_compliance():
    features = {
        "vessel_id": "V1",
        "mmsi": "123",
        "vessel_name": "Alpha",
        "vessel_type": "Tanker",
        "min_distance_km": 1.0,
        "cpa_time_diff_minutes": 5.0,
    }
    feature_scores, final_score, raw_exp = score_vessel_candidate({
        **features,
        "avg_dist_in_zone_km": 1.0,
        "dwell_count": 3,
        "speed_at_cpa_knots": 5.0,
        "heading_diff_deg": 10.0,
    })

    res_rank1 = build_candidate_result(features, feature_scores, final_score, raw_exp, rank=1)
    res_rank2 = build_candidate_result(features, feature_scores, final_score, raw_exp, rank=2)

    assert res_rank1.classification == "Highest-correlated candidate vessel"
    assert res_rank2.classification == "Potentially responsible vessel"

    # Ensure disclaimer exists and no claim of definite responsibility
    assert "definitive proof of responsibility" in res_rank1.explanation
    assert "definitely responsible" not in res_rank1.explanation.lower()


def test_end_to_end_attribution_service():
    spill = create_sample_spill_origin()
    trajectories = generate_mock_vessel_trajectories(spill)

    response = VesselAttributionService.analyze_attribution(spill, trajectories)

    assert response.total_vessels_evaluated == 4
    # Vessel D should be filtered out, so 3 candidate vessels returned
    assert len(response.candidate_vessels) == 3

    # Candidates must be ordered descending by final score
    scores = [c.final_score for c in response.candidate_vessels]
    assert scores == sorted(scores, reverse=True)

    # Rank 1 must be Ocean Titan (the slow-moving tanker near origin)
    rank1_vessel = response.candidate_vessels[0]
    assert rank1_vessel.rank == 1
    assert rank1_vessel.vessel_name == "Ocean Titan"
    assert rank1_vessel.classification == "Highest-correlated candidate vessel"
    assert rank1_vessel.distance_km < 3.0
    assert rank1_vessel.time_difference_minutes < 30.0
