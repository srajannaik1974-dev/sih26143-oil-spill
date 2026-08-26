"""
High-level service interface for the Vessel Attribution module.
Coordinates candidate vessel filtering, feature extraction, explainable scoring, and vessel ranking.
"""

from typing import List
from .schemas import (
    SpillOriginInput,
    AISTrajectoryRecord,
    AttributionResponse,
    CandidateVesselResult,
)
from .features import extract_vessel_features
from .scorer import score_vessel_candidate, build_candidate_result


class VesselAttributionService:
    """
    Service responsible for calculating vessel correlation scores and ranking
    potentially responsible vessels for a given oil spill event.
    """

    @staticmethod
    def analyze_attribution(
        spill: SpillOriginInput,
        vessels: List[AISTrajectoryRecord]
    ) -> AttributionResponse:
        """
        Analyze a list of AIS vessel trajectories against an oil spill origin and release time.

        Processing Pipeline:
        1. Candidate Vessel Filtering (spatial radius & temporal window)
        2. Spatial-Temporal Feature Calculation (distance, time difference, trajectory dwell, speed/heading)
        3. Explainable Correlation Scoring (Distance 30%, Time 25%, Trajectory 25%, Speed/Heading 20%)
        4. Vessel Ranking & Terminology Assignment

        Returns AttributionResponse containing ranked candidate vessels.
        """
        evaluated_candidates = []

        for vessel in vessels:
            features = extract_vessel_features(spill, vessel)
            if features is None:
                # Filtered out (outside spatial or temporal boundaries)
                continue

            feature_scores, final_score, raw_explanation = score_vessel_candidate(
                features,
                max_search_radius_km=spill.max_search_radius_km,
                max_time_window_hours=spill.max_time_window_hours
            )

            evaluated_candidates.append({
                "features": features,
                "feature_scores": feature_scores,
                "final_score": final_score,
                "raw_explanation": raw_explanation,
            })

        # Sort candidates descending by final correlation score
        evaluated_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # Assign ranks and build response items
        candidate_results: List[CandidateVesselResult] = []
        for rank_idx, item in enumerate(evaluated_candidates, start=1):
            res = build_candidate_result(
                features=item["features"],
                feature_scores=item["feature_scores"],
                final_score=item["final_score"],
                raw_explanation=item["raw_explanation"],
                rank=rank_idx
            )
            candidate_results.append(res)

        return AttributionResponse(
            spill_origin=spill,
            total_vessels_evaluated=len(vessels),
            candidate_vessels=candidate_results
        )
