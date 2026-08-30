from datetime import datetime, timezone
from backend.services.vessel_service import BaseVesselService
from backend.schemas.vessel import VesselRankRequest, VesselRankResponse, RankedVessel, AttributionFactor


class MockVesselService(BaseVesselService):
    """
    Mock implementation of vessel attribution ranking service.
    Ranks candidate vessels using mock spatio-temporal proximity and vessel type heuristics.
    """

    async def rank_vessels(self, request: VesselRankRequest) -> VesselRankResponse:
        candidates = request.candidate_vessels

        # Sort candidate vessels by proximity (distance_to_source_km) ascending
        sorted_candidates = sorted(candidates, key=lambda v: v.distance_to_source_km)

        ranked_list = []
        for index, vessel in enumerate(sorted_candidates, start=1):
            # Calculate mock risk score based on distance and vessel type
            dist_factor = max(0.1, 1.0 - (vessel.distance_to_source_km / 50.0))
            type_multiplier = 1.0 if "Tanker" in vessel.vessel_type else 0.7
            raw_risk = dist_factor * type_multiplier
            risk_score = round(min(0.99, max(0.05, raw_risk)), 2)

            factors = [
                AttributionFactor(
                    factor_name="Spatial Proximity",
                    score=round(dist_factor, 2),
                    description=f"Vessel was {vessel.distance_to_source_km} km from estimated spill origin."
                ),
                AttributionFactor(
                    factor_name="Vessel Classification Risk",
                    score=round(type_multiplier, 2),
                    description=f"Vessel type '{vessel.vessel_type}' has standard oil payload risk index."
                )
            ]

            ranked_list.append(
                RankedVessel(
                    vessel=vessel,
                    rank=index,
                    risk_score=risk_score,
                    attribution_factors=factors
                )
            )

        return VesselRankResponse(
            ranked_vessels=ranked_list,
            total_ranked=len(ranked_list),
            ranked_at=datetime.now(timezone.utc),
            disclaimer="MOCK ATTRIBUTION RESULT: Vessel rankings and risk scores are derived using mock proximity rules for integration testing and do NOT represent real legal attribution."
        )
