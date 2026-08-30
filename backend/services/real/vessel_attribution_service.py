from datetime import datetime, timezone
from typing import List, Dict, Optional

from backend.services.vessel_service import BaseVesselService
from backend.schemas.vessel import (
    VesselRankRequest,
    VesselRankResponse,
    RankedVessel,
    AttributionFactor,
)
from backend.schemas.ais import VesselAISData

from attribution.schemas import (
    SpillOriginInput,
    AISTrajectoryRecord,
    AISPosition,
)
from attribution.service import VesselAttributionService


class RealVesselAttributionService(BaseVesselService):
    """
    Adapter service integrating Member 4's Vessel Attribution Engine
    into Member 6's backend API.
    
    Transforms Member 6 request contracts into Member 4's SpillOriginInput & AISTrajectoryRecord,
    calls Member 4's VesselAttributionService.analyze_attribution directly, and maps the results back.
    """

    async def rank_vessels(self, request: VesselRankRequest) -> VesselRankResponse:
        # 1. Map request spill source to Member 4's SpillOriginInput
        release_time = request.estimated_release_time or request.timestamp
        spill_origin = SpillOriginInput(
            latitude=request.spill_source.latitude,
            longitude=request.spill_source.longitude,
            estimated_release_time=release_time,
            max_search_radius_km=request.search_radius_km if request.search_radius_km is not None else 50.0,
            max_time_window_hours=request.time_window_hours if request.time_window_hours is not None else 24.0,
        )

        # Map candidate vessels to AISTrajectoryRecord list
        vessel_map: Dict[str, VesselAISData] = {}
        trajectory_records: List[AISTrajectoryRecord] = []

        for v in request.candidate_vessels:
            vessel_key = v.vessel_id or v.mmsi
            vessel_map[vessel_key] = v
            vessel_map[v.mmsi] = v

            positions_list: List[AISPosition] = []

            if v.positions:
                for pos in v.positions:
                    if isinstance(pos, dict):
                        pos_ts = pos.get("timestamp")
                        if isinstance(pos_ts, str):
                            pos_ts = datetime.fromisoformat(pos_ts.replace("Z", "+00:00"))
                        positions_list.append(
                            AISPosition(
                                timestamp=pos_ts or v.timestamp,
                                latitude=pos.get("latitude", v.latitude),
                                longitude=pos.get("longitude", v.longitude),
                                speed_knots=pos.get("speed_knots", v.speed_knots),
                                heading_deg=pos.get("heading_deg", v.heading_degrees),
                                course_over_ground=pos.get("course_over_ground", v.heading_degrees),
                            )
                        )
                    elif isinstance(pos, AISPosition):
                        positions_list.append(pos)
            else:
                # Default single AIS position from candidate point data
                positions_list.append(
                    AISPosition(
                        timestamp=v.timestamp,
                        latitude=v.latitude,
                        longitude=v.longitude,
                        speed_knots=v.speed_knots,
                        heading_deg=v.heading_degrees,
                        course_over_ground=v.heading_degrees,
                    )
                )

            trajectory_records.append(
                AISTrajectoryRecord(
                    vessel_id=vessel_key,
                    mmsi=v.mmsi,
                    vessel_name=v.vessel_name,
                    vessel_type=v.vessel_type,
                    positions=positions_list,
                )
            )

        # 2. Invoke Member 4's attribution algorithm directly in-process
        attribution_result = VesselAttributionService.analyze_attribution(
            spill=spill_origin,
            vessels=trajectory_records
        )

        # 3. Map Member 4's AttributionResponse back to Member 6's VesselRankResponse
        ranked_vessels: List[RankedVessel] = []

        for cand in attribution_result.candidate_vessels:
            orig_vessel = vessel_map.get(cand.vessel_id) or vessel_map.get(cand.mmsi)
            
            if orig_vessel is None:
                orig_vessel = VesselAISData(
                    mmsi=cand.mmsi,
                    vessel_name=cand.vessel_name,
                    vessel_type=cand.vessel_type,
                    latitude=spill_origin.latitude,
                    longitude=spill_origin.longitude,
                    timestamp=release_time,
                    speed_knots=0.0,
                    heading_degrees=0.0,
                    distance_to_source_km=cand.distance_km,
                    vessel_id=cand.vessel_id
                )
            else:
                # Update distance field with precise value calculated by attribution engine
                orig_vessel = orig_vessel.model_copy(update={"distance_to_source_km": cand.distance_km})

            factors = [
                AttributionFactor(
                    factor_name="Distance Proximity Score",
                    score=round(cand.feature_scores.distance_score / 100.0, 4),
                    description=f"Distance weight (30%): Closest approach is {cand.distance_km:.2f} km from spill origin."
                ),
                AttributionFactor(
                    factor_name="Time Proximity Score",
                    score=round(cand.feature_scores.time_score / 100.0, 4),
                    description=f"Time weight (25%): Time difference is {cand.time_difference_minutes:.1f} minutes from release time."
                ),
                AttributionFactor(
                    factor_name="Trajectory Dwell Score",
                    score=round(cand.feature_scores.trajectory_score / 100.0, 4),
                    description=f"Trajectory weight (25%): Geometry and dwell persistence score is {cand.feature_scores.trajectory_score:.1f}/100."
                ),
                AttributionFactor(
                    factor_name="Speed & Heading Anomaly Score",
                    score=round(cand.feature_scores.speed_heading_score / 100.0, 4),
                    description=f"Behavior weight (20%): Speed changes and heading alignment score is {cand.feature_scores.speed_heading_score:.1f}/100."
                ),
            ]

            risk_score = round(cand.final_score / 100.0, 4)

            ranked_vessels.append(
                RankedVessel(
                    vessel=orig_vessel,
                    rank=cand.rank,
                    risk_score=risk_score,
                    attribution_factors=factors,
                    classification=cand.classification,
                    explanation=cand.explanation,
                    final_score=cand.final_score,
                )
            )

        return VesselRankResponse(
            ranked_vessels=ranked_vessels,
            total_ranked=len(ranked_vessels),
            ranked_at=datetime.now(timezone.utc),
            disclaimer="ATTRIBUTION RESULT: Vessel rankings and correlation scores calculated using Member 4 Vessel Attribution Engine."
        )
