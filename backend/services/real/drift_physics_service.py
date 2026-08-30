from datetime import datetime, timezone
from typing import List, Optional

from backend.services.drift_service import BaseDriftService
from backend.services.mock.drift import MockDriftService
from backend.schemas.drift import (
    BacktrackRequest, BacktrackResponse, TrajectoryPoint, SourceArea
)
from backend.schemas.spill import GeoCoordinate

import drift_adapter


class RealDriftServiceAdapter(BaseDriftService):
    """
    Adapter wrapping Member 2's Ocean Drift Backtracking Engine (drift_adapter.run_drift_analysis).
    Maps Member 6 BacktrackRequest into Member 2's input contract and translates DriftOriginOutput back.
    """

    def __init__(self):
        self.fallback_mock = MockDriftService()

    async def backtrack(self, request: BacktrackRequest) -> BacktrackResponse:
        spill_info = {
            "latitude": request.spill_location.latitude,
            "longitude": request.spill_location.longitude,
            "timestamp": request.timestamp.isoformat(),
            "area_km2": 4.85,
            "confidence": 0.90
        }

        try:
            drift_result = drift_adapter.run_drift_analysis(
                spill_info=spill_info,
                duration_hours=request.drift_hours,
                step_minutes=30.0
            )

            if drift_result is not None:
                # Extract backward simulation trajectory points
                traj_points: List[TrajectoryPoint] = []
                if hasattr(drift_result, "backward_simulation") and drift_result.backward_simulation:
                    for pt in drift_result.backward_simulation.points:
                        traj_points.append(
                            TrajectoryPoint(
                                timestamp=pt.timestamp,
                                latitude=round(pt.latitude, 6),
                                longitude=round(pt.longitude, 6),
                                uncertainty_radius_km=round(getattr(pt, "uncertainty_km", 1.5), 2)
                            )
                        )

                if not traj_points:
                    traj_points.append(
                        TrajectoryPoint(
                            timestamp=request.timestamp,
                            latitude=request.spill_location.latitude,
                            longitude=request.spill_location.longitude,
                            uncertainty_radius_km=1.5
                        )
                    )

                # Extract origin estimation
                origin_lat = request.spill_location.latitude
                origin_lon = request.spill_location.longitude
                origin_radius = 5.0

                if hasattr(drift_result, "estimated_origin") and drift_result.estimated_origin:
                    origin_lat = drift_result.estimated_origin.latitude
                    origin_lon = drift_result.estimated_origin.longitude
                    origin_radius = drift_result.estimated_origin.uncertainty_radius_km

                source_center = GeoCoordinate(latitude=round(origin_lat, 6), longitude=round(origin_lon, 6))
                
                r = 0.02
                boundary_polygon = [
                    GeoCoordinate(latitude=round(source_center.latitude + r, 6), longitude=round(source_center.longitude, 6)),
                    GeoCoordinate(latitude=round(source_center.latitude, 6), longitude=round(source_center.longitude + r, 6)),
                    GeoCoordinate(latitude=round(source_center.latitude - r, 6), longitude=round(source_center.longitude, 6)),
                    GeoCoordinate(latitude=round(source_center.latitude, 6), longitude=round(source_center.longitude - r, 6)),
                    GeoCoordinate(latitude=round(source_center.latitude + r, 6), longitude=round(source_center.longitude, 6)),
                ]

                source_area = SourceArea(
                    center=source_center,
                    radius_km=round(origin_radius, 2),
                    boundary_polygon=boundary_polygon
                )

                return BacktrackResponse(
                    spill_location=request.spill_location,
                    detection_timestamp=request.timestamp,
                    estimated_source_area=source_area,
                    trajectory=traj_points,
                    timestamp=datetime.now(timezone.utc),
                    disclaimer="M2 DRIFT MODEL: Backtrack trajectory calculated using Member 2 Hydrodynamic Ocean Drift Engine."
                )

        except Exception as e:
            print(f"[RealDriftServiceAdapter] Error running drift analysis: {e}. Falling back to mock service.")

        return await self.fallback_mock.backtrack(request)
