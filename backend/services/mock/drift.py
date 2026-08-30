from datetime import datetime, timedelta, timezone
from backend.services.drift_service import BaseDriftService
from backend.schemas.drift import (
    BacktrackRequest, BacktrackResponse, TrajectoryPoint, SourceArea
)
from backend.schemas.spill import GeoCoordinate


class MockDriftService(BaseDriftService):
    """
    Mock implementation of ocean drift backtracking model.
    Simulates backtrack steps by drifting backward against mock ocean current vector.
    """

    async def backtrack(self, request: BacktrackRequest) -> BacktrackResponse:
        start_lat = request.spill_location.latitude
        start_lon = request.spill_location.longitude
        detection_time = request.timestamp
        hours = request.drift_hours

        # Mock vector drift rate: ~0.005 deg/hour lat, -0.003 deg/hour lon backward
        steps = 5
        dt_step = hours / steps
        trajectory = []

        curr_lat = start_lat
        curr_lon = start_lon
        curr_time = detection_time

        for i in range(steps + 1):
            trajectory.append(
                TrajectoryPoint(
                    timestamp=curr_time,
                    latitude=round(curr_lat, 6),
                    longitude=round(curr_lon, 6),
                    uncertainty_radius_km=round(1.5 + (i * 0.8), 2)
                )
            )
            curr_lat -= 0.005 * dt_step
            curr_lon += 0.003 * dt_step
            curr_time -= timedelta(hours=dt_step)

        # Source center is the final point in backtrack trajectory
        source_center = GeoCoordinate(
            latitude=round(curr_lat, 6),
            longitude=round(curr_lon, 6)
        )

        r = 0.02
        source_boundary = [
            GeoCoordinate(latitude=round(source_center.latitude + r, 6), longitude=round(source_center.longitude, 6)),
            GeoCoordinate(latitude=round(source_center.latitude, 6), longitude=round(source_center.longitude + r, 6)),
            GeoCoordinate(latitude=round(source_center.latitude - r, 6), longitude=round(source_center.longitude, 6)),
            GeoCoordinate(latitude=round(source_center.latitude, 6), longitude=round(source_center.longitude - r, 6)),
            GeoCoordinate(latitude=round(source_center.latitude + r, 6), longitude=round(source_center.longitude, 6)),
        ]

        source_area = SourceArea(
            center=source_center,
            radius_km=round(1.5 + (steps * 0.8), 2),
            boundary_polygon=source_boundary
        )

        return BacktrackResponse(
            spill_location=request.spill_location,
            detection_timestamp=request.timestamp,
            estimated_source_area=source_area,
            trajectory=trajectory,
            timestamp=datetime.now(timezone.utc)
        )
