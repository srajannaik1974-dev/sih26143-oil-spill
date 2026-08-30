from datetime import datetime, timezone
from backend.services.ais_service import BaseAISService
from backend.schemas.ais import AISCandidatesRequest, AISCandidatesResponse, VesselAISData


class MockAISService(BaseAISService):
    """
    Mock implementation of AIS data query service.
    Returns simulated candidate vessels with realistic maritime parameters near the query location.
    """

    async def get_candidates(self, request: AISCandidatesRequest) -> AISCandidatesResponse:
        lat = request.source_latitude
        lon = request.source_longitude

        # Generate realistic mock vessels in the vicinity
        candidates = [
            VesselAISData(
                mmsi="413298410",
                vessel_name="OCEAN TITAN",
                vessel_type="Crude Oil Tanker",
                callsign="VRHK8",
                flag="Panama",
                latitude=round(lat + 0.012, 6),
                longitude=round(lon - 0.008, 6),
                timestamp=request.timestamp,
                speed_knots=12.4,
                heading_degrees=145.0,
                distance_to_source_km=1.85
            ),
            VesselAISData(
                mmsi="235091220",
                vessel_name="PACIFIC TRADER",
                vessel_type="Container Ship",
                callsign="2GBC4",
                flag="Liberia",
                latitude=round(lat - 0.035, 6),
                longitude=round(lon + 0.021, 6),
                timestamp=request.timestamp,
                speed_knots=16.8,
                heading_degrees=210.0,
                distance_to_source_km=4.62
            ),
            VesselAISData(
                mmsi="563048910",
                vessel_name="SEA STAR",
                vessel_type="Chemical Tanker",
                callsign="9V9201",
                flag="Singapore",
                latitude=round(lat + 0.065, 6),
                longitude=round(lon + 0.045, 6),
                timestamp=request.timestamp,
                speed_knots=11.1,
                heading_degrees=88.0,
                distance_to_source_km=8.95
            ),
            VesselAISData(
                mmsi="311000845",
                vessel_name="ATLANTIC RUNNER",
                vessel_type="Bulk Carrier",
                callsign="C6BX3",
                flag="Bahamas",
                latitude=round(lat - 0.120, 6),
                longitude=round(lon - 0.095, 6),
                timestamp=request.timestamp,
                speed_knots=14.0,
                heading_degrees=330.0,
                distance_to_source_km=16.40
            ),
        ]

        # Filter out candidates exceeding the requested search radius if necessary
        filtered = [v for v in candidates if v.distance_to_source_km <= request.search_radius_km]

        return AISCandidatesResponse(
            candidates=filtered,
            total_count=len(filtered),
            search_radius_km=request.search_radius_km,
            timestamp=datetime.now(timezone.utc)
        )
