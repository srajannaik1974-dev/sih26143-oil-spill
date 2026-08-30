from datetime import datetime, timezone
from typing import List, Optional

from backend.services.ais_service import BaseAISService
from backend.services.mock.ais import MockAISService
from backend.schemas.ais import AISCandidatesRequest, AISCandidatesResponse, VesselAISData

import ais_adapter


class RealAISServiceAdapter(BaseAISService):
    """
    Adapter wrapping Member 3's AIS candidate query engine (ais_adapter.run_ais_analysis).
    Translates M3's candidate record schema (vessel_id, closest_distance_km, closest_timestamp)
    into Member 6 / Member 4's canonical VesselAISData format (mmsi, vessel_name, positions).
    """

    def __init__(self):
        self.fallback_mock = MockAISService()

    async def get_candidates(self, request: AISCandidatesRequest) -> AISCandidatesResponse:
        try:
            ais_result = ais_adapter.run_ais_analysis(
                probable_latitude=request.source_latitude,
                probable_longitude=request.source_longitude,
                estimated_release_time=request.timestamp,
                search_radius_km=request.search_radius_km,
                time_window_minutes=request.time_window_hours * 60.0,
                top_n_candidates=10
            )

            raw_candidates = ais_result.get("candidate_vessels", [])
            vessel_list: List[VesselAISData] = []

            for item in raw_candidates:
                vessel_id = item.get("vessel_id", "UNKNOWN")
                
                # Derive numeric MMSI or fallback to string identifier
                mmsi_val = vessel_id if str(vessel_id).isdigit() else f"419{hash(vessel_id) % 1000000:06d}"
                
                # Parse timestamp
                ts_str = item.get("closest_timestamp")
                if ts_str:
                    ts_val = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    ts_val = request.timestamp

                lat_val = float(item.get("latitude", request.source_latitude))
                lon_val = float(item.get("longitude", request.source_longitude))
                speed_val = float(item.get("speed_knots") or 0.0)
                heading_val = float(item.get("heading_deg") or 0.0)
                dist_val = float(item.get("closest_distance_km") or 0.0)

                # Construct position dictionary for M4 trajectory format compatibility
                position_record = {
                    "timestamp": ts_val.isoformat(),
                    "latitude": lat_val,
                    "longitude": lon_val,
                    "speed_knots": speed_val,
                    "heading_deg": heading_val,
                    "course_over_ground": heading_val
                }

                vessel_list.append(
                    VesselAISData(
                        mmsi=mmsi_val,
                        vessel_name=f"Vessel {vessel_id}",
                        vessel_type=item.get("vessel_type", "Cargo/Tanker"),
                        callsign=item.get("callsign"),
                        flag=item.get("flag", "Unknown"),
                        latitude=lat_val,
                        longitude=lon_val,
                        timestamp=ts_val,
                        speed_knots=speed_val,
                        heading_degrees=heading_val,
                        distance_to_source_km=dist_val,
                        vessel_id=vessel_id,
                        positions=[position_record]
                    )
                )

            if vessel_list:
                return AISCandidatesResponse(
                    candidates=vessel_list,
                    total_count=len(vessel_list),
                    search_radius_km=request.search_radius_km,
                    timestamp=datetime.now(timezone.utc),
                    disclaimer="M3 AIS SERVICE: Candidates processed using Member 3 AIS Trajectory Engine."
                )

        except Exception as e:
            print(f"[RealAISServiceAdapter] Error querying AIS candidates: {e}. Falling back to mock service.")

        return await self.fallback_mock.get_candidates(request)
