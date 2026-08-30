from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class AISCandidatesRequest(BaseModel):
    source_latitude: float = Field(..., ge=-90.0, le=90.0, description="Estimated spill source latitude")
    source_longitude: float = Field(..., ge=-180.0, le=180.0, description="Estimated spill source longitude")
    timestamp: datetime = Field(..., description="ISO-8601 timestamp of estimated discharge")
    search_radius_km: float = Field(50.0, gt=0, le=500.0, description="Search radius in kilometers around source location")
    time_window_hours: float = Field(12.0, gt=0, le=72.0, description="Time window (+/- hours) around source timestamp to query AIS historical tracks")


class VesselAISData(BaseModel):
    mmsi: str = Field(..., description="Maritime Mobile Service Identity (9 digits)")
    vessel_name: str = Field(..., description="Name of the vessel")
    vessel_type: str = Field(..., description="Vessel category (e.g. Tanker, Cargo, Container, Tug)")
    callsign: Optional[str] = Field(None, description="Radio callsign")
    flag: str = Field("Unknown", description="Country flag of registration")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Vessel position latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Vessel position longitude")
    timestamp: datetime = Field(..., description="ISO-8601 timestamp of AIS broadcast message")
    speed_knots: float = Field(0.0, ge=0.0, description="Speed Over Ground (SOG) in knots")
    heading_degrees: float = Field(0.0, ge=0.0, lt=360.0, description="Course / Heading in degrees")
    distance_to_source_km: float = Field(0.0, description="Calculated distance to estimated spill source in km")
    vessel_id: Optional[str] = Field(None, description="Optional unique vessel ID string")
    positions: Optional[List[dict]] = Field(None, description="Optional trajectory time-series positions list")



class AISCandidatesResponse(BaseModel):
    candidates: List[VesselAISData] = Field(default_factory=list, description="List of candidate vessels matching spatio-temporal query")
    total_count: int = Field(..., description="Total count of candidate vessels found")
    search_radius_km: float = Field(..., description="Radius used for query")
    timestamp: datetime = Field(..., description="ISO-8601 execution timestamp")
    disclaimer: str = Field(
        "MOCK AIS RESULT: Vessel positions and metadata are simulated candidate samples for pipeline integration testing.",
        description="Notice regarding mock data"
    )
