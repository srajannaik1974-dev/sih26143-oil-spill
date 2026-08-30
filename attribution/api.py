"""
FastAPI REST router for the Vessel Attribution module.
Provides HTTP API endpoints for Member 6's core backend integration.
"""

from typing import List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from .schemas import SpillOriginInput, AISTrajectoryRecord, AttributionResponse
from .service import VesselAttributionService


router = APIRouter(prefix="/api/v1/attribution", tags=["Vessel Attribution"])


class RankAttributionRequest(BaseModel):
    """Request payload for ranking candidate vessels."""
    spill_origin: SpillOriginInput = Field(..., description="Estimated oil spill origin location and release timestamp")
    vessel_trajectories: List[AISTrajectoryRecord] = Field(..., description="List of AIS vessel trajectory time-series records")


@router.post("/rank", response_model=AttributionResponse, status_code=status.HTTP_200_OK)
def rank_candidate_vessels(payload: RankAttributionRequest) -> AttributionResponse:
    """
    HTTP POST endpoint to rank candidate vessels for an oil spill incident.

    Accepts spill origin details and candidate AIS vessel trajectories,
    calculates spatial-temporal feature correlations, and returns ranked candidate vessels.
    """
    try:
        response = VesselAttributionService.analyze_attribution(
            spill=payload.spill_origin,
            vessels=payload.vessel_trajectories
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing vessel attribution analysis: {str(exc)}"
        )
