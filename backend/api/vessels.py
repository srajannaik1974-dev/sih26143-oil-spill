from fastapi import APIRouter, Depends, status, HTTPException
from backend.schemas.vessel import VesselRankRequest, VesselRankResponse
from backend.services.vessel_service import BaseVesselService
from backend.dependencies import get_vessel_service

router = APIRouter(prefix="/vessels", tags=["Vessel Attribution & Ranking"])


@router.post(
    "/rank",
    response_model=VesselRankResponse,
    status_code=status.HTTP_200_OK,
    summary="Rank Candidate Suspect Vessels",
    description="Ranks candidate vessels according to spatial-temporal proximity and attribution heuristics."
)
async def rank_vessels(
    request: VesselRankRequest,
    vessel_service: BaseVesselService = Depends(get_vessel_service)
) -> VesselRankResponse:
    try:
        return await vessel_service.rank_vessels(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ranking candidate vessels: {str(e)}"
        )
