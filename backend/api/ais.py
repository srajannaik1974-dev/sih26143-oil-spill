from fastapi import APIRouter, Depends, status, HTTPException
from backend.schemas.ais import AISCandidatesRequest, AISCandidatesResponse
from backend.services.ais_service import BaseAISService
from backend.dependencies import get_ais_service

router = APIRouter(prefix="/ais", tags=["AIS Data Query"])


@router.post(
    "/candidates",
    response_model=AISCandidatesResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Candidate Vessels from AIS Data",
    description="Queries candidate vessels within a spatio-temporal radius of an estimated spill origin."
)
async def get_ais_candidates(
    request: AISCandidatesRequest,
    ais_service: BaseAISService = Depends(get_ais_service)
) -> AISCandidatesResponse:
    try:
        return await ais_service.get_candidates(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying AIS candidate vessels: {str(e)}"
        )
