from fastapi import APIRouter, status
from pydantic import BaseModel, Field
from backend.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Status string")
    service: str = Field(..., description="Service title")
    version: str = Field(..., description="Service version")


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Backend Health Status",
    description="Returns backend health status, service name, and version."
)
async def check_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        version=settings.VERSION
    )
