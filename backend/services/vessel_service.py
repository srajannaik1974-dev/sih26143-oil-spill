from abc import ABC, abstractmethod
from backend.schemas.vessel import VesselRankRequest, VesselRankResponse


class BaseVesselService(ABC):
    """
    Abstract interface for vessel attribution and suspect ranking services.
    Future vessel attribution algorithms (e.g. machine learning risk scorer, spatio-temporal trajectory matcher)
    should inherit from this class and implement rank_vessels.
    """

    @abstractmethod
    async def rank_vessels(self, request: VesselRankRequest) -> VesselRankResponse:
        """
        Rank candidate vessels based on spill origin location/time, velocity patterns, and trajectory overlap.
        """
        pass
