from abc import ABC, abstractmethod
from backend.schemas.ais import AISCandidatesRequest, AISCandidatesResponse


class BaseAISService(ABC):
    """
    Abstract interface for AIS vessel data retrieval services.
    Future real AIS data providers (e.g. Spire AIS, MarineTraffic, AISHub, GFW API)
    should inherit from this class and implement get_candidates.
    """

    @abstractmethod
    async def get_candidates(self, request: AISCandidatesRequest) -> AISCandidatesResponse:
        """
        Query AIS historical/real-time vessel positions around a target location and time window.
        """
        pass
