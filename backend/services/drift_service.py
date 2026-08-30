from abc import ABC, abstractmethod
from backend.schemas.drift import BacktrackRequest, BacktrackResponse


class BaseDriftService(ABC):
    """
    Abstract interface for ocean drift backtrack modeling services.
    Future ocean drift physics implementations (e.g. OpenDrift, GNOME, MetOcean wind/current vector models)
    should inherit from this class and implement backtrack.
    """

    @abstractmethod
    async def backtrack(self, request: BacktrackRequest) -> BacktrackResponse:
        """
        Backtrack detected oil spill location and timestamp to estimate origin region and trajectory.
        """
        pass
