from abc import ABC, abstractmethod
from backend.schemas.spill import SpillDetectionRequest, SpillDetectionResponse


class BaseSpillService(ABC):
    """
    Abstract interface for satellite oil spill detection services.
    Future satellite ML models (e.g. UNet, YOLOv8, ResNet-based SAR segmentation)
    should inherit from this class and implement detect_spill.
    """

    @abstractmethod
    async def detect_spill(self, request: SpillDetectionRequest) -> SpillDetectionResponse:
        """
        Analyze satellite imagery metadata/URL to detect oil spill presence and polygon boundary.
        """
        pass

    @abstractmethod
    async def detect_spill_file(self, file_path: str, original_filename: str) -> SpillDetectionResponse:
        """
        Analyze an uploaded Sentinel-1 GeoTIFF file to detect oil spill presence and polygon boundary.
        """
        pass

