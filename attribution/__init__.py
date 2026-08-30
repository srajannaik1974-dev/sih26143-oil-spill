"""
Vessel Attribution Module (SIH 2026 PS 26143 - Member 4)
Provides candidate vessel filtering, spatial-temporal feature extraction,
explainable correlation scoring, and vessel ranking.
"""

from .schemas import (
    SpillOriginInput,
    AISPosition,
    AISTrajectoryRecord,
    FeatureScores,
    CandidateVesselResult,
    AttributionResponse,
)
from .service import VesselAttributionService

__all__ = [
    "SpillOriginInput",
    "AISPosition",
    "AISTrajectoryRecord",
    "FeatureScores",
    "CandidateVesselResult",
    "AttributionResponse",
    "VesselAttributionService",
]
