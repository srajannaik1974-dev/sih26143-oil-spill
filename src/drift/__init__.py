"""
Oil-spill drift module containing data models, geographic coordinate utilities,
environmental data layer, drift vector physics, forward trajectory simulation,
backward trajectory simulation, probable origin estimation, end-to-end drift pipeline,
and Member 2 integration boundary adapter.
"""

from .models import (
    SpillObservation,
    EnvironmentalObservation,
    TrajectoryPoint,
    BacktrackingResult,
)
from .geo_utils import (
    destination_point,
    distance_between_points,
    bearing_between_points,
    validate_coordinates,
)
from .environment import (
    EnvironmentalDataProvider,
    SyntheticEnvironmentalProvider,
    FileEnvironmentalProvider,
    RealEnvironmentalProvider,
    EnvironmentalDataUnavailableError,
    get_nearest_observation,
)
from .physics import (
    Vector2D,
    wind_to_vector,
    current_to_vector,
    combine_drift_velocity,
    speed_and_bearing_from_vector,
)
from .simulator import (
    ForwardSimulator,
    simulate_forward_trajectory,
)
from .backtracker import (
    BackwardSimulator,
    simulate_backward_trajectory,
)
from .origin import (
    ProbableOriginEstimator,
    estimate_probable_origin,
    ProbableOriginResult,
    InsufficientTrajectoryError,
)
from .pipeline import (
    DriftOriginPipeline,
    run_drift_pipeline,
    DriftPipelineResult,
)
from .integration import (
    DetectedSpillInput,
    DriftOriginOutput,
    process_detected_spill,
)

__all__ = [
    "SpillObservation",
    "EnvironmentalObservation",
    "TrajectoryPoint",
    "BacktrackingResult",
    "destination_point",
    "distance_between_points",
    "bearing_between_points",
    "validate_coordinates",
    "EnvironmentalDataProvider",
    "SyntheticEnvironmentalProvider",
    "FileEnvironmentalProvider",
    "RealEnvironmentalProvider",
    "EnvironmentalDataUnavailableError",
    "get_nearest_observation",
    "Vector2D",
    "wind_to_vector",
    "current_to_vector",
    "combine_drift_velocity",
    "speed_and_bearing_from_vector",
    "ForwardSimulator",
    "simulate_forward_trajectory",
    "BackwardSimulator",
    "simulate_backward_trajectory",
    "ProbableOriginEstimator",
    "estimate_probable_origin",
    "ProbableOriginResult",
    "InsufficientTrajectoryError",
    "DriftOriginPipeline",
    "run_drift_pipeline",
    "DriftPipelineResult",
    "DetectedSpillInput",
    "DriftOriginOutput",
    "process_detected_spill",
]
