"""
End-to-End Member 2 Drift & Probable Origin Pipeline.

Orchestrates environmental data lookup, vector drift physics, backward trajectory simulation,
and probable origin estimation into a unified pipeline.

Scientific Scope & Model Assumptions:
- Software & Prototype Pipeline: Connects Member 2 modules for oil drift analysis.
- Input: Detected spill location (lat/lon) and detection timestamp.
- Output: Probable origin candidate location, estimated release-time candidate, and backward trajectory.
- Uses SyntheticEnvironmentalProvider as the default provider for prototype/testing.
- This is a software prototype; operational use requires real-world environmental observation feeds.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from .geo_utils import validate_coordinates
from .models import TrajectoryPoint
from .environment import (
    EnvironmentalDataProvider,
    SyntheticEnvironmentalProvider,
    EnvironmentalDataUnavailableError,
)
from .backtracker import simulate_backward_trajectory
from .origin import estimate_probable_origin, InsufficientTrajectoryError


class DriftPipelineResult(BaseModel):
    """
    Consolidated end-to-end result model for Member 2 drift and origin analysis.
    """
    model_config = ConfigDict(extra="forbid")

    spill_id: str
    detected_latitude: float
    detected_longitude: float
    detection_timestamp: datetime
    probable_latitude: float
    probable_longitude: float
    estimated_release_time: datetime
    trajectory_points_used: int
    status: str
    message: str
    backward_trajectory: List[TrajectoryPoint]


class DriftOriginPipeline:
    """
    End-to-End pipeline orchestrator for Member 2 (Environmental Data -> Backtracking -> Probable Origin).
    """

    def __init__(
        self,
        env_provider: Optional[EnvironmentalDataProvider] = None,
        default_windage_factor: float = 0.03,
        default_max_gap_minutes: Optional[float] = 60.0,
    ):
        """
        Initialize the pipeline with default environmental provider and configuration.

        :param env_provider: EnvironmentalDataProvider instance (defaults to SyntheticEnvironmentalProvider)
        :param default_windage_factor: Default windage factor (default: 0.03)
        :param default_max_gap_minutes: Default max gap for environmental lookup in minutes (default: 60.0)
        """
        self.env_provider = env_provider or SyntheticEnvironmentalProvider()
        self.default_windage_factor = default_windage_factor
        self.default_max_gap_minutes = default_max_gap_minutes

    def analyze_spill(
        self,
        latitude: float,
        longitude: float,
        detection_timestamp: datetime,
        spill_id: str = "SPILL_UNKNOWN",
        duration_hours: float = 2.0,
        step_minutes: float = 30.0,
        windage_factor: Optional[float] = None,
        max_gap_minutes: Optional[float] = None,
        env_provider: Optional[EnvironmentalDataProvider] = None,
    ) -> DriftPipelineResult:
        """
        Runs the end-to-end drift and probable origin estimation pipeline for a detected oil spill.

        :param latitude: Detected spill latitude [-90, 90]
        :param longitude: Detected spill longitude [-180, 180]
        :param detection_timestamp: Detection timestamp (must be timezone-aware)
        :param spill_id: Optional unique identifier for the spill observation
        :param duration_hours: Backward simulation duration in hours (>= 0, default: 2.0)
        :param step_minutes: Simulation time-step in minutes (> 0, default: 30.0)
        :param windage_factor: Windage coefficient override (default: pipeline default 0.03)
        :param max_gap_minutes: Max environmental gap override (default: pipeline default 60.0)
        :param env_provider: EnvironmentalDataProvider override
        :return: DriftPipelineResult instance
        :raises ValueError: For invalid coordinates or timezone-naive timestamps
        """
        return run_drift_pipeline(
            latitude=latitude,
            longitude=longitude,
            detection_timestamp=detection_timestamp,
            spill_id=spill_id,
            duration_hours=duration_hours,
            step_minutes=step_minutes,
            windage_factor=windage_factor if windage_factor is not None else self.default_windage_factor,
            max_gap_minutes=max_gap_minutes if max_gap_minutes is not None else self.default_max_gap_minutes,
            env_provider=env_provider or self.env_provider,
        )


def run_drift_pipeline(
    latitude: float,
    longitude: float,
    detection_timestamp: datetime,
    spill_id: str = "SPILL_UNKNOWN",
    duration_hours: float = 2.0,
    step_minutes: float = 30.0,
    windage_factor: float = 0.03,
    max_gap_minutes: Optional[float] = 60.0,
    env_provider: Optional[EnvironmentalDataProvider] = None,
) -> DriftPipelineResult:
    """
    Orchestrates backward tracking and probable origin estimation for a detected oil spill.

    :param latitude: Detected spill latitude [-90, 90]
    :param longitude: Detected spill longitude [-180, 180]
    :param detection_timestamp: Detection timestamp (must be timezone-aware)
    :param spill_id: Optional unique identifier for the spill
    :param duration_hours: Backward tracking duration in hours (>= 0)
    :param step_minutes: Simulation step size in minutes (> 0)
    :param windage_factor: Windage coefficient (>= 0)
    :param max_gap_minutes: Maximum allowable gap for environmental lookup in minutes
    :param env_provider: EnvironmentalDataProvider instance (defaults to SyntheticEnvironmentalProvider)
    :return: DriftPipelineResult instance
    :raises ValueError: For invalid input coordinates or timezone-naive timestamps
    """
    # STEP 1: Input Validation
    validate_coordinates(latitude, longitude)

    if detection_timestamp is None or detection_timestamp.tzinfo is None or detection_timestamp.tzinfo.utcoffset(detection_timestamp) is None:
        raise ValueError("detection_timestamp must be timezone-aware.")

    if duration_hours is None or isinstance(duration_hours, bool) or not isinstance(duration_hours, (int, float)):
        raise ValueError("duration_hours must be a numeric value.")
    if float(duration_hours) < 0:
        raise ValueError(f"duration_hours cannot be negative, got {duration_hours}")

    if step_minutes is None or isinstance(step_minutes, bool) or not isinstance(step_minutes, (int, float)):
        raise ValueError("step_minutes must be a numeric value.")
    if float(step_minutes) <= 0:
        raise ValueError(f"step_minutes must be > 0, got {step_minutes}")

    wf = float(windage_factor)
    if wf < 0:
        raise ValueError(f"windage_factor cannot be negative, got {windage_factor}")

    provider = env_provider or SyntheticEnvironmentalProvider()

    # STEP 2: Backward Trajectory Simulation
    try:
        backward_trajectory = simulate_backward_trajectory(
            spill_latitude=latitude,
            spill_longitude=longitude,
            spill_timestamp=detection_timestamp,
            duration_hours=duration_hours,
            step_minutes=step_minutes,
            env_provider=provider,
            windage_factor=wf,
            max_gap_minutes=max_gap_minutes,
        )
    except EnvironmentalDataUnavailableError as e:
        return DriftPipelineResult(
            spill_id=spill_id,
            detected_latitude=float(latitude),
            detected_longitude=float(longitude),
            detection_timestamp=detection_timestamp,
            probable_latitude=float(latitude),
            probable_longitude=float(longitude),
            estimated_release_time=detection_timestamp,
            trajectory_points_used=0,
            status="environmental_data_unavailable",
            message=f"Environmental data unavailable: {str(e)}",
            backward_trajectory=[],
        )

    if len(backward_trajectory) == 0:
        return DriftPipelineResult(
            spill_id=spill_id,
            detected_latitude=float(latitude),
            detected_longitude=float(longitude),
            detection_timestamp=detection_timestamp,
            probable_latitude=float(latitude),
            probable_longitude=float(longitude),
            estimated_release_time=detection_timestamp,
            trajectory_points_used=0,
            status="environmental_data_unavailable",
            message="Environmental data unavailable for backward trajectory simulation.",
            backward_trajectory=[],
        )

    # STEP 3: Probable Origin Estimation
    try:
        origin_result = estimate_probable_origin(
            backward_trajectory=backward_trajectory,
            spill_id=spill_id,
        )
        status = origin_result.status
        message = origin_result.message
        prob_lat = origin_result.candidate_latitude
        prob_lon = origin_result.candidate_longitude
        est_release_time = origin_result.estimated_release_time
        pts_used = origin_result.trajectory_points_used
    except InsufficientTrajectoryError as e:
        status = "insufficient_trajectory"
        message = str(e)
        prob_lat = float(latitude)
        prob_lon = float(longitude)
        est_release_time = detection_timestamp
        pts_used = len(backward_trajectory)

    # STEP 4: Consolidate and Return Pipeline Result
    return DriftPipelineResult(
        spill_id=spill_id,
        detected_latitude=float(latitude),
        detected_longitude=float(longitude),
        detection_timestamp=detection_timestamp,
        probable_latitude=prob_lat,
        probable_longitude=prob_lon,
        estimated_release_time=est_release_time,
        trajectory_points_used=pts_used,
        status=status,
        message=message,
        backward_trajectory=backward_trajectory,
    )
