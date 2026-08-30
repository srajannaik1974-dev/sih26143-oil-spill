"""
Probable Oil-Spill Origin Estimator.

Analyzes backward surface oil-spill trajectories to estimate candidate historical release
locations and release-time windows.

Scientific Scope & Model Assumptions:
- Baseline SIH Prototype approach:
      The oldest valid reconstructed point in the backward trajectory is selected as
      the primary probable-origin location candidate.
      Its timestamp is selected as the estimated release-time candidate.
- This represents an estimated origin candidate based on available backtracking duration
  and current surface drift physics. It is NOT guaranteed to be the exact physical release location.
- Synthetic data from SyntheticEnvironmentalProvider is DEMO data for testing.
- Real environmental data and AIS vessel correlation are required for operational attribution.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from .geo_utils import validate_coordinates
from .models import TrajectoryPoint


class InsufficientTrajectoryError(Exception):
    """Raised when a backward trajectory is empty or too short for origin estimation."""
    pass


class ProbableOriginResult(BaseModel):
    """
    Model representing probable oil-spill origin candidate estimation results.
    """
    model_config = ConfigDict(extra="forbid")

    spill_id: Optional[str] = None
    candidate_latitude: float
    candidate_longitude: float
    estimated_release_time: datetime
    trajectory_points_used: int
    status: str
    message: str


class ProbableOriginEstimator:
    """
    Estimator for identifying the probable origin location and release-time candidate
    from a backward trajectory.
    """

    def estimate(
        self,
        backward_trajectory: List[TrajectoryPoint],
        spill_id: Optional[str] = None,
        min_points_required: int = 2,
    ) -> ProbableOriginResult:
        """
        Estimates probable origin location and release time from a backward trajectory.

        :param backward_trajectory: Chronological list of backward TrajectoryPoint instances
                                    (trajectory[0] = detection point, trajectory[-1] = oldest point)
        :param spill_id: Optional identifier for the oil spill observation
        :param min_points_required: Minimum trajectory points required for valid estimation (default: 2)
        :return: ProbableOriginResult instance
        :raises InsufficientTrajectoryError: If trajectory is empty or None
        """
        return estimate_probable_origin(
            backward_trajectory=backward_trajectory,
            spill_id=spill_id,
            min_points_required=min_points_required,
        )


def estimate_probable_origin(
    backward_trajectory: List[TrajectoryPoint],
    spill_id: Optional[str] = None,
    min_points_required: int = 2,
) -> ProbableOriginResult:
    """
    Analyzes backward trajectory points to extract the oldest reconstructed point as the
    probable origin candidate.

    :param backward_trajectory: List of TrajectoryPoint instances ordered from detection backward
    :param spill_id: Optional spill identifier string
    :param min_points_required: Minimum points required for full candidate estimation (default: 2)
    :return: ProbableOriginResult instance
    :raises InsufficientTrajectoryError: If trajectory is None or empty.
    """
    if backward_trajectory is None or not isinstance(backward_trajectory, list) or len(backward_trajectory) == 0:
        raise InsufficientTrajectoryError("Backward trajectory is empty or None. Cannot estimate origin.")

    points_count = len(backward_trajectory)

    if points_count < min_points_required:
        single_point = backward_trajectory[0]
        validate_coordinates(single_point.latitude, single_point.longitude)
        return ProbableOriginResult(
            spill_id=spill_id,
            candidate_latitude=single_point.latitude,
            candidate_longitude=single_point.longitude,
            estimated_release_time=single_point.timestamp,
            trajectory_points_used=points_count,
            status="insufficient_trajectory",
            message=f"Trajectory contains only {points_count} point. Minimum {min_points_required} points required for origin candidate estimation.",
        )

    # The oldest valid backward trajectory point is the last element in the list
    oldest_point = backward_trajectory[-1]
    validate_coordinates(oldest_point.latitude, oldest_point.longitude)

    if oldest_point.timestamp is None or oldest_point.timestamp.tzinfo is None:
        raise ValueError("Trajectory point timestamps must be timezone-aware.")

    return ProbableOriginResult(
        spill_id=spill_id,
        candidate_latitude=oldest_point.latitude,
        candidate_longitude=oldest_point.longitude,
        estimated_release_time=oldest_point.timestamp,
        trajectory_points_used=points_count,
        status="origin_estimated",
        message="Probable origin candidate estimated from the oldest valid backward trajectory point.",
    )
