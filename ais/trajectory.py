"""Trajectory reconstruction and vessel position sequence grouping."""

from collections import defaultdict
from typing import Dict, List, Optional

from .schemas import AISPoint, VesselTrajectory


def build_trajectories(records: List[AISPoint]) -> Dict[str, VesselTrajectory]:
    """Group AIS observations by vessel_id and sort chronologically.

    Args:
        records: Cleaned list of AISPoint objects.

    Returns:
        Dictionary mapping vessel_id to its VesselTrajectory object.
    """
    grouped_points: Dict[str, List[AISPoint]] = defaultdict(list)

    for pt in records:
        grouped_points[pt.vessel_id].append(pt)

    trajectories: Dict[str, VesselTrajectory] = {}
    for vessel_id, points in grouped_points.items():
        # Ensure chronological ordering
        sorted_points = sorted(points, key=lambda p: p.timestamp)
        trajectories[vessel_id] = VesselTrajectory(
            vessel_id=vessel_id,
            points=sorted_points,
        )

    return trajectories


def get_vessel_trajectory(
    trajectories: Dict[str, VesselTrajectory],
    vessel_id: str,
) -> Optional[VesselTrajectory]:
    """Retrieve the trajectory sequence for a specific vessel identifier.

    Args:
        trajectories: Mapping of vessel_id to VesselTrajectory.
        vessel_id: The target vessel ID to retrieve.

    Returns:
        VesselTrajectory if found, else None.
    """
    clean_id = str(vessel_id).strip()
    return trajectories.get(clean_id)

