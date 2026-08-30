"""
Synthetic mock data generators for standalone testing and demonstration of the Vessel Attribution module.
Allows Member 4 to run, verify, and demonstrate attribution analysis independently.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from .schemas import SpillOriginInput, AISTrajectoryRecord, AISPosition


def create_sample_spill_origin() -> SpillOriginInput:
    """Return a realistic sample oil spill origin near Bombay High off the Mumbai coast."""
    base_time = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
    return SpillOriginInput(
        latitude=19.4167,
        longitude=71.3333,
        estimated_release_time=base_time,
        max_search_radius_km=50.0,
        max_time_window_hours=24.0
    )


def generate_mock_vessel_trajectories(spill: SpillOriginInput) -> List[AISTrajectoryRecord]:
    """
    Generate synthetic vessel trajectories representing different candidate scenarios:
    1. Vessel A (High Correlation Candidate): Passed within ~1.5 km of spill origin 15 mins prior to release, slow speed (4.5 knots).
    2. Vessel B (Medium Correlation Candidate): Passed within ~8.0 km of spill origin 2 hours prior to release, normal transit speed (12.0 knots).
    3. Vessel C (Low Correlation Candidate): Passed within ~28.0 km of spill origin 5 hours after release.
    4. Vessel D (Filtered Out): Outside spatial search radius (> 65 km away).
    """
    spill_time = spill.estimated_release_time
    spill_lat = spill.latitude
    spill_lon = spill.longitude

    # Vessel A - Highly correlated candidate (Tanker slowing down very close to spill point)
    vessel_a_positions = []
    for i in range(-5, 6):
        t = spill_time + timedelta(minutes=i * 15)
        # Passes within ~1.2 km of spill origin at t=spill_time - 15m
        lat_offset = (i * 0.005) + 0.005
        lon_offset = (i * 0.004) + 0.004
        vessel_a_positions.append(
            AISPosition(
                timestamp=t,
                latitude=spill_lat + lat_offset,
                longitude=spill_lon + lon_offset,
                speed_knots=4.5 if abs(i) <= 2 else 11.0,
                heading_deg=135.0,
                course_over_ground=132.0
            )
        )

    vessel_a = AISTrajectoryRecord(
        vessel_id="VESSEL-A-001",
        mmsi="419000101",
        vessel_name="Ocean Titan",
        vessel_type="Crude Oil Tanker",
        positions=vessel_a_positions
    )

    # Vessel B - Medium correlation candidate (Transit cargo ship ~8 km away, 2 hours prior)
    vessel_b_positions = []
    for i in range(-6, 7):
        t = spill_time + timedelta(hours=-2, minutes=i * 20)
        lat_offset = 0.07 + (i * 0.008)
        lon_offset = 0.05 + (i * 0.006)
        vessel_b_positions.append(
            AISPosition(
                timestamp=t,
                latitude=spill_lat + lat_offset,
                longitude=spill_lon + lon_offset,
                speed_knots=13.2,
                heading_deg=210.0,
                course_over_ground=208.0
            )
        )

    vessel_b = AISTrajectoryRecord(
        vessel_id="VESSEL-B-002",
        mmsi="419000102",
        vessel_name="Pacific Voyager",
        vessel_type="Container Ship",
        positions=vessel_b_positions
    )

    # Vessel C - Low correlation candidate (~28 km away, 5 hours after spill)
    vessel_c_positions = []
    for i in range(-4, 5):
        t = spill_time + timedelta(hours=5, minutes=i * 30)
        lat_offset = 0.25 + (i * 0.01)
        lon_offset = 0.20 + (i * 0.01)
        vessel_c_positions.append(
            AISPosition(
                timestamp=t,
                latitude=spill_lat + lat_offset,
                longitude=spill_lon + lon_offset,
                speed_knots=15.0,
                heading_deg=45.0,
                course_over_ground=45.0
            )
        )

    vessel_c = AISTrajectoryRecord(
        vessel_id="VESSEL-C-003",
        mmsi="419000103",
        vessel_name="Star Clipper",
        vessel_type="Bulk Carrier",
        positions=vessel_c_positions
    )

    # Vessel D - Out of bounds (> 65 km away)
    vessel_d_positions = []
    for i in range(-3, 4):
        t = spill_time + timedelta(minutes=i * 30)
        vessel_d_positions.append(
            AISPosition(
                timestamp=t,
                latitude=spill_lat + 0.85,
                longitude=spill_lon + 0.90,
                speed_knots=10.0,
                heading_deg=90.0,
                course_over_ground=90.0
            )
        )

    vessel_d = AISTrajectoryRecord(
        vessel_id="VESSEL-D-004",
        mmsi="419000104",
        vessel_name="Faraway Trader",
        vessel_type="General Cargo",
        positions=vessel_d_positions
    )

    return [vessel_a, vessel_b, vessel_c, vessel_d]
