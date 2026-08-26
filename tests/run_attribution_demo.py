"""
Standalone Python demonstration script for the Vessel Attribution module (Member 4).
Runs attribution analysis on synthetic mock AIS data and prints a formatted report.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from attribution.mock_data import create_sample_spill_origin, generate_mock_vessel_trajectories
from attribution.service import VesselAttributionService


def main():
    print("=" * 65)
    print("      SIH 2026 PS 26143 - VESSEL ATTRIBUTION MODULE DEMO      ")
    print("=" * 65)
    print()

    # 1. Generate sample spill origin and mock trajectories
    spill = create_sample_spill_origin()
    trajectories = generate_mock_vessel_trajectories(spill)

    # 2. Run attribution analysis
    response = VesselAttributionService.analyze_attribution(spill, trajectories)

    # 3. Print Incident Overview (1 to 4)
    print("--- INCIDENT OVERVIEW ---")
    print(f"1. Spill Location         : Lat {response.spill_origin.latitude} deg N, Lon {response.spill_origin.longitude} deg E")
    print(f"2. Spill Estimated Time   : {response.spill_origin.estimated_release_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"3. Total Vessels Evaluated: {response.total_vessels_evaluated}")
    print(f"4. Candidate Vessels      : {len(response.candidate_vessels)} (passed spatial/temporal filtering)")
    print()

    # 4. Print Candidate Vessels (5 to 12)
    print("--- RANKED CANDIDATE VESSELS ---")
    for candidate in response.candidate_vessels:
        print("-" * 65)
        print(f"5. Rank                 : #{candidate.rank}")
        print(f"6. Vessel Name          : {candidate.vessel_name} ({candidate.vessel_type})")
        print(f"7. MMSI                 : {candidate.mmsi} (ID: {candidate.vessel_id})")
        print(f"8. Distance from Spill  : {candidate.distance_km:.2f} km")
        print(f"9. Time Difference      : {candidate.time_difference_minutes:.1f} minutes")
        print(f"10. Final Score         : {candidate.final_score:.2f} / 100.0")
        print(f"11. Classification      : \"{candidate.classification}\"")
        print(f"12. Explanation         :\n    {candidate.explanation}")
        print()

    print("=" * 65)
    print("                     END OF ATTRIBUTION REPORT                 ")
    print("=" * 65)


if __name__ == "__main__":
    main()
