"""End-to-end AIS candidate vessel identification pipeline and CLI."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import AISConfig, DEFAULT_CONFIG
from .filtering import filter_by_distance, filter_by_time
from .loader import load_and_validate_ais_csv, load_spill_input
from .ranking import group_by_vessel, rank_candidates


def run_ais_pipeline(
    spill_input_path: Optional[Union[str, Path]] = None,
    ais_csv_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[AISConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the full AIS candidate identification pipeline.

    Pipeline stages:
      1. Load mock or real spill incident coordinates & estimated timestamp.
      2. Load and validate raw AIS CSV records (rejecting malformed coordinates/timestamps).
      3. Apply temporal filtering around the spill timestamp.
      4. Apply spatial filtering using the Haversine distance formula.
      5. Group candidate records by vessel MMSI and aggregate proximity metrics.
      6. Rank candidate vessels by transparent candidate score.
      7. Save formatted JSON output to disk.

    Args:
        spill_input_path: Path to spill JSON file (defaults to config.default_spill_path).
        ais_csv_path: Path to AIS CSV dataset (defaults to config.default_ais_path).
        output_path: Destination path for result JSON (defaults to config.default_output_path).
        config: Custom AISConfig instance or None for defaults.
        verbose: If True, prints formatted summary to stdout.

    Returns:
        Dict[str, Any]: Complete analysis result containing spill, parameters, and candidates.
    """
    cfg = config or DEFAULT_CONFIG
    spill_file = Path(spill_input_path) if spill_input_path else cfg.default_spill_path
    ais_file = Path(ais_csv_path) if ais_csv_path else cfg.default_ais_path
    out_file = Path(output_path) if output_path else cfg.default_output_path

    if verbose:
        print("=" * 60)
        print("  OIL SPILL INVESTIGATION -- AIS CANDIDATE IDENTIFICATION  ")
        print("=" * 60)
        print(f"Loading spill incident from: {spill_file}")

    # 1. Load spill incident
    spill = load_spill_input(spill_file)
    if verbose:
        print(f"Spill Coordinates: Lat {spill.latitude:.4f}, Lon {spill.longitude:.4f}")
        print(f"Spill Estimated UTC Time: {spill.to_dict()['estimated_time']}")
        print("-" * 60)
        print(f"Loading AIS dataset from: {ais_file}...")

    # 2. Load & validate AIS CSV
    raw_records, validation_errors = load_and_validate_ais_csv(ais_file)
    if verbose:
        print(f"AIS records loaded: {len(raw_records)}")
        if validation_errors:
            print(f"Validation notices: {len(validation_errors)} records skipped or flagged")

    # 3. Apply Temporal Filter
    time_filtered = filter_by_time(raw_records, spill.timestamp, cfg.time_window_minutes)
    if verbose:
        print(f"Applying time filter (+/-{cfg.time_window_minutes:.0f} min)...")
        print(f"Records remaining: {len(time_filtered)}")

    # 4. Apply Spatial Filter (Haversine)
    spatially_filtered = filter_by_distance(
        time_filtered, spill.latitude, spill.longitude, cfg.search_radius_km
    )
    if verbose:
        print(f"Applying {cfg.search_radius_km:.1f} km spatial filter (Haversine)...")
        print(f"Records remaining: {len(spatially_filtered)}")

    # 5. Group by MMSI & Compute candidate scores
    candidates = group_by_vessel(spatially_filtered, cfg)
    if verbose:
        print(f"Unique candidate vessels: {len(candidates)}")

    # 6. Rank candidates
    top_candidates = rank_candidates(candidates, top_n=cfg.top_n_candidates)

    if verbose:
        print("-" * 60)
        if top_candidates:
            print(f"Top candidates (up to {cfg.top_n_candidates}):\n")
            for rank_idx, cand in enumerate(top_candidates, start=1):
                name_str = cand.vessel_name if cand.vessel_name else "Unknown Name"
                type_str = f" ({cand.vessel_type})" if cand.vessel_type else ""
                print(f"{rank_idx}. {name_str}{type_str}")
                print(f"   MMSI: {cand.mmsi}")
                if cand.imo:
                    print(f"   IMO: {cand.imo}")
                print(f"   Closest approach: {cand.minimum_distance_km:.2f} km @ {cand.closest_record_time}")
                print(f"   Time difference: {cand.time_difference_minutes:.1f} min")
                print(f"   Observations: {cand.observations} AIS records")
                print(f"   Candidate Score: {cand.candidate_score:.1f}")
                print()
        else:
            print("No candidate vessels found within specified spatial and temporal thresholds.\n")

    # 7. Construct output JSON payload
    result_payload: Dict[str, Any] = {
        "spill": spill.to_dict(),
        "search_parameters": {
            "radius_km": cfg.search_radius_km,
            "time_window_minutes": cfg.time_window_minutes,
        },
        "candidate_vessels": [cand.to_dict() for cand in top_candidates],
    }

    # Ensure output directory exists and save file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    if verbose:
        print("AIS analysis completed successfully.")
        print(f"Output saved to: {out_file}")
        print("=" * 60)

    return result_payload


def main() -> None:
    """CLI entrypoint for running the AIS pipeline directly."""
    parser = argparse.ArgumentParser(
        description="Search and rank AIS candidate vessels near an oil spill incident."
    )
    parser.add_argument(
        "--spill",
        type=str,
        default=None,
        help="Path to spill incident JSON file (e.g. data/mock_spill.json)",
    )
    parser.add_argument(
        "--ais",
        type=str,
        default=None,
        help="Path to AIS CSV dataset (e.g. data/synthetic_ais.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for output JSON (e.g. output/ais_result.json)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="Search radius in kilometers (default: 10.0)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30.0,
        help="Time window in minutes (default: 30.0)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top candidates to return (default: 5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose terminal output",
    )

    args = parser.parse_args()

    config = AISConfig(
        search_radius_km=args.radius,
        time_window_minutes=args.window,
        top_n_candidates=args.top_n,
    )

    run_ais_pipeline(
        spill_input_path=args.spill,
        ais_csv_path=args.ais,
        output_path=args.output,
        config=config,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
