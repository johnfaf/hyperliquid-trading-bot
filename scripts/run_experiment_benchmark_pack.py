"""
Run the phase-6 experiment benchmark pack over stored decision research cycles.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.analysis.experiment_discipline import run_experiment_benchmark_pack  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiment benchmark pack.")
    parser.add_argument(
        "--limit-cycles",
        type=int,
        default=config.EXPERIMENT_REPORT_LIMIT_CYCLES,
        help="Number of recent decision-research cycles to replay.",
    )
    parser.add_argument(
        "--oos-ratio",
        type=float,
        default=config.EXPERIMENT_OOS_RATIO,
        help="Fraction of cycles reserved for out-of-sample evaluation.",
    )
    parser.add_argument(
        "--output",
        default=config.EXPERIMENT_BENCHMARK_REPORT_PATH,
        help="Where to write the benchmark JSON report.",
    )
    args = parser.parse_args()

    report = run_experiment_benchmark_pack(
        limit_cycles=args.limit_cycles,
        out_of_sample_ratio=args.oos_ratio,
    )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    winner = report.get("promotion_gate", {}).get("winner", "baseline_current")
    approved = report.get("promotion_gate", {}).get("approved_profiles", [])
    print("Experiment benchmark complete")
    print(f"Cycles replayed: {report.get('cycle_count', 0)}")
    print(f"OOS cycles: {report.get('out_of_sample_cycles', 0)}")
    print(f"Winner: {winner}")
    print(f"Approved challengers: {', '.join(approved) if approved else 'none'}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
