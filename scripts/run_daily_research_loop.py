"""
Run the daily research loop on demand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.analysis.daily_research_loop import run_daily_research_loop  # noqa: E402
from src.data import database as db  # noqa: E402
from src.signals.adaptive_learning import (  # noqa: E402
    AdaptiveLearningManager,
    build_adaptive_learning_config,
)
from src.signals.agent_scoring import AgentScorer  # noqa: E402
from src.signals.calibration import CalibrationTracker  # noqa: E402

try:  # noqa: E402
    from src.signals.alpha_arena import AlphaArena
except Exception:  # pragma: no cover - optional for offline runs
    AlphaArena = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the daily research loop.")
    parser.add_argument(
        "--cycle-count",
        type=int,
        default=None,
        help="Optional cycle number to associate with the research run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a run even if the interval has not elapsed.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the daily research payload.",
    )
    args = parser.parse_args(argv)

    db.init_db()
    arena = None
    if AlphaArena is not None:
        try:
            arena = AlphaArena()
        except Exception:
            arena = None

    adaptive_learning = AdaptiveLearningManager(
        build_adaptive_learning_config(config),
        agent_scorer=AgentScorer(),
        calibration=CalibrationTracker(),
        arena=arena,
    )
    result = run_daily_research_loop(
        adaptive_learning=adaptive_learning,
        cycle_count=args.cycle_count,
        force=args.force,
    )

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    recommendation = result.get("recommendation", {}) or {}
    benchmark = result.get("benchmark", {}) or {}
    gate = benchmark.get("promotion_gate", {}) or {}
    approved = gate.get("approved_profiles", []) or []
    print("Daily research loop complete")
    print(f"Run ID: {result.get('run_id', 'n/a')}")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Recommendation: {recommendation.get('action', 'hold')}")
    print(f"Winner: {gate.get('winner', 'baseline_current')}")
    print(f"Approved challengers: {', '.join(approved) if approved else 'none'}")
    rollback_target = recommendation.get("rollback_target_profile")
    if rollback_target:
        print(f"Rollback target: {rollback_target}")
    if args.output:
        print(f"Output: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
