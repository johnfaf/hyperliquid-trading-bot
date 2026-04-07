"""
Run the adaptive-learning recalibration job on demand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adaptive recalibration job.")
    parser.add_argument(
        "--cycle-count",
        type=int,
        default=None,
        help="Optional cycle number to associate with the recalibration run.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the recalibration payload.",
    )
    args = parser.parse_args()

    db.init_db()
    arena = None
    if AlphaArena is not None:
        try:
            arena = AlphaArena()
        except Exception:
            arena = None

    manager = AdaptiveLearningManager(
        build_adaptive_learning_config(config),
        agent_scorer=AgentScorer(),
        calibration=CalibrationTracker(),
        arena=arena,
    )
    result = manager.run_recalibration(cycle_count=args.cycle_count, force=True)

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    print("Adaptive recalibration complete")
    print(f"Run ID: {result.get('run_id', 'n/a')}")
    print(f"Executed: {result.get('executed', False)}")
    print(f"Transitions: {result.get('transition_count', 0)}")
    print(f"Promoted: {result.get('promoted_count', 0)}")
    print(f"Demoted: {result.get('demoted_count', 0)}")
    print(f"Pending: {result.get('pending_count', 0)}")
    print(f"Held: {result.get('held_count', 0)}")
    if args.output:
        print(f"Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
