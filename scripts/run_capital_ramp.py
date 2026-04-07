"""Run the controlled capital-ramp review on demand."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.analysis.capital_ramp import (  # noqa: E402
    CapitalRampManager,
    build_capital_ramp_config,
)
from src.data import database as db  # noqa: E402
from src.signals.capital_governor import CapitalGovernor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the controlled capital-ramp review.")
    parser.add_argument(
        "--cycle-count",
        type=int,
        default=None,
        help="Optional cycle number to associate with the ramp run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a run even if the interval has not elapsed.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the ramp payload.",
    )
    args = parser.parse_args(argv)

    db.init_db()
    manager = CapitalRampManager(
        build_capital_ramp_config(config),
        capital_governor=CapitalGovernor({"enabled": True}),
    )
    result = manager.run(cycle_count=args.cycle_count, force=args.force)

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    limits = result.get("limits", {}) or {}
    print("Capital ramp review complete")
    print(f"Run ID: {result.get('run_id', 'n/a')}")
    print(f"Status: {result.get('status', 'warming_up')}")
    print(f"Applied stage: {result.get('applied_stage', 'bootstrap')}")
    print(f"Approved stage: {result.get('approved_stage', 'bootstrap')}")
    print(f"Recommended stage: {result.get('recommended_stage', 'bootstrap')}")
    print(f"Deployable: {result.get('deployable', False)}")
    print(f"Max order cap: ${float(limits.get('effective_max_order_usd', 0.0) or 0.0):.2f}")
    if args.output:
        print(f"Output: {os.path.abspath(args.output)}")
    return 0 if result.get("deployable", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
