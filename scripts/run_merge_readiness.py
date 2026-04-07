"""Run the final merge-readiness package on demand."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _extract_runtime_profile(argv):
    for index, arg in enumerate(argv):
        if arg.startswith("--runtime-profile="):
            return arg.split("=", 1)[1].strip()
        if arg == "--runtime-profile" and index + 1 < len(argv):
            return str(argv[index + 1]).strip()
    return ""


_runtime_profile_override = _extract_runtime_profile(sys.argv[1:])
if _runtime_profile_override:
    os.environ["BOT_RUNTIME_PROFILE"] = _runtime_profile_override

import config  # noqa: E402
from src.analysis.merge_readiness import (  # noqa: E402
    MergeReadinessManager,
    build_merge_readiness_config,
)
from src.data import database as db  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the final merge-readiness package.")
    parser.add_argument(
        "--runtime-profile",
        choices=("paper", "shadow", "live"),
        default=None,
        help="Apply the named runtime profile before config loads.",
    )
    parser.add_argument(
        "--cycle-count",
        type=int,
        default=None,
        help="Optional cycle number to associate with the readiness run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a run even if the interval has not elapsed.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the readiness payload.",
    )
    parser.add_argument(
        "--skip-regression-suite",
        action="store_true",
        help="Skip the stable regression suite step.",
    )
    parser.add_argument(
        "--skip-benchmark-pack",
        action="store_true",
        help="Reuse the latest daily research benchmark instead of replaying a fresh pack.",
    )
    parser.add_argument(
        "--skip-live-preflight",
        action="store_true",
        help="Reuse the current live trader readiness path without forcing a fresh preflight.",
    )
    args = parser.parse_args(argv)

    db.init_db()
    manager = MergeReadinessManager(build_merge_readiness_config(config))
    result = manager.run(
        cycle_count=args.cycle_count,
        force=args.force,
        strict=True,
        run_regression_suite=not args.skip_regression_suite,
        run_benchmark_pack=not args.skip_benchmark_pack,
        force_live_preflight=not args.skip_live_preflight,
    )

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    print("Merge readiness review complete")
    print(f"Run ID: {result.get('run_id', 'n/a')}")
    print(f"Status: {result.get('status', 'hold')}")
    print(f"Deployable for merge: {result.get('deployable_for_merge', False)}")
    print(f"Branch: {(result.get('git', {}) or {}).get('branch', 'unknown')}")
    print(f"Commit: {(result.get('git', {}) or {}).get('short_commit', 'unknown')}")
    print(f"Summary: {result.get('summary', 'n/a')}")
    print(
        "Required blockers: "
        + (
            ", ".join((result.get("metadata", {}) or {}).get("failed_required_checks", []))
            or "none"
        )
    )
    if args.output:
        print(f"Output: {os.path.abspath(args.output)}")
    return 0 if result.get("deployable_for_merge", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
