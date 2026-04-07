"""Run the shadow-certification pack on demand."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis.shadow_certification import run_shadow_certification  # noqa: E402
from src.analysis.shadow_tracker import ShadowTracker  # noqa: E402
from src.data import database as db  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the shadow certification pack.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the certification payload.",
    )
    args = parser.parse_args(argv)

    db.init_db()
    tracker = ShadowTracker()
    report = run_shadow_certification(shadow_tracker=tracker)

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    print("Shadow certification complete")
    print(f"Run ID: {report.get('run_id', 'n/a')}")
    print(f"Status: {report.get('status', 'warming_up')}")
    print(f"Certified: {report.get('certified', False)}")
    print(f"Summary: {report.get('summary', 'n/a')}")
    print(
        "Top blockers: "
        + (", ".join((report.get("blocked_entry_reasons", {}) or {}).keys()) or "none")
    )
    if args.output:
        print(f"Output: {os.path.abspath(args.output)}")
    return 0 if report.get("certified", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
