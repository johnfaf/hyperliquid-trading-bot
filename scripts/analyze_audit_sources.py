"""Analyze production source decisions and allocator warmup health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.learning.audit_source_analysis import analyze_audit_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--warmup-days", type=int, default=2)
    parser.add_argument("--warmup-min-rejections", type=int, default=25)
    parser.add_argument("--cleanup-short-copy-keys", action="store_true")
    parser.add_argument("--send-warmup-alerts", action="store_true")
    parser.add_argument(
        "--report-out",
        default="reports/audit_source_analysis.json",
        help="Path for the JSON report.",
    )
    args = parser.parse_args()

    report = analyze_audit_sources(
        days=args.days,
        limit=args.limit,
        coverage_threshold=args.coverage_threshold,
        warmup_days=args.warmup_days,
        warmup_min_rejections=args.warmup_min_rejections,
        cleanup_short_copy_keys=args.cleanup_short_copy_keys,
        send_warmup_alerts=args.send_warmup_alerts,
    )

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    summary = report.get("summary", {})
    print(
        "audit source analysis: "
        f"{summary.get('source_count', 0)} sources, "
        f"{summary.get('approvals', 0)} approvals, "
        f"{summary.get('rejections', 0)} rejections, "
        f"{summary.get('warmup_stuck_sources', 0)} warmup-stuck"
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

