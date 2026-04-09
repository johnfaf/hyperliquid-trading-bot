#!/usr/bin/env python3
"""
Decision-Cycle Replay Harness.

Replays recent decision outcomes from persisted audit + trade records and
produces source/regime attribution summaries for fast operator checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data import database as db  # noqa: E402


def _json_load(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _fetch_audit_decisions(since_iso: str, limit: int, source_filter: str = "") -> List[Dict[str, Any]]:
    source_filter = (source_filter or "").strip().lower()
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, action, coin, side, source, details
            FROM audit_trail
            WHERE timestamp >= ?
              AND action IN ('signal_approved', 'signal_rejected')
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (since_iso, limit),
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["details"] = _json_load(item.get("details"))
        item["source"] = str(item.get("source", "unknown") or "unknown").strip().lower()
        if source_filter and item["source"] != source_filter:
            continue
        results.append(item)
    return results


def _fetch_recent_trades(since_iso: str) -> List[Dict[str, Any]]:
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, coin, side, status, opened_at, closed_at, pnl, metadata
            FROM paper_trades
            WHERE opened_at >= ?
               OR (closed_at IS NOT NULL AND closed_at >= ?)
            ORDER BY opened_at ASC
            """,
            (since_iso, since_iso),
        ).fetchall()
    trades = []
    for row in rows:
        item = dict(row)
        item["metadata"] = _json_load(item.get("metadata"))
        trades.append(item)
    return trades


def build_report(hours: int, limit: int, source: str = "") -> Dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(hours=max(hours, 1))
    since_iso = since.isoformat()

    decisions = _fetch_audit_decisions(since_iso=since_iso, limit=max(limit, 1), source_filter=source)
    trades = _fetch_recent_trades(since_iso=since_iso)

    approved = [d for d in decisions if d.get("action") == "signal_approved"]
    rejected = [d for d in decisions if d.get("action") == "signal_rejected"]

    source_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"approved": 0, "rejected": 0, "total": 0}
    )
    rejection_reasons: Dict[str, int] = defaultdict(int)
    for decision in decisions:
        src = decision.get("source", "unknown")
        source_stats[src]["total"] += 1
        if decision.get("action") == "signal_approved":
            source_stats[src]["approved"] += 1
        else:
            source_stats[src]["rejected"] += 1
            reason = str(decision.get("details", {}).get("reason", "unknown"))
            rejection_reasons[reason] += 1

    executed_by_source: Dict[str, int] = defaultdict(int)
    executed_by_regime: Dict[str, int] = defaultdict(int)
    closed_pnl_by_source: Dict[str, float] = defaultdict(float)
    for trade in trades:
        meta = trade.get("metadata", {})
        src = str(meta.get("source", trade.get("strategy_type", "unknown")) or "unknown").lower()
        regime = str(meta.get("regime", "unknown") or "unknown").lower()
        executed_by_source[src] += 1
        executed_by_regime[regime] += 1
        if str(trade.get("status", "")).lower() == "closed":
            closed_pnl_by_source[src] += float(trade.get("pnl", 0.0) or 0.0)

    total = len(decisions)
    pass_rate = (len(approved) / total) if total else 0.0
    top_rejections = sorted(rejection_reasons.items(), key=lambda item: item[1], reverse=True)[:10]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": config.DB_PATH,
        "window": {
            "hours": hours,
            "since": since_iso,
            "limit": limit,
            "source_filter": source or None,
        },
        "decision_summary": {
            "total": total,
            "approved": len(approved),
            "rejected": len(rejected),
            "pass_rate": round(pass_rate, 4),
            "top_rejection_reasons": [
                {"reason": reason, "count": count} for reason, count in top_rejections
            ],
            "by_source": dict(source_stats),
        },
        "execution_summary": {
            "trades_considered": len(trades),
            "executed_by_source": dict(executed_by_source),
            "executed_by_regime": dict(executed_by_regime),
            "closed_pnl_by_source": {
                key: round(value, 2) for key, value in closed_pnl_by_source.items()
            },
        },
    }


def _print_human(report: Dict[str, Any]) -> None:
    window = report["window"]
    decisions = report["decision_summary"]
    execution = report["execution_summary"]

    print("Decision-Cycle Replay")
    print("=" * 80)
    print(f"DB: {report['db_path']}")
    print(f"Window: last {window['hours']}h since {window['since']}")
    if window.get("source_filter"):
        print(f"Source filter: {window['source_filter']}")
    print(
        "Decisions: {total} total | approved={approved} rejected={rejected} pass_rate={pass_rate:.1%}".format(
            total=decisions["total"],
            approved=decisions["approved"],
            rejected=decisions["rejected"],
            pass_rate=float(decisions["pass_rate"]),
        )
    )
    print("Top rejections:")
    top_rejects = decisions.get("top_rejection_reasons", [])
    if top_rejects:
        for item in top_rejects:
            print(f"  - {item['count']:>4}  {item['reason']}")
    else:
        print("  - none")

    print("By source:")
    by_source = decisions.get("by_source", {})
    if by_source:
        for source, stats in sorted(by_source.items(), key=lambda item: item[0]):
            print(
                f"  - {source:<20} total={stats['total']:<4} "
                f"approved={stats['approved']:<4} rejected={stats['rejected']:<4}"
            )
    else:
        print("  - none")

    print("Executed trades by regime:")
    by_regime = execution.get("executed_by_regime", {})
    if by_regime:
        for regime, count in sorted(by_regime.items(), key=lambda item: item[1], reverse=True):
            print(f"  - {regime:<16} {count}")
    else:
        print("  - none")

    print("Closed PnL by source:")
    pnl_map = execution.get("closed_pnl_by_source", {})
    if pnl_map:
        for source, pnl in sorted(pnl_map.items(), key=lambda item: item[1], reverse=True):
            print(f"  - {source:<20} {pnl:+.2f}")
    else:
        print("  - none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay recent decision-cycle outcomes from DB")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours (default: 24)")
    parser.add_argument("--limit", type=int, default=5000, help="Max audit decision rows to replay")
    parser.add_argument("--source", type=str, default="", help="Optional source filter (e.g. copy_trade)")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--output", type=str, default="", help="Optional path to write JSON report")
    args = parser.parse_args()

    report = build_report(hours=args.hours, limit=args.limit, source=args.source)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
