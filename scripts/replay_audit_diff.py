#!/usr/bin/env python3
"""Decision-level diff between a replay run and the live bot's audit trail.

For each (timestamp_bucket, coin, action) bucket, compares:
  - live      : rows recorded by the production bot during the same window
  - replay    : rows recorded by the replay harness against the same prices

Matches are recorded if the bot took the same action on the same coin
within ``--match-window`` seconds. Mismatches are bucketed by reason:
  live_only  : the live bot acted but the replay didn't
  replay_only: the replay acted but the live bot didn't
  both       : matched (timestamps, action, coin all align)

This is the strongest sanity check we can run on the harness: the live
bot saw real market state and made decisions that landed in the
audit_trail. If the replay sees the same prices and stubs everything
else neutrally, the decisions it makes should overlap meaningfully
with the live ones -- though not 100% because the stubs flatten polymarket
/ options / macro signals the live bot consulted.

Usage:
    python scripts/replay_audit_diff.py \\
        --live data/bot.db \\
        --replay data/replay_<run_id>.db \\
        --start 2025-08-01 --end 2025-08-08 \\
        --report-out reports/audit_diff.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("audit-diff")


@dataclass
class AuditRow:
    timestamp_iso: str
    action: str
    coin: Optional[str]
    side: Optional[str]
    source: Optional[str]
    price: Optional[float]
    details: Dict

    @property
    def ts_epoch(self) -> float:
        try:
            return datetime.fromisoformat(
                self.timestamp_iso.replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            return 0.0


def _norm(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _match_key(row: AuditRow) -> Tuple[str, str, str]:
    """Decision identity for replay-vs-live matching.

    Side matters. A replay that approves a BTC long should not be credited for
    matching a live BTC short just because the action and timestamp line up.
    """
    return (_norm(row.coin), _norm(row.action), _norm(row.side))


def _reason_for(row: AuditRow) -> str:
    """Best-effort mismatch reason extraction from old and new audit schemas."""
    details = row.details if isinstance(row.details, dict) else {}
    for key in (
        "reason",
        "rejection_reason",
        "firewall_reason",
        "status_reason",
        "decision_reason",
    ):
        value = details.get(key)
        if value:
            return str(value)[:120]
    for nested_key in ("firewall", "decision", "result"):
        nested = details.get(nested_key)
        if isinstance(nested, dict):
            for key in ("reason", "rejection_reason", "status_reason"):
                value = nested.get(key)
                if value:
                    return str(value)[:120]
    return (row.action or "?")[:120]


def _load_audit(db_path: str, start_iso: str, end_iso: str) -> List[AuditRow]:
    """Read audit_trail rows in [start_iso, end_iso). Tolerates missing table."""
    if not os.path.exists(db_path):
        logger.warning("DB not found: %s", db_path)
        return []
    rows: List[AuditRow] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                """SELECT timestamp, action, coin, side, source, price, details
                   FROM audit_trail
                   WHERE timestamp >= ? AND timestamp < ?
                   ORDER BY timestamp""",
                (start_iso, end_iso),
            )
        except sqlite3.OperationalError as e:
            logger.warning("audit_trail unavailable in %s: %s", db_path, e)
            return []
        for r in cur:
            try:
                details = json.loads(r["details"] or "{}")
            except (TypeError, json.JSONDecodeError):
                details = {}
            rows.append(AuditRow(
                timestamp_iso=r["timestamp"] or "",
                action=r["action"] or "",
                coin=r["coin"],
                side=r["side"],
                source=r["source"],
                price=r["price"],
                details=details,
            ))
    return rows


@dataclass
class DiffResult:
    total_live: int = 0
    total_replay: int = 0
    matched: int = 0
    live_only: int = 0
    replay_only: int = 0

    by_action_live: Counter = field(default_factory=Counter)
    by_action_replay: Counter = field(default_factory=Counter)
    by_action_matched: Counter = field(default_factory=Counter)

    by_coin_live: Counter = field(default_factory=Counter)
    by_coin_replay: Counter = field(default_factory=Counter)
    by_coin_matched: Counter = field(default_factory=Counter)

    by_source_live: Counter = field(default_factory=Counter)
    by_source_replay: Counter = field(default_factory=Counter)

    live_only_reasons: Counter = field(default_factory=Counter)
    replay_only_reasons: Counter = field(default_factory=Counter)

    sample_mismatches: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "totals": {
                "live": self.total_live,
                "replay": self.total_replay,
                "matched": self.matched,
                "live_only": self.live_only,
                "replay_only": self.replay_only,
                "match_rate_live_pct": (
                    round(self.matched / max(self.total_live, 1) * 100, 1)
                ),
                "match_rate_replay_pct": (
                    round(self.matched / max(self.total_replay, 1) * 100, 1)
                ),
            },
            "by_action": {
                "live":   dict(self.by_action_live),
                "replay": dict(self.by_action_replay),
                "matched": dict(self.by_action_matched),
            },
            "by_coin": {
                "live":   dict(self.by_coin_live),
                "replay": dict(self.by_coin_replay),
                "matched": dict(self.by_coin_matched),
            },
            "by_source": {
                "live":   dict(self.by_source_live),
                "replay": dict(self.by_source_replay),
            },
            "reject_reasons": {
                "live_only":   dict(self.live_only_reasons),
                "replay_only": dict(self.replay_only_reasons),
            },
            "sample_mismatches": self.sample_mismatches[:40],
        }


def diff_audit_trails(
    live: List[AuditRow], replay: List[AuditRow], match_window_s: float = 600.0,
) -> DiffResult:
    """Greedy time-sorted matcher using side-aware decision keys.

    Two rows match if action, coin, and side all match and |dt| is within the
    configured window. For each live row we pick the closest unmatched replay
    row in that key bucket. Older replay rows skipped while seeking a match are
    still counted as replay_only later; they are never silently consumed.
    """
    result = DiffResult(total_live=len(live), total_replay=len(replay))

    for rrow in replay:
        result.by_action_replay[rrow.action] += 1
        if rrow.coin:
            result.by_coin_replay[rrow.coin] += 1
        if rrow.source:
            result.by_source_replay[rrow.source] += 1

    # Index by (coin, action, side) and sort each bucket by timestamp.
    buckets_replay: Dict[Tuple[str, str, str], List[Tuple[int, AuditRow]]] = defaultdict(list)
    for idx, r in enumerate(replay):
        buckets_replay[_match_key(r)].append((idx, r))
    for k in buckets_replay:
        buckets_replay[k].sort(key=lambda item: item[1].ts_epoch)

    first_candidate = {k: 0 for k in buckets_replay}
    matched_replay: set[int] = set()

    for lrow in live:
        result.by_action_live[lrow.action] += 1
        if lrow.coin:
            result.by_coin_live[lrow.coin] += 1
        if lrow.source:
            result.by_source_live[lrow.source] += 1

        key = _match_key(lrow)
        bucket = buckets_replay.get(key)
        if not bucket:
            result.live_only += 1
            reason = _reason_for(lrow)
            result.live_only_reasons[reason] += 1
            if len(result.sample_mismatches) < 40:
                result.sample_mismatches.append({
                    "kind": "live_only", "row": _row_to_dict(lrow),
                })
            continue

        # Advance to the first replay row that could still match this live row.
        # Rows before that pointer remain unmatched and will be counted as
        # replay_only at the end.
        lts = lrow.ts_epoch
        idx = first_candidate.get(key, 0)
        while idx < len(bucket):
            _, rrow = bucket[idx]
            if rrow.ts_epoch < lts - match_window_s:
                idx += 1
                continue
            break
        first_candidate[key] = idx

        best_replay_idx: Optional[int] = None
        best_abs_dt: Optional[float] = None
        scan = idx
        while scan < len(bucket):
            replay_idx, rrow = bucket[scan]
            dt = rrow.ts_epoch - lts
            if dt > match_window_s:
                break
            if replay_idx not in matched_replay:
                abs_dt = abs(dt)
                if best_abs_dt is None or abs_dt < best_abs_dt:
                    best_abs_dt = abs_dt
                    best_replay_idx = replay_idx
            scan += 1

        if best_replay_idx is not None:
            matched_replay.add(best_replay_idx)
            result.matched += 1
            result.by_action_matched[lrow.action] += 1
            if lrow.coin:
                result.by_coin_matched[lrow.coin] += 1
        else:
            result.live_only += 1
            reason = _reason_for(lrow)
            result.live_only_reasons[reason] += 1
            if len(result.sample_mismatches) < 40:
                result.sample_mismatches.append({
                    "kind": "live_only", "row": _row_to_dict(lrow),
                })

    # Any replay rows not matched are replay_only.
    for bucket in buckets_replay.values():
        for replay_idx, rrow in bucket:
            if replay_idx in matched_replay:
                continue
            result.replay_only += 1
            reason = _reason_for(rrow)
            result.replay_only_reasons[reason] += 1
            if len(result.sample_mismatches) < 40:
                result.sample_mismatches.append({
                    "kind": "replay_only", "row": _row_to_dict(rrow),
                })

    return result


def _row_to_dict(r: AuditRow) -> Dict:
    return {
        "timestamp": r.timestamp_iso,
        "action": r.action,
        "coin": r.coin,
        "side": r.side,
        "source": r.source,
        "price": r.price,
        "details_reason": r.details.get("reason"),
    }


def _print_summary(diff: DiffResult, args: argparse.Namespace) -> None:
    d = diff.to_dict()
    print()
    print("=" * 78)
    print("  AUDIT TRAIL DIFF")
    print("=" * 78)
    print(f"  Window: {args.start} -> {args.end}")
    print(f"  Match window: +/- {args.match_window} s")
    print()
    print(f"  Live rows:      {d['totals']['live']:>6}")
    print(f"  Replay rows:    {d['totals']['replay']:>6}")
    print(f"  Matched:        {d['totals']['matched']:>6}  "
          f"({d['totals']['match_rate_live_pct']}% of live, "
          f"{d['totals']['match_rate_replay_pct']}% of replay)")
    print(f"  Live-only:      {d['totals']['live_only']:>6}")
    print(f"  Replay-only:    {d['totals']['replay_only']:>6}")
    print()
    if d["by_action"]["live"]:
        print("  Top live actions:")
        for k, v in sorted(d["by_action"]["live"].items(),
                           key=lambda x: -x[1])[:8]:
            m = d["by_action"]["matched"].get(k, 0)
            print(f"    {k:<28} live={v:>5}  matched={m:>5}")
    print()
    if d["reject_reasons"]["live_only"]:
        print("  Top live-only reasons (replay missed them):")
        for k, v in sorted(d["reject_reasons"]["live_only"].items(),
                           key=lambda x: -x[1])[:8]:
            print(f"    {v:>5}  {k}")
    print()
    if d["reject_reasons"]["replay_only"]:
        print("  Top replay-only reasons (replay invented them):")
        for k, v in sorted(d["reject_reasons"]["replay_only"].items(),
                           key=lambda x: -x[1])[:8]:
            print(f"    {v:>5}  {k}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--live", required=True, help="Path to live bot.db")
    p.add_argument("--replay", required=True, help="Path to replay_<run_id>.db")
    p.add_argument("--start", required=True, help="ISO date or datetime, inclusive")
    p.add_argument("--end", required=True, help="ISO date or datetime, exclusive")
    p.add_argument("--match-window", type=float, default=600.0,
                   help="Tolerance window (seconds) for matching rows. Default 600 (10m).")
    p.add_argument("--min-live-match-rate", type=float, default=0.0,
                   help="Fail if matched/live is below this decimal threshold, e.g. 0.70.")
    p.add_argument("--min-replay-match-rate", type=float, default=0.0,
                   help="Fail if matched/replay is below this decimal threshold, e.g. 0.70.")
    p.add_argument("--report-out", help="JSON output path")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Permit a YYYY-MM-DD shortcut.
    def _normalize(s: str) -> str:
        if len(s) == 10:
            return s + "T00:00:00+00:00"
        return s

    start = _normalize(args.start)
    end = _normalize(args.end)

    live = _load_audit(args.live, start, end)
    replay = _load_audit(args.replay, start, end)

    logger.info("Loaded %d live rows, %d replay rows", len(live), len(replay))
    diff = diff_audit_trails(live, replay, match_window_s=args.match_window)

    out = diff.to_dict()
    out["config"] = {
        "live": args.live,
        "replay": args.replay,
        "start": start,
        "end": end,
        "match_window_s": args.match_window,
    }

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(out, f, indent=2, default=str, sort_keys=True)
        logger.info("Report written to %s", args.report_out)

    _print_summary(diff, args)

    # Exit non-zero if there are NO rows at all (likely a config error).
    if diff.total_live == 0 and diff.total_replay == 0:
        logger.error("No rows on either side -- check --live / --replay / --start / --end")
        return 2
    live_rate = diff.matched / max(diff.total_live, 1)
    replay_rate = diff.matched / max(diff.total_replay, 1)
    if args.min_live_match_rate and live_rate < args.min_live_match_rate:
        logger.error(
            "Live match rate %.1f%% below required %.1f%%",
            live_rate * 100,
            args.min_live_match_rate * 100,
        )
        return 1
    if args.min_replay_match_rate and replay_rate < args.min_replay_match_rate:
        logger.error(
            "Replay match rate %.1f%% below required %.1f%%",
            replay_rate * 100,
            args.min_replay_match_rate * 100,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
