#!/usr/bin/env python3
"""Pick a production audit window, replay it, and gate on decision diff overlap.

This automates roadmap items 6-8:
  1. Find a 3-day production window with audit_trail rows.
  2. Run the replay harness over that exact window.
  3. Run/attach the audit diff and fail if match-rate is below threshold.

Example:
    python scripts/run_replay_validation.py \
        --live-db data/bot.db \
        --coins BTC,ETH,SOL,HYPE,XRP,DOGE,BNB,ADA,AVAX,LINK \
        --report-out reports/replay_validation.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class AuditWindow:
    start_iso: str
    end_iso: str
    row_count: int


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def find_best_audit_window(
    live_db: str,
    *,
    days: int = 3,
    min_rows: int = 1,
) -> AuditWindow:
    """Return the densest rolling N-day audit_trail window."""
    db_path = Path(live_db)
    if not db_path.exists():
        raise FileNotFoundError(f"live DB not found: {live_db}")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            rows = conn.execute(
                "SELECT timestamp FROM audit_trail WHERE timestamp IS NOT NULL ORDER BY timestamp"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"audit_trail unavailable in {live_db}: {exc}") from exc

    timestamps = [dt for (raw,) in rows if (dt := _parse_ts(raw)) is not None]
    if not timestamps:
        raise RuntimeError(f"audit_trail has no parseable timestamps in {live_db}")

    window_delta = timedelta(days=days)
    best_start = timestamps[0]
    best_count = 0
    right = 0
    for left, start in enumerate(timestamps):
        end = start + window_delta
        while right < len(timestamps) and timestamps[right] < end:
            right += 1
        count = right - left
        if count > best_count:
            best_count = count
            best_start = start

    if best_count < min_rows:
        raise RuntimeError(
            f"best {days}d audit window has {best_count} rows, below min_rows={min_rows}"
        )
    best_end = best_start + window_delta
    return AuditWindow(best_start.isoformat(), best_end.isoformat(), best_count)


def build_replay_command(args: argparse.Namespace, window: AuditWindow) -> list[str]:
    run_id = args.run_id or (
        "validation_"
        + window.start_iso[:10].replace("-", "")
        + "_"
        + window.end_iso[:10].replace("-", "")
    )
    report_out = args.report_out or f"reports/replay_validation_{run_id}.json"
    diff_report_out = args.diff_report_out or f"reports/replay_validation_diff_{run_id}.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_replay.py"),
        "--start",
        window.start_iso,
        "--end",
        window.end_iso,
        "--step",
        args.step,
        "--coins",
        args.coins,
        "--cache-db",
        args.cache_db,
        "--run-id",
        run_id,
        "--diff-live-db",
        args.live_db,
        "--diff-match-window",
        str(args.match_window),
        "--diff-min-live-match-rate",
        str(args.min_live_match_rate),
        "--diff-min-replay-match-rate",
        str(args.min_replay_match_rate),
        "--diff-report-out",
        diff_report_out,
        "--report-out",
        report_out,
    ]
    if args.strategy_snapshot:
        cmd.extend(["--strategy-snapshot", args.strategy_snapshot])
    if args.frozen_xgb_model:
        cmd.extend(["--frozen-xgb-model", args.frozen_xgb_model])
    if args.lax_api:
        cmd.append("--lax-api")
    if args.allow_network:
        cmd.append("--allow-network")
    if args.halt_on_error:
        cmd.append("--halt-on-error")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-db", default="data/bot.db")
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--coins", default="BTC,ETH,SOL,HYPE,XRP,DOGE,BNB,ADA,AVAX,LINK")
    parser.add_argument("--step", default="1h")
    parser.add_argument("--cache-db", default="data/candle_cache.db")
    parser.add_argument("--strategy-snapshot")
    parser.add_argument("--frozen-xgb-model")
    parser.add_argument("--run-id")
    parser.add_argument("--match-window", type=float, default=600.0)
    parser.add_argument("--min-live-match-rate", type=float, default=0.70)
    parser.add_argument("--min-replay-match-rate", type=float, default=0.70)
    parser.add_argument("--report-out")
    parser.add_argument("--diff-report-out")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--lax-api", action="store_true")
    parser.add_argument("--halt-on-error", action="store_true")
    args = parser.parse_args(argv)

    window = find_best_audit_window(
        args.live_db,
        days=args.window_days,
        min_rows=args.min_rows,
    )
    cmd = build_replay_command(args, window)
    payload = {"selected_window": asdict(window), "command": cmd}
    print(json.dumps(payload, indent=2))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
