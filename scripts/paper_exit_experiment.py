#!/usr/bin/env python3
"""Record public BTC market data or compare three offline exit policies."""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_paper_pilot import COIN, SPEC, STATE_KEY, STRATEGY, metadata
from src.backtest.paper_exit_study import (
    ARM_AFTER_R, ATR_PERIOD, FEE_RATE, HEALTHY_CHECKS, HEALTHY_SPREAD_BPS, MAX_GAP_MS,
    MINUTE_MS, POLICIES, SLIPPAGE, TRAIL_ATR_MULTIPLE, TRAIL_FLOOR_PCT, Entry, build_frames, compare,
)
from src.data.paper_exit_feed import MarketSample, PaperExitFeed, SAMPLE_MS


DB_NAME = "btc-paper-exit-market.sqlite3"
SCHEMA_VERSION = 2
_FIELDS = [f.name for f in fields(MarketSample)]
_COLUMN_LIST = ", ".join(_FIELDS)


def _market_path(path: Path) -> Path:
    path = path.resolve()
    if path.name != DB_NAME:
        raise ValueError(f"market data must use a dedicated {DB_NAME}, not the pilot database")
    return path


def _init_market(conn: sqlite3.Connection):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if version == SCHEMA_VERSION and names == {"market_samples"}:
        return
    if version or names:
        raise ValueError("unrecognized market database; refusing to change it")
    suffix = ", ".join(f"{name} {'TEXT' if name == 'quote_source' else 'REAL'}" for name in _FIELDS[2:])
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"CREATE TABLE market_samples (at_ms INTEGER PRIMARY KEY, status TEXT NOT NULL, {suffix})")
        conn.execute(f"CREATE UNIQUE INDEX sample_bucket ON market_samples (at_ms / {SAMPLE_MS})")
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def _record(conn: sqlite3.Connection, sample: MarketSample):
    values = tuple(getattr(sample, key) for key in _FIELDS)
    with conn:
        conn.execute(f"INSERT INTO market_samples ({_COLUMN_LIST}) VALUES ({', '.join('?' for _ in values)})", values)


def capture(path: Path, duration_seconds: int) -> dict:
    if duration_seconds < SAMPLE_MS // 1000:
        raise ValueError("capture duration must be at least 5 seconds")
    market_path = _market_path(path)
    counts = Counter()
    with closing(sqlite3.connect(market_path, timeout=5)) as conn:
        _init_market(conn)
        feed = PaperExitFeed(COIN)
        stream = feed.make_stream()
        stream.start()
        try:
            deadline = time.monotonic() + duration_seconds
            next_sample = (int(time.time() * 1000) // SAMPLE_MS + 1) * SAMPLE_MS
            while time.monotonic() < deadline:
                wait = max(0.0, (next_sample - time.time() * 1000) / 1000)
                remaining = deadline - time.monotonic()
                if wait >= remaining:
                    break
                time.sleep(wait)
                sample = feed.snapshot(connected=stream.is_connected())
                if sample.at_ms < next_sample:
                    continue
                _record(conn, sample)
                counts[sample.status] += 1
                next_sample = (sample.at_ms // SAMPLE_MS + 1) * SAMPLE_MS
        finally:
            stream.stop()
    return dict(counts)


def _ms(value: str) -> int:
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("timestamps must include an explicit timezone, e.g. 2026-11-15T00:00:00Z")
    return int(stamp.timestamp() * 1000)


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="milliseconds")


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve(strict=True).as_uri()}?mode=ro", uri=True, timeout=5)


def load_entries(path: Path, start: int, end: int) -> list[Entry]:
    with closing(_readonly(path)) as conn:
        conn.row_factory = sqlite3.Row
        spec = conn.execute("SELECT value FROM bot_state WHERE key=?", (STATE_KEY,)).fetchone()
        if not spec or json.loads(spec[0]) != SPEC:
            raise ValueError("expected the unchanged BTC paper pilot database and study configuration")
        # Pilot timestamps are emitted as UTC ISO strings; the epoch comparison below
        # retains the exact boundary when a timestamp has varying fractional precision.
        first = _utc(start)[:10]
        last = (datetime.fromtimestamp(end / 1000, timezone.utc).date() + timedelta(days=1)).isoformat()
        rows = conn.execute(
            "SELECT id, opened_at, coin, side, leverage, entry_price, size, take_profit, metadata FROM paper_trades "
            "WHERE opened_at >= ? AND opened_at < ? ORDER BY opened_at, id", (first, last),
        )
        entries = []
        for row in rows:
            stamp = _ms(row["opened_at"])
            if not start <= stamp < end:
                continue
            meta = metadata(row)
            if row["coin"] != COIN or row["side"] != "long" or row["leverage"] != 1 or meta.get("source_key") != STRATEGY:
                raise ValueError(f"unexpected entry #{row['id']} in pilot database")
            entries.append(Entry(
                row["id"], stamp, row["entry_price"], row["size"], meta["initial_stop_loss"], row["take_profit"],
            ))
    return entries


def load_samples(path: Path, start: int, end: int) -> list[MarketSample]:
    with closing(_readonly(_market_path(path))) as conn:
        if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("unrecognized market recording version")
        return [MarketSample(*row) for row in conn.execute(
            f"SELECT {_COLUMN_LIST} FROM market_samples WHERE at_ms >= ? AND at_ms < ? ORDER BY at_ms",
            (start - (ATR_PERIOD + 2) * MINUTE_MS, end),
        )]


def build_report(entries: list[Entry], samples: list[MarketSample], start: int, end: int) -> dict:
    frames = build_frames(samples)
    cohorts = compare(entries, frames)
    paired = [outcomes for _, outcomes in cohorts if all(
        x.exit_ms is not None and not x.material_gap and not x.unknown_funding_hours for x in outcomes.values()
    )]
    window_frames = [frame for frame in frames if start <= frame.sample.at_ms < end]
    latest_context = None
    if window_frames:
        frame = window_frames[-1]
        latest_context = {
            **asdict(frame.sample), "spread_bps": frame.sample.spread_bps, "atr": frame.atr,
            "one_minute_bps": frame.one_minute_bps, "five_minute_bps": frame.five_minute_bps,
        }
    arm_results = {}
    for name in POLICIES:
        outcomes = [outcome[name] for outcome in paired]
        rebound = [x.subsequent_5m_rebound for x in outcomes if x.subsequent_5m_rebound is not None]
        arm_results[name] = {
            "modeled_pnl_usd": round(sum(x.modeled_pnl for x in outcomes), 6) if outcomes else None,
            "mean_profit_giveback_usd": round(fmean(x.profit_giveback for x in outcomes), 6) if outcomes else None,
            "worst_single_trade_drawdown_usd": round(max(x.worst_trade_drawdown for x in outcomes), 6) if outcomes else None,
            "mean_subsequent_5m_rebound_usd": round(fmean(rebound), 6) if rebound else None,
            "rebound_observations": len(rebound),
            "exit_reasons": dict(Counter(x.reason for x in outcomes)),
            "fills_after_trigger_sample": sum(x.exit_ms > x.trigger_ms for x in outcomes),
        }
    return {
        "mode": "offline_paper_only",
        "methodology": {
            "schema_version": SCHEMA_VERSION, "check_intervals_ms": POLICIES, "funding": "hourly estimate",
            "taker_fee_bps": FEE_RATE * 10_000, "exit_impact_bps": SLIPPAGE * 10_000,
            "arm_after_r": ARM_AFTER_R, "atr_minutes": ATR_PERIOD, "atr_multiple": TRAIL_ATR_MULTIPLE,
            "trail_floor_pct": TRAIL_FLOOR_PCT, "max_spread_bps_to_tighten": HEALTHY_SPREAD_BPS,
            "distinct_liquid_quotes_to_tighten": HEALTHY_CHECKS, "material_gap_ms": MAX_GAP_MS,
        },
        "window_utc": [_utc(start), _utc(end)],
        "market_samples": dict(Counter(frame.sample.status for frame in window_frames)),
        "latest_context": latest_context,
        "entries_total": len(entries),
        "paired_exits_without_material_gaps_or_missing_funding": len(paired),
        "excluded_from_aggregate": len(cohorts) - len(paired),
        "paired_usd_deltas": {
            "faster_fixed_vs_minute": round(arm_results["fixed_5s"]["modeled_pnl_usd"]
                                             - arm_results["fixed_60s"]["modeled_pnl_usd"], 6) if paired else None,
            "trailing_vs_faster_fixed": round(arm_results["trailing_5s"]["modeled_pnl_usd"]
                                             - arm_results["fixed_5s"]["modeled_pnl_usd"], 6) if paired else None,
        },
        "arms": arm_results,
        "per_entry": [
            {"entry": asdict(entry), "arms": {name: asdict(result) for name, result in outcomes.items()}}
            for entry, outcomes in cohorts
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture", help="Collect timestamped public data; no orders or account credentials")
    collect.add_argument("--market-db", type=Path, required=True)
    collect.add_argument("--seconds", type=int, default=60)
    replay = commands.add_parser("compare", help="Read the pilot and market databases and print a paired JSON report")
    replay.add_argument("--market-db", type=Path, required=True)
    replay.add_argument("--pilot-db", type=Path, required=True)
    replay.add_argument("--from", dest="start", type=_ms, required=True)
    replay.add_argument("--until", dest="end", type=_ms, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "capture":
            counts = capture(args.market_db, args.seconds)
            print(json.dumps({"capture_samples": counts}, sort_keys=True))
            return 0 if counts.get("ready", 0) else 1
        if args.start >= args.end:
            raise ValueError("--from must be before --until")
        entries = load_entries(args.pilot_db, args.start, args.end)
        samples = load_samples(args.market_db, args.start, args.end)
        print(json.dumps(build_report(entries, samples, args.start, args.end), indent=2, allow_nan=False))
        return 0
    except (OSError, sqlite3.Error, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Paper exit experiment failed: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
