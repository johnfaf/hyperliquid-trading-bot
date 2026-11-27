#!/usr/bin/env python3
"""Run a fixed, paired SPY exit-policy replay against historical quotes only."""

import argparse
from contextlib import closing
from datetime import UTC, date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
from urllib.error import HTTPError
import zlib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_stock_orb_replay import History, NY, publish, utc
from scripts.run_stock_pullback_holdout import capture_market, read_market
from src.backtest.stock_exit_retry import (
    PROTOCOL,
    REFERENCE_SPREAD_BPS,
    RETRY_COST_BPS,
    RETRY_OFFSETS,
    RETRY_SPREAD_BPS,
    WINDOW_END,
    capture_slots,
    paired_episodes,
)
from src.backtest.stock_pullback import CASES, RULE, proposals

WARMUP, START, END = date(2024, 9, 3), date(2025, 1, 2), date(2025, 9, 1)
FEEDS = ("iex", "sip")
SOURCE_FILES = (
    "scripts/run_stock_exit_retry_holdout.py",
    "scripts/run_stock_pullback_holdout.py",
    "scripts/run_stock_orb_replay.py",
    "src/backtest/stock_exit_retry.py",
    "src/backtest/stock_pullback.py",
    "src/backtest/stock_range_reversal.py",
    "src/backtest/stock_entry_lab.py",
    "src/backtest/stock_orb.py",
    "src/trading/stock_orb.py",
)


def fingerprint():
    return sha256(
        b"".join((ROOT / name).read_bytes() for name in SOURCE_FILES)
    ).hexdigest()


def preregister(db, now):
    if now.astimezone(NY).date() < END:
        raise ValueError("holdout must finish before capture")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS market (id INTEGER PRIMARY KEY CHECK(id=1), receipt TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS windows (
            feed TEXT NOT NULL, slot INTEGER NOT NULL, receipt BLOB NOT NULL, PRIMARY KEY(feed, slot)
        );
    """)
    definition = {
        "protocol": PROTOCOL,
        "entry_rule": RULE,
        "source_sha256": fingerprint(),
        "warmup": str(WARMUP),
        "start": str(START),
        "end_exclusive": str(END),
        "daily_feed": "iex",
        "quote_feeds": list(FEEDS),
        "retry_offsets_seconds": list(RETRY_OFFSETS),
        "reference_spread_bps": REFERENCE_SPREAD_BPS,
        "retry_spread_bps": RETRY_SPREAD_BPS,
        "retry_cost_bps": RETRY_COST_BPS,
    }
    locked = json.dumps(definition, sort_keys=True)
    with db:
        db.execute("INSERT OR IGNORE INTO protocol VALUES(1,?)", (locked,))
    if db.execute("SELECT manifest FROM protocol WHERE id=1").fetchone()[0] != locked:
        raise ValueError("ledger already pinned to different source or rules")
    return definition


def capture_window(history, feed, slot):
    query = dict(
        start=utc(slot + 2),
        end=utc(slot + WINDOW_END),
        feed=feed,
        sort="asc",
        limit=8000,
    )
    seen = set()
    rows = []
    for _ in range(12):
        payload = history.get("quotes", **query)
        batch = payload["quotes"]
        if not isinstance(batch, list):
            raise ValueError("malformed quote batch")
        rows.extend(batch)
        token = payload.get("next_page_token")
        if not token:
            return rows
        if not isinstance(token, str) or not batch or token in seen:
            break
        seen.add(token)
        query["page_token"] = token
    raise ValueError("incomplete quote window")


def collect(db, history, slots):
    for feed in FEEDS:
        existing = {
            slot
            for (slot,) in db.execute("SELECT slot FROM windows WHERE feed=?", (feed,))
        }
        issues = []
        for slot, day in slots.items():
            if slot in existing:
                continue
            try:
                rows = capture_window(history, feed, slot)
                receipt = zlib.compress(
                    json.dumps(rows, allow_nan=False, separators=(",", ":")).encode()
                )
            except HTTPError as error:
                label = f"HTTP {error.code}"
                issues.append({"day": day, "slot": slot, "error": label})
                if error.code in (401, 403, 429):
                    break
                continue
            except (OSError, TypeError, KeyError, ValueError) as error:
                issues.append({"day": day, "slot": slot, "error": type(error).__name__})
                continue
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO windows VALUES(?,?,?)", (feed, slot, receipt)
                )
        yield feed, issues


class WindowLedger:
    def __init__(self, db, feed):
        self.db = db
        self.feed = feed

    def get(self, slot):
        row = self.db.execute(
            "SELECT receipt FROM windows WHERE feed=? AND slot=?", (self.feed, slot)
        ).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None


def evaluate(db, definition, days, plans, slots, issues, now):
    result, traces = {}, {}
    for feed in FEEDS:
        captured = {
            slot
            for (slot,) in db.execute("SELECT slot FROM windows WHERE feed=?", (feed,))
        }
        missing = [
            {"day": day, "slot": slot}
            for slot, day in slots.items()
            if slot not in captured
        ]
        windows = WindowLedger(db, feed)
        cases, traces[feed] = {}, {}
        # Incomplete or inaccessible feeds must not report misleading zero-trade outcomes.
        if not missing:
            for case in CASES:
                cases[case], traces[feed][case] = paired_episodes(
                    days, plans, windows, case
                )
        result[feed] = {
            "captured": len(slots) - len(missing),
            "missing": missing,
            "capture_errors": issues[feed],
            "results": cases,
        }
    report = {
        **definition,
        "published_at": now.isoformat(),
        "opportunities": len(plans),
        "sessions": sum(d.day >= str(START) for d in days),
        "quote_windows_per_feed": len(slots),
        "feeds": result,
        "paper_approval": False,
        "live_approval": False,
    }
    return report, traces


def run(db, history, now, *, capture=True):
    definition = preregister(db, now)
    existing = db.execute("SELECT receipt FROM market WHERE id=1").fetchone()
    if not existing and not capture:
        raise ValueError("no data to reproduce")
    raw = json.loads(existing[0]) if existing else capture_market(history, WARMUP, END)
    days = read_market(raw, WARMUP, START, END)
    if not existing:
        with db:
            db.execute(
                "INSERT INTO market VALUES(1,?)", (json.dumps(raw, allow_nan=False),)
            )
    plans = proposals(days, definition["start"])
    slots = capture_slots(days, plans)
    issues = dict(collect(db, history, slots)) if capture else dict.fromkeys(FEEDS, [])
    return evaluate(db, definition, days, plans, slots, issues, now)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", required=True, type=Path)
    args = p.parse_args()
    key, secret = (os.getenv(k) for k in ("STOCK_PAPER_KEY", "STOCK_PAPER_SECRET"))
    if os.getenv("STOCK_PAPER_SEND_ORDERS") != "false" or not key or not secret:
        p.error("read-only Alpaca market configuration unavailable")
    with closing(sqlite3.connect(args.state_dir / "retry.sqlite3", timeout=5)) as db:
        report, trace = run(db, History(key, secret), datetime.now(UTC))
    publish(args.state_dir / "trace.json", trace)
    publish(args.state_dir / "latest.json", report)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
