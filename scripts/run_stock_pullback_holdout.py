#!/usr/bin/env python3
"""Run one frozen, read-only next-session SPY pullback holdout."""

import argparse
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_stock_orb_replay import History, NY, publish, quote_window
from src.backtest.stock_pullback import (
    CASES,
    RULE,
    WARMUP,
    Day,
    proposals,
    quote_days,
    simulate,
)
from src.trading.stock_orb import positive


def fingerprint():
    paths = (
        "scripts/run_stock_pullback_holdout.py",
        "scripts/run_stock_orb_replay.py",
        "src/backtest/stock_pullback.py",
        "src/backtest/stock_range_reversal.py",
        "src/backtest/stock_entry_lab.py",
        "src/backtest/stock_orb.py",
        "src/trading/stock_orb.py",
    )
    return sha256(b"".join((ROOT / name).read_bytes() for name in paths)).hexdigest()


def preregister(db, warmup, start, end, now):
    if (
        not 85 <= (start - warmup).days <= 135
        or not 160 <= (end - start).days <= 320
        or end > now.astimezone(NY).date()
    ):
        raise ValueError("use bounded warm-up and completed holdout dates")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS market (id INTEGER PRIMARY KEY CHECK(id=1), receipt TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS quotes (day TEXT PRIMARY KEY, receipt TEXT NOT NULL);
    """)
    definition = {
        "rule": RULE,
        "warmup": str(warmup),
        "start": str(start),
        "end": str(end),
        "feed": "iex",
        "source_sha256": fingerprint(),
    }
    locked = json.dumps(definition, sort_keys=True)
    with db:
        db.execute("INSERT OR IGNORE INTO protocol VALUES(1,?)", (locked,))
    if db.execute("SELECT manifest FROM protocol WHERE id=1").fetchone()[0] != locked:
        raise ValueError("this ledger is already pinned to other dates or source")
    return definition


def capture_market(feed, warmup, end):
    days = feed.get("calendar", start=str(warmup), end=str(end - timedelta(days=1)))
    begin = datetime.combine(warmup, time(0), NY)
    finish = datetime.combine(end, time(0), NY) - timedelta(microseconds=1)
    raw = {}
    for mode in ("raw", "split"):
        payload = feed.get(
            "bars",
            timeframe="1Day",
            start=begin.isoformat(),
            end=finish.isoformat(),
            adjustment=mode,
            sort="asc",
            limit=1000,
            feed="iex",
        )
        if payload.get("next_page_token"):
            raise ValueError("incomplete daily bars")
        raw[mode] = payload["bars"]
    return {"calendar": days, **raw}


def read_market(data, warmup, start, end):
    sessions = {}
    for row in data["calendar"]:
        d = date.fromisoformat(row["date"])
        open_at, close_at = (time.fromisoformat(row[k]) for k in ("open", "close"))
        if (
            not warmup <= d < end
            or d.weekday() > 4
            or d in sessions
            or open_at != time(9, 30)
            or close_at.tzinfo is not None
            or not time(12) <= close_at <= time(16)
        ):
            raise ValueError("invalid calendar")
        sessions[d] = tuple(
            int(datetime.combine(d, t, NY).timestamp()) for t in (open_at, close_at)
        )
    if (
        sum(d < start for d in sessions) < WARMUP
        or not 100 <= sum(d >= start for d in sessions) <= 240
    ):
        raise ValueError("insufficient warm-up or holdout sessions")

    def prices(rows):
        found = {}
        for row in rows:
            dt = datetime.fromisoformat(row["t"].replace("Z", "+00:00"))
            if dt.utcoffset() is None:
                raise ValueError("untimed daily bar")
            local = dt.astimezone(NY)
            d = local.date()
            if d not in sessions or d in found or local.time() != time(0):
                raise ValueError("misaligned or duplicate daily bar")
            low, high, close = (positive(row[k]) for k in ("l", "h", "c"))
            if not low <= close <= high:
                raise ValueError("malformed daily bar")
            found[d] = (low, high, close)
        if found.keys() != sessions.keys():
            raise ValueError("missing daily bar")
        return found

    raw = prices(data["raw"])
    if raw != prices(data["split"]):
        raise ValueError("raw and split-adjusted prices differ; no fill comparison")
    return [Day(str(d), *sessions[d], raw[d][0], raw[d][2]) for d in sorted(sessions)]


def collect_quotes(db, feed, days, needed):
    for i in needed:
        day = days[i]
        if db.execute("SELECT 1 FROM quotes WHERE day=?", (day.day,)).fetchone():
            continue
        try:
            sample = quote_window(feed, day.slot)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(
                f"{day.day} quote window unavailable: {type(error).__name__}",
                file=sys.stderr,
            )
            continue
        with db:
            db.execute(
                "INSERT INTO quotes VALUES(?,?)",
                (day.day, json.dumps(sample, allow_nan=False)),
            )
    return {
        day: json.loads(receipt)
        for day, receipt in db.execute("SELECT day,receipt FROM quotes")
    }


def evaluate(definition, days, plans, quotes, needed, now):
    cases, trace = {}, {}
    start = definition["start"]
    for scenario in CASES:
        cases[scenario], trace[scenario] = simulate(
            days, start, plans, quotes, scenario
        )
    no_quotes = [days[i].day for i in needed if days[i].day not in quotes]
    relevant = ("base", "stress", "combined")
    sufficient = not no_quotes and all(
        cases[key]["closed"] >= 20 and not cases[key]["unresolved"] for key in relevant
    )
    better = all(
        cases[key]["without_largest"] is not None
        and cases[key]["passive"] is not None
        and cases[key]["without_largest"] > max(0, cases[key]["passive"])
        and cases[key]["close_mark_drawdown"] <= 10
        for key in relevant
    )
    report = {
        **definition,
        "published_at": now.isoformat(),
        "sessions": sum(d.day >= start for d in days),
        "signals": len(plans),
        "quote_windows": len(needed),
        "missing_quote_windows": no_quotes,
        "cash": 0,
        "capital": 1000,
        "sleeve": 75,
        "results": cases,
        "sufficient_sample": sufficient,
        "passes_economics_screen": sufficient and better,
        "paper_approval": False,
        "live_approval": False,
    }
    return report, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", required=True, type=Path)
    p.add_argument("--warmup", required=True, type=date.fromisoformat)
    p.add_argument("--start", required=True, type=date.fromisoformat)
    p.add_argument("--end", required=True, type=date.fromisoformat)
    args = p.parse_args()
    key, secret = (os.getenv(k) for k in ("STOCK_PAPER_KEY", "STOCK_PAPER_SECRET"))
    if os.getenv("STOCK_PAPER_SEND_ORDERS") != "false" or not key or not secret:
        p.error("read-only Alpaca market configuration unavailable")
    with closing(sqlite3.connect(args.state_dir / "holdout.sqlite3", timeout=5)) as db:
        definition = preregister(
            db, args.warmup, args.start, args.end, datetime.now(UTC)
        )
        feed = History(key, secret)
        receipt = db.execute("SELECT receipt FROM market WHERE id=1").fetchone()
        raw = (
            json.loads(receipt[0])
            if receipt
            else capture_market(feed, args.warmup, args.end)
        )
        days = read_market(raw, args.warmup, args.start, args.end)
        if not receipt:
            with db:
                db.execute(
                    "INSERT INTO market VALUES(1,?)",
                    (json.dumps(raw, allow_nan=False),),
                )
        plans = proposals(days, definition["start"])
        needed = quote_days(days, definition["start"], plans)
        quotes = collect_quotes(db, feed, days, needed)
        report, trace = evaluate(
            definition, days, plans, quotes, needed, datetime.now(UTC)
        )
    publish(args.state_dir / "trace.json", trace)
    publish(args.state_dir / "latest.json", report)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
