#!/usr/bin/env python3
"""Capture fixed stock research observations, without an order or account API."""
import argparse
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.stock_shadow import CAPITAL, COSTS, DAY_LIMIT, GRACE, RULE, SLEEVE, Frame, simulate
from src.trading.stock_signals import NY, completed_closes, sma20_from_closes

ENDPOINTS = {
    "calendar": "https://paper-api.alpaca.markets/v2/calendar",
    "clock": "https://paper-api.alpaca.markets/v2/clock",
    "bars": "https://data.alpaca.markets/v2/stocks/SPY/bars",
    "quote": "https://data.alpaca.markets/v2/stocks/SPY/quotes/latest",
}
AGE = 30
MAX_REPLY = 2_000_000


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class QuoteFeed:
    def __init__(self, key, secret):
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.opener = build_opener(NoRedirects())

    def get(self, endpoint, **params):
        url = ENDPOINTS[endpoint] + ("?" + urlencode(params) if params else "")
        with self.opener.open(Request(url, method="GET", headers=self.headers), timeout=8) as response:
            body = response.read(MAX_REPLY + 1)
        if len(body) > MAX_REPLY:
            raise ValueError("oversized research response")
        return json.loads(body, parse_float=Decimal)


def fingerprint():
    return sha256(b"".join((ROOT / path).read_bytes() for path in (
        "scripts/run_stock_forward_shadow.py", "src/backtest/stock_shadow.py", "src/trading/stock_signals.py",
    ))).hexdigest()


def utc_now():
    return datetime.now(UTC)


def stamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.utcoffset() is None:
        raise ValueError("missing timestamp offset")
    return dt.timestamp()


def clock_slot(received, slot):
    return slot <= received.timestamp() <= slot + GRACE


def install_schema(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK (id=1),
            frozen_at TEXT, start_day TEXT, end_day TEXT, digest TEXT);
        CREATE TABLE IF NOT EXISTS sessions (
            day TEXT PRIMARY KEY, decision_at INTEGER, fill_at INTEGER, closes_at INTEGER);
        CREATE TABLE IF NOT EXISTS frames (day TEXT, slot INTEGER, quote_at INTEGER,
            observed_at INTEGER, bid TEXT, ask TEXT, entry INTEGER, exit INTEGER, inputs TEXT,
            PRIMARY KEY (day,slot));
    """)


def freeze(db, feed, now, digest):
    today = now.astimezone(NY).date()
    start, end = today + timedelta(days=1), today + timedelta(days=1 + DAY_LIMIT)
    lookback = start - timedelta(days=50)
    calendar = feed.get("calendar", start=lookback.isoformat(), end=(end-timedelta(days=1)).isoformat())
    sessions = []
    for row in calendar:
        day = date.fromisoformat(row["date"])
        opened, closed = [time.fromisoformat(row[k]) for k in ("open", "close")]
        if (not lookback <= day < end or day.weekday() > 4 or opened != time(9, 30)
                or closed.tzinfo is not None or not time(10) <= closed <= time(16)):
            raise ValueError("unexpected regular-session calendar")
        opening, closing = [int(datetime.combine(day, at, NY).timestamp()) for at in (opened, closed)]
        sessions.append((day.isoformat(), opening + 15*60, opening + 20*60, closing))
    dates = {row[0] for row in sessions}
    if len(dates) != len(sessions) or sum(day < start.isoformat() for day in dates) < 21 or not any(
            day >= start.isoformat() for day in dates):
        raise ValueError("invalid frozen session coverage")
    with db:
        db.execute("INSERT INTO protocol VALUES (1,?,?,?,?)", (now.isoformat(), start.isoformat(), end.isoformat(), digest))
        db.executemany("INSERT INTO sessions VALUES (?,?,?,?)", sessions)


def capture(db, feed, day, slot, at, closes_at, clock):
    market = feed.get("clock")
    received = clock()
    broker_time = stamp(market["timestamp"])
    if (market["is_open"] is not True or not 0 <= received.timestamp()-broker_time <= AGE
            or stamp(market["next_close"]) != closes_at):
        raise ValueError("unverified regular-session clock")
    entry = exit_signal = None
    inputs = None
    if slot == 0:
        lookback = (date.fromisoformat(day)-timedelta(days=50)).isoformat()
        bars = feed.get("bars", timeframe="1Day", feed="iex", start=lookback, end=day, adjustment="split", limit=100)
        sessions = [row[0] for row in db.execute("SELECT day FROM sessions WHERE day<? ORDER BY day", (day,))]
        closes = None if bars.get("next_page_token") else completed_closes(bars["bars"], sessions, day)
        if closes is not None:
            entry, exit_signal = sma20_from_closes(closes)
            inputs = json.dumps([(session, str(price)) for session, price in closes], separators=(",", ":"))
    quote = feed.get("quote", feed="iex")["quote"]
    received = clock()
    quote_at = stamp(quote["t"])
    bid, ask = Decimal(str(quote["bp"])), Decimal(str(quote["ap"]))
    if (not clock_slot(received, at) or not at <= quote_at <= received.timestamp() <= quote_at+AGE
            or not (bid.is_finite() and ask.is_finite() and Decimal(".01") <= bid <= ask <= 100_000)
            or (ask-bid)/bid > Decimal(".005")):
        raise ValueError("unusable research quote")
    with db:
        db.execute("INSERT OR IGNORE INTO frames VALUES (?,?,?,?,?,?,?,?,?)",
                   (day, slot, int(quote_at), int(received.timestamp()), str(bid), str(ask), entry, exit_signal, inputs))


def last_report(db, now, start, end, error=None):
    days = []
    expected = paired = samples_due = samples_captured = planned = 0
    final_pair = False
    next_at = None
    frames = {(day, slot): Frame(quote, observed, Decimal(bid), Decimal(ask), entry, leave)
              for day, slot, quote, observed, bid, ask, entry, leave in db.execute(
                  "SELECT day,slot,quote_at,observed_at,bid,ask,entry,exit FROM frames")}
    for day, a, b in db.execute(
        "SELECT day,decision_at,fill_at FROM sessions WHERE day>=? AND day<? ORDER BY day", (start, end),
    ):
        planned += 1
        first, second = frames.get((day, 0)), frames.get((day, 1))
        final_pair = (first is not None and second is not None and first.entry is not None
                      and first.exit is not None and first.observed_at < second.quote_at)
        for at, frame in ((a, first), (b, second)):
            if now.timestamp() > at + GRACE or frame is not None:
                samples_due += 1
                samples_captured += frame is not None
            elif next_at is None:
                next_at = at
        expected += now.timestamp() > b + GRACE or second is not None
        paired += final_pair
        days.append((day, first, second))
    complete = now.astimezone(NY).date() >= date.fromisoformat(end)
    state = "Window complete" if complete else "Observing" if samples_captured else "Awaiting forward observations"
    results, trace = {}, {}
    if paired:
        for name, costs in COSTS.items():
            results[name], trace[name] = simulate(days, costs)
    report = {
        "rule": RULE, "state": state, "published_at": now.isoformat(), "start": start, "end": end,
        "capital": int(CAPITAL), "sleeve": int(SLEEVE), "live_approval": False, "market_status": error,
        "sessions_planned": planned, "sessions_due": expected, "paired": paired,
        "samples_due": samples_due, "samples_captured": samples_captured,
        "final_pair": final_pair,
        "next_slot": datetime.fromtimestamp(next_at, UTC).isoformat() if next_at is not None else None,
        "results": results or None,
    }
    return report, trace


def cycle(db, feed, *, clock=utc_now, digest=None):
    now = clock()
    digest = digest or fingerprint()
    install_schema(db)
    protocol = db.execute("SELECT start_day,end_day,digest FROM protocol").fetchone()
    if protocol is None:
        freeze(db, feed, now, digest)
        protocol = db.execute("SELECT start_day,end_day,digest FROM protocol").fetchone()
    start, end, locked_digest = protocol
    if digest != locked_digest:
        raise ValueError("frozen stock protocol mismatch")
    day = now.astimezone(NY).date().isoformat()
    row = db.execute("SELECT decision_at,fill_at,closes_at FROM sessions WHERE day=?", (day,)).fetchone()
    error = None
    if row and start <= day < end:
        for slot, target in enumerate(row[:2]):
            if clock_slot(now, target) and not db.execute("SELECT 1 FROM frames WHERE day=? AND slot=?", (day, slot)).fetchone():
                try:
                    capture(db, feed, day, slot, target, row[2], clock)
                except (OSError, ValueError, ArithmeticError, KeyError, TypeError, AttributeError, OverflowError):
                    error = "Market data not usable in this slot"
    return last_report(db, clock(), start, end, error)


def atomic_json(path, value):
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, allow_nan=False, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    key, secret = os.environ.get("STOCK_PAPER_KEY"), os.environ.get("STOCK_PAPER_SECRET")
    available = os.environ.get("STOCK_PAPER_SEND_ORDERS") == "false" and key and secret
    trace = None
    report = {"rule": RULE, "state": "Unavailable", "published_at": utc_now().isoformat(), "capital": int(CAPITAL),
              "sleeve": int(SLEEVE), "live_approval": False, "market_status": "Missing shadow-only market configuration"}
    if available:
        try:
            with closing(sqlite3.connect(args.state_dir / "market.sqlite3", timeout=10)) as db:
                report, trace = cycle(db, QuoteFeed(key, secret))
        except (OSError, ValueError, ArithmeticError, KeyError, TypeError, AttributeError, sqlite3.Error):
            report["market_status"] = "Frozen protocol or market data unavailable"
    if trace is not None:
        atomic_json(args.state_dir / "trace.json", trace)
    atomic_json(args.state_dir / "latest.json", report)
    print(json.dumps({"state": report["state"], "market_status": report["market_status"],
                      "samples_captured": report.get("samples_captured")}))


if __name__ == "__main__":
    main()
