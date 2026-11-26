#!/usr/bin/env python3
"""Freeze paired copies of a public-data simulation; never issue an order."""
import argparse
from contextlib import closing
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest.copy_forward import CAPITAL, COSTS, FEEDS, RULE, SLEEVE, SYMBOLS, Candidate, Quote, paired

HOUR = 3600
HOURS = 28*24
TIMES = (600, 900, 2400)
GRACE = 55
MAX_CANDIDATES = 300
INFO = "https://api.hyperliquid.xyz/info"


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def utc_now():
    return datetime.now(UTC)


def iso(t):
    return datetime.fromtimestamp(t, UTC).isoformat()


def read_candidates(path, hour, seen):
    with closing(sqlite3.connect(path.resolve(strict=True).as_uri()+"?mode=ro", uri=True, timeout=2)) as db:
        rows = db.execute("""
            SELECT decision_id, coin, side, source, raw_confidence, created_at, signal_timestamp
            FROM decision_snapshots WHERE created_at>=? AND created_at<? AND coin IN ('BTC','ETH','SOL')
            ORDER BY created_at,decision_id LIMIT ?
        """, (iso(hour), iso(hour+TIMES[0]), MAX_CANDIDATES+1)).fetchall()
    if len(rows) > MAX_CANDIDATES:
        raise ValueError("candidate capture exceeded fixed quota")
    found = {}
    for ref, coin, side, source, value, recorded, source_stamp in rows:
        if not isinstance(ref, str) or not ref or len(ref) > 256 or source not in FEEDS or side not in ("long", "short"):
            continue
        try:
            observed, issued = datetime.fromisoformat(recorded), datetime.fromisoformat(source_stamp)
            score = float(value)
        except (ValueError, TypeError, OverflowError):
            continue
        if (observed.utcoffset() is None or issued.utcoffset() is None
                or not hour <= observed.timestamp() < hour+TIMES[0] or observed > seen
                or not 0 <= (observed-issued).total_seconds() <= 600
                or not math.isfinite(score) or not .65 <= score <= 1):
            continue
        ref = sha256(ref.encode()).hexdigest()
        found.setdefault(ref, Candidate(coin, side, source, score, observed.astimezone(UTC).isoformat(), ref))
    return list(found.values())


def l2(coin):
    request = Request(INFO, method="POST", headers={"Content-Type":"application/json"},
                      data=json.dumps({"type":"l2Book","coin":coin}).encode())
    with build_opener(NoRedirects()).open(request, timeout=5) as reply:
        body = reply.read(131073)
    if len(body) > 131072:
        raise ValueError("book response too large")
    return json.loads(body)


def quote(raw, coin, slot, seen):
    observed = int(seen.timestamp()*1000)
    venue = raw["time"]
    bid, ask = (Decimal(level[0]["px"]) for level in raw["levels"])
    bid_qty, ask_qty = (Decimal(level[0]["sz"]) for level in raw["levels"])
    if (raw["coin"] != coin or type(venue) is not int
            or not slot*1000 <= venue <= observed <= min(venue+15000, (slot+GRACE)*1000)
            or not all(x.is_finite() and x > 0 for x in (bid, ask, bid_qty, ask_qty))
            or not bid <= ask <= bid*Decimal("1.005")):
        raise ValueError("invalid, stale or wide quote")
    return Quote(bid, ask, bid_qty, ask_qty, observed, venue)


def protocol_digest():
    return sha256(b"".join((ROOT/name).read_bytes() for name in (
        "scripts/run_copy_forward.py", "src/backtest/copy_forward.py",
    ))).hexdigest()


def cycle(db, upstream, *, clock=utc_now, book=l2, candidate_feed=read_candidates):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK(id=1), start INTEGER, digest TEXT);
        CREATE TABLE IF NOT EXISTS capture (hour INTEGER, slot INTEGER, seen INTEGER, value TEXT,
                                             PRIMARY KEY(hour, slot));
    """)
    now = clock()
    t = int(now.timestamp())
    hour = t//HOUR*HOUR
    with db:
        db.execute("INSERT OR IGNORE INTO protocol VALUES(1,?,?)", (hour+HOUR, protocol_digest()))
    start, pinned = db.execute("SELECT start,digest FROM protocol WHERE id=1").fetchone()
    if pinned != protocol_digest():
        raise ValueError("model changed after preregistration")
    end = start+HOURS*HOUR
    if start <= hour < end:
        for n, offset in enumerate(TIMES):
            at = hour+offset
            if not at <= t <= at+GRACE or db.execute("SELECT 1 FROM capture WHERE hour=? AND slot=?", (hour,n)).fetchone():
                continue
            try:
                if n == 0:
                    values = [vars(r) for r in candidate_feed(upstream, hour, now)]
                else:
                    values = {}
                    for coin in SYMBOLS:
                        raw = book(coin)
                        q = quote(raw, coin, at, clock())
                        values[coin] = {k:str(v) if isinstance(v, Decimal) else v for k,v in vars(q).items()}
                seen = int(clock().timestamp()*1000)
                if seen > (at+GRACE)*1000:
                    raise ValueError("capture expired before commit")
                with db:
                    db.execute("INSERT OR IGNORE INTO capture VALUES(?,?,?,?)", (hour,n,seen,json.dumps(values,allow_nan=False)))
            except (OSError, ValueError, KeyError, IndexError, TypeError, ArithmeticError, sqlite3.Error) as error:
                print(f"Capture slot {n} unavailable: {type(error).__name__}", file=sys.stderr)
    return report(db, clock(), start, end)


def report(db, now, start, end):
    frames = {(h,n): (seen,json.loads(value)) for h,n,seen,value in db.execute("SELECT * FROM capture")}
    observed = int(now.timestamp())
    hours = []
    candidates_seen = copies_seen = 0
    for hour in range(start, end, HOUR):
        if observed < hour+TIMES[-1]+GRACE and (hour,2) not in frames:
            break
        timings = [frames.get((hour,n)) for n in range(3)]
        if any(x is None for x in timings):
            hours.append((hour,None,None,None))
            continue
        inputs, opening, closing = timings
        candidates = [Candidate(**r) for r in inputs[1]]
        quotes = [{c: Quote(*(Decimal(q[k]) for k in ("bid","ask","bid_qty","ask_qty")), q["observed_ms"], q["venue_ms"])
                   for c,q in sample[1].items()} for sample in (opening,closing)]
        if any(set(q) != set(SYMBOLS) for q in quotes) or not all(
                inputs[0] < quotes[0][c].venue_ms <= opening[0] < quotes[1][c].venue_ms <= closing[0] for c in SYMBOLS):
            hours.append((hour,None,None,None))
            continue
        candidates_seen += len(candidates)
        copies_seen += sum(c.source == "copy_trade" for c in candidates)
        hours.append((hour,candidates,*quotes))
    results, traces = {}, {}
    for name, costs in COSTS.items():
        results[name], traces[name] = paired(hours, costs)
    published = {"rule":RULE, "start":iso(start), "end":iso(end), "published_at":now.isoformat(),
                 "state":"Window complete" if observed >= end else "Observing" if hours else "Awaiting scheduled captures",
                 "capital":int(CAPITAL), "sleeve":int(SLEEVE), "live_approval":False, "due_hours":len(hours),
                 "candidates":candidates_seen, "copy_candidates":copies_seen,
                 "capture_hours":sum(rows is not None for _,rows,_,_ in hours),
                 "results":results if any(r["scored_hours"] for r in results.values()) else None}
    return published, traces


def atomic_json(path, content):
    temp = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, encoding="ascii", delete=False) as out:
            temp = Path(out.name)
            json.dump(content, out, allow_nan=False, separators=(",", ":"))
            out.flush()
            os.fsync(out.fileno())
        temp.replace(path)
    finally:
        if temp:
            temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--decision-db", type=Path, required=True)
    args = p.parse_args()
    with closing(sqlite3.connect(args.state_dir/"paired.sqlite3",timeout=5)) as db:
        summary, trace = cycle(db, args.decision_db)
    atomic_json(args.state_dir/"trace.json", trace)
    atomic_json(args.state_dir/"latest.json", summary)
    print(json.dumps({"state":summary["state"],"start":summary["start"],"due_hours":summary["due_hours"]}))


if __name__ == "__main__":
    main()
