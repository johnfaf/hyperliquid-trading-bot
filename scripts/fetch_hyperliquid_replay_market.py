#!/usr/bin/env python3
"""Capture an immutable top-ten Hyperliquid perpetuals candle and funding dataset."""
from argparse import ArgumentParser
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

INFO = "https://api.hyperliquid.xyz/info"
INTERVAL = "15m"
BAR = 900_000
HOUR = 4 * BAR
DAY = 24 * HOUR
WARMUP = 100
CHUNK = 480
CODES = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")


def iso(value):
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def post(body):
    request = Request(INFO, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=15) as response:
                return json.load(response)
        except (OSError, ValueError) as exc:
            if attempt == 3 or (isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}):
                raise
            time.sleep(1.0 + 2 ** attempt)


def ranking(response):
    meta, contexts = response
    universe = meta["universe"]
    if len(universe) != len(contexts):
        raise ValueError("Hyperliquid universe/context mismatch")
    found = {}
    for item, context in zip(universe, contexts):
        coin = item["name"]
        if not CODES.fullmatch(coin) or item.get("isDelisted") or item.get("isHyperp"):
            continue
        notional = float(context.get("dayNtlVlm") or 0)
        if math.isfinite(notional) and notional > 0:
            if coin in found:
                raise ValueError("duplicate perpetual symbol")
            found[coin] = notional
    if len(found) < 10:
        raise ValueError("fewer than ten eligible active perpetuals")
    return sorted(found.items(), key=lambda x: (-x[1], x[0]))[:10]


def fetch_candles(coin, start, end, send=post):
    result = {}
    for lo in range(start, end, CHUNK * BAR):
        hi = min(lo + CHUNK * BAR, end)
        response = send({"type": "candleSnapshot", "req": {"coin": coin, "interval": INTERVAL,
                         "startTime": lo, "endTime": hi - 1}})
        for bar in response:
            stamp = int(bar["t"])
            if not lo <= stamp < hi:
                continue
            o, h, low, c, v = (float(bar[key]) for key in "ohlcv")
            if (bar["s"] != coin or bar["i"] != INTERVAL or stamp % BAR or int(bar["T"]) != stamp + BAR - 1
                    or not all(map(math.isfinite, (o, h, low, c, v))) or not 0 < low <= min(o, c)
                    or not h >= max(o, c) or v < 0):
                raise ValueError(f"{coin}: malformed candle")
            value = (o, h, low, c, v)
            if stamp in result and result[stamp] != value:
                raise ValueError(f"{coin}: conflicting candle")
            result[stamp] = value
        time.sleep(.10)
    return result


def fetch_funding(coin, start, end, send=post):
    result = {}
    first = ((start + HOUR - 1) // HOUR) * HOUR
    for lo in range(first, end, CHUNK * HOUR):
        hi = min(lo + CHUNK * HOUR, end)
        for row in send({"type": "fundingHistory", "coin": coin, "startTime": lo, "endTime": hi - 1}):
            stamp, rate = int(row["time"]), float(row["fundingRate"])
            bucket = (stamp // HOUR) * HOUR
            if not lo <= bucket < hi:
                continue
            if row["coin"] != coin or stamp - bucket > 30_000 or not math.isfinite(rate) or abs(rate) > .10:
                raise ValueError(f"{coin}: malformed funding settlement")
            value = (stamp, rate)
            if bucket in result and result[bucket] != value:
                raise ValueError(f"{coin}: conflicting funding settlement")
            result[bucket] = value
        time.sleep(.10)
    return result


def capture(output, days):
    now = int(time.time() * 1000)
    # Past hour is fully settled even around a provider's funding timestamp jitter.
    end = now // HOUR * HOUR
    begin = end - days * DAY
    first = begin - WARMUP * BAR
    if output.exists():
        raise ValueError("output already exists; captures are not overwritten")
    ranked = ranking(post({"type": "metaAndAssetCtxs"}))
    print(f"Captured universe {iso(now)}: " + ", ".join(c for c, _ in ranked), flush=True)
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with closing(sqlite3.connect(output)) as db, db:
            db.executescript("""
              CREATE TABLE capture (key TEXT PRIMARY KEY, value TEXT NOT NULL);
              CREATE TABLE markets (rank INTEGER PRIMARY KEY, coin TEXT UNIQUE, day_notional REAL,
                                    bars INTEGER, funding_events INTEGER, error TEXT);
              CREATE TABLE candles (coin TEXT, timeframe TEXT, timestamp_ms INTEGER, open REAL, high REAL,
                                    low REAL, close REAL, volume REAL, PRIMARY KEY (coin,timeframe,timestamp_ms));
              CREATE TABLE funding (coin TEXT, hour INTEGER, observed_ms INTEGER, rate REAL, PRIMARY KEY(coin,hour));
            """)
            for rank, (coin, notional) in enumerate(ranked, 1):
                candles, funding, error = {}, {}, None
                try:
                    candles = fetch_candles(coin, first, end)
                    funding = fetch_funding(coin, first, end)
                except (KeyError, ValueError, TypeError, OSError, OverflowError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                db.executemany("INSERT INTO candles VALUES(?,?,?,?,?,?,?,?)",
                               ((coin, INTERVAL, t, *values) for t, values in candles.items()))
                db.executemany("INSERT INTO funding VALUES(?,?,?,?)", ((coin, h, *values) for h, values in funding.items()))
                db.execute("INSERT INTO markets VALUES(?,?,?,?,?,?)", (rank, coin, notional, len(candles), len(funding), error))
                print(f"{rank:2} {coin:<8} {len(candles)} candles; {len(funding)} funding" + (f"; {error}" if error else ""), flush=True)
            metadata = {"format": 1, "source": INFO, "ranking": "native active crypto perps by 24h USD notional at capture",
                        "captured_ms": now, "first_ms": first, "start_ms": begin, "end_ms": end, "interval": INTERVAL}
            db.executemany("INSERT INTO capture VALUES(?,?)", ((k, json.dumps(v)) for k, v in metadata.items()))
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--days", type=int, default=42)
    args = parser.parse_args()
    if not 21 <= args.days <= 50:
        parser.error("use 21-50 days so the requested 15m data and warm-up fit the API retention")
    capture(args.out.resolve(), args.days)


if __name__ == "__main__":
    main()
