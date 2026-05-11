#!/usr/bin/env python3
"""Backfill 1m kline candles for the regime detector's coin universe.

The regime detector takes a weighted vote across BTC + 9 majors. The
replay harness needs each coin in `data/candle_cache.db` to reproduce
the multi-coin regime calculation. This script downloads daily 1m
kline files from Binance Vision (public archive, no API key needed),
parses them, and imports into the candle cache.

By default it downloads USDC quote pairs; if a coin isn't listed with
USDC it falls back to USDT.

Usage:
    python scripts/backfill_multi_coin_klines.py \\
        --coins ETH,SOL,DOGE,AVAX,LINK,SUI \\
        --start 2025-04-05 --end 2026-05-09

For the full regime universe (excluding BTC which we already have):
    python scripts/backfill_multi_coin_klines.py \\
        --coins ETH,SOL,DOGE,ARB,AVAX,MATIC,LINK,OP,SUI \\
        --start 2025-04-05 --end 2026-05-09
"""
from __future__ import annotations

import argparse
import io
import logging
import sqlite3
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Binance Vision URL templates
SPOT_URL_TPL = "https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/{symbol}-1m-{date}.zip"

# Some coins rebranded; map to actual Binance symbol prefixes.
COIN_SYMBOL_OVERRIDE = {
    "MATIC": "POL",   # MATIC -> POL in Sept 2024
}

QUOTE_FALLBACKS = ("USDC", "USDT")

logger = logging.getLogger("backfill")


def _resolve_symbol(coin: str, quote: str) -> str:
    base = COIN_SYMBOL_OVERRIDE.get(coin.upper(), coin.upper())
    return f"{base}{quote}"


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _fetch_one_day(symbol: str, d: date, session: requests.Session) -> Optional[bytes]:
    """Return raw CSV bytes for `symbol` on date `d`, or None if not found."""
    url = SPOT_URL_TPL.format(symbol=symbol, date=d.strftime("%Y-%m-%d"))
    try:
        r = session.get(url, timeout=30)
    except requests.RequestException as e:
        logger.debug("%s %s: request error %s", symbol, d, e)
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        logger.warning("%s %s: HTTP %d", symbol, d, r.status_code)
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not names:
                return None
            return zf.read(names[0])
    except zipfile.BadZipFile:
        logger.warning("%s %s: bad zip", symbol, d)
        return None


def _parse_csv_rows(raw: bytes) -> Iterable[Tuple[int, float, float, float, float, float]]:
    """Yield (open_time_ms, o, h, l, c, v) from a Binance daily 1m CSV.

    Files are headerless. Columns:
      0 open_time (ms or us), 1 open, 2 high, 3 low, 4 close, 5 volume, ...
    Earlier 1s files were microsecond timestamps; 1m files are millisecond.
    Detect by magnitude.
    """
    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        # Some recent files have a header row -- sniff first cell.
        first = parts[0].strip()
        try:
            ts = int(first)
        except ValueError:
            continue
        # 1m files use ms timestamps (13 digits). Older 1s files used us (16).
        if ts > 1_000_000_000_000_000:
            ts //= 1000
        try:
            yield (ts, float(parts[1]), float(parts[2]), float(parts[3]),
                   float(parts[4]), float(parts[5]))
        except ValueError:
            continue


def _ensure_cache_schema(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                coin TEXT NOT NULL, timeframe TEXT NOT NULL, timestamp_ms INTEGER NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (coin, timeframe, timestamp_ms)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles (coin, timeframe, timestamp_ms)")
        conn.commit()


def _insert_rows(db_path: str, coin: str, rows: list) -> int:
    if not rows:
        return 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cur = conn.executemany(
            """INSERT OR IGNORE INTO candles
               (coin, timeframe, timestamp_ms, open, high, low, close, volume)
               VALUES (?, '1m', ?, ?, ?, ?, ?, ?)""",
            [(coin, ts, o, h, l, c, v) for (ts, o, h, l, c, v) in rows],
        )
        conn.commit()
        return cur.rowcount or 0


def backfill_coin(
    coin: str, start: date, end: date, db_path: str,
    session: requests.Session,
) -> dict:
    """Download + import 1m daily klines for one coin across the range.

    Returns a small stats dict.
    """
    logger.info("Backfilling %s: %s -> %s", coin, start, end)
    chosen_symbol: Optional[str] = None
    days_processed = 0
    days_missing = 0
    rows_inserted = 0
    days_total = (end - start).days + 1

    for i, d in enumerate(_daterange(start, end)):
        raw: Optional[bytes] = None
        symbol = chosen_symbol
        if symbol is None:
            for quote in QUOTE_FALLBACKS:
                candidate = _resolve_symbol(coin, quote)
                raw = _fetch_one_day(candidate, d, session)
                if raw is not None:
                    chosen_symbol = candidate
                    symbol = candidate
                    logger.info("  resolved symbol: %s -> %s", coin, symbol)
                    break
            if raw is None:
                days_missing += 1
                continue
        else:
            raw = _fetch_one_day(symbol, d, session)
            if raw is None:
                days_missing += 1
                continue

        rows = list(_parse_csv_rows(raw))
        inserted = _insert_rows(db_path, coin.upper(), rows)
        rows_inserted += inserted
        days_processed += 1

        if (i + 1) % 30 == 0 or (i + 1) == days_total:
            logger.info("  %s: %d/%d days processed (%d rows so far)",
                        coin, i + 1, days_total, rows_inserted)

    return {
        "coin": coin,
        "symbol": chosen_symbol,
        "days_total": days_total,
        "days_processed": days_processed,
        "days_missing": days_missing,
        "rows_inserted": rows_inserted,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coins", required=True, help="Comma-separated coin list, e.g. ETH,SOL,DOGE")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--cache-db", default="data/candle_cache.db",
                   help="Path to candle cache (default data/candle_cache.db)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end_d < start_d:
        logger.error("end before start"); return 2

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    _ensure_cache_schema(args.cache_db)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "hyperliquid-trading-bot/replay-backfill"})

    t0 = time.time()
    all_stats = []
    for coin in coins:
        try:
            stats = backfill_coin(coin, start_d, end_d, args.cache_db, sess)
        except Exception as e:
            logger.error("Backfill failed for %s: %s", coin, e)
            stats = {"coin": coin, "error": str(e)}
        all_stats.append(stats)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("  MULTI-COIN BACKFILL SUMMARY")
    print("=" * 70)
    print(f"  Window: {args.start} -> {args.end}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()
    print(f"  {'Coin':<8} {'Symbol':<12} {'Days':>6} {'Got':>6} {'Miss':>6} {'Rows':>10}")
    print("  " + "-" * 60)
    for s in all_stats:
        if "error" in s:
            print(f"  {s['coin']:<8} ERROR: {s['error']}")
            continue
        print(f"  {s['coin']:<8} {s.get('symbol', '?'):<12} "
              f"{s.get('days_total', 0):>6} {s.get('days_processed', 0):>6} "
              f"{s.get('days_missing', 0):>6} {s.get('rows_inserted', 0):>10,}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
