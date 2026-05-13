"""Backfill Binance Vision spot klines into the local candle cache.

Designed for a daily Railway cron:

    python scripts/backfill_multi_coin_klines.py --coins BTC,ETH,SOL --days 3
    python scripts/backfill_multi_coin_klines.py --coins BTC --timeframe 1s --days 1
"""

from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import requests

BINANCE_VISION_URL = (
    "https://data.binance.vision/data/spot/daily/klines/"
    "{symbol}/{timeframe}/{symbol}-{timeframe}-{day}.zip"
)
TIMEFRAME_MS = {"1s": 1_000, "1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


@dataclass
class KlineRow:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iter_days(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _symbol_for_coin(coin: str, quote: str) -> str:
    coin = str(coin or "").strip().upper()
    quote = str(quote or "USDC").strip().upper()
    if coin.endswith(quote):
        return coin
    return f"{coin}{quote}"


def _cache_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            coin TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT DEFAULT 'binance_vision',
            PRIMARY KEY (coin, timeframe, timestamp_ms)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_candles_lookup
        ON candles (coin, timeframe, timestamp_ms)
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(candles)").fetchall()}
    if "source" not in cols:
        conn.execute("ALTER TABLE candles ADD COLUMN source TEXT DEFAULT 'unknown'")
    conn.commit()


def _parse_zip_payload(payload: bytes) -> List[KlineRow]:
    rows: List[KlineRow] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            return rows
        with archive.open(names[0]) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            reader = csv.reader(text)
            for raw in reader:
                if len(raw) < 6 or raw[0] == "open_time":
                    continue
                try:
                    rows.append(
                        KlineRow(
                            timestamp_ms=int(raw[0]),
                            open=float(raw[1]),
                            high=float(raw[2]),
                            low=float(raw[3]),
                            close=float(raw[4]),
                            volume=float(raw[5]),
                        )
                    )
                except (TypeError, ValueError):
                    continue
    return rows


def _download_day(symbol: str, timeframe: str, day: date, timeout: int = 30) -> List[KlineRow]:
    url = BINANCE_VISION_URL.format(
        symbol=symbol,
        timeframe=timeframe,
        day=day.strftime("%Y-%m-%d"),
    )
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return _parse_zip_payload(resp.content)


def _store_rows(
    conn: sqlite3.Connection,
    *,
    coin: str,
    timeframe: str,
    rows: Iterable[KlineRow],
) -> int:
    payload = [
        (
            coin,
            timeframe,
            row.timestamp_ms,
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
            "binance_vision",
        )
        for row in rows
    ]
    if not payload:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO candles
        (coin, timeframe, timestamp_ms, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def backfill(
    *,
    coins: Iterable[str],
    start: date,
    end: date,
    timeframe: str,
    cache_db: str,
    quote: str = "USDC",
    request_sleep_s: float = 0.15,
) -> dict:
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe {timeframe}; use one of {sorted(TIMEFRAME_MS)}")
    db_path = Path(cache_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        _cache_schema(conn)
        stats = {"cache_db": str(db_path), "timeframe": timeframe, "coins": {}, "inserted": 0}
        for coin_raw in coins:
            coin = str(coin_raw or "").strip().upper()
            if not coin:
                continue
            symbol = _symbol_for_coin(coin, quote)
            coin_stats = {"symbol": symbol, "days": 0, "inserted": 0, "missing_days": 0}
            for day in _iter_days(start, end):
                rows = _download_day(symbol, timeframe, day)
                if not rows:
                    coin_stats["missing_days"] += 1
                    continue
                inserted = _store_rows(conn, coin=coin, timeframe=timeframe, rows=rows)
                conn.commit()
                coin_stats["days"] += 1
                coin_stats["inserted"] += inserted
                stats["inserted"] += inserted
                time.sleep(max(0.0, request_sleep_s))
            stats["coins"][coin] = coin_stats
        return stats
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", default="BTC,ETH,SOL")
    parser.add_argument("--timeframe", default="1m", choices=sorted(TIMEFRAME_MS))
    parser.add_argument("--quote", default="USDC")
    parser.add_argument("--cache-db", default="data/candle_cache.db")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--request-sleep-s", type=float, default=0.15)
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    end = _parse_day(args.end) if args.end else today - timedelta(days=1)
    start = _parse_day(args.start) if args.start else end - timedelta(days=max(1, args.days) - 1)
    coins = [part.strip().upper() for part in args.coins.split(",") if part.strip()]
    stats = backfill(
        coins=coins,
        start=start,
        end=end,
        timeframe=args.timeframe,
        cache_db=args.cache_db,
        quote=args.quote,
        request_sleep_s=args.request_sleep_s,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
