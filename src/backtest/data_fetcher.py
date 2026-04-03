"""
Historical Data Fetcher
=======================
Pulls OHLCV candle data from Hyperliquid's API and imports CSV/JSON files.
All data is cached locally in SQLite so you never re-download the same period.

Supported sources:
  - Hyperliquid candles API (1m, 5m, 15m, 1h, 4h, 1d)
  - CSV import (TradingView, Binance export format, generic OHLCV)
  - JSON import (array of candle objects)

Usage:
    fetcher = DataFetcher()

    # Pull from Hyperliquid
    candles = fetcher.fetch_candles("BTC", "1h", start="2025-01-01", end="2025-03-01")

    # Import from CSV
    candles = fetcher.import_csv("data/btc_1h.csv", coin="BTC", timeframe="1h")

    # Check cache
    cached = fetcher.list_cached()
"""
import os
import csv
import json
import time
import sqlite3
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("data_fetcher")

# Hyperliquid candle API
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# Supported timeframes → milliseconds
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Hyperliquid API interval strings
HL_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Max candles per API request (HL limit)
HL_MAX_CANDLES = 5000


@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    coin: str = ""
    timeframe: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _parse_date(s: str) -> int:
    """Parse date string to milliseconds. Accepts YYYY-MM-DD or ISO format."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")


class DataFetcher:
    """
    Fetches and caches historical candle data.
    Cache is stored in SQLite at data/candle_cache.db.
    """

    def __init__(self, cache_dir: str = "data"):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._db_path = os.path.join(cache_dir, "candle_cache.db")
        self._init_cache_db()

    def _init_cache_db(self):
        """Create cache tables if they don't exist."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (coin, timeframe, timestamp_ms)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT,
                timeframe TEXT,
                start_ms INTEGER,
                end_ms INTEGER,
                candle_count INTEGER,
                source TEXT,
                fetched_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_lookup
            ON candles (coin, timeframe, timestamp_ms)
        """)
        conn.commit()
        conn.close()

    # ─── Hyperliquid API ─────────────────────────────────────────

    def fetch_candles(self, coin: str, timeframe: str,
                      start: str = None, end: str = None,
                      use_cache: bool = True) -> List[Candle]:
        """
        Fetch candles from Hyperliquid API with local caching.

        Args:
            coin: Asset symbol (e.g. "BTC", "ETH")
            timeframe: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
            start: Start date (YYYY-MM-DD). Default: 90 days ago
            end: End date (YYYY-MM-DD). Default: now
            use_cache: If True, serve from cache when available

        Returns:
            List of Candle objects, sorted by timestamp ascending
        """
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {list(TIMEFRAME_MS.keys())}")

        coin = coin.upper()
        now_ms = int(time.time() * 1000)

        start_ms = _parse_date(start) if start else now_ms - (90 * 86_400_000)
        end_ms = _parse_date(end) if end else now_ms

        # Check cache first
        if use_cache:
            cached = self._get_cached(coin, timeframe, start_ms, end_ms)
            expected = (end_ms - start_ms) // TIMEFRAME_MS[timeframe]
            if cached and len(cached) >= expected * 0.95:  # 95% coverage = use cache
                logger.info(f"Cache hit: {coin} {timeframe} — {len(cached)} candles")
                return cached

        # Fetch from API in chunks
        logger.info(f"Fetching {coin} {timeframe} candles from Hyperliquid "
                    f"({start or '90d ago'} -> {end or 'now'})...")

        all_candles = []
        chunk_start = start_ms
        interval_ms = TIMEFRAME_MS[timeframe]
        request_count = 0

        while chunk_start < end_ms:
            chunk_end = min(chunk_start + HL_MAX_CANDLES * interval_ms, end_ms)

            try:
                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": HL_INTERVALS[timeframe],
                        "startTime": chunk_start,
                        "endTime": chunk_end,
                    }
                }
                resp = requests.post(HL_INFO_URL, json=payload, timeout=30)
                resp.raise_for_status()
                raw = resp.json()

                if not raw:
                    break

                for c in raw:
                    ts = c.get("t", c.get("T", 0))
                    candle = Candle(
                        timestamp_ms=int(ts),
                        open=float(c.get("o", c.get("O", 0))),
                        high=float(c.get("h", c.get("H", 0))),
                        low=float(c.get("l", c.get("L", 0))),
                        close=float(c.get("c", c.get("C", 0))),
                        volume=float(c.get("v", c.get("V", 0))),
                        coin=coin,
                        timeframe=timeframe,
                    )
                    all_candles.append(candle)

                request_count += 1

                # Rate limit: 1200 req/min on HL info endpoint
                if request_count % 10 == 0:
                    time.sleep(0.5)

                # Advance
                if raw:
                    last_ts = int(raw[-1].get("t", raw[-1].get("T", 0)))
                    chunk_start = last_ts + interval_ms
                else:
                    break

            except Exception as e:
                logger.error(f"Error fetching {coin} {timeframe} chunk: {e}")
                time.sleep(2)
                chunk_start = chunk_end  # Skip failed chunk

        # Deduplicate and sort
        seen = set()
        unique = []
        for c in all_candles:
            if c.timestamp_ms not in seen:
                seen.add(c.timestamp_ms)
                unique.append(c)
        unique.sort(key=lambda x: x.timestamp_ms)

        # Cache the results
        if unique:
            self._store_candles(unique, source="hyperliquid")
            logger.info(f"Fetched and cached {len(unique)} {coin} {timeframe} candles "
                       f"({request_count} API requests)")

        return unique

    # ─── CSV Import ──────────────────────────────────────────────

    def import_csv(self, filepath: str, coin: str = None,
                   timeframe: str = None) -> List[Candle]:
        """
        Import candle data from a CSV file.

        Auto-detects format from common exports:
        - TradingView: time,open,high,low,close,volume
        - Binance: open_time,open,high,low,close,volume,...
        - Generic: timestamp,open,high,low,close,volume

        Args:
            filepath: Path to CSV file
            coin: Override coin name (auto-detected from filename if None)
            timeframe: Override timeframe (auto-detected if None)

        Returns:
            List of Candle objects
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        # Auto-detect coin from filename (e.g. "BTC_1h.csv" → "BTC")
        basename = os.path.splitext(os.path.basename(filepath))[0].upper()
        if coin is None:
            parts = basename.replace("-", "_").split("_")
            coin = parts[0] if parts else "UNKNOWN"
        coin = coin.upper()

        # Auto-detect timeframe from filename
        if timeframe is None:
            for tf in TIMEFRAME_MS:
                if tf.upper() in basename:
                    timeframe = tf
                    break
            if timeframe is None:
                timeframe = "1h"  # Default assumption
                logger.warning(f"Could not detect timeframe from filename, defaulting to {timeframe}")

        candles = []

        with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError("CSV file is empty")

            # Normalize header
            header_lower = [h.strip().lower().replace(" ", "_") for h in header]

            # Detect column mapping
            col_map = self._detect_csv_columns(header_lower)

            for row_num, row in enumerate(reader, start=2):
                try:
                    ts = self._parse_csv_timestamp(row[col_map["time"]])
                    candle = Candle(
                        timestamp_ms=ts,
                        open=float(row[col_map["open"]]),
                        high=float(row[col_map["high"]]),
                        low=float(row[col_map["low"]]),
                        close=float(row[col_map["close"]]),
                        volume=float(row[col_map["volume"]]) if col_map.get("volume") is not None else 0,
                        coin=coin,
                        timeframe=timeframe,
                    )
                    candles.append(candle)
                except (ValueError, IndexError) as e:
                    if row_num <= 5:
                        logger.warning(f"Skipping CSV row {row_num}: {e}")
                    continue

        candles.sort(key=lambda x: x.timestamp_ms)

        if candles:
            self._store_candles(candles, source=f"csv:{os.path.basename(filepath)}")
            logger.info(f"Imported {len(candles)} candles from {filepath} "
                       f"({coin} {timeframe})")

        return candles

    def _detect_csv_columns(self, header: List[str]) -> Dict[str, int]:
        """Detect column indices for OHLCV from header names."""
        mapping = {}

        # Time column
        for i, h in enumerate(header):
            if h in ("time", "timestamp", "date", "datetime", "open_time",
                     "close_time", "t", "unix", "epoch"):
                mapping["time"] = i
                break
        if "time" not in mapping:
            mapping["time"] = 0  # Assume first column

        # OHLCV
        for field, aliases in [
            ("open", ["open", "o", "open_price", "price_open"]),
            ("high", ["high", "h", "high_price", "price_high", "max"]),
            ("low", ["low", "l", "low_price", "price_low", "min"]),
            ("close", ["close", "c", "close_price", "price_close", "last"]),
            ("volume", ["volume", "v", "vol", "base_volume", "quote_volume"]),
        ]:
            for i, h in enumerate(header):
                if h in aliases:
                    mapping[field] = i
                    break

        # Fallback: assume standard TOHLCV column order
        if "open" not in mapping and len(header) >= 5:
            start = mapping.get("time", 0) + 1
            mapping["open"] = start
            mapping["high"] = start + 1
            mapping["low"] = start + 2
            mapping["close"] = start + 3
            if len(header) > start + 4:
                mapping["volume"] = start + 4

        return mapping

    def _parse_csv_timestamp(self, val: str) -> int:
        """Parse various timestamp formats to milliseconds."""
        val = val.strip()

        # Pure numeric = unix timestamp
        try:
            num = float(val)
            if num > 1e12:
                return int(num)  # Already milliseconds
            elif num > 1e9:
                return int(num * 1000)  # Seconds → ms
            else:
                return int(num * 1000)  # Seconds
        except ValueError:
            pass

        # Date string
        return _parse_date(val)

    # ─── JSON Import ─────────────────────────────────────────────

    def import_json(self, filepath: str, coin: str = None,
                    timeframe: str = None) -> List[Candle]:
        """
        Import candle data from a JSON file.

        Expects an array of objects with at least: timestamp/t, o/open, h/high, l/low, c/close
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")

        basename = os.path.splitext(os.path.basename(filepath))[0].upper()
        if coin is None:
            parts = basename.replace("-", "_").split("_")
            coin = parts[0] if parts else "UNKNOWN"
        coin = coin.upper()

        if timeframe is None:
            for tf in TIMEFRAME_MS:
                if tf.upper() in basename:
                    timeframe = tf
                    break
            if timeframe is None:
                timeframe = "1h"

        with open(filepath, "r") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError("JSON must be an array of candle objects")

        candles = []
        for item in raw:
            try:
                ts = item.get("timestamp_ms") or item.get("t") or item.get("timestamp") or item.get("time") or 0
                if isinstance(ts, str):
                    ts = _parse_date(ts)
                elif ts < 1e12:
                    ts = int(ts * 1000)

                candle = Candle(
                    timestamp_ms=int(ts),
                    open=float(item.get("o") or item.get("open") or 0),
                    high=float(item.get("h") or item.get("high") or 0),
                    low=float(item.get("l") or item.get("low") or 0),
                    close=float(item.get("c") or item.get("close") or 0),
                    volume=float(item.get("v") or item.get("volume") or 0),
                    coin=coin,
                    timeframe=timeframe,
                )
                candles.append(candle)
            except (ValueError, TypeError) as e:
                continue

        candles.sort(key=lambda x: x.timestamp_ms)

        if candles:
            self._store_candles(candles, source=f"json:{os.path.basename(filepath)}")
            logger.info(f"Imported {len(candles)} candles from {filepath}")

        return candles

    # ─── Cache Layer ─────────────────────────────────────────────

    def _get_cached(self, coin: str, timeframe: str,
                    start_ms: int, end_ms: int) -> List[Candle]:
        """Retrieve candles from local cache."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            """SELECT timestamp_ms, open, high, low, close, volume
               FROM candles
               WHERE coin = ? AND timeframe = ? AND timestamp_ms >= ? AND timestamp_ms <= ?
               ORDER BY timestamp_ms""",
            (coin, timeframe, start_ms, end_ms)
        ).fetchall()
        conn.close()

        return [
            Candle(
                timestamp_ms=r[0], open=r[1], high=r[2],
                low=r[3], close=r[4], volume=r[5],
                coin=coin, timeframe=timeframe,
            )
            for r in rows
        ]

    def _store_candles(self, candles: List[Candle], source: str):
        """Store candles in local cache using bulk INSERT OR IGNORE."""
        if not candles:
            return

        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        # Bulk insert with INSERT OR IGNORE (skip duplicates)
        data = [
            (c.coin, c.timeframe, c.timestamp_ms, c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
        conn.executemany(
            """INSERT OR IGNORE INTO candles
               (coin, timeframe, timestamp_ms, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            data
        )

        # Log the fetch
        coin = candles[0].coin
        tf = candles[0].timeframe
        conn.execute(
            """INSERT INTO fetch_log (coin, timeframe, start_ms, end_ms, candle_count, source, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (coin, tf, candles[0].timestamp_ms, candles[-1].timestamp_ms,
             len(candles), source, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

    def list_cached(self) -> List[Dict]:
        """List all cached data summaries."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT coin, timeframe,
                   MIN(timestamp_ms) as start_ms,
                   MAX(timestamp_ms) as end_ms,
                   COUNT(*) as candle_count
            FROM candles
            GROUP BY coin, timeframe
            ORDER BY coin, timeframe
        """).fetchall()
        conn.close()

        result = []
        for r in rows:
            start_dt = datetime.fromtimestamp(r["start_ms"] / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(r["end_ms"] / 1000, tz=timezone.utc)
            result.append({
                "coin": r["coin"],
                "timeframe": r["timeframe"],
                "start": start_dt.strftime("%Y-%m-%d"),
                "end": end_dt.strftime("%Y-%m-%d"),
                "candles": r["candle_count"],
                "days": (r["end_ms"] - r["start_ms"]) / 86_400_000,
            })
        return result

    def clear_cache(self, coin: str = None, timeframe: str = None):
        """Clear cached data, optionally filtered by coin/timeframe."""
        conn = sqlite3.connect(self._db_path)
        if coin and timeframe:
            conn.execute("DELETE FROM candles WHERE coin = ? AND timeframe = ?",
                        (coin.upper(), timeframe))
        elif coin:
            conn.execute("DELETE FROM candles WHERE coin = ?", (coin.upper(),))
        else:
            conn.execute("DELETE FROM candles")
        conn.commit()
        conn.close()
        logger.info(f"Cache cleared: coin={coin}, timeframe={timeframe}")

    def get_cache_stats(self) -> Dict:
        """Get cache database size and stats."""
        conn = sqlite3.connect(self._db_path)
        total = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        coins = conn.execute("SELECT COUNT(DISTINCT coin) FROM candles").fetchone()[0]
        conn.close()

        db_size_mb = os.path.getsize(self._db_path) / (1024 * 1024) if os.path.exists(self._db_path) else 0

        return {
            "total_candles": total,
            "unique_coins": coins,
            "db_size_mb": round(db_size_mb, 2),
            "db_path": self._db_path,
        }
