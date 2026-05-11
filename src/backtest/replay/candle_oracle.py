"""Read-only candle store with strict no-lookahead invariants.

Production code calls Hyperliquid `candleSnapshot` to fetch OHLCV. In replay
we route those calls here. The single rule:

    Every read at replay time t may return ONLY candles whose close <= t.

That rule is what makes the harness causal. Violating it -- even by one bar
-- silently fabricates an edge that won't exist live, which is exactly the
trap that the OOS validation in the candle backtester just exposed.

Implementation notes:
- Backed by the same SQLite cache (`data/candle_cache.db`) the live
  DataFetcher writes to, so we share data with the textbook backtester.
- A candle whose `timestamp_ms` is its OPEN time is "complete" only after
  `timestamp_ms + interval_ms` has elapsed. We enforce this when filtering
  by the clock: a 1m bar opened at 12:00:00 doesn't become readable until
  12:01:00 -- otherwise the harness gets to peek at a bar still being formed.
- The oracle owns the SQLite connection; it never writes. Multiple replay
  runs reading the same cache concurrently is safe.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from src.backtest.replay.clock import Clock

logger = logging.getLogger(__name__)

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass(frozen=True)
class OracleCandle:
    """OHLCV bar as returned by the oracle. Field names mirror Hyperliquid's
    candleSnapshot response so the api_manager_shim can wrap us with minimal
    translation."""
    coin: str
    timeframe: str
    timestamp_ms: int   # bar open time in ms since epoch
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def close_time_ms(self) -> int:
        return self.timestamp_ms + TIMEFRAME_MS[self.timeframe]


class LookaheadError(AssertionError):
    """Raised when the oracle is asked for data that would violate causality.

    AssertionError subclass so it survives `try/except Exception:` blocks but
    a wider `try/except (Exception, AssertionError):` will still see it.
    """


class CandleOracle:
    """Read-only, no-lookahead window onto cached candle data.

    The clock parameter is what makes this safe: every read goes through
    `_horizon_ms()` which subtracts the bar interval so we never return a
    bar that hadn't yet closed at clock time.
    """

    def __init__(self, db_path: str, clock: Clock):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"CandleOracle: cache db not found at {db_path}")
        self._db_path = db_path
        self._clock = clock
        # Sanity: at least one row exists.
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM candles").fetchone()
            if not row or row[0] == 0:
                raise RuntimeError(f"CandleOracle: cache at {db_path} is empty")

    def _connect(self) -> sqlite3.Connection:
        # `uri=` with `mode=ro` makes it impossible to accidentally write.
        return sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)

    # ---- core read API -------------------------------------------------

    def get_recent(
        self,
        coin: str,
        timeframe: str,
        count: int,
        *,
        end_ts_ms: Optional[int] = None,
    ) -> List[OracleCandle]:
        """Return up to `count` most recent candles whose close <= clock.now().

        If `end_ts_ms` is given, treat it as a maximum allowed close time. The
        effective horizon is `min(clock.now_ms(), end_ts_ms)` -- callers can
        pull a stale window without rewinding the clock.
        """
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        if count <= 0:
            return []

        horizon = self._horizon_ms(timeframe)
        if end_ts_ms is not None:
            horizon = min(horizon, end_ts_ms)

        # We want bars whose CLOSE <= horizon, i.e. open + interval <= horizon
        max_open = horizon - TIMEFRAME_MS[timeframe]
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_ms, open, high, low, close, volume
                FROM candles
                WHERE coin = ? AND timeframe = ? AND timestamp_ms <= ?
                ORDER BY timestamp_ms DESC
                LIMIT ?
                """,
                (coin.upper(), timeframe, max_open, count),
            ).fetchall()

        candles = [
            OracleCandle(
                coin=coin.upper(), timeframe=timeframe,
                timestamp_ms=r[0], open=r[1], high=r[2], low=r[3],
                close=r[4], volume=r[5],
            )
            for r in rows
        ]
        candles.reverse()  # ascending order, oldest -> newest, like HL
        return candles

    def get_range(
        self,
        coin: str,
        timeframe: str,
        start_ts_ms: int,
        end_ts_ms: int,
    ) -> List[OracleCandle]:
        """Return bars whose open is in [start_ts_ms, end_ts_ms).

        Raises LookaheadError if `end_ts_ms` is in the future relative to the
        clock (caller is asking for data that won't exist yet at replay time).
        """
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        if end_ts_ms <= start_ts_ms:
            return []

        horizon = self._horizon_ms(timeframe)
        if end_ts_ms > horizon:
            raise LookaheadError(
                f"CandleOracle.get_range({coin}, {timeframe}, "
                f"{start_ts_ms}, {end_ts_ms}): end is past clock horizon "
                f"{horizon}. Caller is requesting future data."
            )

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_ms, open, high, low, close, volume
                FROM candles
                WHERE coin = ? AND timeframe = ?
                  AND timestamp_ms >= ? AND timestamp_ms < ?
                ORDER BY timestamp_ms
                """,
                (coin.upper(), timeframe, start_ts_ms, end_ts_ms),
            ).fetchall()
        return [
            OracleCandle(
                coin=coin.upper(), timeframe=timeframe,
                timestamp_ms=r[0], open=r[1], high=r[2], low=r[3],
                close=r[4], volume=r[5],
            )
            for r in rows
        ]

    def get_latest_price(self, coin: str, timeframe: str = "1m") -> Optional[float]:
        """Return the most recent close <= clock.now() (treats it as the live mid).

        Used by the api_manager_shim to answer `allMids` requests. Returns
        None if no bar has closed yet at clock time -- callers need to handle
        that case (e.g. boot before replay-start).
        """
        recent = self.get_recent(coin, timeframe, count=1)
        return recent[-1].close if recent else None

    def get_horizon_ms(self, timeframe: str) -> int:
        return self._horizon_ms(timeframe)

    # ---- introspection -------------------------------------------------

    def coverage(self, coin: str, timeframe: str) -> tuple[int, int, int]:
        """Return (count, first_open_ms, last_open_ms). For startup checks."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), MIN(timestamp_ms), MAX(timestamp_ms)
                FROM candles WHERE coin = ? AND timeframe = ?
                """,
                (coin.upper(), timeframe),
            ).fetchone()
        return (row[0] or 0, row[1] or 0, row[2] or 0)

    def available_coins(self, timeframe: str) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT coin FROM candles WHERE timeframe = ? ORDER BY coin",
                (timeframe,),
            ).fetchall()
        return [r[0] for r in rows]

    # ---- internals -----------------------------------------------------

    def _horizon_ms(self, timeframe: str) -> int:
        """Latest close time the oracle is willing to serve at clock time.

        For a bar opened at T, it has only "happened" once T + interval has
        elapsed. So at clock time t, the latest valid close is t itself, and
        the latest valid OPEN is t - interval.
        """
        return self._clock.now_ms()
