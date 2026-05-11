"""No-lookahead invariant tests for the replay foundation.

These are the proofs that the harness can never silently see the future.
If any of these fail, every replay result is suspect.
"""
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from src.backtest.replay.clock import LiveClock, ReplayClock
from src.backtest.replay.candle_oracle import (
    CandleOracle, LookaheadError, OracleCandle, TIMEFRAME_MS,
)


# --- LiveClock --------------------------------------------------------

def test_live_clock_now_is_close_to_system_now():
    """LiveClock should track datetime.now() within a small slack."""
    clk = LiveClock()
    drift = (datetime.now(timezone.utc) - clk.now()).total_seconds()
    assert abs(drift) < 1.0


def test_live_clock_iso_has_z_suffix():
    assert LiveClock().now_iso().endswith("Z")


# --- ReplayClock ------------------------------------------------------

def test_replay_clock_unset_raises():
    """A ReplayClock that was never `set` must refuse to be read.

    Without this guard the harness silently uses t=0 (1970) on misuse.
    """
    clk = ReplayClock()
    with pytest.raises(RuntimeError):
        clk.now()
    with pytest.raises(RuntimeError):
        clk.now_ms()
    with pytest.raises(RuntimeError):
        clk.advance(1000)


def test_replay_clock_set_and_advance():
    clk = ReplayClock(start_ts_ms=1_700_000_000_000)
    assert clk.now_ms() == 1_700_000_000_000
    assert clk.now() == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
    clk.advance(60_000)
    assert clk.now_ms() == 1_700_000_060_000


def test_replay_clock_rejects_negative_advance():
    """Negative advance would be a backward time-step -- catch it loudly."""
    clk = ReplayClock(start_ts_ms=1_700_000_000_000)
    with pytest.raises(ValueError):
        clk.advance(-1)


def test_replay_clock_rejects_non_int_set():
    clk = ReplayClock()
    with pytest.raises(TypeError):
        clk.set(1700000000.5)  # float is not OK -- ms must be exact


# --- CandleOracle: setup helpers -------------------------------------

def _make_cache_db(tmp_path, *bars):
    """Create a minimal candle_cache.db with the given (coin, tf, ts_ms, ohlcv) bars."""
    db = tmp_path / "candle_cache.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE candles (
            coin TEXT NOT NULL, timeframe TEXT NOT NULL, timestamp_ms INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (coin, timeframe, timestamp_ms)
        )
    """)
    conn.executemany(
        "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        bars,
    )
    conn.commit()
    conn.close()
    return str(db)


def _bar(coin, tf, open_ms, close_price):
    return (coin, tf, open_ms, close_price, close_price + 1, close_price - 1, close_price, 1.0)


# --- CandleOracle: causality ------------------------------------------

def test_oracle_get_recent_excludes_future_bars(tmp_path):
    """A bar that hasn't yet closed at clock time must NOT be returned."""
    # Bars at 12:00, 12:01, 12:02 (1m)
    bars = [_bar("BTC", "1m", t, 100.0 + i) for i, t in enumerate([
        1_700_000_000_000, 1_700_000_060_000, 1_700_000_120_000,
    ])]
    db = _make_cache_db(tmp_path, *bars)

    # Clock at 12:01:30 -- bars at 12:00 and 12:01 are closed (close at 12:01
    # and 12:02 respectively, only 12:01-close <= 12:01:30 ... wait no:
    # 12:01 bar opens at 12:01 closes at 12:02. So at 12:01:30 only 12:00 bar
    # has closed (close at 12:01).
    clk = ReplayClock(start_ts_ms=1_700_000_090_000)  # 12:01:30
    oracle = CandleOracle(db, clk)
    recent = oracle.get_recent("BTC", "1m", count=10)
    # Only the 12:00 bar should be visible
    assert len(recent) == 1
    assert recent[0].timestamp_ms == 1_700_000_000_000


def test_oracle_get_recent_includes_just_closed_bar(tmp_path):
    """At T = open + interval exactly, the bar IS closed and should be visible."""
    bars = [_bar("BTC", "1m", t, 100.0 + i) for i, t in enumerate([
        1_700_000_000_000, 1_700_000_060_000,
    ])]
    db = _make_cache_db(tmp_path, *bars)
    # Clock right at 12:01:00 -- the 12:00 bar (closes at 12:01) is just now closed
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    recent = oracle.get_recent("BTC", "1m", count=10)
    assert len(recent) == 1
    assert recent[0].timestamp_ms == 1_700_000_000_000


def test_oracle_advancing_clock_reveals_more_bars(tmp_path):
    """As the clock advances, the oracle returns progressively more bars."""
    opens = [1_700_000_000_000 + i * 60_000 for i in range(5)]
    bars = [_bar("BTC", "1m", t, 100.0) for t in opens]
    db = _make_cache_db(tmp_path, *bars)

    clk = ReplayClock(start_ts_ms=opens[0] + 60_000)  # 1 bar visible
    oracle = CandleOracle(db, clk)
    counts = []
    for _ in range(5):
        counts.append(len(oracle.get_recent("BTC", "1m", count=10)))
        clk.advance(60_000)
    # Each advance reveals exactly one more bar (until we run out of cache).
    assert counts == [1, 2, 3, 4, 5]


def test_oracle_get_range_raises_on_future(tmp_path):
    """Asking for a range whose end is past the clock horizon is a hard error."""
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 100.0)]
    db = _make_cache_db(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)  # 12:01:00
    oracle = CandleOracle(db, clk)
    with pytest.raises(LookaheadError):
        oracle.get_range("BTC", "1m", 1_700_000_000_000, 1_700_000_120_000)


def test_oracle_get_range_returns_only_within_clock(tmp_path):
    opens = [1_700_000_000_000 + i * 60_000 for i in range(5)]
    bars = [_bar("BTC", "1m", t, 100.0 + i) for i, t in enumerate(opens)]
    db = _make_cache_db(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=opens[3] + 60_000)  # bars 0..3 closed
    oracle = CandleOracle(db, clk)
    rng = oracle.get_range("BTC", "1m", opens[0], opens[3] + 60_000)
    assert [b.timestamp_ms for b in rng] == opens[:4]


def test_oracle_latest_price_returns_most_recent_close(tmp_path):
    opens = [1_700_000_000_000 + i * 60_000 for i in range(3)]
    bars = [
        _bar("BTC", "1m", opens[0], 100.0),
        _bar("BTC", "1m", opens[1], 200.0),
        _bar("BTC", "1m", opens[2], 300.0),
    ]
    db = _make_cache_db(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=opens[2] + 60_000)  # all 3 closed
    oracle = CandleOracle(db, clk)
    assert oracle.get_latest_price("BTC", "1m") == 300.0
    # Rewinding the clock returns the older close
    clk.set(opens[1] + 60_000)
    assert oracle.get_latest_price("BTC", "1m") == 200.0


def test_oracle_latest_price_none_before_first_close(tmp_path):
    """Before any bar has closed, get_latest_price returns None (not a stale guess)."""
    bars = [_bar("BTC", "1m", 1_700_000_060_000, 100.0)]
    db = _make_cache_db(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)  # bar opens AT clock time, not yet closed
    oracle = CandleOracle(db, clk)
    assert oracle.get_latest_price("BTC", "1m") is None


def test_oracle_does_not_leak_other_coins(tmp_path):
    bars = [
        _bar("BTC", "1m", 1_700_000_000_000, 100.0),
        _bar("ETH", "1m", 1_700_000_000_000, 50.0),
    ]
    db = _make_cache_db(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    btc = oracle.get_recent("BTC", "1m", count=10)
    assert all(b.coin == "BTC" for b in btc)
    eth = oracle.get_recent("ETH", "1m", count=10)
    assert all(b.coin == "ETH" for b in eth)


def test_oracle_does_not_mix_timeframes(tmp_path):
    bars = [
        _bar("BTC", "1m", 1_700_000_000_000, 100.0),
        _bar("BTC", "1h", 1_700_000_000_000, 100.0),
    ]
    db = _make_cache_db(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_004_000_000)
    oracle = CandleOracle(db, clk)
    one_min = oracle.get_recent("BTC", "1m", count=10)
    one_hour = oracle.get_recent("BTC", "1h", count=10)
    assert all(b.timeframe == "1m" for b in one_min)
    assert all(b.timeframe == "1h" for b in one_hour)


def test_oracle_empty_cache_rejected(tmp_path):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE candles (
        coin TEXT, timeframe TEXT, timestamp_ms INTEGER,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY (coin, timeframe, timestamp_ms))""")
    conn.commit()
    conn.close()
    clk = ReplayClock(start_ts_ms=1_700_000_000_000)
    with pytest.raises(RuntimeError):
        CandleOracle(str(db), clk)


def test_oracle_missing_db_file_rejected(tmp_path):
    clk = ReplayClock(start_ts_ms=1_700_000_000_000)
    with pytest.raises(FileNotFoundError):
        CandleOracle(str(tmp_path / "does_not_exist.db"), clk)
