"""Tests for the api_manager shim -- the chokepoint for HL traffic in replay mode.

Two safety guarantees being verified:
  1. The shim returns ONLY data <= clock time (no future leakage through HL calls)
  2. The shim raises on unknown request types instead of silently returning None
     (so a missed code path is loud, not silent).
"""
import sqlite3

import pytest

from src.backtest.replay.candle_oracle import CandleOracle
from src.backtest.replay.clock import ReplayClock
from src.backtest.replay.api_manager_shim import (
    ReplayAPIManager, ReplayInterceptError,
    install_replay_manager, uninstall_replay_manager,
)


def _make_cache(tmp_path, *bars):
    db = tmp_path / "candle_cache.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE candles (
        coin TEXT, timeframe TEXT, timestamp_ms INTEGER,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY (coin, timeframe, timestamp_ms))""")
    conn.executemany(
        "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", bars,
    )
    conn.commit()
    conn.close()
    return str(db)


def _bar(coin, tf, open_ms, close_price):
    return (coin, tf, open_ms, close_price, close_price + 1, close_price - 1, close_price, 1.0)


# --- Basic dispatch ----------------------------------------------------

def test_shim_routes_candle_snapshot_through_oracle(tmp_path):
    opens = [1_700_000_000_000 + i * 60_000 for i in range(5)]
    bars = [_bar("BTC", "1m", t, 100.0 + i) for i, t in enumerate(opens)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=opens[4] + 60_000)  # all 5 closed
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])

    out = shim.post({
        "type": "candleSnapshot",
        "req": {"coin": "BTC", "interval": "1m",
                "startTime": opens[0], "endTime": opens[4] + 60_000},
    })
    assert len(out) == 5
    # Returned in HL shape (string-typed prices, lowercase keys)
    assert all("t" in c and "o" in c and "c" in c for c in out)
    assert all(isinstance(c["o"], str) for c in out)


def test_shim_candle_snapshot_clamps_end_to_clock(tmp_path):
    """Caller asks for end_ts past clock -- shim silently clamps, doesn't leak."""
    opens = [1_700_000_000_000 + i * 60_000 for i in range(5)]
    bars = [_bar("BTC", "1m", t, 100.0 + i) for i, t in enumerate(opens)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=opens[2] + 60_000)  # only first 3 closed
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])

    far_future = opens[4] + 10 * 60_000
    out = shim.post({
        "type": "candleSnapshot",
        "req": {"coin": "BTC", "interval": "1m",
                "startTime": opens[0], "endTime": far_future},
    })
    # Only the first 3 bars should appear, regardless of what caller asked for
    assert len(out) == 3


def test_shim_all_mids_returns_string_typed_prices(tmp_path):
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 50_000.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])

    mids = shim.post({"type": "allMids"})
    assert mids == {"BTC": "50000.0"}
    # Hyperliquid returns prices as strings; downstream parses with float()
    assert all(isinstance(v, str) for v in mids.values())


def test_shim_meta_and_asset_ctxs_shape_matches_hl(tmp_path):
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 50_000.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"], funding_rate_8h=0.0001)

    out = shim.post({"type": "metaAndAssetCtxs"})
    assert isinstance(out, list) and len(out) == 2
    meta, ctxs = out
    assert "universe" in meta
    assert meta["universe"][0]["name"] == "BTC"
    assert ctxs[0]["funding"] == "0.0001"
    assert ctxs[0]["midPx"] == "50000.0"


def test_shim_l2_book_returns_levels(tmp_path):
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 100.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])
    book = shim.post({"type": "l2Book", "coin": "BTC"})
    assert "levels" in book
    assert len(book["levels"]) == 2
    bid_px = float(book["levels"][0][0]["px"])
    ask_px = float(book["levels"][1][0]["px"])
    assert bid_px < 100.0 < ask_px


# --- Safety guarantees -------------------------------------------------

def test_shim_strict_mode_raises_on_unknown_request_type(tmp_path):
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 100.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"], strict=True)
    with pytest.raises(ReplayInterceptError):
        shim.post({"type": "subscribeToOrderUpdates"})


def test_shim_non_strict_mode_returns_none_on_unknown(tmp_path):
    """Non-strict mode is a debugging escape hatch, not the default."""
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 100.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"], strict=False)
    assert shim.post({"type": "made_up_request"}) is None


def test_shim_unknown_coin_returns_empty_not_crash(tmp_path):
    """Asking for a coin not in the cache must not crash; tracking miss for telemetry."""
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 100.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])
    out = shim.post({
        "type": "candleSnapshot",
        "req": {"coin": "DOGE", "interval": "1h",
                "startTime": 1_700_000_000_000, "endTime": 1_700_000_060_000},
    })
    assert out == []
    stats = shim.get_stats()
    assert stats["coin_cache_misses"]["DOGE"] >= 1


def test_shim_telemetry_counts_calls_by_type(tmp_path):
    bars = [_bar("BTC", "1m", 1_700_000_000_000, 100.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])
    shim.post({"type": "allMids"})
    shim.post({"type": "allMids"})
    shim.post({"type": "metaAndAssetCtxs"})
    stats = shim.get_stats()
    assert stats["calls_by_type"]["allMids"] == 2
    assert stats["calls_by_type"]["metaAndAssetCtxs"] == 1


def test_install_uninstall_replay_manager(tmp_path):
    """Verify the singleton swap actually replaces what get_manager() returns."""
    import src.core.api_manager as am

    bars = [_bar("BTC", "1m", 1_700_000_000_000, 100.0)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=1_700_000_060_000)
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])

    install_replay_manager(shim)
    try:
        # After install, get_manager() should return the shim
        m = am.get_manager()
        assert m is shim
        assert m.post({"type": "allMids"}) == {"BTC": "100.0"}
    finally:
        uninstall_replay_manager()
        # Module is reset; next call would lazily build a real APIManager.
        assert am._manager is None


# --- Causality regression -- runs many ticks to stress the invariant --

def test_shim_at_every_tick_only_serves_past(tmp_path):
    """Slide the clock through 100 bars; at each tick allMids must equal that
    bar's close, never further. This is the tightest test of causality."""
    opens = [1_700_000_000_000 + i * 60_000 for i in range(101)]
    bars = [_bar("BTC", "1m", t, 1000.0 + i) for i, t in enumerate(opens)]
    db = _make_cache(tmp_path, *bars)
    clk = ReplayClock(start_ts_ms=opens[0])
    oracle = CandleOracle(db, clk)
    shim = ReplayAPIManager(oracle, clk, known_coins=["BTC"])

    for i in range(1, 101):
        # Clock at the close time of bar (i-1). At this instant, exactly i bars
        # are closed and bar (i-1) is the most recent.
        clk.set(opens[i])
        mids = shim.post({"type": "allMids"})
        # Bar (i-1)'s close was 1000 + (i-1).
        expected = float(1000 + (i - 1))
        assert float(mids["BTC"]) == expected, (
            f"At tick {i} (t={opens[i]}), shim returned {mids['BTC']} "
            f"but should have returned {expected}"
        )
