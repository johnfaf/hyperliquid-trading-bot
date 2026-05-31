"""Replay must seed the process RNGs so a run is bit-for-bit reproducible.

paper_trader's slippage model draws from the global ``random`` module. Left
unseeded, two runs of the same window produced slightly different fill prices
(the regime-cache fix made *which* trades fire deterministic; this pins the
fill prices). ReplayHarness seeds ``random`` (and numpy) from the fixed window
start in _build, so the same window replays identically.
"""
from __future__ import annotations

import random
import sqlite3

from src.backtest.replay.harness import ReplayHarness


def _cache(tmp_path):
    db = tmp_path / "candle_cache.db"
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE candles (coin TEXT, timeframe TEXT, timestamp_ms INTEGER, "
              "open REAL, high REAL, low REAL, close REAL, volume REAL, "
              "PRIMARY KEY (coin, timeframe, timestamp_ms))")
    base = 1_700_000_000_000
    c.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)",
                  [("BTC", "1h", base + i * 3_600_000, 100, 101, 99, 100, 1.0) for i in range(50)])
    c.commit()
    c.close()
    return str(db), base


def test_harness_seeds_global_rng_reproducibly(tmp_path):
    db, base = _cache(tmp_path)
    end = base + 40 * 3_600_000

    with ReplayHarness(base, end, cache_db=db, coins=["BTC"], engage_network_sandbox=False):
        seq1 = [random.random() for _ in range(8)]
    with ReplayHarness(base, end, cache_db=db, coins=["BTC"], engage_network_sandbox=False):
        seq2 = [random.random() for _ in range(8)]

    assert seq1 == seq2, "same window must seed the global RNG identically"


def test_different_windows_seed_differently(tmp_path):
    """Sanity: the seed is derived from the window start, so distinct windows
    get distinct (still-reproducible) RNG streams -- we are not flattening every
    run onto one constant seed."""
    db, base = _cache(tmp_path)
    with ReplayHarness(base, base + 40 * 3_600_000, cache_db=db, coins=["BTC"], engage_network_sandbox=False):
        a = [random.random() for _ in range(8)]
    with ReplayHarness(base + 3_600_000, base + 41 * 3_600_000, cache_db=db, coins=["BTC"], engage_network_sandbox=False):
        b = [random.random() for _ in range(8)]
    assert a != b
