"""Replay must disable the live-feed data-readiness gate.

The firewall's data-readiness gate validates the LIVE feature store /
source-health freshness, which a sandboxed historical replay doesn't populate.
Left on, it rejects every signal -> the replay (and bg-auto-backtest) produce 0
trades. The harness disables it during a run and restores it on teardown.
"""
from __future__ import annotations

import sqlite3

import config as cfg
from src.backtest.replay.harness import ReplayHarness


def _cache(tmp_path):
    db = tmp_path / "candle_cache.db"
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE candles (coin TEXT, timeframe TEXT, timestamp_ms INTEGER, "
              "open REAL, high REAL, low REAL, close REAL, volume REAL, "
              "PRIMARY KEY (coin, timeframe, timestamp_ms))")
    base = 1_700_000_000_000
    c.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)",
                  [("BTC", "1h", base + i*3_600_000, 100, 101, 99, 100, 1.0) for i in range(50)])
    c.commit()
    c.close()
    return str(db), base


def test_harness_disables_and_restores_readiness_gate(tmp_path, monkeypatch):
    db, base = _cache(tmp_path)
    monkeypatch.setattr(cfg, "DATA_READINESS_GATE_ENABLED", True, raising=False)
    with ReplayHarness(base, base + 40*3_600_000, cache_db=db, coins=["BTC"],
                       build_container=False):
        assert cfg.DATA_READINESS_GATE_ENABLED is False   # disabled in-run
    assert cfg.DATA_READINESS_GATE_ENABLED is True         # restored on teardown


def test_harness_leaves_already_disabled_gate_off(tmp_path, monkeypatch):
    db, base = _cache(tmp_path)
    monkeypatch.setattr(cfg, "DATA_READINESS_GATE_ENABLED", False, raising=False)
    with ReplayHarness(base, base + 40*3_600_000, cache_db=db, coins=["BTC"],
                       build_container=False):
        assert cfg.DATA_READINESS_GATE_ENABLED is False
    assert cfg.DATA_READINESS_GATE_ENABLED is False        # untouched
