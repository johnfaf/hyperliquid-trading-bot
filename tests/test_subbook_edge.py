"""Per-(wallet,side,coin) copy sub-book gate (signal #5)."""
from __future__ import annotations

import sqlite3
import time

import config
from src.learning.forward_edge import wallet_subbook_outcomes
from src.trading.copy_trader import CopyTrader

_DAY = 86_400_000.0


def _mk(db, rows):
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE wallet_fills (wallet_address TEXT, coin TEXT, side TEXT, "
              "time_ms REAL, closed_pnl REAL)")
    c.executemany("INSERT INTO wallet_fills VALUES (?,?,?,?,?)", rows)
    c.commit()
    c.close()


def test_subbook_outcomes_filters_coin_and_side(tmp_path):
    db = str(tmp_path / "wf.db")
    now = 1_000_000 * _DAY
    _mk(db, [
        ("0xA", "ETH", "long", now - _DAY, 5.0),
        ("0xA", "ETH", "long", now - 2 * _DAY, -2.0),
        ("0xA", "ETH", "short", now - _DAY, -3.0),   # wrong side
        ("0xA", "SOL", "long", now - _DAY, 4.0),     # wrong coin
    ])
    outs = wallet_subbook_outcomes(db, "0xa", "ETH", "long", now, lookback_days=90)
    assert len(outs) == 2   # only ETH long


def test_subbook_passes_off_by_default(monkeypatch):
    monkeypatch.setattr(config, "COPY_SUBBOOK_EDGE_ENABLED", False, raising=False)
    assert CopyTrader._subbook_passes("0xA", "ETH", "short") is True


def test_subbook_blocks_proven_flat(tmp_path, monkeypatch):
    db = str(tmp_path / "wf.db")
    now = time.time() * 1000
    rows = [("0xA", "ETH", "short", now - (i + 1) * 3_600_000, -2.0) for i in range(10)]
    _mk(db, rows)   # 10 recent losing copy-shorts
    monkeypatch.setattr(config, "COPY_SUBBOOK_EDGE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "DB_PATH", db, raising=False)
    monkeypatch.setattr(config, "COPY_SUBBOOK_MIN_SAMPLES", 8, raising=False)
    monkeypatch.setattr(config, "COPY_SUBBOOK_MIN_EDGE", 0.50, raising=False)
    assert CopyTrader._subbook_passes("0xA", "ETH", "short") is False


def test_subbook_unmeasured_bootstraps(tmp_path, monkeypatch):
    db = str(tmp_path / "wf.db")
    _mk(db, [])   # no evidence for this sub-book
    monkeypatch.setattr(config, "COPY_SUBBOOK_EDGE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "DB_PATH", db, raising=False)
    monkeypatch.setattr(config, "COPY_SUBBOOK_MIN_SAMPLES", 8, raising=False)
    assert CopyTrader._subbook_passes("0xA", "ETH", "short") is True
