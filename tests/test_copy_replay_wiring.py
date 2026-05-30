"""Tests for copy-trade simulation wiring in the replay harness.

Two gaps kept copy trades at 0 in backtests even with the position oracle:
  1. copy_trader scanned the discovery top-N traders, which barely overlapped
     the tracked-copy wallets the oracle serves -> build_traders_from_fills
     makes the trader pool BE the oracle's wallets (copyable + ranked).
  2. REPLAY_PROFILE didn't include copy_trader, so container.copy_trader was
     None and Phase 4b silently skipped.
"""
from __future__ import annotations

import sqlite3

from src.backtest.replay.strategy_seed import build_traders_from_fills


def _make_fills(tmp_path, rows):
    db = tmp_path / "fills.db"
    c = sqlite3.connect(db)
    c.execute(
        "CREATE TABLE wallet_fills (wallet_address TEXT, coin TEXT, side TEXT, "
        "original_price REAL, size REAL, time_ms INTEGER, direction TEXT)"
    )
    c.executemany("INSERT INTO wallet_fills VALUES (?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()
    return str(db)


def test_build_traders_from_fills_makes_copyable(tmp_path):
    a1 = "0x" + "11" * 20
    a2 = "0x" + "22" * 20
    db = _make_fills(tmp_path, [
        (a1, "BTC", "B", 100.0, 1.0, 1000, "Open Long"),
        (a1, "BTC", "A", 110.0, 1.0, 2000, "Close Long"),
        (a2, "ETH", "A", 50.0, 2.0, 1500, "Open Short"),
    ])
    traders = build_traders_from_fills(db)
    by_addr = {t.address: t for t in traders}
    assert set(by_addr) == {a1, a2}
    # Every emitted trader must clear the evidence bar: active, >=10 trades,
    # non-zero pnl (so get_copyable_traders returns them).
    for t in traders:
        assert t.active == 1
        assert t.trade_count >= 10
        assert (t.total_pnl or t.roi_pct)
    # Ranked by fill activity: a1 (2 fills) outranks a2 (1 fill).
    assert by_addr[a1].total_pnl >= by_addr[a2].total_pnl


def test_build_traders_from_fills_missing_db(tmp_path):
    assert build_traders_from_fills(str(tmp_path / "nope.db")) == []


def test_copy_trader_in_replay_profile():
    from src.core.subsystem_registry import REPLAY_PROFILE
    assert "copy_trader" in REPLAY_PROFILE


def test_copy_scan_top_n_config_default():
    import importlib
    import config as cfg
    importlib.reload(cfg)
    try:
        assert cfg.COPY_TRADER_SCAN_TOP_N == 10
    finally:
        importlib.reload(cfg)
