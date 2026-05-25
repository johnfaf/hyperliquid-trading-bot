"""analyze_trader() must upsert the trader row BEFORE saving snapshots.

Background
----------
``position_snapshots.trader_address`` carries a FOREIGN KEY referencing
``traders(address)``.  The original ``analyze_trader()`` ordering was:

  1. compute trade_analysis
  2. ``db.save_position_snapshot(trader_address=address, ...)``  # FK insert
  3. run bot detection
  4. ``db.upsert_trader(address=address, ...)``                   # creates row

For a new trader (no existing ``traders`` row, or one that was just
purged by ``purge_non_golden_wallets`` in Phase 0 of the discovery
cycle) step 2 hits FOREIGN KEY constraint failed and the whole
``analyze_trader`` call returns None via the outer except in
``run_discovery``.  The trader's analysis is lost and they never reach
Phase 2 strategy identification.

Observed at ~5-10% failure rate (40-48 errors per discovery cycle) on
the 2026-05-24 production run; the structural cause of the gap between
1238 prescreened and 816 actually-analyzed traders.

After this fix
--------------
The snapshot loop is moved after ``upsert_trader``.  Each snapshot is
also wrapped in its own try/except so a bad row never aborts the rest
of the analysis (the trader profile is already persisted).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

import src.discovery.trader_discovery as td


@pytest.fixture
def call_order(monkeypatch):
    """Track the order of upsert_trader / save_position_snapshot calls."""
    order = []

    def _upsert(*args, **kwargs):
        order.append("upsert_trader")

    def _snap(*args, **kwargs):
        order.append("save_position_snapshot")

    monkeypatch.setattr(td.db, "upsert_trader", _upsert)
    monkeypatch.setattr(td.db, "save_position_snapshot", _snap)

    return order


def _stub_hyperliquid(monkeypatch, *, positions=None, fills=None):
    """Stub the Hyperliquid client so analyze_trader runs without HTTP."""
    state = {
        "account_value": 10_000.0,
        "positions": positions or [
            {
                "coin": "BTC",
                "side": "long",
                "size": 0.01,
                "entry_price": 50_000.0,
                "leverage": 2.0,
                "unrealized_pnl": 5.0,
                "margin_used": 250.0,
            }
        ],
        "total_margin_used": 250.0,
    }
    monkeypatch.setattr(td.hl, "get_user_state", lambda address: state)
    monkeypatch.setattr(td.hl, "get_user_fills", lambda *a, **k: fills or [])


# ── Headline ordering guarantee ──────────────────────────────


def test_upsert_runs_before_snapshot(call_order, monkeypatch):
    """The trader row must exist BEFORE any position snapshot is inserted."""
    _stub_hyperliquid(monkeypatch)
    import src.discovery.adaptive_bot_detector as abd
    monkeypatch.setattr(abd.AdaptiveBotDetector, "detect", lambda self, *a, **k: MagicMock(
        is_bot=False, bot_probability=0.10, confidence=0.50, reason="ok",
    ), raising=False)

    monkeypatch.setattr(td.db, "get_active_traders", lambda **k: [])
    detector = td.TraderDiscovery()
    result = detector.analyze_trader("0x" + "1" * 40)

    assert result is not None
    assert "upsert_trader" in call_order, "upsert_trader was never called"
    assert "save_position_snapshot" in call_order, "snapshot never written"

    upsert_idx = call_order.index("upsert_trader")
    snap_idx = call_order.index("save_position_snapshot")
    assert upsert_idx < snap_idx, (
        f"upsert_trader must run BEFORE save_position_snapshot to satisfy "
        f"the FK constraint; got order={call_order}"
    )


def test_no_snapshot_when_no_open_positions(call_order, monkeypatch):
    """Closed-book traders skip the snapshot loop entirely."""
    _stub_hyperliquid(monkeypatch, positions=[
        # Zero-size positions are filtered out.
        {
            "coin": "BTC", "side": "long", "size": 0.0,
            "entry_price": 0.0, "leverage": 1.0,
            "unrealized_pnl": 0.0, "margin_used": 0.0,
        },
    ])
    import src.discovery.adaptive_bot_detector as abd
    monkeypatch.setattr(abd.AdaptiveBotDetector, "detect", lambda self, *a, **k: MagicMock(
        is_bot=False, bot_probability=0.10, confidence=0.50, reason="ok",
    ), raising=False)

    monkeypatch.setattr(td.db, "get_active_traders", lambda **k: [])
    detector = td.TraderDiscovery()
    detector.analyze_trader("0x" + "2" * 40)

    assert "upsert_trader" in call_order
    assert "save_position_snapshot" not in call_order


# ── Per-snapshot resilience ──────────────────────────────────


def test_snapshot_failure_does_not_lose_trader(monkeypatch):
    """A snapshot insert failure must not lose the upserted trader row.

    Before this fix the entire analyze_trader call would raise and the
    outer except in run_discovery would drop the trader.  With the
    reordering + per-position try/except, the trader profile is already
    persisted by the time the (now non-fatal) snapshot insert is
    attempted; the trader returns its profile successfully even if the
    snapshot DB fails.
    """
    _stub_hyperliquid(monkeypatch)

    upsert_called = []

    def _upsert(*a, **k):
        upsert_called.append(True)

    def _snap_boom(*a, **k):
        raise RuntimeError("simulated snapshot insert failure")

    monkeypatch.setattr(td.db, "upsert_trader", _upsert)
    monkeypatch.setattr(td.db, "save_position_snapshot", _snap_boom)
    import src.discovery.adaptive_bot_detector as abd
    monkeypatch.setattr(abd.AdaptiveBotDetector, "detect", lambda self, *a, **k: MagicMock(
        is_bot=False, bot_probability=0.10, confidence=0.50, reason="ok",
    ), raising=False)

    monkeypatch.setattr(td.db, "get_active_traders", lambda **k: [])
    detector = td.TraderDiscovery()
    result = detector.analyze_trader("0x" + "3" * 40)

    assert result is not None, (
        "Trader profile must still be returned even if the snapshot insert "
        "fails -- the upsert already persisted the row"
    )
    assert upsert_called, "trader was never upserted"
