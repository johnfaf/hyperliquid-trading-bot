"""Orphan reaper tests."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import config
from src.trading import orphan_reaper
from src.trading.orphan_reaper import reap_orphan_positions


class _FakeTrader:
    """Mock live trader that records close_position calls."""
    def __init__(self, result=None):
        self.closes = []
        self._result = result or {"status": "ok", "size": 0.0166}

    def close_position(self, coin: str):
        self.closes.append(coin)
        return dict(self._result)


class _FakeContainer:
    def __init__(self, trader):
        self.live_trader = trader


@pytest.fixture(autouse=True)
def _enable_reaper(monkeypatch):
    monkeypatch.setattr(config, "ORPHAN_REAPER_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ORPHAN_REAPER_MAX_AGE_HOURS", 24.0, raising=False)
    monkeypatch.setattr(config, "ORPHAN_REAPER_REQUIRE_BREAKEVEN", True, raising=False)


def _orphan_trade(*, coin="ETH", side="short", entry=2252.0, size=0.0166,
                  age_hours=48.0):
    found_at = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    return {
        "id": 1168,
        "coin": coin,
        "side": side,
        "entry_price": entry,
        "size": size,
        "leverage": 2,
        "metadata": json.dumps({
            "orphan_found": True,
            "source": "live_orphan",
            "strategy_type": "orphan_found",
            "orphan_found_at": found_at,
        }),
    }


def test_reaper_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(config, "ORPHAN_REAPER_ENABLED", False, raising=False)
    trader = _FakeTrader()
    monkeypatch.setattr(
        orphan_reaper.db, "get_open_paper_trades",
        lambda: [_orphan_trade(age_hours=99.0)],
    )
    reaped = reap_orphan_positions(_FakeContainer(trader))
    assert reaped == []
    assert trader.closes == []


def test_no_trader_returns_empty(monkeypatch):
    container = _FakeContainer(trader=None)
    container.live_trader = None
    monkeypatch.setattr(
        orphan_reaper.db, "get_open_paper_trades",
        lambda: [_orphan_trade()],
    )
    reaped = reap_orphan_positions(container)
    assert reaped == []


def test_no_orphans_returns_empty(monkeypatch):
    trader = _FakeTrader()
    monkeypatch.setattr(orphan_reaper.db, "get_open_paper_trades", lambda: [])
    reaped = reap_orphan_positions(_FakeContainer(trader))
    assert reaped == []


def test_young_orphan_not_reaped(monkeypatch):
    """An orphan younger than max_age stays open."""
    trader = _FakeTrader()
    monkeypatch.setattr(
        orphan_reaper.db, "get_open_paper_trades",
        lambda: [_orphan_trade(age_hours=2.0)],  # well under 24h default
    )
    reaped = reap_orphan_positions(_FakeContainer(trader))
    assert reaped == []
    assert trader.closes == []


def test_old_orphan_at_breakeven_reaped(monkeypatch):
    """Aged orphan at break-even (mid >= entry for long) closes."""
    trader = _FakeTrader()
    orphan = _orphan_trade(coin="ETH", side="short", entry=2252.0, age_hours=48.0)
    monkeypatch.setattr(
        orphan_reaper.db, "get_open_paper_trades", lambda: [orphan],
    )
    # Mid below entry -> short is profitable -> meets break-even gate
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_mids",
        lambda: {"ETH": 2100.0},
    )
    reaped = reap_orphan_positions(_FakeContainer(trader))
    assert len(reaped) == 1
    assert trader.closes == ["ETH"]
    assert reaped[0]["ok"] is True
    assert reaped[0]["approx_pnl_usd"] > 0


def test_old_orphan_at_loss_held(monkeypatch):
    """Aged orphan that would close at a loss stays open by default."""
    trader = _FakeTrader()
    orphan = _orphan_trade(coin="ETH", side="short", entry=2252.0, age_hours=48.0)
    monkeypatch.setattr(
        orphan_reaper.db, "get_open_paper_trades", lambda: [orphan],
    )
    # Mid above entry -> short is losing -> break-even gate blocks reap
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_mids",
        lambda: {"ETH": 2400.0},
    )
    reaped = reap_orphan_positions(_FakeContainer(trader))
    assert reaped == []
    assert trader.closes == []


def test_breakeven_gate_off_reaps_anyway(monkeypatch):
    """With require_breakeven=False, aged orphans close even at a loss."""
    monkeypatch.setattr(config, "ORPHAN_REAPER_REQUIRE_BREAKEVEN", False, raising=False)
    trader = _FakeTrader()
    orphan = _orphan_trade(coin="ETH", side="short", entry=2252.0, age_hours=48.0)
    monkeypatch.setattr(
        orphan_reaper.db, "get_open_paper_trades", lambda: [orphan],
    )
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_mids",
        lambda: {"ETH": 2400.0},
    )
    reaped = reap_orphan_positions(_FakeContainer(trader))
    assert len(reaped) == 1
    assert trader.closes == ["ETH"]


def test_non_orphan_trade_ignored(monkeypatch):
    """Trades without orphan metadata are not touched."""
    trader = _FakeTrader()
    regular = {
        "id": 99,
        "coin": "BTC",
        "side": "long",
        "entry_price": 67000,
        "size": 0.001,
        "leverage": 3,
        "metadata": json.dumps({"source": "strategy", "strategy_type": "momentum_long"}),
    }
    monkeypatch.setattr(orphan_reaper.db, "get_open_paper_trades", lambda: [regular])
    reaped = reap_orphan_positions(_FakeContainer(trader))
    assert reaped == []
    assert trader.closes == []
