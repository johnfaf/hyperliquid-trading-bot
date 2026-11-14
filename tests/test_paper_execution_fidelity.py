from datetime import datetime, timezone

import pytest

import config
from src.trading import paper_trader as paper


@pytest.fixture
def paper_case(monkeypatch):
    trade = {
        "id": 50, "coin": "BTC", "side": "long", "entry_price": 100,
        "leverage": 1, "size": 2, "stop_loss": 99, "take_profit": 110,
        "opened_at": "2026-01-01T11:59:30+00:00",
        "metadata": {"initial_stop_loss": 95},
    }
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    finalizations = []
    metadata_writes = []
    trader = paper.PaperTrader.__new__(paper.PaperTrader)
    trader._resolve_trade_risk_policy = lambda _: {}
    trader._update_trade_excursion = lambda *args, **kwargs: None
    trader._close_trade = lambda row, **kwargs: finalizations.append((row.copy(), kwargs))
    monkeypatch.setattr(config, "SL_MIN_HOLD_SECONDS", 600)
    monkeypatch.setattr(config, "PAPER_TRADING_FUNDING_ENABLED", False)
    monkeypatch.setattr(paper.clock_provider, "utc_now", lambda: now)
    monkeypatch.setattr(paper.db, "get_open_paper_trades", lambda: [trade])
    monkeypatch.setattr(paper.db, "update_paper_trade_metadata", lambda _, updates: metadata_writes.append(updates))
    return trader, trade, finalizations, metadata_writes


@pytest.mark.parametrize("mid", [None, "NaN", "inf"])
def test_no_fake_time_exit_without_valid_quote(monkeypatch, paper_case, mid):
    trader, trade, finalizations, _ = paper_case
    trade["opened_at"] = "2025-12-30T12:00:00+00:00"
    monkeypatch.setattr(paper.hl, "get_all_mids", lambda: {"BTC": mid})
    assert trader.check_open_positions() == []
    assert not finalizations


def test_original_stop_stays_armed_during_grace_period(monkeypatch, paper_case):
    trader, _, finalizations, _ = paper_case
    mids = {"BTC": 97}
    monkeypatch.setattr(paper.hl, "get_all_mids", lambda: mids)
    trader.check_open_positions()
    assert not finalizations
    mids["BTC"] = 94
    trader.check_open_positions()
    assert finalizations[0][1] == {"current_price": 94, "close_reason": "stop_loss"}


@pytest.mark.parametrize(("side", "expected"), [("long", 0.02), ("short", -0.02)])
def test_hourly_funding_uses_oracle_and_is_not_counted_twice(monkeypatch, paper_case, side, expected):
    trader, trade, finalizations, writes = paper_case
    trade["side"] = side
    if side == "short":
        trade.update(stop_loss=101, metadata={"initial_stop_loss": 105})
    mids = {"BTC": 100}
    monkeypatch.setattr(config, "PAPER_TRADING_FUNDING_ENABLED", True)
    monkeypatch.setattr(paper.hl, "get_all_mids", lambda: mids)
    monkeypatch.setattr(paper.hl, "get_asset_contexts", lambda: {"BTC": {"funding": "0.0001", "oracle_price": 100}})
    trader.check_open_positions()
    assert writes == [{
        "funding_accrued": expected, "last_funding_ts": "2026-01-01T12:00:00+00:00",
        "last_funding_rate": 0.0001, "funding_estimated": True,
    }]
    mids["BTC"] = 94 if side == "long" else 106
    trader.check_open_positions()
    assert len(writes) == 1
    assert finalizations[0][0]["metadata"]["funding_accrued"] == expected
