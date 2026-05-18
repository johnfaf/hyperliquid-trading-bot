"""Walk-forward promotion gate tests.

Covers the three resolution paths (strategy_id -> DB strategy,
copy_trade -> agent_scores, strategy_type -> agent_scores) plus the
fail-closed default when no source data is resolvable.
"""
from __future__ import annotations

import pytest

import config
from src.learning.promotion_gate import is_live_promotable


@pytest.fixture(autouse=True)
def _enable_gate(monkeypatch):
    """Force the gate ON inside this module regardless of env."""
    monkeypatch.setattr(config, "LIVE_PROMOTION_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LIVE_PROMOTION_MIN_TRADES", 30, raising=False)
    monkeypatch.setattr(config, "LIVE_PROMOTION_MIN_WIN_RATE", 0.45, raising=False)
    monkeypatch.setattr(config, "LIVE_PROMOTION_MIN_SCORE", 0.20, raising=False)


def test_gate_disabled_passes_through(monkeypatch):
    monkeypatch.setattr(config, "LIVE_PROMOTION_GATE_ENABLED", False, raising=False)
    promotable, reason = is_live_promotable({"coin": "BTC"})
    assert promotable is True
    assert reason == "gate_disabled"


def test_fail_closed_when_no_source_data():
    """Trade with no strategy_id / source_trader / strategy_type stays paper-only."""
    promotable, reason = is_live_promotable({"coin": "BTC", "size": 1.0})
    assert promotable is False
    assert reason == "no_promotion_data"


def test_strategy_path_promotes_when_thresholds_met(monkeypatch):
    healthy_strategy = {
        "id": 42,
        "trade_count": 60,
        "win_rate": 0.55,
        "current_score": 0.42,
        "strategy_type": "momentum_long",
        "parameters": '{"source_wallet": "0x' + "a" * 40 + '"}',
    }
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.get_strategy",
        lambda sid: healthy_strategy if sid == 42 else None,
    )
    # No quarantine reason for a clean strategy.
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.strategy_quarantine_reason",
        lambda strat: None,
    )
    promotable, reason = is_live_promotable(
        {"coin": "BTC", "size": 1.0, "strategy_id": 42}
    )
    assert promotable is True
    assert reason == "ok"


def test_strategy_path_blocks_when_trades_below_min(monkeypatch):
    thin_strategy = {
        "id": 1,
        "trade_count": 12,
        "win_rate": 0.60,
        "current_score": 0.50,
    }
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.get_strategy", lambda sid: thin_strategy
    )
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.strategy_quarantine_reason",
        lambda strat: None,
    )
    promotable, reason = is_live_promotable({"strategy_id": 1})
    assert promotable is False
    assert reason.startswith("insufficient_trades")


def test_strategy_path_blocks_when_quarantined(monkeypatch):
    syn_strategy = {
        "id": 7,
        "trade_count": 5000,
        "win_rate": 0.999,
        "current_score": 0.7955,
    }
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.get_strategy", lambda sid: syn_strategy
    )
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.strategy_quarantine_reason",
        lambda strat: "synthetic_placeholder_metrics",
    )
    promotable, reason = is_live_promotable({"strategy_id": 7})
    assert promotable is False
    assert "quarantined" in reason


def test_strategy_path_blocks_low_win_rate(monkeypatch):
    losing_strategy = {
        "id": 9,
        "trade_count": 80,
        "win_rate": 0.30,
        "current_score": 0.50,
    }
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.get_strategy", lambda sid: losing_strategy
    )
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.strategy_quarantine_reason",
        lambda strat: None,
    )
    promotable, reason = is_live_promotable({"strategy_id": 9})
    assert promotable is False
    assert reason.startswith("win_rate_too_low")


def test_copy_trade_path_uses_agent_scores(monkeypatch):
    """copy_trade signals look up agent_scores by source_key."""
    fake_row = {"total_signals": 50, "correct_signals": 26, "accuracy": 0.52}

    class _FakeConn:
        def __init__(self, row):
            self._row = row

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):
            class _Cur:
                def fetchone(_self):
                    return fake_row

            return _Cur()

    monkeypatch.setattr(
        "src.learning.promotion_gate.db.get_connection",
        lambda for_read=False: _FakeConn(fake_row),
    )
    trader_addr = "0x" + "b" * 40
    promotable, reason = is_live_promotable(
        {
            "coin": "BTC",
            "metadata": {
                "source": "copy_trade",
                "source_trader": trader_addr,
            },
        }
    )
    assert promotable is True
    assert reason == "ok"


def test_copy_trade_path_blocks_thin_history(monkeypatch):
    fake_row = {"total_signals": 4, "correct_signals": 3, "accuracy": 0.75}

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):
            class _Cur:
                def fetchone(_self):
                    return fake_row

            return _Cur()

    monkeypatch.setattr(
        "src.learning.promotion_gate.db.get_connection",
        lambda for_read=False: _FakeConn(),
    )
    trader_addr = "0x" + "c" * 40
    promotable, reason = is_live_promotable(
        {
            "metadata": {
                "source": "copy_trade",
                "source_trader": trader_addr,
            }
        }
    )
    assert promotable is False
    assert reason.startswith("insufficient_signals")


def test_strategy_type_path_falls_through_to_agent_scores(monkeypatch):
    """When no strategy_id is provided but strategy_type is, fall back to
    agent_scores under the ``strategy:<type>`` key."""
    fake_row = {"total_signals": 35, "correct_signals": 19, "accuracy": 0.54}

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):
            class _Cur:
                def fetchone(_self):
                    return fake_row

            return _Cur()

    monkeypatch.setattr(
        "src.learning.promotion_gate.db.get_connection",
        lambda for_read=False: _FakeConn(),
    )
    promotable, reason = is_live_promotable(
        {"metadata": {"strategy_type": "momentum_short"}}
    )
    assert promotable is True
    assert reason == "ok"


def test_strategy_type_unknown_stays_blocked(monkeypatch):
    """strategy_type == 'unknown' is the same as no data — stay paper-only."""
    promotable, reason = is_live_promotable(
        {"metadata": {"strategy_type": "unknown"}}
    )
    assert promotable is False
    assert reason == "no_promotion_data"
