"""Verify TP defaults dropped from 5:1 to 2.5:1 (the May 2026 audit).

Background
----------
30-day audit found TP fires only 4/308 trades while time_limit fires 28
times with +$188 net. Winners reach a couple R of profit but drift back
before the 5R TP. Defaults were halved to 2.5x stop so the bot captures
the move it actually gets.

This test locks in the new defaults so a future PR can't silently revert
the change.
"""
from __future__ import annotations

import config
from src.signals.signal_schema import RiskParams


def test_riskparams_defaults_to_2_5_to_1():
    """Fresh RiskParams (no overrides) uses the new 2.5:1 shape."""
    rp = RiskParams()
    # Default stop 5% ROE, TP 12.5% ROE (2.5x).
    assert rp.stop_loss_pct == 0.05
    assert rp.take_profit_pct == 0.125
    assert rp.reward_to_risk_ratio == 2.5


def test_riskparams_sync_reward_to_risk_uses_new_ratio():
    """sync_reward_to_risk() recomputes TP from stop * R:R. With a
    different stop, the resulting TP should still be 2.5x."""
    rp = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.0,
                    reward_to_risk_ratio=2.5, enforce_reward_to_risk=True)
    # __post_init__ should have called sync_reward_to_risk()
    assert rp.take_profit_pct == 0.04 * 2.5


def test_paper_trading_config_tp_multiple_is_2_5():
    """config.PAPER_TRADING_TAKE_PROFIT_PCT must equal stop * 2.5
    (with default PAPER_TRADING_TAKE_PROFIT_MULTIPLE)."""
    expected = config.PAPER_TRADING_STOP_LOSS_PCT * 2.5
    assert config.PAPER_TRADING_TAKE_PROFIT_PCT == expected
    assert config.PAPER_TRADING_TAKE_PROFIT_MULTIPLE == 2.5


def test_resolve_trigger_prices_uses_halved_tp_distance(monkeypatch):
    """At entry 100, 25x leverage, default stop/TP, ratio 2.5:1:
    - price stop = 0.05/25 = 20 bps -> 99.80
    - price TP   = 0.125/25 = 50 bps -> 100.50 (was 100 bps / 101.0 pre-fix)"""
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", False, raising=False)
    rp = RiskParams()  # defaults
    sl, tp = rp.resolve_trigger_prices(100.0, "long", 25.0)
    import pytest
    assert sl == pytest.approx(99.80)
    assert tp == pytest.approx(100.50)  # 50 bps above entry (was 100 bps)


def test_can_override_ratio_via_dataclass_field():
    """Operators that want a tighter or wider R:R can still set it
    explicitly on the dataclass."""
    import pytest
    rp = RiskParams(stop_loss_pct=0.05, reward_to_risk_ratio=3.0,
                    enforce_reward_to_risk=True)
    assert rp.take_profit_pct == pytest.approx(0.15)  # 3x stop
