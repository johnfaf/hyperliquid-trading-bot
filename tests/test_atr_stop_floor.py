"""A1: ATR-aware stop-loss floor (config.ATR_STOP_FLOOR_ENABLED).

Audit context: 5/8 of recent week's losses were SL closes at avg -$3.22,
some triggered by adverse moves as small as -0.03%. The default 4% ROE
stop on 25x leverage = 16 bps price stop, which is well inside one
candle of typical noise. This widens the stop (never tightens) to at
least max(k * recent_ATR, noise_floor_bps) when the operator opts in,
and preserves the reward:risk ratio by widening TP by the same factor.

Default-OFF: the floor never engages until ATR_STOP_FLOOR_ENABLED=true.
All "OFF" tests assert byte-identical output to the pre-A1 method.
"""
from __future__ import annotations

import pytest

import config
from src.signals.signal_schema import RiskParams


@pytest.fixture(autouse=True)
def _disable_floor_by_default(monkeypatch):
    """Default-OFF posture: no test sees the floor unless it opts in."""
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", False, raising=False)


def test_default_off_byte_identical_to_pre_a1():
    """When the flag is OFF, output is the same regardless of atr_pct."""
    risk = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe")
    sl_no, tp_no = risk.resolve_trigger_prices(100.0, "long", 25.0)
    sl_with, tp_with = risk.resolve_trigger_prices(100.0, "long", 25.0, atr_pct=0.05)
    assert sl_no == sl_with
    assert tp_no == tp_with


def test_flag_on_but_atr_pct_none_no_change(monkeypatch):
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    risk = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe")
    sl_pre, tp_pre = risk.resolve_trigger_prices(100.0, "long", 25.0)
    sl_post, tp_post = risk.resolve_trigger_prices(100.0, "long", 25.0, atr_pct=None)
    assert sl_pre == sl_post
    assert tp_pre == tp_post


def test_flag_on_but_atr_pct_zero_no_change(monkeypatch):
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    risk = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe")
    sl_pre, tp_pre = risk.resolve_trigger_prices(100.0, "long", 25.0)
    sl_post, tp_post = risk.resolve_trigger_prices(100.0, "long", 25.0, atr_pct=0.0)
    assert sl_pre == sl_post
    assert tp_pre == tp_post


def test_base_stop_wider_than_atr_floor_wins(monkeypatch):
    """If base price-stop is already wider than the ATR floor, do nothing."""
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_ATR_MULTIPLIER", 2.5, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_NOISE_FLOOR_BPS", 10.0, raising=False)
    # 1x leverage on price-basis 5% stop -> 5% price move. ATR floor = 2.5 * 0.01 = 2.5%.
    risk = RiskParams(stop_loss_pct=0.05, take_profit_pct=0.25, risk_basis="price")
    sl, tp = risk.resolve_trigger_prices(100.0, "long", 1.0, atr_pct=0.01)
    assert sl == pytest.approx(95.0)
    assert tp == pytest.approx(125.0)


def test_atr_floor_widens_tight_high_leverage_stop(monkeypatch):
    """The whole point of A1: 25x leverage + 4% ROE stop = 16 bps price.
    With ATR ≈ 1.5% and k=2.5, floor = 3.75% — widens 16 bps → 3.75%."""
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_ATR_MULTIPLIER", 2.5, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_NOISE_FLOOR_BPS", 50.0, raising=False)
    risk = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe")
    sl, tp = risk.resolve_trigger_prices(100.0, "long", 25.0, atr_pct=0.015)
    # ATR floor = max(2.5 * 0.015, 0.005) = 0.0375 -> sl_price = 100 * (1 - 0.0375) = 96.25
    assert sl == pytest.approx(96.25)
    # Reward:risk was 5:1 (TP 0.20/SL 0.04 in ROE). Base price TP would be
    # 0.20/25 = 0.008 (so 100.8); preserved ratio gives TP * (0.0375/0.0016) = 0.1875
    # -> 118.75
    assert tp == pytest.approx(118.75)
    # And reward:risk is preserved (within float epsilon)
    assert (tp - 100.0) / (100.0 - sl) == pytest.approx(5.0, rel=1e-6)


def test_noise_floor_binds_when_atr_too_small(monkeypatch):
    """If 2.5*ATR < noise_floor_bps, the noise floor binds.

    Use ATR=0.001 (10 bps). k*ATR = 25 bps. Noise floor = 50 bps. Stop should
    widen to 50 bps, not 25 bps.
    """
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_ATR_MULTIPLIER", 2.5, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_NOISE_FLOOR_BPS", 50.0, raising=False)
    risk = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe")
    sl, _ = risk.resolve_trigger_prices(100.0, "long", 25.0, atr_pct=0.001)
    # 0.04/25 = 0.0016 = 16 bps base; max(25, 50) = 50 bps -> 0.005
    assert sl == pytest.approx(99.5)


def test_short_side_widens_in_correct_direction(monkeypatch):
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_ATR_MULTIPLIER", 2.5, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_NOISE_FLOOR_BPS", 50.0, raising=False)
    risk = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe")
    sl, tp = risk.resolve_trigger_prices(100.0, "short", 25.0, atr_pct=0.015)
    # short stop is ABOVE entry, TP is BELOW
    assert sl == pytest.approx(103.75)
    assert tp == pytest.approx(81.25)


def test_floor_never_tightens_stop(monkeypatch):
    """Hard invariant: the ATR floor must never make a stop *tighter*."""
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_ATR_MULTIPLIER", 2.5, raising=False)
    monkeypatch.setattr(config, "ATR_STOP_NOISE_FLOOR_BPS", 50.0, raising=False)
    risk = RiskParams(stop_loss_pct=0.10, take_profit_pct=0.50, risk_basis="price")
    sl_off_flag, tp_off_flag = risk.resolve_trigger_prices(100.0, "long", 1.0, atr_pct=None)
    sl_on_flag, tp_on_flag = risk.resolve_trigger_prices(100.0, "long", 1.0, atr_pct=0.015)
    # Base stop 10% > ATR floor max(3.75%, 0.5%) -> unchanged
    assert sl_on_flag == sl_off_flag
    assert tp_on_flag == tp_off_flag


def test_malformed_atr_pct_does_not_crash(monkeypatch):
    """Robustness: a string/garbage atr_pct must not break trigger resolution."""
    monkeypatch.setattr(config, "ATR_STOP_FLOOR_ENABLED", True, raising=False)
    risk = RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe")
    sl_baseline, tp_baseline = risk.resolve_trigger_prices(100.0, "long", 25.0, atr_pct=None)
    sl_garbage, tp_garbage = risk.resolve_trigger_prices(100.0, "long", 25.0, atr_pct="not-a-number")  # type: ignore[arg-type]
    assert sl_garbage == sl_baseline
    assert tp_garbage == tp_baseline
