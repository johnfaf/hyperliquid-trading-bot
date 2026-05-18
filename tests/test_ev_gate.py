"""EV gate tests."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

import config
from src.signals.ev_gate import compute_expected_value, evaluate_signal_ev


@dataclass
class _FakeRisk:
    stop_loss_pct: float = 0.01   # 1% raw price stop
    take_profit_pct: float = 0.03  # 3% raw price target
    risk_basis: str = "price"

    def resolve_roe_stop_loss_pct(self, leverage):
        return self.stop_loss_pct * max(leverage, 1.0)

    def resolve_roe_take_profit_pct(self, leverage):
        return self.take_profit_pct * max(leverage, 1.0)


class _FakeSide:
    def __init__(self, v): self.value = v


def _signal(*, confidence=0.55, leverage=2.0, side="long", live=False, ctx=None):
    s = type("Sig", (), {})()
    s.coin = "BTC"
    s.side = _FakeSide(side)
    s.confidence = confidence
    s.leverage = leverage
    s.risk = _FakeRisk()
    s.context = ctx if ctx is not None else {"live_mirror": live}
    return s


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setattr(config, "EV_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "EV_GATE_MIN_BPS", 10.0, raising=False)
    monkeypatch.setattr(config, "EV_GATE_MIN_COST_RATIO", 1.5, raising=False)
    monkeypatch.setattr(config, "EV_GATE_LIVE_SIGMA_MULT", 2.0, raising=False)


def test_gate_disabled_passes(monkeypatch):
    monkeypatch.setattr(config, "EV_GATE_ENABLED", False, raising=False)
    ok, reason, _ = evaluate_signal_ev(_signal())
    assert ok and reason == "gate_disabled"


def test_strong_edge_passes():
    # 60% win prob, 6R win / 2R loss, low costs -> clearly positive EV
    sig = _signal(confidence=0.60, leverage=2.0)
    ok, reason, br = evaluate_signal_ev(
        sig, costs={"total_bps": 20.0},
    )
    assert ok, f"expected pass, got: {reason} {br}"
    assert br["ev_bps"] > 0


def test_weak_edge_rejected():
    # 45% win prob, symmetric 2R/2R, costs 30bps -> negative EV
    sig = _signal(confidence=0.45, leverage=1.0)
    ok, reason, br = evaluate_signal_ev(
        sig, costs={"total_bps": 30.0},
    )
    assert not ok, f"expected reject, got: {reason} {br}"
    assert "ev_below_threshold" in reason


def test_live_lcb_can_block_at_modest_mean():
    """Live trades require LCB > 0; a borderline-positive EV should fail
    when the variance is large enough."""
    # bucket_n=1 makes sigma_per_trade fully apply; with low p_win and
    # high payoff spread, the live LCB easily goes negative.
    sig = _signal(confidence=0.52, leverage=2.0, live=True)
    ok, reason, br = evaluate_signal_ev(
        sig, costs={"total_bps": 5.0}, bucket_n=1.0,
    )
    # Should reject live even though mean EV > threshold, because
    # mean - 2*sigma <= 0 at small bucket_n.
    assert not ok
    assert "live" in reason.lower() or "lcb" in reason.lower()


def test_live_lcb_passes_with_large_bucket():
    """Same signal, large bucket_n -> sigma collapses, live passes."""
    sig = _signal(confidence=0.58, leverage=2.0, live=True)
    ok, reason, br = evaluate_signal_ev(
        sig, costs={"total_bps": 5.0}, bucket_n=400.0,
    )
    assert ok, f"expected pass with n=400, got: {reason} {br}"


def test_compute_ev_returns_breakdown():
    br = compute_expected_value(_signal(), costs={"total_bps": 25.0})
    for key in ("ev_bps", "sigma_bps", "p_win", "avg_win_bps", "avg_loss_bps", "cost_bps"):
        assert key in br
    assert br["cost_bps"] == 25.0
