"""Trade-cost estimator tests."""
from __future__ import annotations

import pytest

import config
from src.signals.trade_costs import estimate_signal_costs_bps


class _FakeSide:
    def __init__(self, v): self.value = v


def _signal(*, coin="BTC", side="long"):
    s = type("Sig", (), {})()
    s.coin = coin
    s.side = _FakeSide(side)
    return s


@pytest.fixture(autouse=True)
def _stub_funding(monkeypatch):
    monkeypatch.setattr(config, "PAPER_TRADING_MAKER_FEE_BPS", 0.2, raising=False)
    monkeypatch.setattr(config, "PAPER_TRADING_TAKER_FEE_BPS", 2.5, raising=False)
    monkeypatch.setattr(config, "TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS", 5.0, raising=False)
    monkeypatch.setattr(config, "TRADE_COSTS_DEFAULT_HOLDING_HOURS", 24.0, raising=False)
    # Default funding stub: 0.01% per hour (sensible mid)
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_asset_contexts",
        lambda: {"BTC": {"funding": 0.0001}},
    )


def test_taker_default_round_trip():
    costs = estimate_signal_costs_bps(_signal())
    assert costs["fees_bps"] == pytest.approx(5.0, abs=1e-6)        # 2.5 * 2
    assert costs["slippage_bps"] == pytest.approx(10.0, abs=1e-6)    # 5.0 * 2
    # 24h * 0.0001 = 0.0024 = 24 bps; long pays positive funding
    assert costs["funding_bps"] == pytest.approx(24.0, abs=1e-3)
    assert costs["total_bps"] == pytest.approx(39.0, abs=1e-3)


def test_maker_role_lower_fees():
    costs = estimate_signal_costs_bps(_signal(), role="maker")
    assert costs["fees_bps"] == pytest.approx(0.4, abs=1e-6)


def test_short_funding_sign_flips():
    costs = estimate_signal_costs_bps(_signal(side="short"))
    # Short with positive funding earns the long-side premium (negative cost)
    assert costs["funding_bps"] < 0


def test_holding_hours_scales_funding():
    short = estimate_signal_costs_bps(_signal(), holding_hours=1.0)
    long_hold = estimate_signal_costs_bps(_signal(), holding_hours=48.0)
    assert long_hold["funding_bps"] > short["funding_bps"]
    assert long_hold["funding_bps"] == pytest.approx(2 * short["funding_bps"] * 24, abs=1e-3)


def test_missing_funding_doesnt_crash(monkeypatch):
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_asset_contexts",
        lambda: {},
    )
    costs = estimate_signal_costs_bps(_signal())
    assert costs["funding_bps"] == 0.0
    # Fees + slippage still apply
    assert costs["total_bps"] == pytest.approx(15.0, abs=1e-3)
