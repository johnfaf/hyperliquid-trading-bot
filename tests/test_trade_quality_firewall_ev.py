"""Trade-quality gate sourcing edge from the firewall EV breakdown.

Regression cover for the cold-start deadlock: the legacy confidence
proxy collapses edge to ~0 when calibration is cold (confidence pinned
at 0.50), silently vetoing every regime-aligned signal the firewall EV
gate already accepted. The gate must now consume context.ev_breakdown.
"""
from __future__ import annotations

import pytest

import config
from src.trading.paper_trader import PaperTrader


class _Sig:
    def __init__(self, confidence=0.50, context=None):
        self.confidence = confidence
        self.source_accuracy = 0.0
        self.regime = "trending_down"
        self.context = context if context is not None else {}


def _pt():
    # Bypass heavy __init__: the gate methods use only config + the
    # signal/sig args + static helpers, no instance state.
    return PaperTrader.__new__(PaperTrader)


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setattr(config, "TRADE_QUALITY_USE_FIREWALL_EV", True, raising=False)
    monkeypatch.setattr(config, "TRADE_QUALITY_FEE_EV_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "TRADE_QUALITY_STRONG_SHORT_CONFIRMATION", True, raising=False)
    monkeypatch.setattr(config, "TRADE_QUALITY_SHORT_MIN_CONFIDENCE", 0.55, raising=False)
    monkeypatch.setattr(config, "TRADE_QUALITY_MIN_EDGE_COST_MULTIPLE", 1.5, raising=False)


def test_firewall_ev_helper_parses_breakdown():
    sig = _Sig(context={"ev_breakdown": {"ev_bps": 1517.9, "cost_bps": 16.0}})
    assert PaperTrader._firewall_ev(sig) == (1517.9, 16.0)
    assert PaperTrader._firewall_ev(_Sig(context={})) is None
    assert PaperTrader._firewall_ev(_Sig(context={"ev_breakdown": "x"})) is None


def test_edge_uses_firewall_ev_not_confidence_proxy():
    pt = _pt()
    # Confidence 0.50 -> legacy proxy edge would be exactly 0.
    sig_obj = _Sig(confidence=0.50,
                   context={"ev_breakdown": {"ev_bps": 1517.9, "cost_bps": 16.0}})
    edge = pt._estimate_signal_edge_bps(sig_obj, {"side": "short"})
    # gross = ev_bps + cost_bps = 1533.9 (no confirmation nudges)
    assert edge == pytest.approx(1533.9, abs=1e-3)


def test_coldstart_short_passes_quality_gate_with_firewall_ev():
    pt = _pt()
    sig_obj = _Sig(confidence=0.50,
                   context={"ev_breakdown": {"ev_bps": 1517.9, "cost_bps": 16.0}})
    ok, meta = pt._passes_trade_quality_gate(sig_obj, {"side": "short"})
    assert ok is True, meta
    assert meta["reason"] == "passed"


def test_coldstart_short_without_breakdown_still_gated():
    """No ev_breakdown -> legacy behaviour: cold-start short still
    rejected (proxy edge ~0, no confirmation)."""
    pt = _pt()
    sig_obj = _Sig(confidence=0.50, context={})
    ok, meta = pt._passes_trade_quality_gate(sig_obj, {"side": "short"})
    assert ok is False
    assert meta["reason"] in ("short_lacks_confirmation", "edge_below_cost")


def test_negative_firewall_ev_still_rejected():
    """A firewall EV that's net-negative must NOT pass the gate."""
    pt = _pt()
    sig_obj = _Sig(confidence=0.50,
                   context={"ev_breakdown": {"ev_bps": -40.0, "cost_bps": 16.0}})
    ok, meta = pt._passes_trade_quality_gate(sig_obj, {"side": "short"})
    assert ok is False
    # gross = -40 + 16 = -24 < minimum -> edge_below_cost; and
    # firewall_ev_ok is False (ev_bps<=0) so short-confirm also fails.
    assert meta["reason"] in ("edge_below_cost", "short_lacks_confirmation")


def test_knob_off_reverts_to_confidence_proxy(monkeypatch):
    monkeypatch.setattr(config, "TRADE_QUALITY_USE_FIREWALL_EV", False, raising=False)
    pt = _pt()
    sig_obj = _Sig(confidence=0.50,
                   context={"ev_breakdown": {"ev_bps": 1517.9, "cost_bps": 16.0}})
    # With the knob off the firewall EV is ignored -> proxy edge ~0.
    edge = pt._estimate_signal_edge_bps(sig_obj, {"side": "short"})
    assert edge == pytest.approx(0.0, abs=1e-6)
