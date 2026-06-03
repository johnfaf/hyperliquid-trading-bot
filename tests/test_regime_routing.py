"""Regime routing (signal #4): drop strategies that don't fit the current
regime instead of just down-weighting them. Flag-gated default OFF."""
from __future__ import annotations

import config
from src.analysis.regime_strategy_filter import RegimeStrategyFilter


def _strats():
    return [
        {"strategy_type": "momentum_long", "current_score": 0.8},
        {"strategy_type": "mean_reversion", "current_score": 0.8},
        {"strategy_type": "momentum_short", "current_score": 0.8},
    ]


_RANGING = {"regime": "ranging", "confidence": 0.8, "adx": 15.0}


def test_off_keeps_all_strategies(monkeypatch):
    monkeypatch.setattr(config, "REGIME_ROUTING_ENABLED", False, raising=False)
    out = RegimeStrategyFilter().filter(_strats(), _RANGING)
    assert len(out) == 3   # re-scored but none dropped


def test_routing_drops_low_compatibility(monkeypatch):
    # In 'ranging': mean_reversion compat 1.0 (keep), momentum_long/short 0.3.
    monkeypatch.setattr(config, "REGIME_ROUTING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "REGIME_ROUTING_MIN_COMPAT", 0.5, raising=False)
    out = RegimeStrategyFilter().filter(_strats(), _RANGING)
    types = {s.get("strategy_type") for s in out}
    assert "mean_reversion" in types
    assert "momentum_long" not in types and "momentum_short" not in types


def test_routing_keeps_fit_strategy(monkeypatch):
    # In 'trending_up': momentum_long compat 1.0 -> kept even at a high bar.
    monkeypatch.setattr(config, "REGIME_ROUTING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "REGIME_ROUTING_MIN_COMPAT", 0.5, raising=False)
    out = RegimeStrategyFilter().filter(
        [{"strategy_type": "momentum_long", "current_score": 0.8}],
        {"regime": "trending_up", "confidence": 0.8, "adx": 35.0},
    )
    assert any(s.get("strategy_type") == "momentum_long" for s in out)
