"""EV-first ranking (algo #3).

The DecisionEngine's heuristic composite (base score, regime, freshness,
consensus) can rank a high-composite/low-edge signal above a low-composite/
high-edge one. With EV-first enabled, the net-of-cost EV proxy (driven by
calibrated confidence) becomes the primary sort key, so the higher-edge signal
wins -- while remaining strictly more selective (composite floor still applies
plus a minimum-EV gate). OFF reproduces the legacy composite ranking.
"""
from __future__ import annotations

from datetime import timedelta

from src.core import clock_provider
from src.signals.decision_engine import DecisionEngine


def _strats():
    now = clock_provider.utc_now()
    old = (now - timedelta(hours=100)).isoformat()
    fresh = now.isoformat()
    # A: high base score + fresh, but LOW confidence (low EV).
    a = {
        "name": "stratA", "strategy_type": "momentum_long",
        "current_score": 0.90, "confidence": 0.45,
        "discovered_at": fresh,
        "parameters": {"coins": ["AAA"]},
    }
    # B: low base score + stale, but HIGH confidence (high EV).
    b = {
        "name": "stratB", "strategy_type": "momentum_long",
        "current_score": 0.40, "confidence": 0.80,
        "discovered_at": old,
        "parameters": {"coins": ["BBB"]},
    }
    return [a, b]


def _coin(decided):
    params = decided.get("parameters", {})
    coins = params.get("coins") if isinstance(params, dict) else None
    return (coins or [decided.get("_decision_coin")])[0]


def test_legacy_composite_ranks_high_base_first():
    eng = DecisionEngine(config={"ev_first_enabled": False, "min_decision_score": 0.20})
    out = eng.decide(_strats(), regime_data=None, open_positions=[])
    assert out, "both candidates should qualify"
    assert _coin(out[0]) == "AAA", "composite should rank the high-base/fresh signal first"


def test_ev_first_ranks_high_confidence_first():
    eng = DecisionEngine(config={"ev_first_enabled": True, "min_decision_score": 0.20,
                                 "min_ev_r": 0.0})
    out = eng.decide(_strats(), regime_data=None, open_positions=[])
    assert out, "candidates with positive EV should qualify"
    assert _coin(out[0]) == "BBB", "EV-first should rank the higher-edge signal first"
    # EV proxy is attached and the top one is the largest.
    assert out[0]["_ev_proxy"] >= out[-1]["_ev_proxy"]


def test_ev_first_gates_out_negative_ev():
    # A clearly -EV candidate (very low confidence) must be dropped when a
    # positive min_ev_r is required, even if its composite clears the floor.
    now = clock_provider.utc_now().isoformat()
    weak = {
        "name": "weak", "strategy_type": "momentum_long",
        "current_score": 0.95, "confidence": 0.20, "discovered_at": now,
        "parameters": {"coins": ["WEAK"]},
    }
    eng = DecisionEngine(config={"ev_first_enabled": True, "min_decision_score": 0.20,
                                 "min_ev_r": 0.10})
    out = eng.decide([weak], regime_data=None, open_positions=[])
    coins = [_coin(o) for o in out]
    assert "WEAK" not in coins, "negative-EV candidate must be gated out under EV-first"
