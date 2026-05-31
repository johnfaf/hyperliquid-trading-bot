"""Portfolio correlation / net-exposure cap (algo #7).

On a correlated crypto book the engine can stack many same-direction positions
(one big beta bet). The cap limits concurrent same-side positions (existing open
+ new this cycle), dropping the lowest-ranked new same-side candidates. Flag-
gated default OFF.
"""
from __future__ import annotations

from src.signals.decision_engine import DecisionEngine


def _longs(n):
    return [{"name": f"s{i}", "strategy_type": "momentum_long",
             "current_score": 0.80, "confidence": 0.80,
             "parameters": {"coins": [f"C{i}"]}} for i in range(n)]


def _long_count(out):
    return sum(1 for o in out if str(o.get("_decision_side", "")).lower() == "long")


def test_cap_limits_new_same_side():
    eng = DecisionEngine(config={"corr_cap_enabled": True, "max_same_side_positions": 2,
                                 "min_decision_score": 0.0, "max_positions": 8,
                                 "max_prescreen_candidates": 8})
    out = eng.decide(_longs(4), regime_data=None, open_positions=[])
    assert _long_count(out) == 2, "should cap concurrent long positions at 2"


def test_cap_counts_existing_open_positions():
    eng = DecisionEngine(config={"corr_cap_enabled": True, "max_same_side_positions": 2,
                                 "min_decision_score": 0.0, "max_positions": 8,
                                 "max_prescreen_candidates": 8})
    # one long already open -> only 1 more new long allowed to reach the cap of 2
    out = eng.decide(_longs(4), regime_data=None,
                     open_positions=[{"side": "long", "coin": "ZZZ"}])
    assert _long_count(out) == 1


def test_cap_off_does_not_limit():
    eng = DecisionEngine(config={"corr_cap_enabled": False, "min_decision_score": 0.0,
                                 "max_positions": 8, "max_prescreen_candidates": 8})
    out = eng.decide(_longs(4), regime_data=None, open_positions=[])
    assert _long_count(out) >= 3, "no cap when disabled"
