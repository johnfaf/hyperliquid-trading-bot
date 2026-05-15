"""Options-flow per-day cap graduation.

Once an options_flow source has > OPTIONS_FLOW_CAP_MIN_TRADES closed
trades its per-day cap graduates to OPTIONS_FLOW_GRADUATED_CAP,
lifting it out of the warmup/degraded 1-signal/day throttle that
rejected 23 of 93 decisions / 6h in prod. Never overrides paused.
"""
from __future__ import annotations

import pytest

from src.signals.agent_scoring import AgentScorer


def _scorer(**over):
    cfg = {
        "policy_enabled": True,
        "policy_warmup_max_signals_per_day": 1,
        "policy_degraded_max_signals_per_day": 1,
        "policy_active_min_signals_per_day": 3,
        "policy_active_max_signals_per_day": 8,
        "options_flow_cap_graduation_enabled": True,
        "options_flow_cap_min_trades": 3,
        "options_flow_graduated_cap": 4,
    }
    cfg.update(over)
    return AgentScorer(cfg)


def _patch_scorecard(scorer, *, status, completed, source_key):
    scorer.get_scorecard = lambda: [{
        "source_key": source_key,
        "status": status,
        "rank": 1,
        "dynamic_weight": 0.5,
        "weighted_accuracy": 0.5,
        "completed_trades": completed,
        "recent_pnl": 0.0,
        "win_rate": 0.5,
    }]


def test_degraded_options_flow_graduates_after_more_than_3(monkeypatch):
    s = _scorer()
    _patch_scorecard(s, status="degraded", completed=4,
                     source_key="options_flow:options_momentum")
    p = s.get_source_policy("options_flow:options_momentum")
    assert p["max_signals_per_day"] == 4
    assert "options_flow_graduated" in p["dynamic_cap_reason"]


def test_exactly_3_trades_not_yet_graduated(monkeypatch):
    """'above 3' is strictly > 3; at exactly 3 it stays throttled."""
    s = _scorer()
    _patch_scorecard(s, status="degraded", completed=3,
                     source_key="options_flow:options_momentum")
    p = s.get_source_policy("options_flow:options_momentum")
    assert p["max_signals_per_day"] == 1  # degraded fixed cap, not lifted


def test_warmup_options_flow_graduates(monkeypatch):
    s = _scorer()
    _patch_scorecard(s, status="warmup", completed=10,
                     source_key="options_flow")
    p = s.get_source_policy("options_flow")
    assert p["max_signals_per_day"] == 4


def test_paused_options_flow_NOT_graduated(monkeypatch):
    """A hard safety pause must stay hard regardless of trade count."""
    s = _scorer()
    _patch_scorecard(s, status="paused", completed=99,
                     source_key="options_flow:options_momentum")
    p = s.get_source_policy("options_flow:options_momentum")
    assert p["blocked"] is True
    assert p["max_signals_per_day"] == 0


def test_non_options_flow_source_unaffected(monkeypatch):
    s = _scorer()
    _patch_scorecard(s, status="degraded", completed=50,
                     source_key="strategy:momentum_short")
    p = s.get_source_policy("strategy:momentum_short")
    assert p["max_signals_per_day"] == 1  # untouched by graduation


def test_graduation_never_lowers_an_already_higher_cap(monkeypatch):
    s = _scorer(policy_active_max_signals_per_day=8)
    _patch_scorecard(s, status="active", completed=50,
                     source_key="options_flow:options_momentum")
    p = s.get_source_policy("options_flow:options_momentum")
    # active dynamic cap may already exceed 4 -> graduation must not cut it
    assert p["max_signals_per_day"] >= 4


def test_knob_disabled_keeps_throttle(monkeypatch):
    s = _scorer(options_flow_cap_graduation_enabled=False)
    _patch_scorecard(s, status="degraded", completed=20,
                     source_key="options_flow:options_momentum")
    p = s.get_source_policy("options_flow:options_momentum")
    assert p["max_signals_per_day"] == 1
