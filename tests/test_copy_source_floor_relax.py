"""#2 Opt-in copy-source-floor synthetic relaxation.

DEFAULT OFF: the bulk of "Source allocator requires 45% confidence for
copy_trade" is the AgentScorer correctly down-weighting UNPROVEN sources
(a legitimate live-money control). The lever only relaxes the floor for
signals whose confidence was capped by a synthetic / non-authoritative
regime, and only when an operator opts in -- so OFF == zero change.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.signals.decision_firewall import DecisionFirewall


class _Scorer:
    def get_source_policy(self, source_key):
        return {
            "source_key": source_key, "status": "active", "blocked": False,
            "rank": 1, "max_signals_per_day": 0, "size_multiplier": 1.0,
            "min_confidence": 0.45, "dynamic_weight": 0.6,
            "weighted_accuracy": 0.6, "completed_trades": 10, "recent_pnl": 0.0,
        }

    def get_scorecard(self):
        return [self.get_source_policy("copy_trade:0xabc")]


def _sig(conf, *, synthetic=True):
    s = MagicMock()
    s.coin = "BTC"
    s.confidence = conf
    s.side = MagicMock()
    s.side.value = "long"
    s.regime = ""
    s.source = "copy_trade"
    s.context = {"regime_data_quality": "synthetic_warm_start"} if synthetic else {}
    return s


def _fw(**over):
    cfg = {
        "enable_predictive_derisk": False, "funding_risk_enabled": False,
        "agent_scorer": _Scorer(),
    }
    cfg.update(over)
    return DecisionFirewall(cfg)


def test_default_off_still_rejects_below_floor():
    fw = _fw()  # relax disabled by default
    assert fw.copy_source_floor_synthetic_relax_enabled is False
    ok, reason, _ = fw._apply_source_policy(_sig(0.42), "copy_trade:0xabc")
    assert ok is False
    assert "requires 45% confidence" in reason


def test_enabled_relaxes_floor_for_synthetic_capped_copy():
    fw = _fw(copy_source_floor_synthetic_relax_enabled=True,
             copy_source_floor_synthetic_relax=0.07)
    # 0.45 - 0.07 = 0.38; conf 0.42 now clears the relaxed floor
    ok, reason, _ = fw._apply_source_policy(_sig(0.42), "copy_trade:0xabc")
    assert ok is True, reason


def test_enabled_does_not_relax_when_regime_not_synthetic():
    fw = _fw(copy_source_floor_synthetic_relax_enabled=True,
             copy_source_floor_synthetic_relax=0.07)
    ok, reason, _ = fw._apply_source_policy(_sig(0.42, synthetic=False),
                                            "copy_trade:0xabc")
    assert ok is False
    assert "requires 45% confidence" in reason


def test_enabled_does_not_relax_non_copy_sources():
    fw = _fw(copy_source_floor_synthetic_relax_enabled=True,
             copy_source_floor_synthetic_relax=0.07)
    ok, reason, _ = fw._apply_source_policy(_sig(0.42), "strategy:momentum")
    assert ok is False
    assert "requires 45% confidence" in reason


def test_relaxed_floor_still_rejects_well_below():
    fw = _fw(copy_source_floor_synthetic_relax_enabled=True,
             copy_source_floor_synthetic_relax=0.07)
    # 0.30 is still below the relaxed 0.38 floor -> still rejected
    ok, reason, _ = fw._apply_source_policy(_sig(0.30), "copy_trade:0xabc")
    assert ok is False
