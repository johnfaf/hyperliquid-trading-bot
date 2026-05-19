"""#1 Copy-source-floor synthetic EXEMPTION (default ON) + #2 relax fallback.

Evidence (logs.1779171559187): copy confidence is flattened to 0.50 by
the synthetic-regime cap then blended to a CONSTANT ~0.43, sitting
deterministically under the 0.45 source floor -> 100% of copy rejected
while the forecaster is synthetic. That is a structural dead-zone, not
the AgentScorer grading source merit (merit was erased upstream). So a
synthetic-capped copy signal is EXEMPT from the source-confidence floor
by default; all other source-policy checks still apply. Detection is
authoritative via regime_data.forecaster_synthetic_warm_start (the prior
#2 lever gated only on signal.context, which signal_from_copy_trade
never sets -> it was dead in production).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.signals.decision_firewall import DecisionFirewall

_SYNTH = {"forecaster_synthetic_warm_start": True}


class _Scorer:
    def __init__(self, **over):
        self._p = {
            "source_key": "", "status": "active", "blocked": False,
            "rank": 1, "max_signals_per_day": 0, "size_multiplier": 1.0,
            "min_confidence": 0.45, "dynamic_weight": 0.6,
            "weighted_accuracy": 0.6, "completed_trades": 10, "recent_pnl": 0.0,
        }
        self._p.update(over)

    def get_source_policy(self, source_key):
        return {**self._p, "source_key": source_key}

    def get_scorecard(self):
        return [self.get_source_policy("copy_trade:0xabc")]


def _sig(conf, *, ctx_synth=False):
    s = MagicMock()
    s.coin = "BTC"
    s.confidence = conf
    s.side = MagicMock()
    s.side.value = "long"
    s.regime = ""
    s.source = "copy_trade"
    # Production reality: signal_from_copy_trade sets NO context. Only set
    # the legacy context marker when explicitly testing the fallback path.
    s.context = {"regime_data_quality": "synthetic_warm_start"} if ctx_synth else {}
    return s


def _fw(scorer=None, **over):
    cfg = {
        "enable_predictive_derisk": False, "funding_risk_enabled": False,
        "agent_scorer": scorer or _Scorer(),
    }
    cfg.update(over)
    return DecisionFirewall(cfg)


# ── #1: exemption is ON by default and breaks the deadlock ──

def test_exempt_default_on_lets_synthetic_capped_copy_pass():
    fw = _fw()
    assert fw.copy_source_floor_synthetic_exempt_enabled is True
    # conf 0.42 < 0.45 floor, but synthetic-capped via regime_data -> PASS
    ok, reason, _ = fw._apply_source_policy(
        _sig(0.42), "copy_trade:0xabc", regime_data=dict(_SYNTH))
    assert ok is True, reason


def test_exempt_detects_via_regime_data_not_just_context():
    """Production path: signal.context is empty (signal_from_copy_trade
    never sets it); detection must come from regime_data."""
    fw = _fw()
    ok, reason, _ = fw._apply_source_policy(
        _sig(0.30, ctx_synth=False), "copy_trade:0xabc",
        regime_data=dict(_SYNTH))
    assert ok is True, reason  # exempt regardless of how far below floor


def test_exempt_also_honors_legacy_context_marker_fallback():
    fw = _fw()
    ok, _, _ = fw._apply_source_policy(
        _sig(0.42, ctx_synth=True), "copy_trade:0xabc")  # no regime_data
    assert ok is True


# ── guards: exemption is narrow ──

def test_non_synthetic_copy_still_floored():
    """Genuine low-merit copy under a REAL regime is still rejected --
    the exemption only covers the synthetic-cap dead-zone."""
    fw = _fw()
    ok, reason, _ = fw._apply_source_policy(
        _sig(0.42), "copy_trade:0xabc", regime_data={})
    assert ok is False
    assert "requires 45% confidence" in reason


def test_non_copy_source_never_exempt():
    fw = _fw()
    ok, reason, _ = fw._apply_source_policy(
        _sig(0.42), "strategy:momentum", regime_data=dict(_SYNTH))
    assert ok is False
    assert "requires 45% confidence" in reason


def test_exemption_does_not_bypass_blocked_source():
    """Exemption skips ONLY the confidence floor -- a paused/blocked
    source is still rejected."""
    fw = _fw(scorer=_Scorer(blocked=True, status="paused"))
    ok, reason, _ = fw._apply_source_policy(
        _sig(0.42), "copy_trade:0xabc", regime_data=dict(_SYNTH))
    assert ok is False
    assert "paused" in reason.lower()


# ── operator can revert to legacy floor; #2 relax still works as fallback ──

def test_exempt_disabled_reverts_to_legacy_floor():
    fw = _fw(copy_source_floor_synthetic_exempt_enabled=False)
    ok, reason, _ = fw._apply_source_policy(
        _sig(0.42), "copy_trade:0xabc", regime_data=dict(_SYNTH))
    assert ok is False
    assert "requires 45% confidence" in reason


def test_exempt_disabled_relax_fallback_applies():
    fw = _fw(copy_source_floor_synthetic_exempt_enabled=False,
             copy_source_floor_synthetic_relax_enabled=True,
             copy_source_floor_synthetic_relax=0.07)
    # 0.45 - 0.07 = 0.38; conf 0.42 clears the relaxed floor
    ok, reason, _ = fw._apply_source_policy(
        _sig(0.42), "copy_trade:0xabc", regime_data=dict(_SYNTH))
    assert ok is True, reason


def test_exempt_disabled_relax_still_rejects_well_below():
    fw = _fw(copy_source_floor_synthetic_exempt_enabled=False,
             copy_source_floor_synthetic_relax_enabled=True,
             copy_source_floor_synthetic_relax=0.07)
    ok, _, _ = fw._apply_source_policy(
        _sig(0.30), "copy_trade:0xabc", regime_data=dict(_SYNTH))
    assert ok is False  # 0.30 still below relaxed 0.38
