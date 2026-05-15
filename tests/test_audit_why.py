"""Tests for the audit router's why-enter/why-reject parser."""
from __future__ import annotations

from src.ui.v2.routers.audit import _build_why, _loads


def test_loads_handles_json_string_and_dict_and_garbage():
    assert _loads('{"a": 1}') == {"a": 1}
    assert _loads({"b": 2}) == {"b": 2}
    assert _loads("not json") == {}
    assert _loads(None) == {}
    assert _loads("") == {}


def test_build_why_entered_with_ev():
    metadata = {
        "why_entered": {
            "signal_reason": "Strategy: alpha_momentum (momentum_long)",
            "strategy_type": "momentum_long",
            "rejection_reason": None,
        },
        "market_read": {
            "overall_regime": "trending_up",
            "overall_confidence": 0.72,
            "countertrend_block_side": None,
        },
        "risk_and_sizing": {"leverage": 3.0, "position_pct": 0.05, "entry_price": 2300.0},
        "ev_breakdown": {
            "ev_bps": 42.5, "sigma_bps": 18.0, "p_win": 0.58,
            "p_win_source": "calibration_tracker",
            "avg_win_bps": 300.0, "avg_loss_bps": 150.0, "cost_bps": 39.0,
        },
    }
    decision = {"final_status": "firewall_prescreen_approved", "rejection_reason": None}
    why = _build_why(metadata, decision)
    assert why["verdict"] == "FIREWALL_PRESCREEN_APPROVED"
    assert why["regime"] == "trending_up"
    assert why["has_ev"] is True
    assert why["ev"]["ev_bps"] == 42.5
    assert why["ev"]["p_win_source"] == "calibration_tracker"
    assert why["strategy_type"] == "momentum_long"


def test_build_why_reject_marks_verdict_and_reason():
    metadata = {
        "why_entered": {"signal_reason": "copy 0xabc", "strategy_type": ""},
        "market_read": {"overall_regime": "ranging"},
    }
    decision = {
        "final_status": "rejected",
        "rejection_reason": "ev_below_threshold:ev=-5bps<=thr=30bps",
    }
    why = _build_why(metadata, decision)
    assert why["verdict"] == "REJECT"
    assert "ev_below_threshold" in why["rejection_reason"]
    assert why["has_ev"] is False
    assert why["ev"] == {}


def test_build_why_empty_metadata_is_safe():
    why = _build_why({}, {"final_status": "candidate"})
    assert why["verdict"] == "CANDIDATE"
    assert why["has_ev"] is False
    assert why["regime"] == ""
    assert why["rejection_reason"] == ""


def test_build_why_rejection_inferred_from_reason_even_if_status_blank():
    why = _build_why({}, {"final_status": "", "rejection_reason": "rejected_data_readiness"})
    assert why["verdict"] == "REJECT"
    assert why["rejection_reason"] == "rejected_data_readiness"
