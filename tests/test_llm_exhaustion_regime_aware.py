"""Regime-aware LLM exhaustion guard.

A short into RSI<22 while regime==TRENDING_DOWN is trend continuation,
not an exhaustion trap -- it must NOT be hard-blocked (that deadlocked
the bot: LLM pass_rate 4%, 0 orders). Non-aligned / ranging / contra
contexts keep the hard block.
"""
from __future__ import annotations

import pytest

from src.signals.llm_filter import LLMFilter


def _sig(side="short", rsi=18, coin="SOL", conf=0.6):
    return {
        "coin": coin,
        "side": side,
        "confidence": conf,
        "features": {"rsi": rsi, "bollinger_position": 0.0},
    }


def _ctx(regime):
    return {"regime_data": {"overall_regime": regime}}


def _filter(**over):
    cfg = {
        "check_memory": False,
        "check_conflicts": False,
        "check_correlation": False,
        "exhaustion_regime_aware": True,
        "exhaustion_trend_aligned_conf_mult": 0.85,
    }
    cfg.update(over)
    return LLMFilter(cfg)


def test_short_oversold_in_downtrend_is_not_blocked():
    f = _filter()
    ok, conf, reason = f.filter(_sig(side="short", rsi=18, conf=0.6),
                                _ctx("TRENDING_DOWN"))
    assert ok is True, f"trend-aligned oversold short must pass, got: {reason}"
    assert conf == pytest.approx(0.6 * 0.85, abs=1e-6), "should be de-risked, not blocked"
    assert "trend-aligned" in reason


def test_short_oversold_in_ranging_still_hard_blocked():
    f = _filter()
    ok, conf, reason = f.filter(_sig(side="short", rsi=18),
                                _ctx("RANGING"))
    assert ok is False
    assert "Exhaustion block: shorting" in reason


def test_short_oversold_no_regime_still_hard_blocked():
    f = _filter()
    ok, conf, reason = f.filter(_sig(side="short", rsi=18), {"regime_data": {}})
    assert ok is False
    assert "Exhaustion block" in reason


def test_long_overbought_in_uptrend_is_not_blocked():
    f = _filter()
    ok, conf, reason = f.filter(_sig(side="long", rsi=85, conf=0.6),
                                _ctx("TRENDING_UP"))
    assert ok is True, reason
    assert conf == pytest.approx(0.6 * 0.85, abs=1e-6)


def test_long_overbought_in_downtrend_still_blocked():
    """Long + overbought + TRENDING_DOWN: not trend-aligned -> hard block."""
    f = _filter()
    ok, conf, reason = f.filter(_sig(side="long", rsi=85),
                                _ctx("TRENDING_DOWN"))
    assert ok is False
    assert "Exhaustion block: longing" in reason


def test_regime_aware_disabled_reverts_to_hard_block():
    f = _filter(exhaustion_regime_aware=False)
    ok, conf, reason = f.filter(_sig(side="short", rsi=18),
                                _ctx("TRENDING_DOWN"))
    assert ok is False
    assert "Exhaustion block: shorting" in reason


def test_non_extreme_rsi_short_in_downtrend_passes_clean():
    """RSI 45 short in downtrend: not exhaustion territory, full conf."""
    f = _filter()
    ok, conf, reason = f.filter(_sig(side="short", rsi=45, conf=0.6),
                                _ctx("TRENDING_DOWN"))
    assert ok is True
    assert conf == pytest.approx(0.6, abs=1e-6)  # no haircut
