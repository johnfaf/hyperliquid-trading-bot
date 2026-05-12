"""Regression tests for copy_trader confidence math + unit defense.

User report: copy_trader signals consistently rejected at exactly 43%
confidence vs source-allocator's 45% warmup floor. Investigation showed
the math IS correct (`copy_scale_in` base 0.40 + win_rate 0.06 * 0.5
= 0.43) but the function had no defense against win_rate-as-percent,
and the rejection log gave no clue WHY 43%.
"""
import pytest

from src.trading.copy_trader import (
    CopyTrader, _SIGNAL_CONFIDENCE_MODEL,
)


# --- Confidence math regression ---------------------------------------

def test_reproduces_user_43pct_scenario():
    """The user's actual symptom: copy_scale_in on a trader with 6% win
    rate gives 0.43 confidence, which is below the 0.45 warmup floor."""
    conf = CopyTrader._calculate_signal_confidence("copy_scale_in", 0.06)
    assert abs(conf - 0.43) < 0.01, (
        f"Expected ~0.43 (matches user's report), got {conf}"
    )


def test_known_baselines():
    """Spot-check each signal type's base at win_rate=0."""
    for stype, model in _SIGNAL_CONFIDENCE_MODEL.items():
        conf = CopyTrader._calculate_signal_confidence(stype, 0.0)
        assert conf == model["base"], (
            f"{stype}: expected {model['base']} at WR=0, got {conf}"
        )


def test_perfect_win_rate_caps_at_max():
    """A 100% WR trader doesn't push confidence past the cap."""
    for stype, model in _SIGNAL_CONFIDENCE_MODEL.items():
        conf = CopyTrader._calculate_signal_confidence(stype, 1.0)
        assert conf <= model["max"] + 1e-9, f"{stype}: {conf} > cap {model['max']}"


# --- Unit defense -----------------------------------------------------

def test_percent_win_rate_normalized_to_fraction():
    """If win_rate is mistakenly passed as a percent (60.0 meaning 60%),
    the function should produce the same result as 0.60. Without the
    defense, win_rate=60.0 would clamp to 1.0 -> always cap.

    The codebase already uses this `if x > 1.5: x /= 100` idiom in
    src/data/database.py:150 and src/analysis/strategy_scorer.py:176,
    so copy_trader behaving differently was a latent inconsistency.
    """
    as_pct = CopyTrader._calculate_signal_confidence("copy_open", 60.0)
    as_frac = CopyTrader._calculate_signal_confidence("copy_open", 0.60)
    assert abs(as_pct - as_frac) < 1e-9, (
        f"60.0 (pct) gave {as_pct}, 0.60 (frac) gave {as_frac}; "
        "unit defense regressed"
    )


def test_negative_win_rate_clamped_to_zero():
    """Pathological negative win_rate (data corruption) -> base value."""
    conf = CopyTrader._calculate_signal_confidence("copy_open", -0.5)
    assert conf == _SIGNAL_CONFIDENCE_MODEL["copy_open"]["base"]


def test_none_or_zero_win_rate_is_safe():
    """None or 0.0 -> base value, no division-by-zero or coercion errors."""
    for wr in (None, 0, 0.0):
        conf = CopyTrader._calculate_signal_confidence("copy_open", wr)
        assert conf == _SIGNAL_CONFIDENCE_MODEL["copy_open"]["base"]


def test_string_winrate_coerced_or_safe():
    """If win_rate arrives as a string from JSON-loaded DB metadata,
    the function should not crash; treating it as fraction is OK."""
    conf = CopyTrader._calculate_signal_confidence("copy_open", "0.5")
    assert 0.5 < conf <= _SIGNAL_CONFIDENCE_MODEL["copy_open"]["max"]


# --- Observability ----------------------------------------------------

def test_signal_carries_confidence_inputs_for_diagnostics():
    """Every copy_trader signal must carry confidence_inputs so the
    firewall's rejection log can show win_rate + signal_type, which
    is what would have told the user why 43% in the first place."""
    ct = CopyTrader(firewall=None, agent_scorer=None)

    trader = {"address": "0xabc", "win_rate": 0.06, "trade_count": 17, "total_pnl": 100}
    old = {}
    new = {"BTC": {"side": "long", "size": 1.0, "leverage": 5, "entry_price": 50_000}}
    mids = {"BTC": 50_000}

    signals = ct._detect_position_changes("0xabc", old, new, trader, mids)
    assert signals, "Expected at least one signal"
    s = signals[0]
    assert "confidence_inputs" in s, f"Signal missing confidence_inputs: {s}"
    inputs = s["confidence_inputs"]
    assert inputs["win_rate"] == pytest.approx(0.06)
    assert inputs["trade_count"] == 17
    assert inputs["signal_type"] == "copy_open"
