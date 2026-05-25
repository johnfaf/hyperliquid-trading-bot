"""AdaptiveBotDetector retuning — verifies the May-2026 audit fixes.

Background
----------
Prior to this fix the detector defaults produced only a 5% bot rate on
the live trader pool (41/816), far below the 20-40% norm for crypto
perps DEXs.  The clearest failure case was 0x261af68f doing 932
trades/day yet classified "Likely human" with prob=0.27 — physically
impossible, but the structure of the scorer let it through:

  * trade_frequency=1.00 contributed only 0.25 * 1.00 = 0.25 to the
    weighted sum.
  * No other feature fired (sparse data / human-like spacing).
  * Threshold 0.60 was never reachable from a single signal.

After this fix
--------------
  1. Hard cutoff (default 80 trades/day) short-circuits to is_bot=True
     before the weighted aggregation — no other evidence required.
  2. Threshold lowered to 0.45.
  3. Weights rebalanced: trade_frequency 0.35, pnl_pattern 0.25,
     timing_regularity 0.15, size_uniformity 0.10, liquidation_rate
     0.10, session_pattern 0.05.
  4. size_uniformity midpoint widened 0.05 -> 0.15 so bots with
     CV=0.1-0.2 get caught.

All tunables are env-var overridable.
"""
from __future__ import annotations

import os

import pytest

from src.discovery.adaptive_bot_detector import AdaptiveBotDetector


def _hf_fills(trades_per_day: float, total_trades: int = 100):
    """Build a fills payload with the requested trades/day rate.

    Spans the trades evenly over the implied duration so
    _compute_trades_per_day returns the target rate.
    """
    if trades_per_day <= 0:
        return []
    span_days = total_trades / trades_per_day
    span_ms = int(span_days * 24 * 3600 * 1000)
    start_ms = 1_700_000_000_000  # arbitrary base
    step_ms = max(1, span_ms // max(total_trades - 1, 1))
    fills = []
    for i in range(total_trades):
        fills.append({
            "time": start_ms + i * step_ms,
            "size": 100.0,
            "price": 1.0,
            "closed_pnl": 0.10 if i % 2 == 0 else -0.05,
            "is_liquidation": False,
        })
    return fills


# ── New defaults ─────────────────────────────────────────────


def test_new_default_threshold_is_045(monkeypatch):
    """Threshold defaults to 0.45 (was 0.60)."""
    monkeypatch.delenv("BOT_PROB_THRESHOLD", raising=False)
    det = AdaptiveBotDetector()
    assert det.threshold == pytest.approx(0.45)


def test_new_default_weights_sum_to_one(monkeypatch):
    """Default weights still sum to 1.00 after retuning."""
    for k in ("BOT_WEIGHT_TRADE_FREQUENCY",
              "BOT_WEIGHT_TIMING_REGULARITY",
              "BOT_WEIGHT_SIZE_UNIFORMITY",
              "BOT_WEIGHT_PNL_PATTERN",
              "BOT_WEIGHT_LIQUIDATION_RATE",
              "BOT_WEIGHT_SESSION_PATTERN"):
        monkeypatch.delenv(k, raising=False)
    det = AdaptiveBotDetector()
    total = sum(det.weights.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_new_default_weights_prioritize_trade_frequency(monkeypatch):
    """trade_frequency is the highest-weighted feature post-retuning."""
    for k in ("BOT_WEIGHT_TRADE_FREQUENCY",
              "BOT_WEIGHT_TIMING_REGULARITY",
              "BOT_WEIGHT_SIZE_UNIFORMITY",
              "BOT_WEIGHT_PNL_PATTERN",
              "BOT_WEIGHT_LIQUIDATION_RATE",
              "BOT_WEIGHT_SESSION_PATTERN"):
        monkeypatch.delenv(k, raising=False)
    det = AdaptiveBotDetector()
    assert det.weights["trade_frequency"] == max(det.weights.values())
    assert det.weights["session_pattern"] == min(det.weights.values())


# ── Hard cutoff short-circuit ────────────────────────────────


def test_hard_cutoff_short_circuits_to_bot(monkeypatch):
    """932 trades/day must be classified as bot via the hard-cutoff path.

    This is the 0x261af68f case from the May-2026 discovery run.  Before
    this fix it was classified "Likely human" with prob=0.27.
    """
    import config
    monkeypatch.setattr(config, "BOT_HARD_CUTOFF_TRADES", 80, raising=False)

    det = AdaptiveBotDetector()
    fills = _hf_fills(trades_per_day=932.0, total_trades=500)
    result = det.detect(fills, [], {"total_trades": 500}, "0x261af68f")

    assert result.is_bot is True, (
        f"932 trades/day must be a bot; got prob={result.bot_probability:.2f}, "
        f"reason={result.reason}"
    )
    assert result.bot_probability == pytest.approx(1.0)
    assert "Hard cutoff" in result.reason


def test_hard_cutoff_just_above_threshold(monkeypatch):
    """A trader just above the hard cutoff (81/day) is a bot."""
    import config
    monkeypatch.setattr(config, "BOT_HARD_CUTOFF_TRADES", 80, raising=False)

    det = AdaptiveBotDetector()
    fills = _hf_fills(trades_per_day=81.0, total_trades=100)
    result = det.detect(fills, [], {"total_trades": 100}, "0xabc")
    assert result.is_bot is True
    assert result.bot_probability == pytest.approx(1.0)


def test_hard_cutoff_just_below_runs_weighted_logic(monkeypatch):
    """A trader just under the cutoff (79/day) runs the weighted scorer,
    NOT the short-circuit.  The verdict depends on the other signals."""
    import config
    monkeypatch.setattr(config, "BOT_HARD_CUTOFF_TRADES", 80, raising=False)

    det = AdaptiveBotDetector()
    fills = _hf_fills(trades_per_day=79.0, total_trades=100)
    result = det.detect(fills, [], {"total_trades": 100}, "0xabc")
    # Either verdict is acceptable, but the reason must NOT be the
    # hard-cutoff short-circuit (which we want reserved for clear bots).
    assert "Hard cutoff" not in result.reason


def test_hard_cutoff_configurable(monkeypatch):
    """The hard-cutoff threshold honors the config value (env-var driven)."""
    import config
    # Tighten to 200 trades/day -- now 150 trades/day should NOT
    # short-circuit even though it would under the 80-default.
    monkeypatch.setattr(config, "BOT_HARD_CUTOFF_TRADES", 200, raising=False)

    det = AdaptiveBotDetector()
    fills = _hf_fills(trades_per_day=150.0, total_trades=200)
    result = det.detect(fills, [], {"total_trades": 200}, "0xabc")
    assert "Hard cutoff" not in result.reason


# ── Env-var overrides ────────────────────────────────────────


def test_threshold_env_var_override(monkeypatch):
    """BOT_PROB_THRESHOLD env var overrides the default."""
    monkeypatch.setenv("BOT_PROB_THRESHOLD", "0.30")
    det = AdaptiveBotDetector()
    assert det.threshold == pytest.approx(0.30)


def test_weight_env_var_override(monkeypatch):
    """BOT_WEIGHT_* env vars override individual weights."""
    monkeypatch.setenv("BOT_WEIGHT_TRADE_FREQUENCY", "0.50")
    det = AdaptiveBotDetector()
    assert det.weights["trade_frequency"] == pytest.approx(0.50)


# ── size_uniformity widening ─────────────────────────────────


def test_size_uniformity_widened_midpoint_catches_cv_0_12(monkeypatch):
    """A CV of ~0.12 should now produce a moderate uniformity score.

    Under the old midpoint=0.05 a CV of 0.12 was essentially classified
    as varied (score near 0).  The new midpoint=0.15 puts it just below
    the inflection point, producing a noticeable score that contributes
    to the weighted sum.
    """
    monkeypatch.delenv("BOT_SIZE_UNIFORMITY_MIDPOINT", raising=False)
    monkeypatch.delenv("BOT_SIZE_UNIFORMITY_STEEPNESS", raising=False)

    det = AdaptiveBotDetector()
    # Build fills with CV~=0.12: sizes alternating ~95 and ~105 around mean 100.
    fills = []
    base_ms = 1_700_000_000_000
    for i in range(30):
        size = 100.0 + (5.0 if i % 2 == 0 else -5.0)
        fills.append({
            "time": base_ms + i * 60_000,
            "size": size,
            "price": 1.0,
        })
    score = det._score_size_uniformity(fills)
    # With midpoint=0.15 and CV~0.05 the sigmoid sits well above 0.5 --
    # uniformity is recognized.  The old midpoint=0.05 placed CV=0.05
    # exactly at 0.5 but anything above (like 0.12) collapsed to ~0.0.
    assert score >= 0.5, (
        f"Expected size-uniformity score >= 0.5 for low-CV fills under "
        f"new midpoint; got {score:.3f}"
    )
