"""Regression test for the -410k R-multiple bug.

Background
----------
Three call sites in `src/trading/paper_trader.py` previously did::

    stop_roe_pct = max(float(risk_policy.get("stop_roe_pct", 0.0) or 0.0), 1e-9)
    r_mult = current_roe / stop_roe_pct

When the trade had no recorded stop (`stop_roe_pct=0`), the `1e-9` floor
kicked in and a -0.04% adverse move became `-0.0004 / 1e-9 = -400,000 R`.
That number got stamped into `metadata.min_r_multiple` for every affected
trade, poisoning:

* dashboards (impossible R values),
* the A1 retro-sweep (read these as the historical worst-MAE),
* loss attribution (also reads R-multiple),
* agent_scorer / arena scoring.

The DB-level scan found `avg_min_r_multiple = -410003.08` across 166 trades
with path data.

The fix is the new `PaperTrader._safe_r_multiple(roe_pct, stop_roe_pct)`
helper -- returns ``None`` when the stop is missing or implausibly tight,
rather than computing garbage.
"""
from __future__ import annotations

import pytest

from src.trading.paper_trader import PaperTrader


# ── Direct helper contract ──────────────────────────────────────


def test_safe_r_multiple_returns_none_for_zero_stop():
    """The original bug: stop_roe_pct=0 must not produce a giant number."""
    assert PaperTrader._safe_r_multiple(-0.0004, 0.0) is None


def test_safe_r_multiple_returns_none_for_legacy_1e_9_floor():
    """The historical 1e-9 floor was the trigger for -410k. The helper
    must reject this value too (it's well below MIN_VALID_STOP_ROE)."""
    assert PaperTrader._safe_r_multiple(-0.0004, 1e-9) is None


def test_safe_r_multiple_returns_none_for_negative_or_missing_stop():
    assert PaperTrader._safe_r_multiple(-0.01, -0.05) is None
    assert PaperTrader._safe_r_multiple(0.0, None) is None


def test_safe_r_multiple_computes_for_valid_stop():
    """A real stop_roe_pct=4% (0.04) with a -2% adverse move = -0.5 R."""
    r = PaperTrader._safe_r_multiple(-0.02, 0.04)
    assert r == pytest.approx(-0.5)


def test_safe_r_multiple_positive_excursion():
    """+8% ROE excursion against a 4% stop = +2 R."""
    r = PaperTrader._safe_r_multiple(0.08, 0.04)
    assert r == pytest.approx(2.0)


def test_safe_r_multiple_min_valid_threshold_is_1bp():
    """1 bp ROE (0.0001) is the floor. Stops tighter than this are
    treated as missing, not used for R-multiple math."""
    assert PaperTrader._safe_r_multiple(0.001, 5e-5) is None
    # Just above the threshold passes
    assert PaperTrader._safe_r_multiple(0.001, 1.5e-4) == pytest.approx(0.001 / 1.5e-4)


def test_safe_r_multiple_garbage_input_returns_none():
    """The helper must never raise on weird input."""
    assert PaperTrader._safe_r_multiple("not-a-number", 0.04) is None  # type: ignore[arg-type]
    assert PaperTrader._safe_r_multiple(0.01, "bad") is None  # type: ignore[arg-type]


# ── End-to-end: the regression that bit ─────────────────────────


def test_historical_bug_scenario_no_longer_explodes():
    """The exact DB-level scenario: missing stop, tiny adverse move,
    -410k R was written. The new helper returns None."""
    # Synthesise the scenario: trade with no recorded stop, exit on
    # a tiny adverse 4 bps ROE move.
    exit_roe_pct = -0.0004
    stop_roe_pct = 0.0  # missing -> historically floored to 1e-9
    r = PaperTrader._safe_r_multiple(exit_roe_pct, stop_roe_pct)
    # Pre-fix this would have been ~-400_000. Now it's None.
    assert r is None
    # And NEVER something absurd like -410_000
    assert r != pytest.approx(-400_000, abs=10_000)
