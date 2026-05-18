"""Regression tests for the regime-classifier directional bias.

Before the fix, ``_classify_regime`` had a dedicated ``trend_dir > 0``
arm but funnelled every other state into ``else: TRENDING_DOWN``.  A flat
slope, or a downtrend already reversing UP, was therefore labelled
TRENDING_DOWN at 0.7x a high ADX confidence.  Amplified by the 2x
BTC/ETH vote weight in ``get_market_regime`` this poisoned
``overall_regime`` to high-confidence ``trending_down`` and made the
market-side guard veto every long ("BULLISH shown but only SHORT
trades").  The classifier must now treat up/down symmetrically and never
fabricate a high-confidence down trend out of a flat market.
"""
from __future__ import annotations

import pytest

from src.analysis.regime_detector import Regime, RegimeDetector

# adx=40 -> trend_confidence = 0.5 + (40-25)/50 = 0.80
_ADX = 40.0
_FULL = 0.80
_REDUCED = pytest.approx(0.80 * 0.7)


def _clf(trend_dir, momentum):
    d = RegimeDetector.__new__(RegimeDetector)
    # volume_ratio>=0.15 (not low-liq), atr_pct<=0.05 (not volatile), adx>25
    return d._classify_regime(_ADX, 0.02, 1.0, trend_dir, momentum)


def test_uptrend_aligned_full_confidence():
    regime, conf = _clf(0.01, 0.02)
    assert regime is Regime.TRENDING_UP
    assert conf == pytest.approx(_FULL)


def test_downtrend_aligned_full_confidence():
    regime, conf = _clf(-0.01, -0.02)
    assert regime is Regime.TRENDING_DOWN
    assert conf == pytest.approx(_FULL)


def test_uptrend_with_slowing_momentum_is_reduced_not_down():
    regime, conf = _clf(0.01, -0.02)
    assert regime is Regime.TRENDING_UP  # not flipped to DOWN
    assert conf == _REDUCED


def test_downtrend_reversing_up_is_reduced_confidence_symmetric():
    # The exact case that used to hit the buggy ``else``: negative slope
    # but momentum already positive (a bottoming / recovery).
    regime, conf = _clf(-0.01, 0.02)
    assert regime is Regime.TRENDING_DOWN
    # Reduced (0.7x), NOT the full high confidence the old else returned
    # at trend_confidence... it returned 0.7x too, but ONLY because the
    # whole branch was a DOWN sink. The point: it is now symmetric with
    # the uptrend-slowing case, not a catch-all.
    assert conf == _REDUCED


def test_flat_market_is_ranging_not_trending_down():
    """THE bias regression: zero slope + zero momentum at high ADX must
    NOT become a fabricated high-confidence trending_down."""
    regime, conf = _clf(0.0, 0.0)
    assert regime is Regime.RANGING
    assert regime is not Regime.TRENDING_DOWN


def test_flat_slope_positive_momentum_is_up_not_down():
    regime, conf = _clf(0.0, 0.02)
    assert regime is Regime.TRENDING_UP
    assert conf == _REDUCED


def test_flat_slope_negative_momentum_is_down():
    regime, conf = _clf(0.0, -0.02)
    assert regime is Regime.TRENDING_DOWN
    assert conf == _REDUCED


@pytest.mark.parametrize("td,mom", [(0.02, 0.03), (0.02, -0.03), (0.005, 0.0)])
def test_up_down_resolution_is_symmetric(td, mom):
    """classify(+td,+mom) UP-confidence must equal classify(-td,-mom)
    DOWN-confidence — no directional asymmetry remains."""
    up_regime, up_conf = _clf(td, mom)
    dn_regime, dn_conf = _clf(-td, -mom)
    assert up_regime is Regime.TRENDING_UP
    assert dn_regime is Regime.TRENDING_DOWN
    assert up_conf == pytest.approx(dn_conf)
