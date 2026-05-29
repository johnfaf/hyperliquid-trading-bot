"""Tests for volatility-relative forward-return labeling.

Background (May 2026 prod audit): the labeler's fixed +/-1.5%/60min bar
is ~2-3 sigma for BTC, so 43/43 observed forward windows were labeled
"neutral".  Single-class data -> train() hits its degenerate-model guard
-> the XGBoost regime model never trains and silently serves the
weighted-signal fallback (signal=0.0).  Volatility-relative labeling
classifies crash/bullish by how large a move is *relative to the coin's
own realized vol* so the label set becomes a trainable 3-class mix.

These exercise the two pure helpers that carry the logic:
  - _forward_sigma(candles, forward_minutes)
  - _classify_forward_return(ret, sigma_fwd, ...)
plus the config defaults.
"""
from __future__ import annotations

import math

import pytest

from src.signals.xgboost_regime_forecaster import (
    REGIME_LABELS,
    XGBoostRegimeForecaster,
)

CRASH = REGIME_LABELS["crash"]      # 0
NEUTRAL = REGIME_LABELS["neutral"]  # 1
BULLISH = REGIME_LABELS["bullish"]  # 2

_clf = XGBoostRegimeForecaster._classify_forward_return
_sigma = XGBoostRegimeForecaster._forward_sigma


def _vr(ret, sigma_fwd, *, vol_k=1.0, min_abs_move=0.001):
    """Vol-relative classify with absolute bars set wide so they can't fire."""
    return _clf(
        ret,
        sigma_fwd,
        vol_relative=True,
        vol_k=vol_k,
        crash_pct=-0.015,
        bullish_pct=0.015,
        min_abs_move=min_abs_move,
    )


# ── _classify_forward_return: volatility-relative ───────────────


def test_vol_relative_labels_large_negative_as_crash():
    # ret = -1.5 sigma, well past the 0.1% floor -> crash.
    assert _vr(-0.006, 0.004) == CRASH


def test_vol_relative_labels_large_positive_as_bullish():
    assert _vr(+0.006, 0.004) == BULLISH


def test_vol_relative_small_move_is_neutral():
    # ret = +0.5 sigma -> within the band -> neutral.
    assert _vr(+0.002, 0.004) == NEUTRAL


def test_vol_relative_is_the_regression_fix():
    """The exact prod failure: a move that the OLD fixed +/-1.5% bar
    called neutral is now correctly directional when it is large
    relative to the coin's (low) realized vol."""
    ret = -0.006        # -0.6%: under fixed -1.5% bar -> NEUTRAL (the bug)
    sigma_fwd = 0.004   # 0.4% forward sigma -> z = -1.5 -> a real move
    # Fixed-bar mode still calls it neutral...
    assert _clf(
        ret, sigma_fwd, vol_relative=False, vol_k=1.0,
        crash_pct=-0.015, bullish_pct=0.015, min_abs_move=0.001,
    ) == NEUTRAL
    # ...but vol-relative mode labels it crash, giving train() a 2nd class.
    assert _vr(ret, sigma_fwd) == CRASH


def test_vol_relative_floor_blocks_subfee_noise():
    """A move past 1 sigma but below the absolute floor stays neutral so
    dead-flat, low-vol windows don't manufacture phantom regimes."""
    # sigma tiny (0.02%); ret = +2 sigma = 0.04% but < 0.1% floor -> neutral.
    assert _vr(+0.0004, 0.0002, min_abs_move=0.001) == NEUTRAL


def test_vol_k_scales_sensitivity():
    ret, sigma_fwd = -0.006, 0.004  # z = -1.5
    assert _vr(ret, sigma_fwd, vol_k=1.0) == CRASH      # 1.5 >= 1.0
    assert _vr(ret, sigma_fwd, vol_k=2.0) == NEUTRAL    # 1.5 < 2.0


def test_vol_relative_boundary_is_inclusive():
    # ret exactly == vol_k * sigma_fwd (and above floor) counts as directional.
    assert _vr(+0.004, 0.004, vol_k=1.0) == BULLISH


# ── _classify_forward_return: absolute fallback ─────────────────


def test_zero_sigma_falls_back_to_absolute_bars():
    # sigma_fwd == 0 (can't estimate) -> fixed crash_pct/bullish_pct apply.
    assert _clf(
        -0.02, 0.0, vol_relative=True, vol_k=1.0,
        crash_pct=-0.015, bullish_pct=0.015, min_abs_move=0.001,
    ) == CRASH
    assert _clf(
        -0.006, 0.0, vol_relative=True, vol_k=1.0,
        crash_pct=-0.015, bullish_pct=0.015, min_abs_move=0.001,
    ) == NEUTRAL  # -0.6% doesn't clear the -1.5% bar


def test_vol_relative_off_uses_absolute_even_with_sigma():
    # Operator turned vol-relative OFF: legacy fixed bars regardless of sigma.
    assert _clf(
        -0.006, 0.004, vol_relative=False, vol_k=1.0,
        crash_pct=-0.015, bullish_pct=0.015, min_abs_move=0.001,
    ) == NEUTRAL
    assert _clf(
        +0.02, 0.004, vol_relative=False, vol_k=1.0,
        crash_pct=-0.015, bullish_pct=0.015, min_abs_move=0.001,
    ) == BULLISH


# ── _forward_sigma ──────────────────────────────────────────────


def _candles(closes, start_ms=0, step_ms=60_000):
    return [(start_ms + i * step_ms, c) for i, c in enumerate(closes)]


def test_forward_sigma_flat_series_is_zero():
    assert _sigma(_candles([100.0] * 30), 60) == 0.0


def test_forward_sigma_too_few_candles_is_zero():
    assert _sigma(_candles([100.0, 101.0, 102.0]), 60) == 0.0
    assert _sigma([], 60) == 0.0


def test_forward_sigma_positive_for_volatile_series():
    closes = [100.0, 101.0, 99.5, 102.0, 100.5, 98.0, 101.5,
              103.0, 99.0, 100.0, 102.5, 97.5]
    s = _sigma(closes if False else _candles(closes), 60)
    assert s > 0.0
    assert math.isfinite(s)


def test_forward_sigma_scales_with_sqrt_horizon():
    closes = [100.0, 101.0, 99.5, 102.0, 100.5, 98.0, 101.5,
              103.0, 99.0, 100.0, 102.5, 97.5]
    c = _candles(closes)
    s60 = _sigma(c, 60)
    s15 = _sigma(c, 15)
    # 60min sigma should be ~2x the 15min sigma (sqrt(60)/sqrt(15) = 2).
    assert s60 == pytest.approx(s15 * 2.0, rel=1e-6)


def test_forward_sigma_ignores_nonpositive_closes():
    # Zero/negative closes are dropped; remaining count < 10 -> 0.0.
    closes = [0.0, -1.0] + [100.0, 101.0, 99.0]
    assert _sigma(_candles(closes), 60) == 0.0


# ── config defaults ─────────────────────────────────────────────


def test_config_defaults_enable_vol_relative(monkeypatch):
    import importlib

    for k in (
        "XGBOOST_LABELER_VOL_RELATIVE",
        "XGBOOST_LABELER_VOL_K",
        "XGBOOST_LABELER_MIN_ABS_MOVE",
    ):
        monkeypatch.delenv(k, raising=False)
    import config as _cfg
    importlib.reload(_cfg)
    try:
        assert _cfg.XGBOOST_LABELER_VOL_RELATIVE is True
        assert _cfg.XGBOOST_LABELER_VOL_K == 1.0
        assert _cfg.XGBOOST_LABELER_MIN_ABS_MOVE == pytest.approx(0.001)
    finally:
        importlib.reload(_cfg)
