"""Consumers of ``predict_regime()`` must honor PR #20's ``degraded`` flag.

Background
----------
PR #20 (forecaster degraded-feature flag) made the XGBoost regime
forecaster:

  * return ``Optional[float]`` from external feature fetchers
    (funding_rate, volatility_5m, basis_spread) so a network /
    parse failure no longer silently returns 0.0
  * set ``degraded=True`` + cap ``confidence`` proportionally when
    any of those features were missing at inference time
  * list the missing features in ``missing_features`` so downstream
    consumers can log what was lost

But none of the consumers of ``predict_regime()`` actually CHECKED
the flag.  Two were silently making decisions on a zero-padded
feature vector:

  1. ``_reconcile_regimes`` in trading_cycle.py would treat a
     degraded "crash" prediction as authoritative if pred_conf had
     somehow stayed above 0.75 (which can happen on a 1-feature
     miss where confidence floor is 0.67×raw).
  2. ``_run_hedger`` in trading_cycle.py passed the prediction
     straight to ``CrossVenueHedger.check_and_hedge`` which uses
     ``regime == "neutral"`` to CLOSE existing crash hedges -- so a
     degraded prediction during an actual crash could close real
     hedges right when the operator needs them most.

This PR adds the same guard pattern that already exists for the
``synthetic_warm_start`` flag in ``_reconcile_regimes``: when
degraded, skip the override / skip the hedger cycle.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.core.cycles import trading_cycle


# ── _reconcile_regimes ──────────────────────────────────────


def test_reconcile_does_not_override_when_degraded(caplog):
    """A degraded crash prediction must not override the detector."""
    import logging

    forecaster = MagicMock()
    forecaster.predict_regime.return_value = {
        "regime": "crash",
        "confidence": 0.80,        # well above the 0.75 override threshold
        "degraded": True,
        "missing_features": ["funding_rate", "volatility_5m"],
        "synthetic_warm_start": False,
    }
    container = MagicMock(predictive_forecaster=forecaster)

    regime_data = {
        "overall_regime": "trending_up",
        "overall_confidence": 0.7,
        "strategy_guidance": {"activate": ["momentum_long"], "pause": []},
    }
    with caplog.at_level(logging.INFO, logger="src.core.cycles.trading_cycle"):
        out = trading_cycle._reconcile_regimes(regime_data, container)

    # The override would have flipped overall_regime to "volatile" with a
    # regime_override key.  Both must NOT be present when degraded.
    assert out.get("overall_regime") == "trending_up", (
        "Degraded forecaster crashed override the detector regime; "
        "should have left it alone"
    )
    assert "regime_override" not in out, (
        "regime_override key was set despite degraded forecaster"
    )
    # And the new flag is propagated for downstream consumers.
    assert out.get("forecaster_degraded") is True
    assert out.get("forecaster_missing_features") == [
        "funding_rate", "volatility_5m",
    ]
    messages = " ".join(rec.message for rec in caplog.records)
    assert "degraded forecaster cannot override" in messages


def test_reconcile_still_overrides_when_not_degraded():
    """A healthy >=0.75 crash prediction still overrides (regression guard)."""
    forecaster = MagicMock()
    forecaster.predict_regime.return_value = {
        "regime": "crash",
        "confidence": 0.80,
        "degraded": False,
        "missing_features": [],
        "synthetic_warm_start": False,
    }
    container = MagicMock(predictive_forecaster=forecaster)

    regime_data = {
        "overall_regime": "trending_up",
        "overall_confidence": 0.7,
        "strategy_guidance": {"activate": ["momentum_long"], "pause": []},
    }
    out = trading_cycle._reconcile_regimes(regime_data, container)

    # Healthy crash override flips the regime to volatile.
    assert out.get("overall_regime") == "volatile"
    assert out.get("regime_override") == "forecaster_crash"
    assert out.get("forecaster_degraded") is False


def test_reconcile_propagates_degraded_flag_when_no_override():
    """Even when there's no disagreement, the degraded flag is exposed."""
    forecaster = MagicMock()
    forecaster.predict_regime.return_value = {
        "regime": "neutral",
        "confidence": 0.30,
        "degraded": True,
        "missing_features": ["basis_spread"],
        "synthetic_warm_start": False,
    }
    container = MagicMock(predictive_forecaster=forecaster)

    regime_data = {
        "overall_regime": "ranging",
        "overall_confidence": 0.6,
        "strategy_guidance": {"activate": ["mean_reversion"], "pause": []},
    }
    out = trading_cycle._reconcile_regimes(regime_data, container)

    assert out.get("forecaster_degraded") is True
    assert out.get("forecaster_missing_features") == ["basis_spread"]


# ── _run_hedger ─────────────────────────────────────────────


def test_hedger_skips_when_forecaster_degraded(caplog):
    """A degraded prediction must NOT reach the hedger -- skip the cycle."""
    import logging

    forecaster = MagicMock()
    forecaster.predict_regime.return_value = {
        "regime": "neutral",        # would close existing hedges if hedger ran
        "confidence": 0.20,
        "degraded": True,
        "missing_features": ["funding_rate"],
    }
    hedger = MagicMock()
    hedger.check_and_hedge = MagicMock()
    container = MagicMock(
        cross_venue_hedger=hedger,
        predictive_forecaster=forecaster,
    )

    regime_data = {"overall_regime": "ranging", "overall_confidence": 0.6}

    with caplog.at_level(logging.WARNING, logger="src.core.cycles.trading_cycle"):
        trading_cycle._run_hedger(container, regime_data)

    # The hedger must NOT have been called when forecaster is degraded.
    hedger.check_and_hedge.assert_not_called()

    messages = " ".join(rec.message for rec in caplog.records)
    assert "Hedger: skipping cycle" in messages, (
        f"expected hedger-skip warning; saw: {messages}"
    )
    assert "degraded" in messages


def test_hedger_runs_normally_when_not_degraded():
    """Healthy prediction -> hedger.check_and_hedge IS called."""
    forecaster = MagicMock()
    forecaster.predict_regime.return_value = {
        "regime": "crash",
        "confidence": 0.85,
        "degraded": False,
    }
    hedger = MagicMock()
    hedger.check_and_hedge.return_value = {
        "action": "hedged",
        "hedges_placed": 2,
        "hedges_closed": 0,
        "coins_affected": ["BTC", "ETH"],
    }
    container = MagicMock(
        cross_venue_hedger=hedger,
        predictive_forecaster=forecaster,
    )

    # Stub get_execution_open_positions so we don't touch DB
    import src.core.live_execution as live_execution
    orig = live_execution.get_execution_open_positions
    live_execution.get_execution_open_positions = MagicMock(return_value=[])
    try:
        regime_data = {"overall_regime": "volatile", "overall_confidence": 0.7}
        trading_cycle._run_hedger(container, regime_data)
        hedger.check_and_hedge.assert_called_once()
    finally:
        live_execution.get_execution_open_positions = orig


def test_hedger_skip_specifically_protects_crash_hedges_from_being_closed():
    """The bug we're guarding against: a degraded 'neutral' regime triggers
    hedger to close active crash hedges.  After this fix, the hedger
    is not called at all when degraded -- existing hedges stay open.
    """
    forecaster = MagicMock()
    # The exact failure mode: forecaster thinks regime is neutral (so its
    # logic would close hedges), but the underlying inputs are missing.
    forecaster.predict_regime.return_value = {
        "regime": "neutral",
        "confidence": 0.50,
        "degraded": True,
        "missing_features": ["funding_rate", "volatility_5m", "basis_spread"],
    }
    hedger = MagicMock()
    # Sanity: if the hedger DID run with this prediction, it would close hedges.
    hedger.check_and_hedge = MagicMock()
    container = MagicMock(
        cross_venue_hedger=hedger,
        predictive_forecaster=forecaster,
    )
    regime_data = {"overall_regime": "ranging", "overall_confidence": 0.6}

    trading_cycle._run_hedger(container, regime_data)

    # The protective behavior: hedger never runs, hedges stay as-is.
    hedger.check_and_hedge.assert_not_called()


def test_hedger_handles_missing_degraded_field_gracefully():
    """Predictions that pre-date PR #20 (no 'degraded' key) are treated as healthy."""
    forecaster = MagicMock()
    forecaster.predict_regime.return_value = {
        "regime": "neutral",
        "confidence": 0.30,
        # No 'degraded' key at all.  Should be treated as not-degraded.
    }
    hedger = MagicMock()
    hedger.check_and_hedge.return_value = {
        "action": "idle",
        "hedges_placed": 0,
        "hedges_closed": 0,
        "coins_affected": [],
    }
    container = MagicMock(
        cross_venue_hedger=hedger,
        predictive_forecaster=forecaster,
    )

    import src.core.live_execution as live_execution
    orig = live_execution.get_execution_open_positions
    live_execution.get_execution_open_positions = MagicMock(return_value=[])
    try:
        regime_data = {"overall_regime": "ranging", "overall_confidence": 0.6}
        trading_cycle._run_hedger(container, regime_data)
        # No 'degraded' key -> treated as healthy -> hedger DOES run.
        hedger.check_and_hedge.assert_called_once()
    finally:
        live_execution.get_execution_open_positions = orig
