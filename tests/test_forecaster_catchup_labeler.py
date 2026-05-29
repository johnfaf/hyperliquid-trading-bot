"""Tests for the Phase 2 forecaster fixes.

Covers:
- min_training_samples default lowered 100 -> 30
- XGBOOST_MIN_TRAINING_SAMPLES env override is honored
- Reporting-cycle catchup-labeler fires on the right forecaster shape
- Reporting-cycle catchup-labeler skipped silently when forecaster
  doesn't expose the labeler (e.g. the PredictiveRegimeForecaster path)
- env FORECASTER_REPORTING_LABELER_ENABLED=false opts out
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_pipeline_health_streak():
    """Reset the module-level streak counter so this file's tests
    don't pollute the false-positive stale-warning test in
    tests/test_live_controls.py.  See PR #40 CI investigation."""
    from src.core.cycles import reporting_cycle as rc
    saved_streak = dict(rc._PIPELINE_HEALTH_STREAK)
    saved_prev = dict(rc._PIPELINE_HEALTH_PREV)
    rc._PIPELINE_HEALTH_STREAK.clear()
    rc._PIPELINE_HEALTH_PREV.clear()
    yield
    rc._PIPELINE_HEALTH_STREAK.clear()
    rc._PIPELINE_HEALTH_STREAK.update(saved_streak)
    rc._PIPELINE_HEALTH_PREV.clear()
    rc._PIPELINE_HEALTH_PREV.update(saved_prev)


# ── min_training_samples defaults ───────────────────────────────


def test_default_min_training_samples_is_30(monkeypatch):
    """Lower default unblocks the cold-start synthetic loop."""
    monkeypatch.delenv("XGBOOST_MIN_TRAINING_SAMPLES", raising=False)
    from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster
    fc = XGBoostRegimeForecaster(config={})
    assert fc.min_samples == 30


def test_env_overrides_min_training_samples(monkeypatch):
    monkeypatch.setenv("XGBOOST_MIN_TRAINING_SAMPLES", "75")
    from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster
    fc = XGBoostRegimeForecaster(config={})
    assert fc.min_samples == 75


def test_config_dict_wins_over_env(monkeypatch):
    """Explicit config wins over env so operators can pin a per-instance value."""
    monkeypatch.setenv("XGBOOST_MIN_TRAINING_SAMPLES", "75")
    from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster
    fc = XGBoostRegimeForecaster(config={"min_training_samples": 200})
    assert fc.min_samples == 200


def test_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("XGBOOST_MIN_TRAINING_SAMPLES", "not-a-number")
    from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster
    fc = XGBoostRegimeForecaster(config={})
    assert fc.min_samples == 30


# ── Reporting-cycle catchup-labeler ────────────────────────────


def _make_container(forecaster=None):
    """Build a minimal duck-typed container the reporting cycle reads."""
    return SimpleNamespace(
        predictive_forecaster=forecaster,
        shadow_tracker=None,
        cross_venue_hedger=None,
        reporter=None,
        paper_trader=None,
        scorer=None,
        kelly_sizer=None,
        trade_memory=None,
        calibration=None,
        llm_filter=None,
        signal_processor=None,
        arena_incubator=None,
        decision_engine=None,
        multi_scanner=None,
        agent_scorer=None,
    )


def test_catchup_labeler_invokes_forecaster_method(monkeypatch, caplog):
    """When predictive_forecaster exposes the labeler, it gets called."""
    forecaster = MagicMock()
    forecaster.label_predictions_with_forward_returns.return_value = {
        "scanned": 50, "labeled": 47, "no_data": 2, "errors": 1,
    }
    container = _make_container(forecaster=forecaster)

    # Run only the catchup-labeler block to keep the test focused.
    from src.core.cycles import reporting_cycle as rc

    # Re-implement the same block inline so we don't have to mock the
    # entire run_reporting universe.  Mirrors the wiring at the top of
    # the catchup labeler section.
    import config as _cfg
    monkeypatch.setattr(_cfg, "FORECASTER_REPORTING_LABELER_ENABLED", True, raising=False)
    monkeypatch.setattr(_cfg, "FORECASTER_REPORTING_LABELER_BATCH_SIZE", 200, raising=False)

    # Call run_reporting -- but we need to avoid all the other side
    # effects, so we mock out the bits we don't care about.
    monkeypatch.setattr(rc, "logger", logging.getLogger("test_rc"))
    monkeypatch.setattr(rc.db, "backup_to_json", lambda: None)
    monkeypatch.setattr(rc, "_log_module_stats", lambda c: None)

    with caplog.at_level(logging.INFO):
        rc.run_reporting(container, cycle_count=1)

    forecaster.label_predictions_with_forward_returns.assert_called_once()
    call_kwargs = forecaster.label_predictions_with_forward_returns.call_args.kwargs
    assert call_kwargs["batch_size"] == 200
    assert any("Forecaster catchup-labeler" in r.message for r in caplog.records)


def test_catchup_labeler_skipped_when_method_missing(monkeypatch, caplog):
    """A forecaster without the labeler method (e.g. the rules-based
    fallback) must not crash the cycle."""
    # PredictiveRegimeForecaster does NOT have the labeler method.
    forecaster = SimpleNamespace()
    assert not hasattr(forecaster, "label_predictions_with_forward_returns")
    container = _make_container(forecaster=forecaster)

    from src.core.cycles import reporting_cycle as rc
    monkeypatch.setattr(rc, "_log_module_stats", lambda c: None)
    monkeypatch.setattr(rc.db, "backup_to_json", lambda: None)
    # Must not raise.
    rc.run_reporting(container, cycle_count=1)


def test_catchup_labeler_opt_out_via_env(monkeypatch):
    """FORECASTER_REPORTING_LABELER_ENABLED=false disables the call."""
    forecaster = MagicMock()
    forecaster.label_predictions_with_forward_returns.return_value = {
        "scanned": 0, "labeled": 0, "no_data": 0, "errors": 0,
    }
    container = _make_container(forecaster=forecaster)

    from src.core.cycles import reporting_cycle as rc
    import config as _cfg
    monkeypatch.setattr(_cfg, "FORECASTER_REPORTING_LABELER_ENABLED", False, raising=False)
    monkeypatch.setattr(rc, "_log_module_stats", lambda c: None)
    monkeypatch.setattr(rc.db, "backup_to_json", lambda: None)

    rc.run_reporting(container, cycle_count=1)

    forecaster.label_predictions_with_forward_returns.assert_not_called()


def test_catchup_labeler_handles_no_forecaster(monkeypatch):
    """No forecaster at all -- cycle must still complete."""
    container = _make_container(forecaster=None)
    from src.core.cycles import reporting_cycle as rc
    monkeypatch.setattr(rc, "_log_module_stats", lambda c: None)
    monkeypatch.setattr(rc.db, "backup_to_json", lambda: None)
    rc.run_reporting(container, cycle_count=1)


def test_catchup_labeler_handles_labeler_exception(monkeypatch, caplog):
    """Exception from the labeler must NOT propagate -- fail-open."""
    forecaster = MagicMock()
    forecaster.label_predictions_with_forward_returns.side_effect = RuntimeError("boom")
    container = _make_container(forecaster=forecaster)

    from src.core.cycles import reporting_cycle as rc
    monkeypatch.setattr(rc, "_log_module_stats", lambda c: None)
    monkeypatch.setattr(rc.db, "backup_to_json", lambda: None)
    # Must not raise.
    rc.run_reporting(container, cycle_count=1)
