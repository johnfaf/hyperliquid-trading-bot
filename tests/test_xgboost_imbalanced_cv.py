"""Phase 5: XGBoost training survives imbalanced-class labels.

Production tail showed:

  XGBoost labeler: scanned=200 labeled=198 no_data=2 errors=0
  XGBoost walk-forward validation FAILED: Invalid classes inferred
    from unique values of y. Expected: [0 1], got [1 2]
  XGBoost training complete but CV validation failed --
    NOT saving model

The labeler was finally working (Phase 5 PR #43) but every cycle
produced y with only [1, 2] (neutral + bullish) -- no crash labels
in the recent 30-day window.  XGBoost's walk-forward CV inside the
fold rejected the non-contiguous class set against ``num_class=3``,
the outer try/except caught the exception, and the orchestrator
discarded the model so the bot stayed on synthetic warm-start.

The fix: detect class imbalance up-front and use a single train/
test split instead of TimeSeriesSplit when not all 3 classes
appear.  A real model trained on imbalanced data is strictly
better than the synthetic fallback.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def fc(monkeypatch):
    """Build an XGBoostRegimeForecaster without triggering training,
    then drop the threshold for explicit train() calls in the test."""
    monkeypatch.setenv("XGBOOST_MIN_TRAINING_SAMPLES", "999999")
    from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster
    inst = XGBoostRegimeForecaster(config={})
    inst.min_samples = 50  # lower threshold for the explicit test calls
    return inst


def _fake_training_data(*, n: int, classes: list[int]) -> tuple:
    """Return synthetic (X, y) where y only contains the requested classes."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, 6).astype(np.float32)
    y = rng.choice(classes, size=n).astype(np.int32)
    return X, y


def test_imbalanced_two_class_data_trains_and_saves(fc, monkeypatch):
    """The bot's real failure mode: y has only [1, 2].  After the fix
    a model should be saved (i.e. self.model is not reset to None)."""
    pytest.importorskip("xgboost")

    X, y = _fake_training_data(n=200, classes=[1, 2])
    monkeypatch.setattr(fc, "_get_training_data", lambda: (X, y))
    saved = []
    monkeypatch.setattr(fc, "_save_model", lambda: saved.append(True))

    result = fc.train()

    assert fc.model is not None, "imbalanced training must still save a model"
    assert saved == [True], "_save_model should have been called"
    assert result is not None
    assert result["samples"] == 200


def test_three_class_data_still_uses_walk_forward(fc, monkeypatch, caplog):
    """When all 3 regimes appear, the legacy walk-forward CV path runs."""
    pytest.importorskip("xgboost")
    import logging

    X, y = _fake_training_data(n=400, classes=[0, 1, 2])
    monkeypatch.setattr(fc, "_get_training_data", lambda: (X, y))
    monkeypatch.setattr(fc, "_save_model", lambda: None)

    with caplog.at_level(logging.INFO):
        fc.train()

    assert any(
        "walk-forward accuracy" in r.message for r in caplog.records
    ), "Walk-forward CV path should have run with 3-class data"


def test_imbalanced_single_class_data_skips_save(fc, monkeypatch):
    """Degenerate case: y has only one class.  XGBoost can't learn
    anything meaningful, so we still refuse to save."""
    pytest.importorskip("xgboost")

    X, y = _fake_training_data(n=200, classes=[1])
    monkeypatch.setattr(fc, "_get_training_data", lambda: (X, y))
    saved = []
    monkeypatch.setattr(fc, "_save_model", lambda: saved.append(True))

    fc.train()
    # The simple split path runs but XGBoost still can't fit a 3-class
    # objective on 1 class, so the CV step throws and we refuse to
    # save -- mirror legacy safety for the truly degenerate case.
    assert fc.model is None or not saved


def test_logging_marks_imbalanced_mode(fc, monkeypatch, caplog):
    """Operators need a clear log when we fall back to the simple
    split so they know what training mode the saved model used."""
    pytest.importorskip("xgboost")
    import logging

    X, y = _fake_training_data(n=200, classes=[1, 2])
    monkeypatch.setattr(fc, "_get_training_data", lambda: (X, y))
    monkeypatch.setattr(fc, "_save_model", lambda: None)

    with caplog.at_level(logging.INFO):
        fc.train()

    assert any(
        "only" in r.message and "unique class" in r.message
        for r in caplog.records
    ), "Should log the imbalanced-class fallback"
