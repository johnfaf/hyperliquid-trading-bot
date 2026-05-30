"""Tests for the multi-coin regime warm-up in the reporting cycle.

Prod regime_history was BTC-only (~9 rows/day) because the trading cycle's
periodic predict_regime() calls use BTC as a market-regime proxy.  The
warm-up periodically stores predictions for a few extra coins so the
labeler/trainer gets coin diversity.  It is decision-neutral (trades still
use the per-signal predict_regime path) and fail-open.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.cycles import reporting_cycle as rc


@pytest.fixture(autouse=True)
def _warmup_env(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_ENABLED", True, raising=False)
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_EVERY_N_CYCLES", 5, raising=False)
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_COINS", "", raising=False)
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_MAX_COINS", 5, raising=False)
    monkeypatch.setattr(_cfg, "ARENA_COIN_UNIVERSE", ["BTC", "ETH", "SOL"], raising=False)
    yield


def _container(forecaster):
    return SimpleNamespace(predictive_forecaster=forecaster)


def _forecaster():
    fc = MagicMock()
    fc.predict_regime.return_value = {"regime": "neutral", "confidence": 0.5}
    return fc


# ── _warmup_regime_coins ────────────────────────────────────────


def test_coins_default_to_arena_universe():
    assert rc._warmup_regime_coins() == ["BTC", "ETH", "SOL"]


def test_coins_explicit_env_list_wins(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_COINS", "btc, eth , arb", raising=False)
    assert rc._warmup_regime_coins() == ["BTC", "ETH", "ARB"]


def test_coins_deduped_and_capped(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_COINS", "BTC,BTC,ETH,SOL,ARB,OP", raising=False)
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_MAX_COINS", 3, raising=False)
    out = rc._warmup_regime_coins()
    assert out == ["BTC", "ETH", "SOL"]  # dup BTC dropped, capped at 3


# ── _warmup_regime_predictions: cadence ─────────────────────────


def test_warmup_runs_on_nth_cycle():
    fc = _forecaster()
    n = rc._warmup_regime_predictions(_container(fc), cycle_count=10)  # 10 % 5 == 0
    assert n == 3
    called = {c.args[0] for c in fc.predict_regime.call_args_list}
    assert called == {"BTC", "ETH", "SOL"}


def test_warmup_skipped_off_cadence():
    fc = _forecaster()
    assert rc._warmup_regime_predictions(_container(fc), cycle_count=7) == 0  # 7 % 5 != 0
    fc.predict_regime.assert_not_called()


def test_warmup_every_n_one_runs_every_cycle(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_EVERY_N_CYCLES", 1, raising=False)
    fc = _forecaster()
    assert rc._warmup_regime_predictions(_container(fc), cycle_count=3) == 3


# ── _warmup_regime_predictions: gating ──────────────────────────


def test_warmup_disabled_via_env(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "XGBOOST_REGIME_WARMUP_ENABLED", False, raising=False)
    fc = _forecaster()
    assert rc._warmup_regime_predictions(_container(fc), cycle_count=10) == 0
    fc.predict_regime.assert_not_called()


def test_warmup_no_forecaster():
    assert rc._warmup_regime_predictions(_container(None), cycle_count=10) == 0


def test_warmup_forecaster_without_predict_method():
    # A forecaster missing predict_regime must not crash.
    fc = SimpleNamespace()
    assert rc._warmup_regime_predictions(_container(fc), cycle_count=10) == 0


# ── fail-open ───────────────────────────────────────────────────


def test_warmup_fail_open_counts_only_successes():
    fc = MagicMock()
    # BTC raises, ETH/SOL succeed -> no exception propagates, count = 2.
    fc.predict_regime.side_effect = [
        RuntimeError("boom"),
        {"regime": "bullish", "confidence": 0.6},
        {"regime": "crash", "confidence": 0.6},
    ]
    assert rc._warmup_regime_predictions(_container(fc), cycle_count=5) == 2
    assert fc.predict_regime.call_count == 3
