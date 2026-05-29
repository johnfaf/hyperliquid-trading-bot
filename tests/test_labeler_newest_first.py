"""Verify the XGBoost catchup-labeler queries newest predictions first.

Phase 5 root-cause: with ASC ordering, the labeler chewed on the
6-week-old April rows every cycle.  Hyperliquid's 1m candle window
only retains ~30 days, so every fetch returned ``no_data`` and the
model never got real training labels (15,156 unlabeled rows in prod).
DESC pulls predictions whose forward-return candles are still
available so labels accumulate.
"""
from __future__ import annotations

import re

import pytest


@pytest.fixture
def fc(monkeypatch):
    """Build an XGBoostRegimeForecaster without triggering training."""
    monkeypatch.setenv("XGBOOST_MIN_TRAINING_SAMPLES", "999999")
    from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster
    return XGBoostRegimeForecaster(config={})


def _capture_query(fc, monkeypatch):
    """Run label_predictions_with_forward_returns and return the
    SELECT SQL the labeler issued."""
    captured: dict = {}

    class _FakeConn:
        def execute(self, sql, *args):
            captured["sql"] = sql

            class _Res:
                def fetchall(self_):
                    return []
            return _Res()

    def _get_connection(for_read=True):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield _FakeConn()

        return _cm()

    monkeypatch.setattr(
        "src.data.database.get_connection", _get_connection,
    )
    monkeypatch.setattr(
        "src.data.database.get_backend_name", lambda: "postgres",
    )
    fc.label_predictions_with_forward_returns(batch_size=10)
    return captured.get("sql", "")


def _order_in(sql: str) -> str:
    m = re.search(r"ORDER BY timestamp (ASC|DESC)", sql)
    return m.group(1) if m else ""


# ── DESC by default ────────────────────────────────────────────


def test_default_query_is_desc(fc, monkeypatch):
    monkeypatch.delenv("XGBOOST_LABELER_ORDER", raising=False)
    sql = _capture_query(fc, monkeypatch)
    assert _order_in(sql) == "DESC", (
        f"labeler must default to DESC (newest-first); got:\n{sql}"
    )


def test_env_can_force_asc(fc, monkeypatch):
    """Operator can opt into backfill mode (oldest-first) via env."""
    monkeypatch.setenv("XGBOOST_LABELER_ORDER", "ASC")
    sql = _capture_query(fc, monkeypatch)
    assert _order_in(sql) == "ASC"


def test_garbage_env_falls_back_to_desc(fc, monkeypatch):
    monkeypatch.setenv("XGBOOST_LABELER_ORDER", "not-a-valid-order")
    sql = _capture_query(fc, monkeypatch)
    assert _order_in(sql) == "DESC"


def test_env_lowercase_asc_honoured(fc, monkeypatch):
    monkeypatch.setenv("XGBOOST_LABELER_ORDER", "asc")
    sql = _capture_query(fc, monkeypatch)
    assert _order_in(sql) == "ASC"
