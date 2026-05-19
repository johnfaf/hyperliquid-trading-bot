"""Drift-aware promotion gate (wires FeatureDriftMonitor into promotion_gate).

Before this wiring, FeatureDriftMonitor.compare() set
``DriftReport.blocks_promotion=True`` when feature drift exceeded
the configured z-score threshold, persisted the report to
``learning_drift_reports``, and... nothing consumed the flag. The
protection was dead.

These tests cover the new ``_drift_promotion_ok()`` helper + its
wiring into ``is_live_promotable()`` via the
``PROMOTION_REQUIRE_DRIFT_OK`` config flag.

DB-free: the helper queries the DB, so we patch ``db.get_connection``
to return a stub cursor with canned ``fetchone()`` returns.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import config
from src.learning.promotion_gate import _drift_promotion_ok, is_live_promotable


# ── Fake DB plumbing ─────────────────────────────────────────────


@contextmanager
def _stub_conn(row):
    """Yield a connection-like object whose .execute().fetchone() == row."""
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = row
    conn.execute.return_value = cur
    yield conn


def _row(created_at, blocks, status="warn", summary=None):
    """Mapping-style row (matches PG psycopg dict cursor)."""
    return {
        "created_at": created_at,
        "blocks_promotion": blocks,
        "status": status,
        "summary": summary or {},
        "current_dataset_id": "ds_current",
        "baseline_dataset_id": "ds_baseline",
    }


# ── _drift_promotion_ok ──────────────────────────────────────────


def test_no_drift_reports_returns_true():
    """No rows in learning_drift_reports → fail OPEN."""
    with patch(
        "src.learning.promotion_gate.db.get_connection",
        return_value=_stub_conn(None),
    ):
        ok, reason = _drift_promotion_ok(max_age_hours=24.0)
    assert ok is True
    assert "no_reports" in reason


def test_recent_non_blocking_report_returns_true():
    """A 'warn' or 'pass' report with blocks_promotion=False is fine."""
    row = _row(datetime.now(timezone.utc).isoformat(), blocks=False, status="warn")
    with patch(
        "src.learning.promotion_gate.db.get_connection",
        return_value=_stub_conn(row),
    ):
        ok, reason = _drift_promotion_ok(max_age_hours=24.0)
    assert ok is True
    assert "ok_drift" in reason


def test_recent_blocking_report_returns_false():
    """A recent block with blocks_promotion=True downgrades the promotion."""
    row = _row(
        datetime.now(timezone.utc).isoformat(),
        blocks=True, status="block",
    )
    with patch(
        "src.learning.promotion_gate.db.get_connection",
        return_value=_stub_conn(row),
    ):
        ok, reason = _drift_promotion_ok(max_age_hours=24.0)
    assert ok is False
    assert "drift_blocked" in reason


def test_stale_blocking_report_returns_true():
    """A 48-hour-old block is outside the 24h window → no longer blocks."""
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    row = _row(old, blocks=True, status="block")
    with patch(
        "src.learning.promotion_gate.db.get_connection",
        return_value=_stub_conn(row),
    ):
        ok, reason = _drift_promotion_ok(max_age_hours=24.0)
    assert ok is True
    assert "stale_block" in reason


def test_db_failure_returns_true_fail_open():
    """If the DB query raises, the drift check is skipped (fail OPEN).
    A broken drift query MUST NOT block all promotions."""
    @contextmanager
    def _boom():
        raise RuntimeError("simulated DB outage")
        yield None  # pragma: no cover

    with patch(
        "src.learning.promotion_gate.db.get_connection",
        return_value=_boom(),
    ):
        ok, reason = _drift_promotion_ok(max_age_hours=24.0)
    assert ok is True
    assert "drift_check_skipped" in reason


def test_tuple_row_shape_also_parsed():
    """psycopg2 by default returns tuples, not dicts. Verify parsing
    doesn't break."""
    now = datetime.now(timezone.utc).isoformat()
    tuple_row = (now, True, "block", {}, "ds_curr", "ds_base")
    with patch(
        "src.learning.promotion_gate.db.get_connection",
        return_value=_stub_conn(tuple_row),
    ):
        ok, _ = _drift_promotion_ok(max_age_hours=24.0)
    assert ok is False


# ── Wiring: is_live_promotable() with PROMOTION_REQUIRE_DRIFT_OK ────


@pytest.fixture(autouse=True)
def _flags_off_by_default(monkeypatch):
    """Default-OFF posture for all gating flags so wiring tests opt in."""
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DRIFT_OK", False, raising=False)
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DSR", False, raising=False)
    monkeypatch.setattr(config, "LIVE_PROMOTION_GATE_ENABLED", True, raising=False)


def _strategy_row(passing: bool):
    """Build a strategy row that makes _strategy_promotion_ok return True/False."""
    return {
        "id": 42,
        "trade_count": 100,
        "win_rate": 0.60 if passing else 0.20,
        "current_score": 0.50 if passing else 0.05,
        "metadata": {},
    }


@pytest.fixture(autouse=True)
def _stub_quarantine(monkeypatch):
    """Bypass the quarantine check -- it does its own DB reads we don't
    want to mock for every test."""
    monkeypatch.setattr(
        "src.learning.promotion_gate.db.strategy_quarantine_reason",
        lambda _strategy: None,
        raising=False,
    )


def test_drift_gate_off_does_not_call_drift_check(monkeypatch):
    """Flag off: behavior is byte-identical to pre-wiring."""
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DRIFT_OK", False, raising=False)
    with patch("src.learning.promotion_gate.db.get_strategy",
               return_value=_strategy_row(passing=True)):
        with patch(
            "src.learning.promotion_gate._drift_promotion_ok"
        ) as mock_drift:
            mock_drift.return_value = (False, "should-not-be-called")
            ok, _ = is_live_promotable({
                "strategy_id": 42, "metadata": {},
            })
    mock_drift.assert_not_called()
    assert ok is True


def test_drift_gate_on_passes_when_no_recent_block(monkeypatch):
    """Flag on + no recent drift block → promotion still approved."""
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DRIFT_OK", True, raising=False)
    with patch("src.learning.promotion_gate.db.get_strategy",
               return_value=_strategy_row(passing=True)):
        with patch(
            "src.learning.promotion_gate._drift_promotion_ok",
            return_value=(True, "ok_drift:no_reports"),
        ) as mock_drift:
            ok, reason = is_live_promotable({
                "strategy_id": 42, "metadata": {},
            })
    mock_drift.assert_called_once()
    assert ok is True
    assert reason == "ok"


def test_drift_gate_on_downgrades_when_recent_block(monkeypatch):
    """Flag on + recent block → promotion DOWNgraded to False."""
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DRIFT_OK", True, raising=False)
    with patch("src.learning.promotion_gate.db.get_strategy",
               return_value=_strategy_row(passing=True)):
        with patch(
            "src.learning.promotion_gate._drift_promotion_ok",
            return_value=(False, "drift_blocked:status=block,age=2.0h"),
        ) as mock_drift:
            ok, reason = is_live_promotable({
                "strategy_id": 42, "metadata": {},
            })
    mock_drift.assert_called_once()
    assert ok is False
    assert "drift_blocked" in reason


def test_drift_gate_never_upgrades_a_rejection(monkeypatch):
    """Critical invariant: drift gate is downgrade-only. If the base
    strategy gate rejects, the drift gate must NOT be consulted at all
    -- ok=False bypasses the if-ok block."""
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DRIFT_OK", True, raising=False)
    with patch("src.learning.promotion_gate.db.get_strategy",
               return_value=_strategy_row(passing=False)):
        with patch(
            "src.learning.promotion_gate._drift_promotion_ok"
        ) as mock_drift:
            ok, reason = is_live_promotable({
                "strategy_id": 42, "metadata": {},
            })
    # The base gate said False (win_rate too low) → drift gate not reached
    mock_drift.assert_not_called()
    assert ok is False
    assert "win_rate" in reason or "score" in reason


def test_gate_disabled_skips_drift_check(monkeypatch):
    """When LIVE_PROMOTION_GATE_ENABLED=false, no inner gating runs."""
    monkeypatch.setattr(config, "LIVE_PROMOTION_GATE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DRIFT_OK", True, raising=False)
    with patch(
        "src.learning.promotion_gate._drift_promotion_ok"
    ) as mock_drift:
        ok, reason = is_live_promotable({"strategy_id": 42, "metadata": {}})
    mock_drift.assert_not_called()
    assert ok is True
    assert reason == "gate_disabled"
