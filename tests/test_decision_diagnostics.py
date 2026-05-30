"""Tests for the why-not-trading decision diagnostics."""
from __future__ import annotations

from contextlib import contextmanager

from src.analysis.decision_diagnostics import (
    _normalise_reason,
    summarize_recent_decisions,
)


# ── _normalise_reason ────────────────────────────────────────────


def test_normalise_strips_ev_detail():
    assert _normalise_reason(
        "ev_below_threshold:ev=1.8bps<=thr=20.7bps (cost=13.8bps, p_win=0.50)"
    ) == "ev_below_threshold"


def test_normalise_strips_paused_source_fragment():
    assert _normalise_reason(
        "Source allocator paused strategy:momentum_long (paused)"
    ) == "Source allocator paused"


def test_normalise_plain_reason_kept():
    assert _normalise_reason("Regime pauses momentum_short") == "Regime pauses momentum_short"


def test_normalise_empty_is_unknown():
    assert _normalise_reason("") == "unknown"
    assert _normalise_reason(None) == "unknown"


# ── summarize_recent_decisions ───────────────────────────────────


def _wire_db(monkeypatch, total, executed, reason_rows):
    from src.data import database as db_mod

    class _Cur:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def fetchall(self):
            return self._rows

    class _Conn:
        def execute(self, sql):
            s = sql.lower()
            if "count(*)" in s and "action_taken" in s:
                return _Cur([(executed,)])
            if "count(*)" in s:
                return _Cur([(total,)])
            return _Cur(reason_rows)

    @contextmanager
    def fake_get_connection(*a, **k):
        yield _Conn()

    monkeypatch.setattr(db_mod, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db_mod, "get_connection", fake_get_connection)


def test_summarize_counts_rate_and_reasons(monkeypatch):
    _wire_db(
        monkeypatch,
        total=20,
        executed=3,
        reason_rows=[
            ("ev_below_threshold:ev=1.8",),
            ("ev_below_threshold:ev=0.4",),
            ("Source allocator paused strategy:momentum_long (paused)",),
            ("Recent live loss guard",),
        ],
    )
    out = summarize_recent_decisions(hours=6)
    assert out["total"] == 20
    assert out["executed"] == 3
    assert out["rejected"] == 17
    assert out["execution_rate"] == round(3 / 20, 4)
    reasons = {r["reason"]: r["count"] for r in out["top_reasons"]}
    assert reasons["ev_below_threshold"] == 2          # two variants collapsed
    assert reasons["Source allocator paused"] == 1
    assert reasons["Recent live loss guard"] == 1
    assert "error" not in out


def test_summarize_zero_decisions_is_safe(monkeypatch):
    _wire_db(monkeypatch, total=0, executed=0, reason_rows=[])
    out = summarize_recent_decisions(hours=6)
    assert out["total"] == 0 and out["execution_rate"] == 0.0
    assert out["top_reasons"] == []


def test_summarize_with_real_sqlite_rows(monkeypatch):
    """Regression: a real sqlite3.Row names a COUNT(*) column "COUNT(*)",
    not "count" -- the old r["count"] raised KeyError ('No item with that
    key') in prod.  Positional r[0] must work."""
    import sqlite3
    from src.data import database as db_mod

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE decision_outcomes "
        "(created_at TEXT, action_taken INTEGER, rejection_reason TEXT)"
    )
    conn.executemany(
        "INSERT INTO decision_outcomes (created_at, action_taken, rejection_reason) "
        "VALUES (datetime('now'), ?, ?)",
        [
            (1, None),                                   # executed
            (0, "ev_below_threshold:ev=1.8"),
            (0, "ev_below_threshold:ev=0.4"),
            (0, "Source allocator paused strategy:m (paused)"),
        ],
    )
    conn.commit()

    @contextmanager
    def fake_get_connection(*a, **k):
        yield conn

    monkeypatch.setattr(db_mod, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db_mod, "get_connection", fake_get_connection)

    out = summarize_recent_decisions(hours=6)
    assert "error" not in out
    assert out["total"] == 4 and out["executed"] == 1 and out["rejected"] == 3
    reasons = {r["reason"]: r["count"] for r in out["top_reasons"]}
    assert reasons["ev_below_threshold"] == 2
    assert reasons["Source allocator paused"] == 1


def test_summarize_degrades_on_db_error(monkeypatch):
    from src.data import database as db_mod

    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(db_mod, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db_mod, "get_connection", boom)
    out = summarize_recent_decisions(hours=6)
    assert "error" in out and out["total"] == 0   # never raises
