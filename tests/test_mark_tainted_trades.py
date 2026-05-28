"""Tests for scripts/mark_tainted_trades.py (SQLite path).

The Postgres path uses ``psycopg2`` JSONB upserts which we exercise
directly in CI's Postgres job; here we just lock in the SQLite logic
because that's what most operators iterate against locally.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

# Importing the script directly so we don't depend on shell argv.
import importlib.util


@pytest.fixture
def script_module():
    here = Path(__file__).resolve().parent.parent
    path = here / "scripts" / "mark_tainted_trades.py"
    spec = importlib.util.spec_from_file_location("mark_tainted_trades", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sqlite_conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            status TEXT,
            metadata TEXT
        )
        """
    )
    conn.commit()
    return conn


def _insert(conn, trade_id: int, *, status: str = "closed", metadata: dict | None = None):
    conn.execute(
        "INSERT INTO paper_trades (id, status, metadata) VALUES (?, ?, ?)",
        (trade_id, status, json.dumps(metadata or {})),
    )
    conn.commit()


def _meta(conn, trade_id: int) -> dict:
    cur = conn.execute("SELECT metadata FROM paper_trades WHERE id = ?", (trade_id,))
    row = cur.fetchone()
    return json.loads(row[0] or "{}") if row else {}


def test_marks_reconciler_kill_without_mirror_status(script_module, sqlite_conn):
    """The canonical case the bug poisoned: closed via
    live_reconciled_closed AND no live_mirror_status."""
    _insert(sqlite_conn, 1, metadata={"close_reason": "live_reconciled_closed"})

    n_ex, n_up = script_module._run_sqlite(sqlite_conn, dry_run=False)
    assert (n_ex, n_up) == (1, 1)

    m = _meta(sqlite_conn, 1)
    assert m["tainted"] is True
    assert m["taint_reason"] == "reconciler_kill_pre_fix"


def test_skips_mirrored_trade(script_module, sqlite_conn):
    """A reconciler close on a properly-mirrored trade is a legit close
    (live position vanished externally) — NOT tainted."""
    _insert(sqlite_conn, 2, metadata={
        "close_reason": "live_reconciled_closed",
        "live_mirror_status": "success",
    })
    n_ex, n_up = script_module._run_sqlite(sqlite_conn, dry_run=False)
    assert (n_ex, n_up) == (0, 0)
    assert "tainted" not in _meta(sqlite_conn, 2)


def test_skips_non_reconciler_close(script_module, sqlite_conn):
    """Take-profit / stop-loss / time-limit closes are not tainted."""
    _insert(sqlite_conn, 3, metadata={"close_reason": "take_profit"})
    _insert(sqlite_conn, 4, metadata={"close_reason": "time_limit"})
    n_ex, n_up = script_module._run_sqlite(sqlite_conn, dry_run=False)
    assert (n_ex, n_up) == (0, 0)


def test_idempotent_rerun(script_module, sqlite_conn):
    """Running twice marks once."""
    _insert(sqlite_conn, 5, metadata={"close_reason": "live_reconciled_closed"})
    script_module._run_sqlite(sqlite_conn, dry_run=False)
    n_ex, n_up = script_module._run_sqlite(sqlite_conn, dry_run=False)
    assert (n_ex, n_up) == (0, 0)


def test_dry_run_no_writes(script_module, sqlite_conn):
    _insert(sqlite_conn, 6, metadata={"close_reason": "live_reconciled_closed"})
    n_ex, n_up = script_module._run_sqlite(sqlite_conn, dry_run=True)
    assert n_ex == 1
    assert n_up == 0  # dry-run never counts as updated
    assert "tainted" not in _meta(sqlite_conn, 6)


def test_handles_reconciliation_reason_alias(script_module, sqlite_conn):
    """Some legacy rows used `reconciliation_reason` not
    `close_reason`; both must trigger the taint."""
    _insert(sqlite_conn, 7, metadata={"reconciliation_reason": "live_reconciled_closed"})
    n_ex, n_up = script_module._run_sqlite(sqlite_conn, dry_run=False)
    assert (n_ex, n_up) == (1, 1)
