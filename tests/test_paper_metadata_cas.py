"""AUDIT M3 — race-safety tests for ``update_paper_trade_metadata``.

Previously the function did a plain SELECT-then-UPDATE which would lose
writes under concurrent callers (e.g. a funding accrual thread racing
the close handler).  The fix uses a per-trade-id process-local lock
plus a compare-and-swap on the raw metadata text to serialize RMW
across threads and detect cross-process races.

These tests exercise:
  1. Simple merge still works (backward compat).
  2. Many concurrent merges against the SAME trade_id preserve every
     key — no silent drops.
  3. CAS retry fires when a simulated cross-process writer changes
     metadata between our SELECT and UPDATE.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import List

import pytest

import src.data.database as db


def _make_paper_trades_table(tmp_path):
    db_path = tmp_path / "m3.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable WAL so the race-simulation tests can hold a reader open
    # on the main connection while a second connection commits writes
    # — the default rollback journal would deadlock there.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metadata TEXT DEFAULT '{}'
        )
        """
    )
    # Seed one trade row for the tests to mutate.
    conn.execute("INSERT INTO paper_trades (id, metadata) VALUES (1, '{}')")
    conn.commit()
    conn.close()
    return db_path


def _install_fake_connection(monkeypatch, db_path):
    @contextmanager
    def _fake_connection(*, for_read=False):
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _fake_connection)


def _read_metadata(db_path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT metadata FROM paper_trades WHERE id = 1"
    ).fetchone()
    conn.close()
    assert row is not None
    return json.loads(row["metadata"] or "{}")


def _reset_process_locks():
    """Clear the process-local per-trade lock dict between tests so
    tests don't accidentally serialize through a stale lock from an
    earlier run."""
    db._PAPER_TRADE_METADATA_LOCKS.clear()


def test_update_paper_trade_metadata_simple_merge(tmp_path, monkeypatch):
    """Backward compat: a single call merges keys into the existing blob."""
    _reset_process_locks()
    db_path = _make_paper_trades_table(tmp_path)
    _install_fake_connection(monkeypatch, db_path)

    # Seed base metadata first.
    db.update_paper_trade_metadata(1, {"fees_paid": 1.0})
    assert _read_metadata(db_path) == {"fees_paid": 1.0}

    db.update_paper_trade_metadata(1, {"funding_accrued": 0.5})
    assert _read_metadata(db_path) == {"fees_paid": 1.0, "funding_accrued": 0.5}


def test_update_paper_trade_metadata_raises_when_trade_missing(tmp_path, monkeypatch):
    _reset_process_locks()
    db_path = _make_paper_trades_table(tmp_path)
    _install_fake_connection(monkeypatch, db_path)

    with pytest.raises(LookupError):
        db.update_paper_trade_metadata(999, {"foo": "bar"})


def test_concurrent_merges_preserve_every_key(tmp_path, monkeypatch):
    """AUDIT M3: N threads each writing a unique key must all succeed.

    Under the old SELECT-then-UPDATE pattern, concurrent merges would
    cause lost updates — thread B's write clobbers thread A's between
    A's SELECT and A's UPDATE.  The per-trade lock + CAS retry must
    serialize writes so ALL keys land in the final blob.
    """
    _reset_process_locks()
    db_path = _make_paper_trades_table(tmp_path)
    _install_fake_connection(monkeypatch, db_path)

    N = 50
    errors: List[Exception] = []
    start_gate = threading.Event()

    def _worker(i: int):
        start_gate.wait()
        try:
            db.update_paper_trade_metadata(1, {f"key_{i}": i})
        except Exception as exc:  # pragma: no cover - diagnostics
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    # Release them all at once so they stampede the same trade row.
    start_gate.set()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Workers raised: {errors}"
    final = _read_metadata(db_path)
    # All N keys must be present.
    for i in range(N):
        assert f"key_{i}" in final, f"Lost update on key_{i}; final={final}"
        assert final[f"key_{i}"] == i


class _RacingConnectionWrapper:
    """Wraps a sqlite3.Connection and triggers a callback after every
    SELECT metadata statement — lets tests simulate a cross-process
    writer sneaking in between our SELECT and UPDATE."""

    def __init__(self, conn, on_post_select):
        self._conn = conn
        self._on_post_select = on_post_select
        # Propagate the backend attribute our production code checks.
        self.backend = getattr(conn, "backend", "sqlite")

    def execute(self, sql, params=()):
        result = self._conn.execute(sql, params)
        if sql.strip().upper().startswith("SELECT METADATA"):
            self._on_post_select()
        return result

    def commit(self):
        return self._conn.commit()

    def close(self):
        return self._conn.close()

    def __getattr__(self, item):
        return getattr(self._conn, item)


def test_update_paper_trade_metadata_retries_on_cas_miss(tmp_path, monkeypatch):
    """If the first UPDATE misses (simulated cross-process writer
    changed metadata between our SELECT and UPDATE), the function must
    retry the full read-merge-UPDATE cycle until it succeeds or
    exhausts max_retries."""
    _reset_process_locks()
    db_path = _make_paper_trades_table(tmp_path)
    _install_fake_connection(monkeypatch, db_path)

    # Pre-populate a non-empty base so we can see both retry attempts
    # apply their merges atop the freshest read.
    db.update_paper_trade_metadata(1, {"base": "v1"})

    attempt_count = {"n": 0}

    def _sneak_once():
        if attempt_count["n"] == 0:
            attempt_count["n"] += 1
            sneaker = sqlite3.connect(db_path)
            sneaker.execute(
                "UPDATE paper_trades SET metadata = ? WHERE id = 1",
                (json.dumps({"base": "v1", "sneaker": True}),),
            )
            sneaker.commit()
            sneaker.close()

    @contextmanager
    def _racing_connection(*, for_read=False):
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        wrapper = _RacingConnectionWrapper(conn, on_post_select=_sneak_once)
        try:
            yield wrapper
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _racing_connection)

    db.update_paper_trade_metadata(1, {"ours": "v2"})

    final = _read_metadata(db_path)
    # Both the sneaker's write AND our retry-merged write must survive.
    # The retry re-reads the post-sneaker state, merges "ours" on top,
    # and commits.  Sneaker's "sneaker: True" is preserved; "base: v1"
    # from the original seed is preserved; "ours: v2" from our call
    # is preserved.
    assert final.get("base") == "v1", final
    assert final.get("sneaker") is True, final
    assert final.get("ours") == "v2", final
    # We should have taken at least one retry.
    assert attempt_count["n"] >= 1


def test_update_paper_trade_metadata_gives_up_after_max_retries(tmp_path, monkeypatch):
    """If CAS keeps failing (persistent external writer), the function
    raises a RuntimeError after exhausting retries — it does NOT loop
    forever."""
    _reset_process_locks()
    db_path = _make_paper_trades_table(tmp_path)
    _install_fake_connection(monkeypatch, db_path)

    db.update_paper_trade_metadata(1, {"base": "v1"})

    mutation_counter = {"n": 0}

    def _sneak_always():
        mutation_counter["n"] += 1
        sneaker = sqlite3.connect(db_path)
        sneaker.execute(
            "UPDATE paper_trades SET metadata = ? WHERE id = 1",
            (json.dumps({"persistent_mutator": mutation_counter["n"]}),),
        )
        sneaker.commit()
        sneaker.close()

    @contextmanager
    def _always_racing_connection(*, for_read=False):
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        wrapper = _RacingConnectionWrapper(conn, on_post_select=_sneak_always)
        try:
            yield wrapper
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _always_racing_connection)

    with pytest.raises(RuntimeError, match="CAS failed"):
        db.update_paper_trade_metadata(1, {"ours": "v2"}, max_retries=3)
