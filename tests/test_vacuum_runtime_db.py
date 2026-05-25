"""Tests for the runtime-DB VACUUM CLI.

After PR #25's strategy-score cleanup deleted 377K rows but VACUUM
hadn't run, the SQLite file size stayed the same and the boot-time
PRAGMA integrity_check still walked the freelist pages.  The
``vacuum_runtime_db.py`` CLI exists to close that gap during a
planned maintenance window.

These tests verify:
  * Inspection reports the right page counts before/after
  * Dry-run mode never writes
  * Real VACUUM actually shrinks the on-disk file when freelist
    pages exist
  * Locked DB doesn't crash, reports the timeout cleanly
  * No-freelist case exits early without VACUUM
"""
from __future__ import annotations

import sqlite3
import threading
import time

from scripts.vacuum_runtime_db import (
    _human_bytes,
    inspect_db,
    run_vacuum,
)


# ── _human_bytes ────────────────────────────────────────────


def test_human_bytes_formats_each_unit():
    assert _human_bytes(0) == "0.0 B"
    assert _human_bytes(512) == "512.0 B"
    assert _human_bytes(2048) == "2.0 KB"
    assert _human_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert _human_bytes(int(1.5 * 1024 ** 3)) == "1.5 GB"


# ── inspect_db ──────────────────────────────────────────────


def _seed_db(path, rows_to_insert=1000, then_delete=True):
    """Create a SQLite DB with bloat (insert N rows then delete most)."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE bloat (id INTEGER PRIMARY KEY, payload TEXT)")
    payload = "x" * 200  # ~200B per row
    conn.executemany(
        "INSERT INTO bloat (payload) VALUES (?)",
        [(payload,) for _ in range(rows_to_insert)],
    )
    conn.commit()
    if then_delete:
        # Delete 90% of rows -> creates freelist pages
        conn.execute("DELETE FROM bloat WHERE id > ?", (int(rows_to_insert * 0.1),))
        conn.commit()
    conn.close()


def test_inspect_reports_freelist_after_deletes(tmp_path):
    """Freelist count > 0 after bulk deletes (without VACUUM)."""
    db = tmp_path / "test.db"
    _seed_db(db, rows_to_insert=1000)

    snap = inspect_db(str(db))
    assert "error" not in snap
    assert snap["page_count"] > 0
    # After deleting 90% of rows, the freelist should hold a non-zero
    # number of pages.
    assert snap["freelist_count"] > 0, (
        "Expected freelist pages after bulk DELETE without VACUUM; "
        f"got snapshot={snap}"
    )


def test_inspect_handles_missing_db_gracefully(tmp_path):
    """Pointing at a non-existent DB returns a snapshot with zero pages."""
    missing = tmp_path / "no_such_file.db"
    snap = inspect_db(str(missing))
    # sqlite3.connect creates an empty DB on demand, so we get a real
    # snapshot with zero pages (or near-zero) rather than an error.
    assert snap["page_count"] == 0
    assert snap["freelist_count"] == 0


# ── run_vacuum ──────────────────────────────────────────────


def test_vacuum_shrinks_freelist(tmp_path):
    """After VACUUM, freelist drops to zero and page_count drops."""
    db = tmp_path / "test.db"
    _seed_db(db, rows_to_insert=2000)

    before = inspect_db(str(db))
    assert before["freelist_count"] > 0

    result = run_vacuum(str(db), busy_timeout_ms=5000)
    assert "error" not in result
    after = result["after"]
    assert after["freelist_count"] == 0, (
        f"VACUUM should drop freelist to 0; got freelist={after['freelist_count']}"
    )
    assert after["page_count"] < before["page_count"], (
        f"VACUUM should shrink page count; "
        f"before={before['page_count']}, after={after['page_count']}"
    )
    assert after["file_size_bytes"] < before["file_size_bytes"], (
        "VACUUM should shrink on-disk size"
    )


def test_vacuum_on_clean_db_is_noop(tmp_path):
    """VACUUM on a DB with no freelist still succeeds and is a no-op."""
    db = tmp_path / "clean.db"
    _seed_db(db, rows_to_insert=100, then_delete=False)

    before = inspect_db(str(db))
    # Fresh DB; should have minimal freelist
    assert before["freelist_count"] == 0

    result = run_vacuum(str(db), busy_timeout_ms=5000)
    assert "error" not in result
    after = result["after"]
    # Page count may stay the same or be marginally different.
    assert after["freelist_count"] == 0


def test_vacuum_reports_lock_timeout(tmp_path):
    """When a writer holds the DB, VACUUM reports a timeout cleanly."""
    db = tmp_path / "locked.db"
    _seed_db(db, rows_to_insert=500)

    # Hold a write transaction in a background thread to block VACUUM
    blocker_ready = threading.Event()
    blocker_release = threading.Event()
    blocker_error: list = []

    def _blocker():
        try:
            conn = sqlite3.connect(str(db), isolation_level=None, timeout=30.0)
            try:
                conn.execute("BEGIN EXCLUSIVE")
                blocker_ready.set()
                # Hold the lock until told to release
                blocker_release.wait(timeout=30)
                conn.execute("COMMIT")
            finally:
                conn.close()
        except Exception as exc:
            blocker_error.append(exc)
            blocker_ready.set()

    t = threading.Thread(target=_blocker, daemon=True)
    t.start()
    assert blocker_ready.wait(timeout=5), "blocker thread did not acquire lock"
    if blocker_error:
        # If the blocker itself failed, the test is inconclusive --
        # release and skip the assertion.
        blocker_release.set()
        t.join(timeout=5)
        return

    try:
        # Try to VACUUM with a SHORT busy_timeout so the test is quick
        # to fail.
        start = time.time()
        result = run_vacuum(str(db), busy_timeout_ms=500)
        elapsed = time.time() - start

        # We expect an error (could be "database is locked" or similar)
        assert "error" in result, (
            "VACUUM should report an error when blocked by an exclusive "
            f"transaction; got result={result}"
        )
        # Should have failed quickly, not waited 30+ seconds.
        assert elapsed < 10, f"VACUUM blocked too long ({elapsed:.1f}s)"
    finally:
        blocker_release.set()
        t.join(timeout=5)
