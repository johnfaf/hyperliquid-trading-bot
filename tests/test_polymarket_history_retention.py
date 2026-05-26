"""Tests for polymarket history retention pruning.

Production state on 2026-05-25:
  /data/bot.db ~15.4 GB, dominated by
    polymarket_price_points     2,673,890 rows
    polymarket_market_snapshots 1,736,353 rows

The bot only reads recent polymarket data for signal generation
(last few hours -> last 24-48h max), so older rows are pure
storage cost.  This PR adds:

  * ``prune_polymarket_history(retention_days)`` -- bounded
    batched DELETE on rows older than the cutoff
  * ``POLYMARKET_HISTORY_RETENTION_DAYS`` env var (default 30)
  * Opportunistic prune call in polymarket_scanner.py after every
    scan (silently skipped if it fails -- not critical path)
  * scripts/prune_polymarket_history.py CLI for retroactive trim

Tests cover:
  * The prune function only touches rows older than the cutoff
  * Default retention (30 days) preserves recent rows
  * Cap of 0 disables pruning entirely
  * Both target tables are pruned
  * Dimension tables (markets, tokens, trades) are NEVER touched
  * Batched DELETE is idempotent (re-run is a no-op)
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager


def _setup_polymarket_test_db(tmp_path, monkeypatch):
    """Create a minimal polymarket schema + stub db.get_connection."""
    db_path = tmp_path / "test_polymarket.db"
    conn0 = sqlite3.connect(str(db_path))
    conn0.row_factory = sqlite3.Row
    # High-frequency tables: subject to pruning.  Uses the REAL
    # production composite-PK schema (no id, no rowid alias) so the
    # prune SQL exercises the same shape it would on prod.
    conn0.execute("""
        CREATE TABLE polymarket_price_points (
            token_id TEXT,
            timestamp_ms INTEGER,
            price REAL,
            source TEXT,
            metadata TEXT,
            PRIMARY KEY (token_id, timestamp_ms, source)
        )
    """)
    conn0.execute("""
        CREATE TABLE polymarket_market_snapshots (
            market_id TEXT,
            observed_at_ms INTEGER,
            probability REAL,
            raw_market TEXT,
            PRIMARY KEY (market_id, observed_at_ms)
        )
    """)
    # Dimension tables: NEVER pruned
    conn0.execute("""
        CREATE TABLE polymarket_markets (
            market_id TEXT PRIMARY KEY,
            question TEXT,
            first_seen_ms INTEGER,
            last_seen_ms INTEGER
        )
    """)
    conn0.execute("""
        CREATE TABLE polymarket_tokens (
            token_id TEXT PRIMARY KEY,
            market_id TEXT
        )
    """)
    conn0.execute("""
        CREATE TABLE polymarket_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT,
            observed_at_ms INTEGER
        )
    """)
    conn0.commit()
    conn0.close()

    @contextmanager
    def fake_get_connection(*a, **k):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    from src.data import database as real_db
    monkeypatch.setattr(real_db, "get_connection", fake_get_connection)

    return db_path


def _seed_with_old_and_new_rows(db_path, *, old_count: int, new_count: int):
    """Insert N rows older than cutoff and N rows newer."""
    now_ms = int(time.time() * 1000)
    # "old" = 60 days ago, "new" = 1 day ago
    old_ts = now_ms - 60 * 86_400_000
    new_ts = now_ms - 1 * 86_400_000

    conn = sqlite3.connect(str(db_path))
    for i in range(old_count):
        conn.execute(
            "INSERT INTO polymarket_price_points "
            "(token_id, timestamp_ms, price, source, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            ("tok_old", old_ts + i, 0.5, "test", "{}"),
        )
        conn.execute(
            "INSERT INTO polymarket_market_snapshots "
            "(market_id, observed_at_ms, probability, raw_market) "
            "VALUES (?, ?, ?, ?)",
            ("mkt_old", old_ts + i, 0.5, "{}"),
        )
    for i in range(new_count):
        conn.execute(
            "INSERT INTO polymarket_price_points "
            "(token_id, timestamp_ms, price, source, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            ("tok_new", new_ts + i, 0.5, "test", "{}"),
        )
        conn.execute(
            "INSERT INTO polymarket_market_snapshots "
            "(market_id, observed_at_ms, probability, raw_market) "
            "VALUES (?, ?, ?, ?)",
            ("mkt_new", new_ts + i, 0.5, "{}"),
        )
    # Dimension rows -- one of each, should never be deleted
    conn.execute(
        "INSERT INTO polymarket_markets "
        "(market_id, question, first_seen_ms, last_seen_ms) "
        "VALUES (?, ?, ?, ?)",
        ("mkt_old", "Q", old_ts, old_ts),
    )
    conn.execute(
        "INSERT INTO polymarket_tokens (token_id, market_id) VALUES (?, ?)",
        ("tok_old", "mkt_old"),
    )
    conn.execute(
        "INSERT INTO polymarket_trades (trade_id, observed_at_ms) VALUES (?, ?)",
        ("trade_old", old_ts),
    )
    conn.commit()
    conn.close()


# ── Headline guarantees ────────────────────────────────────


def test_prune_deletes_old_rows_only(tmp_path, monkeypatch):
    """30-day retention removes 60-day-old rows but keeps 1-day-old rows."""
    db_path = _setup_polymarket_test_db(tmp_path, monkeypatch)
    _seed_with_old_and_new_rows(db_path, old_count=100, new_count=50)

    from src.data.polymarket_history import prune_polymarket_history
    counts = prune_polymarket_history(retention_days=30)

    assert counts["price_points_deleted"] == 100
    assert counts["snapshots_deleted"] == 100

    conn = sqlite3.connect(str(db_path))
    pp_remain = conn.execute(
        "SELECT COUNT(*) FROM polymarket_price_points"
    ).fetchone()[0]
    sn_remain = conn.execute(
        "SELECT COUNT(*) FROM polymarket_market_snapshots"
    ).fetchone()[0]
    conn.close()
    assert pp_remain == 50  # the 1-day-old rows survive
    assert sn_remain == 50


def test_prune_preserves_dimension_tables(tmp_path, monkeypatch):
    """Markets, tokens, and trades tables are never touched."""
    db_path = _setup_polymarket_test_db(tmp_path, monkeypatch)
    _seed_with_old_and_new_rows(db_path, old_count=50, new_count=10)

    from src.data.polymarket_history import prune_polymarket_history
    prune_polymarket_history(retention_days=30)

    conn = sqlite3.connect(str(db_path))
    assert conn.execute(
        "SELECT COUNT(*) FROM polymarket_markets"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM polymarket_tokens"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM polymarket_trades"
    ).fetchone()[0] == 1
    conn.close()


def test_retention_zero_disables_pruning(tmp_path, monkeypatch):
    """retention_days=0 short-circuits with no deletes."""
    db_path = _setup_polymarket_test_db(tmp_path, monkeypatch)
    _seed_with_old_and_new_rows(db_path, old_count=20, new_count=10)

    from src.data.polymarket_history import prune_polymarket_history
    counts = prune_polymarket_history(retention_days=0)

    assert counts["price_points_deleted"] == 0
    assert counts["snapshots_deleted"] == 0

    # Nothing was deleted.
    conn = sqlite3.connect(str(db_path))
    assert conn.execute(
        "SELECT COUNT(*) FROM polymarket_price_points"
    ).fetchone()[0] == 30
    conn.close()


def test_retention_negative_disables_pruning(tmp_path, monkeypatch):
    """retention_days < 0 is treated as 'disabled' rather than crashing."""
    db_path = _setup_polymarket_test_db(tmp_path, monkeypatch)
    _seed_with_old_and_new_rows(db_path, old_count=10, new_count=5)

    from src.data.polymarket_history import prune_polymarket_history
    counts = prune_polymarket_history(retention_days=-1)
    assert counts["price_points_deleted"] == 0
    assert counts["snapshots_deleted"] == 0


# ── Batching + idempotency ─────────────────────────────────


def test_prune_is_idempotent(tmp_path, monkeypatch):
    """Re-running the prune is a no-op (the old rows are already gone)."""
    db_path = _setup_polymarket_test_db(tmp_path, monkeypatch)
    _seed_with_old_and_new_rows(db_path, old_count=50, new_count=20)

    from src.data.polymarket_history import prune_polymarket_history
    first = prune_polymarket_history(retention_days=30)
    second = prune_polymarket_history(retention_days=30)

    assert first["price_points_deleted"] == 50
    assert first["snapshots_deleted"] == 50
    assert second["price_points_deleted"] == 0
    assert second["snapshots_deleted"] == 0


def test_prune_handles_large_batches(tmp_path, monkeypatch):
    """Pruning 12_000 rows with batch_size=1000 deletes all in 12 batches."""
    db_path = _setup_polymarket_test_db(tmp_path, monkeypatch)
    _seed_with_old_and_new_rows(db_path, old_count=12_000, new_count=100)

    from src.data.polymarket_history import prune_polymarket_history
    counts = prune_polymarket_history(retention_days=30, batch_size=1000)

    assert counts["price_points_deleted"] == 12_000
    assert counts["snapshots_deleted"] == 12_000

    conn = sqlite3.connect(str(db_path))
    pp = conn.execute(
        "SELECT COUNT(*) FROM polymarket_price_points"
    ).fetchone()[0]
    conn.close()
    assert pp == 100


def test_prune_no_rows_to_delete_returns_zero(tmp_path, monkeypatch):
    """Pristine DB with only-new rows: prune returns 0/0."""
    db_path = _setup_polymarket_test_db(tmp_path, monkeypatch)
    _seed_with_old_and_new_rows(db_path, old_count=0, new_count=25)

    from src.data.polymarket_history import prune_polymarket_history
    counts = prune_polymarket_history(retention_days=30)
    assert counts == {"price_points_deleted": 0, "snapshots_deleted": 0}


# ── Boundary check on the cutoff ────────────────────────────


def test_row_exactly_at_cutoff_uses_strict_less_than(tmp_path, monkeypatch):
    """The ranged DELETE uses strict ``< cutoff_ms``, not ``<=``.

    A row right at the cutoff timestamp must survive.  The current
    implementation builds the SQL as ``{ts_col} >= ? AND {ts_col} <
    ?`` with the upper bound bound to ``cutoff_ms``, so a row at
    exactly the cutoff is excluded from the slice.
    """
    import inspect
    from src.data import polymarket_history

    src = inspect.getsource(polymarket_history.prune_polymarket_history)
    # The upper bound of each time slice uses strict <, not <=.
    assert "{ts_col} < ?" in src or "ts_col} < ?" in src, (
        "DELETE must use strict '< cutoff_ms'; an inclusive cutoff "
        "would delete rows right at the boundary.  Source did not "
        "contain the expected upper-bound predicate."
    )


def test_delete_uses_no_rowid_no_id_subquery(tmp_path, monkeypatch):
    """SQL must not reference rowid OR id, since the production tables
    have neither -- composite PKs only.

    Production runs SQLite + Postgres dualwrite.  Two consecutive
    regressions observed on 2026-05-26:
      1. ``WHERE rowid IN (SELECT rowid ...)`` -- failed on Postgres
         (no rowid pseudo-column)
      2. ``WHERE id IN (SELECT id ...)`` -- failed on SQLite
         (no id column, the PK is composite)

    The correct cross-backend form is a plain ranged DELETE on the
    timestamp column with no subquery and no PK enumeration.
    """
    import inspect
    from src.data import polymarket_history

    src = inspect.getsource(polymarket_history.prune_polymarket_history)
    assert "WHERE rowid" not in src and "SELECT rowid" not in src, (
        "Prune must not reference rowid -- it's SQLite-only"
    )
    assert "WHERE id IN" not in src and "SELECT id FROM polymarket" not in src, (
        "Prune must not reference an ``id`` column -- the production "
        "polymarket tables use composite PKs, no id is declared"
    )
    # The correct form: ranged DELETE on the timestamp column via an
    # f-string that interpolates the table name and ts_col.
    assert "polymarket_price_points" in src
    assert "polymarket_market_snapshots" in src
    assert "timestamp_ms" in src and "observed_at_ms" in src
    # The DELETE statement uses ``>= ? AND ... < ?`` as the slice
    # predicate.
    assert ">= ? AND" in src and "< ?" in src
