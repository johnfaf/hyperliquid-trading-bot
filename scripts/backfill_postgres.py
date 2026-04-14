#!/usr/bin/env python3
"""
Backfill Postgres from the SQLite runtime database.

Run once before switching DB_BACKEND from ``sqlite`` to ``dualwrite`` or
``postgres``.  The script:

  1. Reads every row from SQLite authoritative tables.
  2. INSERTs into Postgres (skipping duplicates via ON CONFLICT DO NOTHING).
  3. Resets Postgres sequences so new auto-ids don't collide.
  4. Runs a parity check (row counts, balances, latest audit rows).

Usage::

    # Ensure POSTGRES_DSN is set
    export POSTGRES_DSN="postgresql://user:pass@host:5432/dbname"

    # Run the backfill
    python scripts/backfill_postgres.py

    # Then switch to dualwrite for 24-48h observation
    export DB_BACKEND=dualwrite
"""
import json
import logging
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")

# ─── Tables to migrate (order matters for FK constraints) ──────
TABLES = [
    "traders",
    "strategies",
    "position_snapshots",
    "strategy_scores",
    "regime_history",
    "paper_account",
    "paper_trades",
    "research_logs",
    "bot_state",
    "audit_trail",
    "golden_wallets",
    "wallet_fills",
    "calibration_records",
    "agent_scores",
    "arena_agents",
    "arena_rounds",
    "trade_memory",
    "shadow_trades",
    "shadow_attribution",
]

# Tables with auto-increment sequences to reset
SERIAL_TABLES = [
    "position_snapshots",
    "strategies",
    "strategy_scores",
    "regime_history",
    "paper_trades",
    "research_logs",
    "audit_trail",
    "wallet_fills",
    "calibration_records",
    "shadow_trades",
    "shadow_attribution",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADOW_DB_PATH = os.path.join(PROJECT_ROOT, "shadow.db")


def _sqlite_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _shadow_sqlite_conn():
    if not os.path.exists(SHADOW_DB_PATH):
        return None
    conn = sqlite3.connect(SHADOW_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists_sqlite(conn, name):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _table_exists_pg(cur, name):
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=%s",
        (name,),
    )
    return cur.fetchone() is not None


def _get_columns(sq_conn, table):
    """Get column names from SQLite table."""
    row = sq_conn.execute(f"SELECT * FROM {table} LIMIT 0")
    return [desc[0] for desc in row.description]


def _backfill_table(sq_conn, pg_conn, table):
    """Copy all rows from SQLite to Postgres for one table."""
    if not _table_exists_sqlite(sq_conn, table):
        logger.info("  %-25s SKIP (not in SQLite)", table)
        return 0

    cur = pg_conn.cursor()
    if not _table_exists_pg(cur, table):
        logger.warning("  %-25s SKIP (not in Postgres — run migrations first)", table)
        return 0

    columns = _get_columns(sq_conn, table)
    rows = sq_conn.execute(f"SELECT * FROM {table}").fetchall()

    if not rows:
        logger.info("  %-25s 0 rows (empty)", table)
        return 0

    # Build INSERT ... ON CONFLICT DO NOTHING
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    inserted = 0
    for row in rows:
        values = []
        for col in columns:
            val = row[col]
            # Convert SQLite integer booleans to Python bools for Postgres BOOLEAN cols
            if col in ("active", "is_golden", "connected_to_live", "is_liquidation"):
                val = bool(val) if val is not None else False
            # Ensure metadata/details JSON strings are valid
            if col in ("metadata", "details", "parameters", "trade_history") and isinstance(val, str):
                try:
                    json.loads(val)  # validate
                except (json.JSONDecodeError, TypeError):
                    val = "{}"
            values.append(val)

        try:
            cur.execute(insert_sql, values)
            if cur.rowcount > 0:
                inserted += 1
        except Exception as exc:
            logger.warning("  Row insert failed in %s: %s", table, str(exc)[:80])
            pg_conn.rollback()
            # Try to continue — re-establish transaction
            cur = pg_conn.cursor()

    pg_conn.commit()
    logger.info("  %-25s %d / %d rows inserted", table, inserted, len(rows))
    return inserted


def _backfill_table_from_source(sq_conn, pg_conn, table, source_label: str):
    """Copy rows from a specified SQLite source into Postgres."""
    if sq_conn is None:
        logger.info("  %-25s SKIP (%s missing)", table, source_label)
        return 0
    if not _table_exists_sqlite(sq_conn, table):
        logger.info("  %-25s SKIP (not in %s)", table, source_label)
        return 0
    return _backfill_table(sq_conn, pg_conn, table)


def _reset_sequences(pg_conn):
    """Reset Postgres sequences to max(id) + 1 so new inserts don't collide."""
    cur = pg_conn.cursor()
    for table in SERIAL_TABLES:
        try:
            if not _table_exists_pg(cur, table):
                continue
            cur.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}")
            max_id = cur.fetchone()[0]
            if max_id > 0:
                seq_name = f"{table}_id_seq"
                cur.execute(f"SELECT setval('{seq_name}', %s)", (max_id,))
                logger.info("  Sequence %-30s reset to %d", seq_name, max_id)
        except Exception as exc:
            logger.warning("  Sequence reset failed for %s: %s", table, exc)
            pg_conn.rollback()
            cur = pg_conn.cursor()
    pg_conn.commit()


def _parity_check(sq_conn, pg_conn, shadow_conn=None):
    """Compare row counts and key values between SQLite and Postgres."""
    logger.info("\n=== PARITY CHECK ===")
    cur = pg_conn.cursor()
    all_ok = True

    for table in TABLES:
        if not _table_exists_sqlite(sq_conn, table):
            continue
        if not _table_exists_pg(cur, table):
            continue

        source_conn = shadow_conn if table in {"shadow_trades", "shadow_attribution"} and shadow_conn else sq_conn
        if not _table_exists_sqlite(source_conn, table):
            continue
        sq_count = source_conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
        cur.execute(f"SELECT COUNT(*) as c FROM {table}")
        pg_count = cur.fetchone()["c"]

        status = "OK" if sq_count == pg_count else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        logger.info("  %-25s SQLite=%d  Postgres=%d  %s", table, sq_count, pg_count, status)

    # Check paper_account balance
    sq_acc = sq_conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    if sq_acc:
        cur.execute("SELECT * FROM paper_account WHERE id = 1")
        pg_acc = cur.fetchone()
        if pg_acc:
            for key in ("balance", "total_pnl", "total_trades"):
                sq_val = sq_acc[key]
                pg_val = pg_acc[key]
                match = abs(float(sq_val or 0) - float(pg_val or 0)) < 0.01
                status = "OK" if match else "MISMATCH"
                if not match:
                    all_ok = False
                logger.info("  paper_account.%-15s SQLite=%s  Postgres=%s  %s",
                            key, sq_val, pg_val, status)

    # Compare latest 5 audit rows
    sq_recent = sq_conn.execute(
        "SELECT id, action, coin, timestamp FROM audit_trail ORDER BY id DESC LIMIT 5"
    ).fetchall()
    if sq_recent:
        for row in sq_recent:
            cur.execute(
                "SELECT id, action, coin, timestamp FROM audit_trail WHERE id = %s",
                (row["id"],),
            )
            pg_row = cur.fetchone()
            if pg_row:
                match = pg_row["action"] == row["action"] and pg_row["coin"] == row["coin"]
                status = "OK" if match else "MISMATCH"
            else:
                status = "MISSING"
                all_ok = False
            logger.info("  audit_trail #%-6d %-20s %s", row["id"], row["action"], status)

    return all_ok


def main():
    if not config.POSTGRES_DSN:
        logger.error("POSTGRES_DSN is not set. Cannot backfill.")
        sys.exit(1)

    # Run migrations first
    logger.info("Running Postgres migrations...")
    from src.data.db.migrations import run_migrations
    run_migrations()

    # Open connections
    sq_conn = _sqlite_conn()
    shadow_conn = _shadow_sqlite_conn()

    from src.data.db.postgres import get_connection as pg_get_conn, return_connection as pg_return
    pg_raw = pg_get_conn()

    try:
        logger.info("\n=== BACKFILL START ===")
        logger.info("SQLite: %s", config.DB_PATH)
        logger.info("Postgres: %s", config.POSTGRES_DSN[:40] + "...")

        for table in TABLES:
            if table in {"shadow_trades", "shadow_attribution"}:
                _backfill_table_from_source(shadow_conn, pg_raw, table, "shadow.db")
            else:
                _backfill_table(sq_conn, pg_raw, table)

        logger.info("\n=== RESETTING SEQUENCES ===")
        _reset_sequences(pg_raw)

        ok = _parity_check(sq_conn, pg_raw, shadow_conn=shadow_conn)

        if ok:
            logger.info("\n PARITY CHECK PASSED — safe to enable dualwrite.")
        else:
            logger.warning("\n PARITY MISMATCH — investigate before enabling dualwrite.")

    finally:
        sq_conn.close()
        if shadow_conn is not None:
            shadow_conn.close()
        pg_return(pg_raw)


if __name__ == "__main__":
    main()
