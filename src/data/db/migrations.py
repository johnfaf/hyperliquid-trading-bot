"""
Forward-only migration runner for Postgres.

Tracks applied migrations in a ``schema_migrations`` table.
Migration files live in ``migrations/`` at the project root and are
named ``NNNN_description.sql`` (e.g. ``0001_init_schema.sql``).

Usage::

    from src.data.db.migrations import run_migrations
    run_migrations()   # applies any pending .sql files
"""
from __future__ import annotations

import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

_MIGRATION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "migrations"
)
_MIGRATION_FILE_RE = re.compile(r"^(\d{4})_.+\.sql$")


def _ensure_schema_migrations_table(conn) -> None:
    """Create the tracking table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version  TEXT PRIMARY KEY,
            applied  TIMESTAMPTZ NOT NULL DEFAULT now(),
            filename TEXT NOT NULL
        )
    """)
    conn.commit()


def _applied_versions(conn) -> set:
    """Return set of already-applied migration version strings."""
    cur = conn.execute("SELECT version FROM schema_migrations ORDER BY version")
    return {row["version"] for row in cur.fetchall()}


def _discover_migrations() -> List[tuple]:
    """Scan the migrations/ directory for SQL files, sorted by version.

    Returns list of (version, filename, full_path).
    """
    if not os.path.isdir(_MIGRATION_DIR):
        return []

    results = []
    for fname in sorted(os.listdir(_MIGRATION_DIR)):
        m = _MIGRATION_FILE_RE.match(fname)
        if m:
            version = m.group(1)
            results.append((version, fname, os.path.join(_MIGRATION_DIR, fname)))
    return results


def run_migrations() -> int:
    """Apply all pending migrations. Returns count of newly applied migrations."""
    from src.data.db.postgres import get_connection, return_connection

    conn = get_connection()
    try:
        _ensure_schema_migrations_table(conn)
        applied = _applied_versions(conn)
        pending = [
            (v, fname, path)
            for v, fname, path in _discover_migrations()
            if v not in applied
        ]

        if not pending:
            logger.info("Postgres schema is up to date (no pending migrations).")
            return 0

        for version, fname, path in pending:
            logger.info("Applying migration %s: %s", version, fname)
            with open(path, "r") as f:
                sql = f.read()

            # Execute the migration
            cur = conn.cursor()
            cur.execute(sql)

            # Record it
            cur.execute(
                "INSERT INTO schema_migrations (version, filename) VALUES (%s, %s)",
                (version, fname),
            )
            conn.commit()
            logger.info("Migration %s applied successfully.", version)

        logger.info("Applied %d migration(s).", len(pending))
        return len(pending)

    except Exception:
        conn.rollback()
        raise
    finally:
        return_connection(conn)
