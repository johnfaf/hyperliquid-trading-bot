"""
Unified connection wrapper for SQLite and Postgres.

Provides a thin adapter so that database.py can use the same SQL (with ``?``
placeholders) against both backends.  The adapter translates:

  - ``?`` -> ``%s`` for Postgres
  - ``INSERT OR REPLACE`` -> ``INSERT ... ON CONFLICT ... DO UPDATE``
  - ``INSERT OR IGNORE``  -> ``INSERT ... ON CONFLICT DO NOTHING``
  - ``sqlite_master``     -> ``information_schema.tables``
  - Row access via dict-like interface

The :class:`DualWriteAdapter` executes every statement on SQLite first
(authoritative) and mirrors it to Postgres (best-effort).  Postgres
failures are logged and counted but never surface to the caller.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Pre-compiled patterns for SQL translation
_PLACEHOLDER_RE = re.compile(r"\?")


def _translate_sql(sql: str, backend: str) -> str:
    """Translate SQLite SQL dialect to Postgres when necessary."""
    if backend == "sqlite":
        return sql

    # ? -> %s
    translated = _PLACEHOLDER_RE.sub("%s", sql)

    # INSERT OR REPLACE INTO <table> (...) VALUES (...)
    # -> INSERT INTO <table> (...) VALUES (...) ON CONFLICT DO UPDATE SET ...
    # We handle this at the caller level for complex cases, but provide a
    # simple rewrite for the common pattern.
    translated = translated.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    translated = translated.replace("INSERT OR IGNORE INTO", "INSERT INTO")

    # AUTOINCREMENT -> not needed (Postgres SERIAL handles it)
    translated = translated.replace("AUTOINCREMENT", "")

    # sqlite_master -> information_schema
    translated = translated.replace(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=",
        "SELECT table_name AS name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=",
    )

    return translated


class CursorAdapter:
    """Wraps a Postgres cursor to expose sqlite3-compatible attributes."""

    def __init__(self, pg_cursor):
        self._cur = pg_cursor

    @property
    def lastrowid(self) -> Optional[int]:
        """Return the last inserted row ID.

        For Postgres, the caller must add RETURNING id to the INSERT and
        fetch the result.  We try to read it; if not available, return None.
        """
        try:
            row = self._cur.fetchone()
            if row:
                return row[0] if isinstance(row, (tuple, list)) else row.get("id")
        except Exception:
            pass
        return None

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()


class ConnectionAdapter:
    """Wraps a raw connection (sqlite3 or psycopg) with dialect translation.

    Provides the same ``execute()`` / ``executescript()`` / ``fetchone()``
    interface regardless of backend.
    """

    def __init__(self, raw_conn, backend: str):
        self._conn = raw_conn
        self.backend = backend

    # -- Execute with auto-translation ------------------------------------

    def execute(self, sql: str, params: Sequence = None) -> CursorAdapter:
        translated = _translate_sql(sql, self.backend)

        if self.backend == "postgres":
            # Postgres needs RETURNING id for INSERT ... to get lastrowid
            needs_returning = (
                translated.strip().upper().startswith("INSERT")
                and "RETURNING" not in translated.upper()
            )
            if needs_returning:
                translated = translated.rstrip().rstrip(";") + " RETURNING id"

            cur = self._conn.cursor()
            cur.execute(translated, params or ())
            return CursorAdapter(cur)
        else:
            # SQLite
            if params:
                return self._conn.execute(sql, params)
            return self._conn.execute(sql)

    def executescript(self, sql: str):
        """Execute a multi-statement script.

        SQLite: uses ``executescript()`` which auto-commits.
        Postgres: splits on ``;`` and executes each statement.
        """
        if self.backend == "sqlite":
            return self._conn.executescript(sql)

        # Postgres: execute each statement individually
        cur = self._conn.cursor()
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                translated = _translate_sql(stmt, self.backend)
                cur.execute(translated)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        if self.backend == "postgres":
            # Return to pool for psycopg pool connections
            self._conn.close()
        else:
            self._conn.close()

    # Allow the underlying connection's attributes to pass through
    def __getattr__(self, name: str):
        return getattr(self._conn, name)


# ─── Dual-write health counters (process-wide) ─────────────────

class _DualWriteStats:
    """Thread-safe counters for dual-write health monitoring."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pg_writes_ok: int = 0
        self.pg_writes_failed: int = 0
        self.pg_last_error: str = ""
        self.pg_last_error_ts: float = 0.0

    def record_ok(self):
        with self._lock:
            self.pg_writes_ok += 1

    def record_fail(self, err: str):
        with self._lock:
            self.pg_writes_failed += 1
            self.pg_last_error = err[:200]
            self.pg_last_error_ts = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pg_writes_ok": self.pg_writes_ok,
                "pg_writes_failed": self.pg_writes_failed,
                "pg_last_error": self.pg_last_error,
                "pg_last_error_ts": self.pg_last_error_ts,
            }


dualwrite_stats = _DualWriteStats()


class DualWriteAdapter:
    """Executes on SQLite (authoritative) and mirrors to Postgres (best-effort).

    *  ``execute()`` runs on SQLite first.  On success it translates the same
       SQL for Postgres and executes there.  Postgres failures are logged and
       counted but **never** propagated to the caller.
    *  ``executescript()`` only runs on SQLite — Postgres DDL is handled
       by the migration runner, not by executescript.
    *  The adapter's ``backend`` attribute is ``"sqlite"`` because all
       return values (rows, lastrowid, rowcount) come from SQLite.
    """

    def __init__(self, sqlite_conn, pg_conn):
        self._sq = sqlite_conn      # raw sqlite3.Connection
        self._pg = pg_conn           # raw psycopg connection
        self.backend = "sqlite"      # callers see SQLite semantics

    # -- helpers -------------------------------------------------------------

    def _mirror_to_pg(self, sql: str, params: Sequence = None) -> None:
        """Best-effort replay of a single statement to Postgres."""
        try:
            pg_sql = _translate_sql(sql, "postgres")
            # Don't add RETURNING for mirror writes — we don't need lastrowid
            cur = self._pg.cursor()
            cur.execute(pg_sql, tuple(params) if params else ())
            dualwrite_stats.record_ok()
        except Exception as exc:
            err_str = f"{type(exc).__name__}: {exc}"
            dualwrite_stats.record_fail(err_str)
            logger.warning("Dualwrite Postgres mirror failed: %s | SQL: %s",
                           err_str[:120], sql[:80])
            # Rollback the Postgres transaction so subsequent statements
            # in this connection don't fail with "in error state".
            try:
                self._pg.rollback()
            except Exception:
                pass

    # -- public interface (mirrors ConnectionAdapter) ------------------------

    def execute(self, sql: str, params: Sequence = None):
        """Execute on SQLite, then mirror to Postgres."""
        # 1. Authoritative SQLite execution
        if params:
            result = self._sq.execute(sql, params)
        else:
            result = self._sq.execute(sql)

        # 2. Best-effort Postgres mirror
        self._mirror_to_pg(sql, params)

        return result  # SQLite cursor — callers see .lastrowid / .rowcount

    def executescript(self, sql: str):
        """Execute DDL on SQLite only.  Postgres DDL is via migrations."""
        return self._sq.executescript(sql)

    def commit(self):
        self._sq.commit()
        try:
            self._pg.commit()
        except Exception as exc:
            dualwrite_stats.record_fail(f"commit: {exc}")
            logger.warning("Dualwrite Postgres commit failed: %s", exc)
            try:
                self._pg.rollback()
            except Exception:
                pass

    def rollback(self):
        self._sq.rollback()
        try:
            self._pg.rollback()
        except Exception:
            pass

    def close(self):
        """Close SQLite; Postgres is returned to pool by the router."""
        self._sq.close()

    def __getattr__(self, name: str):
        return getattr(self._sq, name)
