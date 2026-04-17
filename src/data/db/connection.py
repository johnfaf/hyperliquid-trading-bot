"""
Unified connection wrapper for SQLite and Postgres.

Provides a thin adapter so that database.py can use the same SQL (with ``?``
placeholders) against both backends.  The adapter translates:

  - ``?`` -> ``%s`` for Postgres
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
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Pre-compiled patterns for SQL translation
_PLACEHOLDER_RE = re.compile(r"\?")
_PRAGMA_TABLE_INFO_RE = re.compile(
    r"^\s*PRAGMA\s+table_info\((?P<table>[^)]+)\)\s*;?\s*$",
    re.IGNORECASE,
)
_BOOL_COLUMN_EQ_INT_RE = re.compile(
    r"\b(?P<column>active|is_golden|connected_to_live|is_liquidation)\s*=\s*(?P<value>[01])\b",
    re.IGNORECASE,
)
_DATETIME_NOW_MINUS_DAYS_RE = re.compile(
    r"datetime\(\s*'now'\s*,\s*'-(?P<days>\d+)\s+days'\s*\)",
    re.IGNORECASE,
)
_DATETIME_NOW_RE = re.compile(r"datetime\(\s*'now'\s*\)", re.IGNORECASE)
_INSERT_VALUES_RE = re.compile(
    r"(?is)^\s*INSERT\s+(?:OR\s+(?:REPLACE|IGNORE)\s+)?INTO\s+"
    r"(?P<table>[\"`\w\.]+)\s*\((?P<columns>.*?)\)\s*VALUES\s*"
    r"\((?P<values>.*?)\)(?P<suffix>\s*.*)$"
)
_SQLITE_ONLY_PREFIXES = ("PRAGMA",)
_DDL_PREFIXES = ("CREATE ", "ALTER ", "DROP ", "VACUUM", "REINDEX")


def _clean_identifier(raw: str) -> str:
    return raw.strip().strip("\"'`[]")


def _bool_literal_replacer(match: re.Match) -> str:
    return f"{match.group('column')} = {'TRUE' if match.group('value') == '1' else 'FALSE'}"


def _translate_sql(sql: str, backend: str) -> str:
    """Translate SQLite SQL dialect to Postgres when necessary."""
    if backend == "sqlite":
        return sql

    pragma_match = _PRAGMA_TABLE_INFO_RE.match(sql.strip())
    if pragma_match:
        table = _clean_identifier(pragma_match.group("table")).lower()
        return (
            "SELECT column_name AS name FROM information_schema.columns "
            f"WHERE table_schema='public' AND table_name='{table}' "
            "ORDER BY ordinal_position"
        )

    # ? -> %s
    translated = _PLACEHOLDER_RE.sub("%s", sql)

    # UPSERT behaviour is handled by the callers with explicit
    # ``ON CONFLICT`` clauses so both backends share the same semantics.
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
    translated = _DATETIME_NOW_MINUS_DAYS_RE.sub(
        lambda match: f"(now() - INTERVAL '{match.group('days')} days')",
        translated,
    )
    translated = _DATETIME_NOW_RE.sub("CURRENT_TIMESTAMP", translated)
    translated = _BOOL_COLUMN_EQ_INT_RE.sub(_bool_literal_replacer, translated)

    return translated


class CursorAdapter:
    """Wraps a Postgres cursor to expose sqlite3-compatible attributes.

    CRIT-FIX C1: the first row returned by the query is buffered so that
    ``.lastrowid`` and ``.fetchone()`` can both be called without one draining
    the other — a common pattern after ``INSERT ... RETURNING id``.  Previously
    ``.lastrowid`` consumed the cursor, making every subsequent ``.fetchone()``
    miss data.
    """

    def __init__(self, pg_cursor):
        self._cur = pg_cursor
        self._first_row_cached = False
        self._first_row = None
        self._lastrowid_value: Optional[int] = None

    def _ensure_first_row_cached(self) -> None:
        if self._first_row_cached:
            return
        self._first_row_cached = True
        try:
            self._first_row = self._cur.fetchone()
        except Exception:
            self._first_row = None
            return
        row = self._first_row
        if row is None:
            return
        try:
            if isinstance(row, (tuple, list)):
                self._lastrowid_value = row[0]
            elif hasattr(row, "get"):
                self._lastrowid_value = row.get("id") or (
                    next(iter(row.values())) if row else None
                )
            else:
                try:
                    self._lastrowid_value = row["id"]
                except Exception:
                    self._lastrowid_value = None
        except Exception:
            self._lastrowid_value = None

    @property
    def lastrowid(self) -> Optional[int]:
        """Return the last inserted row ID when the query used RETURNING.

        Buffers the first row on first access so subsequent ``fetchone()``
        calls still return it (i.e. the accessor is non-destructive).
        """
        self._ensure_first_row_cached()
        return self._lastrowid_value

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    def fetchone(self):
        if self._first_row_cached and self._first_row is not None:
            row = self._first_row
            self._first_row = None
            return row
        return self._cur.fetchone()

    def fetchall(self):
        if self._first_row_cached and self._first_row is not None:
            remaining = self._cur.fetchall()
            first = self._first_row
            self._first_row = None
            return [first] + list(remaining)
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
        self._sqlite_id_pk_cache: dict[str, bool] = {}

    # -- helpers -------------------------------------------------------------

    def _should_skip_pg_mirror(self, sql: str) -> bool:
        stripped = (sql or "").lstrip().upper()
        return stripped.startswith(_SQLITE_ONLY_PREFIXES) or stripped.startswith(_DDL_PREFIXES)

    def _sqlite_table_has_integer_id_pk(self, table: str) -> bool:
        table = _clean_identifier(table).split(".")[-1]
        cached = self._sqlite_id_pk_cache.get(table)
        if cached is not None:
            return cached
        try:
            rows = self._sq.execute(f"PRAGMA table_info({table})").fetchall()
            has_id_pk = any(
                _clean_identifier(str(row["name"])).lower() == "id" and int(row["pk"] or 0) == 1
                for row in rows
            )
        except Exception:
            has_id_pk = False
        self._sqlite_id_pk_cache[table] = has_id_pk
        return has_id_pk

    def _prepare_pg_mirror(self, sql: str, params: Sequence = None, sqlite_result=None):
        if self._should_skip_pg_mirror(sql):
            return None, None

        insert_match = _INSERT_VALUES_RE.match(sql)
        if not insert_match:
            return sql, tuple(params) if params else ()

        table = _clean_identifier(insert_match.group("table"))
        columns_raw = insert_match.group("columns").strip()
        columns = [_clean_identifier(col).lower() for col in columns_raw.split(",")]
        if "id" in columns:
            return sql, tuple(params) if params else ()

        inserted_id = getattr(sqlite_result, "lastrowid", None)
        if not inserted_id or not self._sqlite_table_has_integer_id_pk(table):
            return sql, tuple(params) if params else ()

        values_raw = insert_match.group("values").strip()
        suffix = insert_match.group("suffix") or ""
        mirror_sql = (
            f"INSERT INTO {insert_match.group('table')} (id, {columns_raw}) "
            f"VALUES (?, {values_raw}){suffix}"
        )
        mirror_params = (inserted_id,) + tuple(params or ())
        return mirror_sql, mirror_params

    def _mirror_to_pg(self, sql: str, params: Sequence = None, sqlite_result=None) -> None:
        """Best-effort replay of a single statement to Postgres."""
        pg_sql, pg_params = self._prepare_pg_mirror(sql, params, sqlite_result=sqlite_result)
        if pg_sql is None:
            return
        try:
            pg_sql = _translate_sql(pg_sql, "postgres")
            # Don't add RETURNING for mirror writes — we don't need lastrowid
            cur = self._pg.cursor()
            cur.execute(pg_sql, pg_params or ())
            self._pg.commit()
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
        self._mirror_to_pg(sql, params, sqlite_result=result)

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
