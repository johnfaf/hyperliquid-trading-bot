"""
Unified connection wrapper for SQLite and Postgres.

Provides a thin adapter so that database.py can use the same SQL (with ``?``
placeholders) against both backends.  The adapter translates:

  - ``?`` -> ``%s`` for Postgres
  - ``sqlite_master``     -> ``information_schema.tables``
  - Row access via dict-like interface

The :class:`DualWriteAdapter` executes every statement on SQLite first
(authoritative) and mirrors write statements to Postgres (best-effort).
Postgres failures are logged and counted but never surface to the caller.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
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
_PAPER_TRADES_METADATA_CAS_QMARK_RE = re.compile(
    r"(?is)^\s*UPDATE\s+paper_trades\s+SET\s+metadata\s*=\s*\?\s+"
    r"WHERE\s+id\s*=\s*\?\s+AND\s+COALESCE\(\s*metadata\s*,\s*''\s*\)\s*=\s*"
    r"COALESCE\(\s*\?\s*,\s*''\s*\)\s*$"
)
_PAPER_TRADES_METADATA_CAS_PERCENT_RE = re.compile(
    r"(?is)^\s*UPDATE\s+paper_trades\s+SET\s+metadata\s*=\s*%s\s+"
    r"WHERE\s+id\s*=\s*%s\s+AND\s+COALESCE\(\s*metadata\s*,\s*''\s*\)\s*=\s*"
    r"COALESCE\(\s*%s\s*,\s*''\s*\)\s*$"
)
_PAPER_TRADES_METADATA_CAS_JSONB_SQL = (
    "UPDATE paper_trades SET metadata = %s::jsonb "
    "WHERE id = %s AND COALESCE(metadata, '{}'::jsonb) = "
    "COALESCE(%s::jsonb, '{}'::jsonb)"
)
_SQLITE_ONLY_PREFIXES = ("PRAGMA",)
_DDL_PREFIXES = ("CREATE ", "ALTER ", "DROP ", "VACUUM", "REINDEX")
_MIRRORED_WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "REPLACE")

# Columns that are BOOLEAN in Postgres but historically stored as 0/1 ints in
# SQLite.  When dualwriting parameterised INSERTs we must coerce these values
# to Python bool so psycopg binds them as the correct Postgres type.  Without
# this, every INSERT into wallet_fills / golden_wallets fails with
# "column is of type boolean but expression is of type smallint".
_POSTGRES_BOOLEAN_COLUMNS = frozenset({
    "active",
    "is_golden",
    "connected_to_live",
    "is_liquidation",
})


def _coerce_bool_params(columns, params):
    """Return a new params tuple with 0/1 ints in boolean columns cast to bool.

    ``columns`` is expected to be a list of lowercased column names lined up
    positionally with ``params``.  Values that are already ``None`` or ``bool``
    pass through untouched.  If the lengths don't match we return ``params``
    unchanged — that path is a caller bug, not something we should guess at.
    """
    if not params:
        return params
    params = tuple(params)
    if len(columns) != len(params):
        return params
    out = list(params)
    for i, col in enumerate(columns):
        if col in _POSTGRES_BOOLEAN_COLUMNS:
            v = out[i]
            if v is None or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out[i] = bool(v)
    return tuple(out)


def _coerce_paper_metadata_cas_params(params):
    """Normalize empty SQLite metadata CAS values into valid JSON for Postgres."""
    if not params:
        return ()
    out = list(tuple(params))
    if len(out) >= 3:
        if out[0] in (None, ""):
            out[0] = "{}"
        if out[2] in (None, ""):
            out[2] = "{}"
    return tuple(out)


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
    if _PAPER_TRADES_METADATA_CAS_PERCENT_RE.match(translated):
        # SQLite stores paper trade metadata as TEXT.  Postgres stores the same
        # column as JSONB, so the mirrored compare-and-swap predicate must avoid
        # COALESCE(metadata, '') because '' is not valid JSON.
        return _PAPER_TRADES_METADATA_CAS_JSONB_SQL

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
    """Thread-safe counters for dual-write health monitoring.

    H4 (audit): dualwrite silently logging Postgres mirror failures means
    the operator can think they have a resilient audit ledger while every
    mirror write has been failing for hours.  This class now also tracks
    a rolling deque of recent failure timestamps so ``is_healthy()`` can
    answer *is Postgres currently keeping up?* as opposed to the
    cumulative ``pg_writes_failed`` counter which never shrinks and can't
    tell a healed outage from an ongoing one.
    """

    # Cap the failure-timestamp deque so a long outage cannot grow the
    # memory footprint without bound.  Any value here above the
    # ``max_failures`` threshold is sufficient for the health check.
    _MAX_FAILURE_TIMESTAMPS: int = 1024

    def __init__(self):
        self._lock = threading.Lock()
        self.pg_writes_ok: int = 0
        self.pg_writes_failed: int = 0
        self.pg_last_error: str = ""
        self.pg_last_error_ts: float = 0.0
        self._recent_failures: deque[float] = deque(
            maxlen=self._MAX_FAILURE_TIMESTAMPS
        )

    def record_ok(self):
        with self._lock:
            self.pg_writes_ok += 1

    def record_fail(self, err: str):
        now = time.time()
        with self._lock:
            self.pg_writes_failed += 1
            self.pg_last_error = err[:200]
            self.pg_last_error_ts = now
            self._recent_failures.append(now)

    def recent_failures(self, window_s: float) -> int:
        """Return the number of mirror failures seen in the last ``window_s`` seconds.

        Counts non-destructively so concurrent callers using different
        windows don't steal each other's history — the deque's
        ``maxlen`` already bounds memory on a long outage.
        """
        cutoff = time.time() - float(window_s)
        with self._lock:
            return sum(1 for ts in self._recent_failures if ts >= cutoff)

    def is_healthy(self, *, window_s: float = 300.0, max_failures: int = 5) -> bool:
        """Return True if recent Postgres mirror failures stay under threshold.

        ``window_s``: size of the rolling window in seconds (default 5 min).
        ``max_failures``: strict upper bound on failures allowed in the
        window before the backend is considered unhealthy.  A single
        transient error does not trip the guard; a sustained outage does.
        """
        return self.recent_failures(window_s) <= int(max_failures)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pg_writes_ok": self.pg_writes_ok,
                "pg_writes_failed": self.pg_writes_failed,
                "pg_last_error": self.pg_last_error,
                "pg_last_error_ts": self.pg_last_error_ts,
                "recent_failures_5m": sum(
                    1 for ts in self._recent_failures
                    if ts >= time.time() - 300.0
                ),
            }


dualwrite_stats = _DualWriteStats()


def dualwrite_is_healthy(*, window_s: float = 300.0, max_failures: int = 5) -> bool:
    """Module-level convenience wrapper around :meth:`_DualWriteStats.is_healthy`.

    Callers that do not want to import the private ``dualwrite_stats``
    instance directly can use this helper.  Returns ``True`` when the
    rolling failure window is clean; ``False`` when Postgres mirroring
    has been failing often enough that scaled live trading should stop
    treating the Postgres ledger as reliable.
    """
    return dualwrite_stats.is_healthy(
        window_s=window_s, max_failures=max_failures,
    )


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
        if not stripped:
            return True
        first_token = stripped.split(None, 1)[0].rstrip(";")
        return (
            stripped.startswith(_SQLITE_ONLY_PREFIXES)
            or stripped.startswith(_DDL_PREFIXES)
            or first_token not in _MIRRORED_WRITE_PREFIXES
        )

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

        if _PAPER_TRADES_METADATA_CAS_QMARK_RE.match(sql or ""):
            return sql, _coerce_paper_metadata_cas_params(params)

        insert_match = _INSERT_VALUES_RE.match(sql)
        if not insert_match:
            return sql, tuple(params) if params else ()

        table = _clean_identifier(insert_match.group("table"))
        columns_raw = insert_match.group("columns").strip()
        columns = [_clean_identifier(col).lower() for col in columns_raw.split(",")]
        coerced_params = _coerce_bool_params(columns, tuple(params) if params else ())
        if "id" in columns:
            return sql, coerced_params

        inserted_id = getattr(sqlite_result, "lastrowid", None)
        if not inserted_id or not self._sqlite_table_has_integer_id_pk(table):
            return sql, coerced_params

        values_raw = insert_match.group("values").strip()
        suffix = insert_match.group("suffix") or ""
        mirror_sql = (
            f"INSERT INTO {insert_match.group('table')} (id, {columns_raw}) "
            f"VALUES (?, {values_raw}){suffix}"
        )
        mirror_params = (inserted_id,) + coerced_params
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
