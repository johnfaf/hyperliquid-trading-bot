"""
Database backend router.

Routes connections based on ``config.DB_BACKEND``:

  - ``sqlite``    - all ops go to SQLite (default, current behaviour)
  - ``dualwrite`` - writes go to both SQLite and Postgres; reads from SQLite
  - ``postgres``  - all ops go to Postgres

The router exposes a single ``get_connection()`` context manager that
returns a :class:`ConnectionAdapter` the rest of ``database.py`` can use
without caring about the backend.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import config

from src.core.env_utils import safe_env_float
from src.data.db.connection import ConnectionAdapter, DualWriteAdapter

logger = logging.getLogger(__name__)
_SQLITE_WRITE_LOCK = threading.RLock()


# SQLite helpers

def _sqlite_connect() -> sqlite3.Connection:
    """Open a fresh SQLite connection with WAL, FK checks, and busy_timeout."""
    import shutil

    db_path = config.DB_PATH
    db_dir = os.path.dirname(os.path.abspath(db_path))
    min_free = safe_env_float("DB_MIN_FREE_MB", 100.0, lo=1.0, hi=100_000.0)
    try:
        usage = shutil.disk_usage(db_dir)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < min_free:
            raise RuntimeError(
                f"Insufficient disk space for DB: {free_mb:.1f}MB free "
                f"(minimum {min_free:.1f}MB)"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        logger.warning("Could not determine SQLite disk usage for %s: %s", db_dir, exc)

    busy_timeout_ms = int(
        safe_env_float("DB_BUSY_TIMEOUT_MS", 15_000.0, lo=1_000.0, hi=600_000.0)
    )
    conn = sqlite3.connect(db_path, timeout=busy_timeout_ms / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    return conn


@contextmanager
def _sqlite_write_guard(enabled: bool):
    """Serialize in-process SQLite write transactions.

    WAL mode still only permits a single writer at a time. The bot runs
    multiple background threads that can all open write-capable connections
    against the same SQLite file, which turns harmless overlap into
    `database is locked` churn during startup and first-cycle scans.

    Serializing write transactions inside this process is the pragmatic fix:
    readers remain concurrent, while writers line up behind one shared guard.
    """
    if not enabled:
        yield
        return

    with _SQLITE_WRITE_LOCK:
        yield


# Postgres helpers

def _pg_connect():
    """Get a Postgres connection from the pool."""
    from src.data.db.postgres import get_connection
    return get_connection()


def _pg_return(conn):
    from src.data.db.postgres import return_connection
    return_connection(conn)


# Unified context manager

@contextmanager
def get_connection(*, for_read: bool = False):
    """Yield a :class:`ConnectionAdapter` for the active backend.

    Parameters
    ----------
    for_read : bool
        Hint that this connection will only read data.  In ``dualwrite``
        mode, reads go to SQLite; writes go to both.  In ``sqlite`` or
        ``postgres`` modes this flag is ignored.
    """
    backend = config.DB_BACKEND

    if backend == "postgres":
        raw = _pg_connect()
        adapter = ConnectionAdapter(raw, "postgres")
        try:
            yield adapter
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            _pg_return(raw)

    elif backend == "dualwrite":
        if for_read:
            raw = _sqlite_connect()
            adapter = ConnectionAdapter(raw, "sqlite")
            try:
                yield adapter
                raw.commit()
            except sqlite3.OperationalError as exc:
                logger.warning("SQLite operational error: %s", exc)
                raw.rollback()
                raise
            except Exception:
                raw.rollback()
                raise
            finally:
                raw.close()
            return

        # Every statement executes on SQLite first (authoritative), then
        # mirrors to Postgres (best-effort).  Postgres failures are logged
        # and counted but never propagate to the caller.
        with _sqlite_write_guard(enabled=True):
            raw_sq = _sqlite_connect()
            pg_raw = None
            try:
                pg_raw = _pg_connect()
            except Exception as exc:
                # Rate-limit this warning to avoid log spam when PG is down
                if not hasattr(get_connection, "_pg_warn_ts") or \
                   (time.time() - get_connection._pg_warn_ts) > 300:
                    logger.warning(
                        "Dualwrite: could not obtain Postgres connection (%s) -- "
                        "falling back to SQLite-only.", exc,
                    )
                    get_connection._pg_warn_ts = time.time()

            if pg_raw is not None:
                adapter = DualWriteAdapter(raw_sq, pg_raw)
                try:
                    yield adapter
                    adapter.commit()
                except sqlite3.OperationalError as exc:
                    logger.warning("SQLite operational error: %s", exc)
                    adapter.rollback()
                    raise
                except Exception:
                    adapter.rollback()
                    raise
                finally:
                    raw_sq.close()
                    _pg_return(pg_raw)
            else:
                # Postgres unavailable - degrade to SQLite-only
                adapter = ConnectionAdapter(raw_sq, "sqlite")
                try:
                    yield adapter
                    raw_sq.commit()
                except sqlite3.OperationalError as exc:
                    logger.warning("SQLite operational error: %s", exc)
                    raw_sq.rollback()
                    raise
                except Exception:
                    raw_sq.rollback()
                    raise
                finally:
                    raw_sq.close()

    else:
        # Default: pure SQLite
        with _sqlite_write_guard(enabled=not for_read):
            raw = _sqlite_connect()
            adapter = ConnectionAdapter(raw, "sqlite")
            try:
                yield adapter
                raw.commit()
            except sqlite3.OperationalError as exc:
                logger.warning("SQLite operational error: %s", exc)
                raw.rollback()
                raise
            except Exception:
                raw.rollback()
                raise
            finally:
                raw.close()


def is_postgres_active() -> bool:
    """True if the primary backend is Postgres (not just dual-write)."""
    return config.DB_BACKEND == "postgres"


def is_dualwrite_active() -> bool:
    """True if dual-write mode is on."""
    return config.DB_BACKEND == "dualwrite"


def _quiet_postgres_server_logs() -> None:
    """No-op placeholder for the old app-side Postgres log mutator.

    Runtime code must not mutate cluster-wide database settings. Managed
    Postgres log verbosity belongs in provider settings or an explicit
    operator-run maintenance script, not inside the trading process. Keeping
    this function as a no-op preserves the init call site while preventing
    poisoned transactions and surprise global DB changes during boot.
    """
    logger.debug(
        "Postgres routine-log configuration is managed outside the bot process"
    )


def init_postgres_schema() -> None:
    """Run pending Postgres migrations if Postgres is in use."""
    if config.DB_BACKEND in ("postgres", "dualwrite"):
        from src.data.db.postgres import get_postgres_config_error

        config_error = get_postgres_config_error(config.DB_BACKEND, config.POSTGRES_DSN)
        if config_error:
            if config.DB_BACKEND == "dualwrite":
                logger.warning(
                    "Dualwrite Postgres init skipped: %s SQLite will remain authoritative.",
                    config_error,
                )
                return
            raise RuntimeError(config_error)

        from src.data.db.migrations import run_migrations
        if config.DB_BACKEND == "dualwrite":
            try:
                run_migrations()
            except Exception as exc:
                logger.warning(
                    "Dualwrite Postgres init skipped because migrations could not run (%s). "
                    "SQLite will remain authoritative.",
                    exc,
                )
                return
        else:
            run_migrations()

        # Database server log policy is managed outside the bot process.
        _quiet_postgres_server_logs()
