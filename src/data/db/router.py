"""
Database backend router.

Routes connections based on ``config.DB_BACKEND``:

  - ``sqlite``    — all ops go to SQLite (default, current behaviour)
  - ``dualwrite`` — writes go to both SQLite and Postgres; reads from SQLite
  - ``postgres``  — all ops go to Postgres

The router exposes a single ``get_connection()`` context manager that
returns a :class:`ConnectionAdapter` the rest of ``database.py`` can use
without caring about the backend.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import config

from src.data.db.connection import ConnectionAdapter, DualWriteAdapter

logger = logging.getLogger(__name__)


# ─── SQLite helpers ─────────────────────────────────────────────

def _sqlite_connect() -> sqlite3.Connection:
    """Open a fresh SQLite connection with WAL + busy_timeout."""
    import shutil

    db_path = config.DB_PATH
    db_dir = os.path.dirname(os.path.abspath(db_path))
    min_free = max(1.0, float(os.environ.get("DB_MIN_FREE_MB", "100")))
    usage = shutil.disk_usage(db_dir)
    free_mb = usage.free / (1024 * 1024)
    if free_mb < min_free:
        raise RuntimeError(
            f"Insufficient disk space for DB: {free_mb:.1f}MB free "
            f"(minimum {min_free:.1f}MB)"
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ─── Postgres helpers ──────────────────────────────────────────

def _pg_connect():
    """Get a Postgres connection from the pool."""
    from src.data.db.postgres import get_connection
    return get_connection()


def _pg_return(conn):
    from src.data.db.postgres import return_connection
    return_connection(conn)


# ─── Unified context manager ───────────────────────────────────

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
        raw_sq = _sqlite_connect()
        pg_raw = None
        try:
            pg_raw = _pg_connect()
        except Exception as exc:
            # Rate-limit this warning to avoid log spam when PG is down
            if not hasattr(get_connection, "_pg_warn_ts") or \
               (time.time() - get_connection._pg_warn_ts) > 300:
                logger.warning(
                    "Dualwrite: could not obtain Postgres connection (%s) — "
                    "falling back to SQLite-only.", exc,
                )
                get_connection._pg_warn_ts = time.time()

        if pg_raw is not None:
            adapter = DualWriteAdapter(raw_sq, pg_raw)
            try:
                yield adapter
                raw_sq.commit()
                try:
                    pg_raw.commit()
                except Exception as exc:
                    logger.warning("Dualwrite Postgres commit failed: %s", exc)
                    try:
                        pg_raw.rollback()
                    except Exception:
                        pass
            except sqlite3.OperationalError as exc:
                logger.warning("SQLite operational error: %s", exc)
                raw_sq.rollback()
                try:
                    pg_raw.rollback()
                except Exception:
                    pass
                raise
            except Exception:
                raw_sq.rollback()
                try:
                    pg_raw.rollback()
                except Exception:
                    pass
                raise
            finally:
                raw_sq.close()
                _pg_return(pg_raw)
        else:
            # Postgres unavailable — degrade to SQLite-only
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
