"""
Postgres connection pool management.

Uses ``psycopg[binary]`` (psycopg 3) with a thread-safe connection pool.
The pool is lazily initialized on first use and shared process-wide.
"""
from __future__ import annotations

import atexit
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    """Lazily initialise the global Postgres connection pool."""
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool

        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        import config

        dsn = config.POSTGRES_DSN
        if not dsn:
            raise RuntimeError(
                "POSTGRES_DSN is required when DB_BACKEND is 'postgres' or 'dualwrite'. "
                "Set it to a connection string like: "
                "postgresql://user:pass@host:5432/dbname?sslmode=require"
            )

        try:
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row

            _pool = ConnectionPool(
                conninfo=dsn,
                min_size=config.POSTGRES_POOL_MIN,
                max_size=config.POSTGRES_POOL_MAX,
                kwargs={
                    "row_factory": dict_row,
                    "options": (
                        f"-c statement_timeout={config.POSTGRES_STATEMENT_TIMEOUT_MS} "
                        f"-c application_name={config.POSTGRES_APP_NAME}"
                    ),
                },
                open=True,
            )
            logger.info(
                "Postgres pool opened: min=%d max=%d app=%s",
                config.POSTGRES_POOL_MIN,
                config.POSTGRES_POOL_MAX,
                config.POSTGRES_APP_NAME,
            )
        except ImportError:
            raise RuntimeError(
                "psycopg[binary] and psycopg_pool are required for Postgres backend. "
                "Install with: pip install 'psycopg[binary]' psycopg_pool"
            )

        atexit.register(_close_pool)
        return _pool


def _close_pool():
    global _pool
    if _pool is not None:
        try:
            _pool.close()
            logger.info("Postgres pool closed.")
        except Exception as exc:
            logger.warning("Error closing Postgres pool: %s", exc)
        _pool = None


def get_connection():
    """Get a connection from the Postgres pool.

    Returns a psycopg connection.  The caller should use it as a context
    manager (``with``) or call ``.close()`` to return it to the pool.
    """
    pool = _get_pool()
    return pool.getconn()


def return_connection(conn):
    """Return a connection back to the pool."""
    pool = _get_pool()
    try:
        pool.putconn(conn)
    except Exception:
        pass


def check_health() -> bool:
    """Quick connectivity check for readiness probes."""
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            conn.execute("SELECT 1")
            conn.commit()
            return True
        finally:
            pool.putconn(conn)
    except Exception as exc:
        logger.warning("Postgres health check failed: %s", exc)
        return False
