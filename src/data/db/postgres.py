"""
Postgres connection pool management.

Uses ``psycopg[binary]`` (psycopg 3) with a thread-safe connection pool.
The pool is lazily initialized on first use and shared process-wide.
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import sys
import threading
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()
_LOCAL_POSTGRES_HOSTS = {"localhost", "127.0.0.1", "::1"}
_HOST_KV_RE = re.compile(r"(?:^|\s)host=(?P<host>[^\s]+)")


def _is_hosted_runtime() -> bool:
    """Best-effort detection for managed deployments like Railway."""
    hosted_markers = (
        "RAILWAY_ENVIRONMENT_ID",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RENDER",
        "K_SERVICE",
        "FLY_APP_NAME",
    )
    return any(os.environ.get(marker) for marker in hosted_markers)


def _extract_dsn_host(dsn: str) -> str:
    """Return the configured Postgres host from a DSN or libpq string."""
    dsn = (dsn or "").strip()
    if not dsn:
        return ""

    if "://" in dsn:
        try:
            parsed = urlparse(dsn)
            return (parsed.hostname or "").strip("[]").lower()
        except ValueError:
            return ""

    match = _HOST_KV_RE.search(dsn)
    if not match:
        return ""
    return match.group("host").strip("[]'\"").lower()


def get_postgres_config_error(backend: str, dsn: str) -> str:
    """Return a human-readable config error, or an empty string if valid."""
    if backend not in {"postgres", "dualwrite"}:
        return ""

    dsn = (dsn or "").strip()
    if not dsn:
        return (
            "POSTGRES_DSN is required when DB_BACKEND is 'postgres' or 'dualwrite'. "
            "Set it to a managed Postgres connection string such as "
            "postgresql://user:pass@host:5432/dbname?sslmode=require"
        )

    host = _extract_dsn_host(dsn)
    if _is_hosted_runtime() and host in _LOCAL_POSTGRES_HOSTS:
        return (
            "POSTGRES_DSN points to localhost inside a hosted deployment. "
            "Use the managed Postgres DATABASE_URL for this environment instead "
            "of a local sample DSN."
        )

    return ""


def _get_pool():
    """Lazily initialise the global Postgres connection pool."""
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        import config

        dsn = config.POSTGRES_DSN
        config_error = get_postgres_config_error(config.DB_BACKEND, dsn)
        if config_error:
            raise RuntimeError(config_error)

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
        try:
            conn.rollback()
        except Exception:
            pass
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
