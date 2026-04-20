"""
Database layer for persisting traders, strategies, scores, and paper trades.

Supports three backends selected by ``DB_BACKEND`` env var:

  - ``sqlite``    — local SQLite file (default, original behaviour)
  - ``dualwrite`` — writes to both SQLite and Postgres; reads from SQLite
  - ``postgres``  — Postgres only

Public API is unchanged — callers keep using ``get_connection()``,
``open_paper_trade()``, etc.
"""
import json
import os
import re
import shutil
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.core.env_utils import safe_env_float

logger = logging.getLogger(__name__)
_TRADER_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Resolved once at import — config.py already tested writability
_DB_PATH = config.DB_PATH
os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
_DB_MIN_FREE_MB = safe_env_float("DB_MIN_FREE_MB", 100.0, lo=1.0, hi=100_000.0)

# Import the router — it handles backend selection internally.
from src.data.db.router import (                       # noqa: E402
    get_connection as _routed_connection,
    is_postgres_active as _is_pg,
    init_postgres_schema,
)


def get_db_path():
    return _DB_PATH


def _is_valid_trader_address(address) -> bool:
    if not isinstance(address, str):
        return False
    return bool(_TRADER_ADDRESS_RE.match(address.strip()))


def _merge_quarantine_metadata(metadata, *, reason: str) -> dict:
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata or "{}")
        except Exception:
            metadata = {}
    metadata = dict(metadata or {})
    metadata["invalid_address_quarantined"] = True
    metadata["invalid_address_reason"] = str(reason or "malformed_eth_address")
    metadata["invalid_address_quarantined_at"] = datetime.now(timezone.utc).isoformat()
    return metadata


def _assert_db_disk_space() -> None:
    """Guard DB writes against low-disk conditions."""
    db_dir = os.path.dirname(os.path.abspath(_DB_PATH))
    usage = shutil.disk_usage(db_dir)
    free_mb = usage.free / (1024 * 1024)
    if free_mb < _DB_MIN_FREE_MB:
        raise RuntimeError(
            f"Insufficient disk space for DB operations: {free_mb:.1f}MB free "
            f"(minimum {_DB_MIN_FREE_MB:.1f}MB)"
        )


@contextmanager
def get_connection(*, for_read: bool = False):
    """Yield a connection for the active backend.

    In ``sqlite`` mode this behaves identically to the original implementation.
    In ``postgres`` or ``dualwrite`` mode the router transparently switches
    the underlying driver while keeping the same interface.
    """
    with _routed_connection(for_read=for_read) as conn:
        yield conn


def _insert_and_get_id(conn, sql: str, params):
    """Execute an insert and return the generated id on both backends."""
    if getattr(conn, "backend", "sqlite") == "postgres":
        cursor = conn.execute(sql.rstrip().rstrip(";") + " RETURNING id", params)
        row = cursor.fetchone()
        if not row:
            return None
        return row["id"] if isinstance(row, dict) else row[0]
    cursor = conn.execute(sql, params)
    return cursor.lastrowid


def table_exists(name: str) -> bool:
    """Check whether a table exists in the active backend."""
    with get_connection(for_read=True) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None


def init_db():
    """Create all tables if they don't exist.

    For Postgres, schema creation is handled by migrations (see
    ``migrations/0001_init_schema.sql``).  This function only runs
    the SQLite DDL when SQLite is the active backend.
    """
    if config.DB_BACKEND in ("postgres", "dualwrite"):
        # Postgres schema is managed by the migration runner.
        init_postgres_schema()

    if _is_pg():
        return

    # H5 (audit): pre-migrate an existing SQLite database to add the
    # ``client_order_id`` column before the main DDL runs.  The
    # ``CREATE UNIQUE INDEX ... ON paper_trades(client_order_id)``
    # statement inside the executescript would otherwise fail on a
    # long-lived DB because ``CREATE TABLE IF NOT EXISTS`` leaves the
    # old (column-less) paper_trades table untouched, and then the
    # index build references a non-existent column.  Running the
    # ALTER first means the index build always sees the new column.
    try:
        with get_connection() as conn:
            if table_exists("paper_trades"):
                try:
                    conn.execute(
                        "ALTER TABLE paper_trades ADD COLUMN client_order_id TEXT"
                    )
                except Exception:
                    # Column already exists — harmless.
                    pass
    except Exception as exc:
        logger.debug("paper_trades pre-migration skipped: %s", exc)

    with get_connection() as conn:
        conn.executescript("""
        -- Top traders we're tracking
        CREATE TABLE IF NOT EXISTS traders (
            address TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            total_pnl REAL DEFAULT 0,
            roi_pct REAL DEFAULT 0,
            account_value REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            metadata TEXT DEFAULT '{}'
        );

        -- Snapshots of trader positions over time
        CREATE TABLE IF NOT EXISTS position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trader_address TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            leverage REAL DEFAULT 1,
            unrealized_pnl REAL DEFAULT 0,
            margin_used REAL DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (trader_address) REFERENCES traders(address)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_trader ON position_snapshots(trader_address, timestamp);
        CREATE INDEX IF NOT EXISTS idx_snapshots_coin ON position_snapshots(coin, timestamp);

        -- Detected trading strategies
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            strategy_type TEXT NOT NULL,
            parameters TEXT DEFAULT '{}',
            discovered_at TEXT NOT NULL,
            last_scored TEXT,
            current_score REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            active INTEGER DEFAULT 1
        );

        -- Strategy performance scores over time (for self-improvement tracking)
        CREATE TABLE IF NOT EXISTS strategy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            score REAL NOT NULL,
            pnl_score REAL DEFAULT 0,
            win_rate_score REAL DEFAULT 0,
            sharpe_score REAL DEFAULT 0,
            consistency_score REAL DEFAULT 0,
            risk_adj_score REAL DEFAULT 0,
            notes TEXT DEFAULT '',
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );
        CREATE INDEX IF NOT EXISTS idx_scores_strategy ON strategy_scores(strategy_id, timestamp);

        -- Paper trading positions and history
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            size REAL NOT NULL,
            leverage REAL DEFAULT 1,
            pnl REAL DEFAULT 0,
            status TEXT DEFAULT 'open',
            stop_loss REAL,
            take_profit REAL,
            -- H5 (audit): idempotency key supplied by the caller so
            -- crash-retry and pipeline-level re-delivery cannot insert
            -- the same logical paper trade twice.  NULL = caller did
            -- not supply a key (always-insert legacy behavior).  The
            -- partial unique index below enforces dedup only for rows
            -- that opt in.
            client_order_id TEXT,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
        -- Partial index: hot dashboard/history query filters by
        -- status='closed' and orders by closed_at DESC.  Mirrors the
        -- Postgres migration 0007; on SQLite this is the WHERE clause
        -- form of a partial index and avoids sorting every closed row
        -- when the table grows past a few thousand entries.
        CREATE INDEX IF NOT EXISTS idx_paper_trades_closed_recent
            ON paper_trades(closed_at DESC)
            WHERE status = 'closed';
        -- H5: partial unique index on the idempotency key.  Existing
        -- rows with NULL client_order_id are unaffected; only opt-in
        -- keyed rows compete for uniqueness.  Mirrors the Postgres
        -- migration 0008.
        CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_trades_client_order_id
            ON paper_trades(client_order_id)
            WHERE client_order_id IS NOT NULL;

        -- Paper trading account state
        CREATE TABLE IF NOT EXISTS paper_account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL NOT NULL,
            total_pnl REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        );

        -- Research logs (what the bot discovered each cycle)
        CREATE TABLE IF NOT EXISTS research_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cycle_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            details TEXT DEFAULT '{}',
            traders_analyzed INTEGER DEFAULT 0,
            strategies_found INTEGER DEFAULT 0,
            strategies_updated INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS bot_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        -- Immutable audit trail: every trading action is logged here.
        -- INSERT-only table — rows are NEVER updated or deleted.
        -- Used for forensic analysis, compliance, and debugging.
        CREATE TABLE IF NOT EXISTS audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            coin TEXT,
            side TEXT,
            price REAL,
            size REAL,
            pnl REAL,
            source TEXT,
            details TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_trail(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_trail(action);
        CREATE INDEX IF NOT EXISTS idx_audit_coin ON audit_trail(coin);
        """)


# ─── Trader CRUD ───────────────────────────────────────────────

def upsert_trader(address, total_pnl=0, roi_pct=0, account_value=0,
                  win_rate=0, trade_count=0, metadata=None, is_active=True):
    now = datetime.now(timezone.utc).isoformat()
    metadata_dict = dict(metadata or {})
    active_value = bool(is_active)
    normalized_address = str(address or "").strip()
    if not _is_valid_trader_address(normalized_address):
        metadata_dict = _merge_quarantine_metadata(
            metadata_dict,
            reason="malformed_eth_address",
        )
        if active_value:
            logger.warning(
                "upsert_trader: quarantining malformed active trader address %s",
                normalized_address[:18] + "..." if len(normalized_address) > 18 else normalized_address,
            )
        active_value = False
    meta_json = json.dumps(metadata_dict)
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO traders (address, first_seen, last_updated, total_pnl,
                                 roi_pct, account_value, win_rate, trade_count, metadata, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
                last_updated = ?,
                total_pnl = ?,
                roi_pct = ?,
                account_value = ?,
                win_rate = ?,
                trade_count = ?,
                metadata = ?,
                active = ?
        """, (normalized_address, now, now, total_pnl, roi_pct, account_value,
              win_rate, trade_count, meta_json, active_value,
              now, total_pnl, roi_pct, account_value, win_rate, trade_count, meta_json, active_value))


def mark_trader_inactive(address):
    """Mark a trader as inactive (e.g. detected as bot)."""
    with get_connection() as conn:
        conn.execute("UPDATE traders SET active = ?, last_updated = ? WHERE address = ?",
                     (False, datetime.now(timezone.utc).isoformat(), address))


def quarantine_trader_address(address, reason="malformed_eth_address"):
    """Deactivate a trader row and annotate metadata with the quarantine reason."""
    normalized_address = str(address or "").strip()
    if not normalized_address:
        return False

    with get_connection() as conn:
        row = conn.execute(
            "SELECT metadata FROM traders WHERE address = ?",
            (normalized_address,),
        ).fetchone()
        if not row:
            return False
        metadata = _merge_quarantine_metadata(row["metadata"], reason=reason)
        conn.execute(
            "UPDATE traders SET active = ?, last_updated = ?, metadata = ? WHERE address = ?",
            (
                False,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata),
                normalized_address,
            ),
        )
    return True


def quarantine_invalid_traders() -> list[str]:
    """Deactivate malformed trader addresses persisted in the runtime DB."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT address, metadata, active FROM traders WHERE active = ?",
            (True,),
        ).fetchall()
        invalid = []
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            address = str(row["address"] or "").strip()
            if _is_valid_trader_address(address):
                continue
            metadata = _merge_quarantine_metadata(
                row["metadata"],
                reason="malformed_eth_address",
            )
            conn.execute(
                "UPDATE traders SET active = ?, last_updated = ?, metadata = ? WHERE address = ?",
                (False, now, json.dumps(metadata), address),
            )
            invalid.append(address)

    if invalid:
        logger.warning(
            "Quarantined %d malformed trader address(es) from live runtime state",
            len(invalid),
        )
    return invalid


def _get_sqlite_strategy_row(strategy_id):
    """Load a strategy row from the local SQLite runtime DB if it exists."""
    if strategy_id is None or not _DB_PATH or not os.path.exists(_DB_PATH):
        return None

    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM strategies WHERE id = ?",
                (strategy_id,),
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("Could not load strategy %s from SQLite fallback: %s", strategy_id, exc)
        return None

    return dict(row) if row else None


def _ensure_postgres_strategy_parent(strategy_id) -> None:
    """Backfill a missing strategy row into Postgres.

    This protects child writes such as ``strategy_scores`` and ``paper_trades``
    during cutover windows where SQLite may still contain the authoritative
    parent row but Postgres has not seen it yet.
    """
    if config.DB_BACKEND not in ("dualwrite", "postgres") or strategy_id is None:
        return

    try:
        from src.data.db.postgres import get_connection as get_pg_connection
        from src.data.db.postgres import return_connection as return_pg_connection
    except Exception:
        return

    strategy = _get_sqlite_strategy_row(strategy_id)
    if not strategy and config.DB_BACKEND == "dualwrite":
        with get_connection(for_read=True) as sqlite_conn:
            row = sqlite_conn.execute(
                "SELECT * FROM strategies WHERE id = ?",
                (strategy_id,),
            ).fetchone()
        strategy = dict(row) if row else None

    if not strategy:
        return

    pg_conn = get_pg_connection()
    try:
        cur = pg_conn.cursor()
        cur.execute("SELECT 1 FROM strategies WHERE id = %s", (strategy_id,))
        if cur.fetchone():
            pg_conn.commit()
            return

        cur.execute(
            """
            INSERT INTO strategies
            (id, name, description, strategy_type, parameters, discovered_at,
             last_scored, current_score, total_pnl, trade_count, win_rate,
             sharpe_ratio, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                strategy_type = EXCLUDED.strategy_type,
                parameters = EXCLUDED.parameters,
                discovered_at = EXCLUDED.discovered_at,
                last_scored = EXCLUDED.last_scored,
                current_score = EXCLUDED.current_score,
                total_pnl = EXCLUDED.total_pnl,
                trade_count = EXCLUDED.trade_count,
                win_rate = EXCLUDED.win_rate,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                active = EXCLUDED.active
            """,
            (
                strategy["id"],
                strategy["name"],
                strategy["description"],
                strategy["strategy_type"],
                strategy["parameters"],
                strategy["discovered_at"],
                strategy["last_scored"],
                strategy["current_score"],
                strategy["total_pnl"],
                strategy["trade_count"],
                strategy["win_rate"],
                strategy["sharpe_ratio"],
                bool(strategy["active"]),
            ),
        )
        cur.execute(
            """
            SELECT setval(
                pg_get_serial_sequence('strategies', 'id'),
                GREATEST((SELECT COALESCE(MAX(id), 1) FROM strategies), %s),
                true
            )
            """,
            (strategy["id"],),
        )
        pg_conn.commit()
    except Exception as exc:
        try:
            pg_conn.rollback()
        except Exception:
            pass
        logger.debug("Could not backfill strategy %s into Postgres: %s", strategy_id, exc)
    finally:
        return_pg_connection(pg_conn)


def get_active_traders(*, valid_only: bool = False, quarantine_invalid: bool = False):
    if quarantine_invalid:
        quarantine_invalid_traders()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM traders WHERE active = ? ORDER BY total_pnl DESC",
            (True,),
        ).fetchall()
    traders = [dict(r) for r in rows]
    if valid_only:
        traders = [t for t in traders if _is_valid_trader_address(t.get("address"))]
    return traders


def get_trader(address):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM traders WHERE address = ?", (address,)).fetchone()
    return dict(row) if row else None


def get_known_bot_addresses() -> set:
    """
    Get all addresses previously detected as bots (active=0).
    Used by trader_discovery to skip known bots entirely on subsequent scans,
    persisting across redeploys since the data lives in SQLite.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT address FROM traders WHERE active = ?",
            (False,),
        ).fetchall()
    return {r["address"] for r in rows}


def get_all_traders_including_bots():
    """Get ALL traders (active and inactive) for backup purposes."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM traders ORDER BY total_pnl DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Position Snapshots ───────────────────────────────────────

def save_position_snapshot(trader_address, coin, side, size, entry_price,
                           leverage=1, unrealized_pnl=0, margin_used=0, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO position_snapshots
            (trader_address, timestamp, coin, side, size, entry_price,
             leverage, unrealized_pnl, margin_used, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trader_address, now, coin, side, size, entry_price,
              leverage, unrealized_pnl, margin_used, json.dumps(metadata or {})))


def get_trader_position_history(trader_address, limit=100):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM position_snapshots
            WHERE trader_address = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (trader_address, limit)).fetchall()
    return [dict(r) for r in rows]


# ─── Strategy CRUD ─────────────────────────────────────────────

def save_strategy(name, description, strategy_type, parameters=None,
                  total_pnl=0, trade_count=0, win_rate=0, sharpe_ratio=0):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        return _insert_and_get_id(conn, """
            INSERT INTO strategies
            (name, description, strategy_type, parameters, discovered_at,
             total_pnl, trade_count, win_rate, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, description, strategy_type, json.dumps(parameters or {}),
              now, total_pnl, trade_count, win_rate, sharpe_ratio))


def save_strategies_batch(strategies_data):
    """Batch insert multiple strategies in a single transaction."""
    now = datetime.now(timezone.utc).isoformat()
    saved_ids = []
    with get_connection() as conn:
        for s in strategies_data:
            saved_ids.append(_insert_and_get_id(conn, """
                INSERT INTO strategies
                (name, description, strategy_type, parameters, discovered_at,
                 total_pnl, trade_count, win_rate, sharpe_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (s["name"], s["description"], s["strategy_type"],
                  json.dumps(s.get("parameters") or {}),
                  now, s.get("total_pnl", 0), s.get("trade_count", 0),
                  s.get("win_rate", 0), s.get("sharpe_ratio", 0))))
    return saved_ids


def update_strategy_score(strategy_id, score):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("""
            UPDATE strategies SET current_score = ?, last_scored = ? WHERE id = ?
        """, (score, now, strategy_id))


def get_active_strategies():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE active = ? ORDER BY current_score DESC",
            (True,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_strategy(strategy_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    return dict(row) if row else None


# ─── Strategy Scores ──────────────────────────────────────────

def save_strategy_score(strategy_id, score, pnl_score=0, win_rate_score=0,
                        sharpe_score=0, consistency_score=0, risk_adj_score=0, notes=""):
    now = datetime.now(timezone.utc).isoformat()
    _ensure_postgres_strategy_parent(strategy_id)
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO strategy_scores
            (strategy_id, timestamp, score, pnl_score, win_rate_score,
             sharpe_score, consistency_score, risk_adj_score, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (strategy_id, now, score, pnl_score, win_rate_score,
              sharpe_score, consistency_score, risk_adj_score, notes))


def get_strategy_score_history(strategy_id, limit=30):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM strategy_scores
            WHERE strategy_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (strategy_id, limit)).fetchall()
    return [dict(r) for r in rows]


# ─── Bot State (generic KV) ───────────────────────────────────

def get_bot_state(key: str, default=None):
    """Read a JSON-serialised value from the bot_state KV table.

    Returns the decoded value, or ``default`` if the key is missing or the
    stored payload cannot be decoded.  Safe to call before start-up
    migrations have run — a missing table is treated as "no value".
    """
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key = ?", (str(key),)
            ).fetchone()
    except Exception as exc:
        logger.debug("get_bot_state(%s): read failed: %s", key, exc)
        return default
    if not row:
        return default
    try:
        raw = row["value"] if hasattr(row, "keys") else row[0]
    except (KeyError, IndexError, TypeError):
        return default
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def set_bot_state(key: str, value) -> bool:
    """Upsert a JSON-serialisable value into the bot_state KV table.

    Returns True on success, False on any error (writes are best-effort and
    must never break the hot path of the caller).
    """
    try:
        payload = json.dumps(value)
    except (TypeError, ValueError) as exc:
        logger.warning("set_bot_state(%s): value not JSON-serialisable: %s", key, exc)
        return False
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO bot_state (key, value) VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (str(key), payload))
        return True
    except Exception as exc:
        logger.warning("set_bot_state(%s): write failed: %s", key, exc)
        return False


# ─── Paper Trading ─────────────────────────────────────────────

def init_paper_account(balance):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO paper_account (id, balance, total_pnl, total_trades, winning_trades, last_updated)
            VALUES (?, ?, 0, 0, 0, ?)
            ON CONFLICT (id) DO UPDATE SET
                balance = EXCLUDED.balance,
                total_pnl = 0,
                total_trades = 0,
                winning_trades = 0,
                last_updated = EXCLUDED.last_updated
        """, (1, balance, now))


def get_paper_account():
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    return dict(row) if row else None


def update_paper_account(balance, total_pnl, total_trades, winning_trades):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE paper_account
            SET balance = ?, total_pnl = ?, total_trades = ?, winning_trades = ?, last_updated = ?
            WHERE id = 1
        """, (balance, total_pnl, total_trades, winning_trades, now))
        if cursor.rowcount == 0:
            raise LookupError("paper_account singleton row (id=1) does not exist")


def open_paper_trade(strategy_id, coin, side, entry_price, size, leverage=1,
                     stop_loss=None, take_profit=None, metadata=None,
                     idempotency_key: Optional[str] = None):
    """Insert a paper trade row and return the row id.

    H5 (audit): when ``idempotency_key`` is supplied, we first look up
    an existing row by ``client_order_id``.  If one is found (e.g. the
    caller is retrying after a crash/network blip) we return the
    existing id instead of inserting a duplicate.  The unique partial
    index on ``client_order_id`` is the authoritative backstop — a
    concurrent racer that slipped past our SELECT still raises, and we
    translate that into a re-lookup so the caller never sees a phantom
    duplicate.

    When ``idempotency_key`` is None, behavior is unchanged: every call
    inserts a new row (legacy caller contract preserved).
    """
    now = datetime.now(timezone.utc).isoformat()
    _ensure_postgres_strategy_parent(strategy_id)

    key = None
    if idempotency_key is not None:
        # Normalize to a non-empty string.  Empty / whitespace-only
        # values are treated as "no key supplied" so callers that build
        # keys from optional metadata can't accidentally collide every
        # row on the empty string.
        key = str(idempotency_key).strip() or None

    with get_connection() as conn:
        if key is not None:
            existing = conn.execute(
                "SELECT id FROM paper_trades WHERE client_order_id = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                logger.info(
                    "open_paper_trade: idempotent replay for key=%s -> trade_id=%s",
                    key[:40], existing["id"],
                )
                return int(existing["id"])

        try:
            return _insert_and_get_id(conn, """
                INSERT INTO paper_trades
                (strategy_id, opened_at, coin, side, entry_price, size, leverage,
                 stop_loss, take_profit, status, client_order_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
            """, (strategy_id, now, coin, side, entry_price, size, leverage,
                  stop_loss, take_profit, key, json.dumps(metadata or {})))
        except Exception:
            if key is None:
                raise
            # H5: when the unique-partial-index races us, re-lookup and
            # return the winner's id.  ``IntegrityError`` / psycopg
            # ``UniqueViolation`` both surface as generic Exception from
            # the adapter, so we narrow by retrying the select.
            existing = conn.execute(
                "SELECT id FROM paper_trades WHERE client_order_id = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                logger.info(
                    "open_paper_trade: insert race resolved by key=%s -> trade_id=%s",
                    key[:40], existing["id"],
                )
                return int(existing["id"])
            raise


# ─────────────────────────────────────────────────────────────────
# AUDIT M3 — optimistic locking for paper_trade metadata RMW.
#
# Previously ``update_paper_trade_metadata`` did a plain SELECT-merge-UPDATE
# sequence, which loses writes under concurrent callers (e.g. a funding
# accrual thread and a close handler racing on the same trade id).
#
# Fix is two-layered:
#   (1) A per-trade-id process-local lock serializes RMW within a single
#       Python process — covers the common "multiple trading-cycle
#       threads" case without schema changes.
#   (2) A compare-and-swap on the raw metadata column catches races from
#       a second process that talks to the same DB (e.g. a stress-test
#       worker, the dashboard, dualwrite mirror).  The UPDATE only
#       commits when the stored metadata still matches what we SELECT-ed;
#       on miss we re-read, re-merge and retry up to ``max_retries``.
#
# Cross-process safety still depends on the underlying engine honoring
# row-level visibility:
#   - SQLite:   the journal/WAL serializes writers, so a concurrent
#               writer's commit is visible to the next SELECT.
#   - Postgres: JSONB stringifies deterministically for equality against
#               a TEXT parameter because we compare via ``::text``; in
#               dualwrite both backends see the same update path.
# ─────────────────────────────────────────────────────────────────

_PAPER_TRADE_METADATA_LOCKS: Dict[int, threading.Lock] = {}
_PAPER_TRADE_METADATA_LOCKS_GUARD = threading.Lock()


def _get_paper_trade_metadata_lock(trade_id: int) -> threading.Lock:
    """Return the per-trade-id lock used to serialize in-process RMW."""
    with _PAPER_TRADE_METADATA_LOCKS_GUARD:
        lock = _PAPER_TRADE_METADATA_LOCKS.get(trade_id)
        if lock is None:
            lock = threading.Lock()
            _PAPER_TRADE_METADATA_LOCKS[trade_id] = lock
        return lock


def update_paper_trade_metadata(
    trade_id: int, extra: dict, *, max_retries: int = 5
) -> None:
    """Merge ``extra`` keys into a paper trade's metadata JSON blob.

    AUDIT M3 — race-safe version.  The old implementation did a plain
    SELECT-then-UPDATE which lost writes under concurrent callers.  The
    merged value is now committed with compare-and-swap semantics: if
    another writer has changed the metadata between our SELECT and our
    UPDATE, the UPDATE matches zero rows and we re-read/re-merge up to
    ``max_retries`` times before raising :class:`RuntimeError`.

    Additionally a process-local per-trade lock serializes RMW within
    this Python process so concurrent threads do not stampede the CAS.
    """
    lock = _get_paper_trade_metadata_lock(trade_id)
    with lock:
        last_seen: Optional[str] = None
        for attempt in range(max_retries):
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT metadata FROM paper_trades WHERE id = ?", (trade_id,)
                ).fetchone()
                if not row:
                    raise LookupError(f"Paper trade {trade_id} does not exist")

                raw_metadata = row["metadata"]
                # Normalize to a string for the CAS predicate.  ``row``
                # may surface JSONB objects as dict on postgres; cast to
                # the same on-wire form we'll write back.
                if isinstance(raw_metadata, (dict, list)):
                    cas_current = json.dumps(raw_metadata)
                elif raw_metadata is None:
                    cas_current = ""
                else:
                    cas_current = str(raw_metadata)

                try:
                    existing = json.loads(cas_current or "{}")
                except Exception:
                    existing = {}
                existing.update(extra)
                new_metadata_str = json.dumps(existing)

                # CAS predicate works on both sqlite (metadata is TEXT)
                # and postgres (cast metadata::text for comparison).  The
                # COALESCE guards the NULL/empty-string boundary.
                backend = getattr(conn, "backend", "sqlite")
                if backend == "postgres":
                    cursor = conn.execute(
                        "UPDATE paper_trades SET metadata = ?::jsonb "
                        "WHERE id = ? AND COALESCE(metadata::text, '') = COALESCE(?, '')",
                        (new_metadata_str, trade_id, cas_current),
                    )
                else:
                    cursor = conn.execute(
                        "UPDATE paper_trades SET metadata = ? "
                        "WHERE id = ? AND COALESCE(metadata, '') = COALESCE(?, '')",
                        (new_metadata_str, trade_id, cas_current),
                    )
                if cursor.rowcount == 1:
                    return
            # CAS failed — another writer raced us.  Short exponential
            # backoff before retrying the read-merge-CAS cycle.
            last_seen = cas_current
            time.sleep(0.001 * (2 ** attempt))
        raise RuntimeError(
            f"update_paper_trade_metadata(trade_id={trade_id}) CAS failed after "
            f"{max_retries} retries; last seen metadata: {last_seen!r}"
        )


def close_paper_trade(trade_id, exit_price, pnl) -> bool:
    """Close a paper trade.  Returns True on success, False if trade_id not found.

    CRIT-FIX CRIT-5: check rowcount — if the UPDATE matches 0 rows the trade was
    already closed or the ID is wrong.  The caller MUST check the return value and
    skip the account PnL credit to prevent phantom double-credit.

    NOTE: Prefer :func:`close_paper_trade_and_credit_account` for the common
    close+credit pattern — it performs both operations in a single transaction,
    eliminating the race window where a crash between close and credit would
    desync the account balance from the trades table.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE paper_trades SET closed_at = ?, exit_price = ?, pnl = ?, status = 'closed'
            WHERE id = ? AND status = 'open'
        """, (now, exit_price, pnl, trade_id))
        if cursor.rowcount == 0:
            logger.error(
                "close_paper_trade: trade_id=%s matched 0 open rows -- "
                "possible double-close or missing record. PnL NOT credited.",
                trade_id,
            )
            return False
    return True


def close_paper_trade_and_credit_account(trade_id, exit_price, pnl) -> bool:
    """Atomically close a paper trade AND credit the paper_account balance.

    Runs both UPDATEs inside a single transaction.  If the trade is already
    closed (rowcount == 0 on the first UPDATE) the transaction is rolled back
    and the account is NOT credited — this is the key invariant that prevents
    phantom double-credits on retries.

    Returns True if both updates succeeded, False if the trade was already
    closed / not found (in which case nothing was credited).

    CRIT-FIX C2: replaces the prior two-statement pattern
    ``close_paper_trade() -> update_paper_account()`` which was non-atomic —
    a crash between the two calls could leave the account stale or, on retry
    of the caller, double-credit the PnL.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        try:
            cursor = conn.execute("""
                UPDATE paper_trades SET closed_at = ?, exit_price = ?, pnl = ?, status = 'closed'
                WHERE id = ? AND status = 'open'
            """, (now, exit_price, pnl, trade_id))
            if cursor.rowcount == 0:
                try:
                    conn.rollback()
                except Exception:
                    pass
                logger.error(
                    "close_paper_trade_and_credit_account: trade_id=%s matched "
                    "0 open rows -- possible double-close. PnL NOT credited.",
                    trade_id,
                )
                return False

            # D10: Use a single relative UPDATE instead of SELECT-then-UPDATE.
            # Even inside a transaction the prior pattern left a read/modify/
            # write sequence that any concurrent writer (dashboard admin
            # console, reconciliation job) could interleave against.  An
            # atomic "balance = balance + ?" form serialises cleanly and also
            # means a failed commit never leaves stale absolute values that
            # were computed from a now-outdated read.
            pnl_delta = float(pnl)
            win_delta = 1 if pnl_delta > 0 else 0
            acct_cursor = conn.execute("""
                UPDATE paper_account
                SET balance = COALESCE(balance, 0) + ?,
                    total_pnl = COALESCE(total_pnl, 0) + ?,
                    total_trades = COALESCE(total_trades, 0) + 1,
                    winning_trades = COALESCE(winning_trades, 0) + ?,
                    last_updated = ?
                WHERE id = 1
            """, (pnl_delta, pnl_delta, win_delta, now))
            if acct_cursor.rowcount == 0:
                try:
                    conn.rollback()
                except Exception:
                    pass
                logger.error(
                    "close_paper_trade_and_credit_account: paper_account UPDATE "
                    "matched 0 rows; rolling back trade_id=%s", trade_id,
                )
                return False
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
    return True


def get_open_paper_trades():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status = 'open'"
        ).fetchall()
    return [dict(r) for r in rows]


def get_paper_trade_history(limit=100):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM paper_trades WHERE status = 'closed'
            ORDER BY closed_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def reset_paper_trades(initial_balance: float = None):
    """
    Wipe all paper trades and reset the paper account to fresh state.
    Returns summary of what was deleted.
    """
    if initial_balance is None:
        initial_balance = 10_000.0

    with get_connection() as conn:
        open_count = conn.execute(
            "SELECT COUNT(*) as c FROM paper_trades WHERE status = 'open'"
        ).fetchone()["c"]
        closed_count = conn.execute(
            "SELECT COUNT(*) as c FROM paper_trades WHERE status = 'closed'"
        ).fetchone()["c"]

        conn.execute("DELETE FROM paper_trades")
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO paper_account (id, balance, total_pnl, total_trades, winning_trades, last_updated)
            VALUES (?, ?, 0, 0, 0, ?)
            ON CONFLICT (id) DO UPDATE SET
                balance = EXCLUDED.balance,
                total_pnl = 0,
                total_trades = 0,
                winning_trades = 0,
                last_updated = EXCLUDED.last_updated
        """, (1, initial_balance, now))

    logger.info(f"Paper trades reset: cleared {open_count} open + {closed_count} closed trades, "
               f"balance reset to ${initial_balance:,.2f}")
    return {
        "open_deleted": open_count,
        "closed_deleted": closed_count,
        "new_balance": initial_balance,
    }


# ─── Research Logs ─────────────────────────────────────────────

def log_research_cycle(cycle_type, summary, details=None,
                       traders_analyzed=0, strategies_found=0, strategies_updated=0):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO research_logs
            (timestamp, cycle_type, summary, details, traders_analyzed,
             strategies_found, strategies_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (now, cycle_type, summary, json.dumps(details or {}),
              traders_analyzed, strategies_found, strategies_updated))


# ─── Audit Trail (immutable trade journal) ────────────────────

def audit_log(action: str, coin: str = None, side: str = None,
              price: float = None, size: float = None, pnl: float = None,
              source: str = None, details: dict = None):
    """
    Append an immutable audit record. This table is INSERT-ONLY.
    Every trade signal, execution, rejection, and error gets logged here
    for forensic analysis and compliance.

    Actions: signal_generated, signal_approved, signal_rejected,
             trade_opened, trade_closed, stop_loss_hit, take_profit_hit,
             circuit_breaker_triggered, rate_limit_hit, websocket_reconnect,
             golden_wallet_connected, bot_detected, error
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO audit_trail (timestamp, action, coin, side, price, size, pnl, source, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, action, coin, side, price, size, pnl, source,
              json.dumps(details or {})))


def get_audit_trail(limit: int = 200, action_filter: str = None,
                    coin_filter: str = None) -> list:
    """Query the audit trail with optional filters."""
    with get_connection() as conn:
        query = "SELECT * FROM audit_trail WHERE 1=1"
        params = []
        if action_filter:
            query += " AND action = ?"
            params.append(action_filter)
        if coin_filter:
            query += " AND coin = ?"
            params.append(coin_filter)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ─── Backup & Restore (for Railway persistence) ─────────────

def backup_to_json(filepath: str = None):
    """
    Backup critical DB state to a JSON file for Railway persistence.

    Includes golden wallets + wallet_fills so the expensive research
    data survives Railway redeploys without re-scanning.

    Set HL_BOT_BACKUP env var to a Railway volume path (e.g. /data/bot_backup.json)
    so it persists across container restarts.
    """
    if filepath is None:
        # Put backup next to the DB file (same volume / same dir)
        db_dir = os.path.dirname(os.path.abspath(_DB_PATH))
        filepath = os.environ.get("HL_BOT_BACKUP",
                                   os.path.join(db_dir, "bot_backup.json"))

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    try:
        data = {
            "version": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "paper_account": get_paper_account(),
            "traders": get_active_traders()[:200],
            "bot_traders": [t for t in get_all_traders_including_bots() if not t.get("active", 1)],
            "strategies": get_active_strategies()[:500],
            "open_trades": get_open_paper_trades(),
            "closed_trades": get_paper_trade_history(limit=500),
        }

        # Golden wallets + fills (the most expensive data to regenerate)
        try:
            with get_connection() as conn:
                if table_exists("golden_wallets"):
                    rows = conn.execute(
                        "SELECT * FROM golden_wallets ORDER BY penalised_pnl DESC"
                    ).fetchall()
                    data["golden_wallets"] = [dict(r) for r in rows]

                if table_exists("wallet_fills"):
                    # Only backup fills from golden wallets (not all fills)
                    rows = conn.execute("""
                        SELECT wf.* FROM wallet_fills wf
                        JOIN golden_wallets gw ON wf.wallet_address = gw.address
                        WHERE gw.is_golden = 1
                        ORDER BY wf.time_ms
                    """).fetchall()
                    data["wallet_fills"] = [dict(r) for r in rows]

                if table_exists("calibration_records"):
                    rows = conn.execute(
                        "SELECT * FROM calibration_records ORDER BY timestamp DESC LIMIT 100"
                    ).fetchall()
                    data["calibration_records"] = [dict(r) for r in rows]
        except Exception as e:
            print(f"Warning: could not backup golden/calibration data: {e}")

        # Atomic write: temp file + os.replace so a crash mid-write cannot
        # corrupt the backup file that restore_from_json() reads on fresh
        # deploys.  Truncated bot_backup.json has caused real data loss.
        tmp_path = f"{filepath}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # fsync not supported on some FS; best-effort only
        os.replace(tmp_path, filepath)

        size_kb = os.path.getsize(filepath) / 1024
        counts = (f"{len(data.get('traders', []))} traders, "
                  f"{len(data.get('bot_traders', []))} bots, "
                  f"{len(data.get('strategies', []))} strategies, "
                  f"{len(data.get('golden_wallets', []))} golden wallets, "
                  f"{len(data.get('wallet_fills', []))} fills")
        print(f"DB backup ({size_kb:.0f} KB): {counts} -> {filepath}")
    except Exception as e:
        print(f"Backup failed: {e}")


def restore_from_json(filepath: str = None):
    """
    Restore DB state from a backup JSON file if DB is empty.

    Includes golden wallets + wallet_fills so the expensive research
    survives Railway redeploys without a full re-scan.
    """
    if filepath is None:
        # Put backup next to the DB file (same volume / same dir)
        db_dir = os.path.dirname(os.path.abspath(_DB_PATH))
        filepath = os.environ.get("HL_BOT_BACKUP",
                                   os.path.join(db_dir, "bot_backup.json"))

    if not os.path.exists(filepath):
        return False

    # Only restore if DB is empty (fresh deploy)
    account = get_paper_account()
    if account:
        return False  # DB already has data

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"Restoring from backup ({data.get('timestamp', '?')})...")

        # Restore paper account
        if data.get("paper_account"):
            acc = data["paper_account"]
            init_paper_account(acc.get("balance", 10000))
            update_paper_account(
                acc.get("balance", 10000),
                acc.get("total_pnl", 0),
                acc.get("total_trades", 0),
                acc.get("winning_trades", 0),
            )

        # Restore active traders
        for t in data.get("traders", []):
            upsert_trader(
                t["address"],
                total_pnl=t.get("total_pnl", 0),
                roi_pct=t.get("roi_pct", 0),
                account_value=t.get("account_value", 0),
                win_rate=t.get("win_rate", 0),
                trade_count=t.get("trade_count", 0),
            )

        # Restore bot traders (so they stay skipped across redeploys)
        for t in data.get("bot_traders", []):
            meta = t.get("metadata", "{}")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            upsert_trader(
                t["address"],
                total_pnl=t.get("total_pnl", 0),
                roi_pct=t.get("roi_pct", 0),
                account_value=t.get("account_value", 0),
                win_rate=t.get("win_rate", 0),
                trade_count=t.get("trade_count", 0),
                metadata=meta,
                is_active=False,
            )

        # Restore strategies
        for s in data.get("strategies", []):
            save_strategy(
                s.get("name", "restored"),
                s.get("description", ""),
                s.get("strategy_type", "unknown"),
                parameters=json.loads(s["parameters"]) if isinstance(s.get("parameters"), str) else s.get("parameters"),
                total_pnl=s.get("total_pnl", 0),
                trade_count=s.get("trade_count", 0),
                win_rate=s.get("win_rate", 0),
                sharpe_ratio=s.get("sharpe_ratio", 0),
            )

        # Restore golden wallets (v2 backup)
        golden_count = 0
        golden_failures = 0
        fills_count = 0
        if data.get("golden_wallets"):
            try:
                from src.discovery.golden_wallet import init_golden_tables
                init_golden_tables()

                _ignore_sql = (
                    "INSERT INTO golden_wallets "
                    "(address, penalised_pnl, raw_pnl, sharpe_ratio, "
                    "max_drawdown_pct, penalised_max_drawdown_pct, "
                    "win_rate, trades_per_day, is_golden, coins_traded, "
                    "best_coin, evaluated_at, connected_to_live) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (address) DO NOTHING"
                )
                with get_connection() as conn:
                    for gw in data["golden_wallets"]:
                        try:
                            conn.execute(_ignore_sql, (
                                gw["address"],
                                gw.get("penalised_pnl", 0),
                                gw.get("raw_pnl", 0),
                                gw.get("sharpe_ratio", 0),
                                gw.get("max_drawdown_pct", 0),
                                gw.get("penalised_max_drawdown_pct", 0),
                                gw.get("win_rate", 0),
                                gw.get("trades_per_day", 0),
                                gw.get("is_golden", 0),
                                gw.get("coins_traded", ""),
                                gw.get("best_coin", ""),
                                gw.get("evaluated_at", datetime.now(timezone.utc).isoformat()),
                                gw.get("connected_to_live", 0),
                            ))
                            golden_count += 1
                        except Exception as e:
                            golden_failures += 1
                            if golden_failures <= 3:
                                logger.warning(
                                    "Restore golden_wallets row failed for %s: %s",
                                    gw.get("address", "?"),
                                    e,
                                )
            except Exception as e:
                print(f"Warning: could not restore golden wallets: {e}")

        # Restore wallet fills (v2 backup)
        # Schema: wallet_address, coin, side, original_price, penalised_price,
        #         size, time_ms, delayed_time_ms, closed_pnl, penalised_pnl,
        #         fee, is_liquidation, direction
        fills_failures = 0
        if data.get("wallet_fills"):
            try:
                with get_connection() as conn:
                    for fill in data["wallet_fills"]:
                        try:
                            conn.execute("""
                                INSERT INTO wallet_fills
                                (wallet_address, coin, side, original_price,
                                 penalised_price, size, time_ms, delayed_time_ms,
                                 closed_pnl, penalised_pnl, fee, is_liquidation,
                                 direction)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT DO NOTHING
                            """, (
                                fill["wallet_address"],
                                fill.get("coin", ""),
                                fill.get("side", ""),
                                fill.get("original_price", 0),
                                fill.get("penalised_price", 0),
                                fill.get("size", 0),
                                fill.get("time_ms", 0),
                                fill.get("delayed_time_ms", 0),
                                fill.get("closed_pnl", 0),
                                fill.get("penalised_pnl", 0),
                                fill.get("fee", 0),
                                fill.get("is_liquidation", 0),
                                fill.get("direction", ""),
                            ))
                            fills_count += 1
                        except Exception as e:
                            fills_failures += 1
                            if fills_failures <= 3:
                                logger.warning(
                                    "Restore wallet_fills row failed for %s %s: %s",
                                    fill.get("wallet_address", "?"),
                                    fill.get("coin", "?"),
                                    e,
                                )
            except Exception as e:
                print(f"Warning: could not restore wallet fills: {e}")

        if golden_failures:
            logger.warning("Restore golden_wallets: %d rows failed", golden_failures)
        if fills_failures:
            logger.warning("Restore wallet_fills: %d rows failed", fills_failures)

        print(f"Restored DB from backup: {len(data.get('traders', []))} traders, "
              f"{len(data.get('bot_traders', []))} bots, "
              f"{len(data.get('strategies', []))} strategies, "
              f"{golden_count} golden wallets, {fills_count} fills")
        return True
    except Exception as e:
        print(f"Restore failed: {e}")
        return False


# ─── Backend-aware helpers for modules migrating off raw sqlite3 ──

def get_backend_name() -> str:
    """Return the active backend name: 'sqlite', 'dualwrite', or 'postgres'."""
    return config.DB_BACKEND


def get_dualwrite_stats() -> dict:
    """Return dual-write health counters.

    Returns a dict with keys:
      - ``pg_writes_ok``      — successful Postgres mirror writes
      - ``pg_writes_failed``  — failed Postgres mirror writes
      - ``pg_last_error``     — last error message (truncated)
      - ``pg_last_error_ts``  — timestamp of last error
      - ``recent_failures_5m`` — failures observed in the last 5 minutes

    Returns an empty dict if dual-write is not active.
    """
    if config.DB_BACKEND != "dualwrite":
        return {}
    from src.data.db.connection import dualwrite_stats
    return dualwrite_stats.snapshot()


def dualwrite_is_healthy(
    *, window_s: float = 300.0, max_failures: int = 5,
) -> bool:
    """Return True when the dualwrite Postgres mirror is keeping up.

    H4 (audit): readiness checks must surface sustained Postgres mirror
    failures to the live trader so scaling decisions don't ride on a
    silently-broken audit ledger.  This wrapper is backend-aware: for
    any backend other than ``dualwrite`` it short-circuits to True, so
    SQLite-only and pure-Postgres deployments are unaffected.

    Arguments match :func:`src.data.db.connection.dualwrite_is_healthy`.
    """
    if config.DB_BACKEND != "dualwrite":
        return True
    from src.data.db.connection import dualwrite_is_healthy as _dw_healthy
    return _dw_healthy(window_s=window_s, max_failures=max_failures)


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {get_db_path()}")
