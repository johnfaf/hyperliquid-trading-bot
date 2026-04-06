"""
SQLite database layer for persisting traders, strategies, scores, and paper trades.
"""
import sqlite3
import json
import os
import logging
import hashlib
from datetime import datetime, timedelta
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# Resolved once at import — config.py already tested writability
_DB_PATH = config.DB_PATH
os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)


def get_db_path():
    return _DB_PATH


@contextmanager
def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_live_timestamp(value=None):
    if value in (None, ""):
        return datetime.utcnow().isoformat()
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1_000_000_000_000:
            timestamp /= 1000.0
        return datetime.utcfromtimestamp(timestamp).isoformat()
    return str(value)


def _normalize_live_side(value):
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "buy" in text or "long" in text:
        return "buy"
    if "sell" in text or "short" in text:
        return "sell"
    return text


def _parse_iso_timestamp(value):
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def _make_live_fill_uid(fill):
    for key in ("tid", "fillId", "id", "oid"):
        value = fill.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"

    fallback = {
        "time": fill.get("time"),
        "coin": fill.get("coin"),
        "dir": fill.get("dir") or fill.get("side"),
        "size": fill.get("sz") or fill.get("size"),
        "price": fill.get("px") or fill.get("price"),
        "closedPnl": fill.get("closedPnl"),
        "fee": fill.get("fee"),
    }
    payload = json.dumps(fallback, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def init_db():
    """Create all tables if they don't exist."""
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
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);

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

        CREATE TABLE IF NOT EXISTS live_equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            public_address TEXT,
            perps_margin REAL,
            spot_usdc REAL,
            total REAL,
            daily_realized_pnl REAL DEFAULT 0,
            orders_today INTEGER DEFAULT 0,
            fills_today INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_live_equity_timestamp ON live_equity_snapshots(timestamp);

        CREATE TABLE IF NOT EXISTS live_position_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            public_address TEXT,
            position_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_live_position_batches_timestamp ON live_position_batches(timestamp);

        CREATE TABLE IF NOT EXISTS live_position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            signed_size REAL NOT NULL,
            entry_price REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            leverage REAL DEFAULT 1,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (snapshot_id) REFERENCES live_position_batches(snapshot_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_live_position_snapshot_unique
            ON live_position_snapshots(snapshot_id, coin, side, signed_size);
        CREATE INDEX IF NOT EXISTS idx_live_position_snapshot_coin ON live_position_snapshots(coin);

        CREATE TABLE IF NOT EXISTS live_fill_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fill_uid TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            public_address TEXT,
            coin TEXT,
            side TEXT,
            dir TEXT,
            size REAL DEFAULT 0,
            price REAL DEFAULT 0,
            fee REAL DEFAULT 0,
            closed_pnl REAL DEFAULT 0,
            order_id TEXT,
            start_position TEXT,
            raw_fill TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_live_fill_timestamp ON live_fill_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_live_fill_coin ON live_fill_events(coin);

        CREATE TABLE IF NOT EXISTS live_execution_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            public_address TEXT,
            signal_id TEXT,
            source TEXT,
            source_key TEXT,
            coin TEXT,
            side TEXT,
            status TEXT NOT NULL,
            execution_role TEXT DEFAULT 'taker',
            requested_size REAL DEFAULT 0,
            submitted_size REAL DEFAULT 0,
            filled_size REAL DEFAULT 0,
            requested_notional REAL DEFAULT 0,
            submitted_notional REAL DEFAULT 0,
            mid_price REAL DEFAULT 0,
            execution_price REAL DEFAULT 0,
            expected_slippage_bps REAL DEFAULT 0,
            realized_slippage_bps REAL DEFAULT 0,
            fill_ratio REAL DEFAULT 0,
            rejection_reason TEXT,
            protective_status TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_live_execution_timestamp
            ON live_execution_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_live_execution_source_key
            ON live_execution_events(source_key, timestamp);
        CREATE INDEX IF NOT EXISTS idx_live_execution_status
            ON live_execution_events(status);

        CREATE TABLE IF NOT EXISTS decision_research_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cycle_number INTEGER DEFAULT 0,
            regime TEXT DEFAULT 'unknown',
            available_slots INTEGER DEFAULT 0,
            candidate_count INTEGER DEFAULT 0,
            qualified_count INTEGER DEFAULT 0,
            executed_count INTEGER DEFAULT 0,
            long_score REAL DEFAULT 0,
            short_score REAL DEFAULT 0,
            market_bias TEXT DEFAULT 'neutral',
            context_json TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_decision_research_cycles_timestamp
            ON decision_research_cycles(timestamp);

        CREATE TABLE IF NOT EXISTS decision_research_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_cycle_id INTEGER NOT NULL,
            candidate_rank INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            name TEXT,
            source TEXT,
            source_key TEXT,
            strategy_type TEXT,
            coin TEXT,
            side TEXT,
            route TEXT,
            composite_score REAL DEFAULT 0,
            confidence REAL DEFAULT 0,
            expected_value_pct REAL DEFAULT 0,
            execution_cost_pct REAL DEFAULT 0,
            blockers_json TEXT DEFAULT '[]',
            score_breakdown_json TEXT DEFAULT '{}',
            candidate_json TEXT DEFAULT '{}',
            FOREIGN KEY (research_cycle_id) REFERENCES decision_research_cycles(id)
        );
        CREATE INDEX IF NOT EXISTS idx_decision_research_candidates_cycle
            ON decision_research_candidates(research_cycle_id, candidate_rank);
        CREATE INDEX IF NOT EXISTS idx_decision_research_candidates_status
            ON decision_research_candidates(status);
        """)


# ─── Trader CRUD ───────────────────────────────────────────────

def upsert_trader(address, total_pnl=0, roi_pct=0, account_value=0,
                  win_rate=0, trade_count=0, metadata=None, is_active=True):
    now = datetime.utcnow().isoformat()
    meta_json = json.dumps(metadata or {})
    active_int = 1 if is_active else 0
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
        """, (address, now, now, total_pnl, roi_pct, account_value,
              win_rate, trade_count, meta_json, active_int,
              now, total_pnl, roi_pct, account_value, win_rate, trade_count, meta_json, active_int))


def mark_trader_inactive(address):
    """Mark a trader as inactive (e.g. detected as bot)."""
    with get_connection() as conn:
        conn.execute("UPDATE traders SET active = 0, last_updated = ? WHERE address = ?",
                     (datetime.utcnow().isoformat(), address))


def get_active_traders():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM traders WHERE active = 1 ORDER BY total_pnl DESC"
        ).fetchall()
    return [dict(r) for r in rows]


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
            "SELECT address FROM traders WHERE active = 0"
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
    now = datetime.utcnow().isoformat()
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
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO strategies
            (name, description, strategy_type, parameters, discovered_at,
             total_pnl, trade_count, win_rate, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, description, strategy_type, json.dumps(parameters or {}),
              now, total_pnl, trade_count, win_rate, sharpe_ratio))
        return cursor.lastrowid


def save_strategies_batch(strategies_data):
    """Batch insert multiple strategies in a single transaction."""
    now = datetime.utcnow().isoformat()
    saved_ids = []
    with get_connection() as conn:
        for s in strategies_data:
            cursor = conn.execute("""
                INSERT INTO strategies
                (name, description, strategy_type, parameters, discovered_at,
                 total_pnl, trade_count, win_rate, sharpe_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (s["name"], s["description"], s["strategy_type"],
                  json.dumps(s.get("parameters") or {}),
                  now, s.get("total_pnl", 0), s.get("trade_count", 0),
                  s.get("win_rate", 0), s.get("sharpe_ratio", 0)))
            saved_ids.append(cursor.lastrowid)
    return saved_ids


def update_strategy_score(strategy_id, score):
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""
            UPDATE strategies SET current_score = ?, last_scored = ? WHERE id = ?
        """, (score, now, strategy_id))


def get_active_strategies():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE active = 1 ORDER BY current_score DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_strategy(strategy_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    return dict(row) if row else None


# ─── Strategy Scores ──────────────────────────────────────────

def save_strategy_score(strategy_id, score, pnl_score=0, win_rate_score=0,
                        sharpe_score=0, consistency_score=0, risk_adj_score=0, notes=""):
    now = datetime.utcnow().isoformat()
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


# ─── Paper Trading ─────────────────────────────────────────────

def init_paper_account(balance):
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO paper_account (id, balance, total_pnl, total_trades, winning_trades, last_updated)
            VALUES (1, ?, 0, 0, 0, ?)
        """, (balance, now))


def get_paper_account():
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    return dict(row) if row else None


def update_paper_account(balance, total_pnl, total_trades, winning_trades):
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""
            UPDATE paper_account
            SET balance = ?, total_pnl = ?, total_trades = ?, winning_trades = ?, last_updated = ?
            WHERE id = 1
        """, (balance, total_pnl, total_trades, winning_trades, now))


def open_paper_trade(strategy_id, coin, side, entry_price, size, leverage=1,
                     stop_loss=None, take_profit=None, metadata=None):
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO paper_trades
            (strategy_id, opened_at, coin, side, entry_price, size, leverage,
             stop_loss, take_profit, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        """, (strategy_id, now, coin, side, entry_price, size, leverage,
              stop_loss, take_profit, json.dumps(metadata or {})))
        return cursor.lastrowid


def update_paper_trade_metadata(trade_id: int, extra: dict):
    """Merge extra keys into a paper trade's metadata JSON blob."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT metadata FROM paper_trades WHERE id = ?", (trade_id,)
        ).fetchone()
        if row:
            try:
                existing = json.loads(row["metadata"] or "{}")
            except Exception:
                existing = {}
            existing.update(extra)
            conn.execute(
                "UPDATE paper_trades SET metadata = ? WHERE id = ?",
                (json.dumps(existing), trade_id)
            )


def close_paper_trade(trade_id, exit_price, pnl) -> bool:
    """Close a paper trade.  Returns True on success, False if trade_id not found.

    CRIT-FIX CRIT-5: check rowcount — if the UPDATE matches 0 rows the trade was
    already closed or the ID is wrong.  The caller MUST check the return value and
    skip the account PnL credit to prevent phantom double-credit.
    """
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE paper_trades SET closed_at = ?, exit_price = ?, pnl = ?, status = 'closed'
            WHERE id = ? AND status = 'open'
        """, (now, exit_price, pnl, trade_id))
        if cursor.rowcount == 0:
            logger.error(
                "close_paper_trade: trade_id=%s matched 0 open rows — "
                "possible double-close or missing record. PnL NOT credited.",
                trade_id,
            )
            return False
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
        conn.execute("""
            INSERT OR REPLACE INTO paper_account (id, balance, total_pnl, total_trades, winning_trades, last_updated)
            VALUES (1, ?, 0, 0, 0, ?)
        """, (initial_balance, datetime.utcnow().isoformat()))

    logger.info(f"Paper trades reset: cleared {open_count} open + {closed_count} closed trades, "
               f"balance reset to ${initial_balance:,.2f}")
    return {
        "open_deleted": open_count,
        "closed_deleted": closed_count,
        "new_balance": initial_balance,
    }


def save_live_equity_snapshot(public_address: str, perps_margin=None, spot_usdc=None,
                              total=None, daily_realized_pnl=0.0, orders_today=0,
                              fills_today=0, metadata=None, timestamp=None):
    snapshot_time = _normalize_live_timestamp(timestamp)
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO live_equity_snapshots
            (timestamp, public_address, perps_margin, spot_usdc, total,
             daily_realized_pnl, orders_today, fills_today, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot_time,
            public_address,
            perps_margin,
            spot_usdc,
            total,
            _safe_float(daily_realized_pnl, 0.0),
            int(orders_today or 0),
            int(fills_today or 0),
            json.dumps(metadata or {}),
        ))
        return cursor.lastrowid


def save_live_position_snapshot(public_address: str, positions: list,
                                metadata=None, timestamp=None):
    snapshot_time = _normalize_live_timestamp(timestamp)
    normalized_positions = []
    for position in positions or []:
        coin = str(position.get("coin", "") or "").strip().upper()
        if not coin:
            continue
        signed_size = _safe_float(position.get("szi", position.get("size", 0)), 0.0)
        size = abs(signed_size) if signed_size else abs(_safe_float(position.get("size", 0), 0.0))
        side = str(position.get("side", "") or "").strip().lower()
        if not side:
            side = "long" if signed_size > 0 else "short" if signed_size < 0 else "flat"
        normalized_positions.append(
            {
                "coin": coin,
                "side": side,
                "size": size,
                "signed_size": signed_size,
                "entry_price": _safe_float(position.get("entry_price", position.get("entryPx", 0)), 0.0),
                "unrealized_pnl": _safe_float(position.get("unrealized_pnl", position.get("unrealizedPnl", 0)), 0.0),
                "leverage": _safe_float(position.get("leverage", 1), 1.0),
                "metadata": json.dumps({"raw": position.get("raw", position)}),
            }
        )

    signature = json.dumps(
        {
            "public_address": public_address,
            "timestamp": snapshot_time,
            "positions": normalized_positions,
        },
        sort_keys=True,
        default=str,
    )
    snapshot_id = hashlib.sha256(signature.encode()).hexdigest()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO live_position_batches
            (snapshot_id, timestamp, public_address, position_count, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            snapshot_id,
            snapshot_time,
            public_address,
            len(normalized_positions),
            json.dumps(metadata or {}),
        ))
        for position in normalized_positions:
            conn.execute("""
                INSERT INTO live_position_snapshots
                (snapshot_id, coin, side, size, signed_size, entry_price,
                 unrealized_pnl, leverage, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                position["coin"],
                position["side"],
                position["size"],
                position["signed_size"],
                position["entry_price"],
                position["unrealized_pnl"],
                position["leverage"],
                position["metadata"],
            ))
    return snapshot_id


def upsert_live_fill_events(public_address: str, fills: list):
    inserted = 0
    updated = 0
    with get_connection() as conn:
        for fill in fills or []:
            fill_uid = _make_live_fill_uid(fill)
            cursor = conn.execute("""
                INSERT INTO live_fill_events
                (fill_uid, timestamp, public_address, coin, side, dir, size, price,
                 fee, closed_pnl, order_id, start_position, raw_fill)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fill_uid) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    public_address = excluded.public_address,
                    coin = excluded.coin,
                    side = excluded.side,
                    dir = excluded.dir,
                    size = excluded.size,
                    price = excluded.price,
                    fee = excluded.fee,
                    closed_pnl = excluded.closed_pnl,
                    order_id = excluded.order_id,
                    start_position = excluded.start_position,
                    raw_fill = excluded.raw_fill
            """, (
                fill_uid,
                _normalize_live_timestamp(fill.get("time")),
                public_address,
                str(fill.get("coin", "") or "").strip().upper() or None,
                _normalize_live_side(fill.get("side") or fill.get("dir")),
                str(fill.get("dir", fill.get("side", "")) or ""),
                _safe_float(fill.get("sz", fill.get("size", 0)), 0.0),
                _safe_float(fill.get("px", fill.get("price", 0)), 0.0),
                _safe_float(fill.get("fee", 0), 0.0),
                _safe_float(fill.get("closedPnl", 0), 0.0),
                str(fill.get("oid", fill.get("orderId", "")) or ""),
                str(fill.get("startPosition", "") or ""),
                json.dumps(fill, sort_keys=True, default=str),
            ))
            if cursor.rowcount:
                inserted += 1

    return {
        "seen": len(fills or []),
        "upserted": inserted,
        "updated": updated,
    }


def get_latest_live_equity_snapshot(public_address: str = None):
    try:
        with get_connection() as conn:
            query = "SELECT * FROM live_equity_snapshots"
            params = []
            if public_address:
                query += " WHERE public_address = ?"
                params.append(public_address)
            query += " ORDER BY id DESC LIMIT 1"
            row = conn.execute(query, params).fetchone()
    except sqlite3.OperationalError:
        return None
    return dict(row) if row else None


def get_recent_live_equity_snapshots(limit: int = 100, public_address: str = None):
    try:
        with get_connection() as conn:
            query = "SELECT * FROM live_equity_snapshots"
            params = []
            if public_address:
                query += " WHERE public_address = ?"
                params.append(public_address)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in reversed(rows)]


def get_latest_live_positions(public_address: str = None):
    try:
        with get_connection() as conn:
            query = "SELECT * FROM live_position_batches"
            params = []
            if public_address:
                query += " WHERE public_address = ?"
                params.append(public_address)
            query += " ORDER BY id DESC LIMIT 1"
            batch = conn.execute(query, params).fetchone()
            if not batch:
                return {"snapshot": None, "positions": []}
            rows = conn.execute("""
                SELECT * FROM live_position_snapshots
                WHERE snapshot_id = ?
                ORDER BY ABS(signed_size) DESC, coin ASC
            """, (batch["snapshot_id"],)).fetchall()
    except sqlite3.OperationalError:
        return {"snapshot": None, "positions": []}
    return {
        "snapshot": dict(batch),
        "positions": [dict(r) for r in rows],
    }


def get_live_fill_summary(public_address: str = None):
    try:
        with get_connection() as conn:
            query = """
                SELECT
                    COUNT(*) AS fill_count,
                    COALESCE(SUM(closed_pnl), 0) AS realized_pnl,
                    COALESCE(SUM(fee), 0) AS fees_paid,
                    MAX(timestamp) AS last_fill_timestamp
                FROM live_fill_events
            """
            params = []
            if public_address:
                query += " WHERE public_address = ?"
                params.append(public_address)
            row = conn.execute(query, params).fetchone()
    except sqlite3.OperationalError:
        row = None
    return dict(row) if row else {
        "fill_count": 0,
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "last_fill_timestamp": None,
    }


def save_live_execution_event(public_address: str, **event):
    timestamp = _normalize_live_timestamp(event.get("timestamp"))
    source = str(event.get("source", "") or "").strip().lower() or None
    source_key = str(event.get("source_key", "") or "").strip() or None
    coin = str(event.get("coin", "") or "").strip().upper() or None
    side = str(event.get("side", "") or "").strip().lower() or None
    status = str(event.get("status", "unknown") or "unknown").strip().lower()
    execution_role = str(event.get("execution_role", "taker") or "taker").strip().lower()
    if execution_role not in {"maker", "taker"}:
        execution_role = "taker"

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO live_execution_events
            (timestamp, public_address, signal_id, source, source_key, coin, side,
             status, execution_role, requested_size, submitted_size, filled_size,
             requested_notional, submitted_notional, mid_price, execution_price,
             expected_slippage_bps, realized_slippage_bps, fill_ratio,
             rejection_reason, protective_status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            public_address,
            str(event.get("signal_id", "") or "").strip() or None,
            source,
            source_key,
            coin,
            side,
            status,
            execution_role,
            _safe_float(event.get("requested_size"), 0.0),
            _safe_float(event.get("submitted_size"), 0.0),
            _safe_float(event.get("filled_size"), 0.0),
            _safe_float(event.get("requested_notional"), 0.0),
            _safe_float(event.get("submitted_notional"), 0.0),
            _safe_float(event.get("mid_price"), 0.0),
            _safe_float(event.get("execution_price"), 0.0),
            _safe_float(event.get("expected_slippage_bps"), 0.0),
            _safe_float(event.get("realized_slippage_bps"), 0.0),
            _safe_float(event.get("fill_ratio"), 0.0),
            str(event.get("rejection_reason", "") or "").strip() or None,
            str(event.get("protective_status", "") or "").strip().lower(),
            json.dumps(event.get("metadata", {}) or {}, sort_keys=True, default=str),
        ))
        return cursor.lastrowid


def _get_execution_quality_row(
    *,
    public_address: str = None,
    source_key: str = None,
    source: str = None,
    lookback_hours: float = 24.0 * 7,
):
    conditions = []
    params = []
    if public_address:
        conditions.append("public_address = ?")
        params.append(public_address)
    if source_key:
        conditions.append("source_key = ?")
        params.append(source_key)
    if source:
        conditions.append("source = ?")
        params.append(str(source).strip().lower())
    if lookback_hours and float(lookback_hours) > 0:
        since = (datetime.utcnow() - timedelta(hours=float(lookback_hours))).isoformat()
        conditions.append("timestamp >= ?")
        params.append(since)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT
            COUNT(*) AS total_events,
            COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS success_count,
            COALESCE(SUM(CASE WHEN status = 'warning' THEN 1 ELSE 0 END), 0) AS warning_count,
            COALESCE(SUM(CASE WHEN status IN ('rejected', 'error') THEN 1 ELSE 0 END), 0) AS rejection_count,
            COALESCE(SUM(CASE WHEN protective_status = 'failed' THEN 1 ELSE 0 END), 0) AS protective_failure_count,
            AVG(CASE WHEN status = 'success' THEN realized_slippage_bps END) AS avg_realized_slippage_bps,
            AVG(CASE WHEN status = 'success' THEN expected_slippage_bps END) AS avg_expected_slippage_bps,
            AVG(CASE WHEN status = 'success' THEN fill_ratio END) AS avg_fill_ratio,
            AVG(
                CASE
                    WHEN status = 'success' AND execution_role = 'maker' THEN 1.0
                    WHEN status = 'success' AND execution_role = 'taker' THEN 0.0
                    ELSE NULL
                END
            ) AS maker_ratio,
            MAX(timestamp) AS last_event_timestamp
        FROM live_execution_events
        {where_clause}
    """

    try:
        with get_connection() as conn:
            row = conn.execute(query, params).fetchone()
    except sqlite3.OperationalError:
        return {}
    return dict(row) if row else {}


def get_execution_quality_summary(
    public_address: str = None,
    source_key: str = None,
    source: str = None,
    lookback_hours: float = 24.0 * 7,
):
    row = _get_execution_quality_row(
        public_address=public_address,
        source_key=source_key,
        source=source,
        lookback_hours=lookback_hours,
    )
    total_events = int(row.get("total_events", 0) or 0)
    success_count = int(row.get("success_count", 0) or 0)
    warning_count = int(row.get("warning_count", 0) or 0)
    rejection_count = int(row.get("rejection_count", 0) or 0)
    protective_failure_count = int(row.get("protective_failure_count", 0) or 0)
    maker_ratio = _safe_float(row.get("maker_ratio"), 0.0)

    return {
        "total_events": total_events,
        "success_count": success_count,
        "warning_count": warning_count,
        "rejection_count": rejection_count,
        "protective_failure_count": protective_failure_count,
        "success_rate": round(success_count / total_events, 4) if total_events else 0.0,
        "warning_rate": round(warning_count / total_events, 4) if total_events else 0.0,
        "rejection_rate": round(rejection_count / total_events, 4) if total_events else 0.0,
        "protective_failure_rate": (
            round(protective_failure_count / success_count, 4) if success_count else 0.0
        ),
        "avg_realized_slippage_bps": round(
            _safe_float(row.get("avg_realized_slippage_bps"), 0.0), 4
        ),
        "avg_expected_slippage_bps": round(
            _safe_float(row.get("avg_expected_slippage_bps"), 0.0), 4
        ),
        "avg_fill_ratio": round(_safe_float(row.get("avg_fill_ratio"), 0.0), 4),
        "maker_ratio": round(maker_ratio, 4),
        "taker_ratio": round(max(0.0, 1.0 - maker_ratio), 4) if success_count else 0.0,
        "dominant_execution_role": "maker" if success_count and maker_ratio >= 0.5 else "taker",
        "last_event_timestamp": row.get("last_event_timestamp"),
        "lookback_hours": float(lookback_hours or 0.0),
    }


def get_execution_quality_by_source(
    public_address: str = None,
    lookback_hours: float = 24.0 * 7,
    limit: int = 10,
):
    conditions = []
    params = []
    if public_address:
        conditions.append("public_address = ?")
        params.append(public_address)
    if lookback_hours and float(lookback_hours) > 0:
        since = (datetime.utcnow() - timedelta(hours=float(lookback_hours))).isoformat()
        conditions.append("timestamp >= ?")
        params.append(since)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT
            COALESCE(source_key, source, 'unknown') AS source_key,
            COALESCE(source, 'unknown') AS source,
            COUNT(*) AS total_events,
            COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS success_count,
            COALESCE(SUM(CASE WHEN status = 'warning' THEN 1 ELSE 0 END), 0) AS warning_count,
            COALESCE(SUM(CASE WHEN status IN ('rejected', 'error') THEN 1 ELSE 0 END), 0) AS rejection_count,
            COALESCE(SUM(CASE WHEN protective_status = 'failed' THEN 1 ELSE 0 END), 0) AS protective_failure_count,
            AVG(CASE WHEN status = 'success' THEN realized_slippage_bps END) AS avg_realized_slippage_bps,
            AVG(CASE WHEN status = 'success' THEN fill_ratio END) AS avg_fill_ratio,
            AVG(
                CASE
                    WHEN status = 'success' AND execution_role = 'maker' THEN 1.0
                    WHEN status = 'success' AND execution_role = 'taker' THEN 0.0
                    ELSE NULL
                END
            ) AS maker_ratio,
            MAX(timestamp) AS last_event_timestamp
        FROM live_execution_events
        {where_clause}
        GROUP BY COALESCE(source_key, source, 'unknown'), COALESCE(source, 'unknown')
        ORDER BY total_events DESC, last_event_timestamp DESC
        LIMIT ?
    """

    try:
        with get_connection() as conn:
            rows = conn.execute(query, [*params, int(limit)]).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        payload = dict(row)
        total_events = int(payload.get("total_events", 0) or 0)
        success_count = int(payload.get("success_count", 0) or 0)
        warning_count = int(payload.get("warning_count", 0) or 0)
        rejection_count = int(payload.get("rejection_count", 0) or 0)
        maker_ratio = _safe_float(payload.get("maker_ratio"), 0.0)
        results.append(
            {
                "source_key": payload.get("source_key"),
                "source": payload.get("source"),
                "total_events": total_events,
                "success_count": success_count,
                "warning_count": warning_count,
                "rejection_count": rejection_count,
                "success_rate": round(success_count / total_events, 4) if total_events else 0.0,
                "warning_rate": round(warning_count / total_events, 4) if total_events else 0.0,
                "rejection_rate": round(rejection_count / total_events, 4) if total_events else 0.0,
                "protective_failure_count": int(payload.get("protective_failure_count", 0) or 0),
                "avg_realized_slippage_bps": round(
                    _safe_float(payload.get("avg_realized_slippage_bps"), 0.0), 4
                ),
                "avg_fill_ratio": round(_safe_float(payload.get("avg_fill_ratio"), 0.0), 4),
                "maker_ratio": round(maker_ratio, 4),
                "taker_ratio": round(max(0.0, 1.0 - maker_ratio), 4) if success_count else 0.0,
                "last_event_timestamp": payload.get("last_event_timestamp"),
            }
        )
    return results


def save_decision_research_snapshot(snapshot: dict) -> int:
    """Persist a replayable decision-cycle snapshot and all candidate rows."""
    timestamp = _normalize_live_timestamp(snapshot.get("timestamp"))
    candidates = snapshot.get("candidates", []) or []
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO decision_research_cycles
            (timestamp, cycle_number, regime, available_slots, candidate_count,
             qualified_count, executed_count, long_score, short_score,
             market_bias, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            int(snapshot.get("cycle_number", 0) or 0),
            str(snapshot.get("regime", "unknown") or "unknown"),
            int(snapshot.get("available_slots", 0) or 0),
            int(snapshot.get("candidate_count", len(candidates)) or 0),
            int(snapshot.get("qualified_count", 0) or 0),
            int(snapshot.get("executed_count", 0) or 0),
            _safe_float(snapshot.get("long_score", 0.0), 0.0),
            _safe_float(snapshot.get("short_score", 0.0), 0.0),
            str(snapshot.get("market_bias", "neutral") or "neutral"),
            json.dumps(snapshot.get("context", {}), sort_keys=True, default=str),
        ))
        research_cycle_id = cursor.lastrowid

        for index, candidate in enumerate(candidates, start=1):
            conn.execute("""
                INSERT INTO decision_research_candidates
                (research_cycle_id, candidate_rank, status, name, source, source_key,
                 strategy_type, coin, side, route, composite_score, confidence,
                 expected_value_pct, execution_cost_pct, blockers_json,
                 score_breakdown_json, candidate_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                research_cycle_id,
                int(candidate.get("rank", index) or index),
                str(candidate.get("status", "unknown") or "unknown"),
                candidate.get("name"),
                candidate.get("source"),
                candidate.get("source_key"),
                candidate.get("strategy_type"),
                candidate.get("coin"),
                candidate.get("side"),
                candidate.get("route"),
                _safe_float(candidate.get("composite_score", 0.0), 0.0),
                _safe_float(candidate.get("confidence", 0.0), 0.0),
                _safe_float(candidate.get("expected_value_pct", 0.0), 0.0),
                _safe_float(candidate.get("execution_cost_pct", 0.0), 0.0),
                json.dumps(candidate.get("blockers", []), sort_keys=True, default=str),
                json.dumps(candidate.get("score_breakdown", {}), sort_keys=True, default=str),
                json.dumps(candidate.get("raw_candidate", {}), sort_keys=True, default=str),
            ))
    return research_cycle_id


def get_decision_research_candidates(research_cycle_id: int) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM decision_research_candidates
                WHERE research_cycle_id = ?
                ORDER BY candidate_rank ASC, id ASC
            """, (int(research_cycle_id),)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


def get_recent_decision_research(limit: int = 20, include_candidates: bool = False) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM decision_research_cycles
                ORDER BY id DESC
                LIMIT ?
            """, (int(limit),)).fetchall()
    except sqlite3.OperationalError:
        return []

    cycles = []
    for row in reversed(rows):
        cycle = dict(row)
        try:
            cycle["context"] = json.loads(cycle.get("context_json") or "{}")
        except Exception:
            cycle["context"] = {}
        if include_candidates:
            candidates = get_decision_research_candidates(cycle["id"])
            for candidate in candidates:
                for key in ("blockers_json", "score_breakdown_json", "candidate_json"):
                    try:
                        parsed = json.loads(candidate.get(key) or ("[]" if key == "blockers_json" else "{}"))
                    except Exception:
                        parsed = [] if key == "blockers_json" else {}
                    if key == "blockers_json":
                        candidate["blockers"] = parsed
                    elif key == "score_breakdown_json":
                        candidate["score_breakdown"] = parsed
                    else:
                        candidate["raw_candidate"] = parsed
                candidate.pop("blockers_json", None)
                candidate.pop("score_breakdown_json", None)
                candidate.pop("candidate_json", None)
            cycle["candidates"] = candidates
        cycles.append(cycle)
    return cycles


def get_decision_funnel_summary(limit_cycles: int = 50) -> dict:
    cycles = get_recent_decision_research(limit=limit_cycles, include_candidates=True)
    summary = {
        "cycles": len(cycles),
        "candidate_count": 0,
        "selected_count": 0,
        "blocked_count": 0,
        "overflow_count": 0,
        "ranked_count": 0,
        "no_trade_cycles": 0,
        "selection_rate": 0.0,
        "avg_candidates_per_cycle": 0.0,
        "blocker_mix": {},
        "status_mix": {},
    }
    blocker_mix = {}
    status_mix = {}

    for cycle in cycles:
        candidates = cycle.get("candidates", []) or []
        summary["candidate_count"] += len(candidates)
        selected_this_cycle = 0
        for candidate in candidates:
            status = str(candidate.get("status", "unknown") or "unknown")
            status_mix[status] = status_mix.get(status, 0) + 1
            if status == "selected":
                summary["selected_count"] += 1
                selected_this_cycle += 1
            elif status == "blocked":
                summary["blocked_count"] += 1
                blockers = candidate.get("blockers") or ["unknown"]
                for blocker in blockers:
                    label = str(blocker or "unknown")
                    blocker_mix[label] = blocker_mix.get(label, 0) + 1
            elif status == "overflow":
                summary["overflow_count"] += 1
            else:
                summary["ranked_count"] += 1
        if selected_this_cycle == 0:
            summary["no_trade_cycles"] += 1

    if summary["candidate_count"] > 0:
        summary["selection_rate"] = round(
            summary["selected_count"] / summary["candidate_count"],
            4,
        )
    if summary["cycles"] > 0:
        summary["avg_candidates_per_cycle"] = round(
            summary["candidate_count"] / summary["cycles"],
            2,
        )

    summary["status_mix"] = dict(
        sorted(status_mix.items(), key=lambda item: (-item[1], item[0]))
    )
    summary["blocker_mix"] = dict(
        sorted(blocker_mix.items(), key=lambda item: (-item[1], item[0]))
    )
    return summary


def get_source_attribution_summary(limit_cycles: int = 50, lookback_hours: float = 24.0 * 7) -> list:
    cycles = get_recent_decision_research(limit=limit_cycles, include_candidates=True)
    cutoff = None
    if lookback_hours and float(lookback_hours) > 0:
        cutoff = datetime.utcnow() - timedelta(hours=float(lookback_hours))

    by_source = {}

    def _bucket_for(source_key: str, source: str):
        key = str(source_key or source or "unknown").strip() or "unknown"
        bucket = by_source.setdefault(
            key,
            {
                "source_key": key,
                "source": str(source or "unknown").strip().lower() or "unknown",
                "candidate_count": 0,
                "selected_count": 0,
                "blocked_count": 0,
                "overflow_count": 0,
                "avg_composite_score": 0.0,
                "avg_expected_value_pct": 0.0,
                "_composite_total": 0.0,
                "_expected_value_total": 0.0,
                "paper_open_count": 0,
                "paper_closed_count": 0,
                "paper_realized_pnl": 0.0,
            },
        )
        if source and bucket["source"] == "unknown":
            bucket["source"] = str(source).strip().lower() or "unknown"
        return bucket

    for cycle in cycles:
        cycle_ts = _parse_iso_timestamp(cycle.get("timestamp"))
        if cutoff and cycle_ts and cycle_ts < cutoff:
            continue
        for candidate in cycle.get("candidates", []) or []:
            bucket = _bucket_for(candidate.get("source_key"), candidate.get("source"))
            bucket["candidate_count"] += 1
            bucket["_composite_total"] += _safe_float(candidate.get("composite_score"), 0.0)
            bucket["_expected_value_total"] += _safe_float(candidate.get("expected_value_pct"), 0.0)
            status = str(candidate.get("status", "") or "").lower()
            if status == "selected":
                bucket["selected_count"] += 1
            elif status == "blocked":
                bucket["blocked_count"] += 1
            elif status == "overflow":
                bucket["overflow_count"] += 1

    try:
        with get_connection() as conn:
            paper_rows = conn.execute(
                """
                SELECT status, pnl, metadata
                FROM paper_trades
                """
            ).fetchall()
    except sqlite3.OperationalError:
        paper_rows = []

    for row in paper_rows:
        metadata = {}
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}
        source_key = str(metadata.get("source_key", "") or "").strip()
        source = str(metadata.get("source", metadata.get("signal_source", "")) or "").strip().lower()
        bucket = _bucket_for(source_key, source)
        status = str(row["status"] or "").lower()
        if status == "open":
            bucket["paper_open_count"] += 1
        elif status == "closed":
            bucket["paper_closed_count"] += 1
            bucket["paper_realized_pnl"] += _safe_float(row["pnl"], 0.0)

    live_by_source = {
        row["source_key"]: row
        for row in get_execution_quality_by_source(
            lookback_hours=lookback_hours,
            limit=max(len(by_source), 1) + 20,
        )
    }

    rows = []
    for bucket in by_source.values():
        candidate_count = bucket["candidate_count"] or 0
        if candidate_count > 0:
            bucket["avg_composite_score"] = round(bucket["_composite_total"] / candidate_count, 4)
            bucket["avg_expected_value_pct"] = round(bucket["_expected_value_total"] / candidate_count, 4)
        live_row = live_by_source.get(bucket["source_key"], {})
        bucket["live_events"] = int(live_row.get("total_events", 0) or 0)
        bucket["live_success_rate"] = round(
            _safe_float(live_row.get("success_rate"), 0.0),
            4,
        )
        bucket["live_rejection_rate"] = round(
            _safe_float(live_row.get("rejection_rate"), 0.0),
            4,
        )
        bucket["live_avg_slippage_bps"] = round(
            _safe_float(live_row.get("avg_realized_slippage_bps"), 0.0),
            4,
        )
        bucket["live_avg_fill_ratio"] = round(
            _safe_float(live_row.get("avg_fill_ratio"), 0.0),
            4,
        )
        bucket["paper_realized_pnl"] = round(bucket["paper_realized_pnl"], 2)
        bucket.pop("_composite_total", None)
        bucket.pop("_expected_value_total", None)
        rows.append(bucket)

    rows.sort(
        key=lambda item: (
            -item["selected_count"],
            -item["live_events"],
            -item["candidate_count"],
            item["source_key"],
        )
    )
    return rows


def get_runtime_divergence_summary(lookback_hours: float = 24.0) -> dict:
    cutoff = None
    if lookback_hours and float(lookback_hours) > 0:
        cutoff = datetime.utcnow() - timedelta(hours=float(lookback_hours))

    cycles = get_recent_decision_research(limit=200, include_candidates=True)
    shadow_selected_count = 0
    for cycle in cycles:
        cycle_ts = _parse_iso_timestamp(cycle.get("timestamp"))
        if cutoff and cycle_ts and cycle_ts < cutoff:
            continue
        shadow_selected_count += sum(
            1
            for candidate in cycle.get("candidates", []) or []
            if str(candidate.get("status", "") or "").lower() == "selected"
        )

    paper_open_count = len(get_open_paper_trades())
    paper_recent_open_count = 0
    paper_recent_closed_count = 0
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT opened_at, closed_at, status FROM paper_trades"
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    for row in rows:
        opened_at = _parse_iso_timestamp(row["opened_at"])
        closed_at = _parse_iso_timestamp(row["closed_at"])
        if cutoff is None or (opened_at and opened_at >= cutoff):
            paper_recent_open_count += 1
        if str(row["status"] or "").lower() == "closed" and (cutoff is None or (closed_at and closed_at >= cutoff)):
            paper_recent_closed_count += 1

    live_positions = get_latest_live_positions()
    live_open_positions = len(live_positions.get("positions", []) or [])
    live_execution = get_execution_quality_summary(lookback_hours=lookback_hours)
    live_execution_total = int(live_execution.get("total_events", 0) or 0)
    live_success_count = int(live_execution.get("success_count", 0) or 0)
    live_rejection_count = int(live_execution.get("rejection_count", 0) or 0)

    paper_live_gap = paper_open_count - live_open_positions
    shadow_live_gap = shadow_selected_count - live_execution_total

    return {
        "lookback_hours": float(lookback_hours or 0.0),
        "shadow_selected_count": shadow_selected_count,
        "paper_open_count": paper_open_count,
        "paper_recent_open_count": paper_recent_open_count,
        "paper_recent_closed_count": paper_recent_closed_count,
        "live_open_positions": live_open_positions,
        "live_execution_total": live_execution_total,
        "live_success_count": live_success_count,
        "live_rejection_count": live_rejection_count,
        "paper_live_open_gap": paper_live_gap,
        "paper_live_open_gap_ratio": round(
            abs(paper_live_gap) / max(1, max(paper_open_count, live_open_positions)),
            4,
        ),
        "shadow_live_execution_gap": shadow_live_gap,
        "shadow_live_execution_gap_ratio": round(
            abs(shadow_live_gap) / max(1, max(shadow_selected_count, live_execution_total)),
            4,
        ),
    }


# ─── Research Logs ─────────────────────────────────────────────

def log_research_cycle(cycle_type, summary, details=None,
                       traders_analyzed=0, strategies_found=0, strategies_updated=0):
    now = datetime.utcnow().isoformat()
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
    now = datetime.utcnow().isoformat()
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
            "timestamp": datetime.utcnow().isoformat(),
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
                def _table_exists(name):
                    return conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)
                    ).fetchone() is not None

                if _table_exists("golden_wallets"):
                    rows = conn.execute(
                        "SELECT * FROM golden_wallets ORDER BY penalised_pnl DESC"
                    ).fetchall()
                    data["golden_wallets"] = [dict(r) for r in rows]

                if _table_exists("wallet_fills"):
                    # Only backup fills from golden wallets (not all fills)
                    rows = conn.execute("""
                        SELECT wf.* FROM wallet_fills wf
                        JOIN golden_wallets gw ON wf.wallet_address = gw.address
                        WHERE gw.is_golden = 1
                        ORDER BY wf.time_ms
                    """).fetchall()
                    data["wallet_fills"] = [dict(r) for r in rows]

                if _table_exists("calibration_records"):
                    rows = conn.execute(
                        "SELECT * FROM calibration_records ORDER BY timestamp DESC LIMIT 100"
                    ).fetchall()
                    data["calibration_records"] = [dict(r) for r in rows]
        except Exception as e:
            print(f"Warning: could not backup golden/calibration data: {e}")

        with open(filepath, "w") as f:
            json.dump(data, f)

        size_kb = os.path.getsize(filepath) / 1024
        counts = (f"{len(data.get('traders', []))} traders, "
                  f"{len(data.get('bot_traders', []))} bots, "
                  f"{len(data.get('strategies', []))} strategies, "
                  f"{len(data.get('golden_wallets', []))} golden wallets, "
                  f"{len(data.get('wallet_fills', []))} fills")
        print(f"DB backup ({size_kb:.0f} KB): {counts} → {filepath}")
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
        fills_count = 0
        if data.get("golden_wallets"):
            try:
                from src.discovery.golden_wallet import init_golden_tables
                init_golden_tables()

                with get_connection() as conn:
                    for gw in data["golden_wallets"]:
                        try:
                            conn.execute("""
                                INSERT OR IGNORE INTO golden_wallets
                                (address, penalised_pnl, raw_pnl, sharpe_ratio,
                                 max_drawdown_pct, penalised_max_drawdown_pct,
                                 win_rate, trades_per_day, is_golden, coins_traded,
                                 best_coin, evaluated_at, connected_to_live)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
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
                                gw.get("evaluated_at", datetime.utcnow().isoformat()),
                                gw.get("connected_to_live", 0),
                            ))
                            golden_count += 1
                        except Exception:
                            pass
            except Exception as e:
                print(f"Warning: could not restore golden wallets: {e}")

        # Restore wallet fills (v2 backup)
        # Schema: wallet_address, coin, side, original_price, penalised_price,
        #         size, time_ms, delayed_time_ms, closed_pnl, penalised_pnl,
        #         fee, is_liquidation, direction
        if data.get("wallet_fills"):
            try:
                with get_connection() as conn:
                    for fill in data["wallet_fills"]:
                        try:
                            conn.execute("""
                                INSERT OR IGNORE INTO wallet_fills
                                (wallet_address, coin, side, original_price,
                                 penalised_price, size, time_ms, delayed_time_ms,
                                 closed_pnl, penalised_pnl, fee, is_liquidation,
                                 direction)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        except Exception:
                            pass
            except Exception as e:
                print(f"Warning: could not restore wallet fills: {e}")

        print(f"Restored DB from backup: {len(data.get('traders', []))} traders, "
              f"{len(data.get('bot_traders', []))} bots, "
              f"{len(data.get('strategies', []))} strategies, "
              f"{golden_count} golden wallets, {fills_count} fills")
        return True
    except Exception as e:
        print(f"Restore failed: {e}")
        return False


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {get_db_path()}")
