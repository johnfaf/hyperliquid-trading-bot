"""
SQLite database layer for persisting traders, strategies, scores, and paper trades.
"""
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_db_path():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    return config.DB_PATH


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


def close_paper_trade(trade_id, exit_price, pnl):
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""
            UPDATE paper_trades SET closed_at = ?, exit_price = ?, pnl = ?, status = 'closed'
            WHERE id = ?
        """, (now, exit_price, pnl, trade_id))


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


# ─── Backup & Restore (for Railway persistence) ─────────────

def backup_to_json(filepath: str = None):
    """Backup critical DB state to a JSON file for Railway persistence."""
    if filepath is None:
        filepath = os.environ.get("HL_BOT_BACKUP", "/tmp/bot_backup.json")
    try:
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "paper_account": get_paper_account(),
            "traders": get_active_traders()[:100],
            "strategies": get_active_strategies()[:200],
            "open_trades": get_open_paper_trades(),
            "closed_trades": get_paper_trade_history(limit=200),
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Backup failed: {e}")


def restore_from_json(filepath: str = None):
    """Restore DB state from a backup JSON file if DB is empty."""
    if filepath is None:
        filepath = os.environ.get("HL_BOT_BACKUP", "/tmp/bot_backup.json")
    if not os.path.exists(filepath):
        return False

    # Only restore if DB is empty (fresh deploy)
    account = get_paper_account()
    if account:
        return False  # DB already has data

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

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

        # Restore traders
        for t in data.get("traders", []):
            upsert_trader(
                t["address"],
                total_pnl=t.get("total_pnl", 0),
                roi_pct=t.get("roi_pct", 0),
                account_value=t.get("account_value", 0),
                win_rate=t.get("win_rate", 0),
                trade_count=t.get("trade_count", 0),
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

        print(f"Restored DB from backup: {len(data.get('traders', []))} traders, {len(data.get('strategies', []))} strategies")
        return True
    except Exception as e:
        print(f"Restore failed: {e}")
        return False


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {get_db_path()}")
