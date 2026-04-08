"""
SQLite database layer for persisting traders, strategies, scores, and paper trades.
"""
import sqlite3
import json
import os
import logging
import hashlib
import re
from datetime import datetime, timedelta
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.core.time_utils import utc_from_timestamp_naive, utc_now_naive

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
        return utc_now_naive().isoformat()
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1_000_000_000_000:
            timestamp /= 1000.0
        return utc_from_timestamp_naive(timestamp).isoformat()
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

        CREATE TABLE IF NOT EXISTS source_health_batches (
            snapshot_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            profile_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_source_health_batches_timestamp
            ON source_health_batches(timestamp);

        CREATE TABLE IF NOT EXISTS source_health_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            source_key TEXT NOT NULL,
            source TEXT,
            status TEXT DEFAULT 'warming_up',
            training_label TEXT DEFAULT 'monitor',
            recommended_action TEXT DEFAULT 'monitor',
            health_score REAL DEFAULT 0,
            weight_multiplier REAL DEFAULT 1,
            confidence_multiplier REAL DEFAULT 1,
            sample_size INTEGER DEFAULT 0,
            recent_sample_size INTEGER DEFAULT 0,
            selection_count INTEGER DEFAULT 0,
            closed_trades INTEGER DEFAULT 0,
            recent_closed_trades INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            recent_win_rate REAL DEFAULT 0,
            avg_return_pct REAL DEFAULT 0,
            recent_avg_return_pct REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            calibration_ece REAL,
            drift_score REAL DEFAULT 0,
            live_success_rate REAL DEFAULT 0,
            live_rejection_rate REAL DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (snapshot_id) REFERENCES source_health_batches(snapshot_id)
        );
        CREATE INDEX IF NOT EXISTS idx_source_health_profiles_snapshot
            ON source_health_profiles(snapshot_id, health_score DESC);
        CREATE INDEX IF NOT EXISTS idx_source_health_profiles_source
            ON source_health_profiles(source_key, snapshot_id);

        CREATE TABLE IF NOT EXISTS arena_review_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            agent_name TEXT,
            strategy_type TEXT,
            previous_status TEXT,
            new_status TEXT,
            action TEXT,
            reason TEXT,
            metrics TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_arena_review_events_timestamp
            ON arena_review_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_arena_review_events_agent
            ON arena_review_events(agent_id, timestamp);

        CREATE TABLE IF NOT EXISTS adaptive_recalibration_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            cycle_count INTEGER,
            profile_count INTEGER DEFAULT 0,
            transition_count INTEGER DEFAULT 0,
            promoted_count INTEGER DEFAULT 0,
            demoted_count INTEGER DEFAULT 0,
            held_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_adaptive_recalibration_runs_timestamp
            ON adaptive_recalibration_runs(timestamp);

        CREATE TABLE IF NOT EXISTS adaptive_promotion_states (
            source_key TEXT PRIMARY KEY,
            source TEXT,
            applied_stage TEXT DEFAULT 'trial',
            raw_stage TEXT DEFAULT 'trial',
            pending_stage TEXT,
            target_stage TEXT DEFAULT 'trial',
            last_good_stage TEXT DEFAULT 'trial',
            promotion_score REAL DEFAULT 0,
            gate_passed INTEGER DEFAULT 0,
            consecutive_promote_runs INTEGER DEFAULT 0,
            consecutive_demote_runs INTEGER DEFAULT 0,
            hold_count INTEGER DEFAULT 0,
            last_transition_at TEXT,
            cooldown_until TEXT,
            last_recalibrated_at TEXT,
            run_id TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_adaptive_promotion_states_stage
            ON adaptive_promotion_states(applied_stage, last_recalibrated_at);

        CREATE TABLE IF NOT EXISTS daily_research_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            cycle_count INTEGER,
            status TEXT DEFAULT 'executed',
            recommendation TEXT DEFAULT 'hold',
            winner_profile TEXT DEFAULT 'baseline_current',
            approved_profile_count INTEGER DEFAULT 0,
            recalibration_run_id TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_daily_research_runs_timestamp
            ON daily_research_runs(timestamp);

        CREATE TABLE IF NOT EXISTS shadow_certification_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            lookback_days INTEGER DEFAULT 7,
            status TEXT DEFAULT 'warming_up',
            certified INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_shadow_certification_runs_timestamp
            ON shadow_certification_runs(timestamp);

        CREATE TABLE IF NOT EXISTS capital_ramp_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            cycle_count INTEGER,
            status TEXT DEFAULT 'warming_up',
            applied_stage TEXT DEFAULT 'bootstrap',
            approved_stage TEXT DEFAULT 'bootstrap',
            recommended_stage TEXT DEFAULT 'bootstrap',
            deployable INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_capital_ramp_runs_timestamp
            ON capital_ramp_runs(timestamp);

        CREATE TABLE IF NOT EXISTS merge_readiness_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            cycle_count INTEGER,
            status TEXT DEFAULT 'hold',
            deployable INTEGER DEFAULT 0,
            branch_name TEXT DEFAULT '',
            commit_hash TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_merge_readiness_runs_timestamp
            ON merge_readiness_runs(timestamp);
        """)


# ─── Trader CRUD ───────────────────────────────────────────────

def upsert_trader(address, total_pnl=0, roi_pct=0, account_value=0,
                  win_rate=0, trade_count=0, metadata=None, is_active=True):
    now = utc_now_naive().isoformat()
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
                     (utc_now_naive().isoformat(), address))


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
    now = utc_now_naive().isoformat()
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
                  total_pnl=0, trade_count=0, win_rate=0, sharpe_ratio=0,
                  discovered_at=None, last_scored=None, current_score=0,
                  active=1):
    discovered_time = discovered_at or utc_now_naive().isoformat()
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO strategies
            (name, description, strategy_type, parameters, discovered_at,
             last_scored, current_score, total_pnl, trade_count, win_rate,
             sharpe_ratio, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, description, strategy_type, json.dumps(parameters or {}),
              discovered_time, last_scored, current_score, total_pnl,
              trade_count, win_rate, sharpe_ratio, int(bool(active))))
        return cursor.lastrowid


def save_strategies_batch(strategies_data):
    """Batch insert multiple strategies in a single transaction."""
    now = utc_now_naive().isoformat()
    saved_ids = []
    with get_connection() as conn:
        for s in strategies_data:
            cursor = conn.execute("""
                INSERT INTO strategies
                (name, description, strategy_type, parameters, discovered_at,
                 last_scored, current_score, total_pnl, trade_count, win_rate,
                 sharpe_ratio, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (s["name"], s["description"], s["strategy_type"],
                  json.dumps(s.get("parameters") or {}),
                  s.get("discovered_at") or now,
                  s.get("last_scored"),
                  s.get("current_score", 0),
                  s.get("total_pnl", 0), s.get("trade_count", 0),
                  s.get("win_rate", 0), s.get("sharpe_ratio", 0),
                  int(bool(s.get("active", 1)))))
            saved_ids.append(cursor.lastrowid)
    return saved_ids


def update_strategy_score(strategy_id, score):
    now = utc_now_naive().isoformat()
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


def get_all_strategies():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY current_score DESC, discovered_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def backfill_strategy_coins_from_name(limit: int = 5000) -> int:
    """
    Backfill missing strategy coin metadata from deterministic name/description tokens.

    This avoids random coin inference during decisioning for legacy strategies that
    were saved without `parameters.coins`.
    """
    stopwords = {
        "alpha", "beta", "gamma", "delta", "epsilon", "theta", "kappa",
        "strategy", "signal", "model", "long", "short", "momentum", "reversion",
        "mean", "swing", "scalp", "scalping", "trend", "following", "breakout",
        "arb", "funding", "neutral", "concentrated", "bet", "copy", "trade",
        "trader", "flow", "options", "polymarket", "arena", "champion", "unknown",
        "legacy", "explicit", "coin", "coins", "param", "params", "profile",
        "without", "with", "token", "tradable", "available", "custom", "pattern",
        "name", "description", "no", "trading",
        "on", "in", "of", "to", "for", "from", "by", "at", "as", "and",
        "daily", "weekly", "monthly", "hourly",
        "1m", "3m", "5m", "15m", "30m", "45m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1mo",
    }

    def _is_plausible_coin_symbol(token: str) -> bool:
        value = str(token or "").strip()
        if not value:
            return False
        lowered = value.lower()
        if lowered in stopwords:
            return False
        if lowered.startswith("0x") or lowered == "unknown":
            return False
        if value[0].isdigit():
            return False
        if not value.isalnum():
            return False
        if len(value) < 2 or len(value) > 8:
            return False
        letters = sum(ch.isalpha() for ch in value)
        return letters >= 2
    updated = 0

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, strategy_type, parameters
            FROM strategies
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

        for row in rows:
            params_raw = row["parameters"]
            if isinstance(params_raw, str):
                try:
                    params = json.loads(params_raw)
                except Exception:
                    params = {}
            elif isinstance(params_raw, dict):
                params = dict(params_raw)
            else:
                params = {}

            existing_coins = params.get("coins", params.get("coins_traded", []))
            if isinstance(existing_coins, str):
                existing_coins = [existing_coins]
            normalized_existing = [
                str(c).strip().upper()
                for c in (existing_coins or [])
                if str(c).strip() and str(c).strip().lower() != "unknown"
            ]
            if normalized_existing:
                continue

            text_parts = [
                str(row["name"] or ""),
                str(row["description"] or ""),
                str(row["strategy_type"] or ""),
            ]
            tokens = re.split(r"[^A-Za-z0-9]+", " ".join(text_parts).lower())
            inferred: list[str] = []
            for token in reversed(tokens):
                token = token.strip()
                if not _is_plausible_coin_symbol(token):
                    continue
                coin = token.upper()
                if coin not in inferred:
                    inferred.append(coin)
                if len(inferred) >= 3:
                    break

            if not inferred:
                continue

            params["coins"] = inferred
            conn.execute(
                "UPDATE strategies SET parameters = ? WHERE id = ?",
                (json.dumps(params), row["id"]),
            )
            updated += 1

    return updated


def get_strategy(strategy_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    return dict(row) if row else None


# ─── Strategy Scores ──────────────────────────────────────────

def save_strategy_score(strategy_id, score, pnl_score=0, win_rate_score=0,
                        sharpe_score=0, consistency_score=0, risk_adj_score=0, notes=""):
    now = utc_now_naive().isoformat()
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
    now = utc_now_naive().isoformat()
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
    now = utc_now_naive().isoformat()
    with get_connection() as conn:
        conn.execute("""
            UPDATE paper_account
            SET balance = ?, total_pnl = ?, total_trades = ?, winning_trades = ?, last_updated = ?
            WHERE id = 1
        """, (balance, total_pnl, total_trades, winning_trades, now))


def open_paper_trade(strategy_id, coin, side, entry_price, size, leverage=1,
                     stop_loss=None, take_profit=None, metadata=None):
    now = utc_now_naive().isoformat()
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
    now = utc_now_naive().isoformat()
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
        """, (initial_balance, utc_now_naive().isoformat()))

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
        since = (utc_now_naive() - timedelta(hours=float(lookback_hours))).isoformat()
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
        since = (utc_now_naive() - timedelta(hours=float(lookback_hours))).isoformat()
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
        cutoff = utc_now_naive() - timedelta(hours=float(lookback_hours))

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


def get_source_trade_outcome_summary(lookback_hours: float = 24.0 * 30) -> list:
    cutoff = None
    if lookback_hours and float(lookback_hours) > 0:
        cutoff = utc_now_naive() - timedelta(hours=float(lookback_hours))

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT coin, side, entry_price, size, pnl, status, metadata, opened_at, closed_at
                FROM paper_trades
                """
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    by_source = {}

    def _bucket_for(source_key: str, source: str):
        key = str(source_key or source or "unknown").strip() or "unknown"
        bucket = by_source.setdefault(
            key,
            {
                "source_key": key,
                "source": str(source or "unknown").strip().lower() or "unknown",
                "open_trades": 0,
                "closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "realized_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "avg_return_pct": 0.0,
                "_return_total": 0.0,
                "last_closed_at": None,
            },
        )
        if source and bucket["source"] == "unknown":
            bucket["source"] = str(source).strip().lower() or "unknown"
        return bucket

    for row in rows:
        metadata = {}
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}

        source_key = str(metadata.get("source_key", "") or "").strip()
        source = str(metadata.get("source", metadata.get("signal_source", "")) or "").strip().lower()
        bucket = _bucket_for(source_key, source)

        status = str(row["status"] or "").lower()
        opened_at = _parse_iso_timestamp(row["opened_at"])
        closed_at = _parse_iso_timestamp(row["closed_at"])

        if status == "open":
            if cutoff is None or (opened_at and opened_at >= cutoff):
                bucket["open_trades"] += 1
            continue

        if status != "closed":
            continue
        if cutoff is not None and (closed_at is None or closed_at < cutoff):
            continue

        pnl = _safe_float(row["pnl"], 0.0)
        entry_price = _safe_float(row["entry_price"], 0.0)
        size = abs(_safe_float(row["size"], 0.0))
        notional = max(entry_price * max(size, 1e-8), 1e-8)
        return_pct = pnl / notional if notional > 0 else 0.0

        bucket["closed_trades"] += 1
        bucket["realized_pnl"] += pnl
        bucket["_return_total"] += return_pct
        if pnl > 0:
            bucket["winning_trades"] += 1
            bucket["gross_profit"] += pnl
        elif pnl < 0:
            bucket["losing_trades"] += 1
            bucket["gross_loss"] += abs(pnl)
        if closed_at:
            last_closed = bucket.get("last_closed_at")
            if not last_closed or str(closed_at.isoformat()) > str(last_closed):
                bucket["last_closed_at"] = closed_at.isoformat()

    results = []
    for bucket in by_source.values():
        closed_trades = int(bucket["closed_trades"] or 0)
        bucket["win_rate"] = round(
            bucket["winning_trades"] / closed_trades,
            4,
        ) if closed_trades else 0.0
        bucket["profit_factor"] = round(
            bucket["gross_profit"] / max(bucket["gross_loss"], 1e-8),
            4,
        ) if bucket["gross_profit"] > 0 and bucket["gross_loss"] > 0 else (
            999.0 if bucket["gross_profit"] > 0 and bucket["gross_loss"] == 0 else 0.0
        )
        bucket["avg_return_pct"] = round(
            bucket["_return_total"] / closed_trades,
            4,
        ) if closed_trades else 0.0
        bucket["realized_pnl"] = round(bucket["realized_pnl"], 2)
        bucket.pop("_return_total", None)
        results.append(bucket)

    results.sort(
        key=lambda item: (
            -item["closed_trades"],
            -item["realized_pnl"],
            item["source_key"],
        )
    )
    return results


def get_context_trade_outcome_summary(
    *,
    lookback_hours: float = 24.0 * 30,
    source_key: str = "",
    source: str = "",
    coin: str = "",
    side: str = "",
    regime: str = "",
    min_closed_trades: int = 0,
) -> list:
    cutoff = None
    if lookback_hours and float(lookback_hours) > 0:
        cutoff = utc_now_naive() - timedelta(hours=float(lookback_hours))

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT coin, side, entry_price, size, pnl, status, metadata, opened_at, closed_at
                FROM paper_trades
                """
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    by_context = {}

    def _bucket_for(source_key_value, source_value, coin_value, side_value, regime_value):
        context_key = "|".join(
            [
                str(source_key_value or source_value or "unknown").strip() or "unknown",
                str(coin_value or "unknown").strip().upper() or "UNKNOWN",
                str(side_value or "unknown").strip().lower() or "unknown",
                str(regime_value or "unknown").strip().lower() or "unknown",
            ]
        )
        return by_context.setdefault(
            context_key,
            {
                "context_key": context_key,
                "source_key": str(source_key_value or source_value or "unknown").strip() or "unknown",
                "source": str(source_value or "unknown").strip().lower() or "unknown",
                "coin": str(coin_value or "unknown").strip().upper() or "UNKNOWN",
                "side": str(side_value or "unknown").strip().lower() or "unknown",
                "regime": str(regime_value or "unknown").strip().lower() or "unknown",
                "open_trades": 0,
                "closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "realized_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "avg_return_pct": 0.0,
                "_return_total": 0.0,
                "last_closed_at": None,
            },
        )

    for row in rows:
        metadata = {}
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}

        source_key_value = str(metadata.get("source_key", "") or "").strip()
        source_value = str(metadata.get("source", metadata.get("signal_source", "")) or "").strip().lower()
        coin_value = str(row["coin"] or metadata.get("coin", "") or "").strip().upper()
        side_value = str(row["side"] or metadata.get("side", "") or "").strip().lower()
        regime_value = str(metadata.get("regime", metadata.get("overall_regime", "")) or "").strip().lower()
        bucket = _bucket_for(source_key_value, source_value, coin_value, side_value, regime_value)

        status = str(row["status"] or "").lower()
        opened_at = _parse_iso_timestamp(row["opened_at"])
        closed_at = _parse_iso_timestamp(row["closed_at"])

        if status == "open":
            if cutoff is None or (opened_at and opened_at >= cutoff):
                bucket["open_trades"] += 1
            continue

        if status != "closed":
            continue
        if cutoff is not None and (closed_at is None or closed_at < cutoff):
            continue

        pnl = _safe_float(row["pnl"], 0.0)
        entry_price = _safe_float(row["entry_price"], 0.0)
        size = abs(_safe_float(row["size"], 0.0))
        notional = max(entry_price * max(size, 1e-8), 1e-8)
        return_pct = pnl / notional if notional > 0 else 0.0

        bucket["closed_trades"] += 1
        bucket["realized_pnl"] += pnl
        bucket["_return_total"] += return_pct
        if pnl > 0:
            bucket["winning_trades"] += 1
            bucket["gross_profit"] += pnl
        elif pnl < 0:
            bucket["losing_trades"] += 1
            bucket["gross_loss"] += abs(pnl)
        if closed_at:
            last_closed = bucket.get("last_closed_at")
            if not last_closed or str(closed_at.isoformat()) > str(last_closed):
                bucket["last_closed_at"] = closed_at.isoformat()

    normalized_filters = {
        "source_key": str(source_key or "").strip(),
        "source": str(source or "").strip().lower(),
        "coin": str(coin or "").strip().upper(),
        "side": str(side or "").strip().lower(),
        "regime": str(regime or "").strip().lower(),
    }

    results = []
    for bucket in by_context.values():
        closed_trades = int(bucket["closed_trades"] or 0)
        bucket["win_rate"] = round(
            bucket["winning_trades"] / closed_trades,
            4,
        ) if closed_trades else 0.0
        bucket["profit_factor"] = round(
            bucket["gross_profit"] / max(bucket["gross_loss"], 1e-8),
            4,
        ) if bucket["gross_profit"] > 0 and bucket["gross_loss"] > 0 else (
            999.0 if bucket["gross_profit"] > 0 and bucket["gross_loss"] == 0 else 0.0
        )
        bucket["avg_return_pct"] = round(
            bucket["_return_total"] / closed_trades,
            4,
        ) if closed_trades else 0.0
        bucket["realized_pnl"] = round(bucket["realized_pnl"], 2)
        bucket.pop("_return_total", None)

        if min_closed_trades and closed_trades < int(min_closed_trades):
            continue
        if normalized_filters["source_key"] and bucket["source_key"] != normalized_filters["source_key"]:
            continue
        if normalized_filters["source"] and bucket["source"] != normalized_filters["source"]:
            continue
        if normalized_filters["coin"] and bucket["coin"] != normalized_filters["coin"]:
            continue
        if normalized_filters["side"] and bucket["side"] != normalized_filters["side"]:
            continue
        if normalized_filters["regime"] and bucket["regime"] != normalized_filters["regime"]:
            continue
        results.append(bucket)

    results.sort(
        key=lambda item: (
            -item["closed_trades"],
            -item["realized_pnl"],
            item["context_key"],
        )
    )
    return results


def save_source_health_snapshot(snapshot: dict) -> str:
    timestamp = _normalize_live_timestamp(snapshot.get("timestamp"))
    profiles = snapshot.get("profiles", []) or []
    metadata = snapshot.get("metadata", {}) or {}
    snapshot_id = str(
        snapshot.get("snapshot_id")
        or f"source-health:{hashlib.sha256(f'{timestamp}:{len(profiles)}'.encode()).hexdigest()[:16]}"
    )

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO source_health_batches
            (snapshot_id, timestamp, profile_count, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                snapshot_id,
                timestamp,
                len(profiles),
                json.dumps(metadata, sort_keys=True, default=str),
            ),
        )
        conn.execute(
            "DELETE FROM source_health_profiles WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        for profile in profiles:
            conn.execute(
                """
                INSERT INTO source_health_profiles
                (snapshot_id, source_key, source, status, training_label, recommended_action,
                 health_score, weight_multiplier, confidence_multiplier, sample_size,
                 recent_sample_size, selection_count, closed_trades, recent_closed_trades,
                 win_rate, recent_win_rate, avg_return_pct, recent_avg_return_pct,
                 realized_pnl, calibration_ece, drift_score, live_success_rate,
                 live_rejection_rate, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    str(profile.get("source_key", "") or "").strip() or "unknown",
                    str(profile.get("source", "") or "").strip().lower() or "unknown",
                    str(profile.get("status", "warming_up") or "warming_up").strip().lower(),
                    str(profile.get("training_label", "monitor") or "monitor").strip().lower(),
                    str(profile.get("recommended_action", "monitor") or "monitor").strip().lower(),
                    _safe_float(profile.get("health_score"), 0.0),
                    _safe_float(profile.get("weight_multiplier"), 1.0),
                    _safe_float(profile.get("confidence_multiplier"), 1.0),
                    int(profile.get("sample_size", 0) or 0),
                    int(profile.get("recent_sample_size", 0) or 0),
                    int(profile.get("selection_count", 0) or 0),
                    int(profile.get("closed_trades", 0) or 0),
                    int(profile.get("recent_closed_trades", 0) or 0),
                    _safe_float(profile.get("win_rate"), 0.0),
                    _safe_float(profile.get("recent_win_rate"), 0.0),
                    _safe_float(profile.get("avg_return_pct"), 0.0),
                    _safe_float(profile.get("recent_avg_return_pct"), 0.0),
                    _safe_float(profile.get("realized_pnl"), 0.0),
                    (
                        None
                        if profile.get("calibration_ece") in (None, "")
                        else _safe_float(profile.get("calibration_ece"), 0.0)
                    ),
                    _safe_float(profile.get("drift_score"), 0.0),
                    _safe_float(profile.get("live_success_rate"), 0.0),
                    _safe_float(profile.get("live_rejection_rate"), 0.0),
                    json.dumps(profile.get("metadata", {}) or {}, sort_keys=True, default=str),
                ),
            )
    return snapshot_id


def get_latest_source_health_snapshot(limit: int = 25) -> dict:
    try:
        with get_connection() as conn:
            batch = conn.execute(
                """
                SELECT * FROM source_health_batches
                ORDER BY timestamp DESC, snapshot_id DESC
                LIMIT 1
                """
            ).fetchone()
            if not batch:
                return {"snapshot": None, "profiles": []}
            rows = conn.execute(
                """
                SELECT * FROM source_health_profiles
                WHERE snapshot_id = ?
                ORDER BY health_score DESC, sample_size DESC, source_key ASC
                LIMIT ?
                """,
                (batch["snapshot_id"], int(limit)),
            ).fetchall()
    except sqlite3.OperationalError:
        return {"snapshot": None, "profiles": []}

    snapshot = dict(batch)
    try:
        snapshot["metadata"] = json.loads(snapshot.get("metadata") or "{}")
    except Exception:
        snapshot["metadata"] = {}

    profiles = []
    for row in rows:
        payload = dict(row)
        try:
            payload["metadata"] = json.loads(payload.get("metadata") or "{}")
        except Exception:
            payload["metadata"] = {}
        for key in (
            "promotion_stage",
            "promotion_score",
            "promotion_gate_passed",
            "promotion_multiplier",
            "promotion_cap_pct",
            "promotion_reasons",
            "raw_promotion_stage",
            "raw_promotion_score",
            "raw_promotion_gate_passed",
            "raw_promotion_reasons",
            "promotion_target_stage",
            "promotion_pending_stage",
            "promotion_last_good_stage",
            "promotion_last_transition_at",
            "promotion_cooldown_until",
            "promotion_transition_state",
            "promotion_confirmed_runs",
        ):
            if key in payload["metadata"] and key not in payload:
                payload[key] = payload["metadata"][key]
        profiles.append(payload)

    return {
        "snapshot": snapshot,
        "profiles": profiles,
    }


def save_arena_review_events(events: list) -> int:
    if not events:
        return 0
    inserted = 0
    with get_connection() as conn:
        for event in events:
            conn.execute(
                """
                INSERT INTO arena_review_events
                (timestamp, agent_id, agent_name, strategy_type, previous_status,
                 new_status, action, reason, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _normalize_live_timestamp(event.get("timestamp")),
                    str(event.get("agent_id", "") or "").strip(),
                    str(event.get("agent_name", "") or "").strip(),
                    str(event.get("strategy_type", "") or "").strip(),
                    str(event.get("previous_status", "") or "").strip().lower(),
                    str(event.get("new_status", "") or "").strip().lower(),
                    str(event.get("action", "") or "").strip().lower(),
                    str(event.get("reason", "") or "").strip(),
                    json.dumps(event.get("metrics", {}) or {}, sort_keys=True, default=str),
                ),
            )
            inserted += 1
    return inserted


def get_recent_arena_review_events(limit: int = 20) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM arena_review_events
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    events = []
    for row in rows:
        payload = dict(row)
        try:
            payload["metrics"] = json.loads(payload.get("metrics") or "{}")
        except Exception:
            payload["metrics"] = {}
        events.append(payload)
    return events


def save_adaptive_recalibration_run(payload: dict) -> str:
    timestamp = _normalize_live_timestamp(payload.get("timestamp"))
    profile_count = int(payload.get("profile_count", 0) or 0)
    default_key = f"{timestamp}:{profile_count}"
    run_id = str(
        payload.get("run_id")
        or f"adaptive-recalibration:{hashlib.sha256(default_key.encode()).hexdigest()[:16]}"
    )
    metadata = payload.get("metadata", {}) or {}

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO adaptive_recalibration_runs
            (run_id, timestamp, cycle_count, profile_count, transition_count,
             promoted_count, demoted_count, held_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                int(payload.get("cycle_count", 0) or 0) if payload.get("cycle_count") is not None else None,
                profile_count,
                int(payload.get("transition_count", 0) or 0),
                int(payload.get("promoted_count", 0) or 0),
                int(payload.get("demoted_count", 0) or 0),
                int(payload.get("held_count", 0) or 0),
                json.dumps(metadata, sort_keys=True, default=str),
            ),
        )
    return run_id


def get_latest_adaptive_recalibration_run() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM adaptive_recalibration_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return {}

    if not row:
        return {}

    payload = dict(row)
    try:
        payload["metadata"] = json.loads(payload.get("metadata") or "{}")
    except Exception:
        payload["metadata"] = {}
    return payload


def get_recent_adaptive_recalibration_runs(limit: int = 10) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM adaptive_recalibration_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        payload = dict(row)
        try:
            payload["metadata"] = json.loads(payload.get("metadata") or "{}")
        except Exception:
            payload["metadata"] = {}
        results.append(payload)
    return results


def save_adaptive_promotion_states(states: list) -> int:
    if not states:
        return 0

    updated = 0
    with get_connection() as conn:
        for state in states:
            conn.execute(
                """
                INSERT OR REPLACE INTO adaptive_promotion_states
                (source_key, source, applied_stage, raw_stage, pending_stage, target_stage,
                 last_good_stage, promotion_score, gate_passed, consecutive_promote_runs,
                 consecutive_demote_runs, hold_count, last_transition_at, cooldown_until,
                 last_recalibrated_at, run_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(state.get("source_key", "") or "").strip(),
                    str(state.get("source", "") or "").strip().lower() or "unknown",
                    str(state.get("applied_stage", "trial") or "trial").strip().lower(),
                    str(state.get("raw_stage", "trial") or "trial").strip().lower(),
                    (
                        str(state.get("pending_stage", "") or "").strip().lower()
                        or None
                    ),
                    str(state.get("target_stage", "trial") or "trial").strip().lower(),
                    str(state.get("last_good_stage", "trial") or "trial").strip().lower(),
                    _safe_float(state.get("promotion_score"), 0.0),
                    1 if bool(state.get("gate_passed", False)) else 0,
                    int(state.get("consecutive_promote_runs", 0) or 0),
                    int(state.get("consecutive_demote_runs", 0) or 0),
                    int(state.get("hold_count", 0) or 0),
                    _normalize_live_timestamp(state.get("last_transition_at")),
                    _normalize_live_timestamp(state.get("cooldown_until")),
                    _normalize_live_timestamp(state.get("last_recalibrated_at")),
                    str(state.get("run_id", "") or "").strip() or None,
                    json.dumps(state.get("metadata", {}) or {}, sort_keys=True, default=str),
                ),
            )
            updated += 1
    return updated


def get_adaptive_promotion_states(limit: int = 0) -> list:
    query = """
        SELECT * FROM adaptive_promotion_states
        ORDER BY
            CASE applied_stage
                WHEN 'full' THEN 4
                WHEN 'scaled' THEN 3
                WHEN 'trial' THEN 2
                WHEN 'incubating' THEN 1
                WHEN 'blocked' THEN 0
                ELSE -1
            END DESC,
            promotion_score DESC,
            source_key ASC
    """
    params = ()
    if int(limit or 0) > 0:
        query += "\nLIMIT ?"
        params = (int(limit),)

    try:
        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        payload = dict(row)
        try:
            payload["metadata"] = json.loads(payload.get("metadata") or "{}")
        except Exception:
            payload["metadata"] = {}
        payload["gate_passed"] = bool(payload.get("gate_passed", 0))
        results.append(payload)
    return results


def save_daily_research_run(payload: dict) -> str:
    timestamp = _normalize_live_timestamp(payload.get("timestamp"))
    status = str(payload.get("status", "executed") or "executed").strip().lower()
    recommendation = str(payload.get("recommendation", "hold") or "hold").strip().lower()
    winner_profile = str(
        payload.get("winner_profile")
        or ((payload.get("benchmark") or {}).get("promotion_gate", {}) or {}).get("winner")
        or "baseline_current"
    ).strip()
    approved_profile_count = int(payload.get("approved_profile_count", 0) or 0)
    cycle_count = payload.get("cycle_count")
    recalibration_run_id = str(payload.get("recalibration_run_id", "") or "").strip() or None
    default_key = f"{timestamp}:{winner_profile}:{recommendation}:{approved_profile_count}"
    run_id = str(
        payload.get("run_id")
        or f"daily-research:{hashlib.sha256(default_key.encode()).hexdigest()[:16]}"
    )
    metadata = payload.get("metadata", {}) or {}

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_research_runs
            (run_id, timestamp, cycle_count, status, recommendation, winner_profile,
             approved_profile_count, recalibration_run_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                int(cycle_count or 0) if cycle_count is not None else None,
                status,
                recommendation,
                winner_profile or "baseline_current",
                approved_profile_count,
                recalibration_run_id,
                json.dumps(metadata, sort_keys=True, default=str),
            ),
        )
    return run_id


def get_latest_daily_research_run() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM daily_research_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return {}

    if not row:
        return {}

    payload = dict(row)
    try:
        payload["metadata"] = json.loads(payload.get("metadata") or "{}")
    except Exception:
        payload["metadata"] = {}
    return payload


def get_recent_daily_research_runs(limit: int = 10) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM daily_research_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        payload = dict(row)
        try:
            payload["metadata"] = json.loads(payload.get("metadata") or "{}")
        except Exception:
            payload["metadata"] = {}
        results.append(payload)
    return results


def save_daily_research_last_known_good(payload: dict) -> None:
    value = json.dumps(payload or {}, sort_keys=True, default=str)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bot_state (key, value)
            VALUES ('daily_research_last_known_good', ?)
            """,
            (value,),
        )


def get_daily_research_last_known_good() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT value FROM bot_state
                WHERE key = 'daily_research_last_known_good'
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return {}

    if not row:
        return {}
    try:
        return json.loads(row["value"] or "{}")
    except Exception:
        return {}


def save_capital_ramp_state(payload: dict) -> None:
    value = json.dumps(payload or {}, sort_keys=True, default=str)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bot_state (key, value)
            VALUES ('capital_ramp_state', ?)
            """,
            (value,),
        )


def get_capital_ramp_state() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT value FROM bot_state
                WHERE key = 'capital_ramp_state'
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return {}

    if not row:
        return {}
    try:
        return json.loads(row["value"] or "{}")
    except Exception:
        return {}


def save_shadow_certification_run(payload: dict) -> str:
    timestamp = _normalize_live_timestamp(payload.get("timestamp"))
    status = str(payload.get("status", "warming_up") or "warming_up").strip().lower()
    certified = 1 if bool(payload.get("certified", False)) else 0
    lookback_days = int(payload.get("lookback_days", 7) or 7)
    default_key = f"{timestamp}:{status}:{lookback_days}:{certified}"
    run_id = str(
        payload.get("run_id")
        or f"shadow-cert:{hashlib.sha256(default_key.encode()).hexdigest()[:16]}"
    )
    metadata = payload.get("metadata", {}) or {}

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO shadow_certification_runs
            (run_id, timestamp, lookback_days, status, certified, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                lookback_days,
                status,
                certified,
                json.dumps(metadata, sort_keys=True, default=str),
            ),
        )
    return run_id


def get_latest_shadow_certification_run() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM shadow_certification_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return {}

    if not row:
        return {}
    payload = dict(row)
    payload["certified"] = bool(payload.get("certified", 0))
    try:
        payload["metadata"] = json.loads(payload.get("metadata") or "{}")
    except Exception:
        payload["metadata"] = {}
    return payload


def get_recent_shadow_certification_runs(limit: int = 10) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM shadow_certification_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        payload = dict(row)
        payload["certified"] = bool(payload.get("certified", 0))
        try:
            payload["metadata"] = json.loads(payload.get("metadata") or "{}")
        except Exception:
            payload["metadata"] = {}
        results.append(payload)
    return results


def save_capital_ramp_run(payload: dict) -> str:
    timestamp = _normalize_live_timestamp(payload.get("timestamp"))
    status = str(payload.get("status", "warming_up") or "warming_up").strip().lower()
    applied_stage = str(payload.get("applied_stage", "bootstrap") or "bootstrap").strip().lower()
    approved_stage = str(payload.get("approved_stage", "bootstrap") or "bootstrap").strip().lower()
    recommended_stage = str(
        payload.get("recommended_stage", applied_stage or "bootstrap") or applied_stage or "bootstrap"
    ).strip().lower()
    deployable = 1 if bool(payload.get("deployable", False)) else 0
    cycle_count = payload.get("cycle_count")
    default_key = f"{timestamp}:{status}:{applied_stage}:{approved_stage}:{recommended_stage}:{deployable}"
    run_id = str(
        payload.get("run_id")
        or f"capital-ramp:{hashlib.sha256(default_key.encode()).hexdigest()[:16]}"
    )
    metadata = payload.get("metadata", {}) or {}

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO capital_ramp_runs
            (run_id, timestamp, cycle_count, status, applied_stage, approved_stage,
             recommended_stage, deployable, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                int(cycle_count or 0) if cycle_count is not None else None,
                status,
                applied_stage,
                approved_stage,
                recommended_stage,
                deployable,
                json.dumps(metadata, sort_keys=True, default=str),
            ),
        )
    return run_id


def get_latest_capital_ramp_run() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM capital_ramp_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return {}

    if not row:
        return {}

    payload = dict(row)
    payload["deployable"] = bool(payload.get("deployable", 0))
    try:
        payload["metadata"] = json.loads(payload.get("metadata") or "{}")
    except Exception:
        payload["metadata"] = {}
    return payload


def get_recent_capital_ramp_runs(limit: int = 10) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM capital_ramp_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        payload = dict(row)
        payload["deployable"] = bool(payload.get("deployable", 0))
        try:
            payload["metadata"] = json.loads(payload.get("metadata") or "{}")
        except Exception:
            payload["metadata"] = {}
        results.append(payload)
    return results


def save_merge_readiness_run(payload: dict) -> str:
    timestamp = _normalize_live_timestamp(payload.get("timestamp"))
    status = str(payload.get("status", "hold") or "hold").strip().lower()
    deployable = 1 if bool(payload.get("deployable_for_merge", payload.get("deployable", False))) else 0
    branch_name = str(
        payload.get("branch_name")
        or ((payload.get("git", {}) or {}).get("branch"))
        or ""
    ).strip()
    commit_hash = str(
        payload.get("commit_hash")
        or ((payload.get("git", {}) or {}).get("commit"))
        or ""
    ).strip()
    cycle_count = payload.get("cycle_count")
    default_key = f"{timestamp}:{status}:{deployable}:{branch_name}:{commit_hash}"
    run_id = str(
        payload.get("run_id")
        or f"merge-readiness:{hashlib.sha256(default_key.encode()).hexdigest()[:16]}"
    )
    metadata = payload.get("metadata", {}) or {}

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO merge_readiness_runs
            (run_id, timestamp, cycle_count, status, deployable, branch_name, commit_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                int(cycle_count or 0) if cycle_count is not None else None,
                status,
                deployable,
                branch_name,
                commit_hash,
                json.dumps(metadata, sort_keys=True, default=str),
            ),
        )
    return run_id


def get_latest_merge_readiness_run() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM merge_readiness_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return {}

    if not row:
        return {}
    payload = dict(row)
    payload["deployable"] = bool(payload.get("deployable", 0))
    payload["deployable_for_merge"] = payload["deployable"]
    try:
        payload["metadata"] = json.loads(payload.get("metadata") or "{}")
    except Exception:
        payload["metadata"] = {}
    payload["summary"] = str(payload["metadata"].get("summary", "") or "")
    payload["strict"] = bool(payload["metadata"].get("strict", False))
    return payload


def get_recent_merge_readiness_runs(limit: int = 10) -> list:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM merge_readiness_runs
                ORDER BY timestamp DESC, run_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        payload = dict(row)
        payload["deployable"] = bool(payload.get("deployable", 0))
        payload["deployable_for_merge"] = payload["deployable"]
        try:
            payload["metadata"] = json.loads(payload.get("metadata") or "{}")
        except Exception:
            payload["metadata"] = {}
        payload["summary"] = str(payload["metadata"].get("summary", "") or "")
        payload["strict"] = bool(payload["metadata"].get("strict", False))
        results.append(payload)
    return results


def get_runtime_divergence_summary(lookback_hours: float = 24.0) -> dict:
    cutoff = None
    if lookback_hours and float(lookback_hours) > 0:
        cutoff = utc_now_naive() - timedelta(hours=float(lookback_hours))

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
    live_rejection_rate = round(
        live_rejection_count / max(1, live_execution_total),
        4,
    ) if live_execution_total > 0 else 0.0
    live_success_rate = round(
        live_success_count / max(1, live_execution_total),
        4,
    ) if live_execution_total > 0 else 0.0

    paper_account = get_paper_account() or {}
    paper_total_pnl = _safe_float(paper_account.get("total_pnl"), 0.0)
    live_fill_summary = get_live_fill_summary()
    live_realized_pnl = _safe_float(live_fill_summary.get("realized_pnl"), 0.0)

    paper_live_gap = paper_open_count - live_open_positions
    shadow_live_gap = shadow_selected_count - live_execution_total
    realized_pnl_gap = round(paper_total_pnl - live_realized_pnl, 2)

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
        "live_success_rate": live_success_rate,
        "live_rejection_rate": live_rejection_rate,
        "paper_total_pnl": round(paper_total_pnl, 2),
        "live_realized_pnl": round(live_realized_pnl, 2),
        "paper_live_realized_pnl_gap": realized_pnl_gap,
        "paper_live_realized_pnl_gap_ratio": round(
            abs(realized_pnl_gap) / max(1.0, abs(paper_total_pnl), abs(live_realized_pnl)),
            4,
        ),
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

def _compute_drawdown_metrics(values: list) -> dict:
    clean_values = [_safe_float(value, 0.0) for value in values or [] if _safe_float(value, 0.0) > 0]
    if not clean_values:
        return {
            "peak_value": 0.0,
            "current_value": 0.0,
            "current_drawdown_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    peak_value = clean_values[0]
    max_drawdown_pct = 0.0
    for value in clean_values:
        peak_value = max(peak_value, value)
        if peak_value > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak_value - value) / peak_value)

    current_value = clean_values[-1]
    current_drawdown_pct = ((peak_value - current_value) / peak_value) if peak_value > 0 else 0.0
    return {
        "peak_value": round(peak_value, 4),
        "current_value": round(current_value, 4),
        "current_drawdown_pct": round(max(0.0, current_drawdown_pct), 4),
        "max_drawdown_pct": round(max(0.0, max_drawdown_pct), 4),
    }


def _compute_return_stats(returns: list) -> dict:
    clean_returns = [float(value) for value in returns or [] if value is not None]
    if not clean_returns:
        return {
            "avg_return_pct": 0.0,
            "sharpe": 0.0,
            "volatility_pct": 0.0,
        }

    avg_return = sum(clean_returns) / len(clean_returns)
    if len(clean_returns) < 2:
        volatility = 0.0
    else:
        variance = sum((value - avg_return) ** 2 for value in clean_returns) / (len(clean_returns) - 1)
        volatility = variance ** 0.5

    sharpe = 0.0
    if volatility > 1e-12:
        sharpe = (avg_return / volatility) * (len(clean_returns) ** 0.5)

    return {
        "avg_return_pct": round(avg_return, 6),
        "sharpe": round(sharpe, 4),
        "volatility_pct": round(volatility, 6),
    }


def get_capital_governor_summary(lookback_hours: float = 24.0 * 14, health_limit: int = 200) -> dict:
    cutoff = None
    if lookback_hours and float(lookback_hours) > 0:
        cutoff = utc_now_naive() - timedelta(hours=float(lookback_hours))

    paper_account = get_paper_account() or {}
    paper_balance = _safe_float(paper_account.get("balance"), 0.0)
    paper_total_pnl = _safe_float(paper_account.get("total_pnl"), 0.0)
    inferred_initial_balance = paper_balance - paper_total_pnl
    if inferred_initial_balance <= 0:
        inferred_initial_balance = _safe_float(
            getattr(config, "PAPER_TRADING_INITIAL_BALANCE", 10_000),
            10_000.0,
        )

    try:
        with get_connection() as conn:
            paper_rows = conn.execute(
                """
                SELECT pnl, closed_at
                FROM paper_trades
                WHERE status = 'closed'
                ORDER BY COALESCE(closed_at, opened_at) ASC
                """
            ).fetchall()
    except sqlite3.OperationalError:
        paper_rows = []

    paper_returns = []
    paper_closed_count = 0
    paper_winning_trades = 0
    equity_value = inferred_initial_balance
    paper_equity_curve = [inferred_initial_balance]
    for row in paper_rows:
        closed_at = _parse_iso_timestamp(row["closed_at"])
        if cutoff and closed_at and closed_at < cutoff:
            continue
        pnl = _safe_float(row["pnl"], 0.0)
        denominator = max(abs(equity_value), 1.0)
        paper_returns.append(pnl / denominator)
        paper_closed_count += 1
        if pnl > 0:
            paper_winning_trades += 1
        equity_value += pnl
        paper_equity_curve.append(equity_value)

    if paper_balance > 0:
        paper_equity_curve.append(paper_balance)

    paper_drawdown = _compute_drawdown_metrics(paper_equity_curve)
    paper_return_stats = _compute_return_stats(paper_returns)

    live_snapshots = get_recent_live_equity_snapshots(
        limit=max(72, int(max(float(lookback_hours or 0.0), 1.0) * 12) + 24)
    )
    filtered_live_snapshots = []
    for snapshot in live_snapshots:
        snapshot_ts = _parse_iso_timestamp(snapshot.get("timestamp"))
        if cutoff and snapshot_ts and snapshot_ts < cutoff:
            continue
        filtered_live_snapshots.append(snapshot)

    live_totals = [
        _safe_float(snapshot.get("total"), 0.0)
        for snapshot in filtered_live_snapshots
        if _safe_float(snapshot.get("total"), 0.0) > 0
    ]
    live_returns = []
    for previous, current in zip(live_totals, live_totals[1:]):
        if previous > 0 and current > 0:
            live_returns.append((current - previous) / previous)

    live_drawdown = _compute_drawdown_metrics(live_totals)
    live_return_stats = _compute_return_stats(live_returns)
    latest_live_snapshot = filtered_live_snapshots[-1] if filtered_live_snapshots else None

    source_snapshot = get_latest_source_health_snapshot(limit=health_limit)
    profiles = list(source_snapshot.get("profiles", []) or [])
    status_counts = {"active": 0, "warming_up": 0, "caution": 0, "blocked": 0}
    avg_health_score = 0.0
    for profile in profiles:
        status = str(profile.get("status", "warming_up") or "warming_up").strip().lower() or "warming_up"
        status_counts[status] = status_counts.get(status, 0) + 1
        avg_health_score += _safe_float(profile.get("health_score"), 0.0)
    if profiles:
        avg_health_score /= len(profiles)

    total_profiles = len(profiles)
    degraded_source_ratio = (
        (status_counts.get("caution", 0) + status_counts.get("blocked", 0)) / max(total_profiles, 1)
        if total_profiles > 0
        else 0.0
    )
    blocked_source_ratio = (
        status_counts.get("blocked", 0) / max(total_profiles, 1)
        if total_profiles > 0
        else 0.0
    )

    return {
        "lookback_hours": float(lookback_hours or 0.0),
        "paper_balance": round(paper_balance, 2),
        "paper_total_pnl": round(paper_total_pnl, 2),
        "paper_initial_balance": round(inferred_initial_balance, 2),
        "paper_closed_trades": paper_closed_count,
        "paper_win_rate": (
            round(paper_winning_trades / max(paper_closed_count, 1), 4)
            if paper_closed_count > 0
            else 0.0
        ),
        "paper_avg_return_pct": paper_return_stats["avg_return_pct"],
        "paper_sharpe": paper_return_stats["sharpe"],
        "paper_volatility_pct": paper_return_stats["volatility_pct"],
        "paper_peak_balance": paper_drawdown["peak_value"],
        "paper_current_drawdown_pct": paper_drawdown["current_drawdown_pct"],
        "paper_max_drawdown_pct": paper_drawdown["max_drawdown_pct"],
        "live_snapshot_count": len(live_totals),
        "live_current_total": live_drawdown["current_value"],
        "live_peak_total": live_drawdown["peak_value"],
        "live_avg_return_pct": live_return_stats["avg_return_pct"],
        "live_sharpe": live_return_stats["sharpe"],
        "live_volatility_pct": live_return_stats["volatility_pct"],
        "live_current_drawdown_pct": live_drawdown["current_drawdown_pct"],
        "live_max_drawdown_pct": live_drawdown["max_drawdown_pct"],
        "live_latest_timestamp": latest_live_snapshot.get("timestamp") if latest_live_snapshot else None,
        "source_profile_count": total_profiles,
        "source_status_counts": status_counts,
        "avg_source_health_score": round(avg_health_score, 4),
        "degraded_source_ratio": round(degraded_source_ratio, 4),
        "blocked_source_ratio": round(blocked_source_ratio, 4),
        "source_snapshot_id": (source_snapshot.get("snapshot") or {}).get("snapshot_id"),
    }


def log_research_cycle(cycle_type, summary, details=None,
                       traders_analyzed=0, strategies_found=0, strategies_updated=0):
    now = utc_now_naive().isoformat()
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
    now = utc_now_naive().isoformat()
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
            "timestamp": utc_now_naive().isoformat(),
            "paper_account": get_paper_account(),
            "traders": get_active_traders()[:200],
            "bot_traders": [t for t in get_all_traders_including_bots() if not t.get("active", 1)],
            "strategies": get_all_strategies()[:1000],
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

    # Only skip restore if the DB already contains meaningful research/trading
    # state. A bootstrap paper account row alone should not block recovery.
    if get_active_traders() or get_all_strategies() or get_open_paper_trades():
        return False

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
                discovered_at=s.get("discovered_at"),
                last_scored=s.get("last_scored"),
                current_score=s.get("current_score", 0),
                active=s.get("active", 1),
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
                                gw.get("evaluated_at", utc_now_naive().isoformat()),
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
