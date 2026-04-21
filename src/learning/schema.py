"""SQLite DDL for continuous-learning tables.

Postgres uses forward SQL migrations. SQLite still exists for local dev/tests and
for any deployment running in dual-write fallback, so this module keeps the new
learning tables available there too.
"""

from __future__ import annotations

SQLITE_CONTINUOUS_LEARNING_DDL = """
CREATE TABLE IF NOT EXISTS continuous_learning_policies (
    policy_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'manual',
    parent_policy_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    promoted_at TEXT,
    non_trainable_limits TEXT NOT NULL DEFAULT '{}',
    promotion_gates TEXT NOT NULL DEFAULT '{}',
    rollback_rules TEXT NOT NULL DEFAULT '{}',
    reporting_contract TEXT NOT NULL DEFAULT '{}',
    model_versions TEXT NOT NULL DEFAULT '{}',
    metrics TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_learning_policies_status
    ON continuous_learning_policies (status, updated_at DESC);

CREATE TABLE IF NOT EXISTS source_inventory (
    source_name TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    required INTEGER NOT NULL DEFAULT 0,
    supports_live INTEGER NOT NULL DEFAULT 1,
    supports_historical INTEGER NOT NULL DEFAULT 0,
    point_in_time_safe INTEGER NOT NULL DEFAULT 0,
    min_history_days INTEGER NOT NULL DEFAULT 0,
    expected_freshness_seconds INTEGER NOT NULL DEFAULT 0,
    owner TEXT NOT NULL DEFAULT 'bot',
    notes TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_source_health_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL,
    source_name TEXT NOT NULL,
    status TEXT NOT NULL,
    freshness_seconds REAL,
    reason TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_source_health_recent
    ON data_source_health_history (source_name, observed_at DESC);

CREATE TABLE IF NOT EXISTS polymarket_markets (
    market_id TEXT PRIMARY KEY,
    question TEXT,
    slug TEXT,
    category TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    closed INTEGER NOT NULL DEFAULT 0,
    end_date TEXT,
    first_seen_ms INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL,
    raw_market TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_polymarket_markets_seen
    ON polymarket_markets (last_seen_ms DESC);

CREATE TABLE IF NOT EXISTS polymarket_tokens (
    token_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    outcome TEXT,
    side TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    first_seen_ms INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_polymarket_tokens_market
    ON polymarket_tokens (market_id);

CREATE TABLE IF NOT EXISTS polymarket_market_snapshots (
    market_id TEXT NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    probability REAL,
    volume REAL,
    volume_24h REAL,
    liquidity REAL,
    best_bid REAL,
    best_ask REAL,
    spread_bps REAL,
    raw_market TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (market_id, observed_at_ms)
);
CREATE INDEX IF NOT EXISTS idx_polymarket_snapshots_recent
    ON polymarket_market_snapshots (observed_at_ms DESC);

CREATE TABLE IF NOT EXISTS polymarket_trades (
    trade_id TEXT PRIMARY KEY,
    market_id TEXT,
    token_id TEXT,
    timestamp_ms INTEGER NOT NULL,
    side TEXT,
    price REAL,
    size REAL,
    maker_address TEXT,
    taker_address TEXT,
    raw_trade TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_polymarket_trades_lookup
    ON polymarket_trades (market_id, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS polymarket_price_points (
    token_id TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    price REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'polymarket',
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (token_id, timestamp_ms, source)
);
CREATE INDEX IF NOT EXISTS idx_polymarket_price_points_recent
    ON polymarket_price_points (token_id, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS funding_history (
    source TEXT NOT NULL,
    coin TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    annualized REAL,
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, coin, timestamp_ms)
);
CREATE INDEX IF NOT EXISTS idx_funding_history_lookup
    ON funding_history (coin, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS open_interest_history (
    source TEXT NOT NULL,
    coin TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    open_interest REAL NOT NULL,
    notional_usd REAL,
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, coin, timestamp_ms)
);
CREATE INDEX IF NOT EXISTS idx_open_interest_history_lookup
    ON open_interest_history (coin, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS options_summary_history (
    source TEXT NOT NULL,
    coin TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    iv_rank REAL,
    iv_percentile REAL,
    skew REAL,
    call_put_ratio REAL,
    net_premium_usd REAL,
    flow_direction TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, coin, timestamp_ms)
);
CREATE INDEX IF NOT EXISTS idx_options_summary_history_lookup
    ON options_summary_history (coin, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS decision_snapshots (
    decision_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    signal_timestamp TEXT,
    policy_id TEXT,
    model_version TEXT,
    coin TEXT,
    side TEXT,
    source TEXT,
    source_key TEXT,
    strategy_type TEXT,
    strategy_id TEXT,
    signal_id TEXT,
    raw_confidence REAL,
    calibrated_confidence REAL,
    firewall_decision TEXT,
    final_status TEXT NOT NULL DEFAULT 'candidate',
    rejection_reason TEXT,
    entry_price REAL,
    proposed_size_usd REAL,
    proposed_position_pct REAL,
    proposed_leverage REAL,
    proposed_sl_roe REAL,
    proposed_tp_roe REAL,
    proposed_sl_price REAL,
    proposed_tp_price REAL,
    paper_trade_id INTEGER,
    live_order_id TEXT,
    features TEXT NOT NULL DEFAULT '{}',
    source_health TEXT NOT NULL DEFAULT '{}',
    regime TEXT NOT NULL DEFAULT '{}',
    raw_signal TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_recent
    ON decision_snapshots (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_status
    ON decision_snapshots (final_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_trade
    ON decision_snapshots (paper_trade_id);
"""


def ensure_sqlite_schema(conn) -> None:
    """Create continuous-learning tables on a SQLite connection."""
    conn.executescript(SQLITE_CONTINUOUS_LEARNING_DDL)
