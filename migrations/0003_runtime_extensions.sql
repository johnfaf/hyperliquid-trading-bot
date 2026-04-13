-- =============================================================
-- Migration 0003: Runtime extension tables
-- Covers modules moved off standalone SQLite stores during
-- the Phase 5-7 Postgres cutover.
-- =============================================================

CREATE TABLE IF NOT EXISTS arena_agents (
    agent_id            TEXT PRIMARY KEY,
    name                TEXT,
    strategy_type       TEXT,
    status              TEXT DEFAULT 'incubating',
    params              TEXT DEFAULT '{}',
    capital_allocated   DOUBLE PRECISION DEFAULT 1000,
    total_pnl           DOUBLE PRECISION DEFAULT 0,
    total_trades        INTEGER DEFAULT 0,
    winning_trades      INTEGER DEFAULT 0,
    sharpe_ratio        DOUBLE PRECISION DEFAULT 0,
    max_drawdown        DOUBLE PRECISION DEFAULT 0,
    win_rate            DOUBLE PRECISION DEFAULT 0,
    generation          INTEGER DEFAULT 0,
    parent_id           TEXT DEFAULT '',
    elo_rating          DOUBLE PRECISION DEFAULT 1000,
    tournament_rank     INTEGER DEFAULT 0,
    backtest_sharpe     DOUBLE PRECISION DEFAULT 0,
    backtest_pnl        DOUBLE PRECISION DEFAULT 0,
    backtest_trades     INTEGER DEFAULT 0,
    backtest_win_rate   DOUBLE PRECISION DEFAULT 0,
    created_at          TIMESTAMPTZ,
    last_updated        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS arena_rounds (
    round_id            BIGINT PRIMARY KEY,
    started_at          TIMESTAMPTZ,
    ended_at            TIMESTAMPTZ,
    agents_entered      INTEGER,
    agents_eliminated   INTEGER,
    agents_promoted     INTEGER,
    agents_spawned      INTEGER,
    best_agent          TEXT,
    best_fitness        DOUBLE PRECISION,
    summary             TEXT
);

CREATE TABLE IF NOT EXISTS shadow_trades (
    id              BIGSERIAL PRIMARY KEY,
    signal_source   TEXT NOT NULL,
    coin            TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    size            DOUBLE PRECISION NOT NULL,
    pnl             DOUBLE PRECISION DEFAULT 0,
    pnl_pct         DOUBLE PRECISION DEFAULT 0,
    entry_ts        TIMESTAMPTZ NOT NULL,
    exit_ts         TIMESTAMPTZ,
    regime_at_entry TEXT,
    confidence      DOUBLE PRECISION DEFAULT 1.0,
    metadata_json   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS shadow_attribution (
    id              BIGSERIAL PRIMARY KEY,
    date            TEXT NOT NULL,
    signal_source   TEXT NOT NULL,
    trade_count     INTEGER DEFAULT 0,
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    avg_pnl_pct     DOUBLE PRECISION DEFAULT 0,
    win_rate        DOUBLE PRECISION DEFAULT 0,
    best_trade_pnl  DOUBLE PRECISION,
    worst_trade_pnl DOUBLE PRECISION,
    sharpe_proxy    DOUBLE PRECISION DEFAULT 0,
    UNIQUE(date, signal_source)
);

CREATE TABLE IF NOT EXISTS trade_memory (
    trade_id        TEXT PRIMARY KEY,
    coin            TEXT NOT NULL,
    side            TEXT NOT NULL,
    strategy_type   TEXT,
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    pnl             DOUBLE PRECISION,
    return_pct      DOUBLE PRECISION,
    win             INTEGER,
    opened_at       TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    confidence      DOUBLE PRECISION,
    source          TEXT,
    regime          TEXT,
    setup_type      TEXT,
    features_json   TEXT,
    feature_vector  TEXT
);

CREATE INDEX IF NOT EXISTS idx_shadow_source
    ON shadow_trades(signal_source);

CREATE INDEX IF NOT EXISTS idx_shadow_entry_ts
    ON shadow_trades(entry_ts);

CREATE INDEX IF NOT EXISTS idx_shadow_exit_ts
    ON shadow_trades(exit_ts);

CREATE INDEX IF NOT EXISTS idx_attribution_date
    ON shadow_attribution(date);

CREATE INDEX IF NOT EXISTS idx_attribution_source
    ON shadow_attribution(signal_source);

CREATE INDEX IF NOT EXISTS idx_trade_memory_coin
    ON trade_memory(coin);

CREATE INDEX IF NOT EXISTS idx_trade_memory_strategy
    ON trade_memory(strategy_type);
