-- =============================================================
-- Migration 0001: Initial Postgres schema
-- Mirrors the SQLite tables used in runtime production.
-- =============================================================

-- Top traders we're tracking
CREATE TABLE IF NOT EXISTS traders (
    address         TEXT PRIMARY KEY,
    first_seen      TIMESTAMPTZ NOT NULL,
    last_updated    TIMESTAMPTZ NOT NULL,
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    roi_pct         DOUBLE PRECISION DEFAULT 0,
    account_value   DOUBLE PRECISION DEFAULT 0,
    win_rate        DOUBLE PRECISION DEFAULT 0,
    trade_count     INTEGER DEFAULT 0,
    active          BOOLEAN DEFAULT TRUE,
    metadata        JSONB DEFAULT '{}'::jsonb
);

-- Snapshots of trader positions over time
CREATE TABLE IF NOT EXISTS position_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    trader_address  TEXT NOT NULL REFERENCES traders(address),
    timestamp       TIMESTAMPTZ NOT NULL,
    coin            TEXT NOT NULL,
    side            TEXT NOT NULL,
    size            DOUBLE PRECISION NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    leverage        DOUBLE PRECISION DEFAULT 1,
    unrealized_pnl  DOUBLE PRECISION DEFAULT 0,
    margin_used     DOUBLE PRECISION DEFAULT 0,
    metadata        JSONB DEFAULT '{}'::jsonb
);

-- Detected trading strategies
CREATE TABLE IF NOT EXISTS strategies (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    strategy_type   TEXT NOT NULL,
    parameters      JSONB DEFAULT '{}'::jsonb,
    discovered_at   TIMESTAMPTZ NOT NULL,
    last_scored     TIMESTAMPTZ,
    current_score   DOUBLE PRECISION DEFAULT 0,
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    trade_count     INTEGER DEFAULT 0,
    win_rate        DOUBLE PRECISION DEFAULT 0,
    sharpe_ratio    DOUBLE PRECISION DEFAULT 0,
    active          BOOLEAN DEFAULT TRUE
);

-- Strategy performance scores over time
CREATE TABLE IF NOT EXISTS strategy_scores (
    id                  BIGSERIAL PRIMARY KEY,
    strategy_id         BIGINT NOT NULL REFERENCES strategies(id),
    timestamp           TIMESTAMPTZ NOT NULL,
    score               DOUBLE PRECISION NOT NULL,
    pnl_score           DOUBLE PRECISION DEFAULT 0,
    win_rate_score      DOUBLE PRECISION DEFAULT 0,
    sharpe_score        DOUBLE PRECISION DEFAULT 0,
    consistency_score   DOUBLE PRECISION DEFAULT 0,
    risk_adj_score      DOUBLE PRECISION DEFAULT 0,
    notes               TEXT DEFAULT ''
);

-- Paper trading positions and history
CREATE TABLE IF NOT EXISTS paper_trades (
    id              BIGSERIAL PRIMARY KEY,
    strategy_id     BIGINT REFERENCES strategies(id),
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    coin            TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    size            DOUBLE PRECISION NOT NULL,
    leverage        DOUBLE PRECISION DEFAULT 1,
    pnl             DOUBLE PRECISION DEFAULT 0,
    status          TEXT DEFAULT 'open',
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    metadata        JSONB DEFAULT '{}'::jsonb
);

-- Paper trading account state (single row, id=1)
CREATE TABLE IF NOT EXISTS paper_account (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    balance         DOUBLE PRECISION NOT NULL,
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    total_trades    INTEGER DEFAULT 0,
    winning_trades  INTEGER DEFAULT 0,
    last_updated    TIMESTAMPTZ NOT NULL
);

-- Research logs (what the bot discovered each cycle)
CREATE TABLE IF NOT EXISTS research_logs (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    cycle_type          TEXT NOT NULL,
    summary             TEXT NOT NULL,
    details             JSONB DEFAULT '{}'::jsonb,
    traders_analyzed    INTEGER DEFAULT 0,
    strategies_found    INTEGER DEFAULT 0,
    strategies_updated  INTEGER DEFAULT 0
);

-- Key-value bot state
CREATE TABLE IF NOT EXISTS bot_state (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

-- Immutable audit trail
CREATE TABLE IF NOT EXISTS audit_trail (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    action      TEXT NOT NULL,
    coin        TEXT,
    side        TEXT,
    price       DOUBLE PRECISION,
    size        DOUBLE PRECISION,
    pnl         DOUBLE PRECISION,
    source      TEXT,
    details     JSONB DEFAULT '{}'::jsonb
);

-- Golden wallets (top-performing traders identified by the scanner)
CREATE TABLE IF NOT EXISTS golden_wallets (
    address                     TEXT PRIMARY KEY,
    bot_score                   INTEGER DEFAULT 0,
    total_fills                 INTEGER DEFAULT 0,
    raw_pnl                     DOUBLE PRECISION DEFAULT 0,
    penalised_pnl               DOUBLE PRECISION DEFAULT 0,
    max_drawdown_pct            DOUBLE PRECISION DEFAULT 0,
    penalised_max_drawdown_pct  DOUBLE PRECISION DEFAULT 0,
    sharpe_ratio                DOUBLE PRECISION DEFAULT 0,
    win_rate                    DOUBLE PRECISION DEFAULT 0,
    trades_per_day              DOUBLE PRECISION DEFAULT 0,
    is_golden                   BOOLEAN DEFAULT FALSE,
    coins_traded                TEXT DEFAULT '[]',
    best_coin                   TEXT DEFAULT '',
    worst_coin                  TEXT DEFAULT '',
    raw_equity_curve            TEXT DEFAULT '[]',
    penalised_equity_curve      TEXT DEFAULT '[]',
    equity_timestamps           TEXT DEFAULT '[]',
    evaluated_at                TIMESTAMPTZ NOT NULL,
    connected_to_live           BOOLEAN DEFAULT FALSE
);

-- Wallet fills (individual trades for golden wallets)
CREATE TABLE IF NOT EXISTS wallet_fills (
    id              BIGSERIAL PRIMARY KEY,
    wallet_address  TEXT NOT NULL REFERENCES golden_wallets(address),
    coin            TEXT NOT NULL,
    side            TEXT NOT NULL,
    original_price  DOUBLE PRECISION NOT NULL,
    penalised_price DOUBLE PRECISION NOT NULL,
    size            DOUBLE PRECISION NOT NULL,
    time_ms         BIGINT NOT NULL,
    delayed_time_ms BIGINT NOT NULL,
    closed_pnl      DOUBLE PRECISION DEFAULT 0,
    penalised_pnl   DOUBLE PRECISION DEFAULT 0,
    fee             DOUBLE PRECISION DEFAULT 0,
    is_liquidation  BOOLEAN DEFAULT FALSE,
    direction       TEXT DEFAULT ''
);

-- Calibration records (signal source calibration tracking)
CREATE TABLE IF NOT EXISTS calibration_records (
    id                      BIGSERIAL PRIMARY KEY,
    source_key              TEXT NOT NULL,
    predicted_confidence    DOUBLE PRECISION NOT NULL,
    actual_win              INTEGER NOT NULL,
    pnl                     DOUBLE PRECISION,
    coin                    TEXT,
    side                    TEXT,
    timestamp               TIMESTAMPTZ
);

-- Agent scoring (per-source performance tracking)
CREATE TABLE IF NOT EXISTS agent_scores (
    source_key      TEXT PRIMARY KEY,
    total_signals   INTEGER DEFAULT 0,
    correct_signals INTEGER DEFAULT 0,
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    total_return    DOUBLE PRECISION DEFAULT 0,
    accuracy        DOUBLE PRECISION DEFAULT 0,
    sharpe          DOUBLE PRECISION DEFAULT 0,
    dynamic_weight  DOUBLE PRECISION DEFAULT 0.5,
    trade_history   JSONB DEFAULT '[]'::jsonb,
    last_updated    TIMESTAMPTZ
)
