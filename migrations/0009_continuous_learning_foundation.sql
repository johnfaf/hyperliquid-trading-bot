-- Continuous learning foundation: safety policies, historical inputs, replay, and decision snapshots.

CREATE TABLE IF NOT EXISTS continuous_learning_policies (
    policy_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'manual',
    parent_policy_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    promoted_at TIMESTAMPTZ,
    non_trainable_limits JSONB NOT NULL DEFAULT '{}'::jsonb,
    promotion_gates JSONB NOT NULL DEFAULT '{}'::jsonb,
    rollback_rules JSONB NOT NULL DEFAULT '{}'::jsonb,
    reporting_contract JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_learning_policies_status
    ON continuous_learning_policies (status, updated_at DESC);

CREATE TABLE IF NOT EXISTS source_inventory (
    source_name TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    required BOOLEAN NOT NULL DEFAULT FALSE,
    supports_live BOOLEAN NOT NULL DEFAULT TRUE,
    supports_historical BOOLEAN NOT NULL DEFAULT FALSE,
    point_in_time_safe BOOLEAN NOT NULL DEFAULT FALSE,
    min_history_days INTEGER NOT NULL DEFAULT 0,
    expected_freshness_seconds INTEGER NOT NULL DEFAULT 0,
    owner TEXT NOT NULL DEFAULT 'bot',
    notes TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS data_source_health_history (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_name TEXT NOT NULL,
    status TEXT NOT NULL,
    freshness_seconds DOUBLE PRECISION,
    reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_source_health_recent
    ON data_source_health_history (source_name, observed_at DESC);

CREATE TABLE IF NOT EXISTS polymarket_markets (
    market_id TEXT PRIMARY KEY,
    question TEXT,
    slug TEXT,
    category TEXT,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    closed BOOLEAN NOT NULL DEFAULT FALSE,
    end_date TIMESTAMPTZ,
    first_seen_ms BIGINT NOT NULL,
    last_seen_ms BIGINT NOT NULL,
    raw_market JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_polymarket_markets_seen
    ON polymarket_markets (last_seen_ms DESC);

CREATE TABLE IF NOT EXISTS polymarket_tokens (
    token_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES polymarket_markets(market_id) ON DELETE CASCADE,
    outcome TEXT,
    side TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_ms BIGINT NOT NULL,
    last_seen_ms BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_polymarket_tokens_market
    ON polymarket_tokens (market_id);

CREATE TABLE IF NOT EXISTS polymarket_market_snapshots (
    market_id TEXT NOT NULL REFERENCES polymarket_markets(market_id) ON DELETE CASCADE,
    observed_at_ms BIGINT NOT NULL,
    probability DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    volume_24h DOUBLE PRECISION,
    liquidity DOUBLE PRECISION,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    raw_market JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (market_id, observed_at_ms)
);

CREATE INDEX IF NOT EXISTS idx_polymarket_snapshots_recent
    ON polymarket_market_snapshots (observed_at_ms DESC);

CREATE TABLE IF NOT EXISTS polymarket_trades (
    trade_id TEXT PRIMARY KEY,
    market_id TEXT,
    token_id TEXT,
    timestamp_ms BIGINT NOT NULL,
    side TEXT,
    price DOUBLE PRECISION,
    size DOUBLE PRECISION,
    maker_address TEXT,
    taker_address TEXT,
    raw_trade JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_polymarket_trades_lookup
    ON polymarket_trades (market_id, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS polymarket_price_points (
    token_id TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    source TEXT NOT NULL DEFAULT 'polymarket',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (token_id, timestamp_ms, source)
);

CREATE INDEX IF NOT EXISTS idx_polymarket_price_points_recent
    ON polymarket_price_points (token_id, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS funding_history (
    source TEXT NOT NULL,
    coin TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    funding_rate DOUBLE PRECISION NOT NULL,
    annualized DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source, coin, timestamp_ms)
);

CREATE INDEX IF NOT EXISTS idx_funding_history_lookup
    ON funding_history (coin, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS open_interest_history (
    source TEXT NOT NULL,
    coin TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    open_interest DOUBLE PRECISION NOT NULL,
    notional_usd DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source, coin, timestamp_ms)
);

CREATE INDEX IF NOT EXISTS idx_open_interest_history_lookup
    ON open_interest_history (coin, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS options_summary_history (
    source TEXT NOT NULL,
    coin TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    iv_rank DOUBLE PRECISION,
    iv_percentile DOUBLE PRECISION,
    skew DOUBLE PRECISION,
    call_put_ratio DOUBLE PRECISION,
    net_premium_usd DOUBLE PRECISION,
    flow_direction TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source, coin, timestamp_ms)
);

CREATE INDEX IF NOT EXISTS idx_options_summary_history_lookup
    ON options_summary_history (coin, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS decision_snapshots (
    decision_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    signal_timestamp TIMESTAMPTZ,
    policy_id TEXT,
    model_version TEXT,
    coin TEXT,
    side TEXT,
    source TEXT,
    source_key TEXT,
    strategy_type TEXT,
    strategy_id TEXT,
    signal_id TEXT,
    raw_confidence DOUBLE PRECISION,
    calibrated_confidence DOUBLE PRECISION,
    firewall_decision TEXT,
    final_status TEXT NOT NULL DEFAULT 'candidate',
    rejection_reason TEXT,
    entry_price DOUBLE PRECISION,
    proposed_size_usd DOUBLE PRECISION,
    proposed_position_pct DOUBLE PRECISION,
    proposed_leverage DOUBLE PRECISION,
    proposed_sl_roe DOUBLE PRECISION,
    proposed_tp_roe DOUBLE PRECISION,
    proposed_sl_price DOUBLE PRECISION,
    proposed_tp_price DOUBLE PRECISION,
    paper_trade_id BIGINT,
    live_order_id TEXT,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_health JSONB NOT NULL DEFAULT '{}'::jsonb,
    regime JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_signal JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_decision_snapshots_recent
    ON decision_snapshots (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_status
    ON decision_snapshots (final_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_trade
    ON decision_snapshots (paper_trade_id);
