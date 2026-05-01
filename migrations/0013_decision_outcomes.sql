-- Decision learning upgrade: durable stage events and outcome labels.
--
-- These tables make every candidate decision auditable from generation through
-- rejection/open/close, and provide one row per decision for offline training.

CREATE TABLE IF NOT EXISTS decision_stage_events (
    event_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    confidence DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_decision_stage_events_decision
    ON decision_stage_events (decision_id, created_at);
CREATE INDEX IF NOT EXISTS idx_decision_stage_events_stage
    ON decision_stage_events (stage, status, created_at DESC);

CREATE TABLE IF NOT EXISTS decision_outcomes (
    decision_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    coin TEXT,
    side TEXT,
    source TEXT,
    source_key TEXT,
    strategy_type TEXT,
    final_status TEXT,
    action_taken BOOLEAN NOT NULL DEFAULT FALSE,
    paper_trade_id BIGINT,
    label_win INTEGER,
    outcome_pnl DOUBLE PRECISION,
    outcome_return_pct DOUBLE PRECISION,
    exit_reason TEXT,
    hold_minutes DOUBLE PRECISION,
    max_favorable_r DOUBLE PRECISION,
    max_adverse_r DOUBLE PRECISION,
    forward_return_15m DOUBLE PRECISION,
    forward_return_1h DOUBLE PRECISION,
    forward_return_4h DOUBLE PRECISION,
    forward_return_24h DOUBLE PRECISION,
    would_have_won INTEGER,
    side_correct INTEGER,
    missed_profit_usd DOUBLE PRECISION,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    decision_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    outcome_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    explanation TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_decision_outcomes_source
    ON decision_outcomes (source, source_key, strategy_type);
CREATE INDEX IF NOT EXISTS idx_decision_outcomes_status
    ON decision_outcomes (final_status, action_taken);
CREATE INDEX IF NOT EXISTS idx_decision_outcomes_created
    ON decision_outcomes (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_decision_calibrators (
    calibrator_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset_id TEXT,
    model_type TEXT NOT NULL DEFAULT 'conservative_empirical',
    feature_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    global_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    gates JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_decision_calibrators_recent
    ON learning_decision_calibrators (created_at DESC);
