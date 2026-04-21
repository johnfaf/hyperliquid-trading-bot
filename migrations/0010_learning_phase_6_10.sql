-- Continuous learning Phase 6-10: dataset manifests, replay results, shadow evaluation,
-- improvement runs, and manual promotion records.

CREATE TABLE IF NOT EXISTS learning_datasets (
    dataset_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_policy_id TEXT,
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ,
    row_count INTEGER NOT NULL DEFAULT 0,
    positive_labels INTEGER NOT NULL DEFAULT 0,
    executed_count INTEGER NOT NULL DEFAULT 0,
    feature_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    label_definition JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_datasets_recent
    ON learning_datasets (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset_id TEXT,
    policy_id TEXT,
    candidate_policy_id TEXT,
    backtest_type TEXT NOT NULL DEFAULT 'decision_replay',
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ,
    trade_count INTEGER NOT NULL DEFAULT 0,
    win_rate DOUBLE PRECISION,
    total_pnl DOUBLE PRECISION,
    avg_pnl DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    sharpe_like DOUBLE PRECISION,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    passed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_learning_backtest_runs_recent
    ON learning_backtest_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_learning_backtest_runs_dataset
    ON learning_backtest_runs (dataset_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_shadow_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    champion_policy_id TEXT NOT NULL,
    challenger_policy_id TEXT NOT NULL,
    dataset_id TEXT,
    champion_run_id TEXT,
    challenger_run_id TEXT,
    verdict TEXT NOT NULL,
    gates_passed BOOLEAN NOT NULL DEFAULT FALSE,
    gate_results JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics_delta JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_learning_shadow_recent
    ON learning_shadow_evaluations (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_improvement_runs (
    improvement_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset_id TEXT,
    best_candidate_policy_id TEXT,
    mode TEXT NOT NULL DEFAULT 'offline',
    status TEXT NOT NULL DEFAULT 'completed',
    search_space JSONB NOT NULL DEFAULT '{}'::jsonb,
    candidate_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    next_action TEXT NOT NULL DEFAULT 'manual_review',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_improvement_recent
    ON learning_improvement_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_promotion_decisions (
    decision_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    candidate_policy_id TEXT NOT NULL,
    target_policy_id TEXT,
    requested_by TEXT NOT NULL DEFAULT 'codex',
    decision TEXT NOT NULL,
    approved BOOLEAN NOT NULL DEFAULT FALSE,
    requires_manual_approval BOOLEAN NOT NULL DEFAULT TRUE,
    shadow_evaluation_id TEXT,
    rollback_policy_id TEXT,
    reason TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_promotion_recent
    ON learning_promotion_decisions (created_at DESC);
