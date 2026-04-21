-- Continuous learning Phase 11-19: quality gates, candidate registry,
-- feature attribution, drift monitoring, promotion packages, scheduler audit,
-- and operator reporting. These tables store offline evidence only; live
-- policy mutation remains manual.

CREATE TABLE IF NOT EXISTS learning_data_quality_reports (
    report_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset_id TEXT NOT NULL,
    status TEXT NOT NULL,
    checks JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    blocks_training BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_quality_dataset
    ON learning_data_quality_reports (dataset_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_policy_candidates (
    candidate_policy_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    parent_policy_id TEXT NOT NULL,
    source_improvement_id TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    trainable_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    non_trainable_limits_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    safety_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_learning_candidates_recent
    ON learning_policy_candidates (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_feature_attributions (
    attribution_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset_id TEXT NOT NULL,
    candidate_policy_id TEXT,
    method TEXT NOT NULL DEFAULT 'win_loss_mean_delta',
    feature_scores JSONB NOT NULL DEFAULT '[]'::jsonb,
    top_features JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_attribution_dataset
    ON learning_feature_attributions (dataset_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_drift_reports (
    drift_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    baseline_dataset_id TEXT NOT NULL,
    current_dataset_id TEXT NOT NULL,
    status TEXT NOT NULL,
    feature_drift JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    blocks_promotion BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_drift_recent
    ON learning_drift_reports (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_shadow_periods (
    shadow_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    candidate_policy_id TEXT NOT NULL,
    champion_policy_id TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    min_shadow_days INTEGER NOT NULL DEFAULT 7,
    status TEXT NOT NULL DEFAULT 'planned',
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_learning_shadow_periods_recent
    ON learning_shadow_periods (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_rollback_checks (
    rollback_check_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    candidate_policy_id TEXT NOT NULL,
    rollback_policy_id TEXT NOT NULL,
    status TEXT NOT NULL,
    checks JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_rollback_recent
    ON learning_rollback_checks (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_promotion_packages (
    package_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    candidate_policy_id TEXT NOT NULL,
    dataset_id TEXT,
    shadow_evaluation_id TEXT,
    promotion_decision_id TEXT,
    readiness TEXT NOT NULL,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    operator_summary TEXT NOT NULL DEFAULT '',
    requires_manual_approval BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_packages_recent
    ON learning_promotion_packages (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_scheduler_runs (
    schedule_run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    dataset_id TEXT,
    improvement_id TEXT,
    package_id TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    errors JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_scheduler_recent
    ON learning_scheduler_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS learning_operator_reports (
    report_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    package_id TEXT,
    report_type TEXT NOT NULL DEFAULT 'promotion_readiness',
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_learning_operator_reports_recent
    ON learning_operator_reports (created_at DESC);
