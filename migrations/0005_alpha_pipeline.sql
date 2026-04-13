-- Phase B: ML alpha pipeline on top of the Postgres feature store.
-- Stores walk-forward training runs plus live model predictions.

CREATE TABLE IF NOT EXISTS alpha_model_runs (
    id                         BIGSERIAL PRIMARY KEY,
    trained_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    timeframe                  TEXT NOT NULL,
    horizon                    TEXT NOT NULL,
    model_version              TEXT NOT NULL,
    sample_count               INTEGER NOT NULL DEFAULT 0,
    oof_sample_count           INTEGER NOT NULL DEFAULT 0,
    active_trade_count         INTEGER NOT NULL DEFAULT 0,
    accuracy                   DOUBLE PRECISION,
    balanced_accuracy          DOUBLE PRECISION,
    brier_score                DOUBLE PRECISION,
    log_loss                   DOUBLE PRECISION,
    ece                        DOUBLE PRECISION,
    win_rate                   DOUBLE PRECISION,
    avg_signed_return_bps      DOUBLE PRECISION,
    significance_pvalue        DOUBLE PRECISION,
    bootstrap_ci_low_bps       DOUBLE PRECISION,
    bootstrap_ci_high_bps      DOUBLE PRECISION,
    eligible                   BOOLEAN NOT NULL DEFAULT FALSE,
    feature_names              JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics                    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_alpha_model_runs_lookup
    ON alpha_model_runs (timeframe, horizon, trained_at DESC);

CREATE TABLE IF NOT EXISTS alpha_predictions (
    id                         BIGSERIAL PRIMARY KEY,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    coin                       TEXT NOT NULL,
    timeframe                  TEXT NOT NULL,
    horizon                    TEXT NOT NULL,
    feature_timestamp_ms       BIGINT NOT NULL,
    model_version              TEXT NOT NULL,
    raw_probability_up         DOUBLE PRECISION NOT NULL,
    calibrated_probability_up  DOUBLE PRECISION NOT NULL,
    predicted_side             TEXT NOT NULL,
    confidence                 DOUBLE PRECISION NOT NULL,
    expected_return_bps        DOUBLE PRECISION,
    significance_pvalue        DOUBLE PRECISION,
    eligible                   BOOLEAN NOT NULL DEFAULT FALSE,
    metadata                   JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_alpha_predictions_recent
    ON alpha_predictions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_alpha_predictions_coin_horizon
    ON alpha_predictions (coin, horizon, created_at DESC);
