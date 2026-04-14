-- =============================================================
-- Migration 0006: Regime history for XGBoostRegimeForecaster
-- Mirrors the SQLite training-history table with Postgres-native
-- types so dual-write and full Postgres mode share the same store.
-- =============================================================

CREATE TABLE IF NOT EXISTS regime_history (
    id                      BIGSERIAL PRIMARY KEY,
    timestamp               TIMESTAMPTZ NOT NULL DEFAULT now(),
    coin                    TEXT DEFAULT 'BTC',
    funding_rate            DOUBLE PRECISION DEFAULT 0,
    funding_slope           DOUBLE PRECISION DEFAULT 0,
    orderbook_imbalance     DOUBLE PRECISION DEFAULT 0,
    arkham_flow             DOUBLE PRECISION DEFAULT 0,
    volatility_5m           DOUBLE PRECISION DEFAULT 0,
    basis_spread            DOUBLE PRECISION DEFAULT 0,
    polymarket_sentiment    DOUBLE PRECISION DEFAULT 0,
    options_flow_conviction DOUBLE PRECISION DEFAULT 0,
    regime_label            INTEGER,
    confidence              DOUBLE PRECISION DEFAULT 0,
    predicted_regime        TEXT DEFAULT 'neutral',
    label_source            TEXT DEFAULT 'predicted'
);

CREATE INDEX IF NOT EXISTS idx_regime_history_timestamp
    ON regime_history (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_regime_history_coin_timestamp
    ON regime_history (coin, timestamp DESC);
