-- =============================================================
-- Migration 0017: firewall_shadow_signals
--
-- Side table for src/signals/firewall_shadow.py.  When
-- FIREWALL_SHADOW_MODE_FRACTION > 0, a fraction of REJECTED signals
-- get sampled into this table with their predicted side + confidence
-- + entry-price snapshot.  A periodic evaluator computes a simulated
-- win/loss outcome from the next mid-price observation
-- (FIREWALL_SHADOW_HOLD_MINUTES later) and feeds it to
-- CalibrationTracker so the bot's calibration curves can rebuild
-- even when no real trades are flowing.
--
-- Default OFF (env flag).  Pure observability -- never opens trades.
-- =============================================================

CREATE TABLE IF NOT EXISTS firewall_shadow_signals (
    id                     BIGSERIAL PRIMARY KEY,
    coin                   TEXT NOT NULL,
    side                   TEXT NOT NULL,
    confidence             DOUBLE PRECISION NOT NULL,
    source_key             TEXT,
    entry_price            DOUBLE PRECISION NOT NULL,
    rejection_reason       TEXT,
    regime                 TEXT,
    opened_at              TIMESTAMPTZ NOT NULL,
    evaluated              BOOLEAN NOT NULL DEFAULT FALSE,
    evaluated_at           TIMESTAMPTZ,
    simulated_win          BOOLEAN,
    simulated_exit_price   DOUBLE PRECISION,
    simulated_pnl_pct      DOUBLE PRECISION
);

-- The evaluator's hot query: pull pending rows older than the hold
-- window, oldest first, capped at FIREWALL_SHADOW_MAX_EVAL_PER_CYCLE.
CREATE INDEX IF NOT EXISTS idx_firewall_shadow_pending
    ON firewall_shadow_signals (evaluated, opened_at);
