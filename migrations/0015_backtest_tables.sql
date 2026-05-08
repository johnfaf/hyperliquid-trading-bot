-- =============================================================
-- Migration 0015: Backtest result tables (Postgres mirror)
--
-- These tables are created on-demand at runtime by
-- src/backtest/backtest_engine.py (init_backtest_tables) using
-- SQLite-flavoured DDL. The dualwrite adapter deliberately skips
-- DDL from mirroring (schema is meant to flow through migrations),
-- which left every backtest INSERT firing
-- "UndefinedTable: relation 'backtest_results' does not exist"
-- against Postgres — 207 such warnings in 7.5 minutes was the
-- straw that triggered this migration.
--
-- Schema must match the SQLite version in backtest_engine.py:
--   backtest_results        — per (address, timeframe) summary
--   backtest_coin_perf      — per (address, coin) attribution
--   backtest_time_analysis  — per (address, analysis_type, bucket)
--                             time-of-day / day-of-week analysis
-- =============================================================

CREATE TABLE IF NOT EXISTS backtest_results (
    address              TEXT NOT NULL,
    timeframe            TEXT NOT NULL,
    total_periods        INTEGER DEFAULT 0,
    active_periods       INTEGER DEFAULT 0,
    total_pnl            DOUBLE PRECISION DEFAULT 0,
    total_penalised_pnl  DOUBLE PRECISION DEFAULT 0,
    avg_period_pnl       DOUBLE PRECISION DEFAULT 0,
    std_period_pnl       DOUBLE PRECISION DEFAULT 0,
    best_period_pnl      DOUBLE PRECISION DEFAULT 0,
    worst_period_pnl     DOUBLE PRECISION DEFAULT 0,
    profitable_periods   INTEGER DEFAULT 0,
    profitable_pct       DOUBLE PRECISION DEFAULT 0,
    consistency_score    DOUBLE PRECISION DEFAULT 0,
    -- The backtester serialises an array of period summaries here as a
    -- JSON blob. Postgres can store this as JSONB cleanly even though the
    -- SQLite mirror keeps it as TEXT — psycopg coerces the string.
    periods_json         JSONB NOT NULL DEFAULT '[]'::jsonb,
    evaluated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (address, timeframe)
);

CREATE TABLE IF NOT EXISTS backtest_coin_perf (
    address    TEXT NOT NULL,
    coin       TEXT NOT NULL,
    fills      INTEGER DEFAULT 0,
    closing    INTEGER DEFAULT 0,
    raw_pnl    DOUBLE PRECISION DEFAULT 0,
    pen_pnl    DOUBLE PRECISION DEFAULT 0,
    volume     DOUBLE PRECISION DEFAULT 0,
    wins       INTEGER DEFAULT 0,
    losses     INTEGER DEFAULT 0,
    win_rate   DOUBLE PRECISION DEFAULT 0,
    PRIMARY KEY (address, coin)
);

CREATE TABLE IF NOT EXISTS backtest_time_analysis (
    address       TEXT NOT NULL,
    analysis_type TEXT NOT NULL,
    bucket        INTEGER NOT NULL,
    pnl           DOUBLE PRECISION DEFAULT 0,
    PRIMARY KEY (address, analysis_type, bucket)
);

-- Useful read indexes mirroring how the dashboard queries these tables
-- (see src/ui/backtest_dashboard.py and src/ui/v2/routers/backtest.py):
--   "SELECT ... FROM backtest_results WHERE timeframe = ? ORDER BY ..."
--   "SELECT ... FROM backtest_coin_perf WHERE address = ? ORDER BY pen_pnl DESC"
CREATE INDEX IF NOT EXISTS idx_backtest_results_timeframe
    ON backtest_results(timeframe, total_pnl DESC);

CREATE INDEX IF NOT EXISTS idx_backtest_coin_perf_pen_pnl
    ON backtest_coin_perf(address, pen_pnl DESC);
