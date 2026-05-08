-- =============================================================
-- Migration 0016: Arena trade events + readiness probe
--
-- Same pattern as 0015 (backtest tables): both of these were created
-- on-demand at runtime in SQLite-flavoured DDL with no matching
-- migration, so the dualwrite mirror was failing on every INSERT
-- with "UndefinedTable: relation '...' does not exist".
--
--   arena_trade_events   - written by src/signals/alpha_arena.py per
--                          trade outcome; used to evaluate agents
--                          across rounds.
--   readiness_probe      - written by src/core/readiness.py to verify
--                          DB writability for /health and similar.
-- =============================================================

CREATE TABLE IF NOT EXISTS arena_trade_events (
    event_id       TEXT PRIMARY KEY,
    recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    trade_id       TEXT,
    agent_id       TEXT,
    strategy_type  TEXT,
    pnl            DOUBLE PRECISION,
    return_pct     DOUBLE PRECISION,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_arena_trade_events_agent
    ON arena_trade_events(agent_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_arena_trade_events_recorded_at
    ON arena_trade_events(recorded_at DESC);


CREATE TABLE IF NOT EXISTS readiness_probe (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    touched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
