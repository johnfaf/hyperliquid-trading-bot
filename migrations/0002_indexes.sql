-- =============================================================
-- Migration 0002: Indexes
-- Mirrors the SQLite indexes for query performance.
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_snapshots_trader
    ON position_snapshots(trader_address, timestamp);

CREATE INDEX IF NOT EXISTS idx_snapshots_coin
    ON position_snapshots(coin, timestamp);

CREATE INDEX IF NOT EXISTS idx_scores_strategy
    ON strategy_scores(strategy_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_paper_trades_status
    ON paper_trades(status);

CREATE INDEX IF NOT EXISTS idx_paper_trades_coin
    ON paper_trades(coin, status);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp
    ON audit_trail(timestamp);

CREATE INDEX IF NOT EXISTS idx_audit_action
    ON audit_trail(action);

CREATE INDEX IF NOT EXISTS idx_audit_coin
    ON audit_trail(coin);

CREATE INDEX IF NOT EXISTS idx_wf_addr
    ON wallet_fills(wallet_address);

CREATE INDEX IF NOT EXISTS idx_wf_time
    ON wallet_fills(time_ms);

CREATE INDEX IF NOT EXISTS idx_wf_coin
    ON wallet_fills(coin);

CREATE INDEX IF NOT EXISTS idx_calibration_source
    ON calibration_records(source_key);
