-- Migration 0012: golden-wallet dualwrite schema parity.
--
-- SQLite gained these columns after the original Postgres schema was created.
-- Without them, mirrored golden_wallets upserts fail; wallet_fills then hit
-- the golden_wallets FK and spam dualwrite warnings while SQLite remains the
-- only complete ledger.

ALTER TABLE golden_wallets
    ADD COLUMN IF NOT EXISTS avg_hold_time_hours DOUBLE PRECISION DEFAULT 0;

ALTER TABLE golden_wallets
    ADD COLUMN IF NOT EXISTS last_fill_sync_time BIGINT DEFAULT 0;
