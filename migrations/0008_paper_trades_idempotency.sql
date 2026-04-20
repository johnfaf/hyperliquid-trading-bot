-- =============================================================
-- Migration 0008: paper_trades idempotency key
--
-- H5 (audit): ``open_paper_trade()`` has no idempotency key, so a
-- crash between the SQLite commit and the Postgres mirror could
-- replay a logically-identical insert on retry, and a higher-level
-- retry (network blip re-triggering the signal pipeline) could insert
-- the same trade twice.  We add a nullable ``client_order_id`` column
-- that the caller fills with a stable per-signal key, plus a unique
-- partial index so Postgres rejects duplicates at the storage layer.
--
-- Existing rows keep ``client_order_id = NULL`` and the partial
-- uniqueness clause (``WHERE client_order_id IS NOT NULL``) avoids
-- a one-shot conflict on the backfill.  Callers that do not supply
-- a key keep the current behavior (always-insert) — the idempotency
-- guard is opt-in.
-- =============================================================

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS client_order_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_trades_client_order_id
    ON paper_trades (client_order_id)
    WHERE client_order_id IS NOT NULL;
