-- =============================================================
-- Migration 0007: paper_trades closed-trade index
--
-- The hot dashboard/history query is:
--     SELECT * FROM paper_trades
--      WHERE status = 'closed'
--      ORDER BY closed_at DESC
--      LIMIT ?
--
-- The existing single-column ``idx_paper_trades_status(status)`` helps
-- only the WHERE filter; Postgres still has to sort every closed row by
-- ``closed_at`` to satisfy the ORDER BY + LIMIT.  As the paper trade
-- history grows past ~5k rows the sort puts the query over the
-- ``statement_timeout`` and dual-write logs fill with
-- ``QueryCanceled: canceling statement due to statement timeout``.
--
-- A partial index on ``closed_at DESC`` (filtered by status='closed')
-- lets the planner do an index-only range scan over the top-N newest
-- closes without any sort at all.  The partial predicate keeps the
-- index small (open rows are not indexed).
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_paper_trades_closed_recent
    ON paper_trades (closed_at DESC)
    WHERE status = 'closed';
