"""Polymarket schema must include the timestamp_ms-leading index.

Background
----------
PR #29's retention prune issues ranged DELETEs of the form:

    DELETE FROM polymarket_price_points
    WHERE timestamp_ms >= ? AND timestamp_ms < ?

Without a leading-``timestamp_ms`` index on Postgres this falls back
to a sequential scan over the whole table, which on 2026-05-26's
2.7M-row mirror exceeded Postgres's statement_timeout.  The pre-
existing ``idx_polymarket_price_points_recent`` is
``(token_id, timestamp_ms DESC)`` so it doesn't help -- it requires
the predicate to constrain token_id.

This test pins the canonical schema in ``src/learning/schema.py`` so
the timestamp-only index lands on every fresh deploy / developer DB.
"""
from __future__ import annotations

from src.learning import schema


def test_polymarket_price_points_has_timestamp_only_index():
    """The schema declares a ``(timestamp_ms)`` index on price_points."""
    sql = schema.SQLITE_DDL if hasattr(schema, "SQLITE_DDL") else ""
    # Some builds expose the DDL via different attribute names; fall
    # back to scanning the module source for the index declaration.
    if "idx_polymarket_price_points_ts" not in sql:
        import inspect
        sql = inspect.getsource(schema)

    assert "idx_polymarket_price_points_ts" in sql, (
        "Schema must declare idx_polymarket_price_points_ts so the "
        "retention prune's ranged DELETE doesn't seq-scan the full "
        "polymarket_price_points table"
    )
    # Must be ON polymarket_price_points (timestamp_ms) -- not compound
    # with token_id leading.
    assert (
        "ON polymarket_price_points (timestamp_ms)" in sql
        or "ON polymarket_price_points(timestamp_ms)" in sql
    ), (
        "idx_polymarket_price_points_ts must be on (timestamp_ms) alone, "
        "leading with timestamp_ms (the existing _recent index leads "
        "with token_id which doesn't help range scans)"
    )


def test_polymarket_market_snapshots_index_unchanged():
    """The existing snapshots index leading with ``observed_at_ms`` stays."""
    import inspect
    sql = inspect.getsource(schema)
    assert "idx_polymarket_snapshots_recent" in sql
    assert "(observed_at_ms DESC)" in sql
