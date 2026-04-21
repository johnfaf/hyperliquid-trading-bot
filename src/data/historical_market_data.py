"""Historical derivative/source data helpers for the learning layer.

These stores are intentionally simple append/upsert tables. They give the
backtester and feature builder one canonical place for funding, open interest,
and options summaries without touching the live execution path.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


def _json(value: Dict[str, Any]) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _rows_as_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def store_funding_points(rows: Iterable[Dict[str, Any]]) -> int:
    """Upsert funding history rows.

    Expected keys: source, coin, timestamp_ms, funding_rate. Optional:
    annualized, metadata.
    """
    from src.data import database as db

    items = list(rows or [])
    if not items:
        return 0
    with db.get_connection() as conn:
        for row in items:
            conn.execute(
                """
                INSERT INTO funding_history
                (source, coin, timestamp_ms, funding_rate, annualized, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, coin, timestamp_ms) DO UPDATE SET
                    funding_rate = EXCLUDED.funding_rate,
                    annualized = EXCLUDED.annualized,
                    metadata = EXCLUDED.metadata
                """,
                (
                    str(row.get("source", "unknown") or "unknown"),
                    str(row.get("coin", "") or "").upper(),
                    int(row.get("timestamp_ms", 0) or 0),
                    float(row.get("funding_rate", 0.0) or 0.0),
                    row.get("annualized"),
                    _json(row.get("metadata", {})),
                ),
            )
    return len(items)


def store_open_interest_points(rows: Iterable[Dict[str, Any]]) -> int:
    """Upsert open-interest history rows."""
    from src.data import database as db

    items = list(rows or [])
    if not items:
        return 0
    with db.get_connection() as conn:
        for row in items:
            conn.execute(
                """
                INSERT INTO open_interest_history
                (source, coin, timestamp_ms, open_interest, notional_usd, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, coin, timestamp_ms) DO UPDATE SET
                    open_interest = EXCLUDED.open_interest,
                    notional_usd = EXCLUDED.notional_usd,
                    metadata = EXCLUDED.metadata
                """,
                (
                    str(row.get("source", "unknown") or "unknown"),
                    str(row.get("coin", "") or "").upper(),
                    int(row.get("timestamp_ms", 0) or 0),
                    float(row.get("open_interest", 0.0) or 0.0),
                    row.get("notional_usd"),
                    _json(row.get("metadata", {})),
                ),
            )
    return len(items)


def store_options_summaries(rows: Iterable[Dict[str, Any]]) -> int:
    """Upsert options summary history rows."""
    from src.data import database as db

    items = list(rows or [])
    if not items:
        return 0
    with db.get_connection() as conn:
        for row in items:
            conn.execute(
                """
                INSERT INTO options_summary_history
                (source, coin, timestamp_ms, iv_rank, iv_percentile, skew,
                 call_put_ratio, net_premium_usd, flow_direction, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, coin, timestamp_ms) DO UPDATE SET
                    iv_rank = EXCLUDED.iv_rank,
                    iv_percentile = EXCLUDED.iv_percentile,
                    skew = EXCLUDED.skew,
                    call_put_ratio = EXCLUDED.call_put_ratio,
                    net_premium_usd = EXCLUDED.net_premium_usd,
                    flow_direction = EXCLUDED.flow_direction,
                    metadata = EXCLUDED.metadata
                """,
                (
                    str(row.get("source", "unknown") or "unknown"),
                    str(row.get("coin", "") or "").upper(),
                    int(row.get("timestamp_ms", 0) or 0),
                    row.get("iv_rank"),
                    row.get("iv_percentile"),
                    row.get("skew"),
                    row.get("call_put_ratio"),
                    row.get("net_premium_usd"),
                    row.get("flow_direction"),
                    _json(row.get("metadata", {})),
                ),
            )
    return len(items)


def get_funding_history(coin: str, limit: int = 100, source: Optional[str] = None) -> List[Dict[str, Any]]:
    from src.data import database as db

    params: List[Any] = [str(coin or "").upper()]
    source_clause = ""
    if source:
        source_clause = "AND source = ?"
        params.append(source)
    params.append(int(limit))
    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM funding_history
            WHERE coin = ? {source_clause}
            ORDER BY timestamp_ms DESC LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return _rows_as_dicts(rows)


def get_open_interest_history(coin: str, limit: int = 100, source: Optional[str] = None) -> List[Dict[str, Any]]:
    from src.data import database as db

    params: List[Any] = [str(coin or "").upper()]
    source_clause = ""
    if source:
        source_clause = "AND source = ?"
        params.append(source)
    params.append(int(limit))
    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM open_interest_history
            WHERE coin = ? {source_clause}
            ORDER BY timestamp_ms DESC LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return _rows_as_dicts(rows)


def get_latest_options_summary(coin: str, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from src.data import database as db

    params: List[Any] = [str(coin or "").upper()]
    source_clause = ""
    if source:
        source_clause = "AND source = ?"
        params.append(source)
    with db.get_connection(for_read=True) as conn:
        row = conn.execute(
            f"""
            SELECT * FROM options_summary_history
            WHERE coin = ? {source_clause}
            ORDER BY timestamp_ms DESC LIMIT 1
            """,
            tuple(params),
        ).fetchone()
    return dict(row) if row else None
