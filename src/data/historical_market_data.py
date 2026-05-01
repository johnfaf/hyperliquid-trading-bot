"""Historical derivative/source data helpers for the learning layer.

These stores are intentionally simple append/upsert tables. They give the
backtester and feature builder one canonical place for funding, open interest,
and options summaries without touching the live execution path.
"""

from __future__ import annotations

import json
import time
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


def snapshot_live_derivatives_history(
    coins: Optional[Iterable[str]] = None,
    *,
    observed_at_ms: Optional[int] = None,
) -> Dict[str, int]:
    """Persist one current funding/OI snapshot from Hyperliquid asset contexts."""
    from src.data import hyperliquid_client as hl

    observed = int(observed_at_ms or time.time() * 1000)
    contexts = hl.get_asset_contexts() or {}
    if not contexts:
        return {"funding_rows": 0, "open_interest_rows": 0}

    coin_filter = {
        str(coin or "").strip().upper()
        for coin in (coins or contexts.keys())
        if str(coin or "").strip()
    }
    funding_rows = []
    oi_rows = []
    for coin in sorted(coin_filter):
        ctx = contexts.get(coin) or {}
        if not ctx:
            continue
        funding_rate = float(ctx.get("funding", 0.0) or 0.0)
        open_interest = float(ctx.get("open_interest", 0.0) or 0.0)
        mark_price = float(ctx.get("mark_price", 0.0) or 0.0)
        annualized = funding_rate * 24.0 * 365.0
        funding_rows.append(
            {
                "source": "hyperliquid",
                "coin": coin,
                "timestamp_ms": observed,
                "funding_rate": funding_rate,
                "annualized": annualized,
                "metadata": {
                    "mark_price": mark_price,
                    "oracle_price": float(ctx.get("oracle_price", 0.0) or 0.0),
                    "premium": float(ctx.get("premium", 0.0) or 0.0),
                },
            }
        )
        oi_rows.append(
            {
                "source": "hyperliquid",
                "coin": coin,
                "timestamp_ms": observed,
                "open_interest": open_interest,
                "notional_usd": (open_interest * mark_price) if mark_price > 0 else None,
                "metadata": {
                    "mark_price": mark_price,
                    "day_volume": float(ctx.get("day_volume", 0.0) or 0.0),
                },
            }
        )

    return {
        "funding_rows": store_funding_points(funding_rows),
        "open_interest_rows": store_open_interest_points(oi_rows),
    }


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
