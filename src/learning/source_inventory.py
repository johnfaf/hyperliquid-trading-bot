"""Data-source inventory and health history for continuous learning."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

DEFAULT_SOURCE_INVENTORY: List[Dict[str, Any]] = [
    {
        "source_name": "hyperliquid_candles",
        "category": "price_history",
        "required": True,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 180,
        "expected_freshness_seconds": 300,
        "notes": "Primary OHLCV feed and backtest price base.",
    },
    {
        "source_name": "hyperliquid_orderbook",
        "category": "microstructure",
        "required": True,
        "supports_live": True,
        "supports_historical": False,
        "point_in_time_safe": False,
        "min_history_days": 0,
        "expected_freshness_seconds": 10,
        "notes": "Live spread/imbalance/slippage guard.",
    },
    {
        "source_name": "hyperliquid_funding",
        "category": "derivatives",
        "required": False,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 180,
        "expected_freshness_seconds": 3600,
        "notes": "Funding-rate feature and carry risk input.",
    },
    {
        "source_name": "binance_candles",
        "category": "price_history",
        "required": False,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 365,
        "expected_freshness_seconds": 300,
        "notes": "Depth/confirmation history for broader liquidity context.",
    },
    {
        "source_name": "binance_open_interest",
        "category": "derivatives",
        "required": False,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 180,
        "expected_freshness_seconds": 3600,
        "notes": "OI pressure and divergence features.",
    },
    {
        "source_name": "deribit_options",
        "category": "options_flow",
        "required": False,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 180,
        "expected_freshness_seconds": 900,
        "notes": "IV rank, skew and directional flow features.",
    },
    {
        "source_name": "polymarket",
        "category": "prediction_market",
        "required": False,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 180,
        "expected_freshness_seconds": 300,
        "notes": "Event probability and event-flow feature source.",
    },
    {
        "source_name": "golden_wallets",
        "category": "copy_trading",
        "required": False,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 180,
        "expected_freshness_seconds": 3600,
        "notes": "Wallet discovery/scoring and copy-trader priors.",
    },
    {
        "source_name": "feature_store",
        "category": "derived_features",
        "required": True,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 180,
        "expected_freshness_seconds": 300,
        "notes": "Canonical model input table.",
    },
    {
        "source_name": "decision_journal",
        "category": "labels",
        "required": True,
        "supports_live": True,
        "supports_historical": True,
        "point_in_time_safe": True,
        "min_history_days": 30,
        "expected_freshness_seconds": 60,
        "notes": "Decision snapshots for training labels and attribution.",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Dict[str, Any]) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def seed_source_inventory(
    rows: Iterable[Dict[str, Any]] = DEFAULT_SOURCE_INVENTORY,
    *,
    mirror_to_postgres: bool = True,
) -> int:
    """Upsert the static inventory of learning data sources."""
    from src.data import database as db

    now = _now()
    count = 0
    with db.get_connection(for_read=not mirror_to_postgres) as conn:
        for row in rows:
            conn.execute(
                """
                INSERT INTO source_inventory
                (source_name, category, required, supports_live, supports_historical,
                 point_in_time_safe, min_history_days, expected_freshness_seconds,
                 owner, notes, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_name) DO UPDATE SET
                    category = EXCLUDED.category,
                    required = EXCLUDED.required,
                    supports_live = EXCLUDED.supports_live,
                    supports_historical = EXCLUDED.supports_historical,
                    point_in_time_safe = EXCLUDED.point_in_time_safe,
                    min_history_days = EXCLUDED.min_history_days,
                    expected_freshness_seconds = EXCLUDED.expected_freshness_seconds,
                    owner = EXCLUDED.owner,
                    notes = EXCLUDED.notes,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    row["source_name"],
                    row["category"],
                    bool(row.get("required", False)),
                    bool(row.get("supports_live", True)),
                    bool(row.get("supports_historical", False)),
                    bool(row.get("point_in_time_safe", False)),
                    int(row.get("min_history_days", 0) or 0),
                    int(row.get("expected_freshness_seconds", 0) or 0),
                    row.get("owner", "bot"),
                    row.get("notes", ""),
                    _json(row.get("metadata", {})),
                    now,
                    now,
                ),
            )
            count += 1
    return count


def record_source_health_snapshot(registry, *, mirror_to_postgres: bool = True) -> int:
    """Persist a point-in-time health snapshot from DataSourceRegistry."""
    if registry is None or not hasattr(registry, "snapshot"):
        return 0
    from src.data import database as db

    observed_at = _now()
    now_ts = time.time()
    snapshot = registry.snapshot()
    with db.get_connection(for_read=not mirror_to_postgres) as conn:
        for name, state in snapshot.items():
            last_success = float(state.get("last_success_at", 0.0) or 0.0)
            freshness = max(0.0, now_ts - last_success) if last_success > 0 else None
            conn.execute(
                """
                INSERT INTO data_source_health_history
                (observed_at, source_name, status, freshness_seconds, reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observed_at,
                    name,
                    state.get("state", "UNKNOWN"),
                    freshness,
                    state.get("reason", ""),
                    _json(state.get("metadata", {})),
                ),
            )
    return len(snapshot)
