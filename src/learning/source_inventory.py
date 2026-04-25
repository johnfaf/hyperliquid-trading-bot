"""Data-source inventory and persisted health snapshots for continuous learning."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config

_LAST_SOURCE_HEALTH_SNAPSHOT_TS = 0.0

_STATE_SEVERITY = {
    "UP": 1,
    "UNKNOWN": 2,
    "DEGRADED": 3,
    "DOWN": 4,
    "FAILED": 5,
}


def _default_source_inventory() -> List[Dict[str, Any]]:
    feature_store_required = bool(getattr(config, "POSTGRES_DSN", ""))
    return [
        {
            "source_name": "hyperliquid_candles",
            "category": "price_history",
            "required": False,
            "supports_live": True,
            "supports_historical": True,
            "point_in_time_safe": True,
            "min_history_days": 180,
            "expected_freshness_seconds": 300,
            "notes": "Local candle cache and Hyperliquid price history used for backtests.",
            "metadata": {"health_probe": "candle_cache"},
        },
        {
            "source_name": "hyperliquid_orderbook",
            "category": "microstructure",
            "required": False,
            "supports_live": True,
            "supports_historical": False,
            "point_in_time_safe": False,
            "min_history_days": 0,
            "expected_freshness_seconds": 10,
            "notes": "Live Hyperliquid mids/orderbook probe. Core runtime health is tracked separately.",
            "metadata": {"health_probe": "hyperliquid_mids"},
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
            "metadata": {"health_probe": "funding_history"},
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
            "metadata": {"health_probe": "candle_cache"},
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
            "metadata": {"health_probe": "open_interest_history"},
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
            "metadata": {"health_aliases": ["options_flow", "deribit"]},
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
            "metadata": {"health_aliases": ["polymarket"]},
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
            "metadata": {"health_probe": "golden_wallets"},
        },
        {
            "source_name": "feature_store",
            "category": "derived_features",
            "required": feature_store_required,
            "supports_live": True,
            "supports_historical": True,
            "point_in_time_safe": True,
            "min_history_days": 180,
            "expected_freshness_seconds": 300,
            "notes": "Canonical model input table.",
            "metadata": {"health_probe": "feature_store"},
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
            "metadata": {"health_probe": "decision_journal"},
        },
    ]


DEFAULT_SOURCE_INVENTORY: List[Dict[str, Any]] = _default_source_inventory()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Dict[str, Any]) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _read_json(value: Any, default: Optional[dict] = None) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return dict(default or {})
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else dict(default or {})
    except Exception:
        return dict(default or {})


def _normalize_state(value: Any) -> str:
    state = str(value or "UNKNOWN").strip().upper() or "UNKNOWN"
    return state if state in _STATE_SEVERITY else "UNKNOWN"


def _best_state(*states: str) -> str:
    normalized = [_normalize_state(state) for state in states if str(state or "").strip()]
    if not normalized:
        return "UNKNOWN"
    return max(normalized, key=lambda item: _STATE_SEVERITY.get(item, 0))


def _snapshot_row(
    source_name: str,
    status: str,
    *,
    freshness_seconds: Optional[float],
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source_name": str(source_name or "").strip(),
        "status": _normalize_state(status),
        "freshness_seconds": None if freshness_seconds is None else max(0.0, float(freshness_seconds)),
        "reason": str(reason or ""),
        "metadata": dict(metadata or {}),
    }


def _resolve_candle_cache_path() -> Path:
    candidates = [
        Path(config.DB_PATH).resolve().with_name("candle_cache.db"),
        Path.cwd() / "data" / "candle_cache.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _derive_candle_cache_health(source_name: str) -> Dict[str, Any]:
    path = _resolve_candle_cache_path()
    if not path.exists():
        return _snapshot_row(
            source_name,
            "DEGRADED",
            freshness_seconds=None,
            reason="candle cache file missing",
            metadata={"path": str(path)},
        )
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute(
            """
            SELECT COUNT(*) AS c,
                   MAX(timestamp_ms) AS latest_ts,
                   COUNT(DISTINCT coin) AS coins,
                   COUNT(DISTINCT timeframe) AS timeframes
            FROM candles
            """
        ).fetchone()
        conn.close()
        count = int(row[0] or 0)
        latest_ts = int(row[1] or 0)
        latest_age_seconds = (time.time() - (latest_ts / 1000.0)) if latest_ts > 0 else None
        if count <= 0:
            return _snapshot_row(
                source_name,
                "DEGRADED",
                freshness_seconds=None,
                reason="candle cache is empty",
                metadata={"path": str(path)},
            )
        # ★ M22 FIX: previously hardcoded freshness_seconds=0.0 even though
        # latest_age_seconds was computed correctly into metadata.  Downstream
        # SLA checks gating on freshness_seconds saw "fresh" when the cache
        # could be hours old.  Use the real age.
        return _snapshot_row(
            source_name,
            "UP",
            freshness_seconds=latest_age_seconds,
            reason=f"candle cache has {count} rows",
            metadata={
                "path": str(path),
                "row_count": count,
                "coin_count": int(row[2] or 0),
                "timeframe_count": int(row[3] or 0),
                "latest_candle_age_seconds": latest_age_seconds,
            },
        )
    except Exception as exc:
        return _snapshot_row(
            source_name,
            "DEGRADED",
            freshness_seconds=None,
            reason=f"candle cache probe failed: {exc}",
            metadata={"path": str(path)},
        )


def _derive_hyperliquid_orderbook_health() -> Dict[str, Any]:
    try:
        from src.data.hyperliquid_client import get_all_mids

        mids = get_all_mids() or {}
        if mids:
            return _snapshot_row(
                "hyperliquid_orderbook",
                "UP",
                freshness_seconds=0.0,
                reason=f"mids probe ok ({len(mids)} assets)",
                metadata={"asset_count": len(mids)},
            )
        return _snapshot_row(
            "hyperliquid_orderbook",
            "DEGRADED",
            freshness_seconds=None,
            reason="mids probe returned no assets",
            metadata={},
        )
    except Exception as exc:
        return _snapshot_row(
            "hyperliquid_orderbook",
            "DEGRADED",
            freshness_seconds=None,
            reason=f"mids probe failed: {exc}",
            metadata={},
        )


def _derive_timestamp_table_health(source_name: str, table: str, column: str) -> Dict[str, Any]:
    from src.data import database as db

    try:
        with db.get_connection(for_read=True) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS c, MAX({column}) AS latest_ts FROM {table}"
            ).fetchone()
        count = int(row["c"] if hasattr(row, "keys") else row[0] or 0)
        latest_ts = row["latest_ts"] if hasattr(row, "keys") else row[1]
        if count <= 0 or latest_ts in (None, "", 0):
            return _snapshot_row(
                source_name,
                "DEGRADED",
                freshness_seconds=None,
                reason=f"{table} has no persisted rows",
                metadata={"table": table, "row_count": count},
            )
        freshness = time.time() - (float(latest_ts) / 1000.0)
        return _snapshot_row(
            source_name,
            "UP",
            freshness_seconds=freshness,
            reason=f"{table} contains persisted history",
            metadata={"table": table, "row_count": count},
        )
    except Exception as exc:
        return _snapshot_row(
            source_name,
            "DEGRADED",
            freshness_seconds=None,
            reason=f"{table} probe failed: {exc}",
            metadata={"table": table},
        )


def _derive_golden_wallet_health() -> Dict[str, Any]:
    from src.data import database as db

    try:
        with db.get_connection(for_read=True) as conn:
            wallet_count = conn.execute("SELECT COUNT(*) AS c FROM golden_wallets").fetchone()
            latest_fill = conn.execute("SELECT MAX(time_ms) AS latest_ts FROM wallet_fills").fetchone()
        count = int(wallet_count["c"] if hasattr(wallet_count, "keys") else wallet_count[0] or 0)
        latest_ts = latest_fill["latest_ts"] if hasattr(latest_fill, "keys") else latest_fill[0]
        if count <= 0:
            return _snapshot_row(
                "golden_wallets",
                "DEGRADED",
                freshness_seconds=None,
                reason="golden_wallets table has no rows",
                metadata={"wallet_count": count},
            )
        latest_fill_age_seconds = None
        if latest_ts not in (None, "", 0):
            latest_fill_age_seconds = time.time() - (float(latest_ts) / 1000.0)
        return _snapshot_row(
            "golden_wallets",
            "UP",
            freshness_seconds=0.0,
            reason=f"{count} golden wallets tracked",
            metadata={
                "wallet_count": count,
                "latest_fill_age_seconds": latest_fill_age_seconds,
            },
        )
    except Exception as exc:
        return _snapshot_row(
            "golden_wallets",
            "DEGRADED",
            freshness_seconds=None,
            reason=f"golden wallet probe failed: {exc}",
            metadata={},
        )


def _derive_decision_journal_health() -> Dict[str, Any]:
    from src.data import database as db

    try:
        with db.get_connection(for_read=True) as conn:
            journal_row = conn.execute(
                """
                SELECT COUNT(*) AS c,
                       MAX(COALESCE(updated_at, created_at)) AS latest_ts
                FROM decision_snapshots
                """
            ).fetchone()
        count = int(journal_row["c"] if hasattr(journal_row, "keys") else journal_row[0] or 0)
        latest_raw = journal_row["latest_ts"] if hasattr(journal_row, "keys") else journal_row[1]
        freshness = None
        if latest_raw:
            try:
                freshness = max(
                    0.0,
                    (datetime.now(timezone.utc) - datetime.fromisoformat(str(latest_raw).replace("Z", "+00:00"))).total_seconds(),
                )
            except Exception:
                freshness = None
        return _snapshot_row(
            "decision_journal",
            "UP",
            freshness_seconds=0.0,
            reason="decision journal tables available",
            metadata={
                "decision_count": count,
                "last_activity_age_seconds": freshness,
            },
        )
    except Exception as exc:
        return _snapshot_row(
            "decision_journal",
            "DOWN",
            freshness_seconds=None,
            reason=f"decision journal probe failed: {exc}",
            metadata={},
        )


def _derive_feature_store_health() -> Dict[str, Any]:
    try:
        from src.data import feature_store as fs
    except Exception as exc:
        return _snapshot_row(
            "feature_store",
            "DEGRADED",
            freshness_seconds=None,
            reason=f"feature store import failed: {exc}",
            metadata={},
        )

    if not getattr(config, "POSTGRES_DSN", ""):
        return _snapshot_row(
            "feature_store",
            "DEGRADED",
            freshness_seconds=None,
            reason="feature store disabled (POSTGRES_DSN not configured)",
            metadata={"postgres_enabled": False},
        )
    try:
        available = bool(fs._pg_available())
        if not available:
            return _snapshot_row(
                "feature_store",
                "DOWN",
                freshness_seconds=None,
                reason="feature store Postgres backend unavailable",
                metadata={"postgres_enabled": True},
            )
        candle_count = int(fs.get_candle_count() or 0)
        feature_count = int(fs.get_feature_count() or 0)
        status = "UP" if (candle_count > 0 or feature_count > 0) else "DEGRADED"
        reason = (
            f"feature store ready ({candle_count} candles, {feature_count} features)"
            if status == "UP"
            else "feature store reachable but empty"
        )
        return _snapshot_row(
            "feature_store",
            status,
            freshness_seconds=0.0,
            reason=reason,
            metadata={"candle_count": candle_count, "feature_count": feature_count},
        )
    except Exception as exc:
        return _snapshot_row(
            "feature_store",
            "DEGRADED",
            freshness_seconds=None,
            reason=f"feature store probe failed: {exc}",
            metadata={"postgres_enabled": True},
        )


def _registry_row(name: str, state: Dict[str, Any], *, now_ts: float) -> Dict[str, Any]:
    last_success = float(state.get("last_success_at", 0.0) or 0.0)
    freshness = max(0.0, now_ts - last_success) if last_success > 0 else None
    return _snapshot_row(
        name,
        state.get("state", "UNKNOWN"),
        freshness_seconds=freshness,
        reason=state.get("reason", ""),
        metadata=state.get("metadata", {}),
    )


def _alias_registry_row(
    source_name: str,
    registry_snapshot: Dict[str, Dict[str, Any]],
    aliases: Iterable[str],
) -> Dict[str, Any]:
    alias_rows = []
    now_ts = time.time()
    for alias in aliases:
        state = registry_snapshot.get(str(alias or "").strip().lower())
        if state:
            alias_rows.append(_registry_row(source_name, state, now_ts=now_ts))
    if not alias_rows:
        return _snapshot_row(
            source_name,
            "DEGRADED",
            freshness_seconds=None,
            reason=f"no live registry snapshot for aliases: {', '.join(aliases)}",
            metadata={"aliases": list(aliases)},
        )
    chosen = max(alias_rows, key=lambda row: _STATE_SEVERITY.get(row["status"], 0))
    metadata = dict(chosen.get("metadata") or {})
    metadata["aliases"] = list(aliases)
    return _snapshot_row(
        source_name,
        chosen["status"],
        freshness_seconds=chosen.get("freshness_seconds"),
        reason=chosen.get("reason", ""),
        metadata=metadata,
    )


def collect_source_health_snapshot_rows(registry=None) -> List[Dict[str, Any]]:
    registry_snapshot = {}
    if registry is not None and hasattr(registry, "snapshot"):
        try:
            registry_snapshot = registry.snapshot() or {}
        except Exception:
            registry_snapshot = {}
    registry_snapshot = {
        str(name or "").strip().lower(): dict(state or {})
        for name, state in dict(registry_snapshot).items()
        if str(name or "").strip()
    }

    rows_by_source: Dict[str, Dict[str, Any]] = {}
    now_ts = time.time()
    for name, state in registry_snapshot.items():
        rows_by_source[name] = _registry_row(name, state, now_ts=now_ts)

    rows_by_source["hyperliquid_candles"] = _derive_candle_cache_health("hyperliquid_candles")
    rows_by_source["binance_candles"] = _derive_candle_cache_health("binance_candles")
    rows_by_source["hyperliquid_orderbook"] = _derive_hyperliquid_orderbook_health()
    rows_by_source["hyperliquid_funding"] = _derive_timestamp_table_health(
        "hyperliquid_funding", "funding_history", "timestamp_ms"
    )
    rows_by_source["binance_open_interest"] = _derive_timestamp_table_health(
        "binance_open_interest", "open_interest_history", "timestamp_ms"
    )
    rows_by_source["deribit_options"] = _alias_registry_row(
        "deribit_options", registry_snapshot, ["options_flow", "deribit"]
    )
    if "polymarket" not in rows_by_source:
        rows_by_source["polymarket"] = _alias_registry_row(
            "polymarket", registry_snapshot, ["polymarket"]
        )
    rows_by_source["golden_wallets"] = _derive_golden_wallet_health()
    rows_by_source["feature_store"] = _derive_feature_store_health()
    rows_by_source["decision_journal"] = _derive_decision_journal_health()

    return [
        row
        for _, row in sorted(rows_by_source.items(), key=lambda item: item[0])
        if row.get("source_name")
    ]


def seed_source_inventory(
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    mirror_to_postgres: bool = True,
) -> int:
    """Upsert the static inventory of learning data sources."""
    from src.data import database as db

    now = _now()
    count = 0
    inventory_rows = list(rows or _default_source_inventory())
    with db.get_connection(for_read=not mirror_to_postgres) as conn:
        for row in inventory_rows:
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
    """Persist a point-in-time source-health snapshot from live and derived probes."""
    from src.data import database as db

    observed_at = _now()
    rows = collect_source_health_snapshot_rows(registry)
    if not rows:
        return 0
    with db.get_connection(for_read=not mirror_to_postgres) as conn:
        for row in rows:
            conn.execute(
                """
                INSERT INTO data_source_health_history
                (observed_at, source_name, status, freshness_seconds, reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observed_at,
                    row["source_name"],
                    row["status"],
                    row.get("freshness_seconds"),
                    row.get("reason", ""),
                    _json(row.get("metadata", {})),
                ),
            )
    return len(rows)


def persist_source_health_snapshot(
    registry=None,
    *,
    force: bool = False,
    mirror_to_postgres: bool = True,
) -> int:
    """Persist source health at a bounded cadence so freshness ages over time."""
    global _LAST_SOURCE_HEALTH_SNAPSHOT_TS

    interval = max(5, int(getattr(config, "SOURCE_HEALTH_SNAPSHOT_INTERVAL_S", 60) or 60))
    now_ts = time.time()
    if not force and (now_ts - _LAST_SOURCE_HEALTH_SNAPSHOT_TS) < interval:
        return 0
    count = int(record_source_health_snapshot(registry, mirror_to_postgres=mirror_to_postgres) or 0)
    _LAST_SOURCE_HEALTH_SNAPSHOT_TS = now_ts
    return count
