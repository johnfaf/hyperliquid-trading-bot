"""Decision snapshot journal.

This records the full signal context at the point decisions are made so future
training/backtests can learn from accepted, rejected, and executed candidates.
All functions are best-effort: logging data must never block trading.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import config
from src.learning.policy_registry import CHAMPION_POLICY_ID

logger = logging.getLogger(__name__)
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False
_SCHEMA_WARNED = False
_WRITE_FAILURE_LOCK = threading.Lock()
_WRITE_FAILURES: Counter[str] = Counter()
_WRITE_FAILURE_LAST_LOG: dict[str, float] = {}
_WRITE_FAILURE_LOG_INTERVAL_S = 60.0


def _record_write_failure(area: str, exc: Exception) -> None:
    now = time.time()
    with _WRITE_FAILURE_LOCK:
        _WRITE_FAILURES[str(area)] += 1
        count = int(_WRITE_FAILURES[str(area)])
        last_log = float(_WRITE_FAILURE_LAST_LOG.get(str(area), 0.0) or 0.0)
        should_log = now - last_log >= _WRITE_FAILURE_LOG_INTERVAL_S
        if should_log:
            _WRITE_FAILURE_LAST_LOG[str(area)] = now
    if should_log:
        logger.warning(
            "Decision journal %s failed (%d total): %s: %s",
            area,
            count,
            type(exc).__name__,
            exc,
        )


def get_write_failure_stats() -> Dict[str, Any]:
    with _WRITE_FAILURE_LOCK:
        by_area = dict(_WRITE_FAILURES)
        last_log = dict(_WRITE_FAILURE_LAST_LOG)
    return {
        "total": int(sum(by_area.values())),
        "by_area": by_area,
        "last_log_ts": last_log,
    }


POSTGRES_DECISION_SNAPSHOT_DDL = """
CREATE TABLE IF NOT EXISTS decision_snapshots (
    decision_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    signal_timestamp TIMESTAMPTZ,
    policy_id TEXT,
    model_version TEXT,
    coin TEXT,
    side TEXT,
    source TEXT,
    source_key TEXT,
    strategy_type TEXT,
    strategy_id TEXT,
    signal_id TEXT,
    raw_confidence DOUBLE PRECISION,
    calibrated_confidence DOUBLE PRECISION,
    firewall_decision TEXT,
    final_status TEXT NOT NULL DEFAULT 'candidate',
    rejection_reason TEXT,
    entry_price DOUBLE PRECISION,
    proposed_size_usd DOUBLE PRECISION,
    proposed_position_pct DOUBLE PRECISION,
    proposed_leverage DOUBLE PRECISION,
    proposed_sl_roe DOUBLE PRECISION,
    proposed_tp_roe DOUBLE PRECISION,
    proposed_sl_price DOUBLE PRECISION,
    proposed_tp_price DOUBLE PRECISION,
    paper_trade_id BIGINT,
    live_order_id TEXT,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_health JSONB NOT NULL DEFAULT '{}'::jsonb,
    regime JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_signal JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_recent
    ON decision_snapshots (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_status
    ON decision_snapshots (final_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_snapshots_trade
    ON decision_snapshots (paper_trade_id);

CREATE TABLE IF NOT EXISTS decision_stage_events (
    event_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    confidence DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_decision_stage_events_decision
    ON decision_stage_events (decision_id, created_at);
CREATE INDEX IF NOT EXISTS idx_decision_stage_events_stage
    ON decision_stage_events (stage, status, created_at DESC);

CREATE TABLE IF NOT EXISTS decision_outcomes (
    decision_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    coin TEXT,
    side TEXT,
    source TEXT,
    source_key TEXT,
    strategy_type TEXT,
    final_status TEXT,
    action_taken BOOLEAN NOT NULL DEFAULT FALSE,
    paper_trade_id BIGINT,
    label_win INTEGER,
    outcome_pnl DOUBLE PRECISION,
    outcome_return_pct DOUBLE PRECISION,
    exit_reason TEXT,
    hold_minutes DOUBLE PRECISION,
    max_favorable_r DOUBLE PRECISION,
    max_adverse_r DOUBLE PRECISION,
    forward_return_15m DOUBLE PRECISION,
    forward_return_1h DOUBLE PRECISION,
    forward_return_4h DOUBLE PRECISION,
    forward_return_24h DOUBLE PRECISION,
    would_have_won INTEGER,
    side_correct INTEGER,
    missed_profit_usd DOUBLE PRECISION,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    decision_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    outcome_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    explanation TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_decision_outcomes_source
    ON decision_outcomes (source, source_key, strategy_type);
CREATE INDEX IF NOT EXISTS idx_decision_outcomes_status
    ON decision_outcomes (final_status, action_taken);
CREATE INDEX IF NOT EXISTS idx_decision_outcomes_created
    ON decision_outcomes (created_at DESC);
"""


def _enabled() -> bool:
    return str(os.environ.get("DECISION_JOURNAL_ENABLED", "true")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def ensure_schema_ready(force: bool = False) -> bool:
    """Ensure the journal table exists on the active mirror before writing.

    Live log 2026-04-21 showed the dual-write safety gate tripping because
    Postgres was missing ``decision_snapshots`` while SQLite already had it.
    This best-effort bootstrap prevents an observability table from poisoning
    the dual-write health window.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY and not force:
        return True
    with _SCHEMA_LOCK:
        if _SCHEMA_READY and not force:
            return True
        try:
            from src.data import database as db
            from src.learning.schema import ensure_sqlite_schema

            if config.DB_BACKEND in ("sqlite", "dualwrite"):
                try:
                    with db.get_connection() as conn:
                        ensure_sqlite_schema(conn)
                except Exception as exc:
                    _record_write_failure("sqlite_schema_ensure", exc)

            if config.DB_BACKEND in ("postgres", "dualwrite"):
                from src.data.db.postgres import get_connection, return_connection

                pg_conn = get_connection()
                try:
                    cur = pg_conn.cursor()
                    for stmt in [s.strip() for s in POSTGRES_DECISION_SNAPSHOT_DDL.split(";") if s.strip()]:
                        cur.execute(stmt)
                    pg_conn.commit()
                except Exception:
                    try:
                        pg_conn.rollback()
                    except Exception:
                        pass
                    raise
                finally:
                    return_connection(pg_conn)
            _SCHEMA_READY = True
            return True
        except Exception as exc:
            _record_write_failure("schema_ensure", exc)
            return False


def _schema_or_skip() -> bool:
    """Return True when journal writes are safe; warn once when disabled."""
    global _SCHEMA_WARNED
    if ensure_schema_ready():
        return True
    if not _SCHEMA_WARNED:
        logger.warning(
            "Decision journal writes disabled because schema bootstrap failed; "
            "trading continues without optional decision snapshots."
        )
        _SCHEMA_WARNED = True
    return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        loaded = json.loads(value or "{}")
        return dict(loaded) if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "")


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_decision_id(signal: Any) -> str:
    """Return the stable decision id for a TradeSignal-like object."""
    context = getattr(signal, "context", None)
    if isinstance(context, dict):
        decision_id = str(context.get("decision_id") or "").strip()
        if decision_id:
            return decision_id
    return str(getattr(signal, "signal_id", "") or "").strip()


def _raw_signal(signal: Any, raw_signal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(raw_signal, dict):
        return dict(raw_signal)
    if hasattr(signal, "to_dict"):
        try:
            value = signal.to_dict()
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            pass
    return {
        "coin": getattr(signal, "coin", None),
        "side": _enum_value(getattr(signal, "side", "")),
        "confidence": getattr(signal, "confidence", None),
        "source": _enum_value(getattr(signal, "source", "")),
        "strategy_type": getattr(signal, "strategy_type", ""),
        "signal_id": getattr(signal, "signal_id", ""),
    }


def _proposed_size_usd(signal: Any, account_balance: Optional[float]) -> Optional[float]:
    entry = _float(getattr(signal, "entry_price", None), None)
    size = _float(getattr(signal, "size", None), None)
    if entry and entry > 0 and size and size > 0:
        return abs(entry * size)
    position_pct = _float(getattr(signal, "position_pct", None), None)
    if account_balance and position_pct:
        return abs(float(account_balance) * position_pct)
    return None


def _risk_snapshot(signal: Any) -> Dict[str, Optional[float]]:
    risk = getattr(signal, "risk", None)
    entry = _float(getattr(signal, "entry_price", None), None)
    leverage = _float(getattr(signal, "leverage", None), 1.0) or 1.0
    side = _enum_value(getattr(signal, "side", ""))
    snapshot: Dict[str, Optional[float]] = {
        "sl_roe": None,
        "tp_roe": None,
        "sl_price": None,
        "tp_price": None,
    }
    if not risk:
        return snapshot
    try:
        if hasattr(risk, "resolve_roe_stop_loss_pct"):
            snapshot["sl_roe"] = float(risk.resolve_roe_stop_loss_pct(leverage))
        else:
            snapshot["sl_roe"] = _float(getattr(risk, "stop_loss_pct", None), None)
        if hasattr(risk, "resolve_roe_take_profit_pct"):
            snapshot["tp_roe"] = float(risk.resolve_roe_take_profit_pct(leverage))
        else:
            snapshot["tp_roe"] = _float(getattr(risk, "take_profit_pct", None), None)
        if entry and entry > 0 and hasattr(risk, "resolve_trigger_prices"):
            sl_price, tp_price = risk.resolve_trigger_prices(entry, side, leverage)
            snapshot["sl_price"] = float(sl_price)
            snapshot["tp_price"] = float(tp_price)
    except Exception:
        logger.debug("Could not derive risk snapshot", exc_info=True)
    return snapshot


def record_decision_snapshot(
    signal: Any,
    *,
    raw_signal: Optional[Dict[str, Any]] = None,
    regime_data: Optional[Dict[str, Any]] = None,
    source_health: Optional[Dict[str, Any]] = None,
    account_balance: Optional[float] = None,
    final_status: str = "candidate",
    firewall_decision: Optional[str] = "pending",
    rejection_reason: Optional[str] = None,
    policy_id: str = CHAMPION_POLICY_ID,
    model_version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Insert/update a decision snapshot for a TradeSignal-like object."""
    if not _enabled():
        return None
    if not _schema_or_skip():
        return None
    try:
        from src.data import database as db

        decision_id = resolve_decision_id(signal)
        if not decision_id:
            return None
        now = _now()
        raw = _raw_signal(signal, raw_signal)
        context = getattr(signal, "context", None)
        context = context if isinstance(context, dict) else {}
        features = raw.get("features") or context.get("features") or {}
        risk = _risk_snapshot(signal)
        columns = [
            "decision_id",
            "created_at",
            "updated_at",
            "signal_timestamp",
            "policy_id",
            "model_version",
            "coin",
            "side",
            "source",
            "source_key",
            "strategy_type",
            "strategy_id",
            "signal_id",
            "raw_confidence",
            "calibrated_confidence",
            "firewall_decision",
            "final_status",
            "rejection_reason",
            "entry_price",
            "proposed_size_usd",
            "proposed_position_pct",
            "proposed_leverage",
            "proposed_sl_roe",
            "proposed_tp_roe",
            "proposed_sl_price",
            "proposed_tp_price",
            "paper_trade_id",
            "live_order_id",
            "features",
            "source_health",
            "regime",
            "raw_signal",
            "metadata",
        ]
        values = {
            "decision_id": decision_id,
            "created_at": now,
            "updated_at": now,
            "signal_timestamp": getattr(signal, "timestamp", now),
            "policy_id": policy_id,
            "model_version": model_version,
            "coin": getattr(signal, "coin", None),
            "side": _enum_value(getattr(signal, "side", "")),
            "source": _enum_value(getattr(signal, "source", raw.get("source", ""))),
            "source_key": raw.get("source_key") or context.get("source_key"),
            "strategy_type": getattr(signal, "strategy_type", raw.get("strategy_type", "")),
            "strategy_id": str(getattr(signal, "strategy_id", raw.get("strategy_id", "")) or ""),
            "signal_id": str(raw.get("signal_id") or getattr(signal, "signal_id", "") or ""),
            "raw_confidence": _float(raw.get("raw_confidence", raw.get("confidence", None)), None),
            "calibrated_confidence": _float(getattr(signal, "confidence", raw.get("confidence", None)), None),
            "firewall_decision": firewall_decision,
            "final_status": final_status,
            "rejection_reason": rejection_reason,
            "entry_price": _float(getattr(signal, "entry_price", raw.get("price", None)), None),
            "proposed_size_usd": _proposed_size_usd(signal, account_balance),
            "proposed_position_pct": _float(getattr(signal, "position_pct", None), None),
            "proposed_leverage": _float(getattr(signal, "leverage", None), None),
            "proposed_sl_roe": risk["sl_roe"],
            "proposed_tp_roe": risk["tp_roe"],
            "proposed_sl_price": risk["sl_price"],
            "proposed_tp_price": risk["tp_price"],
            "paper_trade_id": None,
            "live_order_id": None,
            "features": _json(features),
            "source_health": _json(source_health or context.get("source_health") or {}),
            "regime": _json(regime_data or context.get("regime_data") or {}),
            "raw_signal": _json(raw),
            "metadata": _json(metadata or {}),
        }
        placeholders = ", ".join(["?"] * len(columns))
        update_columns = [c for c in columns if c not in {"decision_id", "created_at"}]
        update_parts = []
        for column in update_columns:
            if column in {"paper_trade_id", "live_order_id"}:
                update_parts.append(
                    f"{column} = COALESCE(EXCLUDED.{column}, decision_snapshots.{column})"
                )
            else:
                update_parts.append(f"{column} = EXCLUDED.{column}")
        update_sql = ", ".join(update_parts)
        with db.get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO decision_snapshots ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(decision_id) DO UPDATE SET {update_sql}
                """,
                tuple(values[c] for c in columns),
            )
        return decision_id
    except Exception as exc:
        _record_write_failure("snapshot_write", exc)
        return None


def update_decision_status(
    decision_id: str,
    *,
    final_status: str,
    firewall_decision: Optional[str] = None,
    rejection_reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update the terminal/current status for a decision snapshot."""
    if not _enabled() or not decision_id:
        return False
    if not _schema_or_skip():
        return False
    try:
        from src.data import database as db

        now = _now()
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT metadata FROM decision_snapshots WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            merged_meta = _loads(row["metadata"] if row else None)
            if metadata:
                merged_meta.update(metadata)
            if row:
                conn.execute(
                    """
                    UPDATE decision_snapshots
                    SET updated_at = ?, final_status = ?, firewall_decision = ?,
                        rejection_reason = ?, metadata = ?
                    WHERE decision_id = ?
                    """,
                    (
                        now,
                        final_status,
                        firewall_decision,
                        rejection_reason,
                        _json(merged_meta),
                        decision_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO decision_snapshots
                    (decision_id, created_at, updated_at, final_status,
                     firewall_decision, rejection_reason, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        now,
                        now,
                        final_status,
                        firewall_decision,
                        rejection_reason,
                        _json(merged_meta),
                    ),
                )
        return True
    except Exception as exc:
        _record_write_failure("status_update", exc)
        return False


def link_paper_trade(
    decision_id: str,
    paper_trade_id: Any,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Attach the opened paper trade id to an existing decision snapshot."""
    if not _enabled() or not decision_id or paper_trade_id is None:
        return False
    if not _schema_or_skip():
        return False
    try:
        from src.data import database as db

        now = _now()
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT metadata FROM decision_snapshots WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            merged_meta = _loads(row["metadata"] if row else None)
            if metadata:
                merged_meta.update(metadata)
            if row:
                conn.execute(
                    """
                    UPDATE decision_snapshots
                    SET updated_at = ?, paper_trade_id = ?, final_status = ?, metadata = ?
                    WHERE decision_id = ?
                    """,
                    (now, int(paper_trade_id), "paper_opened", _json(merged_meta), decision_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO decision_snapshots
                    (decision_id, created_at, updated_at, final_status, paper_trade_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (decision_id, now, now, "paper_opened", int(paper_trade_id), _json(merged_meta)),
                )
        return True
    except Exception as exc:
        _record_write_failure("paper_trade_link", exc)
        return False


def record_stage_event(
    decision_id: str,
    *,
    stage: str,
    status: str,
    reason: Optional[str] = None,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Append a durable stage event for a decision.

    Stage events are the causal breadcrumb trail for later training: generated,
    feature-adjusted, rejected by a specific gate, opened, closed, and so on.
    Writes are best-effort and must never block trading.
    """
    if not _enabled() or not decision_id:
        return False
    if not _schema_or_skip():
        return False
    try:
        from src.data import database as db

        now = _now()
        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO decision_stage_events
                (event_id, decision_id, created_at, stage, status, reason,
                 confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    str(decision_id),
                    now,
                    str(stage or "unknown"),
                    str(status or "unknown"),
                    reason,
                    _float(confidence, None),
                    _json(metadata or {}),
                ),
            )
        return True
    except Exception as exc:
        _record_write_failure("stage_event_write", exc)
        return False


def finalize_decision(
    decision_id: str,
    *,
    final_status: str,
    stage: str,
    reason: Optional[str] = None,
    firewall_decision: Optional[str] = None,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Record a terminal/current decision state and matching stage event."""
    event_ok = record_stage_event(
        decision_id,
        stage=stage,
        status=final_status,
        reason=reason,
        confidence=confidence,
        metadata=metadata,
    )
    status_ok = update_decision_status(
        decision_id,
        final_status=final_status,
        firewall_decision=firewall_decision,
        rejection_reason=reason,
        metadata=metadata,
    )
    return bool(event_ok or status_ok)


def _brief_json_items(value: Any, limit: int = 5) -> list[str]:
    data = _loads(value)
    items: list[str] = []
    for key, raw in data.items():
        if len(items) >= limit:
            break
        if raw in (None, "", {}, []):
            continue
        try:
            if isinstance(raw, float):
                rendered = f"{raw:.4g}"
            else:
                rendered = str(raw)
        except Exception:
            rendered = "?"
        items.append(f"{key}={rendered}")
    return items


def _source_health_items(value: Any, limit: int = 5) -> list[str]:
    data = _loads(value)
    if not data:
        return []
    items: list[str] = []

    def walk(prefix: str, raw: Any) -> None:
        if len(items) >= limit:
            return
        if isinstance(raw, dict):
            status = str(raw.get("status") or raw.get("state") or "").strip()
            if status:
                if status.lower() not in {"up", "healthy", "ok"}:
                    items.append(f"{prefix or raw.get('source', 'source')}={status}")
                return
            for key, child in raw.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
            return
        status = str(raw or "").strip()
        if status and status.lower() not in {"up", "healthy", "ok"}:
            items.append(f"{prefix}={status}")

    walk("", data)
    return items[:limit]


def build_decision_explanation(
    decision: Dict[str, Any],
    *,
    outcome: Optional[Dict[str, Any]] = None,
    stage_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a compact human-readable explanation for training review."""
    outcome = outcome or {}
    stage_summary = stage_summary or {}
    parts = []
    source = decision.get("source") or "unknown"
    source_key = decision.get("source_key") or source
    side = decision.get("side") or "?"
    coin = decision.get("coin") or "?"
    confidence = _float(decision.get("calibrated_confidence"), None)
    if confidence is None:
        confidence = _float(decision.get("raw_confidence"), None)
    confidence_text = f", confidence={confidence:.2f}" if confidence is not None else ""
    parts.append(f"Opened candidate: {side} {coin} from {source_key}{confidence_text}.")

    status_bits = []
    for label, column in (
        ("firewall", "firewall_decision"),
        ("status", "final_status"),
        ("reject", "rejection_reason"),
    ):
        val = decision.get(column)
        if val not in (None, ""):
            status_bits.append(f"{label}={val}")
    if status_bits:
        parts.append("Decision: " + ", ".join(status_bits) + ".")

    risk_bits = []
    for label, column in (
        ("SL_ROE", "proposed_sl_roe"),
        ("TP_ROE", "proposed_tp_roe"),
        ("SL_price", "proposed_sl_price"),
        ("TP_price", "proposed_tp_price"),
        ("size_usd", "proposed_size_usd"),
        ("lev", "proposed_leverage"),
    ):
        val = decision.get(column)
        if val not in (None, ""):
            risk_bits.append(f"{label}={val}")
    if risk_bits:
        parts.append("Risk: " + ", ".join(str(v) for v in risk_bits) + ".")

    feature_bits = _brief_json_items(decision.get("features"), limit=4)
    if feature_bits:
        parts.append("Key features: " + ", ".join(feature_bits) + ".")

    regime_bits = _brief_json_items(decision.get("regime"), limit=3)
    if regime_bits:
        parts.append("Regime: " + ", ".join(regime_bits) + ".")

    unhealthy_sources = _source_health_items(decision.get("source_health"), limit=5)
    if unhealthy_sources:
        parts.append("Data health: " + ", ".join(unhealthy_sources) + ".")

    if stage_summary:
        last_stage = stage_summary.get("last_stage")
        last_reason = stage_summary.get("last_reason")
        if last_stage:
            parts.append(f"Decision path ended at {last_stage}: {last_reason or 'no reason'}.")

    if outcome:
        pnl = outcome.get("outcome_pnl")
        label = outcome.get("label_win")
        exit_reason = outcome.get("exit_reason") or outcome.get("close_reason")
        result = "win" if label == 1 else "loss" if label == 0 else "unlabelled"
        outcome_bits = [result]
        if pnl is not None:
            outcome_bits.append(f"pnl={pnl}")
        if exit_reason:
            outcome_bits.append(f"exit={exit_reason}")
        parts.append("Outcome: " + ", ".join(outcome_bits) + ".")

    return " ".join(parts)


def _fetch_stage_summary(conn: Any, decision_id: str) -> Dict[str, Any]:
    try:
        rows = conn.execute(
            """
            SELECT stage, status, reason, created_at
            FROM decision_stage_events
            WHERE decision_id = ?
            ORDER BY created_at ASC
            """,
            (decision_id,),
        ).fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    stages = [dict(row) for row in rows]
    last = stages[-1]
    return {
        "count": len(stages),
        "stages": [row.get("stage") for row in stages],
        "last_stage": last.get("stage"),
        "last_status": last.get("status"),
        "last_reason": last.get("reason"),
    }


def record_decision_outcome(
    decision_id: str,
    *,
    outcome: Optional[Dict[str, Any]] = None,
    forward_labels: Optional[Dict[str, Any]] = None,
    explanation: Optional[str] = None,
) -> bool:
    """Upsert the one-row training outcome for a decision snapshot."""
    if not _enabled() or not decision_id:
        return False
    if not _schema_or_skip():
        return False
    try:
        from src.data import database as db

        now = _now()
        outcome = dict(outcome or {})
        forward = dict(forward_labels or {})
        with db.get_connection() as conn:
            decision_row = conn.execute(
                "SELECT * FROM decision_snapshots WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            if not decision_row:
                return False
            decision = dict(decision_row)
            paper_trade_id = decision.get("paper_trade_id")
            paper = None
            if paper_trade_id is not None:
                try:
                    paper = conn.execute(
                        "SELECT * FROM paper_trades WHERE id = ?",
                        (paper_trade_id,),
                    ).fetchone()
                except Exception:
                    paper = None
            paper_dict = dict(paper) if paper else {}
            pnl = outcome.get("outcome_pnl")
            if pnl is None:
                pnl = paper_dict.get("pnl")
            pnl_float = _float(pnl, None)
            action_taken = bool(paper_trade_id is not None)
            if pnl_float is None and not action_taken:
                preferred_forward = None
                for key in ("forward_return_4h", "forward_return_1h", "forward_return_15m", "forward_return_24h"):
                    if forward.get(key) is not None:
                        preferred_forward = _float(forward.get(key), None)
                        break
                proposed_size = _float(decision.get("proposed_size_usd"), 0.0) or 0.0
                proposed_leverage = _float(decision.get("proposed_leverage"), 1.0) or 1.0
                if preferred_forward is not None and proposed_size > 0:
                    pnl_float = preferred_forward * proposed_size * max(proposed_leverage, 1.0)
            label_win = outcome.get("label_win")
            if label_win is None and pnl_float is not None:
                label_win = 1 if pnl_float > 0 else 0
            if label_win is None and forward.get("would_have_won") is not None:
                label_win = int(forward.get("would_have_won"))
            entry = _float(paper_dict.get("entry_price") or decision.get("entry_price"), None)
            size = _float(paper_dict.get("size"), None)
            leverage = _float(paper_dict.get("leverage") or decision.get("proposed_leverage"), 1.0) or 1.0
            notional = (entry or 0.0) * (size or 0.0) * max(leverage, 1.0)
            return_pct = outcome.get("outcome_return_pct")
            if return_pct is None and pnl_float is not None and notional > 0:
                return_pct = pnl_float / notional
            decision_meta = _loads(decision.get("metadata"))
            decision_meta.update(
                {
                    "raw_confidence": decision.get("raw_confidence"),
                    "calibrated_confidence": decision.get("calibrated_confidence"),
                    "firewall_decision": decision.get("firewall_decision"),
                    "final_status": decision.get("final_status"),
                    "proposed_size_usd": decision.get("proposed_size_usd"),
                    "proposed_position_pct": decision.get("proposed_position_pct"),
                    "proposed_leverage": decision.get("proposed_leverage"),
                    "proposed_sl_roe": decision.get("proposed_sl_roe"),
                    "proposed_tp_roe": decision.get("proposed_tp_roe"),
                    "proposed_sl_price": decision.get("proposed_sl_price"),
                    "proposed_tp_price": decision.get("proposed_tp_price"),
                    "rejection_reason": decision.get("rejection_reason"),
                    "source_health": _loads(decision.get("source_health")),
                    "regime": _loads(decision.get("regime")),
                }
            )
            outcome_meta = dict(outcome.get("metadata") or {})
            if paper_dict.get("metadata"):
                outcome_meta["paper_metadata"] = _loads(paper_dict.get("metadata"))
            forward_extra = {
                key: value
                for key, value in forward.items()
                if key
                not in {
                    "forward_return_15m",
                    "forward_return_1h",
                    "forward_return_4h",
                    "forward_return_24h",
                    "would_have_won",
                    "side_correct",
                    "missed_profit_usd",
                }
            }
            if forward_extra:
                outcome_meta["forward_label_metadata"] = forward_extra
            stage_summary = _fetch_stage_summary(conn, decision_id)
            payload = {
                "outcome_pnl": pnl_float,
                "label_win": label_win,
                "exit_reason": outcome.get("exit_reason") or outcome_meta.get("close_reason"),
                "close_reason": outcome.get("close_reason") or outcome_meta.get("close_reason"),
            }
            explanation_text = explanation or build_decision_explanation(
                decision,
                outcome=payload,
                stage_summary=stage_summary,
            )
            columns = [
                "decision_id", "created_at", "updated_at", "coin", "side",
                "source", "source_key", "strategy_type", "final_status",
                "action_taken", "paper_trade_id", "label_win", "outcome_pnl",
                "outcome_return_pct", "exit_reason", "hold_minutes",
                "max_favorable_r", "max_adverse_r", "forward_return_15m",
                "forward_return_1h", "forward_return_4h", "forward_return_24h",
                "would_have_won", "side_correct", "missed_profit_usd",
                "features", "decision_metadata", "outcome_metadata", "explanation",
            ]
            values = {
                "decision_id": decision_id,
                "created_at": decision.get("created_at") or now,
                "updated_at": now,
                "coin": decision.get("coin"),
                "side": decision.get("side"),
                "source": decision.get("source"),
                "source_key": decision.get("source_key"),
                "strategy_type": decision.get("strategy_type"),
                "final_status": decision.get("final_status"),
                "action_taken": action_taken,
                "paper_trade_id": paper_trade_id,
                "label_win": label_win,
                "outcome_pnl": pnl_float,
                "outcome_return_pct": _float(return_pct, None),
                "exit_reason": payload["exit_reason"],
                "hold_minutes": _float(outcome.get("hold_minutes"), None),
                "max_favorable_r": _float(outcome.get("max_favorable_r"), None),
                "max_adverse_r": _float(outcome.get("max_adverse_r"), None),
                "forward_return_15m": _float(forward.get("forward_return_15m"), None),
                "forward_return_1h": _float(forward.get("forward_return_1h"), None),
                "forward_return_4h": _float(forward.get("forward_return_4h"), None),
                "forward_return_24h": _float(forward.get("forward_return_24h"), None),
                "would_have_won": forward.get("would_have_won"),
                "side_correct": forward.get("side_correct"),
                "missed_profit_usd": _float(forward.get("missed_profit_usd"), None),
                "features": _json(_loads(decision.get("features"))),
                "decision_metadata": _json({**decision_meta, "stage_summary": stage_summary}),
                "outcome_metadata": _json(outcome_meta),
                "explanation": explanation_text,
            }
            placeholders = ", ".join(["?"] * len(columns))
            update_sql = ", ".join(
                f"{column} = EXCLUDED.{column}"
                for column in columns
                if column not in {"decision_id", "created_at"}
            )
            conn.execute(
                f"""
                INSERT INTO decision_outcomes ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(decision_id) DO UPDATE SET {update_sql}
                """,
                tuple(values[column] for column in columns),
            )
        return True
    except Exception as exc:
        _record_write_failure("outcome_write", exc)
        return False
