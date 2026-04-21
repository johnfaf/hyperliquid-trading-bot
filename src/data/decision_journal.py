"""Decision snapshot journal.

This records the full signal context at the point decisions are made so future
training/backtests can learn from accepted, rejected, and executed candidates.
All functions are best-effort: logging data must never block trading.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.learning.policy_registry import CHAMPION_POLICY_ID

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return str(os.environ.get("DECISION_JOURNAL_ENABLED", "true")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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
        logger.debug("Decision snapshot write skipped: %s", exc)
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
        logger.debug("Decision status update skipped: %s", exc)
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
        logger.debug("Decision paper-trade link skipped: %s", exc)
        return False
