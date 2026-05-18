"""Audit trail tab — kill-switch transitions, calibration quarantines,
and recent decision-journal rows in one chronological feed.

Reads three sources:
  * ``data/kill_switch_log/YYYYMMDD.jsonl`` (last N days)
  * ``CalibrationTracker.get_quarantined_sources()`` (current state)
  * ``decision_snapshots`` table via direct SQL (recent rows, optionally
    filtered by ``decision_id`` / ``final_status``)

Each source contributes a row with a ``kind`` discriminator so the
frontend can colour-code without having to introspect.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.ui.v2.auth import require_auth
from src.ui.v2.state import get_components

logger = logging.getLogger(__name__)

router = APIRouter()

_KILL_LOG_DIR = Path("data") / "kill_switch_log"
_DEFAULT_DAYS = 7
_MAX_DECISIONS = 200


def _read_kill_switch_log(days: int = _DEFAULT_DAYS) -> List[Dict[str, Any]]:
    """Slurp the last N days of kill-switch JSONL. Skips missing files."""
    if not _KILL_LOG_DIR.exists():
        return []
    today = datetime.now(timezone.utc).date()
    out: List[Dict[str, Any]] = []
    for offset in range(days):
        d = today - timedelta(days=offset)
        path = _KILL_LOG_DIR / f"{d:%Y%m%d}.jsonl"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    payload["kind"] = (
                        "kill_switch_cleared"
                        if payload.get("event") == "kill_switch_cleared"
                        else "kill_switch_activated"
                    )
                    out.append(payload)
        except Exception as exc:
            logger.debug("kill_switch_log read failed for %s: %s", path, exc)
    out.sort(key=lambda r: str(r.get("timestamp_utc") or ""), reverse=True)
    return out


def _read_calibration_quarantines() -> List[Dict[str, Any]]:
    cal = get_components().calibration
    if cal is None:
        return []
    try:
        sources = cal.get_quarantined_sources() or []
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        rows.append({
            "kind": "calibration_quarantine",
            "source": src.get("source"),
            "side": src.get("side"),
            "regime": src.get("regime"),
            "ece": src.get("ece"),
            "samples": src.get("samples"),
            "key": src.get("source_key"),
        })
    return rows


def _loads(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _build_why(metadata: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the snapshot metadata into a compact why-enter/why-reject view.

    The snapshot's metadata carries ``why_entered`` / ``market_read`` /
    ``risk_and_sizing`` / ``ev_breakdown`` (written by
    ``decision_journal._decision_reason_metadata``). This shapes them
    into the few fields the dashboard actually renders so the template
    stays dumb.
    """
    why = metadata.get("why_entered") or {}
    market = metadata.get("market_read") or {}
    risk = metadata.get("risk_and_sizing") or {}
    ev = metadata.get("ev_breakdown") or {}

    final_status = str(decision.get("final_status") or "").lower()
    is_reject = "reject" in final_status or bool(decision.get("rejection_reason"))

    return {
        "verdict": "REJECT" if is_reject else (final_status.upper() or "—"),
        "rejection_reason": decision.get("rejection_reason") or why.get("rejection_reason") or "",
        "signal_reason": why.get("signal_reason") or "",
        "strategy_type": why.get("strategy_type") or "",
        "regime": market.get("overall_regime") or "",
        "regime_confidence": market.get("overall_confidence"),
        "countertrend_block_side": market.get("countertrend_block_side") or "",
        "market_side_alignment": market.get("market_side_alignment"),
        "ev": {
            "ev_bps": ev.get("ev_bps"),
            "sigma_bps": ev.get("sigma_bps"),
            "p_win": ev.get("p_win"),
            "p_win_source": ev.get("p_win_source"),
            "avg_win_bps": ev.get("avg_win_bps"),
            "avg_loss_bps": ev.get("avg_loss_bps"),
            "cost_bps": ev.get("cost_bps"),
        } if ev else {},
        "risk": {
            "leverage": risk.get("leverage"),
            "position_pct": risk.get("position_pct"),
            "entry_price": risk.get("entry_price"),
            "risk_policy": risk.get("risk_policy") or {},
        },
        "source_health": market.get("source_health") or {},
        "has_ev": bool(ev),
    }


def _read_decisions(decision_id: Optional[str], status: Optional[str], limit: int) -> List[Dict[str, Any]]:
    try:
        from src.data import database as db
    except Exception as exc:
        logger.debug("audit: db unavailable: %s", exc)
        return []
    limit = max(1, min(int(limit or _MAX_DECISIONS), 500))
    sql_parts = [
        "SELECT decision_id, created_at, coin, side, source, source_key, "
        "raw_confidence, calibrated_confidence, final_status, rejection_reason, "
        "regime, entry_price, proposed_sl_price, proposed_tp_price, "
        "proposed_size_usd, proposed_leverage, metadata "
        "FROM decision_snapshots"
    ]
    where_clauses: List[str] = []
    params: List[Any] = []
    if decision_id:
        where_clauses.append("decision_id = ?")
        params.append(decision_id)
    if status:
        where_clauses.append("final_status = ?")
        params.append(status)
    if where_clauses:
        sql_parts.append("WHERE " + " AND ".join(where_clauses))
    sql_parts.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)
    sql = " ".join(sql_parts)
    try:
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        logger.debug("audit: decision_snapshots read failed: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = dict(row) if not isinstance(row, dict) else row
        d["kind"] = "decision"
        metadata = _loads(d.pop("metadata", None))
        d["why"] = _build_why(metadata, d)
        out.append(d)
    return out


def _summary_payload(
    decision_id: Optional[str] = None,
    status: Optional[str] = None,
    days: int = _DEFAULT_DAYS,
    limit: int = _MAX_DECISIONS,
) -> Dict[str, Any]:
    kill_rows = _read_kill_switch_log(days=days)
    quarantine_rows = _read_calibration_quarantines()
    decision_rows = _read_decisions(decision_id, status, limit)

    counts = {
        "kill_switch_activations": sum(1 for r in kill_rows if r["kind"] == "kill_switch_activated"),
        "kill_switch_clears": sum(1 for r in kill_rows if r["kind"] == "kill_switch_cleared"),
        "active_quarantines": len(quarantine_rows),
        "decisions": len(decision_rows),
    }

    return {
        "filters": {"decision_id": decision_id or "", "status": status or "", "days": days, "limit": limit},
        "counts": counts,
        "kill_switch_log": kill_rows[:200],
        "calibration_quarantines": quarantine_rows,
        "decisions": decision_rows,
    }


@router.get("/api/audit", response_class=JSONResponse)
async def audit_data(
    request: Request,
    decision_id: Optional[str] = None,
    status: Optional[str] = None,
    days: int = _DEFAULT_DAYS,
    limit: int = _MAX_DECISIONS,
):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(_summary_payload(decision_id, status, days, limit))


@router.get("/audit", response_class=HTMLResponse)
async def audit_page(
    request: Request,
    decision_id: Optional[str] = None,
    status: Optional[str] = None,
):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect
    from src.ui.v2.app import get_templates
    return get_templates().TemplateResponse(
        request,
        "audit.html",
        {
            "title": "Audit",
            "data": _summary_payload(decision_id, status),
        },
    )
