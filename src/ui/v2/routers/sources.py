"""Source scoreboard tab — agent-scorer status, calibration metrics, and
the live source-allocator policy in one row per source.

The page joins three streams:

* ``AgentScorer.get_scorecard()``: warmup/active/degraded/paused +
  dynamic_weight + win_rate + recent_pnl.
* ``CalibrationTracker``: per-source aggregate ECE/Brier (rolled up
  across (side, regime) breakdowns we already track separately on
  the calibration tab).
* ``DecisionFirewall``: surfaces source-cap reasons via the same
  ``get_source_policy`` API the firewall uses at runtime, so the
  display matches what live trading sees.

If a particular subsystem isn't initialised the row falls back to
``None`` for the missing field rather than dropping the row.
"""
from __future__ import annotations

from collections import defaultdict
import json
import logging
import math
from typing import Any, Dict, List

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.ui.v2.auth import require_auth, verify_cookie
from src.ui.v2.state import get_components

logger = logging.getLogger(__name__)

router = APIRouter()


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _strategy_breakdown(limit: int = 25) -> List[Dict[str, Any]]:
    """Aggregate recent decision outcomes by strategy_type for Sources."""
    try:
        from src.data import database as db

        if db.table_exists("decision_outcomes"):
            with db.get_connection(for_read=True) as conn:
                rows = conn.execute(
                    """
                    SELECT strategy_type, source, source_key, side, action_taken,
                           label_win, outcome_pnl, final_status, created_at
                    FROM decision_outcomes
                    ORDER BY created_at DESC
                    LIMIT 5000
                    """
                ).fetchall()
        elif db.table_exists("paper_trades"):
            with db.get_connection(for_read=True) as conn:
                rows = conn.execute(
                    """
                    SELECT pt.side, pt.pnl AS outcome_pnl, pt.status AS final_status,
                           pt.metadata, s.strategy_type
                    FROM paper_trades pt
                    LEFT JOIN strategies s ON s.id = pt.strategy_id
                    ORDER BY COALESCE(pt.closed_at, pt.opened_at) DESC
                    LIMIT 5000
                    """
                ).fetchall()
        else:
            return []
    except Exception as exc:
        logger.debug("strategy breakdown query failed: %s", exc)
        return []

    acc: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "strategy_type": "unknown",
            "decisions": 0,
            "executed": 0,
            "wins": 0,
            "labelled": 0,
            "pnl": 0.0,
            "longs": 0,
            "shorts": 0,
            "sources": set(),
            "latest_status": "",
        }
    )
    for row in rows:
        item = dict(row)
        metadata = _loads_dict(item.get("metadata"))
        strategy = str(
            item.get("strategy_type")
            or metadata.get("strategy_type")
            or "unknown"
        ).strip().lower() or "unknown"
        bucket = acc[strategy]
        bucket["strategy_type"] = strategy
        bucket["decisions"] += 1
        if bool(item.get("action_taken")) or str(item.get("final_status") or "").lower() == "closed":
            bucket["executed"] += 1
        label = item.get("label_win")
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            label_int = -1
        if label_int in (0, 1):
            bucket["labelled"] += 1
            if label_int == 1:
                bucket["wins"] += 1
        pnl = _safe_float(item.get("outcome_pnl"), 0.0)
        bucket["pnl"] += pnl
        side = str(item.get("side") or "").strip().lower()
        if side in {"long", "buy", "b"}:
            bucket["longs"] += 1
        elif side in {"short", "sell", "a"}:
            bucket["shorts"] += 1
        source = str(item.get("source_key") or item.get("source") or metadata.get("source_key") or "").strip()
        if source:
            bucket["sources"].add(source)
        if not bucket["latest_status"]:
            bucket["latest_status"] = str(item.get("final_status") or "")

    rows_out: List[Dict[str, Any]] = []
    for bucket in acc.values():
        labelled = int(bucket["labelled"])
        win_rate = (bucket["wins"] / labelled) if labelled else None
        decisions = int(bucket["decisions"])
        rows_out.append(
            {
                "strategy_type": bucket["strategy_type"],
                "decisions": decisions,
                "executed": int(bucket["executed"]),
                "win_rate": win_rate,
                "pnl": round(float(bucket["pnl"]), 6),
                "avg_pnl": round(float(bucket["pnl"]) / max(1, decisions), 6),
                "longs": int(bucket["longs"]),
                "shorts": int(bucket["shorts"]),
                "sources": len(bucket["sources"]),
                "latest_status": bucket["latest_status"] or "unknown",
            }
        )
    rows_out.sort(key=lambda r: (-r["decisions"], -abs(r["pnl"]), r["strategy_type"]))
    return rows_out[: max(1, int(limit or 25))]


def _aggregate_calibration_for_source(cal, source: str) -> Dict[str, Any]:
    """Roll up per-(source|side|regime) keys into a single source view.

    The calibration tab already shows the breakdown; this scoreboard is
    the bird's-eye view, so we collapse to the source identity. Sample
    counts sum, ECE is sample-weighted, ``quarantined`` is true if any
    sub-key is quarantined.
    """
    if cal is None:
        return {"samples": None, "ece": None, "brier": None, "quarantined": False, "subkeys": 0}
    try:
        from src.signals.calibration import (
            decompose_calibration_key,
        )
        all_stats = cal.get_all_stats()
    except Exception:
        return {"samples": None, "ece": None, "brier": None, "quarantined": False, "subkeys": 0}
    matched_keys = []
    for key in all_stats:
        if key == "global":
            continue
        try:
            src, _, _ = decompose_calibration_key(key)
        except Exception:
            src = key
        if src == source:
            matched_keys.append(key)
    if not matched_keys:
        return {"samples": None, "ece": None, "brier": None, "quarantined": False, "subkeys": 0}
    total_samples = 0.0
    weighted_ece = 0.0
    weighted_brier = 0.0
    brier_weight = 0.0
    quarantined = False
    for key in matched_keys:
        stats = all_stats.get(key, {}) or {}
        samples = float(stats.get("total_records") or 0.0)
        total_samples += samples
        ece = stats.get("ece")
        brier = stats.get("brier")
        if ece is not None and samples > 0:
            weighted_ece += float(ece) * samples
        if brier is not None and samples > 0:
            weighted_brier += float(brier) * samples
            brier_weight += samples
        if stats.get("quarantined"):
            quarantined = True
    avg_ece = (weighted_ece / total_samples) if total_samples > 0 else None
    avg_brier = (weighted_brier / brier_weight) if brier_weight > 0 else None
    return {
        "samples": total_samples,
        "ece": avg_ece,
        "brier": avg_brier,
        "quarantined": quarantined,
        "subkeys": len(matched_keys),
    }


def _summary_payload() -> Dict[str, Any]:
    components = get_components()
    scorer = components.agent_scorer
    cal = components.calibration
    strategies = _strategy_breakdown()

    rows: List[Dict[str, Any]] = []
    if scorer is None:
        return {"available": False, "rows": [], "totals": {}, "strategies": strategies}

    try:
        scorecard = scorer.get_scorecard() or []
    except Exception as exc:
        logger.warning("agent_scorer.get_scorecard failed: %s", exc)
        scorecard = []

    for entry in scorecard:
        source_key = str(entry.get("source_key") or "")
        if not source_key:
            continue
        cal_stats = _aggregate_calibration_for_source(cal, source_key)
        rows.append({
            "source": source_key,
            "rank": entry.get("rank"),
            "status": entry.get("status"),
            "completed": entry.get("completed_trades", 0),
            "win_rate": entry.get("win_rate"),
            "weighted_accuracy": entry.get("weighted_accuracy"),
            "dynamic_weight": entry.get("dynamic_weight"),
            "sharpe": entry.get("sharpe"),
            "avg_return": entry.get("avg_return"),
            "recent_pnl": entry.get("recent_pnl"),
            "total_pnl": entry.get("total_pnl"),
            "last_trade_at": entry.get("last_trade_at"),
            "calibration": cal_stats,
        })

    counts = {"active": 0, "warmup": 0, "degraded": 0, "paused": 0, "other": 0}
    for r in rows:
        bucket = r["status"] if r["status"] in counts else "other"
        counts[bucket] += 1

    return {
        "available": True,
        "rows": rows,
        "totals": counts,
        "strategies": strategies,
    }


@router.get("/api/sources", response_class=JSONResponse)
async def sources_data(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(_summary_payload())


@router.post("/api/sources/clear_quarantine")
async def clear_quarantine(
    request: Request,
    key: str = Form(...),
    audit_reason: str = Form(""),
):
    """Authenticated endpoint to lift a calibration quarantine on a
    single (source, side, regime) key.

    Drops the per-key bin counts, deletes the matching DB rows, and
    rebuilds the global aggregate. The source returns to cold-start.
    Requires a non-empty audit reason -- it lands in the v2 dashboard
    audit log via the calibration tracker's WARNING log.
    """
    if not verify_cookie(request):
        return JSONResponse({"error": "auth_required"}, status_code=401)
    composed = (key or "").strip()
    if not composed:
        return JSONResponse({"error": "key_required"}, status_code=400)
    reason = (audit_reason or "").strip()
    if len(reason) < 4:
        return JSONResponse(
            {"error": "audit_reason_required",
             "message": "Provide a short note (4+ chars) for the audit log."},
            status_code=400,
        )
    cal = get_components().calibration
    if cal is None:
        return JSONResponse({"error": "calibration_unavailable"}, status_code=503)
    try:
        result = cal.operator_clear_quarantine(composed, audit_reason=reason)
    except AttributeError:
        return JSONResponse({"error": "operator_clear_unsupported"}, status_code=501)
    except Exception as exc:
        logger.error("clear_quarantine failed: %s", exc, exc_info=True)
        return JSONResponse({"error": "clear_failed", "message": str(exc)}, status_code=500)
    # Push to WS so the audit feed updates live.
    try:
        from src.ui.v2.events import publish_event
        publish_event("calibration", transition="quarantine_cleared", key=composed, reason=reason)
    except Exception:
        pass
    return JSONResponse({"ok": True, "result": result})


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect
    from src.ui.v2.app import get_templates
    return get_templates().TemplateResponse(
        request,
        "sources.html",
        {"title": "Sources", "data": _summary_payload()},
    )
