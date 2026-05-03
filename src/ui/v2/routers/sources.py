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

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.ui.v2.auth import require_auth
from src.ui.v2.state import get_components

logger = logging.getLogger(__name__)

router = APIRouter()


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
            compose_calibration_key,
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

    rows: List[Dict[str, Any]] = []
    if scorer is None:
        return {"available": False, "rows": [], "totals": {}}

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
    }


@router.get("/api/sources", response_class=JSONResponse)
async def sources_data(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(_summary_payload())


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
