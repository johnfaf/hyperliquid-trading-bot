"""Calibration tab — the v2 dashboard's first concrete page.

Surfaces what the ``CalibrationTracker`` learned from recent trades:
global ECE/Brier, the live-pause flag, and a per-source breakdown
keyed on (source, side, regime). The new calibrator is the operator's
single best lens on whether confidence is meaningful — this page
makes that visible without reading logs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.ui.v2.auth import require_auth
from src.ui.v2.state import get_components

logger = logging.getLogger(__name__)

router = APIRouter()


def _templates() -> Jinja2Templates:
    # Lazy so tests can override the search path.
    from src.ui.v2.app import get_templates
    return get_templates()


def _serialize_curve(cal, key: str) -> List[Dict[str, Any]]:
    try:
        return cal.get_calibration_curve(key)
    except Exception as exc:
        logger.debug("curve fetch failed for %s: %s", key, exc)
        return []


def _summary_payload(cal) -> Dict[str, Any]:
    try:
        global_ece = cal.get_ece("global")
    except Exception:
        global_ece = None
    try:
        global_brier = cal.get_brier("global")
    except Exception:
        global_brier = None
    try:
        live_paused = cal.is_live_paused()
    except Exception:
        live_paused = False
    try:
        quarantined = cal.get_quarantined_sources()
    except Exception:
        quarantined = []
    try:
        all_stats = cal.get_all_stats()
    except Exception:
        all_stats = {}

    rows: List[Dict[str, Any]] = []
    for key, stats in all_stats.items():
        if key == "global":
            continue
        # Decompose the (source, side, regime) key for display.
        try:
            from src.signals.calibration import decompose_calibration_key
            source, side, regime = decompose_calibration_key(key)
        except Exception:
            source, side, regime = key, "_", "any"
        rows.append({
            "key": key,
            "source": source,
            "side": side,
            "regime": regime,
            "samples": stats.get("total_records", 0),
            "ece": stats.get("ece"),
            "brier": stats.get("brier"),
            "quality": stats.get("calibration_quality"),
            "quarantined": bool(stats.get("quarantined")),
        })
    rows.sort(
        key=lambda r: (
            0 if r["quarantined"] else 1,
            -(r["ece"] if r["ece"] is not None else -1),
            -r["samples"],
        )
    )
    return {
        "global": {
            "ece": global_ece,
            "brier": global_brier,
            "live_paused": bool(live_paused),
            "n_sources": len(rows),
            "n_quarantined": len(quarantined),
            "thresholds": {
                "live_pause_ece": getattr(cal, "live_pause_ece", None),
                "quarantine_ece": getattr(cal, "quarantine_ece", None),
                "min_outcomes": getattr(cal, "min_outcomes", None),
                "isotonic_min_outcomes": getattr(cal, "isotonic_min_outcomes", None),
                "half_life_days": getattr(cal, "half_life_days", None),
            },
        },
        "sources": rows,
        "quarantined": quarantined,
    }


@router.get("/api/calibration", response_class=JSONResponse)
async def calibration_data(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)

    cal = get_components().calibration
    if cal is None:
        return JSONResponse({"error": "calibration_unavailable"}, status_code=503)
    return JSONResponse(_summary_payload(cal))


@router.get("/api/calibration/curve", response_class=JSONResponse)
async def calibration_curve(request: Request, key: str = "global"):
    """Return the bin-by-bin calibration curve for one (source|side|regime) key."""
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)

    cal = get_components().calibration
    if cal is None:
        return JSONResponse({"error": "calibration_unavailable"}, status_code=503)
    return JSONResponse({"key": key, "curve": _serialize_curve(cal, key)})


@router.get("/calibration", response_class=HTMLResponse)
async def calibration_page(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect

    cal = get_components().calibration
    payload: Optional[Dict[str, Any]] = (
        _summary_payload(cal) if cal is not None else None
    )
    return _templates().TemplateResponse(
        request,
        "calibration.html",
        {
            "title": "Calibration",
            "data": payload,
            "available": cal is not None,
        },
    )
