"""Positions tab — live positions, kill-switch state, operator clear button.

This is the page operators reach for first when something is happening.
We render exchange-truth positions, surface the daily-PnL guard, expose
the calibration live-pause flag, and let an authenticated operator
clear the sticky kill switch through ``operator_clear_kill_switch`` on
:class:`LiveTrader`.

Dry-run / paper-only deployments degrade gracefully: when no live
trader is wired in we return an empty positions list and disable the
clear-switch button. Tests rely on this behaviour to avoid pulling in
the full LiveTrader stack.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from src.ui.v2.auth import require_auth, verify_cookie
from src.ui.v2.state import get_components

logger = logging.getLogger(__name__)

router = APIRouter()


def _safe_call(obj: Any, name: str, *args, **kwargs):
    """Call a method if it exists, swallowing exceptions.

    The dashboard must never crash because a subsystem returned
    ``None`` or threw. We log and degrade.
    """
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("%s.%s failed: %s", type(obj).__name__, name, exc)
        return None


def _format_position(raw: Dict[str, Any]) -> Dict[str, Any]:
    coin = str(raw.get("coin") or "")
    size = float(raw.get("size") or 0.0)
    side = str(raw.get("side") or ("long" if size > 0 else "short"))
    entry = float(raw.get("entry_price") or 0.0)
    upnl = float(raw.get("unrealized_pnl") or 0.0)
    leverage = float(raw.get("leverage") or 1.0)
    notional = abs(size) * entry
    return {
        "coin": coin,
        "side": side,
        "size": abs(size),
        "entry_price": entry,
        "unrealized_pnl": upnl,
        "leverage": leverage,
        "notional": notional,
        "is_cross": bool(raw.get("is_cross", True)),
    }


def _summary_payload() -> Dict[str, Any]:
    components = get_components()
    live_trader = components.live_trader
    cal = components.calibration

    positions: List[Dict[str, Any]] = []
    if live_trader is not None:
        raw_positions = _safe_call(live_trader, "get_positions") or []
        positions = [
            _format_position(p)
            for p in raw_positions
            if isinstance(p, dict) and abs(float(p.get("size") or 0.0)) > 0
        ]

    kill_state: Dict[str, Any] = {"active": False, "reason": None, "status_reason": None}
    if live_trader is not None:
        ks = _safe_call(live_trader, "get_kill_switch_state")
        if isinstance(ks, dict):
            kill_state = ks

    daily_pnl = None
    daily_realized = None
    daily_unrealized = None
    if live_trader is not None:
        daily_pnl = getattr(live_trader, "daily_pnl", None)
        daily_realized = getattr(live_trader, "daily_realized_pnl", None)
        daily_unrealized = getattr(live_trader, "daily_unrealized_pnl", None)

    calibration_live_paused = False
    global_ece = None
    if cal is not None:
        try:
            calibration_live_paused = bool(cal.is_live_paused())
        except Exception:
            calibration_live_paused = False
        try:
            global_ece = cal.get_ece("global")
        except Exception:
            global_ece = None

    total_unrealized = sum(p["unrealized_pnl"] for p in positions)
    total_notional = sum(p["notional"] for p in positions)

    return {
        "live_available": live_trader is not None,
        "dry_run": bool(getattr(live_trader, "dry_run", True)) if live_trader else True,
        "positions": positions,
        "totals": {
            "count": len(positions),
            "unrealized_pnl": total_unrealized,
            "notional": total_notional,
        },
        "daily": {
            "pnl": daily_pnl,
            "realized": daily_realized,
            "unrealized": daily_unrealized,
            "max_loss": getattr(live_trader, "max_daily_loss", None) if live_trader else None,
        },
        "kill_switch": kill_state,
        "calibration": {
            "live_paused": calibration_live_paused,
            "global_ece": global_ece,
        },
    }


@router.get("/api/positions", response_class=JSONResponse)
async def positions_data(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(_summary_payload())


@router.post("/api/operator/clear_kill_switch")
async def clear_kill_switch(
    request: Request,
    audit_reason: str = Form(""),
):
    """Authenticated endpoint to clear the sticky kill switch.

    Requires a non-empty ``audit_reason`` so the operator's intent is
    captured in the kill-switch JSONL log. Public-read deployments
    cannot reach this -- ``verify_cookie`` is mandatory for POSTs
    even when ``DASHBOARD_PUBLIC_READ`` is on.
    """
    if not verify_cookie(request):
        return JSONResponse({"error": "auth_required"}, status_code=401)

    reason = (audit_reason or "").strip()
    if len(reason) < 4:
        return JSONResponse(
            {"error": "audit_reason_required",
             "message": "Provide a short note explaining why (at least 4 chars)."},
            status_code=400,
        )

    live_trader = get_components().live_trader
    if live_trader is None:
        return JSONResponse({"error": "live_trader_unavailable"}, status_code=503)

    operator = request.cookies.get("dashboard_v2_auth", "")[:16] or "dashboard"
    try:
        result = live_trader.operator_clear_kill_switch(reason=reason, operator=f"dashboard:{operator}")
    except AttributeError:
        return JSONResponse({"error": "operator_clear_unsupported"}, status_code=501)
    except Exception as exc:
        logger.error("operator_clear_kill_switch failed: %s", exc, exc_info=True)
        return JSONResponse({"error": "clear_failed", "message": str(exc)}, status_code=500)

    return JSONResponse({"ok": True, "result": result})


@router.get("/positions", response_class=HTMLResponse)
async def positions_page(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect
    from src.ui.v2.app import get_templates
    return get_templates().TemplateResponse(
        request,
        "positions.html",
        {"title": "Positions", "data": _summary_payload()},
    )
