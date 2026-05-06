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
from typing import Any, Dict, List

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

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
        "sl_price": None,
        "tp_price": None,
        "mark_price": None,
    }


def _attach_protective_legs(positions: List[Dict[str, Any]], live_trader: Any) -> None:
    """Best-effort: tag each position with sl_price/tp_price from open orders."""
    if live_trader is None:
        return
    open_orders = _safe_call(live_trader, "get_open_orders") or []
    if not isinstance(open_orders, list):
        return
    classifier = getattr(live_trader, "_classify_protective_leg", None)
    if classifier is None:
        return
    by_coin: Dict[str, Dict[str, float]] = {}
    for order in open_orders:
        try:
            classified = classifier(order)
        except Exception:
            classified = None
        if not classified:
            continue
        coin = str(classified.get("coin") or "").upper()
        leg = classified.get("leg")
        trigger_px = classified.get("trigger_px")
        if not coin or leg not in ("sl", "tp") or trigger_px is None:
            continue
        by_coin.setdefault(coin, {})[leg] = float(trigger_px)
    for pos in positions:
        legs = by_coin.get(pos["coin"].upper())
        if not legs:
            continue
        pos["sl_price"] = legs.get("sl")
        pos["tp_price"] = legs.get("tp")


def _attach_mark_prices(positions: List[Dict[str, Any]], live_trader: Any) -> None:
    """Pull mid prices for each open coin so the spark can plot entry vs mark."""
    if live_trader is None or not positions:
        return
    fn = getattr(live_trader, "_get_mid_price", None)
    if fn is None:
        return
    for pos in positions:
        try:
            mid = fn(pos["coin"])
        except Exception:
            mid = None
        if isinstance(mid, (int, float)) and mid > 0:
            pos["mark_price"] = float(mid)


def _recent_fills(live_trader: Any, limit: int = 25) -> List[Dict[str, Any]]:
    """Fetch the bot's most recent fills for the operator ticker.

    The trader holds the cached api_manager already used for daily-PnL
    refreshes, so we reuse it. Failures degrade silently -- the ticker
    just shows "no recent fills" rather than blocking the page.
    """
    if live_trader is None:
        return []
    api = getattr(live_trader, "api_manager", None)
    address = getattr(live_trader, "public_address", None)
    if api is None or not address:
        return []
    try:
        from src.exchanges.api_manager import Priority  # type: ignore
        priority = Priority.NORMAL
    except Exception:
        priority = None
    try:
        if priority is not None:
            raw = api.post(
                {"type": "userFills", "user": address},
                priority=priority,
                timeout=5,
            )
        else:
            raw = api.post({"type": "userFills", "user": address}, timeout=5)
    except Exception as exc:
        logger.debug("recent_fills userFills failed: %s", exc)
        return []
    if not isinstance(raw, list):
        return []
    fills = sorted(
        (f for f in raw if isinstance(f, dict)),
        key=lambda f: f.get("time", 0),
        reverse=True,
    )[: max(1, int(limit or 25))]
    out: List[Dict[str, Any]] = []
    for f in fills:
        ts = f.get("time")
        try:
            ts_ms = int(ts) if ts is not None else None
        except (TypeError, ValueError):
            ts_ms = None
        out.append({
            "ts_ms": ts_ms,
            "coin": str(f.get("coin") or ""),
            "side": str(f.get("side") or "").lower(),  # "B"=buy/long, "A"=sell/short
            "size": float(f.get("sz") or f.get("size") or 0.0),
            "price": float(f.get("px") or f.get("price") or 0.0),
            "closed_pnl": float(f.get("closedPnl") or f.get("closed_pnl") or 0.0),
            "fee": float(f.get("fee") or 0.0),
        })
    return out


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
        _attach_protective_legs(positions, live_trader)
        _attach_mark_prices(positions, live_trader)

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


@router.get("/api/fills/recent", response_class=JSONResponse)
async def recent_fills(request: Request, limit: int = 25):
    """Most-recent fills for the live ticker on the positions page."""
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    components = get_components()
    fills = _recent_fills(components.live_trader, limit=limit)
    return JSONResponse({"fills": fills, "count": len(fills)})


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


@router.post("/api/operator/close_position")
async def close_live_position(
    request: Request,
    coin: str = Form(...),
    audit_reason: str = Form(""),
):
    """Authenticated operator override to flatten one live position.

    This is intentionally one-coin-at-a-time and audit-reason gated: it is
    meant for stuck SL/TP or emergency manual intervention, not routine
    strategy execution.
    """
    if not verify_cookie(request):
        return JSONResponse({"error": "auth_required"}, status_code=401)

    target_coin = str(coin or "").strip().upper()
    if not target_coin:
        return JSONResponse({"error": "coin_required"}, status_code=400)
    reason = (audit_reason or "").strip()
    if len(reason) < 4:
        return JSONResponse(
            {
                "error": "audit_reason_required",
                "message": "Provide a short note explaining why (at least 4 chars).",
            },
            status_code=400,
        )

    live_trader = get_components().live_trader
    if live_trader is None:
        return JSONResponse({"error": "live_trader_unavailable"}, status_code=503)

    operator = request.cookies.get("dashboard_v2_auth", "")[:16] or "dashboard"
    try:
        pre_cancel = _safe_call(live_trader, "cancel_all_orders_detailed", coin=target_coin)
        result = live_trader.close_position(target_coin)
        post_cancel = _safe_call(live_trader, "cancel_all_orders_detailed", coin=target_coin)
        logger.warning(
            "dashboard operator close_position coin=%s operator=%s reason=%s result=%s",
            target_coin,
            operator,
            reason,
            result,
        )
        try:
            from src.ui.v2.events import publish_event

            publish_event(
                "operator",
                action="close_position",
                coin=target_coin,
                operator=f"dashboard:{operator}",
                reason=reason,
                result=result,
            )
        except Exception:
            pass
        return JSONResponse(
            {
                "ok": True,
                "coin": target_coin,
                "result": result,
                "pre_cancel": pre_cancel,
                "post_cancel": post_cancel,
            }
        )
    except AttributeError:
        return JSONResponse({"error": "close_position_unsupported"}, status_code=501)
    except Exception as exc:
        logger.error("close_position override failed: %s", exc, exc_info=True)
        return JSONResponse({"error": "close_failed", "message": str(exc)}, status_code=500)


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
