"""Traders directory — every tracked address with its bot probability,
PnL, win rate, and a mark/unmark control.

The page joins ``db.get_active_traders`` (the active set used for
copy-trade analysis) with the ``known_bot_addresses`` set so the
operator can see at a glance who is being tracked vs already
classified as a bot. The mark/unmark endpoint flips the ``active``
flag; the next discovery cycle picks up the change.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.ui.v2.auth import require_auth, verify_cookie

logger = logging.getLogger(__name__)

router = APIRouter()


def _bot_tone(score: Optional[float]) -> str:
    """Map bot_score (0-N) to a UI tone bucket."""
    if score is None:
        return "unknown"
    s = float(score)
    if s >= 3:
        return "bot"
    if s >= 1:
        return "suspect"
    return "human"


def _format_trader(row: Dict[str, Any], known_bots: set) -> Dict[str, Any]:
    addr = str(row.get("address") or "")
    bot_score = row.get("bot_score")
    if bot_score is None:
        # Active list omits bots; use the membership signal as a proxy.
        bot_score = 0 if addr not in known_bots else 3
    return {
        "address": addr,
        "address_short": addr[:10] if addr else "",
        "active": bool(row.get("active", True)),
        "total_pnl": float(row.get("total_pnl") or 0.0),
        "roi_pct": float(row.get("roi_pct") or 0.0),
        "win_rate": float(row.get("win_rate") or 0.0),
        "trade_count": int(row.get("trade_count") or 0),
        "account_value": float(row.get("account_value") or 0.0),
        "bot_score": bot_score,
        "bot_tone": _bot_tone(bot_score),
        "is_known_bot": addr in known_bots,
    }


def _summary_payload() -> Dict[str, Any]:
    try:
        from src.data import database as db
    except Exception as exc:
        logger.warning("traders: db unavailable: %s", exc)
        return {"available": False, "rows": [], "totals": {}}

    try:
        active = db.get_active_traders()
    except Exception as exc:
        logger.warning("traders: get_active_traders failed: %s", exc)
        active = []
    try:
        known_bots = db.get_known_bot_addresses() or set()
    except Exception:
        known_bots = set()

    rows: List[Dict[str, Any]] = []
    for row in active:
        if not isinstance(row, dict):
            continue
        rows.append(_format_trader(row, known_bots))
    rows.sort(key=lambda r: (-r["total_pnl"], r["address"]))

    totals = {
        "active": sum(1 for r in rows if not r["is_known_bot"]),
        "known_bots": len(known_bots),
        "suspect": sum(1 for r in rows if r["bot_tone"] == "suspect"),
        "total_tracked": len(rows) + len(known_bots),
    }
    return {"available": True, "rows": rows, "totals": totals}


@router.get("/api/traders", response_class=JSONResponse)
async def traders_data(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(_summary_payload())


@router.post("/api/traders/{address}/mark_bot")
async def mark_bot(address: str, request: Request, audit_reason: str = Form("")):
    if not verify_cookie(request):
        return JSONResponse({"error": "auth_required"}, status_code=401)
    reason = (audit_reason or "operator-tagged").strip()[:500]
    try:
        from src.data import database as db
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE traders SET active = ? WHERE address = ?",
                (False, address),
            )
        logger.warning(
            "traders: address %s marked as bot by dashboard operator (reason=%s)",
            address[:10], reason,
        )
        return JSONResponse({"ok": True, "address": address, "active": False})
    except Exception as exc:
        logger.error("traders.mark_bot failed: %s", exc, exc_info=True)
        return JSONResponse({"error": "mark_failed", "message": str(exc)}, status_code=500)


@router.post("/api/traders/{address}/restore")
async def restore_trader(address: str, request: Request):
    if not verify_cookie(request):
        return JSONResponse({"error": "auth_required"}, status_code=401)
    try:
        from src.data import database as db
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE traders SET active = ? WHERE address = ?",
                (True, address),
            )
        logger.warning(
            "traders: address %s restored to active by dashboard operator",
            address[:10],
        )
        return JSONResponse({"ok": True, "address": address, "active": True})
    except Exception as exc:
        logger.error("traders.restore failed: %s", exc, exc_info=True)
        return JSONResponse({"error": "restore_failed", "message": str(exc)}, status_code=500)


@router.get("/traders", response_class=HTMLResponse)
async def traders_page(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect
    from src.ui.v2.app import get_templates
    return get_templates().TemplateResponse(
        request,
        "traders.html",
        {"title": "Traders", "data": _summary_payload()},
    )
