"""Top-level page routes for the v2 dashboard.

Each route renders an HTMX-friendly Jinja template. Sub-pages live
in their own routers (``calibration``, future: ``positions``,
``traders``, ``audit``).
"""
from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from src.data.decision_coverage import read_report as decision_coverage_report
from src.ui.v2.auth import require_auth, verify_cookie
from src.ui.v2.cache import get_ttl
from src.ui.v2.desk import STOCK_LABELS, snapshot
from src.ui.v2.events import get_recent_events
from src.ui.v2.trade_review import detail, review

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, hours: int = Query(default=24)):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect
    from src.ui.v2.app import get_templates
    hours = 168 if hours == 168 else 24
    desk = await run_in_threadpool(get_ttl, ("desk", hours), 15, lambda: snapshot(hours))
    return get_templates().TemplateResponse(
        request, "dashboard.html", {"title": "Overview", "desk": desk}
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    if not verify_cookie(request):
        return RedirectResponse("/login", status_code=303)
    from src.ui.v2.app import get_templates
    desk = await run_in_threadpool(get_ttl, ("desk", 24), 15, snapshot)
    return get_templates().TemplateResponse(request, "settings.html", {"title": "Settings", "desk": desk})


@router.get("/stock-pilot", response_class=HTMLResponse)
async def stock_pilot(request: Request):
    if not verify_cookie(request):
        return RedirectResponse("/login", status_code=303)
    from src.ui.v2.app import get_templates
    desk = await run_in_threadpool(get_ttl, ("desk", 24), 15, snapshot)
    return get_templates().TemplateResponse(
        request, "stock_pilot.html", {"title": "Stock Pilot", "desk": desk, "labels": STOCK_LABELS},
    )


@router.get("/trades", response_class=HTMLResponse)
async def trades(request: Request, days: int = 7, group: str = "source", scope: str = "copy"):
    if not verify_cookie(request):
        return RedirectResponse("/login", status_code=303)
    from src.ui.v2.app import get_templates
    days = days if days in (1, 7, 30) else 7
    group = group if group in ("source", "asset", "side") else "source"
    scope = "all" if scope == "all" else "copy"
    report = await run_in_threadpool(get_ttl, ("trade-review", days, group, scope), 20, lambda: review(days, group, scope))
    return get_templates().TemplateResponse(request, "trade_review.html", {"title": "Trade review", "report": report})


@router.get("/trades/{trade_id}", response_class=HTMLResponse)
async def trade_details(request: Request, trade_id: int):
    if not verify_cookie(request):
        return RedirectResponse("/login", status_code=303)
    from src.ui.v2.app import get_templates
    try:
        trade = await run_in_threadpool(detail, trade_id)
    except sqlite3.Error:
        raise HTTPException(503, "Paper trade ledger unavailable") from None
    if not trade:
        raise HTTPException(404, "Paper trade not found")
    return get_templates().TemplateResponse(request, "trade_details.html", {"title": f"Trade {trade_id}", "trade": trade})


@router.get("/decision-coverage", response_class=HTMLResponse)
async def decision_coverage(request: Request):
    if not verify_cookie(request):
        return RedirectResponse("/login", status_code=303)
    from src.ui.v2.app import get_templates
    report = await run_in_threadpool(get_ttl, "decision-coverage", 10, decision_coverage_report)
    return get_templates().TemplateResponse(request, "coverage.html", {"title": "Decision Coverage", "report": report})


@router.get("/api/events/recent", response_class=JSONResponse)
async def recent_events(request: Request, limit: int = 100):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    events = get_recent_events(limit=limit)
    return JSONResponse({"events": events, "count": len(events)})
