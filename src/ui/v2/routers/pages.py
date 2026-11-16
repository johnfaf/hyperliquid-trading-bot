"""Top-level page routes for the v2 dashboard.

Each route renders an HTMX-friendly Jinja template. Sub-pages live
in their own routers (``calibration``, future: ``positions``,
``traders``, ``audit``).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from src.ui.v2.auth import require_auth, verify_cookie
from src.ui.v2.cache import get_ttl
from src.ui.v2.desk import STOCK_LABELS, snapshot
from src.ui.v2.events import get_recent_events

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


@router.get("/api/events/recent", response_class=JSONResponse)
async def recent_events(request: Request, limit: int = 100):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    events = get_recent_events(limit=limit)
    return JSONResponse({"events": events, "count": len(events)})
