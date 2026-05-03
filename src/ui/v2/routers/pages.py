"""Top-level page routes for the v2 dashboard.

Each route renders an HTMX-friendly Jinja template. Sub-pages live
in their own routers (``calibration``, future: ``positions``,
``traders``, ``audit``).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.ui.v2.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect
    from src.ui.v2.app import get_templates
    return get_templates().TemplateResponse(
        request, "dashboard.html", {"title": "Overview"}
    )
