"""A7: ``/metrics`` endpoint for Prometheus scraping.

Renders the trading-domain metric registry built in
:mod:`src.notifications.metrics`. Returns an empty body (with the
correct Content-Type) when prometheus_client is not installed, so
scraper health checks stay green even on bare requirements.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.notifications.metrics import content_type, render_metrics
from src.ui.v2.auth import verify_cookie


router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    if not verify_cookie(request):
        return Response(content="unauthorized", status_code=401)
    body = render_metrics()
    return Response(content=body, media_type=content_type(), status_code=200)
