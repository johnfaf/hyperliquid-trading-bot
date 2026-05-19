"""A7: ``/metrics`` endpoint for Prometheus scraping.

Renders the trading-domain metric registry built in
:mod:`src.notifications.metrics`. Returns an empty body (with the
correct Content-Type) when prometheus_client is not installed, so
scraper health checks stay green even on bare requirements.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from src.notifications.metrics import content_type, render_metrics


router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    body = render_metrics()
    return Response(content=body, media_type=content_type(), status_code=200)
