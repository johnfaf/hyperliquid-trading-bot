"""Health & readiness endpoints for the v2 dashboard.

Mirrors v1's contract so existing probes (Railway, Telegram, etc.)
keep working when v2 is fronted by the same port.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.get("/api/ready")
async def ready() -> JSONResponse:
    """Aggregated readiness check.

    Surfaces whatever ``evaluate_readiness`` reports today; the
    dashboard renders this in the header bar so operators see degraded
    state at a glance.
    """
    try:
        from src.core.readiness import evaluate_readiness
        result = evaluate_readiness()
    except Exception as exc:
        logger.warning("readiness check failed: %s", exc)
        return JSONResponse(
            {"ready": False, "error": str(exc)}, status_code=503
        )
    return JSONResponse(result)


@router.get("/api/live_ready")
async def live_ready() -> JSONResponse:
    """Same probe as ``/api/ready`` but expected to be hit by the
    exchange-facing health monitor. Kept separate so we can later
    diverge the strictness without breaking operator dashboards."""
    return await ready()
