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


@router.get("/api/health/strip")
async def health_strip() -> JSONResponse:
    """Compact health snapshot for the dashboard header strip.

    Returns ``{tone, label, summary, reasons}`` where ``tone`` is one
    of ``green | amber | rose`` so the frontend can colour without
    interpreting the readiness payload.
    """
    payload: dict = {"tone": "amber", "label": "unknown", "summary": "no data", "reasons": []}
    try:
        from src.core.readiness import evaluate_readiness
        result = evaluate_readiness() or {}
    except Exception as exc:
        logger.warning("health strip readiness failed: %s", exc)
        return JSONResponse({
            "tone": "rose",
            "label": "error",
            "summary": "readiness probe crashed",
            "reasons": [str(exc)[:160]],
        })

    ready_flag = bool(result.get("ready"))
    live_ready_flag = bool(result.get("live_ready"))
    reasons = result.get("reasons") or []
    if ready_flag and live_ready_flag:
        payload = {
            "tone": "green",
            "label": "live ready",
            "summary": "all checks ok",
            "reasons": [],
        }
    elif ready_flag and not live_ready_flag:
        payload = {
            "tone": "amber",
            "label": "paper only",
            "summary": "ready but not live-deployable",
            "reasons": reasons[:5],
        }
    else:
        payload = {
            "tone": "rose",
            "label": "not ready",
            "summary": "core checks failing",
            "reasons": reasons[:5],
        }
    return JSONResponse(payload)
