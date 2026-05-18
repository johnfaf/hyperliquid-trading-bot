"""Health & readiness endpoints for the v2 dashboard.

Mirrors v1's contract so existing probes (Railway, Telegram, etc.)
keep working when v2 is fronted by the same port.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.ui.v2.cache import get_ttl

logger = logging.getLogger(__name__)

router = APIRouter()


def _cache_ttl(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return default


@router.get("/api/health")
async def health() -> JSONResponse:
    # Surface the running build so "is the bot even on the merged commit?"
    # is answerable without shelling into the container. Best-effort.
    version: dict = {}
    try:
        from src.core.build_info import get_build_info

        version = get_build_info()
    except Exception as exc:  # never let health fail on this
        logger.debug("health: build_info unavailable: %s", exc)
    return JSONResponse({"status": "ok", "version": version})


@router.get("/api/ready")
async def ready() -> JSONResponse:
    """Aggregated readiness check.

    Surfaces whatever ``evaluate_readiness`` reports today; the
    dashboard renders this in the header bar so operators see degraded
    state at a glance.
    """
    try:
        result = await run_in_threadpool(
            _cached_readiness,
            "dashboard_ready",
            _cache_ttl("DASHBOARD_V2_HEALTH_CACHE_SECONDS", 5.0),
        )
    except Exception as exc:
        logger.warning("readiness check failed: %s", exc)
        return JSONResponse({"ready": False, "error": str(exc)}, status_code=503)
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
        result = await run_in_threadpool(
            _cached_readiness,
            "dashboard_health_strip",
            _cache_ttl(
                "DASHBOARD_V2_HEALTH_STRIP_CACHE_SECONDS",
                _cache_ttl("DASHBOARD_V2_HEALTH_CACHE_SECONDS", 5.0),
            ),
        ) or {}
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


def _cached_readiness(key: str, ttl_s: float) -> dict:
    return get_ttl(key, ttl_s, _evaluate_readiness)


def _evaluate_readiness() -> dict:
    from src.core.readiness import evaluate_readiness

    return evaluate_readiness()
