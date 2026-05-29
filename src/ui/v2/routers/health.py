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
            _health_strip_include_db_audit(),
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


def _cached_readiness(key: str, ttl_s: float, include_db_audit: bool = True) -> dict:
    return get_ttl(
        key,
        ttl_s,
        lambda: _evaluate_readiness(include_db_audit=include_db_audit),
    )


def _health_strip_include_db_audit() -> bool:
    return os.environ.get(
        "DASHBOARD_V2_HEALTH_STRIP_INCLUDE_DB_AUDIT",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _evaluate_readiness(*, include_db_audit: bool = True) -> dict:
    from src.core.readiness import evaluate_readiness
    from src.ui.v2.state import get_components

    # Pass the live component container so readiness can actually see the
    # live trader.  Without it, evaluate_readiness has no handle on
    # ``container.live_trader`` (there is no fallback for it) and reports
    # ``live_requested=false`` / ``live_ready=false`` even when the bot is
    # running in LIVE mode -- which is exactly the stale reading the
    # /api/live_ready endpoint was serving.  ``get_stats()`` is in-memory
    # (state-lock read, no network), so this stays cheap.  health_registry
    # falls back to the global singleton inside evaluate_readiness when None.
    components = get_components()
    return evaluate_readiness(
        container=components,
        health_registry=getattr(components, "health_registry", None),
        include_db_audit=include_db_audit,
    )
