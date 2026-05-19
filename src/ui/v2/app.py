"""FastAPI app factory + uvicorn launcher for the v2 dashboard.

The app is split across small routers under :mod:`src.ui.v2.routers`.
Templates live in :file:`src/ui/v2/templates`. Static assets in
:file:`src/ui/v2/static`; Tailwind is compiled locally so tab changes do
not trigger browser-side CSS generation.

To launch standalone:

    python -m uvicorn src.ui.v2.app:create_app --factory --host 0.0.0.0 --port 8081

In production the bot's boot sequence calls :func:`start_server`
which runs uvicorn on a thread so the trading cycle continues to
own the main thread.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.ui.v2.cache import invalidate as invalidate_dashboard_cache
from src.ui.v2.events import get_bus

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"

_templates: Optional[Jinja2Templates] = None


def _configure_stdio_encoding() -> None:
    """Keep Windows consoles from crashing on Unicode log messages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True, encoding="utf-8", errors="backslashreplace")
        except Exception:
            # Best effort only; logging must not fail because a wrapped stream
            # refuses reconfiguration.
            pass


def get_templates() -> Jinja2Templates:
    global _templates
    if _templates is None:
        _templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    return _templates


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Bind the event bus to the app's running loop so producer threads
    # can publish through ``run_coroutine_threadsafe``.
    get_bus().bind_loop(asyncio.get_running_loop())
    try:
        yield
    finally:
        # Nothing to tear down -- subscribers go away with their sockets.
        pass


def create_app() -> FastAPI:
    _configure_stdio_encoding()
    invalidate_dashboard_cache()
    app = FastAPI(
        title="Hyperliquid Trading Bot — Dashboard v2",
        docs_url="/api/docs" if _docs_enabled() else None,
        redoc_url=None,
        lifespan=_lifespan,
    )

    @app.middleware("http")
    async def _request_timing_middleware(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        slow_ms = _slow_request_threshold_ms()
        if elapsed_ms >= slow_ms:
            logger.warning(
                "dashboard_v2 slow request method=%s path=%s status=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                getattr(response, "status_code", "?"),
                elapsed_ms,
            )
        else:
            logger.debug(
                "dashboard_v2 request method=%s path=%s status=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                getattr(response, "status_code", "?"),
                elapsed_ms,
            )
        return response

    # Routers — each module owns a tightly-scoped slice of the app.
    from src.ui.v2.routers import audit as audit_router
    from src.ui.v2.routers import auth as auth_router
    from src.ui.v2.routers import backtest as backtest_router
    from src.ui.v2.routers import calibration as calibration_router
    from src.ui.v2.routers import health as health_router
    from src.ui.v2.routers import metrics as metrics_router  # A7
    from src.ui.v2.routers import pages as pages_router
    from src.ui.v2.routers import positions as positions_router
    from src.ui.v2.routers import sources as sources_router
    from src.ui.v2.routers import stream as stream_router
    from src.ui.v2.routers import traders as traders_router

    app.include_router(health_router.router)
    app.include_router(metrics_router.router)  # A7: /metrics for Prometheus scraping
    app.include_router(auth_router.router)
    app.include_router(pages_router.router)
    app.include_router(calibration_router.router)
    app.include_router(positions_router.router)
    app.include_router(sources_router.router)
    app.include_router(traders_router.router)
    app.include_router(audit_router.router)
    app.include_router(backtest_router.router)
    app.include_router(stream_router.router)

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


def _docs_enabled() -> bool:
    return os.environ.get("DASHBOARD_V2_DOCS", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _slow_request_threshold_ms() -> float:
    raw = os.environ.get("DASHBOARD_V2_SLOW_REQUEST_MS", "500")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 500.0


def start_server(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    log_level: str = "info",
    daemon: bool = True,
) -> threading.Thread:
    """Run uvicorn in a background thread.

    Returns the launcher thread so callers can join it on shutdown.
    The dashboard owns no critical state -- if it crashes, trading
    continues. We log the exception and exit the thread rather than
    propagate.
    """
    import uvicorn

    bind_host = host or os.environ.get("DASHBOARD_V2_HOST", "0.0.0.0")
    bind_port = int(port or os.environ.get("DASHBOARD_V2_PORT", "8081"))

    _configure_stdio_encoding()
    app = create_app()
    # Tell uvicorn to mute its WebSocket lifecycle "INFO: connection open"
    # writes -- on Railway those go to stderr and get misclassified as
    # ERROR by the log shipper, polluting error dashboards. We only want
    # warnings+ from uvicorn itself; access logs are off, app logs flow
    # through the standard logger configured upstream.
    quiet_loggers = {
        "uvicorn": {"level": "WARNING"},
        "uvicorn.error": {"level": "WARNING"},
        "uvicorn.access": {"level": "WARNING"},
    }
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
        },
        "loggers": {
            name: {"handlers": ["default"], "level": cfg["level"], "propagate": False}
            for name, cfg in quiet_loggers.items()
        },
        "root": {"handlers": ["default"], "level": "INFO"},
    }
    config = uvicorn.Config(
        app,
        host=bind_host,
        port=bind_port,
        log_level=log_level,
        access_log=False,
        loop="asyncio",
        log_config=log_config,
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        try:
            server.run()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("v2 dashboard server crashed: %s", exc)

    thread = threading.Thread(
        target=_run, name="dashboard-v2", daemon=daemon
    )
    thread.start()
    logger.info("Dashboard v2 listening on http://%s:%d", bind_host, bind_port)
    return thread
