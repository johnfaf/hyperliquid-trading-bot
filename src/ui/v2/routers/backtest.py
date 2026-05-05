"""Backtest runner — start a golden-wallet scan + backtest, poll status,
read results.

Mirrors v1's ``/api/backtest/run`` semantics (one job at a time, runs
on a daemon thread) but exposes status to HTMX so the page can poll
without a full reload. Auth: GET status is gated through
``require_auth`` (cookie); POST run additionally requires
``verify_cookie`` so a public-read deployment cannot kick off jobs.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.ui.v2.auth import require_auth, verify_cookie

logger = logging.getLogger(__name__)

router = APIRouter()


# Single-job lock matching v1's contract; v1's lock lives in src.ui.dashboard
# and we want v2 to share it so an operator can't kick off two scans by
# bouncing between dashboards. We import lazily so the v2 module remains
# usable in tests that don't import the v1 dashboard.
_LOCAL_JOB_LOCK = threading.Lock()
_LOCAL_JOB_RUNNING = False
_LOCAL_JOB_STARTED_AT: Optional[float] = None
_LOCAL_JOB_RESULT: Optional[Dict[str, Any]] = None
_LOCAL_JOB_ERROR: Optional[str] = None


def _v1_state() -> Optional[Dict[str, Any]]:
    """Best-effort read of v1's job state. Returns None when v1 is absent."""
    try:
        from src.ui import dashboard as v1
    except Exception:
        return None
    try:
        return {
            "running": bool(getattr(v1, "_BACKTEST_JOB_RUNNING", False)),
            "started_at": getattr(v1, "_BACKTEST_JOB_STARTED_AT", None),
        }
    except Exception:
        return None


def _status_payload() -> Dict[str, Any]:
    v1 = _v1_state()
    started = _LOCAL_JOB_STARTED_AT
    running = _LOCAL_JOB_RUNNING
    if v1 is not None:
        running = running or bool(v1.get("running"))
        if v1.get("started_at"):
            started = started or v1["started_at"]
    elapsed = (time.time() - started) if (running and started) else None
    try:
        from src.data import database as db
        with db.get_connection(for_read=True) as conn:
            try:
                rows = conn.execute(
                    "SELECT trader_address, total_pnl, win_rate, trade_count, "
                    "completed_at FROM backtest_results "
                    "ORDER BY completed_at DESC LIMIT 25"
                ).fetchall()
                recent_results = [dict(r) for r in rows]
            except Exception:
                recent_results = []
    except Exception:
        recent_results = []
    return {
        "running": running,
        "started_at": started,
        "elapsed_s": elapsed,
        "last_result": _LOCAL_JOB_RESULT,
        "last_error": _LOCAL_JOB_ERROR,
        "recent_results": recent_results,
    }


@router.get("/api/backtest/status", response_class=JSONResponse)
async def backtest_status(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(_status_payload())


@router.post("/api/backtest/run")
async def backtest_run(request: Request, max_wallets: int = Form(30)):
    if not verify_cookie(request):
        return JSONResponse({"error": "auth_required"}, status_code=401)

    global _LOCAL_JOB_RUNNING, _LOCAL_JOB_STARTED_AT, _LOCAL_JOB_RESULT, _LOCAL_JOB_ERROR

    with _LOCAL_JOB_LOCK:
        if _LOCAL_JOB_RUNNING:
            return JSONResponse(
                {"status": "running", "started_at": _LOCAL_JOB_STARTED_AT},
                status_code=409,
            )
        _LOCAL_JOB_RUNNING = True
        _LOCAL_JOB_STARTED_AT = time.time()
        _LOCAL_JOB_RESULT = None
        _LOCAL_JOB_ERROR = None

    capped = max(1, min(int(max_wallets or 30), 200))

    def _run() -> None:
        global _LOCAL_JOB_RUNNING, _LOCAL_JOB_STARTED_AT, _LOCAL_JOB_RESULT, _LOCAL_JOB_ERROR
        started = _LOCAL_JOB_STARTED_AT
        try:
            from src.discovery.golden_wallet import run_golden_scan, init_golden_tables
            from src.backtest.backtest_engine import (
                run_all_backtests, save_backtest_result, init_backtest_tables,
            )
            init_golden_tables()
            init_backtest_tables()
            golden = run_golden_scan(max_wallets=capped) or []
            results = run_all_backtests() or []
            for r in results:
                try:
                    save_backtest_result(r)
                except Exception as save_exc:
                    logger.debug("save_backtest_result failed: %s", save_exc)
            _LOCAL_JOB_RESULT = {
                "scanned_wallets": len(golden),
                "backtests_run": len(results),
                "duration_s": time.time() - (started or time.time()),
            }
            logger.info(
                "Dashboard backtest complete: %d wallets scanned, %d backtests",
                len(golden), len(results),
            )
        except Exception as exc:
            logger.error("Dashboard backtest failed: %s", exc, exc_info=True)
            _LOCAL_JOB_ERROR = str(exc)
        finally:
            with _LOCAL_JOB_LOCK:
                _LOCAL_JOB_RUNNING = False
                # Keep _LOCAL_JOB_STARTED_AT so the UI can show the last-run time.

    thread = threading.Thread(target=_run, name="dashboard-v2-backtest", daemon=True)
    thread.start()
    return JSONResponse({"status": "started", "started_at": _LOCAL_JOB_STARTED_AT, "max_wallets": capped})


@router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect
    from src.ui.v2.app import get_templates
    return get_templates().TemplateResponse(
        request,
        "backtest.html",
        {"title": "Backtest", "data": _status_payload()},
    )
