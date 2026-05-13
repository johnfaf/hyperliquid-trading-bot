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

from src.ui.replay_dashboard import (
    DEFAULT_REPLAY_COINS,
    get_replay_status,
    params_from_mapping,
    start_replay_validation,
)
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
_LOCAL_JOB_PROGRESS: Dict[str, Any] = {"phase": "idle"}
_LOCAL_JOB_MARKED_V1_RUNNING = False


def _v1_module():
    try:
        from src.ui import dashboard as v1
        return v1
    except Exception:
        return None


def _v1_state() -> Optional[Dict[str, Any]]:
    """Best-effort read of v1's job state. Returns None when v1 is absent."""
    v1 = _v1_module()
    if v1 is None:
        return None
    try:
        return {
            "running": bool(getattr(v1, "_BACKTEST_JOB_RUNNING", False)),
            "started_at": getattr(v1, "_BACKTEST_JOB_STARTED_AT", None),
        }
    except Exception:
        return None


def _shared_job_lock():
    v1 = _v1_module()
    return getattr(v1, "_BACKTEST_JOB_LOCK", _LOCAL_JOB_LOCK) if v1 is not None else _LOCAL_JOB_LOCK


def _set_progress(phase: str, **fields: Any) -> None:
    _LOCAL_JOB_PROGRESS.update({"phase": phase, "updated_at": time.time(), **fields})


def _coerce_max_wallets(value: Any, default: int = 30) -> tuple[int, int]:
    try:
        requested = int(value if value is not None else default)
    except (TypeError, ValueError):
        requested = default
    return requested, max(1, min(requested, 200))


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
                    """
                    SELECT address AS trader_address,
                           active_periods AS trade_count,
                           profitable_pct / 100.0 AS win_rate,
                           total_penalised_pnl AS total_pnl,
                           evaluated_at AS completed_at
                    FROM backtest_results
                    WHERE timeframe = '1d'
                    ORDER BY evaluated_at DESC
                    LIMIT 25
                    """
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
        "progress": dict(_LOCAL_JOB_PROGRESS),
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
    global _LOCAL_JOB_MARKED_V1_RUNNING

    requested, capped = _coerce_max_wallets(max_wallets)
    lock = _shared_job_lock()
    v1 = _v1_module()

    with lock:
        if _LOCAL_JOB_RUNNING or bool(getattr(v1, "_BACKTEST_JOB_RUNNING", False)):
            return JSONResponse(
                {
                    "status": "running",
                    "started_at": _LOCAL_JOB_STARTED_AT or getattr(v1, "_BACKTEST_JOB_STARTED_AT", None),
                    "progress": dict(_LOCAL_JOB_PROGRESS),
                },
                status_code=409,
            )
        _LOCAL_JOB_RUNNING = True
        _LOCAL_JOB_STARTED_AT = time.time()
        _LOCAL_JOB_RESULT = None
        _LOCAL_JOB_ERROR = None
        _set_progress(
            "queued",
            requested_max_wallets=requested,
            max_wallets=capped,
            started_at=_LOCAL_JOB_STARTED_AT,
        )
        _LOCAL_JOB_MARKED_V1_RUNNING = False
        if v1 is not None:
            try:
                setattr(v1, "_BACKTEST_JOB_RUNNING", True)
                setattr(v1, "_BACKTEST_JOB_STARTED_AT", _LOCAL_JOB_STARTED_AT)
                _LOCAL_JOB_MARKED_V1_RUNNING = True
            except Exception:
                _LOCAL_JOB_MARKED_V1_RUNNING = False

    def _run() -> None:
        global _LOCAL_JOB_RUNNING, _LOCAL_JOB_STARTED_AT, _LOCAL_JOB_RESULT, _LOCAL_JOB_ERROR
        global _LOCAL_JOB_MARKED_V1_RUNNING
        started = _LOCAL_JOB_STARTED_AT
        try:
            from src.discovery.golden_wallet import run_golden_scan, init_golden_tables
            from src.backtest.backtest_engine import (
                run_all_backtests, save_backtest_result, init_backtest_tables,
            )
            _set_progress("init_tables")
            init_golden_tables()
            init_backtest_tables()
            _set_progress("golden_scan", requested_max_wallets=requested, max_wallets=capped)
            golden_summary = run_golden_scan(max_wallets=capped) or {}
            if isinstance(golden_summary, dict):
                scanned_wallets = int(golden_summary.get("scanned") or 0)
                golden_wallets = int(golden_summary.get("golden") or 0)
                scan_error = golden_summary.get("error")
            else:
                scanned_wallets = len(golden_summary)
                golden_wallets = 0
                scan_error = None
            _set_progress(
                "backtest",
                requested_max_wallets=requested,
                max_wallets=capped,
                scanned_wallets=scanned_wallets,
                golden_wallets=golden_wallets,
                scan_error=scan_error,
            )
            results = run_all_backtests() or []
            _set_progress(
                "saving_results",
                requested_max_wallets=requested,
                max_wallets=capped,
                scanned_wallets=scanned_wallets,
                golden_wallets=golden_wallets,
                backtests_run=len(results),
            )
            saved = 0
            for r in results:
                try:
                    save_backtest_result(r)
                    saved += 1
                except Exception as save_exc:
                    logger.warning("save_backtest_result failed: %s", save_exc)
            _LOCAL_JOB_RESULT = {
                "requested_max_wallets": requested,
                "max_wallets": capped,
                "scanned_wallets": scanned_wallets,
                "golden_wallets": golden_wallets,
                "backtests_run": len(results),
                "results_saved": saved,
                "scan_error": scan_error,
                "duration_s": time.time() - (started or time.time()),
            }
            _set_progress("complete", **_LOCAL_JOB_RESULT)
            logger.info(
                "Dashboard backtest complete: %d wallets scanned, %d golden, %d backtests",
                scanned_wallets, golden_wallets, len(results),
            )
        except Exception as exc:
            logger.error("Dashboard backtest failed: %s", exc, exc_info=True)
            _LOCAL_JOB_ERROR = str(exc)
            _set_progress("error", error=str(exc))
        finally:
            with lock:
                _LOCAL_JOB_RUNNING = False
                if _LOCAL_JOB_MARKED_V1_RUNNING and v1 is not None:
                    try:
                        setattr(v1, "_BACKTEST_JOB_RUNNING", False)
                        setattr(v1, "_BACKTEST_JOB_STARTED_AT", None)
                    except Exception:
                        pass
                _LOCAL_JOB_MARKED_V1_RUNNING = False
                # Keep _LOCAL_JOB_STARTED_AT so the UI can show the last-run time.

    thread = threading.Thread(target=_run, name="dashboard-v2-backtest", daemon=True)
    thread.start()
    return JSONResponse({
        "status": "started",
        "started_at": _LOCAL_JOB_STARTED_AT,
        "requested_max_wallets": requested,
        "max_wallets": capped,
    })


@router.get("/api/replay/status", response_class=JSONResponse)
async def replay_status(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(get_replay_status())


@router.post("/api/replay/run")
async def replay_run(
    request: Request,
    coins: str = Form(DEFAULT_REPLAY_COINS),
    window_days: int = Form(3),
    min_rows: int = Form(1),
    step: str = Form("1h"),
    min_live_match_rate: float = Form(0.70),
    min_replay_match_rate: float = Form(0.70),
):
    if not verify_cookie(request):
        return JSONResponse({"error": "auth_required"}, status_code=401)

    params = params_from_mapping({
        "coins": coins,
        "window_days": window_days,
        "min_rows": min_rows,
        "step": step,
        "min_live_match_rate": min_live_match_rate,
        "min_replay_match_rate": min_replay_match_rate,
    })
    result = start_replay_validation(params)
    if result.get("error") == "replay_job_already_running":
        return JSONResponse(result, status_code=409)
    return JSONResponse(result)


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
