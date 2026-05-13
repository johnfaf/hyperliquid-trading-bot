"""Dashboard-facing replay validation job runner.

This module keeps the dashboard thin: it launches the existing
``scripts/run_replay_validation.py`` command in a background thread, reads the
JSON reports it produces, and exposes a compact status payload for both the
legacy and v2 dashboards.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"

DEFAULT_REPLAY_COINS = "BTC,ETH,SOL,HYPE,XRP,DOGE,BNB,ADA,AVAX,LINK"
DEFAULT_LIVE_DB = "data/bot.db"
DEFAULT_CACHE_DB = "data/candle_cache.db"
DEFAULT_WINDOW_DAYS = 3
DEFAULT_MIN_ROWS = 1
DEFAULT_MATCH_WINDOW_S = 600.0
DEFAULT_MIN_MATCH_RATE = 0.70
DEFAULT_TIMEOUT_S = 60 * 60
_STDIO_LIMIT = 8000

_JOB_LOCK = threading.Lock()
_JOB_RUNNING = False
_JOB_STARTED_AT: float | None = None
_JOB_RESULT: dict[str, Any] | None = None
_JOB_ERROR: str | None = None
_JOB_PROGRESS: dict[str, Any] = {"phase": "idle"}


@dataclass(frozen=True)
class ReplayRunParams:
    live_db: str = DEFAULT_LIVE_DB
    coins: str = DEFAULT_REPLAY_COINS
    step: str = "1h"
    cache_db: str = DEFAULT_CACHE_DB
    window_days: int = DEFAULT_WINDOW_DAYS
    min_rows: int = DEFAULT_MIN_ROWS
    match_window_s: float = DEFAULT_MATCH_WINDOW_S
    min_live_match_rate: float = DEFAULT_MIN_MATCH_RATE
    min_replay_match_rate: float = DEFAULT_MIN_MATCH_RATE
    strategy_snapshot: str | None = None
    frozen_xgb_model: str | None = None
    allow_network: bool = False
    lax_api: bool = True
    halt_on_error: bool = False
    timeout_s: int = DEFAULT_TIMEOUT_S


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truncate_stdio(value: str | None) -> str:
    text = str(value or "")
    if len(text) <= _STDIO_LIMIT:
        return text
    return text[-_STDIO_LIMIT:]


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(parsed, hi))


def _coerce_float(value: Any, default: float, *, lo: float, hi: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(parsed, hi))


def _clean_path(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_coins(value: Any) -> str:
    raw = str(value or DEFAULT_REPLAY_COINS)
    coins: list[str] = []
    for item in raw.split(","):
        coin = item.strip().upper()
        if coin and coin not in coins:
            coins.append(coin)
    return ",".join(coins or DEFAULT_REPLAY_COINS.split(","))


def params_from_mapping(values: dict[str, Any] | None) -> ReplayRunParams:
    """Build bounded replay parameters from dashboard form/JSON input."""
    values = values or {}
    return ReplayRunParams(
        live_db=str(values.get("live_db") or DEFAULT_LIVE_DB).strip() or DEFAULT_LIVE_DB,
        coins=_normalize_coins(values.get("coins")),
        step=str(values.get("step") or "1h").strip() or "1h",
        cache_db=str(values.get("cache_db") or DEFAULT_CACHE_DB).strip() or DEFAULT_CACHE_DB,
        window_days=_coerce_int(
            values.get("window_days"),
            DEFAULT_WINDOW_DAYS,
            lo=1,
            hi=30,
        ),
        min_rows=_coerce_int(values.get("min_rows"), DEFAULT_MIN_ROWS, lo=1, hi=100_000),
        match_window_s=_coerce_float(
            values.get("match_window_s"),
            DEFAULT_MATCH_WINDOW_S,
            lo=30.0,
            hi=86_400.0,
        ),
        min_live_match_rate=_coerce_float(
            values.get("min_live_match_rate"),
            DEFAULT_MIN_MATCH_RATE,
            lo=0.0,
            hi=1.0,
        ),
        min_replay_match_rate=_coerce_float(
            values.get("min_replay_match_rate"),
            DEFAULT_MIN_MATCH_RATE,
            lo=0.0,
            hi=1.0,
        ),
        strategy_snapshot=_clean_path(values.get("strategy_snapshot")),
        frozen_xgb_model=_clean_path(values.get("frozen_xgb_model")),
        allow_network=_coerce_bool(values.get("allow_network"), default=False),
        lax_api=_coerce_bool(values.get("lax_api"), default=True),
        halt_on_error=_coerce_bool(values.get("halt_on_error"), default=False),
        timeout_s=_coerce_int(values.get("timeout_s"), DEFAULT_TIMEOUT_S, lo=60, hi=6 * 60 * 60),
    )


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Failed to read replay dashboard report %s: %s", path, exc)
        return None


def _extract_summary(
    *,
    report_path: Path | None,
    diff_path: Path | None,
    returncode: int | None = None,
    started_at: float | None = None,
    completed_at: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    report = _load_json(report_path) or {}
    diff = _load_json(diff_path) or report.get("decision_diff") or {}
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    diagnostics = diff.get("diagnostics") if isinstance(diff.get("diagnostics"), dict) else {}
    totals = diff.get("totals") if isinstance(diff.get("totals"), dict) else {}
    outputs = report.get("outputs") if isinstance(report.get("outputs"), dict) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), dict) else {}
    api_activity = report.get("api_activity") if isinstance(report.get("api_activity"), dict) else {}
    selected_window = report.get("selected_window")

    if not selected_window and isinstance(config, dict):
        selected_window = {
            "start_iso": config.get("start"),
            "end_iso": config.get("end"),
        }

    return {
        "status": diagnostics.get("status") or ("complete" if returncode == 0 else "failed"),
        "trustworthy": bool(diagnostics.get("trustworthy", False)),
        "returncode": returncode,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": (
            round(completed_at - started_at, 3)
            if started_at is not None and completed_at is not None
            else None
        ),
        "run_id": config.get("run_id"),
        "window": selected_window,
        "coins": config.get("coins") or [],
        "step": config.get("step"),
        "replay_db_path": report.get("replay_db_path"),
        "report_path": str(report_path) if report_path else None,
        "diff_report_path": str(diff_path) if diff_path else None,
        "totals": totals,
        "diagnostics": diagnostics,
        "outputs": outputs,
        "execution": execution,
        "api_activity": api_activity,
        "top_live_only_reasons": list(
            (diff.get("reject_reasons") or {}).get("live_only", {}).items()
        )[:8],
        "top_replay_only_reasons": list(
            (diff.get("reject_reasons") or {}).get("replay_only", {}).items()
        )[:8],
        "sample_mismatches": (diff.get("sample_mismatches") or [])[:10],
        "stdout_tail": _truncate_stdio(stdout),
        "stderr_tail": _truncate_stdio(stderr),
    }


def list_recent_replay_results(limit: int = 5) -> list[dict[str, Any]]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (
            p for p in REPORTS_DIR.glob("dashboard_replay_validation_*.json")
            if not p.name.startswith("dashboard_replay_validation_diff_")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for report_path in paths[: max(1, min(limit, 25))]:
        suffix = report_path.name.removeprefix("dashboard_replay_validation_")
        diff_path = report_path.with_name(f"dashboard_replay_validation_diff_{suffix}")
        summary = _extract_summary(report_path=report_path, diff_path=diff_path)
        summary["completed_at"] = summary.get("completed_at") or report_path.stat().st_mtime
        results.append(summary)
    return results


def get_replay_status() -> dict[str, Any]:
    with _JOB_LOCK:
        running = _JOB_RUNNING
        started = _JOB_STARTED_AT
        result = dict(_JOB_RESULT) if isinstance(_JOB_RESULT, dict) else None
        error = _JOB_ERROR
        progress = dict(_JOB_PROGRESS)
    elapsed = time.time() - started if running and started else None
    return {
        "running": running,
        "started_at": started,
        "elapsed_s": elapsed,
        "last_result": result,
        "last_error": error,
        "progress": progress,
        "recent_results": list_recent_replay_results(limit=5),
    }


def _build_command(params: ReplayRunParams, report_path: Path, diff_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_replay_validation.py"),
        "--live-db",
        params.live_db,
        "--window-days",
        str(params.window_days),
        "--min-rows",
        str(params.min_rows),
        "--coins",
        params.coins,
        "--step",
        params.step,
        "--cache-db",
        params.cache_db,
        "--match-window",
        str(params.match_window_s),
        "--min-live-match-rate",
        str(params.min_live_match_rate),
        "--min-replay-match-rate",
        str(params.min_replay_match_rate),
        "--report-out",
        str(report_path),
        "--diff-report-out",
        str(diff_path),
    ]
    if params.strategy_snapshot:
        cmd.extend(["--strategy-snapshot", params.strategy_snapshot])
    if params.frozen_xgb_model:
        cmd.extend(["--frozen-xgb-model", params.frozen_xgb_model])
    if params.allow_network:
        cmd.append("--allow-network")
    if params.lax_api:
        cmd.append("--lax-api")
    if params.halt_on_error:
        cmd.append("--halt-on-error")
    return cmd


def _run_replay_validation(params: ReplayRunParams, started_at: float) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _now_ms()
    report_path = REPORTS_DIR / f"dashboard_replay_validation_{stamp}.json"
    diff_path = REPORTS_DIR / f"dashboard_replay_validation_diff_{stamp}.json"
    cmd = _build_command(params, report_path, diff_path)

    logger.info("Dashboard replay validation starting: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=params.timeout_s,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    completed_at = time.time()
    summary = _extract_summary(
        report_path=report_path,
        diff_path=diff_path,
        returncode=completed.returncode,
        started_at=started_at,
        completed_at=completed_at,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    summary["command"] = cmd
    summary["ok"] = completed.returncode == 0
    return summary


def start_replay_validation(params: ReplayRunParams | None = None) -> dict[str, Any]:
    """Start a background replay validation job if one is not already running."""
    params = params or ReplayRunParams()
    global _JOB_RUNNING, _JOB_STARTED_AT, _JOB_RESULT, _JOB_ERROR, _JOB_PROGRESS
    with _JOB_LOCK:
        if _JOB_RUNNING:
            return {
                "status": "running",
                "error": "replay_job_already_running",
                "started_at": _JOB_STARTED_AT,
                "progress": dict(_JOB_PROGRESS),
            }
        _JOB_RUNNING = True
        _JOB_STARTED_AT = time.time()
        _JOB_RESULT = None
        _JOB_ERROR = None
        _JOB_PROGRESS = {
            "phase": "queued",
            "started_at": _JOB_STARTED_AT,
            "coins": params.coins,
            "window_days": params.window_days,
            "step": params.step,
        }
        started_at = _JOB_STARTED_AT

    def _thread_main() -> None:
        global _JOB_RUNNING, _JOB_RESULT, _JOB_ERROR, _JOB_PROGRESS
        try:
            with _JOB_LOCK:
                _JOB_PROGRESS = {**_JOB_PROGRESS, "phase": "running", "updated_at": time.time()}
            result = _run_replay_validation(params, started_at or time.time())
            with _JOB_LOCK:
                _JOB_RESULT = result
                _JOB_ERROR = None if result.get("ok") else (
                    result.get("stderr_tail") or "replay validation failed"
                )
                _JOB_PROGRESS = {
                    **_JOB_PROGRESS,
                    "phase": "complete" if result.get("ok") else "failed",
                    "updated_at": time.time(),
                    "status": result.get("status"),
                    "trustworthy": result.get("trustworthy"),
                    "returncode": result.get("returncode"),
                }
        except subprocess.TimeoutExpired as exc:
            logger.error("Dashboard replay validation timed out: %s", exc)
            with _JOB_LOCK:
                _JOB_ERROR = f"timeout after {params.timeout_s}s"
                _JOB_PROGRESS = {**_JOB_PROGRESS, "phase": "timeout", "updated_at": time.time()}
        except Exception as exc:
            logger.error("Dashboard replay validation failed: %s", exc, exc_info=True)
            with _JOB_LOCK:
                _JOB_ERROR = str(exc)
                _JOB_PROGRESS = {**_JOB_PROGRESS, "phase": "error", "updated_at": time.time()}
        finally:
            with _JOB_LOCK:
                _JOB_RUNNING = False

    thread = threading.Thread(
        target=_thread_main,
        name="dashboard-replay-validation",
        daemon=True,
    )
    thread.start()
    return {
        "status": "started",
        "started_at": started_at,
        "params": params.__dict__,
    }


def reset_replay_dashboard_state_for_tests() -> None:
    global _JOB_RUNNING, _JOB_STARTED_AT, _JOB_RESULT, _JOB_ERROR, _JOB_PROGRESS
    with _JOB_LOCK:
        _JOB_RUNNING = False
        _JOB_STARTED_AT = None
        _JOB_RESULT = None
        _JOB_ERROR = None
        _JOB_PROGRESS = {"phase": "idle"}
