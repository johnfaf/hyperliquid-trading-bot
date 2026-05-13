"""Automated offline backtest and improvement loop.

The loop is deliberately operator-safe:
  * runs offline learning/backtests on persisted decisions and candle data
  * writes JSON reports under ``reports/auto_backtest``
  * records scheduler rows in the learning tables when available
  * never mutates live trading config, SL/TP, sizing, or kill-switch state

Promotion remains a manual/shadow-mode workflow handled by the existing
continuous-learning orchestrator.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.learning.orchestrator import ContinuousLearningOrchestrator

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = "reports/auto_backtest"
DEFAULT_COINS = "BTC,ETH,SOL,HYPE,XRP,DOGE,BNB,ADA,AVAX,LINK"
_STDIO_LIMIT = 8000
_CYCLE_REPORT_RE = re.compile(r"^auto_bt_\d{8}_\d{6}$")
_ACTIVE_STATUS: dict[str, Any] = {
    "running": False,
    "cycle_id": None,
    "started_at": None,
    "phase": "idle",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(value, hi))


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(value, hi))


def _truncate(value: str | None) -> str:
    text = str(value or "")
    if len(text) <= _STDIO_LIMIT:
        return text
    return text[-_STDIO_LIMIT:]


def _normalize_coins(value: str | None) -> str:
    raw = value or DEFAULT_COINS
    coins: list[str] = []
    for item in raw.split(","):
        coin = item.strip().upper()
        if coin and coin not in coins:
            coins.append(coin)
    return ",".join(coins or DEFAULT_COINS.split(","))


@dataclass(frozen=True)
class AutoBacktestConfig:
    enabled: bool = False
    interval_seconds: int = 6 * 60 * 60
    startup_delay_seconds: int = 5 * 60
    dataset_limit: int = 5000
    reports_dir: str = DEFAULT_REPORTS_DIR
    run_offline_learning: bool = True
    run_replay_validation: bool = True
    run_candle_research: bool = True
    coins: str = DEFAULT_COINS
    live_db: str = "data/bot.db"
    cache_dir: str = "data"
    cache_db: str = "data/candle_cache.db"
    timeframe: str = "1h"
    replay_window_days: int = 3
    replay_min_rows: int = 1
    replay_match_window_s: float = 600.0
    replay_min_live_match_rate: float = 0.70
    replay_min_replay_match_rate: float = 0.70
    replay_allow_network: bool = False
    replay_lax_api: bool = True
    candle_research_days: int = 90
    candle_fetch_missing: bool = False
    candle_min_candles: int = 50
    command_timeout_s: int = 60 * 60
    manual_approval: bool = False

    @classmethod
    def from_env(cls) -> "AutoBacktestConfig":
        return cls(
            enabled=_truthy(os.environ.get("AUTO_BACKTEST_LOOP_ENABLED"), default=False),
            interval_seconds=_env_int(
                "AUTO_BACKTEST_INTERVAL_SECONDS",
                6 * 60 * 60,
                lo=900,
                hi=7 * 86400,
            ),
            startup_delay_seconds=_env_int(
                "AUTO_BACKTEST_STARTUP_DELAY_SECONDS",
                5 * 60,
                lo=0,
                hi=24 * 3600,
            ),
            dataset_limit=_env_int("AUTO_BACKTEST_DATASET_LIMIT", 5000, lo=100, hi=100_000),
            reports_dir=os.environ.get("AUTO_BACKTEST_REPORTS_DIR", DEFAULT_REPORTS_DIR),
            run_offline_learning=_truthy(
                os.environ.get("AUTO_BACKTEST_RUN_OFFLINE_LEARNING"),
                default=True,
            ),
            run_replay_validation=_truthy(
                os.environ.get("AUTO_BACKTEST_RUN_REPLAY_VALIDATION"),
                default=True,
            ),
            run_candle_research=_truthy(
                os.environ.get("AUTO_BACKTEST_RUN_CANDLE_RESEARCH"),
                default=True,
            ),
            coins=_normalize_coins(os.environ.get("AUTO_BACKTEST_COINS")),
            live_db=os.environ.get("AUTO_BACKTEST_LIVE_DB", "data/bot.db"),
            cache_dir=os.environ.get("AUTO_BACKTEST_CACHE_DIR", "data"),
            cache_db=os.environ.get("AUTO_BACKTEST_CACHE_DB", "data/candle_cache.db"),
            timeframe=os.environ.get("AUTO_BACKTEST_TIMEFRAME", "1h").strip() or "1h",
            replay_window_days=_env_int("AUTO_BACKTEST_REPLAY_WINDOW_DAYS", 3, lo=1, hi=30),
            replay_min_rows=_env_int("AUTO_BACKTEST_REPLAY_MIN_ROWS", 1, lo=1, hi=100_000),
            replay_match_window_s=_env_float(
                "AUTO_BACKTEST_REPLAY_MATCH_WINDOW_S",
                600.0,
                lo=30.0,
                hi=86_400.0,
            ),
            replay_min_live_match_rate=_env_float(
                "AUTO_BACKTEST_REPLAY_MIN_LIVE_MATCH_RATE",
                0.70,
                lo=0.0,
                hi=1.0,
            ),
            replay_min_replay_match_rate=_env_float(
                "AUTO_BACKTEST_REPLAY_MIN_REPLAY_MATCH_RATE",
                0.70,
                lo=0.0,
                hi=1.0,
            ),
            replay_allow_network=_truthy(
                os.environ.get("AUTO_BACKTEST_REPLAY_ALLOW_NETWORK"),
                default=False,
            ),
            replay_lax_api=_truthy(os.environ.get("AUTO_BACKTEST_REPLAY_LAX_API"), default=True),
            candle_research_days=_env_int("AUTO_BACKTEST_CANDLE_RESEARCH_DAYS", 90, lo=7, hi=730),
            candle_fetch_missing=_truthy(
                os.environ.get("AUTO_BACKTEST_CANDLE_FETCH_MISSING"),
                default=False,
            ),
            candle_min_candles=_env_int("AUTO_BACKTEST_CANDLE_MIN_CANDLES", 50, lo=10, hi=10_000),
            command_timeout_s=_env_int(
                "AUTO_BACKTEST_COMMAND_TIMEOUT_SECONDS",
                60 * 60,
                lo=60,
                hi=12 * 60 * 60,
            ),
            # Kept false by default. Even True only records approval packages;
            # live deployment remains a separate operator step.
            manual_approval=_truthy(
                os.environ.get("AUTO_BACKTEST_MANUAL_APPROVAL"),
                default=False,
            ),
        )


@dataclass
class AutoBacktestCycleResult:
    cycle_id: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    next_actions: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutoBacktestLoop:
    """Runs one auto-backtest cycle or a forever loop for a service process."""

    def __init__(self, config: AutoBacktestConfig | None = None):
        self.config = config or AutoBacktestConfig.from_env()
        self.reports_dir = (ROOT / self.config.reports_dir).resolve()

    def _cycle_id(self) -> str:
        return f"auto_bt_{_stamp()}"

    def _report_path(self, cycle_id: str) -> Path:
        return self.reports_dir / f"{cycle_id}.json"

    def _write_report(self, result: AutoBacktestCycleResult) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self._report_path(result.cycle_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, sort_keys=True, default=str)
        return path

    def _run_command(
        self,
        *,
        step_name: str,
        cmd: list[str],
        report_path: Path,
    ) -> dict[str, Any]:
        started = time.time()
        logger.info("Auto-backtest step %s starting: %s", step_name, " ".join(cmd))
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_s,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            duration = time.time() - started
            return {
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "duration_s": round(duration, 3),
                "report_path": str(report_path),
                "command": cmd,
                "stdout_tail": _truncate(completed.stdout),
                "stderr_tail": _truncate(completed.stderr),
            }
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - started
            return {
                "status": "timeout",
                "returncode": None,
                "duration_s": round(duration, 3),
                "report_path": str(report_path),
                "command": cmd,
                "stdout_tail": _truncate(exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout),
                "stderr_tail": _truncate(exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr),
                "error": f"timeout after {self.config.command_timeout_s}s",
            }

    def _run_replay_validation(self, cycle_id: str) -> dict[str, Any]:
        live_db = ROOT / self.config.live_db
        if not live_db.exists():
            return {
                "status": "skipped",
                "reason": f"live sqlite db not found: {self.config.live_db}",
                "hint": "Set AUTO_BACKTEST_LIVE_DB to a readable SQLite bot DB for replay diff.",
            }
        report_path = self.reports_dir / f"{cycle_id}_replay_validation.json"
        diff_path = self.reports_dir / f"{cycle_id}_replay_validation_diff.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_replay_validation.py"),
            "--live-db",
            self.config.live_db,
            "--window-days",
            str(self.config.replay_window_days),
            "--min-rows",
            str(self.config.replay_min_rows),
            "--coins",
            self.config.coins,
            "--step",
            self.config.timeframe,
            "--cache-db",
            self.config.cache_db,
            "--match-window",
            str(self.config.replay_match_window_s),
            "--min-live-match-rate",
            str(self.config.replay_min_live_match_rate),
            "--min-replay-match-rate",
            str(self.config.replay_min_replay_match_rate),
            "--report-out",
            str(report_path),
            "--diff-report-out",
            str(diff_path),
        ]
        if self.config.replay_lax_api:
            cmd.append("--lax-api")
        if self.config.replay_allow_network:
            cmd.append("--allow-network")
        result = self._run_command(
            step_name="replay_validation",
            cmd=cmd,
            report_path=report_path,
        )
        result["diff_report_path"] = str(diff_path)
        return result

    def _run_candle_research(self, cycle_id: str) -> dict[str, Any]:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=self.config.candle_research_days)
        report_path = self.reports_dir / f"{cycle_id}_candle_research.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_candle_research.py"),
            "--cache-dir",
            self.config.cache_dir,
            "--timeframe",
            self.config.timeframe,
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
            "--coins",
            self.config.coins,
            "--min-candles",
            str(self.config.candle_min_candles),
            "--report-out",
            str(report_path),
        ]
        if self.config.candle_fetch_missing:
            cmd.append("--fetch-missing")
        return self._run_command(
            step_name="candle_research",
            cmd=cmd,
            report_path=report_path,
        )

    def _run_offline_learning(self) -> dict[str, Any]:
        started = time.time()
        try:
            result = ContinuousLearningOrchestrator().run_offline_cycle(
                limit=self.config.dataset_limit,
                manual_approval=self.config.manual_approval,
                persist=True,
            )
            payload = result.to_dict()
            return {
                "status": "completed" if result.status == "completed" else result.status,
                "duration_s": round(time.time() - started, 3),
                "result": payload,
                "dataset_id": result.dataset_id,
                "improvement_id": result.improvement_id,
                "package_id": result.package_id,
            }
        except Exception as exc:
            logger.error("Auto-backtest offline learning failed: %s", exc, exc_info=True)
            return {
                "status": "failed",
                "duration_s": round(time.time() - started, 3),
                "error": str(exc),
            }

    @staticmethod
    def _record_scheduler_result(result: AutoBacktestCycleResult) -> None:
        try:
            from src.data import database as db

            with db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO learning_scheduler_runs
                    (schedule_run_id, created_at, run_type, status, dataset_id,
                     improvement_id, package_id, started_at, finished_at,
                     metrics, errors, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(schedule_run_id) DO UPDATE SET
                        created_at = EXCLUDED.created_at,
                        status = EXCLUDED.status,
                        finished_at = EXCLUDED.finished_at,
                        metrics = EXCLUDED.metrics,
                        errors = EXCLUDED.errors,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        result.cycle_id,
                        _now(),
                        "auto_backtest_loop",
                        result.status,
                        result.metrics.get("dataset_id"),
                        result.metrics.get("improvement_id"),
                        result.metrics.get("package_id"),
                        result.started_at,
                        result.finished_at,
                        _json(result.metrics),
                        _json(result.errors),
                        _json({
                            "steps": result.steps,
                            "artifacts": result.artifacts,
                            "next_actions": result.next_actions,
                            "no_live_config_mutation": True,
                        }),
                    ),
                )
        except Exception as exc:
            logger.warning("Auto-backtest scheduler result not persisted: %s", exc)

    def _derive_next_actions(self, result: AutoBacktestCycleResult) -> list[str]:
        actions: list[str] = []
        learning = result.steps.get("offline_learning") or {}
        learning_result = (learning.get("result") or {}) if isinstance(learning, dict) else {}
        if learning.get("status") in {"blocked_data_quality", "blocked_no_candidate"}:
            actions.append(f"learning blocked: {learning.get('status')}")
        package_id = learning.get("package_id")
        if package_id:
            actions.append(f"review promotion package {package_id}")

        replay = result.steps.get("replay_validation") or {}
        if replay.get("status") == "skipped":
            actions.append("replay validation skipped; configure AUTO_BACKTEST_LIVE_DB if needed")
        elif replay.get("status") not in {None, "completed"}:
            actions.append("investigate replay validation failure before trusting research")

        candle = result.steps.get("candle_research") or {}
        if candle.get("status") not in {None, "completed"}:
            actions.append("refresh candle cache or enable AUTO_BACKTEST_CANDLE_FETCH_MISSING")

        if learning_result.get("status") == "completed" and not package_id:
            actions.append("collect more labelled decision outcomes before promotion")
        return actions

    def run_cycle(self) -> AutoBacktestCycleResult:
        cycle_id = self._cycle_id()
        result = AutoBacktestCycleResult(
            cycle_id=cycle_id,
            started_at=_now(),
            config={
                **asdict(self.config),
                "safety": "offline_only_no_live_config_mutation",
            },
        )
        logger.info("Auto-backtest cycle started: %s", cycle_id)
        _ACTIVE_STATUS.update({
            "running": True,
            "cycle_id": cycle_id,
            "started_at": result.started_at,
            "phase": "starting",
        })
        self._write_report(result)

        if self.config.run_offline_learning:
            _ACTIVE_STATUS.update({"phase": "offline_learning"})
            result.steps["offline_learning"] = self._run_offline_learning()
            if result.steps["offline_learning"].get("dataset_id"):
                result.metrics["dataset_id"] = result.steps["offline_learning"]["dataset_id"]
            if result.steps["offline_learning"].get("improvement_id"):
                result.metrics["improvement_id"] = result.steps["offline_learning"]["improvement_id"]
            if result.steps["offline_learning"].get("package_id"):
                result.metrics["package_id"] = result.steps["offline_learning"]["package_id"]

        if self.config.run_replay_validation:
            _ACTIVE_STATUS.update({"phase": "replay_validation"})
            result.steps["replay_validation"] = self._run_replay_validation(cycle_id)

        if self.config.run_candle_research:
            _ACTIVE_STATUS.update({"phase": "candle_research"})
            result.steps["candle_research"] = self._run_candle_research(cycle_id)

        failed_steps = [
            name for name, step in result.steps.items()
            if step.get("status") in {"failed", "timeout"}
        ]
        skipped_steps = [
            name for name, step in result.steps.items()
            if step.get("status") == "skipped"
        ]
        result.finished_at = _now()
        result.status = "failed" if failed_steps else ("completed_with_skips" if skipped_steps else "completed")
        result.errors = [
            f"{name}: {result.steps[name].get('error') or result.steps[name].get('stderr_tail') or result.steps[name].get('reason')}"
            for name in failed_steps
        ]
        result.artifacts = {
            name: {
                key: step.get(key)
                for key in ("report_path", "diff_report_path")
                if step.get(key)
            }
            for name, step in result.steps.items()
        }
        result.next_actions = self._derive_next_actions(result)
        report_path = self._write_report(result)
        result.artifacts["cycle_report"] = str(report_path)
        self._write_report(result)
        self._record_scheduler_result(result)
        _ACTIVE_STATUS.update({
            "running": False,
            "cycle_id": cycle_id,
            "started_at": result.started_at,
            "phase": result.status,
            "finished_at": result.finished_at,
        })
        logger.info(
            "Auto-backtest cycle %s finished with status=%s report=%s",
            cycle_id,
            result.status,
            report_path,
        )
        return result

    def run_forever(self) -> None:
        if self.config.startup_delay_seconds > 0:
            logger.info(
                "Auto-backtest loop startup delay: %ss",
                self.config.startup_delay_seconds,
            )
            time.sleep(self.config.startup_delay_seconds)
        while True:
            self.run_cycle()
            logger.info(
                "Auto-backtest loop sleeping for %ss",
                self.config.interval_seconds,
            )
            time.sleep(self.config.interval_seconds)


def latest_auto_backtest_reports(
    reports_dir: str = DEFAULT_REPORTS_DIR,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    path = (ROOT / reports_dir).resolve()
    if not path.exists():
        return []
    reports = sorted(
        (p for p in path.glob("auto_bt_*.json") if _CYCLE_REPORT_RE.match(p.stem)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for report in reports[: max(1, min(limit, 25))]:
        try:
            with report.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["report_path"] = str(report)
                out.append(data)
        except Exception as exc:
            logger.debug("Could not read auto-backtest report %s: %s", report, exc)
    return out


def get_auto_backtest_status(
    config: AutoBacktestConfig | None = None,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    cfg = config or AutoBacktestConfig.from_env()
    return {
        "enabled": bool(cfg.enabled),
        "interval_seconds": cfg.interval_seconds,
        "startup_delay_seconds": cfg.startup_delay_seconds,
        "running": bool(_ACTIVE_STATUS.get("running")),
        "active": dict(_ACTIVE_STATUS),
        "recent_results": latest_auto_backtest_reports(cfg.reports_dir, limit=limit),
        "safety": "offline_only_no_live_config_mutation",
    }
