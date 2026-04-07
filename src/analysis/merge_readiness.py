"""Final merge-readiness packaging for operator go/no-go review."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config
from src.analysis.experiment_discipline import run_experiment_benchmark_pack
from src.core.time_utils import utc_now_iso, utc_now_naive
from src.data import database as db

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_STABLE_REGRESSION_CMD = [
    sys.executable,
    "-m",
    "pytest",
    "tests/test_decision_firewall.py",
    "tests/test_live_controls.py",
    "tests/test_options_flow.py",
    "tests/test_portfolio_rotation.py",
    "-q",
]


def build_merge_readiness_config(cfg=config) -> Dict:
    return {
        "enabled": bool(getattr(cfg, "MERGE_READINESS_ENABLED", True)),
        "interval_hours": float(getattr(cfg, "MERGE_READINESS_INTERVAL_HOURS", 24.0)),
        "require_daily_research_clear": bool(
            getattr(cfg, "MERGE_READINESS_REQUIRE_DAILY_RESEARCH_CLEAR", True)
        ),
        "require_shadow_certified": bool(
            getattr(cfg, "MERGE_READINESS_REQUIRE_SHADOW_CERTIFIED", True)
        ),
        "require_capital_ramp_deployable": bool(
            getattr(cfg, "MERGE_READINESS_REQUIRE_CAPITAL_RAMP_DEPLOYABLE", True)
        ),
        "require_benchmark_clear": bool(
            getattr(cfg, "MERGE_READINESS_REQUIRE_BENCHMARK_CLEAR", True)
        ),
        "require_live_certified": bool(
            getattr(cfg, "MERGE_READINESS_REQUIRE_LIVE_CERTIFIED", True)
        ),
        "require_regression_suite": bool(
            getattr(cfg, "MERGE_READINESS_REQUIRE_REGRESSION_SUITE", False)
        ),
        "require_clean_worktree": bool(
            getattr(cfg, "MERGE_READINESS_REQUIRE_CLEAN_WORKTREE", False)
        ),
        "benchmark_limit_cycles": int(
            getattr(cfg, "MERGE_READINESS_BENCHMARK_LIMIT_CYCLES", 120)
        ),
        "benchmark_oos_ratio": float(
            getattr(cfg, "MERGE_READINESS_BENCHMARK_OOS_RATIO", 0.30)
        ),
        "report_path": os.path.abspath(
            getattr(cfg, "MERGE_READINESS_REPORT_PATH", "reports/merge_readiness_report.json")
        ),
    }


def _write_json(path: str, payload: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _tail_lines(text: str, limit: int = 10) -> List[str]:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-limit:]


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "go", "certified", "ready"}


def _git_command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git command failed")
    return str(result.stdout or "").strip()


def _get_git_summary() -> Dict:
    summary = {
        "branch": "",
        "commit": "",
        "short_commit": "",
        "dirty": False,
    }
    try:
        summary["branch"] = _git_command("rev-parse", "--abbrev-ref", "HEAD")
        summary["commit"] = _git_command("rev-parse", "HEAD")
        summary["short_commit"] = summary["commit"][:7]
        summary["dirty"] = bool(_git_command("status", "--porcelain"))
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _summarize_benchmark(benchmark: Dict) -> Dict:
    if not benchmark:
        return {}
    gate = (benchmark.get("promotion_gate", {}) or {})
    baseline = str(gate.get("baseline", "baseline_current") or "baseline_current")
    winner = str(gate.get("winner", baseline) or baseline)
    approved = list(gate.get("approved_profiles", []) or [])
    profiles = benchmark.get("profiles", {}) or {}
    winner_profile = (profiles.get(winner, {}) or {})
    winner_oos = (winner_profile.get("out_of_sample", {}) or {})
    winner_promotion = (winner_profile.get("promotion", {}) or {})
    clear = bool(winner == baseline or winner in approved)
    return {
        "baseline_profile": baseline,
        "winner_profile": winner,
        "approved_profiles": approved,
        "clear": clear,
        "avg_selected_ev_pct": round(float(winner_oos.get("avg_selected_ev_pct", 0.0) or 0.0), 4),
        "avg_selected_execution_cost_pct": round(
            float(winner_oos.get("avg_selected_execution_cost_pct", 0.0) or 0.0),
            4,
        ),
        "no_trade_rate": round(float(winner_oos.get("no_trade_rate", 0.0) or 0.0), 4),
        "winner_ev_delta_pct": round(float(winner_promotion.get("ev_delta_pct", 0.0) or 0.0), 4),
        "winner_execution_cost_delta_pct": round(
            float(winner_promotion.get("execution_cost_delta_pct", 0.0) or 0.0),
            4,
        ),
        "winner_no_trade_delta": round(
            float(winner_promotion.get("no_trade_delta", 0.0) or 0.0),
            4,
        ),
    }


def _benchmark_from_daily_research(daily_research: Dict) -> Dict:
    metadata = (daily_research.get("metadata", {}) or {}) if isinstance(daily_research, dict) else {}
    benchmark = metadata.get("benchmark", {}) if isinstance(metadata, dict) else {}
    return _summarize_benchmark(benchmark if isinstance(benchmark, dict) else {})


def _run_regression_suite(command: Optional[List[str]] = None) -> Dict:
    cmd = list(command or _STABLE_REGRESSION_CMD)
    try:
        completed = subprocess.run(
            cmd,
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "executed": True,
            "passed": False,
            "timed_out": True,
            "returncode": None,
            "command": " ".join(cmd),
            "summary": "Regression suite timed out.",
            "stdout_tail": _tail_lines(exc.stdout or ""),
            "stderr_tail": _tail_lines(exc.stderr or ""),
        }

    stdout_tail = _tail_lines(completed.stdout)
    stderr_tail = _tail_lines(completed.stderr)
    summary = "Regression suite passed." if completed.returncode == 0 else "Regression suite failed."
    if stdout_tail:
        summary = stdout_tail[-1]
    elif stderr_tail:
        summary = stderr_tail[-1]

    return {
        "executed": True,
        "passed": completed.returncode == 0,
        "timed_out": False,
        "returncode": completed.returncode,
        "command": " ".join(cmd),
        "summary": summary,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


class MergeReadinessManager:
    """Build and persist the final merge go/no-go package."""

    def __init__(self, cfg: Dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.interval_hours = float(cfg.get("interval_hours", 24.0))
        self.require_daily_research_clear = bool(cfg.get("require_daily_research_clear", True))
        self.require_shadow_certified = bool(cfg.get("require_shadow_certified", True))
        self.require_capital_ramp_deployable = bool(
            cfg.get("require_capital_ramp_deployable", True)
        )
        self.require_benchmark_clear = bool(cfg.get("require_benchmark_clear", True))
        self.require_live_certified = bool(cfg.get("require_live_certified", True))
        self.require_regression_suite = bool(cfg.get("require_regression_suite", False))
        self.require_clean_worktree = bool(cfg.get("require_clean_worktree", False))
        self.benchmark_limit_cycles = int(cfg.get("benchmark_limit_cycles", 120))
        self.benchmark_oos_ratio = float(cfg.get("benchmark_oos_ratio", 0.30))
        self.report_path = str(cfg.get("report_path", "") or "").strip()
        self.last_result: Dict = {}
        self.last_run_at: Optional[str] = None
        self.last_run_id: Optional[str] = None

    def _is_due(self, now: Optional[datetime] = None) -> bool:
        if not self.enabled:
            return False
        reference = _parse_timestamp(self.last_run_at)
        if reference is None:
            latest = db.get_latest_merge_readiness_run()
            if latest:
                self.last_result = latest
                self.last_run_at = latest.get("timestamp")
                self.last_run_id = latest.get("run_id")
                reference = _parse_timestamp(self.last_run_at)
        if reference is None:
            return True
        now = now or utc_now_naive()
        return now - reference >= timedelta(hours=max(self.interval_hours, 0.0))

    def _collect_live_readiness(self, *, live_trader=None, force_preflight: bool = False) -> Dict:
        if live_trader is None:
            from src.signals.decision_firewall import DecisionFirewall
            from src.trading.live_trader import LiveTrader

            firewall = DecisionFirewall(
                {
                    "funding_risk_enabled": False,
                    "enable_predictive_derisk": False,
                    "min_confidence": getattr(config, "FIREWALL_MIN_CONFIDENCE", 0.45),
                }
            )
            live_trader = LiveTrader(
                firewall=firewall,
                dry_run=not getattr(config, "LIVE_TRADING_ENABLED", False),
                max_daily_loss=float(getattr(config, "LIVE_MAX_DAILY_LOSS_USD", 500)),
                max_order_usd=float(getattr(config, "LIVE_MAX_ORDER_USD", 12.0)),
            )

        try:
            preflight = live_trader.run_preflight(force=force_preflight)
            activation = live_trader.evaluate_activation_guard()
            readiness = live_trader.get_live_readiness(force_preflight=False)
        except Exception as exc:
            return {
                "preflight": {
                    "status": "error",
                    "deployable": False,
                    "blocking_checks": ["preflight_exception"],
                    "warning_checks": [],
                },
                "activation_guard": {
                    "status": "error",
                    "deployable": False,
                    "blocking_checks": ["activation_exception"],
                    "warning_checks": [],
                },
                "live_readiness": {
                    "status": "blocked",
                    "deployable": False,
                    "status_reason": "merge_readiness_live_check_failed",
                    "blocking_checks": ["merge_readiness_live_check_failed"],
                    "warning_checks": [],
                },
                "certified_for_live_entries": False,
                "error": str(exc),
            }

        return {
            "preflight": preflight,
            "activation_guard": activation,
            "live_readiness": readiness,
            "certified_for_live_entries": bool(readiness.get("deployable", False)),
        }

    def _collect_regression_suite(self, *, run_regression_suite: bool) -> Dict:
        if not run_regression_suite:
            return {
                "executed": False,
                "passed": False,
                "timed_out": False,
                "returncode": None,
                "command": " ".join(_STABLE_REGRESSION_CMD),
                "summary": "Regression suite not run.",
            }
        return _run_regression_suite()

    def _collect_benchmark_summary(self, *, daily_research: Dict, run_benchmark_pack: bool) -> Dict:
        if run_benchmark_pack:
            benchmark = run_experiment_benchmark_pack(
                limit_cycles=self.benchmark_limit_cycles,
                out_of_sample_ratio=self.benchmark_oos_ratio,
            )
            return _summarize_benchmark(benchmark)
        return _benchmark_from_daily_research(daily_research)

    def _build_checks(
        self,
        *,
        strict: bool,
        daily_research: Dict,
        shadow_certification: Dict,
        capital_ramp: Dict,
        benchmark_summary: Dict,
        live_certification: Dict,
        regression_suite: Dict,
        git_summary: Dict,
    ) -> List[Dict]:
        checks = []

        def add_check(
            name: str,
            *,
            passed: bool,
            required: bool,
            value,
            threshold,
            warming_up: bool = False,
        ) -> None:
            checks.append(
                {
                    "name": name,
                    "passed": bool(passed),
                    "required": bool(required),
                    "warming_up": bool(warming_up),
                    "value": value,
                    "threshold": threshold,
                }
            )

        daily_recommendation = str(daily_research.get("recommendation", "") or "").strip().lower()
        add_check(
            "daily_research_clear",
            passed=daily_recommendation in {"hold", "promote"},
            required=self.require_daily_research_clear,
            warming_up=not bool(daily_research),
            value=daily_recommendation or "missing",
            threshold="hold_or_promote",
        )
        add_check(
            "shadow_certified",
            passed=_coerce_bool(shadow_certification.get("certified", False)),
            required=self.require_shadow_certified,
            warming_up=not bool(shadow_certification),
            value=_coerce_bool(shadow_certification.get("certified", False)),
            threshold=True,
        )
        add_check(
            "capital_ramp_deployable",
            passed=_coerce_bool(capital_ramp.get("deployable", False)),
            required=self.require_capital_ramp_deployable,
            warming_up=not bool(capital_ramp),
            value=_coerce_bool(capital_ramp.get("deployable", False)),
            threshold=True,
        )
        add_check(
            "benchmark_clear",
            passed=_coerce_bool(benchmark_summary.get("clear", False)),
            required=self.require_benchmark_clear,
            warming_up=not bool(benchmark_summary),
            value=benchmark_summary.get("winner_profile", "missing"),
            threshold="baseline_or_approved_challenger",
        )
        live_readiness = (live_certification.get("live_readiness", {}) or {})
        add_check(
            "live_certified",
            passed=_coerce_bool(live_certification.get("certified_for_live_entries", False)),
            required=self.require_live_certified,
            warming_up=not bool(live_certification),
            value=live_readiness.get("status", "missing"),
            threshold="ready",
        )
        regression_required = strict and self.require_regression_suite
        add_check(
            "regression_suite_passed",
            passed=bool(regression_suite.get("passed", False)),
            required=regression_required,
            warming_up=regression_required and not regression_suite.get("executed", False),
            value=regression_suite.get("summary", "not_run"),
            threshold="stable_suite_pass",
        )
        add_check(
            "git_worktree_clean",
            passed=not bool(git_summary.get("dirty", False)),
            required=self.require_clean_worktree,
            warming_up=not bool(git_summary),
            value="dirty" if git_summary.get("dirty", False) else "clean",
            threshold="clean",
        )
        return checks

    @staticmethod
    def _summarize_dependencies(
        daily_research: Dict,
        shadow_certification: Dict,
        capital_ramp: Dict,
    ) -> Dict:
        return {
            "daily_research": {
                "run_id": daily_research.get("run_id"),
                "recommendation": daily_research.get("recommendation"),
                "winner_profile": daily_research.get("winner_profile"),
                "timestamp": daily_research.get("timestamp"),
            },
            "shadow_certification": {
                "run_id": shadow_certification.get("run_id"),
                "status": shadow_certification.get("status"),
                "certified": shadow_certification.get("certified"),
                "timestamp": shadow_certification.get("timestamp"),
            },
            "capital_ramp": {
                "run_id": capital_ramp.get("run_id"),
                "status": capital_ramp.get("status"),
                "applied_stage": capital_ramp.get("applied_stage"),
                "approved_stage": capital_ramp.get("approved_stage"),
                "deployable": capital_ramp.get("deployable"),
                "timestamp": capital_ramp.get("timestamp"),
            },
        }

    @staticmethod
    def _build_status(checks: List[Dict]) -> Dict:
        failed_required = [
            item["name"]
            for item in checks
            if item.get("required", False)
            and not item.get("passed", False)
            and not item.get("warming_up", False)
        ]
        warming_required = [
            item["name"]
            for item in checks
            if item.get("required", False) and item.get("warming_up", False)
        ]
        optional_failures = [
            item["name"]
            for item in checks
            if not item.get("required", False) and not item.get("passed", False)
        ]

        if failed_required:
            return {
                "status": "no_go",
                "deployable_for_merge": False,
                "summary": "Merge readiness blocked by: " + ", ".join(failed_required),
                "failed_required_checks": failed_required,
                "warming_required_checks": warming_required,
                "optional_failures": optional_failures,
            }
        if warming_required:
            return {
                "status": "hold",
                "deployable_for_merge": False,
                "summary": "Merge readiness waiting on: " + ", ".join(warming_required),
                "failed_required_checks": failed_required,
                "warming_required_checks": warming_required,
                "optional_failures": optional_failures,
            }
        if optional_failures:
            return {
                "status": "go",
                "deployable_for_merge": True,
                "summary": "Merge readiness clear with non-blocking warnings: " + ", ".join(optional_failures),
                "failed_required_checks": failed_required,
                "warming_required_checks": warming_required,
                "optional_failures": optional_failures,
            }
        return {
            "status": "go",
            "deployable_for_merge": True,
            "summary": "Merge readiness clear for merge.",
            "failed_required_checks": failed_required,
            "warming_required_checks": warming_required,
            "optional_failures": optional_failures,
        }

    def run(
        self,
        *,
        cycle_count: Optional[int] = None,
        force: bool = False,
        strict: bool = True,
        run_regression_suite: bool = False,
        run_benchmark_pack: bool = False,
        force_live_preflight: bool = False,
        live_trader=None,
    ) -> Dict:
        if not self.enabled:
            payload = {
                "run_id": None,
                "timestamp": utc_now_iso(),
                "status": "disabled",
                "deployable_for_merge": False,
                "summary": "Merge readiness disabled.",
                "checks": [],
                "git": _get_git_summary(),
                "dependencies": {},
            }
            self.last_result = payload
            return payload

        now = utc_now_naive()
        if not force and not self._is_due(now):
            latest = db.get_latest_merge_readiness_run()
            if latest:
                self.last_result = latest
                self.last_run_at = latest.get("timestamp")
                self.last_run_id = latest.get("run_id")
                return latest

        timestamp = utc_now_iso()
        daily_research = db.get_latest_daily_research_run()
        shadow_certification = db.get_latest_shadow_certification_run()
        capital_ramp = db.get_latest_capital_ramp_run()
        benchmark_summary = self._collect_benchmark_summary(
            daily_research=daily_research,
            run_benchmark_pack=run_benchmark_pack,
        )
        live_certification = self._collect_live_readiness(
            live_trader=live_trader,
            force_preflight=force_live_preflight,
        )
        regression_suite = self._collect_regression_suite(
            run_regression_suite=run_regression_suite,
        )
        git_summary = _get_git_summary()
        checks = self._build_checks(
            strict=strict,
            daily_research=daily_research,
            shadow_certification=shadow_certification,
            capital_ramp=capital_ramp,
            benchmark_summary=benchmark_summary,
            live_certification=live_certification,
            regression_suite=regression_suite,
            git_summary=git_summary,
        )
        status_payload = self._build_status(checks)
        dependencies = self._summarize_dependencies(
            daily_research,
            shadow_certification,
            capital_ramp,
        )
        metadata = {
            "summary": status_payload["summary"],
            "strict": strict,
            "checks": checks,
            "failed_required_checks": status_payload["failed_required_checks"],
            "warming_required_checks": status_payload["warming_required_checks"],
            "optional_failures": status_payload["optional_failures"],
            "git": git_summary,
            "dependencies": dependencies,
            "benchmark": benchmark_summary,
            "live_certification": live_certification,
            "regression_suite": regression_suite,
        }
        payload = {
            "timestamp": timestamp,
            "cycle_count": cycle_count,
            "status": status_payload["status"],
            "deployable_for_merge": status_payload["deployable_for_merge"],
            "summary": status_payload["summary"],
            "branch_name": git_summary.get("branch", ""),
            "commit_hash": git_summary.get("commit", ""),
            "checks": checks,
            "git": git_summary,
            "dependencies": dependencies,
            "benchmark": benchmark_summary,
            "live_certification": live_certification,
            "regression_suite": regression_suite,
            "metadata": metadata,
        }
        run_id = db.save_merge_readiness_run(payload)
        payload["run_id"] = run_id

        if self.report_path:
            _write_json(self.report_path, payload)

        self.last_result = payload
        self.last_run_at = timestamp
        self.last_run_id = run_id
        return payload

    def get_dashboard_payload(self, limit: int = 10) -> Dict:
        latest = self.last_result or db.get_latest_merge_readiness_run()
        return {
            "latest": latest,
            "recent": db.get_recent_merge_readiness_runs(limit=limit),
        }


def run_merge_readiness_review(
    *,
    cycle_count: Optional[int] = None,
    force: bool = False,
    strict: bool = True,
    run_regression_suite: bool = False,
    run_benchmark_pack: bool = False,
    force_live_preflight: bool = False,
    live_trader=None,
    manager: Optional[MergeReadinessManager] = None,
) -> Dict:
    manager = manager or MergeReadinessManager(build_merge_readiness_config(config))
    return manager.run(
        cycle_count=cycle_count,
        force=force,
        strict=strict,
        run_regression_suite=run_regression_suite,
        run_benchmark_pack=run_benchmark_pack,
        force_live_preflight=force_live_preflight,
        live_trader=live_trader,
    )
