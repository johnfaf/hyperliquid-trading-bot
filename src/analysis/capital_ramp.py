"""Controlled capital-ramp manager for staged live deployment."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config
from src.core.time_utils import utc_now_iso, utc_now_naive
from src.data import database as db

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_STAGE_ORDER = ("bootstrap", "trial", "scaled", "full")


def build_capital_ramp_config(cfg=config) -> Dict:
    return {
        "enabled": bool(getattr(cfg, "CAPITAL_RAMP_ENABLED", True)),
        "interval_hours": float(getattr(cfg, "CAPITAL_RAMP_INTERVAL_HOURS", 24.0)),
        "lookback_hours": float(getattr(cfg, "CAPITAL_RAMP_LOOKBACK_HOURS", 24.0 * 14)),
        "default_stage": str(getattr(cfg, "CAPITAL_RAMP_DEFAULT_STAGE", "bootstrap") or "bootstrap"),
        "approved_stage": str(getattr(cfg, "CAPITAL_RAMP_APPROVED_STAGE", "bootstrap") or "bootstrap"),
        "live_min_order_usd": float(getattr(cfg, "LIVE_MIN_ORDER_USD", 11.0)),
        "base_max_order_usd": float(getattr(cfg, "LIVE_MAX_ORDER_USD", 12.0)),
        "require_shadow_certified": bool(getattr(cfg, "CAPITAL_RAMP_REQUIRE_SHADOW_CERTIFIED", True)),
        "require_benchmark_clear": bool(getattr(cfg, "CAPITAL_RAMP_REQUIRE_BENCHMARK_CLEAR", True)),
        "max_paper_drawdown_pct": float(getattr(cfg, "CAPITAL_RAMP_MAX_PAPER_DRAWDOWN_PCT", 0.12)),
        "max_live_drawdown_pct": float(getattr(cfg, "CAPITAL_RAMP_MAX_LIVE_DRAWDOWN_PCT", 0.08)),
        "min_live_snapshots": int(getattr(cfg, "CAPITAL_RAMP_MIN_LIVE_SNAPSHOTS", 12)),
        "max_slippage_drift_bps": float(getattr(cfg, "CAPITAL_RAMP_MAX_SLIPPAGE_DRIFT_BPS", 2.0)),
        "max_degraded_source_ratio": float(getattr(cfg, "CAPITAL_RAMP_MAX_DEGRADED_SOURCE_RATIO", 0.30)),
        "max_realized_pnl_gap_ratio": float(
            getattr(cfg, "CAPITAL_RAMP_MAX_REALIZED_PNL_GAP_RATIO", 0.45)
        ),
        "report_path": os.path.abspath(
            getattr(cfg, "CAPITAL_RAMP_REPORT_PATH", "reports/capital_ramp_report.json")
        ),
        "stages": {
            "bootstrap": {
                "order_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_BOOTSTRAP_ORDER_CAP_MULTIPLIER", 0.35)
                ),
                "source_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_BOOTSTRAP_SOURCE_CAP_MULTIPLIER", 0.40)
                ),
            },
            "trial": {
                "order_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_TRIAL_ORDER_CAP_MULTIPLIER", 0.55)
                ),
                "source_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_TRIAL_SOURCE_CAP_MULTIPLIER", 0.60)
                ),
            },
            "scaled": {
                "order_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_SCALED_ORDER_CAP_MULTIPLIER", 0.80)
                ),
                "source_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_SCALED_SOURCE_CAP_MULTIPLIER", 0.80)
                ),
            },
            "full": {
                "order_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_FULL_ORDER_CAP_MULTIPLIER", 1.00)
                ),
                "source_cap_multiplier": float(
                    getattr(cfg, "CAPITAL_RAMP_FULL_SOURCE_CAP_MULTIPLIER", 1.00)
                ),
            },
        },
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


class CapitalRampManager:
    """Evaluate staged live-capital progression and expose runtime limits."""

    def __init__(self, cfg: Dict, *, capital_governor=None):
        self.enabled = bool(cfg.get("enabled", True))
        self.interval_hours = float(cfg.get("interval_hours", 24.0))
        self.lookback_hours = float(cfg.get("lookback_hours", 24.0 * 14))
        self.default_stage = self._normalize_stage(cfg.get("default_stage", "bootstrap"))
        self.approved_stage = self._normalize_stage(cfg.get("approved_stage", self.default_stage))
        self.live_min_order_usd = float(cfg.get("live_min_order_usd", 11.0))
        self.base_max_order_usd = float(cfg.get("base_max_order_usd", 12.0))
        self.require_shadow_certified = bool(cfg.get("require_shadow_certified", True))
        self.require_benchmark_clear = bool(cfg.get("require_benchmark_clear", True))
        self.max_paper_drawdown_pct = float(cfg.get("max_paper_drawdown_pct", 0.12))
        self.max_live_drawdown_pct = float(cfg.get("max_live_drawdown_pct", 0.08))
        self.min_live_snapshots = int(cfg.get("min_live_snapshots", 12))
        self.max_slippage_drift_bps = float(cfg.get("max_slippage_drift_bps", 2.0))
        self.max_degraded_source_ratio = float(cfg.get("max_degraded_source_ratio", 0.30))
        self.max_realized_pnl_gap_ratio = float(cfg.get("max_realized_pnl_gap_ratio", 0.45))
        self.report_path = str(cfg.get("report_path", "") or "").strip()
        self.stages = dict(cfg.get("stages", {}) or {})
        self.capital_governor = capital_governor

        self.last_result: Dict = {}
        self.last_run_at: Optional[str] = None
        self.last_run_id: Optional[str] = None

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_stage(self, stage: str) -> str:
        text = str(stage or "").strip().lower()
        return text if text in _STAGE_ORDER else "bootstrap"

    def _stage_index(self, stage: str) -> int:
        return _STAGE_ORDER.index(self._normalize_stage(stage))

    def _stage_config(self, stage: str) -> Dict:
        normalized = self._normalize_stage(stage)
        payload = dict(self.stages.get(normalized, {}) or {})
        payload.setdefault("order_cap_multiplier", 1.0)
        payload.setdefault("source_cap_multiplier", 1.0)
        payload["stage"] = normalized
        payload["stage_index"] = self._stage_index(normalized)
        return payload

    def _load_state(self) -> Dict:
        state = db.get_capital_ramp_state() or {}
        applied_stage = self._normalize_stage(
            state.get("applied_stage")
            or state.get("current_stage")
            or self.default_stage
        )
        state["applied_stage"] = applied_stage
        state["approved_stage"] = self._normalize_stage(
            state.get("approved_stage") or self.approved_stage
        )
        return state

    def _is_due(self, now: Optional[datetime] = None) -> bool:
        if not self.enabled:
            return False
        reference = _parse_timestamp(self.last_run_at)
        if reference is None:
            latest = db.get_latest_capital_ramp_run()
            if latest:
                self.last_result = latest
                self.last_run_at = latest.get("timestamp")
                self.last_run_id = latest.get("run_id")
                reference = _parse_timestamp(self.last_run_at)
        if reference is None:
            return True
        now = now or utc_now_naive()
        return now - reference >= timedelta(hours=max(self.interval_hours, 0.0))

    def _build_checks(self) -> List[Dict]:
        shadow = db.get_latest_shadow_certification_run()
        daily = db.get_latest_daily_research_run()
        capital_summary = db.get_capital_governor_summary(lookback_hours=self.lookback_hours)
        execution_quality = db.get_execution_quality_summary(lookback_hours=self.lookback_hours)
        divergence = db.get_runtime_divergence_summary(lookback_hours=self.lookback_hours)

        slippage_drift_bps = round(
            self._safe_float(execution_quality.get("avg_realized_slippage_bps"), 0.0)
            - self._safe_float(execution_quality.get("avg_expected_slippage_bps"), 0.0),
            4,
        )
        live_snapshot_count = int(capital_summary.get("live_snapshot_count", 0) or 0)
        benchmark_recommendation = str(daily.get("recommendation", "") or "").strip().lower()

        checks = [
            {
                "name": "shadow_certified",
                "passed": bool(shadow.get("certified", False)) if self.require_shadow_certified else True,
                "warming_up": self.require_shadow_certified and not shadow,
                "value": bool(shadow.get("certified", False)),
                "threshold": True,
            },
            {
                "name": "benchmark_clear",
                "passed": (
                    benchmark_recommendation not in {"", "rollback"}
                    if self.require_benchmark_clear
                    else True
                ),
                "warming_up": self.require_benchmark_clear and not daily,
                "value": benchmark_recommendation or "missing",
                "threshold": "hold_or_promote",
            },
            {
                "name": "paper_drawdown_pct",
                "passed": self._safe_float(capital_summary.get("paper_current_drawdown_pct"), 0.0)
                <= self.max_paper_drawdown_pct,
                "warming_up": False,
                "value": self._safe_float(capital_summary.get("paper_current_drawdown_pct"), 0.0),
                "threshold": self.max_paper_drawdown_pct,
            },
            {
                "name": "live_drawdown_pct",
                "passed": (
                    self._safe_float(capital_summary.get("live_current_drawdown_pct"), 0.0)
                    <= self.max_live_drawdown_pct
                ) if live_snapshot_count >= self.min_live_snapshots else True,
                "warming_up": live_snapshot_count < self.min_live_snapshots,
                "value": self._safe_float(capital_summary.get("live_current_drawdown_pct"), 0.0),
                "threshold": self.max_live_drawdown_pct,
            },
            {
                "name": "slippage_drift_bps",
                "passed": abs(slippage_drift_bps) <= self.max_slippage_drift_bps,
                "warming_up": False,
                "value": slippage_drift_bps,
                "threshold": self.max_slippage_drift_bps,
            },
            {
                "name": "degraded_source_ratio",
                "passed": self._safe_float(capital_summary.get("degraded_source_ratio"), 0.0)
                <= self.max_degraded_source_ratio,
                "warming_up": False,
                "value": self._safe_float(capital_summary.get("degraded_source_ratio"), 0.0),
                "threshold": self.max_degraded_source_ratio,
            },
            {
                "name": "realized_pnl_gap_ratio",
                "passed": self._safe_float(divergence.get("realized_pnl_gap_ratio"), 0.0)
                <= self.max_realized_pnl_gap_ratio,
                "warming_up": False,
                "value": self._safe_float(divergence.get("realized_pnl_gap_ratio"), 0.0),
                "threshold": self.max_realized_pnl_gap_ratio,
            },
        ]
        return checks

    def _recommended_stage(self, applied_stage: str, checks: List[Dict]) -> Dict:
        applied_index = self._stage_index(applied_stage)
        approved_index = self._stage_index(self.approved_stage)
        failed_checks = [item["name"] for item in checks if not item.get("passed", False) and not item.get("warming_up", False)]
        warming_checks = [item["name"] for item in checks if item.get("warming_up", False)]

        if approved_index < applied_index:
            recommended = self.approved_stage
            status = "demoted"
            summary = f"Approved stage lowered to {self.approved_stage}; demoting from {applied_stage}."
        elif warming_checks:
            recommended = applied_stage
            status = "warming_up"
            summary = "Capital ramp waiting on warm-up checks: " + ", ".join(warming_checks)
        elif failed_checks and applied_index > 0:
            recommended = _STAGE_ORDER[applied_index - 1]
            status = "demoted"
            summary = "Capital ramp demoted due to failed checks: " + ", ".join(failed_checks)
        elif failed_checks:
            recommended = applied_stage
            status = "blocked"
            summary = "Capital ramp blocked at bootstrap: " + ", ".join(failed_checks)
        elif applied_index < approved_index:
            recommended = _STAGE_ORDER[applied_index + 1]
            status = "promoted"
            summary = f"Capital ramp eligible to advance from {applied_stage} to {recommended}."
        else:
            recommended = applied_stage
            status = "holding"
            summary = f"Capital ramp holding at {applied_stage}; all active checks clear."

        return {
            "recommended_stage": self._normalize_stage(recommended),
            "status": status,
            "summary": summary,
        }

    def _runtime_limits_from_stage(self, applied_stage: str) -> Dict:
        approved_stage = self._normalize_stage(self.approved_stage)
        effective_index = min(self._stage_index(applied_stage), self._stage_index(approved_stage))
        effective_stage = _STAGE_ORDER[effective_index]
        stage_cfg = self._stage_config(effective_stage)
        configured_max_order_usd = max(self.live_min_order_usd, self.base_max_order_usd)
        effective_max_order_usd = max(
            self.live_min_order_usd,
            configured_max_order_usd * self._safe_float(stage_cfg.get("order_cap_multiplier"), 1.0),
        )
        return {
            "applied_stage": self._normalize_stage(applied_stage),
            "approved_stage": approved_stage,
            "effective_stage": effective_stage,
            "configured_max_order_usd": round(configured_max_order_usd, 4),
            "effective_max_order_usd": round(effective_max_order_usd, 4),
            "order_cap_multiplier": round(
                self._safe_float(stage_cfg.get("order_cap_multiplier"), 1.0),
                4,
            ),
            "source_cap_multiplier": round(
                self._safe_float(stage_cfg.get("source_cap_multiplier"), 1.0),
                4,
            ),
        }

    def get_runtime_limits(self) -> Dict:
        state = self._load_state()
        limits = self._runtime_limits_from_stage(state.get("applied_stage", self.default_stage))
        latest = self.last_result or db.get_latest_capital_ramp_run()
        if latest:
            limits["last_status"] = str(latest.get("status", "warming_up") or "warming_up")
            limits["last_summary"] = str(
                ((latest.get("metadata", {}) or {}).get("summary"))
                or latest.get("summary")
                or ""
            )
            limits["last_run_id"] = latest.get("run_id")
            limits["last_run_at"] = latest.get("timestamp")
        else:
            limits["last_status"] = "warming_up"
            limits["last_summary"] = "Capital ramp has not run yet."
            limits["last_run_id"] = None
            limits["last_run_at"] = None
        limits["enabled"] = self.enabled
        return limits

    def run(self, *, cycle_count: Optional[int] = None, force: bool = False) -> Dict:
        if not self.enabled:
            payload = {
                "run_id": None,
                "timestamp": utc_now_iso(),
                "status": "disabled",
                "applied_stage": self.default_stage,
                "approved_stage": self.approved_stage,
                "recommended_stage": self.default_stage,
                "deployable": False,
                "summary": "Capital ramp disabled.",
                "checks": [],
                "limits": self._runtime_limits_from_stage(self.default_stage),
            }
            self.last_result = payload
            return payload

        now = utc_now_naive()
        if not force and not self._is_due(now):
            latest = db.get_latest_capital_ramp_run()
            if latest:
                self.last_result = latest
                self.last_run_at = latest.get("timestamp")
                self.last_run_id = latest.get("run_id")
                return latest

        state = self._load_state()
        applied_stage = self._normalize_stage(state.get("applied_stage", self.default_stage))
        checks = self._build_checks()
        recommendation = self._recommended_stage(applied_stage, checks)
        recommended_stage = recommendation["recommended_stage"]
        failed_checks = [item["name"] for item in checks if not item.get("passed", False) and not item.get("warming_up", False)]
        warming_checks = [item["name"] for item in checks if item.get("warming_up", False)]

        transitioned = recommended_stage != applied_stage
        next_stage = recommended_stage if transitioned else applied_stage
        limits = self._runtime_limits_from_stage(next_stage)
        deployable = not failed_checks and not warming_checks
        timestamp = utc_now_iso()
        metadata = {
            "summary": recommendation["summary"],
            "checks": checks,
            "failed_checks": failed_checks,
            "warming_checks": warming_checks,
            "limits": limits,
            "previous_stage": applied_stage,
            "transitioned": transitioned,
        }
        payload = {
            "timestamp": timestamp,
            "cycle_count": cycle_count,
            "status": recommendation["status"],
            "applied_stage": next_stage,
            "approved_stage": self.approved_stage,
            "recommended_stage": recommended_stage,
            "deployable": deployable,
            "summary": recommendation["summary"],
            "checks": checks,
            "limits": limits,
            "metadata": metadata,
        }
        run_id = db.save_capital_ramp_run(payload)
        payload["run_id"] = run_id

        db.save_capital_ramp_state(
            {
                "applied_stage": next_stage,
                "approved_stage": self.approved_stage,
                "last_transition_at": timestamp if transitioned else state.get("last_transition_at"),
                "last_run_id": run_id,
                "last_run_at": timestamp,
            }
        )
        if self.report_path:
            _write_json(self.report_path, payload)

        self.last_result = payload
        self.last_run_at = timestamp
        self.last_run_id = run_id
        return payload

    def get_dashboard_payload(self, limit: int = 10) -> Dict:
        latest = self.last_result or db.get_latest_capital_ramp_run()
        return {
            "runtime": self.get_runtime_limits(),
            "latest": latest,
            "recent": db.get_recent_capital_ramp_runs(limit=limit),
        }


def run_capital_ramp(*, cycle_count: Optional[int] = None, force: bool = False, manager: Optional[CapitalRampManager] = None) -> Dict:
    manager = manager or CapitalRampManager(build_capital_ramp_config(config))
    return manager.run(cycle_count=cycle_count, force=force)
