"""Daily research loop orchestration for recalibration and benchmark review."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import config
from src.analysis.experiment_discipline import run_experiment_benchmark_pack
from src.core.time_utils import utc_now_iso, utc_now_naive
from src.data import database as db

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def build_daily_research_config(cfg=config) -> Dict:
    return {
        "enabled": getattr(cfg, "DAILY_RESEARCH_LOOP_ENABLED", True),
        "interval_hours": getattr(cfg, "DAILY_RESEARCH_LOOP_INTERVAL_HOURS", 24.0),
        "benchmark_limit_cycles": getattr(
            cfg,
            "DAILY_RESEARCH_LOOP_BENCHMARK_LIMIT_CYCLES",
            getattr(cfg, "EXPERIMENT_REPORT_LIMIT_CYCLES", 120),
        ),
        "out_of_sample_ratio": getattr(
            cfg,
            "DAILY_RESEARCH_LOOP_OOS_RATIO",
            getattr(cfg, "EXPERIMENT_OOS_RATIO", 0.30),
        ),
        "benchmark_report_path": os.path.abspath(
            getattr(cfg, "EXPERIMENT_BENCHMARK_REPORT_PATH", "reports/experiment_benchmark_pack.json")
        ),
        "report_path": os.path.abspath(
            getattr(cfg, "DAILY_RESEARCH_REPORT_PATH", "reports/daily_research_loop.json")
        ),
        "rollback_ev_tolerance_pct": float(
            getattr(cfg, "DAILY_RESEARCH_LOOP_ROLLBACK_EV_TOLERANCE_PCT", 0.0005)
        ),
    }


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _write_json(path: str, payload: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class DailyResearchLoop:
    """Runs the daily recalibration + benchmark review and persists the result."""

    def __init__(self, cfg: Dict, *, adaptive_learning=None):
        self.enabled = bool(cfg.get("enabled", True))
        self.interval_hours = float(cfg.get("interval_hours", 24.0))
        self.benchmark_limit_cycles = int(cfg.get("benchmark_limit_cycles", 120))
        self.out_of_sample_ratio = float(cfg.get("out_of_sample_ratio", 0.30))
        self.benchmark_report_path = str(cfg.get("benchmark_report_path", "") or "").strip()
        self.report_path = str(cfg.get("report_path", "") or "").strip()
        self.rollback_ev_tolerance_pct = float(cfg.get("rollback_ev_tolerance_pct", 0.0005))
        self.adaptive_learning = adaptive_learning
        self.last_run_at: Optional[str] = None
        self.last_run_id: Optional[str] = None
        self.last_result: Dict = {}

    def _is_due(self, now: Optional[datetime] = None) -> bool:
        if not self.enabled:
            return False
        reference = _parse_timestamp(self.last_run_at)
        if reference is None:
            latest = db.get_latest_daily_research_run()
            if latest:
                self.last_run_at = latest.get("timestamp")
                self.last_run_id = latest.get("run_id")
                reference = _parse_timestamp(self.last_run_at)
                self.last_result = latest
        if reference is None:
            return True
        now = now or utc_now_naive()
        return now - reference >= timedelta(hours=max(self.interval_hours, 0.0))

    @staticmethod
    def _profile_view(report: Dict, profile_name: str) -> Dict:
        return ((report.get("profiles", {}) or {}).get(profile_name, {}) or {})

    def _build_delta(self, benchmark: Dict, previous_run: Dict) -> Dict:
        gate = (benchmark.get("promotion_gate", {}) or {})
        baseline = str(gate.get("baseline", "baseline_current") or "baseline_current")
        winner = str(gate.get("winner", baseline) or baseline)
        baseline_report = self._profile_view(benchmark, baseline)
        winner_report = self._profile_view(benchmark, winner)
        baseline_oos = baseline_report.get("out_of_sample", {}) or {}
        winner_oos = winner_report.get("out_of_sample", {}) or {}
        winner_promotion = winner_report.get("promotion", {}) or {}

        previous_meta = (previous_run.get("metadata", {}) or {}) if previous_run else {}
        previous_delta = (previous_meta.get("delta", {}) or {}) if previous_meta else {}
        previous_winner_ev = float(previous_delta.get("winner_ev_delta_pct", 0.0) or 0.0)
        current_winner_ev = float(winner_promotion.get("ev_delta_pct", 0.0) or 0.0)

        return {
            "baseline_profile": baseline,
            "winner_profile": winner,
            "approved_profiles": list(gate.get("approved_profiles", []) or []),
            "baseline_avg_selected_ev_pct": round(
                float(baseline_oos.get("avg_selected_ev_pct", 0.0) or 0.0),
                4,
            ),
            "winner_avg_selected_ev_pct": round(
                float(winner_oos.get("avg_selected_ev_pct", 0.0) or 0.0),
                4,
            ),
            "winner_avg_selected_execution_cost_pct": round(
                float(winner_oos.get("avg_selected_execution_cost_pct", 0.0) or 0.0),
                4,
            ),
            "winner_no_trade_rate": round(
                float(winner_oos.get("no_trade_rate", 0.0) or 0.0),
                4,
            ),
            "winner_ev_delta_pct": round(current_winner_ev, 4),
            "winner_execution_cost_delta_pct": round(
                float(winner_promotion.get("execution_cost_delta_pct", 0.0) or 0.0),
                4,
            ),
            "winner_no_trade_delta": round(
                float(winner_promotion.get("no_trade_delta", 0.0) or 0.0),
                4,
            ),
            "previous_winner_profile": str(previous_run.get("winner_profile", "") or ""),
            "previous_recommendation": str(previous_run.get("recommendation", "") or ""),
            "winner_vs_previous_ev_delta_pct": round(current_winner_ev - previous_winner_ev, 4),
        }

    def _build_recommendation(self, benchmark: Dict, delta: Dict, last_known_good: Dict) -> Dict:
        winner = str(delta.get("winner_profile", "baseline_current") or "baseline_current")
        baseline = str(delta.get("baseline_profile", "baseline_current") or "baseline_current")
        approved = set(delta.get("approved_profiles", []) or [])
        winner_ev_delta = float(delta.get("winner_ev_delta_pct", 0.0) or 0.0)
        last_good_profile = str(last_known_good.get("profile_name", "") or "")

        if winner != baseline and winner in approved:
            return {
                "action": "promote",
                "status": "approved",
                "summary": f"Promote {winner}; approved out-of-sample EV delta {winner_ev_delta:+.4f}.",
                "rollback_target_profile": last_good_profile or baseline,
            }

        if winner_ev_delta < -abs(self.rollback_ev_tolerance_pct) and last_good_profile:
            return {
                "action": "rollback",
                "status": "underperforming",
                "summary": (
                    f"Current benchmark underperformed baseline by {winner_ev_delta:+.4f}; "
                    f"recommend rollback to {last_good_profile}."
                ),
                "rollback_target_profile": last_good_profile,
            }

        return {
            "action": "hold",
            "status": "stable",
            "summary": (
                f"Hold current parameters; winner {winner} is not strong enough to replace baseline."
            ),
            "rollback_target_profile": last_good_profile or baseline,
        }

    def _persist_last_known_good(self, benchmark: Dict, recommendation: Dict, last_known_good: Dict, timestamp: str) -> Dict:
        gate = (benchmark.get("promotion_gate", {}) or {})
        winner = str(gate.get("winner", "baseline_current") or "baseline_current")
        baseline = str(gate.get("baseline", "baseline_current") or "baseline_current")
        approved = set(gate.get("approved_profiles", []) or [])

        updated = dict(last_known_good or {})
        if not updated:
            updated = {
                "profile_name": baseline,
                "overrides": {},
                "saved_at": timestamp,
                "reason": "initial_baseline",
            }

        if recommendation.get("action") == "promote" and winner in approved:
            updated = {
                "profile_name": winner,
                "overrides": dict(self._profile_view(benchmark, winner).get("overrides", {}) or {}),
                "saved_at": timestamp,
                "reason": "daily_benchmark_promote",
            }
        elif not updated.get("profile_name"):
            updated["profile_name"] = baseline
            updated["saved_at"] = timestamp
            updated["reason"] = "baseline_fallback"

        db.save_daily_research_last_known_good(updated)
        return updated

    def run(self, *, cycle_count: Optional[int] = None, force: bool = False) -> Dict:
        if not self.enabled:
            return self.get_stats()

        now = utc_now_naive()
        if not force and not self._is_due(now):
            latest = db.get_latest_daily_research_run()
            if latest:
                self.last_result = latest
                self.last_run_at = latest.get("timestamp")
                self.last_run_id = latest.get("run_id")
            return self.get_stats()

        timestamp = utc_now_iso()
        previous_run = db.get_latest_daily_research_run()
        last_known_good = db.get_daily_research_last_known_good()

        recalibration = {}
        if self.adaptive_learning:
            try:
                recalibration = self.adaptive_learning.run_recalibration(
                    cycle_count=cycle_count,
                    force=True,
                )
            except Exception as exc:
                logger.warning("Daily research loop recalibration error: %s", exc)
                recalibration = {
                    "executed": False,
                    "reasons": ["adaptive_recalibration_failed"],
                    "error": str(exc),
                }
        else:
            recalibration = {
                "executed": False,
                "reasons": ["adaptive_learning_unavailable"],
            }

        benchmark = run_experiment_benchmark_pack(
            limit_cycles=self.benchmark_limit_cycles,
            out_of_sample_ratio=self.out_of_sample_ratio,
        )
        delta = self._build_delta(benchmark, previous_run)
        recommendation = self._build_recommendation(benchmark, delta, last_known_good)
        updated_last_known_good = self._persist_last_known_good(
            benchmark,
            recommendation,
            last_known_good,
            timestamp,
        )

        if self.benchmark_report_path:
            _write_json(self.benchmark_report_path, benchmark)

        payload = {
            "timestamp": timestamp,
            "cycle_count": cycle_count,
            "status": "executed",
            "recommendation": recommendation.get("action", "hold"),
            "winner_profile": delta.get("winner_profile", "baseline_current"),
            "approved_profile_count": len(delta.get("approved_profiles", []) or []),
            "recalibration_run_id": recalibration.get("run_id"),
            "metadata": {
                "recommendation": recommendation,
                "delta": delta,
                "benchmark": benchmark,
                "recalibration": recalibration,
                "last_known_good_before": last_known_good,
                "last_known_good_after": updated_last_known_good,
            },
        }
        run_id = db.save_daily_research_run(payload)
        payload["run_id"] = run_id

        full_report = {
            "run_id": run_id,
            "timestamp": timestamp,
            "cycle_count": cycle_count,
            "status": "executed",
            "recommendation": recommendation,
            "delta": delta,
            "recalibration": recalibration,
            "benchmark": benchmark,
            "last_known_good_before": last_known_good,
            "last_known_good_after": updated_last_known_good,
        }
        if self.report_path:
            _write_json(self.report_path, full_report)

        self.last_run_at = timestamp
        self.last_run_id = run_id
        self.last_result = {
            **payload,
            "metadata": payload["metadata"],
        }
        return full_report

    def get_stats(self) -> Dict:
        latest = self.last_result or db.get_latest_daily_research_run()
        return {
            "enabled": self.enabled,
            "interval_hours": self.interval_hours,
            "benchmark_limit_cycles": self.benchmark_limit_cycles,
            "out_of_sample_ratio": self.out_of_sample_ratio,
            "last_run_at": self.last_run_at or latest.get("timestamp"),
            "last_run_id": self.last_run_id or latest.get("run_id"),
            "latest": latest,
        }


def run_daily_research_loop(*, adaptive_learning=None, cycle_count: Optional[int] = None, force: bool = False) -> Dict:
    loop = DailyResearchLoop(
        build_daily_research_config(config),
        adaptive_learning=adaptive_learning,
    )
    return loop.run(cycle_count=cycle_count, force=force)
