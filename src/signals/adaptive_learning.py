"""
Adaptive learning and source-health management.

Builds a drift-aware health profile for each signal source, persists snapshots
for auditability, and applies a lightweight promotion/demotion review to arena
agents based on realized outcomes.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

from src.data import database as db
from src.signals.signal_schema import build_source_key

logger = logging.getLogger(__name__)


def _status_value(status) -> str:
    return str(getattr(status, "value", status) or "").strip().lower()


def _coerce_status(previous_status, desired: str):
    enum_cls = type(previous_status)
    try:
        return enum_cls(desired)
    except Exception:
        return desired


class AdaptiveLearningManager:
    """Central source-health and promotion policy manager."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        *,
        agent_scorer=None,
        calibration=None,
        arena=None,
    ):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.agent_scorer = agent_scorer
        self.calibration = calibration
        self.arena = arena

        self.lookback_hours = float(cfg.get("lookback_hours", 24.0 * 30))
        self.recent_lookback_hours = float(cfg.get("recent_lookback_hours", 24.0 * 3))
        self.report_limit_cycles = int(cfg.get("report_limit_cycles", 160))
        self.refresh_interval_cycles = int(cfg.get("refresh_interval_cycles", 96))
        self.min_closed_trades = int(cfg.get("min_closed_trades", 5))
        self.min_recent_closed_trades = int(cfg.get("min_recent_closed_trades", 3))
        self.min_selected_candidates = int(cfg.get("min_selected_candidates", 5))
        self.caution_health_floor = float(cfg.get("caution_health_floor", 0.46))
        self.promotion_health_floor = float(cfg.get("promotion_health_floor", 0.70))
        self.caution_drift_threshold = float(cfg.get("caution_drift_threshold", 0.18))
        self.block_drift_threshold = float(cfg.get("block_drift_threshold", 0.33))
        self.max_calibration_ece = float(cfg.get("max_calibration_ece", 0.20))
        self.min_weight_multiplier = float(cfg.get("min_weight_multiplier", 0.12))
        self.return_scale = float(cfg.get("return_scale", 0.04))
        self.base_reference_confidence = float(cfg.get("base_reference_confidence", 0.70))

        self.arena_min_trades = int(cfg.get("arena_min_trades", 10))
        self.arena_min_win_rate = float(cfg.get("arena_min_win_rate", 0.52))
        self.arena_min_sharpe = float(cfg.get("arena_min_sharpe", 0.05))
        self.arena_max_drawdown = float(cfg.get("arena_max_drawdown", 0.22))

        self.source_profiles: Dict[str, Dict] = {}
        self.summary: Dict = {}
        self.last_snapshot_id: Optional[str] = None
        self.last_run_at: Optional[str] = None
        self.last_run_cycle: Optional[int] = None
        self.refresh_count = 0
        self.latest_arena_review = []

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(value, upper))

    def attach(self, *, arena=None, agent_scorer=None, calibration=None):
        if arena is not None:
            self.arena = arena
        if agent_scorer is not None:
            self.agent_scorer = agent_scorer
        if calibration is not None:
            self.calibration = calibration

    def ensure_profiles(self):
        if self.enabled and not self.source_profiles:
            self.refresh(force=True)

    def run_cycle(self, cycle_count: Optional[int] = None, *, force: bool = False) -> Dict:
        if not self.enabled:
            return self.get_stats()

        if (
            not force
            and cycle_count is not None
            and self.last_run_cycle is not None
            and cycle_count - self.last_run_cycle < self.refresh_interval_cycles
        ):
            return self.get_stats()

        return self.refresh(force=force, cycle_count=cycle_count)

    def refresh(self, *, force: bool = False, cycle_count: Optional[int] = None) -> Dict:
        if not self.enabled and not force:
            return self.get_stats()

        attribution_rows = db.get_source_attribution_summary(
            limit_cycles=self.report_limit_cycles,
            lookback_hours=self.lookback_hours,
        )
        baseline_rows = db.get_source_trade_outcome_summary(self.lookback_hours)
        recent_rows = db.get_source_trade_outcome_summary(self.recent_lookback_hours)
        profiles = self._build_profiles(attribution_rows, baseline_rows, recent_rows)

        summary = self._summarize_profiles(profiles)
        snapshot_metadata = {
            "summary": summary,
            "lookback_hours": self.lookback_hours,
            "recent_lookback_hours": self.recent_lookback_hours,
            "cycle_count": cycle_count,
        }
        snapshot_id = db.save_source_health_snapshot(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": snapshot_metadata,
                "profiles": list(profiles.values()),
            }
        )

        arena_review = self._review_arena_agents(profiles)
        if arena_review:
            db.save_arena_review_events(arena_review)

        self.source_profiles = profiles
        self.summary = summary
        self.last_snapshot_id = snapshot_id
        self.last_run_at = datetime.utcnow().isoformat()
        self.last_run_cycle = cycle_count
        self.refresh_count += 1
        self.latest_arena_review = arena_review
        return self.get_stats()

    def _build_profiles(self, attribution_rows: list, baseline_rows: list, recent_rows: list) -> Dict[str, Dict]:
        attribution_by_key = {
            str(row.get("source_key", "") or "").strip(): row
            for row in (attribution_rows or [])
            if str(row.get("source_key", "") or "").strip()
        }
        baseline_by_key = {
            str(row.get("source_key", "") or "").strip(): row
            for row in (baseline_rows or [])
            if str(row.get("source_key", "") or "").strip()
        }
        recent_by_key = {
            str(row.get("source_key", "") or "").strip(): row
            for row in (recent_rows or [])
            if str(row.get("source_key", "") or "").strip()
        }

        score_by_key = {}
        if self.agent_scorer and hasattr(self.agent_scorer, "get_all_scores"):
            for row in self.agent_scorer.get_all_scores() or []:
                key = str(row.get("source_key", "") or "").strip()
                if key:
                    score_by_key[key] = row

        calibration_keys = set()
        global_ece = None
        if self.calibration:
            try:
                global_ece = self.calibration.get_ece("global")
                calibration_keys = {
                    key
                    for key in (self.calibration.get_all_stats() or {}).keys()
                    if key and key != "global"
                }
            except Exception:
                calibration_keys = set()

        total_selected = sum(
            int(row.get("selected_count", 0) or 0) for row in (attribution_rows or [])
        )
        all_keys = (
            set(attribution_by_key)
            | set(baseline_by_key)
            | set(recent_by_key)
            | set(score_by_key)
            | calibration_keys
        )

        profiles = {}
        for source_key in sorted(all_keys):
            if not source_key or source_key == "global":
                continue

            attr = attribution_by_key.get(source_key, {})
            baseline = baseline_by_key.get(source_key, {})
            recent = recent_by_key.get(source_key, {})
            score = score_by_key.get(source_key, {})

            source = str(
                attr.get("source")
                or baseline.get("source")
                or recent.get("source")
                or "unknown"
            ).strip().lower() or "unknown"
            selection_count = int(attr.get("selected_count", 0) or 0)
            closed_trades = int(baseline.get("closed_trades", 0) or 0)
            recent_closed_trades = int(recent.get("closed_trades", 0) or 0)
            sample_size = max(closed_trades, int(score.get("total_signals", 0) or 0), selection_count)

            dynamic_weight = float(score.get("dynamic_weight", 0.5) or 0.5)
            weighted_accuracy = float(
                score.get(
                    "weighted_accuracy",
                    score.get("accuracy", baseline.get("win_rate", 0.0)),
                )
                or 0.0
            )
            baseline_win_rate = float(baseline.get("win_rate", weighted_accuracy) or 0.0)
            recent_win_rate = float(recent.get("win_rate", baseline_win_rate) or 0.0)
            avg_return_pct = float(baseline.get("avg_return_pct", 0.0) or 0.0)
            recent_avg_return_pct = float(recent.get("avg_return_pct", avg_return_pct) or 0.0)
            realized_pnl = float(baseline.get("realized_pnl", 0.0) or 0.0)
            live_success_rate = float(attr.get("live_success_rate", 0.0) or 0.0)
            live_rejection_rate = float(attr.get("live_rejection_rate", 0.0) or 0.0)

            calibration_ece = global_ece
            confidence_multiplier = 1.0
            if self.calibration:
                try:
                    calibration_ece = self.calibration.get_ece(source_key)
                    if calibration_ece is None:
                        calibration_ece = global_ece
                    adjusted_conf = self.calibration.get_adjustment_factor(
                        source_key,
                        self.base_reference_confidence,
                    )
                    confidence_multiplier = self._clamp(
                        adjusted_conf / max(self.base_reference_confidence, 1e-8),
                        0.75,
                        1.15,
                    )
                except Exception:
                    confidence_multiplier = 1.0

            health_score = self._compute_health_score(
                dynamic_weight=dynamic_weight,
                baseline_win_rate=baseline_win_rate,
                avg_return_pct=avg_return_pct,
                calibration_ece=calibration_ece,
                live_rejection_rate=live_rejection_rate,
                sample_size=sample_size,
            )
            drift_score = self._compute_drift_score(
                baseline_win_rate=baseline_win_rate,
                recent_win_rate=recent_win_rate,
                avg_return_pct=avg_return_pct,
                recent_avg_return_pct=recent_avg_return_pct,
                sample_size=sample_size,
                recent_sample_size=recent_closed_trades,
            )
            status = self._classify_status(
                sample_size=sample_size,
                selection_count=selection_count,
                health_score=health_score,
                drift_score=drift_score,
                calibration_ece=calibration_ece,
                live_rejection_rate=live_rejection_rate,
            )
            status_multiplier = {
                "active": 1.0,
                "warming_up": 0.95,
                "caution": 0.70,
                "blocked": 0.30,
            }.get(status, 1.0)
            confidence_status_multiplier = {
                "active": 1.0,
                "warming_up": 0.98,
                "caution": 0.88,
                "blocked": 0.60,
            }.get(status, 1.0)
            weight_multiplier = self._clamp(
                (0.70 + health_score * 0.60) * status_multiplier,
                self.min_weight_multiplier,
                1.25,
            )
            confidence_multiplier = self._clamp(
                confidence_multiplier * (0.90 + health_score * 0.20) * confidence_status_multiplier,
                0.45,
                1.15,
            )
            training_label, recommended_action = self._training_label_for(
                status=status,
                health_score=health_score,
                drift_score=drift_score,
                sample_size=sample_size,
            )

            profiles[source_key] = {
                "source_key": source_key,
                "source": source,
                "status": status,
                "training_label": training_label,
                "recommended_action": recommended_action,
                "health_score": round(health_score, 4),
                "weight_multiplier": round(weight_multiplier, 4),
                "confidence_multiplier": round(confidence_multiplier, 4),
                "sample_size": int(sample_size),
                "recent_sample_size": int(recent_closed_trades),
                "selection_count": selection_count,
                "closed_trades": closed_trades,
                "recent_closed_trades": recent_closed_trades,
                "win_rate": round(baseline_win_rate, 4),
                "recent_win_rate": round(recent_win_rate, 4),
                "avg_return_pct": round(avg_return_pct, 4),
                "recent_avg_return_pct": round(recent_avg_return_pct, 4),
                "realized_pnl": round(realized_pnl, 2),
                "calibration_ece": None if calibration_ece is None else round(float(calibration_ece), 4),
                "drift_score": round(drift_score, 4),
                "live_success_rate": round(live_success_rate, 4),
                "live_rejection_rate": round(live_rejection_rate, 4),
                "metadata": {
                    "dynamic_weight": round(dynamic_weight, 4),
                    "weighted_accuracy": round(weighted_accuracy, 4),
                    "selection_share": round(selection_count / max(total_selected, 1), 4),
                    "paper_open_count": int(attr.get("paper_open_count", 0) or 0),
                    "paper_closed_count": int(attr.get("paper_closed_count", 0) or 0),
                    "live_events": int(attr.get("live_events", 0) or 0),
                    "avg_composite_score": float(attr.get("avg_composite_score", 0.0) or 0.0),
                    "avg_expected_value_pct": float(attr.get("avg_expected_value_pct", 0.0) or 0.0),
                },
            }
        return profiles

    def _compute_health_score(
        self,
        *,
        dynamic_weight: float,
        baseline_win_rate: float,
        avg_return_pct: float,
        calibration_ece,
        live_rejection_rate: float,
        sample_size: int,
    ) -> float:
        return_component = self._clamp(0.5 + (avg_return_pct / max(self.return_scale, 1e-8)), 0.0, 1.0)
        ece_penalty = 0.25
        if calibration_ece is not None:
            ece_penalty = self._clamp(float(calibration_ece) / max(self.max_calibration_ece, 1e-8), 0.0, 1.0)
        sample_multiplier = self._clamp(sample_size / max(self.min_closed_trades * 3, 1), 0.45, 1.0)
        raw = (
            0.34 * self._clamp(dynamic_weight, 0.0, 1.0)
            + 0.24 * self._clamp(baseline_win_rate, 0.0, 1.0)
            + 0.20 * return_component
            + 0.12 * (1.0 - self._clamp(live_rejection_rate, 0.0, 1.0))
            + 0.10 * (1.0 - ece_penalty)
        )
        return self._clamp(raw * sample_multiplier, 0.0, 1.0)

    def _compute_drift_score(
        self,
        *,
        baseline_win_rate: float,
        recent_win_rate: float,
        avg_return_pct: float,
        recent_avg_return_pct: float,
        sample_size: int,
        recent_sample_size: int,
    ) -> float:
        if sample_size < self.min_closed_trades or recent_sample_size < self.min_recent_closed_trades:
            return 0.0

        win_rate_drift = max(0.0, baseline_win_rate - recent_win_rate)
        return_drift = max(0.0, avg_return_pct - recent_avg_return_pct)
        drift = (
            min(win_rate_drift * 1.8, 0.65)
            + min(return_drift / max(self.return_scale, 1e-8), 0.55)
        )
        return self._clamp(drift, 0.0, 1.0)

    def _classify_status(
        self,
        *,
        sample_size: int,
        selection_count: int,
        health_score: float,
        drift_score: float,
        calibration_ece,
        live_rejection_rate: float,
    ) -> str:
        if sample_size < self.min_closed_trades and selection_count < self.min_selected_candidates:
            return "warming_up"
        if (
            drift_score >= self.block_drift_threshold
            or health_score < (self.caution_health_floor * 0.65)
            or live_rejection_rate >= 0.55
        ):
            return "blocked"
        if (
            drift_score >= self.caution_drift_threshold
            or health_score < self.caution_health_floor
            or (calibration_ece is not None and float(calibration_ece) > self.max_calibration_ece)
            or live_rejection_rate >= 0.35
        ):
            return "caution"
        return "active"

    def _training_label_for(
        self,
        *,
        status: str,
        health_score: float,
        drift_score: float,
        sample_size: int,
    ):
        if status == "blocked":
            return "freeze", "disable source until retrained"
        if status == "caution":
            return "monitor", "reduce allocation and monitor drift"
        if sample_size >= self.min_closed_trades and health_score >= self.promotion_health_floor and drift_score < self.caution_drift_threshold:
            return "promote", "increase weight and keep in main ranking set"
        if status == "warming_up":
            return "incubate", "collect more evidence before promoting"
        return "monitor", "keep active with current allocation"

    def _summarize_profiles(self, profiles: Dict[str, Dict]) -> Dict:
        counts = {"active": 0, "warming_up": 0, "caution": 0, "blocked": 0}
        promote = 0
        freeze = 0
        for profile in profiles.values():
            status = str(profile.get("status", "warming_up") or "warming_up")
            counts[status] = counts.get(status, 0) + 1
            if profile.get("training_label") == "promote":
                promote += 1
            if profile.get("training_label") == "freeze":
                freeze += 1

        top_profiles = sorted(
            profiles.values(),
            key=lambda item: (-float(item.get("health_score", 0.0) or 0.0), item["source_key"]),
        )[:5]
        constrained = [
            item
            for item in sorted(
                profiles.values(),
                key=lambda entry: (-float(entry.get("drift_score", 0.0) or 0.0), entry["source_key"]),
            )
            if item.get("status") in {"caution", "blocked"}
        ][:5]

        return {
            "sources_tracked": len(profiles),
            "status_counts": counts,
            "promote_count": promote,
            "freeze_count": freeze,
            "top_sources": [
                {
                    "source_key": item["source_key"],
                    "health_score": item["health_score"],
                    "status": item["status"],
                }
                for item in top_profiles
            ],
            "constrained_sources": [
                {
                    "source_key": item["source_key"],
                    "drift_score": item["drift_score"],
                    "status": item["status"],
                }
                for item in constrained
            ],
        }

    def _review_arena_agents(self, profiles: Dict[str, Dict]) -> list:
        if not self.arena or not getattr(self.arena, "agents", None):
            return []

        events = []
        for agent in self.arena.agents.values():
            previous_status = agent.status
            previous_status_value = _status_value(previous_status)
            if previous_status_value == "eliminated" or int(agent.total_trades or 0) < self.arena_min_trades:
                continue

            source_key = build_source_key(
                "arena_champion",
                strategy_type=agent.strategy_type,
                agent_id=agent.agent_id,
            )
            profile = profiles.get(source_key, {})
            health_score = float(profile.get("health_score", 0.5) or 0.5)
            drift_score = float(profile.get("drift_score", 0.0) or 0.0)
            new_status = previous_status
            action = ""
            reason = ""

            if (
                drift_score >= self.block_drift_threshold
                or float(agent.max_drawdown or 0.0) >= self.arena_max_drawdown
                or (int(agent.total_trades or 0) >= self.arena_min_trades and float(agent.sharpe_ratio or 0.0) < -0.10)
            ):
                new_status = _coerce_status(previous_status, "probation")
                action = "demote" if previous_status_value == "champion" else "flag"
                reason = "arena health degraded"
            elif (
                health_score >= self.promotion_health_floor
                and float(agent.win_rate or 0.0) >= self.arena_min_win_rate
                and float(agent.sharpe_ratio or 0.0) >= self.arena_min_sharpe
            ):
                new_status = _coerce_status(previous_status, "champion")
                action = "promote" if previous_status_value != "champion" else ""
                reason = "arena health approved"
            elif previous_status_value == "probation" and health_score >= self.caution_health_floor and drift_score < self.caution_drift_threshold:
                new_status = _coerce_status(previous_status, "active")
                action = "restore"
                reason = "arena health recovered"
            elif previous_status_value == "champion" and (
                health_score < self.caution_health_floor or drift_score >= self.caution_drift_threshold
            ):
                new_status = _coerce_status(previous_status, "active")
                action = "derisk"
                reason = "champion downgraded due to caution profile"

            if new_status != previous_status:
                agent.status = new_status
                events.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent_id": agent.agent_id,
                        "agent_name": agent.name,
                        "strategy_type": agent.strategy_type,
                        "previous_status": previous_status_value,
                        "new_status": _status_value(new_status),
                        "action": action or "update",
                        "reason": reason,
                        "metrics": {
                            "health_score": round(health_score, 4),
                            "drift_score": round(drift_score, 4),
                            "win_rate": round(float(agent.win_rate or 0.0), 4),
                            "sharpe_ratio": round(float(agent.sharpe_ratio or 0.0), 4),
                            "max_drawdown": round(float(agent.max_drawdown or 0.0), 4),
                            "total_trades": int(agent.total_trades or 0),
                        },
                    }
                )

        if events and hasattr(self.arena, "_save_agents"):
            try:
                self.arena._save_agents()
            except Exception as exc:
                logger.debug("adaptive arena save error: %s", exc)
        return events

    def get_source_profile(self, source_key: str = "", source: str = "") -> Dict:
        if not self.enabled:
            return {}
        self.ensure_profiles()

        normalized_key = str(source_key or "").strip()
        if normalized_key and normalized_key in self.source_profiles:
            return self.source_profiles[normalized_key]

        normalized_source = str(source or "").strip().lower()
        if normalized_source:
            for profile in self.source_profiles.values():
                if profile.get("source") == normalized_source:
                    return profile
        return {}

    def get_stats(self) -> Dict:
        return {
            "enabled": self.enabled,
            "refresh_interval_cycles": self.refresh_interval_cycles,
            "lookback_hours": self.lookback_hours,
            "recent_lookback_hours": self.recent_lookback_hours,
            "last_run_at": self.last_run_at,
            "last_snapshot_id": self.last_snapshot_id,
            "refresh_count": self.refresh_count,
            "summary": self.summary or self._summarize_profiles(self.source_profiles),
            "arena_review": self.latest_arena_review[:10],
        }

    def get_dashboard_payload(self, limit: int = 12) -> Dict:
        self.ensure_profiles()
        ordered = sorted(
            self.source_profiles.values(),
            key=lambda item: (
                {
                    "blocked": 0,
                    "caution": 1,
                    "warming_up": 2,
                    "active": 3,
                }.get(str(item.get("status", "warming_up") or "warming_up"), 4),
                -float(item.get("health_score", 0.0) or 0.0),
                item["source_key"],
            ),
        )
        return {
            **self.get_stats(),
            "profiles": ordered[:limit],
            "recent_arena_reviews": db.get_recent_arena_review_events(limit=10),
        }
