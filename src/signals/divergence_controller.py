"""
Live-vs-paper divergence control.

This layer promotes experiment-discipline divergence reporting into a real
runtime guardrail. It watches paper/shadow/live drift globally and by source,
then returns a neutral, caution, or blocked posture that higher layers can use
to rank lower, size down, or stop new entries outright.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

import config
from src.core.time_utils import utc_now
from src.data import database as db

logger = logging.getLogger(__name__)


class DivergenceController:
    """Evaluate runtime divergence between paper, shadow, and live execution."""

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.live_trading_enabled = bool(
            cfg.get("live_trading_enabled", getattr(config, "LIVE_TRADING_ENABLED", False))
        )
        self.lookback_hours = float(
            cfg.get("lookback_hours", getattr(config, "EXPERIMENT_DIVERGENCE_LOOKBACK_HOURS", 24.0))
        )
        self.refresh_interval_seconds = float(cfg.get("refresh_interval_seconds", 300.0))
        self.min_live_events = int(cfg.get("min_live_events", 3))
        self.source_min_selected = int(cfg.get("source_min_selected", 3))
        self.caution_multiplier = float(cfg.get("caution_multiplier", 0.65))
        self.blocked_multiplier = float(cfg.get("blocked_multiplier", 0.0))
        self.block_on_status = bool(cfg.get("block_on_status", True))
        self.global_caution_open_gap_ratio = float(cfg.get("global_caution_open_gap_ratio", 0.20))
        self.global_block_open_gap_ratio = float(cfg.get("global_block_open_gap_ratio", 0.40))
        self.global_caution_execution_gap_ratio = float(
            cfg.get("global_caution_execution_gap_ratio", 0.25)
        )
        self.global_block_execution_gap_ratio = float(
            cfg.get("global_block_execution_gap_ratio", 0.50)
        )
        self.global_caution_pnl_gap_ratio = float(cfg.get("global_caution_pnl_gap_ratio", 0.20))
        self.global_block_pnl_gap_ratio = float(cfg.get("global_block_pnl_gap_ratio", 0.45))
        self.global_caution_rejection_rate = float(cfg.get("global_caution_rejection_rate", 0.18))
        self.global_block_rejection_rate = float(cfg.get("global_block_rejection_rate", 0.30))
        self.source_caution_execution_gap_ratio = float(
            cfg.get("source_caution_execution_gap_ratio", 0.30)
        )
        self.source_block_execution_gap_ratio = float(
            cfg.get("source_block_execution_gap_ratio", 0.60)
        )
        self.source_caution_rejection_rate = float(
            cfg.get("source_caution_rejection_rate", 0.20)
        )
        self.source_block_rejection_rate = float(
            cfg.get("source_block_rejection_rate", 0.35)
        )
        self.source_caution_fill_ratio = float(cfg.get("source_caution_fill_ratio", 0.70))
        self.source_block_fill_ratio = float(cfg.get("source_block_fill_ratio", 0.45))

        self._last_refresh_at: Optional[datetime] = None
        self._runtime_summary: Dict = {}
        self._global_assessment: Dict = {}
        self._source_by_key: Dict[str, Dict] = {}
        self._source_by_source: Dict[str, Dict] = {}
        self._last_evaluation: Dict = {}

        self.stats = {
            "enabled": self.enabled,
            "live_trading_enabled": self.live_trading_enabled,
            "lookback_hours": self.lookback_hours,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "evaluations": 0,
            "blocked": 0,
            "caution": 0,
            "last_refresh_at": None,
            "last_evaluation": None,
        }

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_source(source) -> str:
        if hasattr(source, "value"):
            source = source.value
        return str(source or "").strip().lower()

    @staticmethod
    def _severity_for_threshold(value: float, caution_threshold: float, block_threshold: float) -> int:
        if value >= block_threshold:
            return 2
        if value >= caution_threshold:
            return 1
        return 0

    def _build_assessment(
        self,
        *,
        scope: str,
        scope_key: str,
        severity: int,
        reasons: List[str],
        metrics: Dict,
    ) -> Dict:
        if not self.enabled:
            status = "disabled"
            multiplier = 1.0
            score = 0.5
        elif not self.live_trading_enabled:
            status = "disabled"
            multiplier = 1.0
            score = 0.5
        elif severity >= 2:
            status = "blocked"
            multiplier = self.blocked_multiplier
            score = 0.0
        elif severity == 1:
            status = "caution"
            multiplier = self.caution_multiplier
            score = 0.35
        elif reasons and (
            "insufficient_live_history" in reasons
            or "insufficient_paper_history" in reasons
        ):
            status = "warming_up"
            multiplier = 1.0
            score = 0.5
        else:
            status = "healthy"
            multiplier = 1.0
            score = 1.0

        assessment = {
            "scope": scope,
            "scope_key": scope_key,
            "status": status,
            "severity": severity,
            "blocked": bool(self.block_on_status and status == "blocked"),
            "multiplier": round(float(multiplier or 0.0), 4),
            "divergence_score": round(float(score), 4),
            "reasons": list(dict.fromkeys(reasons)),
            "metrics": dict(metrics or {}),
        }
        return assessment

    def _evaluate_global(self, summary: Dict) -> Dict:
        if not self.enabled or not self.live_trading_enabled:
            return self._build_assessment(
                scope="global",
                scope_key="global",
                severity=0,
                reasons=["live_trading_disabled"],
                metrics=summary,
            )

        shadow_selected_count = int(summary.get("shadow_selected_count", 0) or 0)
        live_execution_total = int(summary.get("live_execution_total", 0) or 0)
        paper_open_count = int(summary.get("paper_open_count", 0) or 0)
        paper_recent_open_count = int(summary.get("paper_recent_open_count", 0) or 0)
        paper_recent_closed_count = int(summary.get("paper_recent_closed_count", 0) or 0)
        live_open_positions = int(summary.get("live_open_positions", 0) or 0)
        activity = max(
            shadow_selected_count,
            live_execution_total,
            paper_open_count,
            live_open_positions,
        )

        reasons: List[str] = []
        severity = 0
        if activity < self.min_live_events:
            reasons.append("insufficient_live_history")
            return self._build_assessment(
                scope="global",
                scope_key="global",
                severity=0,
                reasons=reasons,
                metrics=summary,
            )

        # Shadow selections are not real paper trades. After a paper reset we
        # can still have live/shadow history in the DB, and counting that as
        # paper activity creates a deadlock where gap ratios block paper from
        # ever warming up again. Require real paper exposure or recent paper
        # closes before treating live-vs-paper divergence as meaningful.
        paper_activity = max(paper_open_count, paper_recent_open_count) + paper_recent_closed_count
        if paper_activity < self.min_live_events:
            reasons.append("insufficient_paper_history")
            return self._build_assessment(
                scope="global",
                scope_key="global",
                severity=0,
                reasons=reasons,
                metrics=summary,
            )

        open_gap_ratio = self._safe_float(summary.get("paper_live_open_gap_ratio"), 0.0)
        exec_gap_ratio = self._safe_float(summary.get("shadow_live_execution_gap_ratio"), 0.0)
        pnl_gap_ratio = self._safe_float(summary.get("paper_live_realized_pnl_gap_ratio"), 0.0)
        live_rejection_rate = self._safe_float(summary.get("live_rejection_rate"), 0.0)

        severity = max(
            severity,
            self._severity_for_threshold(
                open_gap_ratio,
                self.global_caution_open_gap_ratio,
                self.global_block_open_gap_ratio,
            ),
        )
        if open_gap_ratio >= self.global_block_open_gap_ratio:
            reasons.append("global_open_gap_block")
        elif open_gap_ratio >= self.global_caution_open_gap_ratio:
            reasons.append("global_open_gap_caution")

        severity = max(
            severity,
            self._severity_for_threshold(
                exec_gap_ratio,
                self.global_caution_execution_gap_ratio,
                self.global_block_execution_gap_ratio,
            ),
        )
        if exec_gap_ratio >= self.global_block_execution_gap_ratio:
            reasons.append("global_execution_gap_block")
        elif exec_gap_ratio >= self.global_caution_execution_gap_ratio:
            reasons.append("global_execution_gap_caution")

        severity = max(
            severity,
            self._severity_for_threshold(
                pnl_gap_ratio,
                self.global_caution_pnl_gap_ratio,
                self.global_block_pnl_gap_ratio,
            ),
        )
        if pnl_gap_ratio >= self.global_block_pnl_gap_ratio:
            reasons.append("global_realized_pnl_gap_block")
        elif pnl_gap_ratio >= self.global_caution_pnl_gap_ratio:
            reasons.append("global_realized_pnl_gap_caution")

        severity = max(
            severity,
            self._severity_for_threshold(
                live_rejection_rate,
                self.global_caution_rejection_rate,
                self.global_block_rejection_rate,
            ),
        )
        if live_rejection_rate >= self.global_block_rejection_rate:
            reasons.append("global_live_rejection_block")
        elif live_rejection_rate >= self.global_caution_rejection_rate:
            reasons.append("global_live_rejection_caution")

        return self._build_assessment(
            scope="global",
            scope_key="global",
            severity=severity,
            reasons=reasons,
            metrics=summary,
        )

    def _evaluate_source_metrics(self, scope_key: str, metrics: Dict) -> Dict:
        if not self.enabled or not self.live_trading_enabled:
            return self._build_assessment(
                scope="source",
                scope_key=scope_key,
                severity=0,
                reasons=["live_trading_disabled"],
                metrics=metrics,
            )

        selected_count = int(metrics.get("selected_count", 0) or 0)
        live_events = int(metrics.get("live_events", 0) or 0)
        paper_closed_count = int(metrics.get("paper_closed_count", 0) or 0)
        activity = max(selected_count, live_events, paper_closed_count)

        reasons: List[str] = []
        severity = 0
        if activity < self.source_min_selected:
            reasons.append("insufficient_live_history")
            return self._build_assessment(
                scope="source",
                scope_key=scope_key,
                severity=0,
                reasons=reasons,
                metrics=metrics,
            )

        execution_gap_ratio = round(
            abs(selected_count - live_events) / max(1, max(selected_count, live_events)),
            4,
        )
        live_rejection_rate = self._safe_float(metrics.get("live_rejection_rate"), 0.0)
        live_fill_ratio = self._safe_float(metrics.get("live_avg_fill_ratio"), 0.0)

        enriched_metrics = dict(metrics)
        enriched_metrics["execution_gap_ratio"] = execution_gap_ratio

        severity = max(
            severity,
            self._severity_for_threshold(
                execution_gap_ratio,
                self.source_caution_execution_gap_ratio,
                self.source_block_execution_gap_ratio,
            ),
        )
        if execution_gap_ratio >= self.source_block_execution_gap_ratio:
            reasons.append("source_execution_gap_block")
        elif execution_gap_ratio >= self.source_caution_execution_gap_ratio:
            reasons.append("source_execution_gap_caution")

        severity = max(
            severity,
            self._severity_for_threshold(
                live_rejection_rate,
                self.source_caution_rejection_rate,
                self.source_block_rejection_rate,
            ),
        )
        if live_rejection_rate >= self.source_block_rejection_rate:
            reasons.append("source_live_rejection_block")
        elif live_rejection_rate >= self.source_caution_rejection_rate:
            reasons.append("source_live_rejection_caution")

        if live_fill_ratio > 0:
            if live_fill_ratio <= self.source_block_fill_ratio:
                severity = max(severity, 2)
                reasons.append("source_live_fill_block")
            elif live_fill_ratio <= self.source_caution_fill_ratio:
                severity = max(severity, 1)
                reasons.append("source_live_fill_caution")

        return self._build_assessment(
            scope="source",
            scope_key=scope_key,
            severity=severity,
            reasons=reasons,
            metrics=enriched_metrics,
        )

    def _refresh(self, force: bool = False) -> None:
        now = utc_now()
        if (
            not force
            and self._last_refresh_at
            and (now - self._last_refresh_at).total_seconds() < self.refresh_interval_seconds
        ):
            return

        try:
            self._runtime_summary = db.get_runtime_divergence_summary(
                lookback_hours=self.lookback_hours
            )
        except Exception as exc:
            logger.debug("divergence summary refresh error: %s", exc)
            self._runtime_summary = {}

        try:
            source_rows = db.get_source_attribution_summary(
                limit_cycles=getattr(config, "EXPERIMENT_REPORT_LIMIT_CYCLES", 120),
                lookback_hours=self.lookback_hours,
            )
        except Exception as exc:
            logger.debug("divergence source attribution refresh error: %s", exc)
            source_rows = []

        self._global_assessment = self._evaluate_global(self._runtime_summary)
        self._source_by_key = {}
        self._source_by_source = {}
        for row in source_rows or []:
            source_key = str(row.get("source_key", "") or row.get("source") or "unknown").strip() or "unknown"
            source = self._coerce_source(row.get("source")) or "unknown"
            assessment = self._evaluate_source_metrics(source_key, dict(row))
            self._source_by_key[source_key] = assessment
            current = self._source_by_source.get(source)
            if current is None or int(assessment.get("severity", 0) or 0) > int(current.get("severity", 0) or 0):
                self._source_by_source[source] = assessment

        self._last_refresh_at = now
        self.stats["last_refresh_at"] = now.isoformat()

    def _resolve_source_identity(
        self,
        *,
        strategy: Optional[Dict] = None,
        source_key: str = "",
        source: str = "",
    ) -> Tuple[str, str]:
        strategy = strategy or {}
        metadata = strategy.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        resolved_key = str(
            source_key
            or strategy.get("source_key")
            or metadata.get("source_key")
            or ""
        ).strip()
        resolved_source = self._coerce_source(
            source or strategy.get("source") or metadata.get("source")
        ) or "unknown"
        return resolved_key, resolved_source

    @staticmethod
    def _severity_rank(status: str) -> int:
        return {
            "disabled": 0,
            "warming_up": 0,
            "healthy": 1,
            "caution": 2,
            "blocked": 3,
        }.get(str(status or "").strip().lower(), 0)

    def evaluate(
        self,
        *,
        strategy: Optional[Dict] = None,
        source_key: str = "",
        source: str = "",
    ) -> Dict:
        self._refresh()
        resolved_key, resolved_source = self._resolve_source_identity(
            strategy=strategy,
            source_key=source_key,
            source=source,
        )
        global_assessment = dict(self._global_assessment or self._build_assessment(
            scope="global",
            scope_key="global",
            severity=0,
            reasons=["not_refreshed"],
            metrics={},
        ))
        source_assessment = dict(
            self._source_by_key.get(resolved_key)
            or self._source_by_source.get(resolved_source)
            or self._build_assessment(
                scope="source",
                scope_key=resolved_key or resolved_source or "unknown",
                severity=0,
                reasons=["insufficient_live_history"],
                metrics={},
            )
        )

        status_candidates = [global_assessment.get("status", "warming_up"), source_assessment.get("status", "warming_up")]
        status = max(status_candidates, key=self._severity_rank)
        multiplier = min(
            self._safe_float(global_assessment.get("multiplier"), 1.0),
            self._safe_float(source_assessment.get("multiplier"), 1.0),
        )
        divergence_score = min(
            self._safe_float(global_assessment.get("divergence_score"), 0.5),
            self._safe_float(source_assessment.get("divergence_score"), 0.5),
        )
        reasons = list(
            dict.fromkeys(
                list(global_assessment.get("reasons", []) or [])
                + list(source_assessment.get("reasons", []) or [])
            )
        )

        result = {
            "status": status,
            "blocked": bool(self.block_on_status and status == "blocked"),
            "multiplier": round(multiplier, 4),
            "divergence_score": round(divergence_score, 4),
            "source_key": resolved_key,
            "source": resolved_source,
            "reasons": reasons,
            "global": global_assessment,
            "source_profile": source_assessment,
        }

        self.stats["evaluations"] += 1
        if status == "blocked":
            self.stats["blocked"] += 1
        elif status == "caution":
            self.stats["caution"] += 1
        self._last_evaluation = result
        self.stats["last_evaluation"] = result
        return result

    def get_stats(self) -> Dict:
        self._refresh()
        return {
            **dict(self.stats),
            "global_status": self._global_assessment.get("status", "unknown"),
            "global_reasons": list(self._global_assessment.get("reasons", []) or []),
            "tracked_sources": len(self._source_by_key),
        }

    def get_dashboard_payload(self, limit: int = 12) -> Dict:
        self._refresh()
        profiles = list(self._source_by_key.values())
        profiles.sort(
            key=lambda item: (
                -int(item.get("severity", 0) or 0),
                -self._safe_float((item.get("metrics") or {}).get("execution_gap_ratio"), 0.0),
                item.get("scope_key", ""),
            )
        )
        return {
            **self.get_stats(),
            "global": dict(self._global_assessment),
            "profiles": profiles[:limit],
            "runtime_summary": dict(self._runtime_summary),
        }
