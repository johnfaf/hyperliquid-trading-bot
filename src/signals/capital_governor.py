"""
Adaptive capital governor.

This layer turns recent portfolio quality into a runtime risk budget. It looks
at paper and live drawdown, rolling return quality, source-health degradation,
and regime confidence, then decides whether the bot should trade normally,
scale down, move into risk-off mode, or stop opening new entries.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
from typing import Dict, Optional

import config
from src.data import database as db
from src.core.time_utils import utc_now

logger = logging.getLogger(__name__)
_TRUE_VALUES = {"true", "1", "yes"}


class CapitalGovernor:
    """Convert recent portfolio quality into a global risk posture."""

    def __init__(self, cfg: Optional[Dict] = None, divergence_controller=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lookback_hours = float(cfg.get("lookback_hours", 24.0 * 14))
        self.refresh_interval_seconds = float(cfg.get("refresh_interval_seconds", 300.0))
        self.min_paper_trades = int(cfg.get("min_paper_trades", 5))
        self.min_live_snapshots = int(cfg.get("min_live_snapshots", 12))
        self.min_source_profiles = int(cfg.get("min_source_profiles", 3))
        self.caution_multiplier = float(cfg.get("caution_multiplier", 0.75))
        self.risk_off_multiplier = float(cfg.get("risk_off_multiplier", 0.40))
        self.blocked_multiplier = float(cfg.get("blocked_multiplier", 0.0))
        self.block_on_risk_off = bool(cfg.get("block_on_risk_off", False))
        self.caution_paper_drawdown_pct = float(cfg.get("caution_paper_drawdown_pct", 0.08))
        self.risk_off_paper_drawdown_pct = float(cfg.get("risk_off_paper_drawdown_pct", 0.15))
        self.block_paper_drawdown_pct = float(cfg.get("block_paper_drawdown_pct", 0.22))
        self.caution_live_drawdown_pct = float(cfg.get("caution_live_drawdown_pct", 0.05))
        self.risk_off_live_drawdown_pct = float(cfg.get("risk_off_live_drawdown_pct", 0.10))
        self.block_live_drawdown_pct = float(cfg.get("block_live_drawdown_pct", 0.16))
        self.caution_paper_sharpe = float(cfg.get("caution_paper_sharpe", 0.0))
        self.risk_off_paper_sharpe = float(cfg.get("risk_off_paper_sharpe", -0.35))
        self.caution_live_sharpe = float(cfg.get("caution_live_sharpe", 0.0))
        self.risk_off_live_sharpe = float(cfg.get("risk_off_live_sharpe", -0.25))
        self.caution_degraded_source_ratio = float(cfg.get("caution_degraded_source_ratio", 0.35))
        self.risk_off_blocked_source_ratio = float(cfg.get("risk_off_blocked_source_ratio", 0.30))
        self.low_regime_confidence = float(cfg.get("low_regime_confidence", 0.45))
        self.divergence_blocks = bool(cfg.get("divergence_blocks", True))
        self.operator_risk_off_blocks = bool(cfg.get("operator_risk_off_blocks", True))
        self.divergence_controller = divergence_controller

        self._last_refresh_at: Optional[datetime] = None
        self._summary: Dict = {}
        self._last_evaluation: Dict = {}
        self.stats = {
            "enabled": self.enabled,
            "lookback_hours": self.lookback_hours,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "evaluations": 0,
            "caution": 0,
            "risk_off": 0,
            "blocked": 0,
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
    def _runtime_bool(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in _TRUE_VALUES

    @staticmethod
    def _runtime_text(name: str, default: str = "") -> str:
        value = os.environ.get(name)
        if value is None:
            return str(default or "").strip()
        return str(value).strip()

    def _get_operator_override(self) -> Dict:
        return {
            "enabled": self._runtime_bool(
                "OPERATOR_RISK_OFF_ENABLED",
                getattr(config, "OPERATOR_RISK_OFF_ENABLED", False),
            ),
            "reason": self._runtime_text(
                "OPERATOR_RISK_OFF_REASON",
                getattr(config, "OPERATOR_RISK_OFF_REASON", ""),
            ),
            "set_by": self._runtime_text(
                "OPERATOR_RISK_OFF_SET_BY",
                getattr(config, "OPERATOR_RISK_OFF_SET_BY", ""),
            ),
            "set_at": self._runtime_text(
                "OPERATOR_RISK_OFF_SET_AT",
                getattr(config, "OPERATOR_RISK_OFF_SET_AT", ""),
            ),
        }

    def _runtime_live_enabled(self) -> bool:
        profile = self._runtime_text(
            "BOT_RUNTIME_PROFILE",
            getattr(config, "RUNTIME_PROFILE", "paper"),
        ).lower() or "paper"
        live_enabled = self._runtime_bool(
            "LIVE_TRADING_ENABLED",
            getattr(config, "LIVE_TRADING_ENABLED", False),
        )
        return profile == "live" or live_enabled

    @staticmethod
    def _severity_for_high(value: float, caution_threshold: float, risk_off_threshold: float, block_threshold: float) -> int:
        if value >= block_threshold:
            return 3
        if value >= risk_off_threshold:
            return 2
        if value >= caution_threshold:
            return 1
        return 0

    @staticmethod
    def _severity_for_low(value: float, caution_floor: float, risk_off_floor: float) -> int:
        if value <= risk_off_floor:
            return 2
        if value <= caution_floor:
            return 1
        return 0

    def _refresh(self, force: bool = False) -> None:
        now = utc_now()
        if (
            not force
            and self._last_refresh_at
            and (now - self._last_refresh_at).total_seconds() < self.refresh_interval_seconds
        ):
            return

        try:
            self._summary = db.get_capital_governor_summary(lookback_hours=self.lookback_hours)
        except Exception as exc:
            logger.debug("capital governor summary refresh error: %s", exc)
            self._summary = {}

        self._last_refresh_at = now
        self.stats["last_refresh_at"] = now.isoformat()

    def _build_assessment(
        self,
        *,
        severity: int,
        reasons: list,
        metrics: Dict,
        regime_name: str = "",
        regime_confidence: float = 0.0,
        divergence_status: str = "unknown",
    ) -> Dict:
        if not self.enabled:
            status = "disabled"
            multiplier = 1.0
            score = 0.5
        elif severity >= 3:
            status = "blocked"
            multiplier = self.blocked_multiplier
            score = 0.0
        elif severity == 2:
            status = "risk_off"
            multiplier = self.risk_off_multiplier
            score = 0.25
        elif severity == 1:
            status = "caution"
            multiplier = self.caution_multiplier
            score = 0.60
        elif reasons and "insufficient_capital_history" in reasons:
            status = "warming_up"
            multiplier = 1.0
            score = 0.5
        else:
            status = "healthy"
            multiplier = 1.0
            score = 1.0

        blocked = status == "blocked" or (status == "risk_off" and self.block_on_risk_off)
        return {
            "status": status,
            "blocked": blocked,
            "multiplier": round(float(multiplier or 0.0), 4),
            "capital_score": round(float(score), 4),
            "severity": severity,
            "reasons": list(dict.fromkeys(reasons)),
            "metrics": dict(metrics or {}),
            "regime_name": regime_name,
            "regime_confidence": round(float(regime_confidence or 0.0), 4),
            "divergence_status": divergence_status,
        }

    def evaluate(self, regime_data: Optional[Dict] = None) -> Dict:
        self._refresh()
        if not self.enabled:
            result = self._build_assessment(
                severity=0,
                reasons=["capital_governor_disabled"],
                metrics=self._summary,
            )
            self._last_evaluation = result
            self.stats["last_evaluation"] = result
            return result

        summary = dict(self._summary or {})
        runtime_live_enabled = self._runtime_live_enabled()
        if not runtime_live_enabled:
            summary["live_snapshot_count"] = 0
            summary["live_current_drawdown_pct"] = 0.0
            summary["live_sharpe"] = 0.0
            summary["live_metrics_ignored"] = True
        paper_closed_trades = int(summary.get("paper_closed_trades", 0) or 0)
        live_snapshot_count = int(summary.get("live_snapshot_count", 0) or 0)
        source_profile_count = int(summary.get("source_profile_count", 0) or 0)

        reasons = []
        severity = 0
        if paper_closed_trades < self.min_paper_trades and live_snapshot_count < self.min_live_snapshots:
            reasons.append("insufficient_capital_history")

        if paper_closed_trades >= self.min_paper_trades:
            paper_drawdown = self._safe_float(summary.get("paper_current_drawdown_pct"), 0.0)
            paper_drawdown_severity = self._severity_for_high(
                paper_drawdown,
                self.caution_paper_drawdown_pct,
                self.risk_off_paper_drawdown_pct,
                self.block_paper_drawdown_pct,
            )
            severity = max(severity, paper_drawdown_severity)
            if paper_drawdown_severity >= 3:
                reasons.append("paper_drawdown_block")
            elif paper_drawdown_severity == 2:
                reasons.append("paper_drawdown_risk_off")
            elif paper_drawdown_severity == 1:
                reasons.append("paper_drawdown_caution")

            paper_sharpe = self._safe_float(summary.get("paper_sharpe"), 0.0)
            paper_sharpe_severity = self._severity_for_low(
                paper_sharpe,
                self.caution_paper_sharpe,
                self.risk_off_paper_sharpe,
            )
            severity = max(severity, paper_sharpe_severity)
            if paper_sharpe_severity == 2:
                reasons.append("paper_sharpe_risk_off")
            elif paper_sharpe_severity == 1:
                reasons.append("paper_sharpe_caution")

        if live_snapshot_count >= self.min_live_snapshots:
            live_drawdown = self._safe_float(summary.get("live_current_drawdown_pct"), 0.0)
            live_drawdown_severity = self._severity_for_high(
                live_drawdown,
                self.caution_live_drawdown_pct,
                self.risk_off_live_drawdown_pct,
                self.block_live_drawdown_pct,
            )
            severity = max(severity, live_drawdown_severity)
            if live_drawdown_severity >= 3:
                reasons.append("live_drawdown_block")
            elif live_drawdown_severity == 2:
                reasons.append("live_drawdown_risk_off")
            elif live_drawdown_severity == 1:
                reasons.append("live_drawdown_caution")

            live_sharpe = self._safe_float(summary.get("live_sharpe"), 0.0)
            live_sharpe_severity = self._severity_for_low(
                live_sharpe,
                self.caution_live_sharpe,
                self.risk_off_live_sharpe,
            )
            severity = max(severity, live_sharpe_severity)
            if live_sharpe_severity == 2:
                reasons.append("live_sharpe_risk_off")
            elif live_sharpe_severity == 1:
                reasons.append("live_sharpe_caution")

        if source_profile_count >= self.min_source_profiles:
            degraded_source_ratio = self._safe_float(summary.get("degraded_source_ratio"), 0.0)
            blocked_source_ratio = self._safe_float(summary.get("blocked_source_ratio"), 0.0)
            if blocked_source_ratio >= self.risk_off_blocked_source_ratio:
                severity = max(severity, 2)
                reasons.append("blocked_source_ratio_risk_off")
            elif degraded_source_ratio >= self.caution_degraded_source_ratio:
                severity = max(severity, 1)
                reasons.append("degraded_source_ratio_caution")

        divergence_status = "unknown"
        if self.divergence_controller:
            try:
                divergence_stats = self.divergence_controller.get_stats()
                divergence_status = str(divergence_stats.get("global_status", "unknown") or "unknown").strip().lower()
            except Exception as exc:
                logger.debug("capital governor divergence lookup error: %s", exc)
                divergence_status = "unknown"
        if self.divergence_blocks and divergence_status == "blocked":
            severity = max(severity, 3)
            reasons.append("divergence_runtime_block")
        elif divergence_status == "caution":
            severity = max(severity, 1)
            reasons.append("divergence_runtime_caution")

        operator_override = self._get_operator_override()
        if self.operator_risk_off_blocks and operator_override.get("enabled", False):
            severity = max(severity, 3)
            reasons.append("operator_risk_off")

        regime_name = ""
        regime_confidence = 0.0
        if isinstance(regime_data, dict):
            regime_name = str(regime_data.get("overall_regime", "") or "")
            regime_confidence = self._safe_float(regime_data.get("overall_confidence"), 0.0)
        if regime_name and regime_confidence > 0 and regime_confidence < self.low_regime_confidence:
            severity = max(severity, 1)
            reasons.append("low_regime_confidence")

        result = self._build_assessment(
            severity=severity,
            reasons=reasons,
            metrics=summary,
            regime_name=regime_name,
            regime_confidence=regime_confidence,
            divergence_status=divergence_status,
        )
        result["operator_risk_off_enabled"] = bool(operator_override.get("enabled", False))
        result["operator_risk_off_reason"] = str(operator_override.get("reason", "") or "")
        result["operator_risk_off_set_by"] = str(operator_override.get("set_by", "") or "")
        result["operator_risk_off_set_at"] = str(operator_override.get("set_at", "") or "")

        self.stats["evaluations"] += 1
        if result["status"] == "caution":
            self.stats["caution"] += 1
        elif result["status"] == "risk_off":
            self.stats["risk_off"] += 1
        elif result["status"] == "blocked":
            self.stats["blocked"] += 1
        self._last_evaluation = result
        self.stats["last_evaluation"] = result
        return result

    def get_stats(self) -> Dict:
        if not self._last_evaluation:
            self.evaluate()
        else:
            self._refresh()
        return {
            **dict(self.stats),
            "global_status": self._last_evaluation.get("status", "warming_up"),
            "global_reasons": list(self._last_evaluation.get("reasons", []) or []),
            "summary": dict((self._last_evaluation or {}).get("metrics", self._summary) or {}),
        }

    def get_dashboard_payload(self) -> Dict:
        if not self._last_evaluation:
            self.evaluate()
        else:
            self._refresh()
        return {
            **self.get_stats(),
            "runtime": dict(self._last_evaluation or {}),
        }
