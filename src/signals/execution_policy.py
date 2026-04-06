"""
Execution policy recommendations for live order routing.

Chooses between taker-style market entry and maker-style post-only limit entry
using observed execution quality, adaptive source health, and candidate urgency.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

from src.data import database as db

logger = logging.getLogger(__name__)


class ExecutionPolicyManager:
    """Recommend order-routing policy for entry execution."""

    def __init__(self, config: Optional[Dict] = None, *, adaptive_learning=None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.adaptive_learning = adaptive_learning
        self.lookback_hours = float(cfg.get("lookback_hours", 24.0 * 7))
        self.min_events = int(cfg.get("min_events", 3))
        self.maker_rejection_ceiling = float(cfg.get("maker_rejection_ceiling", 0.18))
        self.maker_fill_floor = float(cfg.get("maker_fill_floor", 0.72))
        self.maker_confidence_ceiling = float(cfg.get("maker_confidence_ceiling", 0.78))
        self.maker_source_quality_floor = float(cfg.get("maker_source_quality_floor", 0.52))
        self.maker_offset_bps = float(cfg.get("maker_offset_bps", 1.5))
        self.max_maker_offset_bps = float(cfg.get("max_maker_offset_bps", 5.0))
        self.min_maker_offset_bps = float(cfg.get("min_maker_offset_bps", 0.5))
        self.maker_timeout_seconds = float(cfg.get("maker_timeout_seconds", 5.0))
        self.fallback_confidence_threshold = float(cfg.get("fallback_confidence_threshold", 0.72))
        self.fallback_source_quality_threshold = float(cfg.get("fallback_source_quality_threshold", 0.62))
        self.market_slippage_multiplier = float(cfg.get("market_slippage_multiplier", 1.0))
        self.maker_slippage_floor_bps = float(cfg.get("maker_slippage_floor_bps", 0.4))
        self.default_market_slippage_bps = float(cfg.get("default_market_slippage_bps", 3.0))
        self.default_execution_role = str(cfg.get("default_execution_role", "taker") or "taker").lower()
        self.urgent_sources = {
            str(item).strip().lower()
            for item in cfg.get(
                "urgent_sources",
                ["options_flow", "polymarket", "liquidation_strategy", "arena_champion"],
            )
        }

        self.stats = {
            "recommendations": 0,
            "market_route_count": 0,
            "maker_route_count": 0,
        }
        self._recent_recommendations: deque = deque(maxlen=50)

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(value, upper))

    def _lookup_execution_quality(self, source_key: str, source: str) -> Dict:
        if source_key:
            profile = db.get_execution_quality_summary(
                source_key=source_key,
                lookback_hours=self.lookback_hours,
            )
            if int(profile.get("total_events", 0) or 0) >= self.min_events:
                return profile
        if source:
            profile = db.get_execution_quality_summary(
                source=source,
                lookback_hours=self.lookback_hours,
            )
            if int(profile.get("total_events", 0) or 0) >= self.min_events:
                return profile
        return {}

    def recommend(
        self,
        *,
        strategy: Dict,
        metadata: Optional[Dict] = None,
        confidence: float,
        source_quality: float,
    ) -> Dict:
        metadata = dict(metadata or {})
        if not self.enabled:
            return {
                "route": "market",
                "execution_route": "market",
                "execution_role": self.default_execution_role,
                "expected_slippage_bps": float(
                    metadata.get("expected_slippage_bps", self.default_market_slippage_bps)
                    or self.default_market_slippage_bps
                ),
                "policy_reason": "disabled",
            }

        source_key = str(strategy.get("source_key", metadata.get("source_key", "")) or "").strip()
        source = str(strategy.get("source", metadata.get("source", "strategy")) or "strategy").strip().lower()
        execution_quality = self._lookup_execution_quality(source_key, source)

        adaptive_profile = {}
        if self.adaptive_learning:
            try:
                adaptive_profile = self.adaptive_learning.get_source_profile(
                    source_key=source_key,
                    source=source,
                ) or {}
            except Exception as exc:
                logger.debug("execution policy adaptive lookup error: %s", exc)

        avg_slippage_bps = float(
            execution_quality.get(
                "avg_realized_slippage_bps",
                metadata.get("expected_slippage_bps", self.default_market_slippage_bps),
            )
            or self.default_market_slippage_bps
        )
        maker_ratio = float(execution_quality.get("maker_ratio", 0.0) or 0.0)
        fill_ratio = float(execution_quality.get("avg_fill_ratio", 1.0) or 1.0)
        rejection_rate = float(execution_quality.get("rejection_rate", 0.0) or 0.0)
        adaptive_status = str(adaptive_profile.get("status", "active") or "active").strip().lower()
        adaptive_health = float(adaptive_profile.get("health_score", 0.5) or 0.5)

        source_urgent = source in self.urgent_sources
        urgency_score = self._clamp(
            0.55 * float(confidence or 0.0)
            + 0.20 * float(source_quality or 0.0)
            + (0.20 if source_urgent else 0.0)
            + (0.10 if adaptive_status == "caution" else 0.0),
            0.0,
            1.0,
        )

        can_use_maker = (
            not source_urgent
            and adaptive_status != "blocked"
            and confidence <= self.maker_confidence_ceiling
            and source_quality >= self.maker_source_quality_floor
            and fill_ratio >= self.maker_fill_floor
            and rejection_rate <= self.maker_rejection_ceiling
            and adaptive_health >= 0.45
        )

        route = "maker_limit" if can_use_maker else "market"
        execution_role = "maker" if can_use_maker else "taker"
        maker_price_offset_bps = self._clamp(
            self.maker_offset_bps + (avg_slippage_bps * 0.12) - (maker_ratio * 0.7),
            self.min_maker_offset_bps,
            self.max_maker_offset_bps,
        )
        expected_slippage_bps = (
            max(self.maker_slippage_floor_bps, avg_slippage_bps * 0.35)
            if can_use_maker
            else max(avg_slippage_bps, self.default_market_slippage_bps) * self.market_slippage_multiplier
        )
        fallback_to_market = bool(
            can_use_maker and (
                confidence >= self.fallback_confidence_threshold
                or source_quality >= self.fallback_source_quality_threshold
                or adaptive_status == "caution"
            )
        )

        reasons = []
        if source_urgent:
            reasons.append("urgent_source")
        if adaptive_status == "blocked":
            reasons.append("adaptive_blocked")
        elif adaptive_status == "caution":
            reasons.append("adaptive_caution")
        if rejection_rate > self.maker_rejection_ceiling:
            reasons.append("high_rejection_rate")
        if fill_ratio < self.maker_fill_floor:
            reasons.append("low_fill_ratio")
        if route == "maker_limit":
            reasons.append("maker_cost_saving")
        else:
            reasons.append("taker_immediacy")

        recommendation = {
            "route": route,
            "execution_route": route,
            "execution_role": execution_role,
            "expected_slippage_bps": round(expected_slippage_bps, 4),
            "maker_price_offset_bps": round(maker_price_offset_bps, 4),
            "maker_timeout_seconds": round(self.maker_timeout_seconds, 2),
            "fallback_to_market": fallback_to_market,
            "limit_tif": "Alo" if can_use_maker else None,
            "policy_reason": ",".join(reasons[:4]),
            "urgency_score": round(urgency_score, 4),
            "historical_maker_ratio": round(maker_ratio, 4),
            "historical_fill_ratio": round(fill_ratio, 4),
            "historical_rejection_rate": round(rejection_rate, 4),
        }

        self.stats["recommendations"] += 1
        if route == "maker_limit":
            self.stats["maker_route_count"] += 1
        else:
            self.stats["market_route_count"] += 1

        self._recent_recommendations.append(
            {
                "source_key": source_key or source,
                "route": route,
                "execution_role": execution_role,
                "urgency_score": recommendation["urgency_score"],
                "policy_reason": recommendation["policy_reason"],
            }
        )
        return recommendation

    def get_stats(self) -> Dict:
        recommendations = max(int(self.stats["recommendations"]), 1)
        return {
            **self.stats,
            "enabled": self.enabled,
            "lookback_hours": self.lookback_hours,
            "maker_route_rate": round(self.stats["maker_route_count"] / recommendations, 4),
            "market_route_rate": round(self.stats["market_route_count"] / recommendations, 4),
            "recent_recommendations": list(self._recent_recommendations)[-10:],
        }
