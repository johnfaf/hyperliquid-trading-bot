"""
Source-level capital allocation and budget governance.

This layer sits after signal generation and portfolio sizing. It uses adaptive
learning profiles plus recent attribution / trade outcomes to scale stronger
sources up, derisk weaker ones, and block degraded sources before they consume
fresh capital.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

import config
from src.core.time_utils import utc_now
from src.data import database as db
from src.signals.signal_schema import build_source_key

logger = logging.getLogger(__name__)


@dataclass
class SourceBudgetDecision:
    source_key: str
    source: str
    status: str
    original_position_pct: float
    position_pct: float
    allocation_multiplier: float
    source_cap_pct: float
    source_exposure_pct: float
    source_headroom_pct: float
    health_score: float
    weight_multiplier: float
    confidence_multiplier: float
    promotion_stage: str = ""
    promotion_multiplier: float = 1.0
    promotion_cap_pct: float = 0.0
    divergence_status: str = ""
    divergence_multiplier: float = 1.0
    capital_status: str = ""
    capital_multiplier: float = 1.0
    blocked: bool = False
    block_reason: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class SourceBudgetAllocator:
    """Govern source budgets using adaptive health, outcomes, and live quality."""

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        adaptive_learning=None,
        divergence_controller=None,
        capital_governor=None,
    ):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lookback_hours = float(cfg.get("lookback_hours", 24 * 30))
        self.refresh_interval_seconds = float(cfg.get("refresh_interval_seconds", 300.0))
        self.min_position_pct = float(cfg.get("min_position_pct", config.PORTFOLIO_MIN_POSITION_PCT))
        self.min_multiplier = float(cfg.get("min_multiplier", 0.10))
        self.max_multiplier = float(cfg.get("max_multiplier", 1.20))
        self.active_multiplier = float(cfg.get("active_multiplier", 1.00))
        self.warming_multiplier = float(cfg.get("warming_multiplier", 0.78))
        self.caution_multiplier = float(cfg.get("caution_multiplier", 0.50))
        self.blocked_multiplier = float(cfg.get("blocked_multiplier", 0.0))
        self.active_cap_pct = float(cfg.get("active_cap_pct", 0.30))
        self.warming_cap_pct = float(cfg.get("warming_cap_pct", 0.16))
        self.caution_cap_pct = float(cfg.get("caution_cap_pct", 0.10))
        self.blocked_cap_pct = float(cfg.get("blocked_cap_pct", 0.0))
        self.min_health_score = float(cfg.get("min_health_score", 0.42))
        self.min_closed_trades = int(cfg.get("min_closed_trades", 3))
        self.return_scale = float(cfg.get("return_scale", 0.04))
        self.live_rejection_ceiling = float(cfg.get("live_rejection_ceiling", 0.25))
        self.live_fill_floor = float(cfg.get("live_fill_floor", 0.60))
        self.block_on_status = bool(cfg.get("block_on_status", True))
        self.divergence_enabled = bool(cfg.get("divergence_enabled", divergence_controller is not None))
        self.capital_governor_enabled = bool(cfg.get("capital_governor_enabled", capital_governor is not None))
        self.promotion_ladder_enabled = bool(cfg.get("promotion_ladder_enabled", True))
        self.adaptive_learning = adaptive_learning
        self.divergence_controller = divergence_controller
        self.capital_governor = capital_governor

        self._last_refresh_at: Optional[datetime] = None
        self._outcome_by_key: Dict[str, Dict] = {}
        self._outcome_by_source: Dict[str, Dict] = {}
        self._attribution_by_key: Dict[str, Dict] = {}
        self._attribution_by_source: Dict[str, Dict] = {}

        self.stats = {
            "enabled": self.enabled,
            "lookback_hours": self.lookback_hours,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "evaluations": 0,
            "blocked": 0,
            "size_reduced": 0,
            "last_refresh_at": None,
            "status_counts": {
                "active": 0,
                "warming_up": 0,
                "caution": 0,
                "blocked": 0,
            },
            "promotion_stage_counts": {
                "blocked": 0,
                "incubating": 0,
                "trial": 0,
                "scaled": 0,
                "full": 0,
            },
            "divergence_blocked": 0,
            "divergence_scaled": 0,
            "capital_blocked": 0,
            "capital_scaled": 0,
            "promotion_blocked": 0,
            "promotion_scaled": 0,
            "last_decision": None,
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
    def _coerce_source_name(source) -> str:
        if hasattr(source, "value"):
            source = source.value
        return str(source or "").strip().lower()

    @staticmethod
    def _coerce_metadata(metadata) -> Dict:
        if isinstance(metadata, dict):
            return dict(metadata)
        if isinstance(metadata, str):
            try:
                import json

                parsed = json.loads(metadata or "{}")
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _resolve_source_identity(self, signal_obj, signal: Optional[Dict]) -> Tuple[str, str]:
        payload = signal or {}
        metadata = self._coerce_metadata(payload.get("metadata", {}))

        source_key = str(
            getattr(signal_obj, "source_key", "") or payload.get("source_key") or payload.get("_source_key") or ""
        ).strip()
        source = self._coerce_source_name(
            getattr(signal_obj, "source", "") or payload.get("source") or metadata.get("source")
        )

        if not source and payload.get("source_trader"):
            source = "copy_trade"

        if not source_key and source == "copy_trade":
            trader = str(payload.get("source_trader", "") or metadata.get("source_trader", "")).strip()
            if trader:
                source_key = f"copy_trade:{trader}"

        if not source_key:
            strategy_type = str(
                getattr(signal_obj, "strategy_type", "")
                or payload.get("strategy_type")
                or payload.get("type")
                or metadata.get("strategy_type", "")
            ).strip()
            trader_address = str(
                payload.get("trader_address")
                or payload.get("source_trader")
                or metadata.get("trader_address")
                or metadata.get("source_trader")
                or ""
            ).strip()
            coin = str(getattr(signal_obj, "coin", "") or payload.get("coin") or "").strip().upper()
            agent_id = str(
                payload.get("agent_id") or metadata.get("agent_id") or getattr(signal_obj, "agent_id", "") or ""
            ).strip()
            if source:
                source_key = build_source_key(
                    source,
                    strategy_type=strategy_type,
                    trader_address=trader_address,
                    coin=coin,
                    agent_id=agent_id,
                )

        if not source:
            source = str(source_key.split(":", 1)[0] if source_key else "unknown").strip().lower() or "unknown"

        return source_key, source

    def _refresh_summaries(self, force: bool = False) -> None:
        if not self.enabled:
            return

        now = utc_now()
        if (
            not force
            and self._last_refresh_at
            and (now - self._last_refresh_at).total_seconds() < self.refresh_interval_seconds
        ):
            return

        try:
            outcome_rows = db.get_source_trade_outcome_summary(lookback_hours=self.lookback_hours)
        except Exception as exc:
            logger.debug("source allocator outcome refresh error: %s", exc)
            outcome_rows = []

        try:
            attribution_rows = db.get_source_attribution_summary(
                limit_cycles=getattr(config, "EXPERIMENT_REPORT_LIMIT_CYCLES", 120),
                lookback_hours=self.lookback_hours,
            )
        except Exception as exc:
            logger.debug("source allocator attribution refresh error: %s", exc)
            attribution_rows = []

        self._outcome_by_key, self._outcome_by_source = self._index_outcomes(outcome_rows)
        self._attribution_by_key, self._attribution_by_source = self._index_attribution(attribution_rows)
        self._last_refresh_at = now
        self.stats["last_refresh_at"] = now.isoformat()

    def _index_outcomes(self, rows: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        by_key = {}
        by_source = {}

        for row in rows or []:
            key = str(row.get("source_key", "") or row.get("source") or "unknown").strip() or "unknown"
            source = self._coerce_source_name(row.get("source"))
            by_key[key] = dict(row)

            bucket = by_source.setdefault(
                source or "unknown",
                {
                    "source": source or "unknown",
                    "open_trades": 0,
                    "closed_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "realized_pnl": 0.0,
                    "_return_total": 0.0,
                    "avg_return_pct": 0.0,
                },
            )
            bucket["open_trades"] += int(row.get("open_trades", 0) or 0)
            bucket["closed_trades"] += int(row.get("closed_trades", 0) or 0)
            bucket["winning_trades"] += int(row.get("winning_trades", 0) or 0)
            bucket["losing_trades"] += int(row.get("losing_trades", 0) or 0)
            bucket["realized_pnl"] += self._safe_float(row.get("realized_pnl"), 0.0)
            bucket["_return_total"] += self._safe_float(row.get("avg_return_pct"), 0.0) * int(
                row.get("closed_trades", 0) or 0
            )

        for bucket in by_source.values():
            closed = max(int(bucket.get("closed_trades", 0) or 0), 0)
            if closed > 0:
                bucket["avg_return_pct"] = round(bucket["_return_total"] / closed, 4)
            bucket["realized_pnl"] = round(bucket["realized_pnl"], 2)
            bucket.pop("_return_total", None)

        return by_key, by_source

    def _index_attribution(self, rows: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        by_key = {}
        by_source = {}

        for row in rows or []:
            key = str(row.get("source_key", "") or row.get("source") or "unknown").strip() or "unknown"
            source = self._coerce_source_name(row.get("source"))
            by_key[key] = dict(row)

            bucket = by_source.setdefault(
                source or "unknown",
                {
                    "source": source or "unknown",
                    "candidate_count": 0,
                    "selected_count": 0,
                    "blocked_count": 0,
                    "overflow_count": 0,
                    "paper_open_count": 0,
                    "paper_closed_count": 0,
                    "paper_realized_pnl": 0.0,
                    "live_events": 0,
                    "_live_rejection_total": 0.0,
                    "_live_fill_total": 0.0,
                },
            )
            bucket["candidate_count"] += int(row.get("candidate_count", 0) or 0)
            bucket["selected_count"] += int(row.get("selected_count", 0) or 0)
            bucket["blocked_count"] += int(row.get("blocked_count", 0) or 0)
            bucket["overflow_count"] += int(row.get("overflow_count", 0) or 0)
            bucket["paper_open_count"] += int(row.get("paper_open_count", 0) or 0)
            bucket["paper_closed_count"] += int(row.get("paper_closed_count", 0) or 0)
            bucket["paper_realized_pnl"] += self._safe_float(row.get("paper_realized_pnl"), 0.0)
            live_events = int(row.get("live_events", 0) or 0)
            bucket["live_events"] += live_events
            bucket["_live_rejection_total"] += self._safe_float(row.get("live_rejection_rate"), 0.0) * live_events
            bucket["_live_fill_total"] += self._safe_float(row.get("live_avg_fill_ratio"), 0.0) * live_events

        for bucket in by_source.values():
            live_events = max(int(bucket.get("live_events", 0) or 0), 0)
            if live_events > 0:
                bucket["live_rejection_rate"] = round(bucket["_live_rejection_total"] / live_events, 4)
                bucket["live_avg_fill_ratio"] = round(bucket["_live_fill_total"] / live_events, 4)
            else:
                bucket["live_rejection_rate"] = 0.0
                bucket["live_avg_fill_ratio"] = 0.0
            bucket["paper_realized_pnl"] = round(bucket["paper_realized_pnl"], 2)
            bucket.pop("_live_rejection_total", None)
            bucket.pop("_live_fill_total", None)

        return by_key, by_source

    def _get_profile(self, source_key: str, source: str) -> Dict:
        if not self.adaptive_learning:
            return {}
        try:
            return self.adaptive_learning.get_source_profile(source_key=source_key, source=source) or {}
        except Exception as exc:
            logger.debug("source allocator profile error for %s: %s", source_key or source, exc)
            return {}

    def _get_promotion_assessment(self, profile: Dict) -> Dict:
        if not self.promotion_ladder_enabled or not profile:
            return {}
        metadata = self._coerce_metadata(profile.get("metadata", {}))
        explicit_keys = {
            "promotion_stage",
            "promotion_multiplier",
            "promotion_cap_pct",
            "promotion_reasons",
        }
        if not any(key in profile or key in metadata for key in explicit_keys):
            return {}
        stage = str(
            profile.get("promotion_stage", metadata.get("promotion_stage", "trial")) or "trial"
        ).strip().lower() or "trial"
        multiplier = self._safe_float(
            profile.get("promotion_multiplier", metadata.get("promotion_multiplier", 1.0)),
            1.0,
        )
        cap_pct = self._safe_float(
            profile.get("promotion_cap_pct", metadata.get("promotion_cap_pct", 0.0)),
            0.0,
        )
        reasons = list(
            profile.get("promotion_reasons", metadata.get("promotion_reasons", [])) or []
        )
        blocked = stage == "blocked" or multiplier <= 0 or cap_pct <= 0
        return {
            "stage": stage,
            "multiplier": multiplier,
            "cap_pct": cap_pct,
            "blocked": blocked,
            "reasons": reasons,
        }

    def _get_divergence_assessment(self, source_key: str, source: str) -> Dict:
        if not self.divergence_enabled or not self.divergence_controller:
            return {}
        try:
            return self.divergence_controller.evaluate(source_key=source_key, source=source) or {}
        except Exception as exc:
            logger.debug("source allocator divergence error for %s: %s", source_key or source, exc)
            return {}

    def _get_capital_assessment(self) -> Dict:
        if not self.capital_governor_enabled or not self.capital_governor:
            return {}
        try:
            return self.capital_governor.evaluate() or {}
        except Exception as exc:
            logger.debug("source allocator capital governor error: %s", exc)
            return {}

    def _status_defaults(self, status: str) -> Tuple[float, float]:
        normalized = str(status or "warming_up").strip().lower() or "warming_up"
        multipliers = {
            "active": self.active_multiplier,
            "warming_up": self.warming_multiplier,
            "caution": self.caution_multiplier,
            "blocked": self.blocked_multiplier,
        }
        caps = {
            "active": self.active_cap_pct,
            "warming_up": self.warming_cap_pct,
            "caution": self.caution_cap_pct,
            "blocked": self.blocked_cap_pct,
        }
        return multipliers.get(normalized, self.warming_multiplier), caps.get(normalized, self.warming_cap_pct)

    def _estimate_allocation_multiplier(self, profile: Dict, outcome: Dict, attribution: Dict, reasons: List[str]) -> float:
        status = str(profile.get("status", "warming_up") or "warming_up").strip().lower() or "warming_up"
        allocation, _ = self._status_defaults(status)
        health_score = self._safe_float(profile.get("health_score"), 0.50 if not profile else 0.0)
        weight_multiplier = self._safe_float(profile.get("weight_multiplier"), 1.0)
        confidence_multiplier = self._safe_float(profile.get("confidence_multiplier"), 1.0)

        allocation *= max(self.min_multiplier, min(weight_multiplier, self.max_multiplier))
        allocation *= max(self.min_multiplier, min(confidence_multiplier, self.max_multiplier))

        if profile:
            allocation *= max(0.35, min(1.15, 0.55 + health_score))
            if health_score < self.min_health_score:
                reasons.append("low_health_score")
                allocation *= 0.70
        else:
            reasons.append("no_profile_history")
            allocation *= 0.85

        closed_trades = int(outcome.get("closed_trades", 0) or 0)
        if closed_trades >= self.min_closed_trades:
            winning_trades = int(outcome.get("winning_trades", 0) or 0)
            win_rate = winning_trades / max(closed_trades, 1)
            avg_return_pct = self._safe_float(outcome.get("avg_return_pct"), 0.0)
            realized_pnl = self._safe_float(outcome.get("realized_pnl"), 0.0)

            allocation *= max(0.45, min(1.20, 1.0 + max(-0.35, min(0.25, avg_return_pct / max(self.return_scale, 1e-8)))))
            allocation *= max(0.85, min(1.12, 1.0 + ((win_rate - 0.5) * 0.5)))

            if realized_pnl < 0:
                reasons.append("negative_realized_pnl")
        else:
            reasons.append("limited_closed_trade_history")
            allocation *= 0.90

        rejection_rate = self._safe_float(attribution.get("live_rejection_rate"), 0.0)
        if rejection_rate > self.live_rejection_ceiling:
            reasons.append("live_rejection_penalty")
            allocation *= max(0.45, 1.0 - rejection_rate)

        fill_ratio = self._safe_float(attribution.get("live_avg_fill_ratio"), 0.0)
        if fill_ratio > 0 and fill_ratio < self.live_fill_floor:
            reasons.append("live_fill_penalty")
            allocation *= max(0.55, fill_ratio / max(self.live_fill_floor, 1e-8))

        return max(0.0, min(allocation, self.max_multiplier))

    def _position_pct(self, position: Dict, account_balance: float) -> float:
        if account_balance <= 0:
            return 0.0

        metadata = self._coerce_metadata(position.get("metadata", {}))
        explicit_pct = self._safe_float(position.get("position_pct", metadata.get("position_pct")), 0.0)
        if explicit_pct > 0:
            return explicit_pct

        size = abs(self._safe_float(position.get("size", position.get("szi")), 0.0))
        entry_price = self._safe_float(position.get("entry_price", position.get("entryPx")), 0.0)
        if size > 0 and entry_price > 0:
            return (size * entry_price) / account_balance
        return 0.0

    def _source_matches(self, position: Dict, source_key: str, source: str) -> bool:
        metadata = self._coerce_metadata(position.get("metadata", {}))
        position_source_key = str(position.get("source_key") or metadata.get("source_key") or "").strip()
        position_source = self._coerce_source_name(position.get("source") or metadata.get("source"))
        if source_key and position_source_key == source_key:
            return True
        return bool(source and position_source == source)

    def _source_exposure_pct(
        self,
        open_positions: List[Dict],
        *,
        source_key: str,
        source: str,
        account_balance: float,
    ) -> float:
        total = 0.0
        for position in open_positions or []:
            if not self._source_matches(position, source_key, source):
                continue
            total += self._position_pct(position, account_balance)
        return round(total, 6)

    def evaluate(
        self,
        signal_obj,
        *,
        signal: Optional[Dict] = None,
        open_positions: Optional[List[Dict]] = None,
        account_balance: float = 0.0,
    ) -> SourceBudgetDecision:
        source_key, source = self._resolve_source_identity(signal_obj, signal)
        original_position_pct = max(self._safe_float(getattr(signal_obj, "position_pct", 0.0), 0.0), 0.0)
        reasons: List[str] = []

        if not self.enabled:
            decision = SourceBudgetDecision(
                source_key=source_key,
                source=source or "unknown",
                status="disabled",
                original_position_pct=round(original_position_pct, 6),
                position_pct=round(original_position_pct, 6),
                allocation_multiplier=1.0,
                source_cap_pct=1.0,
                source_exposure_pct=0.0,
                source_headroom_pct=1.0,
                health_score=0.0,
                weight_multiplier=1.0,
                confidence_multiplier=1.0,
            )
            self.stats["last_decision"] = decision.to_dict()
            return decision

        self._refresh_summaries()

        profile = self._get_profile(source_key, source)
        status = str(profile.get("status", "warming_up") or "warming_up").strip().lower() or "warming_up"
        health_score = self._safe_float(profile.get("health_score"), 0.50 if not profile else 0.0)
        weight_multiplier = self._safe_float(profile.get("weight_multiplier"), 1.0)
        confidence_multiplier = self._safe_float(profile.get("confidence_multiplier"), 1.0)
        promotion = self._get_promotion_assessment(profile)
        promotion_stage = str(promotion.get("stage", "") or "")
        promotion_multiplier = self._safe_float(promotion.get("multiplier"), 1.0)
        promotion_cap_pct = self._safe_float(promotion.get("cap_pct"), 0.0)
        _, source_cap_pct = self._status_defaults(status)
        if promotion and promotion_cap_pct > 0:
            source_cap_pct = min(source_cap_pct, promotion_cap_pct)
        outcome = self._outcome_by_key.get(source_key) or self._outcome_by_source.get(source, {})
        attribution = self._attribution_by_key.get(source_key) or self._attribution_by_source.get(source, {})
        allocation_multiplier = self._estimate_allocation_multiplier(profile, outcome, attribution, reasons)
        if promotion and not bool(promotion.get("blocked", False)):
            allocation_multiplier *= promotion_multiplier
            if promotion_stage in {"incubating", "trial", "scaled"}:
                reasons.append("promotion_ladder_scaled")
        if promotion:
            reasons.extend([reason for reason in promotion.get("reasons", []) or [] if reason])
        source_exposure_pct = self._source_exposure_pct(
            open_positions or [],
            source_key=source_key,
            source=source,
            account_balance=account_balance,
        )
        source_headroom_pct = max(0.0, source_cap_pct - source_exposure_pct)
        adjusted_position_pct = min(original_position_pct * allocation_multiplier, source_headroom_pct)
        divergence = self._get_divergence_assessment(source_key, source)
        divergence_status = str(divergence.get("status", "") or "")
        divergence_multiplier = self._safe_float(divergence.get("multiplier"), 1.0)
        if divergence and not bool(divergence.get("blocked", False)):
            adjusted_position_pct *= divergence_multiplier
            allocation_multiplier *= divergence_multiplier
            if divergence_status == "caution":
                reasons.append("divergence_caution_scaled")
        if divergence:
            reasons.extend([reason for reason in divergence.get("reasons", []) or [] if reason])
        capital = self._get_capital_assessment()
        capital_status = str(capital.get("status", "") or "")
        capital_multiplier = self._safe_float(capital.get("multiplier"), 1.0)
        if capital and not bool(capital.get("blocked", False)):
            adjusted_position_pct *= capital_multiplier
            allocation_multiplier *= capital_multiplier
            if capital_status in {"caution", "risk_off"}:
                reasons.append("capital_governor_scaled")
        if capital:
            reasons.extend([reason for reason in capital.get("reasons", []) or [] if reason])

        blocked = False
        block_reason = ""

        if promotion and bool(promotion.get("blocked", False)):
            blocked = True
            block_reason = "promotion_ladder_blocked"
        elif capital and bool(capital.get("blocked", False)):
            blocked = True
            block_reason = "capital_governor_blocked"
        elif divergence and bool(divergence.get("blocked", False)):
            blocked = True
            block_reason = "divergence_control_blocked"
        elif status == "blocked" and self.block_on_status:
            blocked = True
            block_reason = "adaptive_learning_blocked"
            reasons.append("blocked_status")
        elif source_cap_pct <= 0:
            blocked = True
            block_reason = "source_budget_disabled"
        elif source_headroom_pct <= 0:
            blocked = True
            block_reason = "source_budget_exhausted"
        elif adjusted_position_pct <= 0:
            blocked = True
            block_reason = "no_source_budget_headroom"
        elif adjusted_position_pct < self.min_position_pct:
            if source_headroom_pct < self.min_position_pct:
                blocked = True
                block_reason = "insufficient_source_budget_headroom"
            else:
                adjusted_position_pct = self.min_position_pct

        if adjusted_position_pct < original_position_pct:
            reasons.append("source_budget_scaled")

        decision = SourceBudgetDecision(
            source_key=source_key,
            source=source or "unknown",
            status=status,
            original_position_pct=round(original_position_pct, 6),
            position_pct=round(0.0 if blocked else adjusted_position_pct, 6),
            allocation_multiplier=round(allocation_multiplier, 6),
            source_cap_pct=round(source_cap_pct, 6),
            source_exposure_pct=round(source_exposure_pct, 6),
            source_headroom_pct=round(source_headroom_pct, 6),
            health_score=round(health_score, 6),
            weight_multiplier=round(weight_multiplier, 6),
            confidence_multiplier=round(confidence_multiplier, 6),
            promotion_stage=promotion_stage,
            promotion_multiplier=round(promotion_multiplier, 6),
            promotion_cap_pct=round(promotion_cap_pct, 6),
            divergence_status=divergence_status,
            divergence_multiplier=round(divergence_multiplier, 6),
            capital_status=capital_status,
            capital_multiplier=round(capital_multiplier, 6),
            blocked=blocked,
            block_reason=block_reason,
            reasons=reasons,
        )

        self.stats["evaluations"] += 1
        if blocked:
            self.stats["blocked"] += 1
            if block_reason == "promotion_ladder_blocked":
                self.stats["promotion_blocked"] += 1
            elif block_reason == "capital_governor_blocked":
                self.stats["capital_blocked"] += 1
            elif block_reason == "divergence_control_blocked":
                self.stats["divergence_blocked"] += 1
        if decision.position_pct < decision.original_position_pct:
            self.stats["size_reduced"] += 1
            if promotion_stage in {"incubating", "trial", "scaled"}:
                self.stats["promotion_scaled"] += 1
            if capital_status in {"caution", "risk_off"}:
                self.stats["capital_scaled"] += 1
            if divergence_status == "caution":
                self.stats["divergence_scaled"] += 1
        status_counts = self.stats.setdefault("status_counts", {})
        status_counts[status] = status_counts.get(status, 0) + 1
        stage_counts = self.stats.setdefault("promotion_stage_counts", {})
        normalized_stage = promotion_stage or "trial"
        stage_counts[normalized_stage] = stage_counts.get(normalized_stage, 0) + 1
        self.stats["last_decision"] = decision.to_dict()
        return decision

    def apply_to_signal(
        self,
        signal_obj,
        *,
        signal: Optional[Dict] = None,
        open_positions: Optional[List[Dict]] = None,
        account_balance: float = 0.0,
    ) -> SourceBudgetDecision:
        decision = self.evaluate(
            signal_obj,
            signal=signal,
            open_positions=open_positions,
            account_balance=account_balance,
        )
        if not decision.blocked:
            signal_obj.position_pct = decision.position_pct
        return decision

    def get_stats(self) -> Dict:
        return dict(self.stats)

    def get_dashboard_payload(self, limit: int = 12) -> Dict:
        self._refresh_summaries()

        profiles = []
        known_profiles = {}
        if self.adaptive_learning:
            try:
                self.adaptive_learning.ensure_profiles()
                known_profiles = dict(getattr(self.adaptive_learning, "source_profiles", {}) or {})
            except Exception as exc:
                logger.debug("source allocator dashboard profile error: %s", exc)

        keys = set(known_profiles.keys()) | set(self._outcome_by_key.keys()) | set(self._attribution_by_key.keys())
        for source_key in keys:
            profile = known_profiles.get(source_key, {})
            source = self._coerce_source_name(
                profile.get("source")
                or self._outcome_by_key.get(source_key, {}).get("source")
                or self._attribution_by_key.get(source_key, {}).get("source")
            )
            reasons: List[str] = []
            outcome = self._outcome_by_key.get(source_key, {})
            attribution = self._attribution_by_key.get(source_key, {})
            estimated_multiplier = self._estimate_allocation_multiplier(profile, outcome, attribution, reasons)
            promotion = self._get_promotion_assessment(profile)
            status = str(profile.get("status", "warming_up") or "warming_up").strip().lower() or "warming_up"
            _, cap_pct = self._status_defaults(status)
            if promotion and self._safe_float(promotion.get("cap_pct"), 0.0) > 0:
                cap_pct = min(cap_pct, self._safe_float(promotion.get("cap_pct"), cap_pct))
            profiles.append(
                {
                    "source_key": source_key,
                    "source": source or "unknown",
                    "status": status,
                    "health_score": round(self._safe_float(profile.get("health_score"), 0.0), 4),
                    "allocation_multiplier": round(
                        estimated_multiplier * self._safe_float(promotion.get("multiplier"), 1.0),
                        4,
                    ),
                    "source_cap_pct": round(cap_pct, 4),
                    "promotion_stage": str(promotion.get("stage", "") or ""),
                    "promotion_multiplier": round(
                        self._safe_float(promotion.get("multiplier"), 1.0),
                        4,
                    ),
                    "closed_trades": int(outcome.get("closed_trades", 0) or 0),
                    "realized_pnl": round(self._safe_float(outcome.get("realized_pnl"), 0.0), 2),
                    "avg_return_pct": round(self._safe_float(outcome.get("avg_return_pct"), 0.0), 4),
                    "live_rejection_rate": round(self._safe_float(attribution.get("live_rejection_rate"), 0.0), 4),
                    "live_avg_fill_ratio": round(self._safe_float(attribution.get("live_avg_fill_ratio"), 0.0), 4),
                    "reasons": reasons + list(promotion.get("reasons", []) or []),
                }
            )

        profiles.sort(
            key=lambda item: (
                {"blocked": 0, "caution": 1, "warming_up": 2, "active": 3}.get(item["status"], 4),
                -float(item["health_score"] or 0.0),
                item["source_key"],
            )
        )
        return {
            **self.get_stats(),
            "divergence": (
                self.divergence_controller.get_dashboard_payload(limit=limit)
                if self.divergence_enabled and self.divergence_controller
                else {}
            ),
            "capital_governor": (
                self.capital_governor.get_dashboard_payload()
                if self.capital_governor_enabled and self.capital_governor
                else {}
            ),
            "profiles": profiles[:limit],
        }
