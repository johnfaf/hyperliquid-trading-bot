"""
Portfolio rotation helpers for paper trading.

This keeps the position cap "soft" without removing portfolio discipline:
new candidates can open directly while normal capacity is available, high-
conviction names can use reserved slots, and once the book is full the bot
only rotates when a materially better idea appears.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config


@dataclass(frozen=True)
class RotationDecision:
    action: str
    reason: str
    candidate_score: float
    incumbent_score: float = 0.0
    replacement_trade_id: Optional[int] = None


class PortfolioRotationManager:
    """Scores candidates vs incumbents and decides whether to open, rotate, or skip."""

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.target_positions = max(
            1, int(cfg.get("target_positions", config.PORTFOLIO_TARGET_POSITIONS))
        )
        self.hard_max_positions = max(
            self.target_positions,
            int(cfg.get("hard_max_positions", config.PORTFOLIO_HARD_MAX_POSITIONS)),
        )
        self.reserved_high_conviction_slots = max(
            0,
            min(
                self.target_positions,
                int(
                    cfg.get(
                        "reserved_high_conviction_slots",
                        config.PORTFOLIO_RESERVED_HIGH_CONVICTION_SLOTS,
                    )
                ),
            ),
        )
        self.high_conviction_threshold = float(
            cfg.get(
                "high_conviction_threshold",
                config.PORTFOLIO_HIGH_CONVICTION_THRESHOLD,
            )
        )
        self.min_hold_minutes = max(
            0, int(cfg.get("min_hold_minutes", config.PORTFOLIO_MIN_HOLD_MINUTES))
        )
        self.replacement_threshold = float(
            cfg.get(
                "replacement_threshold",
                config.PORTFOLIO_REPLACEMENT_THRESHOLD,
            )
        )
        self.max_replacements_per_cycle = max(
            0,
            int(
                cfg.get(
                    "max_replacements_per_cycle",
                    config.PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE,
                )
            ),
        )

        # Heuristic penalties/bonuses for stale, crowded, or winning incumbents.
        self._age_decay_start_minutes = 6 * 60
        self._age_decay_full_minutes = 24 * 60
        self._crowding_penalty = 0.04
        self._stale_flat_penalty = 0.05
        self._winner_bonus = 0.05
        self._winner_threshold = 0.02
        self._regime_alignment_bonus = 0.08
        self._pnl_weight = 0.35
        self._source_accuracy_weight = 0.10

    def candidate_score(self, signal, regime_data: Optional[Dict] = None) -> float:
        """Score a new signal on the same scale as open positions."""
        score = self._clamp(float(getattr(signal, "confidence", 0.0)), 0.0, 1.0)
        source_accuracy = self._clamp(float(getattr(signal, "source_accuracy", 0.0)), 0.0, 1.0)
        score += source_accuracy * self._source_accuracy_weight
        score += self._alignment_bonus(
            side=self._side_value(getattr(signal, "side", "")),
            regime_data=regime_data,
        )
        return round(score, 4)

    def position_score(
        self,
        trade: Dict,
        open_positions: List[Dict],
        regime_data: Optional[Dict] = None,
    ) -> float:
        """Score an incumbent position: confidence + regime/pnl - stale/crowding."""
        metadata = self._metadata(trade)
        score = self._clamp(float(metadata.get("confidence", 0.35) or 0.35), 0.0, 1.0)
        score += (
            self._clamp(float(metadata.get("source_accuracy", 0.0) or 0.0), 0.0, 1.0)
            * self._source_accuracy_weight
        )
        score += self._alignment_bonus(trade.get("side", ""), regime_data)

        return_pct = self._return_pct(trade)
        score += self._clamp(return_pct, -0.20, 0.20) * self._pnl_weight

        age_minutes = self._age_minutes(trade.get("opened_at", ""))
        score -= self._age_decay(age_minutes)
        if age_minutes >= self._age_decay_start_minutes and abs(return_pct) < 0.01:
            score -= self._stale_flat_penalty

        duplicates = sum(1 for pos in open_positions if pos.get("coin") == trade.get("coin"))
        if duplicates > 1:
            score -= (duplicates - 1) * self._crowding_penalty

        if return_pct > self._winner_threshold:
            score += self._winner_bonus

        return round(score, 4)

    def decide(
        self,
        signal,
        open_positions: List[Dict],
        regime_data: Optional[Dict] = None,
        replacements_used: int = 0,
    ) -> RotationDecision:
        """Decide whether a candidate can open directly, should rotate, or should wait."""
        candidate_score = self.candidate_score(signal, regime_data=regime_data)
        open_count = len(open_positions)
        normal_slot_limit = max(self.target_positions - self.reserved_high_conviction_slots, 0)

        if open_count < normal_slot_limit:
            return RotationDecision(
                action="open",
                reason=f"normal capacity available ({open_count}/{normal_slot_limit})",
                candidate_score=candidate_score,
            )

        if open_count < self.target_positions:
            if self.is_high_conviction(candidate_score):
                return RotationDecision(
                    action="open",
                    reason=(
                        f"reserved high-conviction slot used "
                        f"({open_count}/{self.target_positions})"
                    ),
                    candidate_score=candidate_score,
                )
            return RotationDecision(
                action="reject",
                reason=(
                    "reserved slots held for stronger arrivals "
                    f"(score {candidate_score:.2f} < {self.high_conviction_threshold:.2f})"
                ),
                candidate_score=candidate_score,
            )

        if replacements_used >= self.max_replacements_per_cycle:
            return RotationDecision(
                action="reject",
                reason=(
                    f"replacement budget exhausted "
                    f"({replacements_used}/{self.max_replacements_per_cycle})"
                ),
                candidate_score=candidate_score,
            )

        eligible = []
        for trade in open_positions:
            age_minutes = self._age_minutes(trade.get("opened_at", ""))
            if age_minutes < self.min_hold_minutes:
                continue
            score = self.position_score(trade, open_positions, regime_data=regime_data)
            eligible.append((score, trade))

        if not eligible:
            return RotationDecision(
                action="reject",
                reason=f"all incumbents are inside the {self.min_hold_minutes}m hold window",
                candidate_score=candidate_score,
            )

        weakest_score, weakest_trade = min(eligible, key=lambda item: item[0])
        improvement = candidate_score - weakest_score
        if improvement <= self.replacement_threshold:
            return RotationDecision(
                action="reject",
                reason=(
                    f"improvement {improvement:.2f} <= switch threshold "
                    f"{self.replacement_threshold:.2f}"
                ),
                candidate_score=candidate_score,
                incumbent_score=weakest_score,
                replacement_trade_id=weakest_trade.get("id"),
            )

        return RotationDecision(
            action="replace",
            reason=(
                f"candidate {candidate_score:.2f} beats weakest {weakest_score:.2f} "
                f"by {improvement:.2f}"
            ),
            candidate_score=candidate_score,
            incumbent_score=weakest_score,
            replacement_trade_id=weakest_trade.get("id"),
        )

    def is_high_conviction(self, score: float) -> bool:
        return score >= self.high_conviction_threshold

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _side_value(side) -> str:
        if hasattr(side, "value"):
            return str(side.value).lower()
        return str(side).lower()

    @staticmethod
    def _metadata(trade: Dict) -> Dict:
        metadata = trade.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            try:
                return json.loads(metadata or "{}")
            except Exception:
                return {}
        return {}

    def _return_pct(self, trade: Dict) -> float:
        entry = float(trade.get("entry_price", 0) or 0)
        size = float(trade.get("size", 0) or 0)
        leverage = float(trade.get("leverage", 1) or 1)
        current_price = float(trade.get("current_price", entry) or entry)
        if entry <= 0 or size <= 0:
            return 0.0

        if str(trade.get("side", "")).lower() == "long":
            pnl = (current_price - entry) * size * leverage
        else:
            pnl = (entry - current_price) * size * leverage

        notional = entry * size * leverage
        if notional <= 0:
            return 0.0
        return pnl / notional

    def _alignment_bonus(self, side: str, regime_data: Optional[Dict]) -> float:
        regime = str((regime_data or {}).get("overall_regime", "")).lower()
        side = self._side_value(side)
        if regime in {"bull", "bullish"}:
            return self._regime_alignment_bonus if side == "long" else -self._regime_alignment_bonus
        if regime in {"bear", "bearish", "crash"}:
            return self._regime_alignment_bonus if side == "short" else -self._regime_alignment_bonus
        return 0.0

    def _age_decay(self, age_minutes: float) -> float:
        if age_minutes <= self._age_decay_start_minutes:
            return 0.0
        if age_minutes >= self._age_decay_full_minutes:
            return 0.15

        span = max(self._age_decay_full_minutes - self._age_decay_start_minutes, 1)
        progress = (age_minutes - self._age_decay_start_minutes) / span
        return progress * 0.15

    @staticmethod
    def _age_minutes(opened_at: str) -> float:
        if not opened_at:
            return 0.0

        try:
            ts = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.0

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max((now - ts).total_seconds() / 60.0, 0.0)
