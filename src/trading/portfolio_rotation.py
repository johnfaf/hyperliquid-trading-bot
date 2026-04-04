"""
Portfolio rotation helpers for paper trading.

This keeps the position cap "soft" without removing portfolio discipline:
new candidates can open directly while normal capacity is available, high-
conviction names can use reserved slots, and once the book is full the bot
only rotates when a materially better idea appears.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


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
        self.transaction_cost_weight = float(
            cfg.get(
                "transaction_cost_weight",
                config.PORTFOLIO_TRANSACTION_COST_WEIGHT,
            )
        )
        self.churn_penalty = float(
            cfg.get("churn_penalty", config.PORTFOLIO_CHURN_PENALTY)
        )
        self.expected_slippage_bps = float(
            cfg.get("expected_slippage_bps", config.PORTFOLIO_EXPECTED_SLIPPAGE_BPS)
        )
        self.max_replacements_per_hour = max(
            0,
            int(
                cfg.get(
                    "max_replacements_per_hour",
                    config.PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR,
                )
            ),
        )
        self.max_replacements_per_day = max(
            0,
            int(
                cfg.get(
                    "max_replacements_per_day",
                    config.PORTFOLIO_MAX_REPLACEMENTS_PER_DAY,
                )
            ),
        )
        self.forced_exit_cooldown_minutes = max(
            0,
            int(
                cfg.get(
                    "forced_exit_cooldown_minutes",
                    config.PORTFOLIO_FORCED_EXIT_COOLDOWN_MINUTES,
                )
            ),
        )
        self.round_trip_block_minutes = max(
            0,
            int(
                cfg.get(
                    "round_trip_block_minutes",
                    config.PORTFOLIO_ROUND_TRIP_BLOCK_MINUTES,
                )
            ),
        )
        self.max_coin_exposure_pct = float(
            cfg.get("max_coin_exposure_pct", config.PORTFOLIO_MAX_COIN_EXPOSURE_PCT)
        )
        self.max_side_exposure_pct = float(
            cfg.get("max_side_exposure_pct", config.PORTFOLIO_MAX_SIDE_EXPOSURE_PCT)
        )
        self.max_cluster_exposure_pct = float(
            cfg.get(
                "max_cluster_exposure_pct",
                config.PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT,
            )
        )

        # Heuristic penalties/bonuses for stale, crowded, or winning incumbents.
        self._age_decay_start_minutes = 6 * 60
        self._age_decay_full_minutes = 24 * 60
        self._crowding_penalty = 0.12
        self._stale_flat_penalty = 0.05
        self._winner_bonus = 0.03
        self._winner_threshold = 0.02
        self._regime_alignment_bonus = 0.08
        self._pnl_weight = 0.28
        self._source_accuracy_weight = 0.22
        self._replacement_timestamps: List[datetime] = []
        self._symbol_cooldown_until: Dict[str, datetime] = {}
        self._last_forced_exit: Dict[str, Dict[str, str]] = {}
        self._telemetry = {
            "decisions": 0,
            "open": 0,
            "reject": 0,
            "replace": 0,
            "replacement_count": 0,
            "estimated_churn_cost": 0.0,
            "rejection_reasons": {},
        }
        self._cluster_map = {
            "majors": {"BTC", "ETH"},
            "l1": {"SOL", "AVAX", "APT", "SUI", "INJ", "SEI"},
            "infra": {"ARB", "OP", "LINK"},
            "memes": {"DOGE", "PEPE", "WIF", "BONK"},
        }

    def candidate_score(self, signal, regime_data: Optional[Dict] = None) -> float:
        """Score a new signal on the same scale as open positions."""
        score = self._clamp(float(getattr(signal, "confidence", 0.0)), 0.0, 1.0)
        source_accuracy = self._clamp(float(getattr(signal, "source_accuracy", 0.0)), 0.0, 1.0)
        score += source_accuracy * self._source_accuracy_weight
        score += self._alignment_bonus(
            side=self._side_value(getattr(signal, "side", "")),
            regime_data=regime_data,
        )
        # CRIT-FIX HIGH-3: clamp to [0.0, 1.0] — source_accuracy_weight (+0.22)
        # and alignment_bonus (+0.08) can push score above 1.0, breaking threshold
        # comparisons and is_high_conviction() checks.
        return round(self._clamp(score, 0.0, 1.0), 4)

    def position_score(
        self,
        trade: Dict,
        open_positions: List[Dict],
        regime_data: Optional[Dict] = None,
    ) -> float:
        """Score an incumbent position on a normalized scale (0..1)."""
        metadata = self._metadata(trade)
        confidence = self._clamp(float(metadata.get("confidence", 0.35) or 0.35), 0.0, 1.0)
        source_quality = self._clamp(float(metadata.get("source_accuracy", 0.0) or 0.0), 0.0, 1.0)
        alignment_bonus = self._alignment_bonus(trade.get("side", ""), regime_data)
        alignment_score = self._clamp(0.5 + alignment_bonus, 0.0, 1.0)

        unrealized_pct = self._return_pct(trade)
        realized_pct = float(metadata.get("realized_pnl_pct", 0.0) or 0.0)
        pnl_state = unrealized_pct * 0.65 + realized_pct * 0.35
        pnl_state_score = self._normalize_signed(pnl_state, cap=0.15)

        age_minutes = self._age_minutes(trade.get("opened_at", ""))
        age_penalty = self._age_decay(age_minutes)
        if age_minutes >= self._age_decay_start_minutes and abs(unrealized_pct) < 0.01:
            age_penalty += self._stale_flat_penalty
        freshness_score = self._clamp(1.0 - age_penalty, 0.0, 1.0)

        duplicates = sum(1 for pos in open_positions if pos.get("coin") == trade.get("coin"))
        crowding_penalty = max(duplicates - 1, 0) * self._crowding_penalty
        crowding_score = self._clamp(1.0 - crowding_penalty, 0.0, 1.0)

        regime_drift_penalty = self._regime_drift_penalty(
            trade_side=trade.get("side", ""),
            entry_regime=str(metadata.get("regime", "") or ""),
            regime_data=regime_data,
        )
        regime_stability = self._clamp(1.0 - regime_drift_penalty, 0.0, 1.0)

        score = (
            confidence * 0.22
            + source_quality * 0.22
            + pnl_state_score * self._pnl_weight
            + alignment_score * 0.12
            + freshness_score * 0.08
            + regime_stability * 0.05
            + crowding_score * 0.03
        )
        if pnl_state > self._winner_threshold:
            score += self._winner_bonus
        score = self._clamp(score, 0.0, 1.0)

        return round(score, 4)

    def decide(
        self,
        signal,
        open_positions: List[Dict],
        regime_data: Optional[Dict] = None,
        replacements_used: int = 0,
    ) -> RotationDecision:
        """Decide whether a candidate can open directly, should rotate, or should wait."""
        now = datetime.now(timezone.utc)
        self._cleanup_guardrail_state(now)
        candidate_score = self.candidate_score(signal, regime_data=regime_data)
        open_count = len(open_positions)
        normal_slot_limit = max(self.target_positions - self.reserved_high_conviction_slots, 0)

        blocked, block_reason = self._guardrail_block_reason(signal, now)
        if blocked:
            decision = RotationDecision(
                action="reject",
                reason=block_reason,
                candidate_score=candidate_score,
            )
            self._record_decision(decision)
            return decision

        if open_count < normal_slot_limit:
            decision = RotationDecision(
                action="open",
                reason=f"normal capacity available ({open_count}/{normal_slot_limit})",
                candidate_score=candidate_score,
            )
            self._record_decision(decision)
            return decision

        if open_count < self.target_positions:
            if self.is_high_conviction(candidate_score):
                decision = RotationDecision(
                    action="open",
                    reason=(
                        f"reserved high-conviction slot used "
                        f"({open_count}/{self.target_positions})"
                    ),
                    candidate_score=candidate_score,
                )
                self._record_decision(decision)
                return decision
            decision = RotationDecision(
                action="reject",
                reason=(
                    "reserved slots held for stronger arrivals "
                    f"(score {candidate_score:.2f} < {self.high_conviction_threshold:.2f})"
                ),
                candidate_score=candidate_score,
            )
            self._record_decision(decision)
            return decision

        if replacements_used >= self.max_replacements_per_cycle:
            decision = RotationDecision(
                action="reject",
                reason=(
                    f"replacement budget exhausted "
                    f"({replacements_used}/{self.max_replacements_per_cycle})"
                ),
                candidate_score=candidate_score,
            )
            self._record_decision(decision)
            return decision

        eligible = []
        for trade in open_positions:
            age_minutes = self._age_minutes(trade.get("opened_at", ""))
            if age_minutes < self.min_hold_minutes:
                continue
            score = self.position_score(trade, open_positions, regime_data=regime_data)
            eligible.append((score, trade))

        if not eligible:
            decision = RotationDecision(
                action="reject",
                reason=f"all incumbents are inside the {self.min_hold_minutes}m hold window",
                candidate_score=candidate_score,
            )
            self._record_decision(decision)
            return decision

        for incumbent_score, incumbent_trade in sorted(eligible, key=lambda item: item[0]):
            exposure_blocked, exposure_reason = self._would_worsen_concentration(
                signal, open_positions, incumbent_trade
            )
            if exposure_blocked:
                continue

            required_improvement = self._replacement_threshold_with_costs(
                victim_trade=incumbent_trade,
                candidate_signal=signal,
            )
            improvement = candidate_score - incumbent_score
            if improvement <= required_improvement:
                continue

            decision = RotationDecision(
                action="replace",
                reason=(
                    f"candidate {candidate_score:.2f} beats incumbent {incumbent_score:.2f} "
                    f"by {improvement:.2f} > required {required_improvement:.2f}"
                ),
                candidate_score=candidate_score,
                incumbent_score=incumbent_score,
                replacement_trade_id=incumbent_trade.get("id"),
            )
            self._record_decision(decision)
            return decision

        weakest_score, weakest_trade = min(eligible, key=lambda item: item[0])
        blocked_concentration, reason = self._would_worsen_concentration(signal, open_positions, weakest_trade)
        if blocked_concentration:
            decision = RotationDecision(
                action="reject",
                reason=reason,
                candidate_score=candidate_score,
                incumbent_score=weakest_score,
                replacement_trade_id=weakest_trade.get("id"),
            )
            self._record_decision(decision)
            return decision
        required = self._replacement_threshold_with_costs(weakest_trade, signal)
        improvement = candidate_score - weakest_score
        decision = RotationDecision(
            action="reject",
            reason=(
                f"improvement {improvement:.2f} <= required threshold {required:.2f} "
                "(includes fees/slippage/churn)"
            ),
            candidate_score=candidate_score,
            incumbent_score=weakest_score,
            replacement_trade_id=weakest_trade.get("id"),
        )
        self._record_decision(decision)
        return decision

    def register_replacement(
        self,
        replaced_trade: Dict,
        new_coin: str,
        new_side: str,
    ) -> None:
        """Track replacement event for hourly/day caps and cooldown guardrails."""
        now = datetime.now(timezone.utc)
        self._replacement_timestamps.append(now)

        coin = str(replaced_trade.get("coin", "")).upper()
        side = self._side_value(replaced_trade.get("side", ""))
        if coin:
            cooldown_until = now
            if self.forced_exit_cooldown_minutes > 0:
                from datetime import timedelta
                cooldown_until = now + timedelta(minutes=self.forced_exit_cooldown_minutes)
            self._symbol_cooldown_until[coin] = cooldown_until
            self._last_forced_exit[coin] = {
                "at": now.isoformat(),
                "side": side,
                "entered_coin": str(new_coin or "").upper(),
                "entered_side": self._side_value(new_side),
            }
        self._telemetry["replacement_count"] += 1
        churn_cost = self._replacement_threshold_with_costs(replaced_trade, type("S", (), {"coin": new_coin, "side": new_side})())
        self._telemetry["estimated_churn_cost"] += float(max(churn_cost, 0.0))
        self._cleanup_guardrail_state(now)

    def record_dry_run_replacement_skip(
        self,
        decision: RotationDecision,
        reason_key: str = "dry_run_rotation_disabled",
    ) -> None:
        """Telemetry hook when replacement was proposed but intentionally skipped."""
        if decision.action != "replace":
            return
        reasons = self._telemetry["rejection_reasons"]
        key = reason_key or "dry_run_rotation_disabled"
        reasons[key] = reasons.get(key, 0) + 1

    def should_bypass_reject_in_shadow_mode(
        self,
        decision: RotationDecision,
        open_positions_count: int,
    ) -> bool:
        """
        Shadow mode should not alter actual trading decisions when capacity exists.

        If rotation telemetry-only mode is active and the book still has open
        slots, a rotation-specific reject should not block the underlying trade.
        """
        return (
            decision.action == "reject"
            and open_positions_count < self.target_positions
        )

    def get_stats(self) -> Dict:
        """Expose rotation telemetry for dashboards and reports."""
        return {
            "decisions": int(self._telemetry["decisions"]),
            "open": int(self._telemetry["open"]),
            "reject": int(self._telemetry["reject"]),
            "replace": int(self._telemetry["replace"]),
            "replacement_count": int(self._telemetry["replacement_count"]),
            "estimated_churn_cost": round(float(self._telemetry["estimated_churn_cost"]), 4),
            "rejection_reasons": dict(self._telemetry["rejection_reasons"]),
            "replacements_last_hour": self._count_recent_replacements(hours=1),
            "replacements_last_day": self._count_recent_replacements(hours=24),
        }

    def is_high_conviction(self, score: float) -> bool:
        return score >= self.high_conviction_threshold

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _normalize_signed(value: float, cap: float) -> float:
        if cap <= 0:
            return 0.5
        return max(0.0, min(1.0, 0.5 + (value / (2 * cap))))

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
        # MED-NOTE MED-3: leverage cancels identically in numerator and denominator
        # (pnl = delta * size * lev; notional = entry * size * lev), so this
        # simplifies to (current - entry) / entry — a pure price return.
        # All positions regardless of leverage receive an equivalent score for
        # the same price move; no leverage bias in rotation decisions.
        return pnl / notional

    def _alignment_bonus(self, side: str, regime_data: Optional[Dict]) -> float:
        regime = str((regime_data or {}).get("overall_regime", "")).lower()
        side = self._side_value(side)
        if regime in {"bull", "bullish"}:
            return self._regime_alignment_bonus if side == "long" else -self._regime_alignment_bonus
        if regime in {"bear", "bearish", "crash"}:
            return self._regime_alignment_bonus if side == "short" else -self._regime_alignment_bonus
        return 0.0

    def _regime_drift_penalty(self, trade_side: str, entry_regime: str, regime_data: Optional[Dict]) -> float:
        current_regime = str((regime_data or {}).get("overall_regime", "")).lower()
        entry_regime = str(entry_regime or "").lower()
        if not current_regime:
            return 0.0
        if entry_regime and entry_regime != current_regime:
            return 0.10
        align_now = self._alignment_bonus(trade_side, regime_data)
        if align_now < 0:
            return 0.12
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

    def _replacement_threshold_with_costs(self, victim_trade: Dict, candidate_signal) -> float:
        fee_rate = max(config.PAPER_TRADING_TAKER_FEE_BPS, 0.0) / 10_000
        slippage_rate = max(self.expected_slippage_bps, 0.0) / 10_000
        roundtrip_cost_rate = (fee_rate + slippage_rate) * 2.0
        cost_penalty = roundtrip_cost_rate * self.transaction_cost_weight
        return max(self.replacement_threshold + self.churn_penalty + cost_penalty, self.replacement_threshold)

    def _trade_notional(self, trade: Dict) -> float:
        entry = float(trade.get("entry_price", 0) or 0)
        current = float(trade.get("current_price", entry) or entry)
        size = float(trade.get("size", 0) or 0)
        leverage = float(trade.get("leverage", 1) or 1)
        px = current if current > 0 else entry
        return max(px * size * max(leverage, 1.0), 0.0)

    def _candidate_notional(self, signal, fallback: float) -> float:
        # LOW-FIX LOW-5: removed the erroneous * 100.0 multiplier.
        # position_pct is a fraction (e.g. 0.08), not a percentage integer.
        # Without account balance we cannot compute an accurate absolute notional,
        # so we use the victim's notional (fallback) as the candidate proxy —
        # in a rotation the replacement is approximately the same size as the trade
        # it replaces, so the concentration check remains correct.
        return max(fallback, 1.0)

    def _cluster_for_coin(self, coin: str) -> str:
        coin = str(coin or "").upper()
        for cluster, coins in self._cluster_map.items():
            if coin in coins:
                return cluster
        return coin

    def _would_worsen_concentration(
        self, signal, open_positions: List[Dict], victim_trade: Dict
    ) -> Tuple[bool, str]:
        if not open_positions:
            return False, ""
        candidate_coin = str(getattr(signal, "coin", "")).upper()
        candidate_side = self._side_value(getattr(signal, "side", ""))
        victim_id = victim_trade.get("id")

        before_book = list(open_positions)
        after_book = [t for t in open_positions if t.get("id") != victim_id]

        victim_notional = max(self._trade_notional(victim_trade), 1.0)
        synthetic_candidate = {
            "coin": candidate_coin,
            "side": candidate_side,
            "entry_price": victim_trade.get("current_price", victim_trade.get("entry_price", 0)),
            "current_price": victim_trade.get("current_price", victim_trade.get("entry_price", 0)),
            "size": 1.0,
            "leverage": 1.0,
        }
        cand_notional = self._candidate_notional(signal, fallback=victim_notional)
        synthetic_candidate["size"] = cand_notional / max(float(synthetic_candidate["entry_price"] or 1), 1)
        after_book.append(synthetic_candidate)

        total_before = sum(self._trade_notional(t) for t in before_book) or 1.0
        total_after = sum(self._trade_notional(t) for t in after_book) or 1.0

        def share(book, total, key_fn):
            val = sum(self._trade_notional(t) for t in book if key_fn(t))
            return val / total

        coin_before = share(before_book, total_before, lambda t: str(t.get("coin", "")).upper() == candidate_coin)
        coin_after = share(after_book, total_after, lambda t: str(t.get("coin", "")).upper() == candidate_coin)
        if coin_after > self.max_coin_exposure_pct and coin_after > coin_before + 1e-6:
            return True, f"replacement would worsen coin concentration for {candidate_coin} ({coin_after:.0%})"

        side_before = share(before_book, total_before, lambda t: self._side_value(t.get("side", "")) == candidate_side)
        side_after = share(after_book, total_after, lambda t: self._side_value(t.get("side", "")) == candidate_side)
        if side_after > self.max_side_exposure_pct and side_after > side_before + 1e-6:
            return True, f"replacement would worsen {candidate_side} side concentration ({side_after:.0%})"

        candidate_cluster = self._cluster_for_coin(candidate_coin)
        cluster_before = share(
            before_book,
            total_before,
            lambda t: self._cluster_for_coin(t.get("coin", "")) == candidate_cluster,
        )
        cluster_after = share(
            after_book,
            total_after,
            lambda t: self._cluster_for_coin(t.get("coin", "")) == candidate_cluster,
        )
        if cluster_after > self.max_cluster_exposure_pct and cluster_after > cluster_before + 1e-6:
            return True, f"replacement would worsen {candidate_cluster} cluster concentration ({cluster_after:.0%})"

        return False, ""

    def _cleanup_guardrail_state(self, now: datetime) -> None:
        from datetime import timedelta
        day_cut = now - timedelta(days=1)
        self._replacement_timestamps = [ts for ts in self._replacement_timestamps if ts >= day_cut]
        self._symbol_cooldown_until = {
            coin: until for coin, until in self._symbol_cooldown_until.items() if until > now
        }
        # Keep last forced exits bounded to recent day.
        cleaned = {}
        for coin, info in self._last_forced_exit.items():
            try:
                ts = datetime.fromisoformat(str(info.get("at", "")).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= day_cut:
                    cleaned[coin] = info
            except Exception:
                continue
        self._last_forced_exit = cleaned

    def _guardrail_block_reason(self, signal, now: datetime) -> Tuple[bool, str]:
        from datetime import timedelta
        if self.max_replacements_per_hour > 0:
            hour_cut = now - timedelta(hours=1)
            count_1h = sum(1 for ts in self._replacement_timestamps if ts >= hour_cut)
            if count_1h >= self.max_replacements_per_hour:
                return True, f"replacement cap hit: {count_1h}/{self.max_replacements_per_hour} in last hour"
        if self.max_replacements_per_day > 0:
            count_1d = len(self._replacement_timestamps)
            if count_1d >= self.max_replacements_per_day:
                return True, f"replacement cap hit: {count_1d}/{self.max_replacements_per_day} in last day"

        coin = str(getattr(signal, "coin", "")).upper()
        side = self._side_value(getattr(signal, "side", ""))
        cooldown_until = self._symbol_cooldown_until.get(coin)
        if cooldown_until and cooldown_until > now:
            remaining_m = int((cooldown_until - now).total_seconds() // 60) + 1
            return True, f"{coin} in forced-exit cooldown ({remaining_m}m remaining)"

        last_exit = self._last_forced_exit.get(coin)
        if last_exit:
            try:
                exit_ts = datetime.fromisoformat(str(last_exit.get("at", "")).replace("Z", "+00:00"))
                if exit_ts.tzinfo is None:
                    exit_ts = exit_ts.replace(tzinfo=timezone.utc)
                if (now - exit_ts).total_seconds() <= self.round_trip_block_minutes * 60:
                    prev_side = str(last_exit.get("side", "")).lower()
                    if prev_side and prev_side != side:
                        return True, f"blocked immediate round-trip re-entry on {coin} ({prev_side}->{side})"
                    return True, f"blocked immediate re-entry on {coin} after forced exit"
            except Exception as e:
                # MED-FIX MED-5: log warning instead of silently swallowing —
                # a parse failure here disables the round-trip guardrail entirely
                # for this coin, which is a safety bypass that must be visible.
                logger.warning(
                    "_guardrail_block_reason: could not parse forced-exit timestamp "
                    "for %s (%r): %s — round-trip block bypassed for this signal",
                    coin, last_exit.get("at"), e,
                )
        return False, ""

    def _record_decision(self, decision: RotationDecision) -> None:
        self._telemetry["decisions"] += 1
        action = decision.action if decision.action in ("open", "reject", "replace") else "reject"
        self._telemetry[action] += 1
        if action == "reject":
            reasons = self._telemetry["rejection_reasons"]
            key = str(decision.reason or "unknown_reject")
            reasons[key] = reasons.get(key, 0) + 1

    def _count_recent_replacements(self, hours: int) -> int:
        if hours <= 0:
            return 0
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        return sum(1 for ts in self._replacement_timestamps if ts >= cutoff)
