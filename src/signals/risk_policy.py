"""
Dynamic risk policy engine.

This module resolves trade-specific stop-loss, take-profit, trailing, and
time-stop settings from the signal edge and current market context instead of
relying on one global hardcoded TP/SL pair.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

from src.signals.signal_schema import SignalSource, TradeSignal


@dataclass
class ResolvedRiskPolicy:
    stop_roe_pct: float
    take_profit_roe_pct: float
    reward_multiple: float
    time_limit_hours: float
    breakeven_at_r: float
    breakeven_buffer_roe_pct: float
    trail_after_r: float
    trailing_enabled: bool
    trailing_distance_roe_pct: float
    volatility_pct: float = 0.0
    regime: str = "unknown"
    regime_confidence: float = 0.0
    source_quality: float = 0.5
    rr_mode: str = "dynamic_bounded"  # H11: surface the active rr_mode
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def apply_to_signal(self, signal: TradeSignal) -> TradeSignal:
        adjusted = copy.deepcopy(signal)
        adjusted.risk.stop_loss_pct = self.stop_roe_pct
        adjusted.risk.take_profit_pct = self.take_profit_roe_pct
        adjusted.risk.risk_basis = "roe"
        adjusted.risk.reward_to_risk_ratio = self.reward_multiple
        adjusted.risk.time_limit_hours = self.time_limit_hours
        adjusted.risk.trailing_stop = self.trailing_enabled
        adjusted.risk.trailing_pct = self.trailing_distance_roe_pct
        adjusted.risk.break_even_at_r = self.breakeven_at_r
        adjusted.risk.break_even_buffer_pct = self.breakeven_buffer_roe_pct
        adjusted.risk.trail_activate_at_r = self.trail_after_r
        adjusted.risk.enforce_reward_to_risk = False
        adjusted.context = dict(adjusted.context or {})
        adjusted.context["risk_policy"] = self.to_dict()
        return adjusted


class RiskPolicyEngine:
    """Resolve dynamic risk controls from regime, volatility, and source edge."""

    # H11 (audit): Explicit R-multiple modes make the reward/risk policy
    # auditable and deployable per tier.  Prior behavior was a single
    # dynamic adjustment with hidden bounds (min 1.75R, max 4.5R); this
    # named-mode selector surfaces the choice to the operator.
    #
    #   fixed_5r        - hard set reward_multiple = 5.0; skip dynamic
    #                     adjustments (predictable, conservative TP/SL
    #                     for early tiers — bootstrap phase)
    #   dynamic_bounded - current legacy behavior (min 1.75, max 4.5);
    #                     the dynamic adjustments from regime/confidence
    #                     still apply (advanced tiers with stable edge)
    #   hybrid_min_5r   - use dynamic adjustments but floor the final R
    #                     at 5.0 (takes dynamic upside but keeps the
    #                     fixed_5r floor for consistency)
    RR_MODES = {"fixed_5r", "dynamic_bounded", "hybrid_min_5r"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        raw_mode = str(cfg.get("rr_mode", "dynamic_bounded") or "dynamic_bounded").strip().lower()
        if raw_mode not in self.RR_MODES:
            raise ValueError(
                f"Invalid rr_mode={raw_mode!r}.  Expected one of {sorted(self.RR_MODES)}."
            )
        self.rr_mode = raw_mode
        self.fixed_r_target = float(cfg.get("fixed_r_target", 5.0))
        self.hybrid_min_r_floor = float(cfg.get("hybrid_min_r_floor", 5.0))
        self.default_reward_multiple = float(cfg.get("default_reward_multiple", 3.25))
        self.min_reward_multiple = float(cfg.get("min_reward_multiple", 1.75))
        self.max_reward_multiple = float(cfg.get("max_reward_multiple", 4.5))
        self.atr_stop_multiplier = float(cfg.get("atr_stop_multiplier", 1.0))
        self.min_stop_roe_pct = float(cfg.get("min_stop_roe_pct", 0.01))
        self.max_stop_roe_pct = float(cfg.get("max_stop_roe_pct", 0.15))
        self.min_stop_price_pct = float(cfg.get("min_stop_price_pct", 0.004))
        self.max_stop_price_pct = float(cfg.get("max_stop_price_pct", 0.025))
        self.max_take_profit_price_pct = float(
            cfg.get("max_take_profit_price_pct", 0.07)
        )
        self.stop_vol_cap_multiplier = float(
            cfg.get("stop_vol_cap_multiplier", 2.5)
        )
        self.target_vol_cap_multiplier = float(
            cfg.get("target_vol_cap_multiplier", 6.0)
        )
        self.default_time_limit_hours = float(cfg.get("default_time_limit_hours", 18.0))
        self.default_break_even_at_r = float(cfg.get("default_break_even_at_r", 0.85))
        self.default_break_even_buffer_roe_pct = float(
            cfg.get("default_break_even_buffer_roe_pct", 0.005)
        )
        self.default_trail_after_r = float(cfg.get("default_trail_after_r", 1.35))
        self.default_trailing_distance_ratio = float(
            cfg.get("default_trailing_distance_ratio", 0.65)
        )
        default_source_profiles: Dict[str, Dict[str, float]] = {
            "strategy": {"time_limit_hours": 24.0, "breakeven_at_r": 1.0, "trail_after_r": 2.0},
            "copy_trade": {"time_limit_hours": 18.0, "breakeven_at_r": 1.0, "trail_after_r": 2.0},
            "options_flow": {"time_limit_hours": 8.0, "breakeven_at_r": 0.75, "trail_after_r": 1.5},
            "liquidation_strategy": {"time_limit_hours": 6.0, "breakeven_at_r": 0.75, "trail_after_r": 1.25},
            "arena_champion": {"time_limit_hours": 12.0, "breakeven_at_r": 1.0, "trail_after_r": 1.75},
            "polymarket": {"time_limit_hours": 16.0, "breakeven_at_r": 1.0, "trail_after_r": 2.0},
            "whale_trade": {"time_limit_hours": 10.0, "breakeven_at_r": 0.75, "trail_after_r": 1.5},
        }
        configured_profiles = cfg.get("source_profiles", {})
        self.source_profiles = dict(default_source_profiles)
        if isinstance(configured_profiles, dict):
            for key, value in configured_profiles.items():
                if not isinstance(value, dict):
                    continue
                merged = dict(self.source_profiles.get(str(key).lower(), default_source_profiles["strategy"]))
                merged.update(value)
                self.source_profiles[str(key).lower()] = merged

    def resolve(
        self,
        signal: TradeSignal,
        regime_data: Optional[Dict[str, Any]] = None,
        source_policy: Optional[Dict[str, Any]] = None,
    ) -> ResolvedRiskPolicy:
        leverage = max(float(signal.leverage or 1.0), 1.0)
        regime = self._extract_regime(signal, regime_data)
        regime_confidence = self._extract_regime_confidence(regime_data)
        volatility_pct = self._extract_volatility(signal)
        expected_move_pct = self._extract_expected_move(signal)
        source_quality = self._extract_source_quality(signal, source_policy)
        source_key = self._source_key(signal)
        profile = self.source_profiles.get(source_key, self.source_profiles["strategy"])

        base_stop_roe_pct = max(signal.risk.resolve_roe_stop_loss_pct(leverage), self.min_stop_roe_pct)
        atr_stop_roe_pct = volatility_pct * leverage * self.atr_stop_multiplier

        stop_multiplier = 1.0
        rationale = []
        if volatility_pct > 0:
            rationale.append(f"atr={volatility_pct:.4f}")

        if regime in {"crash", "volatile"}:
            stop_multiplier *= 1.30 if regime == "crash" else 1.45
            rationale.append(f"regime={regime}: wider stop")
        elif regime in {"ranging", "low_liquidity"}:
            stop_multiplier *= 0.90
            rationale.append(f"regime={regime}: slightly tighter stop")

        if signal.confidence < 0.45:
            stop_multiplier *= 1.10
            rationale.append("low_confidence: wider stop")
        elif signal.confidence > 0.75:
            stop_multiplier *= 0.95
            rationale.append("high_confidence: slightly tighter stop")

        stop_roe_pct = max(base_stop_roe_pct * stop_multiplier, atr_stop_roe_pct)
        stop_roe_pct = min(max(stop_roe_pct, self.min_stop_roe_pct), self.max_stop_roe_pct)

        # H11: reward_multiple derivation depends on the configured rr_mode.
        # ``fixed_5r`` short-circuits all dynamic adjustments for predictable
        # TP/SL; ``dynamic_bounded`` preserves legacy behavior; ``hybrid_min_5r``
        # runs the dynamic path but floors the final R at hybrid_min_r_floor.
        if self.rr_mode == "fixed_5r":
            reward_multiple = self.fixed_r_target
            rationale.append(f"rr_mode=fixed_5r:target={self.fixed_r_target:.2f}R")
        else:
            reward_multiple = self.default_reward_multiple
            if regime in {"trending_up", "trending_down", "bullish", "bearish"}:
                reward_multiple += 0.75
                rationale.append("trend regime: larger target")
            elif regime in {"ranging"}:
                reward_multiple -= 1.75
                rationale.append("range regime: smaller target")
            elif regime in {"crash", "volatile"}:
                reward_multiple -= 1.25
                rationale.append("chaotic regime: reduce target")

            if source_quality >= 0.60:
                reward_multiple += 0.5
                rationale.append("strong source quality")
            elif source_quality <= 0.45:
                reward_multiple -= 0.75
                rationale.append("weak source quality")

            if signal.confidence >= 0.75:
                reward_multiple += 0.5
            elif signal.confidence <= 0.45:
                reward_multiple -= 0.75

            if isinstance(source_policy, dict):
                status = str(source_policy.get("status", "") or "").lower()
                if status == "degraded":
                    reward_multiple -= 0.5
                    rationale.append("source degraded")
                elif status == "paused":
                    reward_multiple = min(reward_multiple, 2.0)
                    rationale.append("source paused: capped target")

            if expected_move_pct > 0 and stop_roe_pct > 0:
                expected_rr = (expected_move_pct * leverage) / stop_roe_pct
                reward_multiple = min(reward_multiple, max(self.min_reward_multiple, expected_rr * 1.25))
                rationale.append(f"expected_move_cap={expected_rr:.2f}R")

            reward_multiple = min(max(reward_multiple, self.min_reward_multiple), self.max_reward_multiple)

            if self.rr_mode == "hybrid_min_5r":
                # Floor the dynamic result at hybrid_min_r_floor so the
                # mixed mode never under-targets below the fixed baseline.
                if reward_multiple < self.hybrid_min_r_floor:
                    rationale.append(
                        f"rr_mode=hybrid_min_5r:floored_from={reward_multiple:.2f}R"
                        f"_to={self.hybrid_min_r_floor:.2f}R"
                    )
                    reward_multiple = self.hybrid_min_r_floor
                else:
                    rationale.append(
                        f"rr_mode=hybrid_min_5r:dynamic={reward_multiple:.2f}R"
                    )
        take_profit_roe_pct = stop_roe_pct * reward_multiple

        # H11: effective min/max R bounds depend on rr_mode.  fixed_5r and
        # hybrid_min_5r may require R above max_reward_multiple (which was
        # set for the legacy dynamic_bounded mode).
        if self.rr_mode == "fixed_5r":
            effective_min_rr = self.fixed_r_target
            effective_max_rr = max(self.fixed_r_target, self.max_reward_multiple)
        elif self.rr_mode == "hybrid_min_5r":
            effective_min_rr = self.hybrid_min_r_floor
            effective_max_rr = max(self.hybrid_min_r_floor, self.max_reward_multiple)
        else:
            effective_min_rr = self.min_reward_multiple
            effective_max_rr = self.max_reward_multiple

        stop_price_pct = stop_roe_pct / leverage
        take_profit_price_pct = take_profit_roe_pct / leverage

        dynamic_stop_cap = self.max_stop_price_pct
        if volatility_pct > 0:
            dynamic_stop_cap = min(
                dynamic_stop_cap,
                max(self.min_stop_price_pct, volatility_pct * self.stop_vol_cap_multiplier),
            )
        stop_price_pct = min(max(stop_price_pct, self.min_stop_price_pct), dynamic_stop_cap)

        # H11: target price cap must honor the effective min/max R.  For
        # fixed_5r we also widen max_take_profit_price_pct to
        # stop_price_pct * fixed_r_target so the cap doesn't truncate the
        # fixed target back to a smaller effective R.
        tp_price_cap_base = max(
            self.max_take_profit_price_pct,
            stop_price_pct * effective_max_rr,
        )
        dynamic_target_cap = tp_price_cap_base
        if volatility_pct > 0:
            dynamic_target_cap = min(
                dynamic_target_cap,
                max(stop_price_pct * effective_min_rr, volatility_pct * self.target_vol_cap_multiplier),
            )
        if expected_move_pct > 0:
            dynamic_target_cap = min(
                dynamic_target_cap,
                max(stop_price_pct * effective_min_rr, expected_move_pct * 1.15),
            )
        take_profit_price_pct = min(
            max(take_profit_price_pct, stop_price_pct * effective_min_rr),
            dynamic_target_cap,
        )

        stop_roe_pct = min(
            max(stop_price_pct * leverage, self.min_stop_roe_pct),
            self.max_stop_roe_pct,
        )
        take_profit_roe_pct = take_profit_price_pct * leverage
        reward_multiple = max(
            1.0,
            min(effective_max_rr, take_profit_roe_pct / stop_roe_pct),
        )

        if stop_price_pct >= dynamic_stop_cap - 1e-9:
            rationale.append(f"stop_cap={dynamic_stop_cap:.3%}")
        if take_profit_price_pct >= dynamic_target_cap - 1e-9:
            rationale.append(f"target_cap={dynamic_target_cap:.3%}")

        time_limit_hours = float(profile.get("time_limit_hours", self.default_time_limit_hours))
        if regime in {"crash", "volatile"}:
            time_limit_hours = max(2.0, time_limit_hours - 4.0)
        elif regime in {"trending_up", "trending_down", "bullish", "bearish"}:
            time_limit_hours = min(72.0, time_limit_hours + 4.0)
        if stop_price_pct >= self.max_stop_price_pct * 0.9:
            time_limit_hours = min(time_limit_hours, max(6.0, self.default_time_limit_hours))

        breakeven_at_r = float(profile.get("breakeven_at_r", self.default_break_even_at_r))
        if signal.confidence >= 0.80:
            breakeven_at_r += 0.15
        elif regime in {"crash", "volatile"}:
            breakeven_at_r = max(0.5, breakeven_at_r - 0.25)

        trail_after_r = max(
            breakeven_at_r + 0.20,
            float(profile.get("trail_after_r", self.default_trail_after_r)),
        )
        trailing_distance_roe_pct = max(
            stop_roe_pct * self.default_trailing_distance_ratio,
            atr_stop_roe_pct * 0.75,
            self.min_stop_roe_pct,
        )
        trailing_distance_roe_pct = min(trailing_distance_roe_pct, max(stop_roe_pct, self.max_stop_roe_pct * 0.75))

        breakeven_buffer_roe_pct = min(
            max(stop_roe_pct * 0.10, self.default_break_even_buffer_roe_pct),
            stop_roe_pct * 0.35,
        )

        return ResolvedRiskPolicy(
            stop_roe_pct=round(stop_roe_pct, 6),
            take_profit_roe_pct=round(take_profit_roe_pct, 6),
            reward_multiple=round(reward_multiple, 4),
            time_limit_hours=round(time_limit_hours, 2),
            breakeven_at_r=round(breakeven_at_r, 3),
            breakeven_buffer_roe_pct=round(breakeven_buffer_roe_pct, 6),
            trail_after_r=round(trail_after_r, 3),
            trailing_enabled=bool(signal.risk.trailing_stop),
            trailing_distance_roe_pct=round(trailing_distance_roe_pct, 6),
            volatility_pct=round(volatility_pct, 6),
            regime=regime,
            regime_confidence=round(regime_confidence, 4),
            source_quality=round(source_quality, 4),
            rr_mode=self.rr_mode,
            rationale=rationale,
        )

    def apply(
        self,
        signal: TradeSignal,
        regime_data: Optional[Dict[str, Any]] = None,
        source_policy: Optional[Dict[str, Any]] = None,
    ) -> TradeSignal:
        existing_policy = dict((signal.context or {}).get("risk_policy", {}) or {})
        if existing_policy.get("policy_version") == "dynamic_v1":
            return signal

        resolved = self.resolve(signal, regime_data=regime_data, source_policy=source_policy)
        adjusted = resolved.apply_to_signal(signal)
        adjusted.context["risk_policy"]["policy_version"] = "dynamic_v1"
        return adjusted

    @staticmethod
    def current_r_multiple(
        entry_price: float,
        current_price: float,
        side: str,
        leverage: float,
        stop_roe_pct: float,
    ) -> float:
        if entry_price <= 0 or current_price <= 0 or leverage <= 0 or stop_roe_pct <= 0:
            return 0.0
        direction = 1.0 if str(side or "").lower() == "long" else -1.0
        move_pct = ((current_price - entry_price) / entry_price) * direction
        roe_move_pct = move_pct * leverage
        return roe_move_pct / stop_roe_pct

    @staticmethod
    def _source_key(signal: TradeSignal) -> str:
        source = signal.source.value if isinstance(signal.source, SignalSource) else str(signal.source)
        return source.lower().strip() or "strategy"

    @staticmethod
    def _extract_regime(signal: TradeSignal, regime_data: Optional[Dict[str, Any]]) -> str:
        candidates: Iterable[Any] = (
            (regime_data or {}).get("regime"),
            (regime_data or {}).get("overall_regime"),
            signal.regime,
            (signal.context or {}).get("regime"),
        )
        for item in candidates:
            value = str(item or "").strip().lower()
            if value:
                return value
        return "unknown"

    @staticmethod
    def _extract_regime_confidence(regime_data: Optional[Dict[str, Any]]) -> float:
        if not isinstance(regime_data, dict):
            return 0.0
        for key in ("confidence", "overall_confidence"):
            try:
                value = float(regime_data.get(key, 0.0) or 0.0)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _extract_volatility(signal: TradeSignal) -> float:
        context = signal.context or {}
        features = context.get("features", {})
        candidates = (
            context.get("atr_pct"),
            context.get("volatility"),
            context.get("volatility_pct"),
            context.get("expected_volatility"),
            features.get("atr_pct") if isinstance(features, dict) else None,
            features.get("volatility") if isinstance(features, dict) else None,
        )
        for value in candidates:
            try:
                vol = float(value or 0.0)
            except (TypeError, ValueError):
                continue
            if vol > 0:
                return vol
        return 0.0

    @staticmethod
    def _extract_expected_move(signal: TradeSignal) -> float:
        context = signal.context or {}
        features = context.get("features", {})
        candidates = (
            context.get("expected_return"),
            context.get("expected_move_pct"),
            features.get("expected_return") if isinstance(features, dict) else None,
        )
        for value in candidates:
            try:
                expected = float(value or 0.0)
            except (TypeError, ValueError):
                continue
            if expected > 0:
                return expected
        return 0.0

    @staticmethod
    def _extract_source_quality(signal: TradeSignal, source_policy: Optional[Dict[str, Any]]) -> float:
        if isinstance(source_policy, dict):
            for key in ("quality", "score", "win_rate"):
                try:
                    value = float(source_policy.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value if value <= 1.0 else min(value / 100.0, 1.0)
        try:
            value = float(signal.source_accuracy or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return value if value > 0 else 0.5
