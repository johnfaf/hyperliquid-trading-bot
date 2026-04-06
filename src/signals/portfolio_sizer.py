"""
Portfolio-aware sizing and adaptive exit management.

This layer sits between raw signal generation and execution so the bot sizes
new entries in the context of the existing book instead of treating every trade
as if the portfolio were empty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


COIN_CLUSTER_MAP = {
    "BTC": "btc",
    "ETH": "majors",
    "SOL": "majors",
    "XRP": "majors",
    "HYPE": "majors",
    "DOGE": "memes",
    "WIF": "memes",
    "PEPE": "memes",
    "BONK": "memes",
    "SHIB": "memes",
    "AVAX": "layer1",
    "SUI": "layer1",
    "SEI": "layer1",
    "TIA": "layer1",
    "NEAR": "layer1",
    "APT": "layer1",
    "INJ": "layer1",
    "ARB": "layer2",
    "OP": "layer2",
    "MATIC": "layer2",
    "LINK": "infra",
    "ONDO": "defi",
    "AAVE": "defi",
    "UNI": "defi",
    "JUP": "defi",
    "FET": "ai",
    "RENDER": "ai",
    "WLD": "ai",
    "TAO": "ai",
}

CLUSTER_BETA_MAP = {
    "btc": 1.00,
    "majors": 0.90,
    "layer1": 1.05,
    "layer2": 1.00,
    "memes": 1.30,
    "infra": 0.95,
    "defi": 1.00,
    "ai": 1.15,
    "alts": 1.00,
}

COIN_BETA_OVERRIDES = {
    "BTC": 1.00,
    "ETH": 0.92,
    "SOL": 1.08,
    "XRP": 0.96,
    "DOGE": 1.28,
    "HYPE": 1.08,
}


@dataclass
class PortfolioSizingDecision:
    position_pct: float
    original_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    time_limit_hours: float
    cluster: str
    volatility_pct: float
    side_exposure_pct: float
    cluster_exposure_pct: float
    coin_exposure_pct: float
    portfolio_beta: float
    projected_beta: float
    size_multiplier: float
    blocked: bool = False
    block_reason: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class PortfolioSizer:
    """Size trades with concentration, volatility, and beta awareness."""

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.min_position_pct = float(cfg.get("min_position_pct", 0.01))
        self.max_position_pct = float(
            cfg.get("max_position_pct", getattr(config, "PAPER_TRADING_MAX_POSITION_PCT", 0.08))
        )
        self.max_coin_exposure_pct = float(
            cfg.get("max_coin_exposure_pct", config.PORTFOLIO_MAX_COIN_EXPOSURE_PCT)
        )
        self.max_side_exposure_pct = float(
            cfg.get("max_side_exposure_pct", config.PORTFOLIO_MAX_SIDE_EXPOSURE_PCT)
        )
        self.max_cluster_exposure_pct = float(
            cfg.get("max_cluster_exposure_pct", config.PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT)
        )
        self.max_beta_abs = float(cfg.get("max_beta_abs", 1.40))
        self.target_volatility_pct = float(cfg.get("target_volatility_pct", 0.025))
        self.stop_loss_vol_multiplier = float(cfg.get("stop_loss_vol_multiplier", 0.85))
        self.min_stop_loss_pct = float(cfg.get("min_stop_loss_pct", 0.0075))
        self.max_stop_loss_pct = float(cfg.get("max_stop_loss_pct", 0.06))
        self.trend_reward_risk = float(cfg.get("trend_reward_risk", 2.4))
        self.base_reward_risk = float(cfg.get("base_reward_risk", 1.9))
        self.volatile_reward_risk = float(cfg.get("volatile_reward_risk", 1.35))
        self.trend_time_limit_hours = float(cfg.get("trend_time_limit_hours", 36.0))
        self.base_time_limit_hours = float(cfg.get("base_time_limit_hours", 24.0))
        self.volatile_time_limit_hours = float(cfg.get("volatile_time_limit_hours", 12.0))

        self.stats = {
            "total_adjustments": 0,
            "blocked": 0,
            "size_reduced": 0,
            "last_decision": None,
        }

    @staticmethod
    def classify_cluster(coin: str) -> str:
        normalized = str(coin or "").strip().upper()
        return COIN_CLUSTER_MAP.get(normalized, "alts")

    @classmethod
    def get_beta(cls, coin: str) -> float:
        normalized = str(coin or "").strip().upper()
        if normalized in COIN_BETA_OVERRIDES:
            return COIN_BETA_OVERRIDES[normalized]
        cluster = cls.classify_cluster(normalized)
        return CLUSTER_BETA_MAP.get(cluster, 1.0)

    @staticmethod
    def _coerce_side(value) -> str:
        raw = getattr(value, "value", value)
        return str(raw or "").strip().lower() or "long"

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _notional_pct(self, position: Dict, account_balance: float) -> float:
        if account_balance <= 0:
            return 0.0
        size = abs(self._safe_float(position.get("size", position.get("szi")), 0.0))
        entry_price = self._safe_float(position.get("entry_price", position.get("entryPx")), 0.0)
        leverage = max(self._safe_float(position.get("leverage"), 1.0), 1.0)
        if size > 0 and entry_price > 0:
            return (size * entry_price * leverage) / account_balance
        position_pct = abs(self._safe_float(position.get("position_pct"), 0.0))
        if position_pct > 0:
            return position_pct * leverage
        return 0.0

    def summarize_exposure(self, open_positions: List[Dict], account_balance: float) -> Dict:
        summary = {
            "by_coin": {},
            "by_side": {"long": 0.0, "short": 0.0},
            "by_cluster": {},
            "signed_beta": 0.0,
        }
        for position in open_positions or []:
            coin = str(position.get("coin", "")).strip().upper()
            if not coin:
                continue
            side = self._coerce_side(position.get("side"))
            cluster = self.classify_cluster(coin)
            notional_pct = self._notional_pct(position, account_balance)
            if notional_pct <= 0:
                continue
            summary["by_coin"][coin] = summary["by_coin"].get(coin, 0.0) + notional_pct
            summary["by_side"][side] = summary["by_side"].get(side, 0.0) + notional_pct
            summary["by_cluster"][cluster] = summary["by_cluster"].get(cluster, 0.0) + notional_pct
            direction = 1.0 if side == "long" else -1.0
            summary["signed_beta"] += notional_pct * direction * self.get_beta(coin)
        return summary

    def _extract_volatility(self, coin: str, regime_data: Optional[Dict], features: Optional[Dict]) -> float:
        if isinstance(features, dict):
            value = self._safe_float(features.get("volatility"), 0.0)
            if value > 0:
                return value

        if isinstance(regime_data, dict):
            per_coin = regime_data.get("per_coin", {})
            coin_state = per_coin.get(str(coin or "").strip().upper(), {})
            if isinstance(coin_state, dict):
                value = self._safe_float(coin_state.get("atr_pct"), 0.0)
                if value > 0:
                    return value

        return self.target_volatility_pct

    def _solve_beta_cap(
        self,
        current_beta: float,
        impact_per_position_pct: float,
        base_position_pct: float,
    ) -> float:
        if impact_per_position_pct == 0:
            return base_position_pct
        projected = current_beta + (impact_per_position_pct * base_position_pct)
        if abs(projected) <= self.max_beta_abs:
            return base_position_pct

        target = self.max_beta_abs if impact_per_position_pct > 0 else -self.max_beta_abs
        allowed = (target - current_beta) / impact_per_position_pct
        return max(0.0, min(base_position_pct, allowed))

    def adjust_signal(
        self,
        signal,
        *,
        open_positions: Optional[List[Dict]] = None,
        account_balance: float = 0.0,
        regime_data: Optional[Dict] = None,
        features: Optional[Dict] = None,
    ) -> PortfolioSizingDecision:
        coin = str(getattr(signal, "coin", "") or "").strip().upper()
        side = self._coerce_side(getattr(signal, "side", "long"))
        leverage = max(self._safe_float(getattr(signal, "leverage", 1.0), 1.0), 1.0)
        base_position_pct = self._safe_float(getattr(signal, "position_pct", 0.0), 0.0)
        base_position_pct = max(self.min_position_pct, min(base_position_pct, self.max_position_pct))
        regime_name = ""
        if isinstance(regime_data, dict):
            regime_name = str(regime_data.get("overall_regime", "") or "")
        regime_name = regime_name or str(getattr(signal, "regime", "") or "")
        cluster = self.classify_cluster(coin)

        exposures = self.summarize_exposure(open_positions or [], account_balance)
        coin_exposure_pct = exposures["by_coin"].get(coin, 0.0)
        side_exposure_pct = exposures["by_side"].get(side, 0.0)
        cluster_exposure_pct = exposures["by_cluster"].get(cluster, 0.0)
        portfolio_beta = exposures["signed_beta"]
        volatility_pct = self._extract_volatility(coin, regime_data, features)

        allowed_by_coin = max(0.0, (self.max_coin_exposure_pct - coin_exposure_pct) / leverage)
        allowed_by_side = max(0.0, (self.max_side_exposure_pct - side_exposure_pct) / leverage)
        allowed_by_cluster = max(0.0, (self.max_cluster_exposure_pct - cluster_exposure_pct) / leverage)
        impact_per_position_pct = (1.0 if side == "long" else -1.0) * leverage * self.get_beta(coin)
        allowed_by_beta = self._solve_beta_cap(portfolio_beta, impact_per_position_pct, base_position_pct)
        projected_beta = portfolio_beta + (impact_per_position_pct * min(base_position_pct, allowed_by_beta))

        hard_cap_pct = min(base_position_pct, allowed_by_coin, allowed_by_side, allowed_by_cluster, allowed_by_beta)
        reasons: List[str] = []
        if allowed_by_coin < base_position_pct:
            reasons.append("coin_exposure_cap")
        if allowed_by_side < base_position_pct:
            reasons.append("side_exposure_cap")
        if allowed_by_cluster < base_position_pct:
            reasons.append("cluster_exposure_cap")
        if allowed_by_beta < base_position_pct:
            reasons.append("btc_beta_cap")

        blocked = False
        block_reason = ""
        if not self.enabled:
            hard_cap_pct = base_position_pct
        elif hard_cap_pct <= 0:
            blocked = True
            if allowed_by_side <= 0:
                block_reason = "no same-side exposure headroom"
            elif allowed_by_cluster <= 0:
                block_reason = f"no {cluster} cluster headroom"
            elif allowed_by_coin <= 0:
                block_reason = f"no {coin} coin headroom"
            else:
                block_reason = "btc beta limit reached"

        regime_multiplier = 1.0
        if isinstance(regime_data, dict):
            guidance = regime_data.get("strategy_guidance", {})
            regime_multiplier = self._safe_float(guidance.get("size_modifier"), 1.0)
        if regime_name in {"volatile", "low_liquidity"}:
            regime_multiplier = min(regime_multiplier, 0.60)

        volatility_multiplier = 1.0
        if volatility_pct > self.target_volatility_pct > 0:
            volatility_multiplier = max(0.35, self.target_volatility_pct / volatility_pct)
            reasons.append("volatility_targeting")

        adjusted_position_pct = min(base_position_pct * regime_multiplier * volatility_multiplier, hard_cap_pct)
        if not blocked and adjusted_position_pct < self.min_position_pct:
            if hard_cap_pct < self.min_position_pct:
                blocked = True
                block_reason = "insufficient portfolio headroom after sizing"
            else:
                adjusted_position_pct = self.min_position_pct

        aligned_with_regime = (
            (regime_name == "trending_up" and side == "long")
            or (regime_name == "trending_down" and side == "short")
        )
        base_stop_loss_pct = max(
            self._safe_float(getattr(getattr(signal, "risk", None), "stop_loss_pct", 0.0), 0.0),
            self.min_stop_loss_pct,
        )
        adaptive_stop_loss_pct = min(
            max(base_stop_loss_pct, volatility_pct * self.stop_loss_vol_multiplier),
            self.max_stop_loss_pct,
        )
        if regime_name in {"volatile", "low_liquidity"}:
            reward_risk = self.volatile_reward_risk
            time_limit_hours = self.volatile_time_limit_hours
        elif aligned_with_regime:
            reward_risk = self.trend_reward_risk
            time_limit_hours = self.trend_time_limit_hours
        else:
            reward_risk = self.base_reward_risk
            time_limit_hours = self.base_time_limit_hours

        base_take_profit_pct = self._safe_float(
            getattr(getattr(signal, "risk", None), "take_profit_pct", 0.0),
            adaptive_stop_loss_pct * reward_risk,
        )
        adaptive_take_profit_pct = max(base_take_profit_pct, adaptive_stop_loss_pct * reward_risk)
        if volatility_pct > self.target_volatility_pct * 1.5:
            time_limit_hours = min(time_limit_hours, self.volatile_time_limit_hours)

        size_multiplier = adjusted_position_pct / base_position_pct if base_position_pct > 0 else 0.0
        decision = PortfolioSizingDecision(
            position_pct=round(adjusted_position_pct, 6),
            original_position_pct=round(base_position_pct, 6),
            stop_loss_pct=round(adaptive_stop_loss_pct, 6),
            take_profit_pct=round(adaptive_take_profit_pct, 6),
            time_limit_hours=round(time_limit_hours, 2),
            cluster=cluster,
            volatility_pct=round(volatility_pct, 6),
            side_exposure_pct=round(side_exposure_pct, 6),
            cluster_exposure_pct=round(cluster_exposure_pct, 6),
            coin_exposure_pct=round(coin_exposure_pct, 6),
            portfolio_beta=round(portfolio_beta, 6),
            projected_beta=round(projected_beta, 6),
            size_multiplier=round(size_multiplier, 6),
            blocked=blocked,
            block_reason=block_reason,
            reasons=reasons,
        )

        self.stats["total_adjustments"] += 1
        if blocked:
            self.stats["blocked"] += 1
        if decision.position_pct < decision.original_position_pct:
            self.stats["size_reduced"] += 1
        self.stats["last_decision"] = decision.to_dict()
        return decision

    def apply_to_signal(
        self,
        signal,
        *,
        open_positions: Optional[List[Dict]] = None,
        account_balance: float = 0.0,
        regime_data: Optional[Dict] = None,
        features: Optional[Dict] = None,
    ) -> PortfolioSizingDecision:
        decision = self.adjust_signal(
            signal,
            open_positions=open_positions,
            account_balance=account_balance,
            regime_data=regime_data,
            features=features,
        )
        if decision.blocked:
            return decision

        signal.position_pct = decision.position_pct
        signal.risk.stop_loss_pct = decision.stop_loss_pct
        signal.risk.take_profit_pct = decision.take_profit_pct
        signal.risk.time_limit_hours = decision.time_limit_hours
        return decision

    def get_stats(self) -> Dict:
        return dict(self.stats)
