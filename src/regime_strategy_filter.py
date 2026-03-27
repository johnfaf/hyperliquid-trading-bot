"""
Regime-Aware Strategy Filter
==============================
Matches strategies to market regimes. Not all strategies work in all conditions.
Momentum strategies thrive in trends, mean reversion in ranges, etc.

Core insight: Strategy effectiveness varies by regime. This filter:
  1. Maps each strategy to regime compatibility scores (0-1)
  2. Measures regime signal strength (ADX, volatility confidence)
  3. Blends compatibility with regime strength to adjust strategy scores
  4. Returns ranked strategies suited to current market conditions

Regimes:
  - trending_up: Strong uptrend (ADX>25, momentum>0)
  - trending_down: Strong downtrend (ADX>25, momentum<0)
  - ranging: Sideways consolidation (ADX<20, low momentum)
  - volatile: High volatility without clear direction
  - low_liquidity: Thin markets, high spreads
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# Strategy-Regime Compatibility Matrix
# Scores 0-1 indicate how well each strategy performs in each regime
# 1.0 = perfect match, 0.1 = poor match, 0.5 = neutral
REGIME_COMPATIBILITY = {
    "trending_up": {
        "momentum_long": 1.0,         # Perfect match
        "trend_following": 0.9,       # Excellent
        "breakout": 0.8,              # Good
        "swing_trading": 0.7,         # Decent
        "concentrated_bet": 0.6,      # Moderate
        "mean_reversion": 0.2,        # Counter-trend = risky
        "momentum_short": 0.1,        # Wrong direction
        "delta_neutral": 0.5,         # Neutral
        "funding_arb": 0.6,           # Can work
        "scalping": 0.5,              # Neutral
    },
    "trending_down": {
        "momentum_short": 1.0,        # Perfect match
        "trend_following": 0.9,       # Excellent
        "breakout": 0.7,              # Good but less reliable than up
        "swing_trading": 0.6,         # Decent
        "concentrated_bet": 0.5,      # Moderate
        "mean_reversion": 0.2,        # Counter-trend = risky
        "momentum_long": 0.1,         # Wrong direction
        "delta_neutral": 0.5,         # Neutral
        "funding_arb": 0.6,           # Can work
        "scalping": 0.5,              # Neutral
    },
    "ranging": {
        "mean_reversion": 1.0,        # Perfect match
        "scalping": 0.9,              # Excellent (tight chop)
        "funding_arb": 0.8,           # Good (low directional risk)
        "delta_neutral": 0.8,         # Good (flat market)
        "swing_trading": 0.6,         # Decent (range trading)
        "momentum_long": 0.3,         # Whipsaws
        "momentum_short": 0.3,        # Whipsaws
        "trend_following": 0.2,       # Fails in chop
        "breakout": 0.3,              # False breakouts
        "concentrated_bet": 0.4,      # Risky in range
    },
    "volatile": {
        "scalping": 0.7,              # Tight stops work
        "breakout": 0.8,              # Breakouts happen often
        "momentum_long": 0.5,         # Whipsaw risk
        "momentum_short": 0.5,        # Whipsaw risk
        "mean_reversion": 0.3,        # Reversals are violent
        "concentrated_bet": 0.3,      # High risk
        "delta_neutral": 0.4,         # Some cushion
        "funding_arb": 0.4,           # Can work
        "swing_trading": 0.5,         # Stop hunts
        "trend_following": 0.6,       # Works if trend clear
    },
    "low_liquidity": {
        "delta_neutral": 0.3,         # Hard to balance
        "funding_arb": 0.3,           # Slippage kills profit
        "scalping": 0.2,              # Spread too wide
        "mean_reversion": 0.4,        # Reversal slow
        "swing_trading": 0.5,         # Can work (long holds)
        "momentum_long": 0.4,         # Slippage risk
        "momentum_short": 0.4,        # Slippage risk
        "trend_following": 0.4,       # Limited liquidity
        "breakout": 0.3,              # Hard to scale in/out
        "concentrated_bet": 0.3,      # Execution risk
    },
}

# Default compatibility if strategy not in matrix (conservative)
DEFAULT_COMPATIBILITY = 0.5


@dataclass
class RegimeStrategyFilterResult:
    """Result of filtering strategies for a regime."""
    filtered_strategies: List[Dict] = field(default_factory=list)
    regime_name: str = ""
    regime_strength: float = 0.0  # 0-1, how confident the regime signal is
    report: str = ""


class RegimeStrategyFilter:
    """
    Filters and re-scores strategies based on market regime compatibility.

    Usage:
        filter = RegimeStrategyFilter()
        regime = {"regime": "trending_up", "adx": 35, "confidence": 0.8}
        strategies = [
            {"type": "momentum_long", "score": 0.75},
            {"type": "mean_reversion", "score": 0.60},
        ]
        filtered = filter.filter(strategies, regime)
    """

    def __init__(self):
        logger.info("RegimeStrategyFilter initialized")

    def filter(
        self,
        strategies: List[Dict],
        regime: Dict,
    ) -> List[Dict]:
        """
        Filter and re-score strategies for the given regime.

        Args:
            strategies: List of strategy dicts with keys:
                - type: strategy type (e.g., "momentum_long")
                - score: current strategy score (0-1)
                - confidence: (optional) strategy confidence
                - ... (other fields preserved)

            regime: Regime dict with keys:
                - regime: regime name string
                - confidence: regime confidence (0-1)
                - adx: ADX value (0-100, for trend strength)
                - atr_pct: ATR as % of price (volatility)
                - (other optional fields)

        Returns:
            List of strategies, re-scored and sorted by adjusted score (highest first)
        """
        if not strategies or not regime:
            return strategies

        # Detector outputs "overall_regime" / "overall_confidence" at market level.
        # Accept both key names so the filter works whether it receives a per-coin
        # RegimeState dict OR the market-level summary dict from get_market_regime().
        regime_name = (
            regime.get("overall_regime")
            or regime.get("regime")
            or "unknown"
        )
        regime_confidence = (
            regime.get("overall_confidence")
            or regime.get("confidence")
            or 0.5
        )

        # Get regime strength (how strong is this regime signal?)
        regime_strength = self._get_regime_strength(regime)

        # Re-score each strategy
        adjusted_strategies = []
        for strat in strategies:
            strat_type = strat.get("type", "unknown")
            base_score = strat.get("score", 0.5)

            # Get compatibility for this strategy in this regime
            compatibility = self._get_compatibility(strat_type, regime_name)

            # Blend: strong regime → lean on compatibility, weak regime → stay neutral
            # final_score = base_score * (compatibility * regime_strength + (1 - regime_strength) * 0.5)
            blended_compat = (
                compatibility * regime_strength +
                (1 - regime_strength) * 0.5
            )
            adjusted_score = base_score * blended_compat

            # Create adjusted strategy dict (preserve all original fields)
            adjusted = strat.copy()
            adjusted["adjusted_score"] = float(adjusted_score)
            adjusted["regime_compatibility"] = float(compatibility)
            adjusted["blended_compatibility"] = float(blended_compat)
            adjusted["regime_adjustment_factor"] = float(blended_compat)

            adjusted_strategies.append(adjusted)

        # Sort by adjusted score (highest first)
        adjusted_strategies.sort(
            key=lambda s: s.get("adjusted_score", 0),
            reverse=True
        )

        logger.debug(
            f"Regime filter: {regime_name} (strength={regime_strength:.2f}), "
            f"re-scored {len(adjusted_strategies)} strategies"
        )

        return adjusted_strategies

    def _get_compatibility(self, strategy_type: str, regime_name: str) -> float:
        """
        Get compatibility score for a strategy in a regime.

        Args:
            strategy_type: Strategy type (e.g., "momentum_long")
            regime_name: Regime name (e.g., "trending_up")

        Returns:
            Compatibility score 0-1
        """
        regime_matrix = REGIME_COMPATIBILITY.get(regime_name, {})
        return regime_matrix.get(strategy_type, DEFAULT_COMPATIBILITY)

    def _get_regime_strength(self, regime: Dict) -> float:
        """
        Compute how strong the regime signal is (0-1).

        A strong regime signal means we should trust the regime classification
        and heavily adjust strategy scores. A weak signal means stay neutral.

        Uses:
          - ADX (trend strength): >25 = strong trend, <20 = weak trend
          - ATR percentile: used to gauge confidence
          - Regime confidence: reported confidence from detector

        Args:
            regime: Regime dict with optional keys: adx, atr_pct, confidence, volume_ratio

        Returns:
            Strength 0-1 (0=very weak, 1=very strong)
        """
        # Accept both market-level and per-coin dicts
        regime_name = (
            regime.get("overall_regime")
            or regime.get("regime")
            or "unknown"
        )
        reported_confidence = (
            regime.get("overall_confidence")
            or regime.get("confidence")
            or 0.5
        )

        # ADX lives in per-coin dict; for market-level dict extract BTC's ADX
        adx = regime.get("adx", 0)
        if adx == 0:
            per_coin = regime.get("per_coin", {})
            btc_state = per_coin.get("BTC", per_coin.get("ETH", {}))
            adx = btc_state.get("adx", 0) if isinstance(btc_state, dict) else 0
        if regime_name in ("trending_up", "trending_down"):
            # ADX: 0-25 = no trend, 25-50 = strong trend, >50 = very strong
            if adx >= 50:
                adx_strength = 1.0
            elif adx >= 25:
                adx_strength = (adx - 25) / 25  # Linear from 0.0 to 1.0
            else:
                adx_strength = 0.0  # No trend detected
        else:
            # For non-trending regimes, ADX should be low
            adx_strength = 1.0 - min(adx / 25, 1.0)

        # Combine ADX strength with reported confidence
        regime_strength = 0.7 * adx_strength + 0.3 * reported_confidence

        # Clamp to [0, 1]
        regime_strength = max(0.0, min(1.0, regime_strength))

        return regime_strength

    def get_regime_report(self, strategies: List[Dict], regime: Dict) -> str:
        """
        Generate a detailed report on strategy filtering for this regime.

        Args:
            strategies: List of strategy dicts
            regime: Regime dict

        Returns:
            Human-readable report string
        """
        if not strategies or not regime:
            return "No strategies or regime data to report"

        regime_name = (
            regime.get("overall_regime")
            or regime.get("regime")
            or "unknown"
        )
        regime_strength = self._get_regime_strength(regime)
        # ADX — try per-coin BTC first for market-level dicts
        adx = regime.get("adx", 0)
        if adx == 0:
            per_coin = regime.get("per_coin", {})
            btc_state = per_coin.get("BTC", per_coin.get("ETH", {}))
            adx = btc_state.get("adx", 0) if isinstance(btc_state, dict) else 0
        atr_pct = regime.get("atr_pct", 0)
        if atr_pct == 0:
            per_coin = regime.get("per_coin", {})
            btc_state = per_coin.get("BTC", per_coin.get("ETH", {}))
            atr_pct = btc_state.get("atr_pct", 0) if isinstance(btc_state, dict) else 0
        confidence = (
            regime.get("overall_confidence")
            or regime.get("confidence")
            or 0
        )

        lines = [
            f"Regime Report: {regime_name}",
            f"  Strength: {regime_strength:.2%} (ADX={adx:.1f}, ATR%={atr_pct:.2%}, Confidence={confidence:.2%})",
        ]

        # Show top 3 strategies
        filtered = self.filter(strategies, regime)
        lines.append(f"  Top Strategies ({len(filtered)} total):")

        for i, strat in enumerate(filtered[:3], 1):
            stype = strat.get("type", "unknown")
            base_score = strat.get("score", 0)
            adj_score = strat.get("adjusted_score", 0)
            compat = strat.get("regime_compatibility", 0)
            lines.append(
                f"    {i}. {stype}: {base_score:.2f} → {adj_score:.2f} "
                f"(compatibility={compat:.2f})"
            )

        # Show worst strategies (low compatibility)
        worst = filtered[-3:] if len(filtered) > 3 else []
        if worst:
            lines.append(f"  Low Compatibility (avoid):")
            for strat in worst[:3]:
                stype = strat.get("type", "unknown")
                adj_score = strat.get("adjusted_score", 0)
                compat = strat.get("regime_compatibility", 0)
                lines.append(f"    - {stype}: {adj_score:.2f} (compatibility={compat:.2f})")

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")

    # Example usage
    filter = RegimeStrategyFilter()

    # Sample strategies
    sample_strategies = [
        {"type": "momentum_long", "score": 0.75, "confidence": 0.8},
        {"type": "mean_reversion", "score": 0.60, "confidence": 0.7},
        {"type": "trend_following", "score": 0.70, "confidence": 0.75},
        {"type": "scalping", "score": 0.50, "confidence": 0.6},
        {"type": "delta_neutral", "score": 0.55, "confidence": 0.65},
    ]

    # Sample regimes
    sample_regimes = [
        {
            "regime": "trending_up",
            "confidence": 0.85,
            "adx": 35.0,
            "atr_pct": 0.02,
        },
        {
            "regime": "ranging",
            "confidence": 0.80,
            "adx": 15.0,
            "atr_pct": 0.015,
        },
    ]

    # Run filter for each regime
    for regime in sample_regimes:
        filtered = filter.filter(sample_strategies, regime)
        report = filter.get_regime_report(sample_strategies, regime)
        print(report)
        print()

    print("Filtered strategies (trending_up):")
    for s in filter.filter(sample_strategies, sample_regimes[0]):
        print(f"  {s['type']}: base={s['score']:.2f} → adjusted={s['adjusted_score']:.2f}")
