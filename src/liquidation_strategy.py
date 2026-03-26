"""
Liquidation Cascade Reversal Strategy (LCRS)
=============================================
The first concrete, structural edge hypothesis for Hyperliquid.

Core thesis:
  When funding is extreme + open interest is rising + momentum is stalling,
  the market is overcrowded on one side. A reversal (liquidation cascade) is likely.

This is NOT indicator-based guessing. It's targeting a real market microstructure
phenomenon: leveraged positions getting forcefully liquidated create predictable
price spikes in the opposite direction.

Hyperliquid-specific advantages:
  - Transparent funding rates (updated every hour)
  - Open interest data via API
  - Fast liquidation engine creates sharp wicks
  - Lower fees than CEXes for the reversal capture

Inputs required:
  - funding_rate: current 8h funding rate
  - oi_change: % change in open interest over last 4-8h
  - price_change: % price change over same window
  - trend_strength: 0-1 (ADX-normalized or consistency measure)
  - volatility: ATR% or similar
  - volume_ratio: current volume vs 20-period average

Outputs:
  - Signal dict compatible with paper_trader pipeline
  - Or None if no trade setup detected
"""
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LiquidationSetup:
    """A detected liquidation cascade setup."""
    coin: str
    side: str               # "long" or "short" — the direction WE trade (opposite of crowded side)
    confidence: float       # 0-1
    expected_return: float  # Expected % return
    funding_rate: float
    oi_change: float
    price_change: float
    trend_strength: float
    exhaustion_score: float  # How exhausted the move looks (0-1)
    crowding_score: float    # How crowded one side is (0-1)
    setup_type: str          # "funding_extreme", "oi_spike", "momentum_exhaustion", "full_confluence"

    def to_dict(self) -> Dict:
        return asdict(self)


class LiquidationStrategy:
    """
    Detects overcrowded positions on Hyperliquid and generates
    reversal signals when momentum exhaustion is detected.

    Three sub-signals that combine for confluence:
    1. Funding Rate Extreme — crowded positioning
    2. OI Spike + Stalling Price — new positions but no follow-through
    3. Momentum Exhaustion — trend weakening after strong move
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Funding rate thresholds
        self.funding_extreme_long = cfg.get("funding_extreme_long", 0.0005)   # 0.05% per 8h
        self.funding_extreme_short = cfg.get("funding_extreme_short", -0.0005)
        self.funding_very_extreme = cfg.get("funding_very_extreme", 0.001)    # 0.1% — very crowded

        # OI thresholds
        self.oi_spike_threshold = cfg.get("oi_spike_threshold", 0.03)    # 3% OI increase
        self.oi_surge_threshold = cfg.get("oi_surge_threshold", 0.08)    # 8% — massive new positions

        # Momentum exhaustion
        self.exhaustion_trend_max = cfg.get("exhaustion_trend_max", 0.35)  # Trend weakening below this
        self.price_move_min = cfg.get("price_move_min", 0.015)            # At least 1.5% move occurred

        # Volatility filter
        self.min_volatility = cfg.get("min_volatility", 0.005)  # Need some volatility
        self.max_volatility = cfg.get("max_volatility", 0.08)   # Not during chaos

        # Volume confirmation
        self.volume_confirmation = cfg.get("volume_confirmation", 1.2)  # Above-average volume

        # Position sizing based on confidence
        self.base_position_pct = cfg.get("base_position_pct", 0.06)  # 6% base
        self.max_position_pct = cfg.get("max_position_pct", 0.10)    # 10% max

        # Risk params
        self.stop_loss_pct = cfg.get("stop_loss_pct", 0.025)    # 2.5% stop (tight for reversals)
        self.take_profit_pct = cfg.get("take_profit_pct", 0.05)  # 5% TP (2:1 R:R)
        self.max_leverage = cfg.get("max_leverage", 3)            # Conservative leverage

        # Stats
        self.signals_generated = 0
        self.setups_detected = 0

        logger.info(f"LiquidationStrategy initialized: funding_threshold={self.funding_extreme_long}, "
                    f"oi_spike={self.oi_spike_threshold}, exhaustion_max={self.exhaustion_trend_max}")

    def analyze(self, coin: str, features: Dict) -> Optional[LiquidationSetup]:
        """
        Analyze market features for a liquidation cascade setup.

        Args:
            coin: Token symbol (e.g. "BTC")
            features: Dict with keys:
                - funding_rate: float (current 8h rate)
                - oi_change: float (% change in OI)
                - price_change: float (% price change)
                - trend_strength: float (0-1)
                - volatility: float (ATR%)
                - volume_ratio: float (vs 20-period avg)

        Returns:
            LiquidationSetup if a setup is detected, None otherwise.
        """
        funding = features.get("funding_rate", 0)
        oi_change = features.get("oi_change", 0)
        price_change = features.get("price_change", 0)
        trend_strength = features.get("trend_strength", 0.5)
        volatility = features.get("volatility", 0)
        volume_ratio = features.get("volume_ratio", 1.0)
        rsi = features.get("rsi", 50)

        # ─── Volatility Filter ───────────────────────────────────
        # Skip if volatility is too low (no opportunity) or too high (chaos)
        if volatility < self.min_volatility or volatility > self.max_volatility:
            return None

        # ─── Score each sub-signal ───────────────────────────────

        # 1. Funding crowding score (0-1)
        crowding_score = 0.0
        crowded_side = None  # Which side is crowded (we trade OPPOSITE)

        if funding > self.funding_extreme_long:
            crowding_score = min((funding - self.funding_extreme_long) / self.funding_very_extreme, 1.0)
            crowded_side = "long"  # Longs are crowded → we SHORT
        elif funding < self.funding_extreme_short:
            crowding_score = min((abs(funding) - abs(self.funding_extreme_short)) / abs(self.funding_very_extreme), 1.0)
            crowded_side = "short"  # Shorts are crowded → we LONG

        # 2. OI spike score (0-1)
        oi_score = 0.0
        if oi_change > self.oi_spike_threshold:
            oi_score = min((oi_change - self.oi_spike_threshold) /
                           (self.oi_surge_threshold - self.oi_spike_threshold), 1.0)

        # 3. Momentum exhaustion score (0-1)
        exhaustion_score = 0.0
        price_moved = abs(price_change) >= self.price_move_min

        if price_moved and trend_strength < self.exhaustion_trend_max:
            # Price moved significantly but trend is weakening = exhaustion
            exhaustion_score = (self.exhaustion_trend_max - trend_strength) / self.exhaustion_trend_max
            exhaustion_score = min(exhaustion_score, 1.0)

        # RSI extremes boost exhaustion
        if rsi > 75:
            exhaustion_score = min(exhaustion_score + 0.2, 1.0)
        elif rsi < 25:
            exhaustion_score = min(exhaustion_score + 0.2, 1.0)

        # 4. Volume confirmation
        volume_confirmed = volume_ratio >= self.volume_confirmation

        # ─── Determine if setup is valid ─────────────────────────

        # We need at least crowding signal to trade
        if crowding_score < 0.2:
            return None

        # Calculate composite confidence
        # Crowding is the primary signal, exhaustion and OI are confirming
        raw_confidence = (
            crowding_score * 0.40 +     # Funding crowding is king
            exhaustion_score * 0.30 +   # Momentum exhaustion confirms
            oi_score * 0.20 +           # Rising OI means new positions
            (0.10 if volume_confirmed else 0.0)  # Volume confirms
        )

        # Minimum confidence threshold
        if raw_confidence < 0.35:
            return None

        # Determine setup type based on which signals are active
        active_signals = []
        if crowding_score >= 0.5:
            active_signals.append("funding_extreme")
        if oi_score >= 0.3:
            active_signals.append("oi_spike")
        if exhaustion_score >= 0.4:
            active_signals.append("momentum_exhaustion")

        if len(active_signals) >= 3:
            setup_type = "full_confluence"
            raw_confidence = min(raw_confidence * 1.2, 0.95)  # Boost for full confluence
        elif len(active_signals) >= 2:
            setup_type = "+".join(active_signals)
        elif active_signals:
            setup_type = active_signals[0]
        else:
            return None  # No strong sub-signal

        # Determine trade direction (OPPOSITE of crowded side)
        if crowded_side == "long":
            trade_side = "short"  # Crowded longs → we short the reversal
        elif crowded_side == "short":
            trade_side = "long"  # Crowded shorts → we long the reversal
        else:
            # Use price direction for exhaustion-only setups
            if price_change > 0:
                trade_side = "short"  # Price went up, exhausted → short
            else:
                trade_side = "long"   # Price went down, exhausted → long

        # Expected return based on setup quality
        expected_return = 0.01 + (raw_confidence - 0.35) * 0.05  # 1-3.5% expected

        self.setups_detected += 1
        logger.debug(f"LCRS setup [{coin}]: {setup_type} → {trade_side.upper()} "
                     f"(confidence={raw_confidence:.0%}, funding={funding:.4f}, "
                     f"OI_chg={oi_change:.1%}, exhaustion={exhaustion_score:.2f})")

        return LiquidationSetup(
            coin=coin,
            side=trade_side,
            confidence=raw_confidence,
            expected_return=expected_return,
            funding_rate=funding,
            oi_change=oi_change,
            price_change=price_change,
            trend_strength=trend_strength,
            exhaustion_score=exhaustion_score,
            crowding_score=crowding_score,
            setup_type=setup_type,
        )

    def generate_signal(self, coin: str, features: Dict,
                         current_price: float) -> Optional[Dict]:
        """
        Generate a trade signal compatible with the paper_trader pipeline.

        Returns a signal dict or None.
        """
        setup = self.analyze(coin, features)
        if not setup:
            return None

        # Position sizing: scale by confidence
        position_pct = self.base_position_pct + (
            (self.max_position_pct - self.base_position_pct) *
            (setup.confidence - 0.35) / 0.60  # Scale from base to max
        )
        position_pct = min(position_pct, self.max_position_pct)

        leverage = min(
            1 + int(setup.confidence * 3),  # 1x-3x based on confidence
            self.max_leverage
        )

        # SL/TP adjusted by leverage
        if setup.side == "long":
            stop_loss = current_price * (1 - self.stop_loss_pct / leverage)
            take_profit = current_price * (1 + self.take_profit_pct / leverage)
        else:
            stop_loss = current_price * (1 + self.stop_loss_pct / leverage)
            take_profit = current_price * (1 - self.take_profit_pct / leverage)

        self.signals_generated += 1

        return {
            "coin": coin,
            "side": setup.side,
            "price": current_price,
            "size": 0,  # Will be calculated by paper_trader from position_pct
            "leverage": leverage,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "confidence": setup.confidence,
            "strategy_type": "liquidation_reversal",
            "position_pct": position_pct,
            "features": {
                "funding_rate": setup.funding_rate,
                "oi_change": setup.oi_change,
                "price_change": setup.price_change,
                "exhaustion_score": setup.exhaustion_score,
                "crowding_score": setup.crowding_score,
                "setup_type": setup.setup_type,
            },
            "source": "liquidation_strategy",
        }

    def scan_multiple(self, coins: List[str], features_map: Dict[str, Dict],
                       prices: Dict[str, float]) -> List[Dict]:
        """
        Scan multiple coins for liquidation setups.

        Args:
            coins: List of coin symbols to scan
            features_map: {coin: features_dict}
            prices: {coin: current_price}

        Returns:
            List of signal dicts, sorted by confidence (highest first).
        """
        signals = []
        for coin in coins:
            if coin not in features_map or coin not in prices:
                continue
            try:
                sig = self.generate_signal(coin, features_map[coin], prices[coin])
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"LCRS scan error for {coin}: {e}")

        # Sort by confidence descending
        signals.sort(key=lambda s: s["confidence"], reverse=True)
        return signals

    def get_stats(self) -> Dict:
        """Return strategy statistics."""
        return {
            "setups_detected": self.setups_detected,
            "signals_generated": self.signals_generated,
            "hit_rate": self.signals_generated / max(self.setups_detected, 1),
        }
