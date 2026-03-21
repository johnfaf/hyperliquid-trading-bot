"""
Market Regime Detection
=======================
Classifies market conditions into regimes so strategies can adapt.
Uses ADX (trend strength), ATR (volatility), and volume anomalies.

Regimes:
  TRENDING_UP    - Strong uptrend, momentum strategies excel
  TRENDING_DOWN  - Strong downtrend, short momentum / trend-following
  RANGING        - Sideways chop, mean reversion works
  VOLATILE       - High volatility without clear trend, reduce size
  LOW_LIQUIDITY  - Thin order books, avoid trading

Each regime maps to which strategy types should be active vs paused.
"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import hyperliquid_client as hl
from src.exchange_aggregator import ExchangeAggregator

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    UNKNOWN = "unknown"


# Maps regime → which strategy types should be active
REGIME_STRATEGY_MAP = {
    Regime.TRENDING_UP: {
        "activate": ["momentum_long", "trend_following", "breakout", "swing_trading"],
        "pause": ["mean_reversion", "funding_arb", "delta_neutral"],
        "size_modifier": 1.0,  # Full size
    },
    Regime.TRENDING_DOWN: {
        "activate": ["momentum_short", "trend_following", "breakout"],
        "pause": ["momentum_long", "funding_arb", "delta_neutral"],
        "size_modifier": 0.8,  # Slightly reduced (shorts are riskier)
    },
    Regime.RANGING: {
        "activate": ["mean_reversion", "funding_arb", "delta_neutral", "scalping"],
        "pause": ["momentum_long", "momentum_short", "breakout"],
        "size_modifier": 0.7,  # Reduced size in chop
    },
    Regime.VOLATILE: {
        "activate": ["scalping"],  # Only fast in/out
        "pause": ["momentum_long", "momentum_short", "swing_trading", "breakout",
                   "mean_reversion", "trend_following"],
        "size_modifier": 0.4,  # Heavily reduced
    },
    Regime.LOW_LIQUIDITY: {
        "activate": [],  # Don't trade
        "pause": ["all"],
        "size_modifier": 0.0,
    },
    Regime.UNKNOWN: {
        "activate": ["mean_reversion", "funding_arb"],  # Conservative defaults
        "pause": ["momentum_long", "momentum_short", "breakout"],
        "size_modifier": 0.5,
    },
}


@dataclass
class RegimeState:
    """Current market regime for a specific asset or overall market."""
    regime: Regime
    confidence: float       # 0-1, how confident we are in this classification
    adx: float              # Average Directional Index (0-100, >25 = trending)
    atr_pct: float          # ATR as % of price (volatility measure)
    volume_ratio: float     # Current volume / 20-period avg volume
    trend_direction: float  # Positive = up, negative = down
    momentum: float         # Rate of change
    timestamp: str

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


class RegimeDetector:
    """
    Detects market regime from price data.

    Uses a combination of:
    - ADX (trend strength): >25 = trending, <20 = ranging
    - ATR% (volatility): historical percentile ranking
    - Volume ratio: current vs average (spikes = regime change)
    - Price momentum: direction and acceleration

    Can work with Hyperliquid candle data or multi-exchange aggregated data.
    """

    def __init__(self, exchange_agg: Optional[ExchangeAggregator] = None):
        self.exchange_agg = exchange_agg
        self._cache: Dict[str, RegimeState] = {}
        self._cache_ts: Dict[str, float] = {}
        self._history: Dict[str, List[RegimeState]] = defaultdict(list)
        self.CACHE_TTL = 120  # 2 minutes
        logger.info("RegimeDetector initialized")

    # ─── Core Detection ───────────────────────────────────────

    def detect_regime(self, coin: str, candles: Optional[List[Dict]] = None) -> RegimeState:
        """
        Detect the current market regime for a coin.

        If candles are not provided, fetches from Hyperliquid.
        Candles should be OHLCV dicts with keys: open, high, low, close, volume.

        Returns RegimeState with classification and metrics.
        """
        # Check cache
        now = time.time()
        if coin in self._cache and (now - self._cache_ts.get(coin, 0)) < self.CACHE_TTL:
            return self._cache[coin]

        # Fetch candles if not provided
        if not candles:
            candles = self._fetch_candles(coin)

        if not candles or len(candles) < 30:
            logger.debug(f"Insufficient data for {coin} regime detection ({len(candles) if candles else 0} candles)")
            return RegimeState(
                regime=Regime.UNKNOWN, confidence=0.0,
                adx=0, atr_pct=0, volume_ratio=0,
                trend_direction=0, momentum=0,
                timestamp=datetime.utcnow().isoformat(),
            )

        # Extract OHLCV arrays
        closes = np.array([c["close"] for c in candles], dtype=float)
        highs = np.array([c["high"] for c in candles], dtype=float)
        lows = np.array([c["low"] for c in candles], dtype=float)
        volumes = np.array([c.get("volume", 0) for c in candles], dtype=float)

        # Calculate indicators
        adx = self._calculate_adx(highs, lows, closes, period=14)
        atr_pct = self._calculate_atr_pct(highs, lows, closes, period=14)
        volume_ratio = self._calculate_volume_ratio(volumes, period=20)
        trend_dir = self._calculate_trend_direction(closes, period=20)
        momentum = self._calculate_momentum(closes, period=10)

        # Classify regime
        regime, confidence = self._classify_regime(
            adx, atr_pct, volume_ratio, trend_dir, momentum
        )

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            adx=round(adx, 2),
            atr_pct=round(atr_pct, 4),
            volume_ratio=round(volume_ratio, 2),
            trend_direction=round(trend_dir, 4),
            momentum=round(momentum, 4),
            timestamp=datetime.utcnow().isoformat(),
        )

        # Cache and log
        self._cache[coin] = state
        self._cache_ts[coin] = now
        self._history[coin].append(state)
        # Keep last 100 regime states per coin
        self._history[coin] = self._history[coin][-100:]

        logger.info(f"Regime {coin}: {regime.value} (confidence={confidence:.0%}, "
                    f"ADX={adx:.1f}, ATR%={atr_pct:.2%}, vol_ratio={volume_ratio:.1f}x, "
                    f"trend={trend_dir:+.3f}, momentum={momentum:+.3f})")

        return state

    def _classify_regime(self, adx: float, atr_pct: float,
                         volume_ratio: float, trend_dir: float,
                         momentum: float) -> Tuple[Regime, float]:
        """
        Classify regime from computed indicators.

        Decision tree:
        1. If volume_ratio < 0.3 → LOW_LIQUIDITY
        2. If atr_pct > 95th percentile AND adx < 20 → VOLATILE (chaos)
        3. If adx > 25 → TRENDING (direction from trend_dir)
        4. If adx < 20 → RANGING
        5. Otherwise → UNKNOWN with moderate confidence

        ATR% thresholds (crypto-adjusted):
        - Normal: 1-3%
        - High: 3-5%
        - Extreme: >5%
        """
        confidence = 0.5  # Base confidence

        # 1. Low liquidity check
        if volume_ratio < 0.3:
            confidence = 0.7 + (0.3 - volume_ratio) * 0.5
            return Regime.LOW_LIQUIDITY, min(confidence, 0.95)

        # 2. High volatility without trend = chaos
        if atr_pct > 0.05 and adx < 20:
            confidence = 0.6 + min(atr_pct - 0.05, 0.05) * 4
            return Regime.VOLATILE, min(confidence, 0.95)

        # 3. Strong trend
        if adx > 25:
            # Higher ADX = more confidence in trend
            trend_confidence = 0.5 + (adx - 25) / 50  # 25→0.5, 75→1.0
            trend_confidence = min(trend_confidence, 0.95)

            # Use trend direction + momentum for up/down
            if trend_dir > 0 and momentum > 0:
                return Regime.TRENDING_UP, trend_confidence
            elif trend_dir < 0 and momentum < 0:
                return Regime.TRENDING_DOWN, trend_confidence
            elif trend_dir > 0:
                # Trend up but momentum slowing — less confident
                return Regime.TRENDING_UP, trend_confidence * 0.7
            else:
                return Regime.TRENDING_DOWN, trend_confidence * 0.7

        # 4. Ranging (low ADX)
        if adx < 20:
            range_confidence = 0.5 + (20 - adx) / 20 * 0.3
            # If volatility is also low, more confident it's ranging
            if atr_pct < 0.02:
                range_confidence += 0.15
            return Regime.RANGING, min(range_confidence, 0.95)

        # 5. Transition zone (ADX 20-25) — uncertain
        return Regime.UNKNOWN, 0.4

    # ─── Technical Indicators ─────────────────────────────────

    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX).
        Measures trend strength regardless of direction.
        0-20: weak/no trend, 20-40: strong trend, 40+: very strong trend
        """
        n = len(closes)
        if n < period + 1:
            return 0.0

        # True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )

        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smoothed averages (Wilder's smoothing)
        atr = self._wilder_smooth(tr, period)
        plus_di = self._wilder_smooth(plus_dm, period)
        minus_di = self._wilder_smooth(minus_dm, period)

        # Directional Indicators
        if atr[-1] > 0:
            plus_di_val = (plus_di[-1] / atr[-1]) * 100
            minus_di_val = (minus_di[-1] / atr[-1]) * 100
        else:
            return 0.0

        # DX and ADX
        di_sum = plus_di_val + minus_di_val
        if di_sum > 0:
            dx = abs(plus_di_val - minus_di_val) / di_sum * 100
        else:
            dx = 0

        return dx  # Simplified: use latest DX as proxy for ADX

    def _calculate_atr_pct(self, highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR as percentage of price (normalized volatility)."""
        n = len(closes)
        if n < period + 1:
            return 0.0

        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )

        atr = np.mean(tr[-period:])
        current_price = closes[-1]
        return atr / current_price if current_price > 0 else 0.0

    def _calculate_volume_ratio(self, volumes: np.ndarray, period: int = 20) -> float:
        """Current volume vs average of last N periods."""
        if len(volumes) < period + 1:
            return 1.0
        avg_vol = np.mean(volumes[-period - 1:-1])  # Exclude current
        current_vol = volumes[-1]
        return current_vol / avg_vol if avg_vol > 0 else 1.0

    def _calculate_trend_direction(self, closes: np.ndarray, period: int = 20) -> float:
        """
        Linear regression slope over last N periods, normalized by price.
        Positive = uptrend, negative = downtrend.
        """
        if len(closes) < period:
            return 0.0
        recent = closes[-period:]
        x = np.arange(period)
        # Linear regression slope
        slope = np.polyfit(x, recent, 1)[0]
        # Normalize by current price
        return slope / recent[-1] if recent[-1] > 0 else 0.0

    def _calculate_momentum(self, closes: np.ndarray, period: int = 10) -> float:
        """Rate of change over last N periods, as a fraction."""
        if len(closes) < period + 1:
            return 0.0
        return (closes[-1] - closes[-period]) / closes[-period] if closes[-period] > 0 else 0.0

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's exponential smoothing (used in ADX calculation)."""
        result = np.zeros(len(data))
        if len(data) < period:
            return result
        result[period] = np.mean(data[1:period + 1])
        for i in range(period + 1, len(data)):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period
        return result

    # ─── Data Fetching ────────────────────────────────────────

    def _fetch_candles(self, coin: str, interval: str = "1h", count: int = 100) -> List[Dict]:
        """
        Fetch OHLCV candles from Hyperliquid.
        Falls back to computing from recent trades if candle endpoint unavailable.
        """
        try:
            # Try Hyperliquid candle snapshot endpoint
            import requests
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": int((datetime.utcnow().timestamp() - count * 3600) * 1000),
                    "endTime": int(datetime.utcnow().timestamp() * 1000),
                }
            }
            resp = requests.post("https://api.hyperliquid.xyz/info",
                                 json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    candles = []
                    for c in data:
                        # Hyperliquid candle format: {t, T, s, o, c, h, l, v, n}
                        candles.append({
                            "timestamp": c.get("t", 0),
                            "open": float(c.get("o", 0)),
                            "high": float(c.get("h", 0)),
                            "low": float(c.get("l", 0)),
                            "close": float(c.get("c", 0)),
                            "volume": float(c.get("v", 0)),
                        })
                    return candles

        except Exception as e:
            logger.debug(f"Candle fetch for {coin}: {e}")

        # Fallback: use mid prices to build synthetic candles
        try:
            mids = hl.get_all_mids()
            if mids and coin in mids:
                price = mids[coin]
                # Return a minimal single-candle (can't compute indicators well)
                return [{"open": price, "high": price * 1.001,
                         "low": price * 0.999, "close": price, "volume": 1}]
        except Exception:
            pass

        return []

    # ─── Multi-Asset Overview ─────────────────────────────────

    def get_market_regime(self, coins: Optional[List[str]] = None) -> Dict:
        """
        Detect regime for multiple coins and compute an overall market regime.

        Returns:
        {
            "overall_regime": "trending_up",
            "overall_confidence": 0.72,
            "per_coin": {"BTC": RegimeState, "ETH": RegimeState, ...},
            "strategy_guidance": {"activate": [...], "pause": [...], "size_modifier": 0.8},
        }
        """
        if not coins:
            coins = ["BTC", "ETH", "SOL", "DOGE", "ARB", "AVAX", "MATIC",
                     "LINK", "OP", "SUI"]

        per_coin = {}
        regime_votes = defaultdict(float)

        for coin in coins:
            try:
                state = self.detect_regime(coin)
                per_coin[coin] = state

                # Weight by confidence — BTC/ETH get 2x weight
                weight = state.confidence
                if coin in ("BTC", "ETH"):
                    weight *= 2.0
                regime_votes[state.regime] += weight

            except Exception as e:
                logger.debug(f"Regime detection failed for {coin}: {e}")
            time.sleep(0.2)

        # Determine overall regime by weighted vote
        if regime_votes:
            total_weight = sum(regime_votes.values())
            overall_regime = max(regime_votes, key=regime_votes.get)
            overall_confidence = regime_votes[overall_regime] / total_weight if total_weight > 0 else 0
        else:
            overall_regime = Regime.UNKNOWN
            overall_confidence = 0.0

        # Get strategy guidance for the overall regime
        guidance = REGIME_STRATEGY_MAP.get(overall_regime, REGIME_STRATEGY_MAP[Regime.UNKNOWN])

        result = {
            "overall_regime": overall_regime.value,
            "overall_confidence": round(overall_confidence, 3),
            "per_coin": {coin: state.to_dict() for coin, state in per_coin.items()},
            "regime_votes": {r.value: round(w, 2) for r, w in regime_votes.items()},
            "strategy_guidance": guidance,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Market regime: {overall_regime.value} (confidence={overall_confidence:.0%})")
        logger.info(f"  Activate: {guidance['activate']}")
        logger.info(f"  Pause: {guidance['pause']}")
        logger.info(f"  Size modifier: {guidance['size_modifier']:.0%}")

        return result

    # ─── Strategy Filtering ───────────────────────────────────

    def filter_strategies_by_regime(self, strategies: List[Dict],
                                     regime_data: Optional[Dict] = None) -> List[Dict]:
        """
        Filter and adjust strategies based on current market regime.
        Pauses strategies that don't fit the regime, boosts those that do.

        Returns filtered list with adjusted confidence scores.
        """
        if not regime_data:
            regime_data = self.get_market_regime()

        guidance = regime_data.get("strategy_guidance", {})
        active_types = set(guidance.get("activate", []))
        paused_types = set(guidance.get("pause", []))
        size_mod = guidance.get("size_modifier", 1.0)
        regime_confidence = regime_data.get("overall_confidence", 0.5)

        filtered = []
        for strategy in strategies:
            stype = strategy.get("strategy_type", "").lower()

            # Skip paused strategies (unless "all" is in pause list)
            if "all" in paused_types or stype in paused_types:
                logger.debug(f"Regime pauses {stype}: {strategy.get('name', '?')}")
                continue

            # Copy strategy and adjust score
            adjusted = dict(strategy)

            if stype in active_types:
                # Boost active strategies proportional to regime confidence
                boost = 1.0 + regime_confidence * 0.3
                adjusted["current_score"] = strategy.get("current_score", 0) * boost
                adjusted["regime_boost"] = boost
                adjusted["regime_status"] = "active"
            else:
                # Neutral strategies — slight reduction
                adjusted["current_score"] = strategy.get("current_score", 0) * 0.8
                adjusted["regime_boost"] = 0.8
                adjusted["regime_status"] = "neutral"

            # Apply size modifier
            adjusted["regime_size_modifier"] = size_mod

            filtered.append(adjusted)

        # Re-sort by adjusted score
        filtered.sort(key=lambda s: s.get("current_score", 0), reverse=True)

        logger.info(f"Regime filter: {len(strategies)} → {len(filtered)} strategies "
                    f"(regime={regime_data.get('overall_regime', '?')}, "
                    f"size_mod={size_mod:.0%})")

        return filtered
