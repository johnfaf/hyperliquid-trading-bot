"""
Feature / Signal Layer
======================
Transforms raw market data into structured features before they reach
strategy identification or agent decision-making.

This reduces hallucination risk (if LLMs are used) and makes signals
consistent and comparable across different data sources.

Features computed:
  - Trend strength (ADX-based)
  - Volatility (ATR%, Bollinger width)
  - Momentum (RSI, ROC, MACD-like)
  - Liquidity (spread, depth, volume profile)
  - Funding rate signal
"""
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MarketFeatures:
    """Structured feature vector for a single coin at a point in time."""
    coin: str

    # Trend
    trend_strength: float = 0.0    # 0-1, from ADX (0=no trend, 1=very strong)
    trend_direction: float = 0.0   # -1 to +1 (negative=down, positive=up)
    trend_duration: int = 0        # Candles in current trend direction

    # Volatility
    volatility: float = 0.0        # ATR% (0-1 range, typical 0.01-0.05 for crypto)
    volatility_rank: float = 0.0   # Percentile rank vs last 50 periods (0-1)
    bollinger_width: float = 0.0   # BB width as % of price
    bollinger_position: float = 0.0  # Where price sits in BB (-1=lower, 0=mid, +1=upper)

    # Momentum
    rsi: float = 50.0              # RSI (0-100)
    rsi_signal: str = "neutral"    # "oversold", "neutral", "overbought"
    rate_of_change: float = 0.0    # % change over lookback
    momentum_score: float = 0.0    # Composite momentum (-1 to +1)

    # Liquidity
    spread_bps: float = 0.0       # Bid-ask spread in basis points
    volume_ratio: float = 1.0     # Current volume vs 20-period average
    volume_trend: str = "normal"  # "surging", "normal", "drying_up"

    # Funding
    funding_rate: float = 0.0     # Current funding rate
    funding_signal: str = "neutral"  # "extreme_long" / "extreme_short" / "neutral"

    # Composite
    overall_score: float = 0.0    # -1 (very bearish) to +1 (very bullish)
    confidence: float = 0.0       # How reliable these features are (data quality)

    def to_dict(self) -> Dict:
        return asdict(self)


class FeatureEngine:
    """
    Computes structured features from raw OHLCV + market data.

    Usage:
        engine = FeatureEngine()
        features = engine.compute(coin="BTC", candles=candles, funding=0.0001)
    """

    def compute(self, coin: str, candles: List[Dict],
                funding_rate: float = 0.0,
                spread_bps: float = 0.0) -> MarketFeatures:
        """Compute all features from candle data."""
        if not candles or len(candles) < 20:
            return MarketFeatures(coin=coin, confidence=0.0)

        closes = np.array([c["close"] for c in candles], dtype=float)
        highs = np.array([c["high"] for c in candles], dtype=float)
        lows = np.array([c["low"] for c in candles], dtype=float)
        volumes = np.array([c.get("volume", 0) for c in candles], dtype=float)

        features = MarketFeatures(coin=coin)

        # ─── Trend ────────────────────────────────────────────
        trend_dir = self._linear_regression_slope(closes, period=20)
        features.trend_direction = np.clip(trend_dir * 100, -1, 1)  # Normalize

        # Trend strength from price consistency
        if len(closes) >= 20:
            recent = closes[-20:]
            up_moves = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
            features.trend_strength = abs(up_moves / 19 - 0.5) * 2  # 0=choppy, 1=consistent

        # Trend duration
        features.trend_duration = self._count_trend_duration(closes)

        # ─── Volatility ──────────────────────────────────────
        atr_pct = self._atr_percent(highs, lows, closes, period=14)
        features.volatility = atr_pct

        # Volatility rank (percentile)
        if len(closes) >= 50:
            atr_series = []
            for i in range(14, min(len(closes), 64)):
                a = self._atr_percent(highs[:i+1], lows[:i+1], closes[:i+1], period=14)
                atr_series.append(a)
            if atr_series:
                features.volatility_rank = sum(1 for a in atr_series if a <= atr_pct) / len(atr_series)

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(closes, period=20, std_dev=2)
        if bb_upper > bb_lower:
            features.bollinger_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
            features.bollinger_position = (closes[-1] - bb_mid) / (bb_upper - bb_mid) if bb_upper != bb_mid else 0

        # ─── Momentum ────────────────────────────────────────
        features.rsi = self._rsi(closes, period=14)
        if features.rsi < 30:
            features.rsi_signal = "oversold"
        elif features.rsi > 70:
            features.rsi_signal = "overbought"
        else:
            features.rsi_signal = "neutral"

        # Rate of change (10 periods)
        if len(closes) >= 11:
            features.rate_of_change = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0

        # Composite momentum score (-1 to +1)
        rsi_score = (features.rsi - 50) / 50  # -1 to +1
        roc_score = np.clip(features.rate_of_change * 10, -1, 1)
        features.momentum_score = 0.5 * rsi_score + 0.5 * roc_score

        # ─── Liquidity ───────────────────────────────────────
        features.spread_bps = spread_bps

        if len(volumes) >= 21:
            avg_vol = np.mean(volumes[-21:-1])
            features.volume_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
            if features.volume_ratio > 2.0:
                features.volume_trend = "surging"
            elif features.volume_ratio < 0.5:
                features.volume_trend = "drying_up"
            else:
                features.volume_trend = "normal"

        # ─── Funding ─────────────────────────────────────────
        features.funding_rate = funding_rate
        if funding_rate > 0.0005:
            features.funding_signal = "extreme_long"
        elif funding_rate < -0.0005:
            features.funding_signal = "extreme_short"
        else:
            features.funding_signal = "neutral"

        # ─── Composite Score ─────────────────────────────────
        # Bullish signals: positive trend, RSI not overbought, volume surging, funding not extreme long
        bullish = 0.0
        bearish = 0.0

        bullish += max(features.trend_direction, 0) * 0.3
        bearish += abs(min(features.trend_direction, 0)) * 0.3
        bullish += max(features.momentum_score, 0) * 0.25
        bearish += abs(min(features.momentum_score, 0)) * 0.25

        # Contrarian funding signal
        if features.funding_signal == "extreme_long":
            bearish += 0.15  # Crowded long = bearish signal
        elif features.funding_signal == "extreme_short":
            bullish += 0.15

        # Volume confirmation
        if features.volume_ratio > 1.5:
            if features.trend_direction > 0:
                bullish += 0.1
            else:
                bearish += 0.1

        # Bollinger mean reversion
        if features.bollinger_position < -0.8:
            bullish += 0.15  # Near lower band
        elif features.bollinger_position > 0.8:
            bearish += 0.15  # Near upper band

        features.overall_score = np.clip(bullish - bearish, -1, 1)
        features.confidence = min(len(candles) / 50, 1.0)  # More data = higher confidence

        return features

    # ─── Indicator Implementations ────────────────────────────

    def _linear_regression_slope(self, data: np.ndarray, period: int = 20) -> float:
        if len(data) < period:
            return 0.0
        recent = data[-period:]
        x = np.arange(period)
        slope = np.polyfit(x, recent, 1)[0]
        return slope / recent[-1] if recent[-1] > 0 else 0.0

    def _atr_percent(self, highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
        n = len(closes)
        if n < period + 1:
            return 0.0
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr = np.mean(tr[-period:])
        return atr / closes[-1] if closes[-1] > 0 else 0.0

    def _bollinger_bands(self, closes: np.ndarray, period: int = 20,
                         std_dev: float = 2.0):
        if len(closes) < period:
            p = closes[-1] if len(closes) > 0 else 0
            return p, p, p
        recent = closes[-period:]
        mid = np.mean(recent)
        std = np.std(recent)
        return mid + std_dev * std, mid, mid - std_dev * std

    def _rsi(self, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _count_trend_duration(self, closes: np.ndarray) -> int:
        if len(closes) < 2:
            return 0
        count = 0
        direction = 1 if closes[-1] > closes[-2] else -1
        for i in range(len(closes) - 2, 0, -1):
            if direction > 0 and closes[i] > closes[i - 1]:
                count += 1
            elif direction < 0 and closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count
