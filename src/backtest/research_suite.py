"""Research helpers for cross-coin and regime-conditional backtests."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.backtest.candle_backtester import (
    CandleBacktestConfig,
    CandleBacktester,
    STRATEGY_MAP,
)

logger = logging.getLogger(__name__)

DEFAULT_10_COIN_UNIVERSE = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "AVAX",
    "LINK",
    "ARB",
    "OP",
)

DEFAULT_12_STRATEGIES = (
    "momentum",
    "mean_reversion",
    "breakout",
    "rsi",
    "macd",
    "vwap_reversion",
    "stochastic",
    "adx_trend",
    "supertrend",
    "ema_rsi_combo",
    "volume_breakout",
    "ichimoku",
)

TREND_STRATEGIES = {
    "momentum",
    "breakout",
    "macd",
    "macd_histogram",
    "adx_trend",
    "supertrend",
    "ema_rsi_combo",
    "volume_breakout",
    "ichimoku",
    "ma_crossover",
}
RANGE_STRATEGIES = {"mean_reversion", "rsi", "vwap_reversion", "stochastic"}
VOLATILE_STRATEGIES = {"breakout", "supertrend", "adx_trend", "volume_breakout"}


@dataclass
class RegimeSegmentResult:
    coin: str
    strategy: str
    regime: str
    allowed: bool
    start_ts: int
    end_ts: int
    candle_count: int
    total_trades: int
    total_pnl: float
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    error: str = ""


@dataclass
class RegimeGateReport:
    summary: Dict[str, Any]
    by_strategy: List[Dict[str, Any]] = field(default_factory=list)
    by_coin: List[Dict[str, Any]] = field(default_factory=list)
    segments: List[RegimeSegmentResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["segments"] = [asdict(item) for item in self.segments]
        return out


def _candle_get(candle: Any, key: str, default: Any = None) -> Any:
    if isinstance(candle, Mapping):
        return candle.get(key, default)
    return getattr(candle, key, default)


def _timestamp(candle: Any) -> int:
    return int(
        _candle_get(
            candle,
            "timestamp_ms",
            _candle_get(candle, "t", _candle_get(candle, "time", 0)),
        )
        or 0
    )


def _close(candle: Any) -> float:
    return float(_candle_get(candle, "close", _candle_get(candle, "c", 0.0)) or 0.0)


def infer_candle_regime(candles: Sequence[Any]) -> str:
    """Infer a coarse regime from a candle segment.

    This is intentionally simple and deterministic for research. Live regime
    detection can be richer, but the backtest gate needs a transparent baseline.
    """
    closes = [_close(c) for c in candles if _close(c) > 0]
    if len(closes) < 8:
        return "unknown"
    start = closes[0]
    end = closes[-1]
    total_return = (end / start) - 1.0 if start else 0.0
    returns = [
        (closes[i] / closes[i - 1]) - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    avg_abs_return = sum(abs(r) for r in returns) / len(returns) if returns else 0.0
    realized_vol = (sum((r - (sum(returns) / len(returns))) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0.0

    if total_return <= -0.08 and realized_vol > 0.015:
        return "crash"
    if realized_vol > max(0.035, avg_abs_return * 3.0):
        return "volatile"
    trend_threshold = max(0.012, avg_abs_return * 2.5)
    if total_return > trend_threshold:
        return "trending_up"
    if total_return < -trend_threshold:
        return "trending_down"
    return "ranging"


def strategy_allowed_in_regime(strategy: str, regime: str) -> bool:
    strategy = str(strategy or "").strip().lower()
    regime = str(regime or "").strip().lower()
    if regime in {"trending_up", "trending_down", "trend", "bullish", "bearish", "crash"}:
        return strategy in TREND_STRATEGIES
    if regime in {"ranging", "range", "neutral", "low_volatility"}:
        return strategy in RANGE_STRATEGIES
    if regime in {"volatile", "high_volatility"}:
        return strategy in VOLATILE_STRATEGIES
    return True


def _aggregate_segments(segments: Iterable[RegimeSegmentResult], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for seg in segments:
        label = getattr(seg, key)
        bucket = buckets.setdefault(
            label,
            {
                key: label,
                "segments": 0,
                "allowed_segments": 0,
                "always_trades": 0,
                "always_pnl": 0.0,
                "gated_trades": 0,
                "gated_pnl": 0.0,
                "always_max_drawdown_pct": 0.0,
                "gated_max_drawdown_pct": 0.0,
            },
        )
        bucket["segments"] += 1
        bucket["always_trades"] += int(seg.total_trades or 0)
        bucket["always_pnl"] += float(seg.total_pnl or 0.0)
        bucket["always_max_drawdown_pct"] = max(
            bucket["always_max_drawdown_pct"],
            float(seg.max_drawdown_pct or 0.0),
        )
        if seg.allowed:
            bucket["allowed_segments"] += 1
            bucket["gated_trades"] += int(seg.total_trades or 0)
            bucket["gated_pnl"] += float(seg.total_pnl or 0.0)
            bucket["gated_max_drawdown_pct"] = max(
                bucket["gated_max_drawdown_pct"],
                float(seg.max_drawdown_pct or 0.0),
            )

    rows = []
    for bucket in buckets.values():
        always_pnl = float(bucket["always_pnl"])
        gated_pnl = float(bucket["gated_pnl"])
        bucket["always_pnl"] = round(always_pnl, 4)
        bucket["gated_pnl"] = round(gated_pnl, 4)
        bucket["pnl_delta"] = round(gated_pnl - always_pnl, 4)
        bucket["beats_always_on"] = gated_pnl > always_pnl
        rows.append(bucket)
    rows.sort(key=lambda row: (row["pnl_delta"], row["gated_pnl"]), reverse=True)
    return rows


def run_regime_conditional_strategy_enablement(
    candle_sets: Mapping[str, Sequence[Any]],
    *,
    strategies: Optional[Sequence[str]] = None,
    config: Optional[CandleBacktestConfig] = None,
    segment_candles: int = 168,
    min_segment_candles: int = 60,
) -> RegimeGateReport:
    """Compare always-on strategies with regime-gated enablement."""
    strategies = [
        s for s in (strategies or DEFAULT_12_STRATEGIES)
        if s in STRATEGY_MAP
    ]
    cfg = config or CandleBacktestConfig()
    backtester = CandleBacktester(cfg)
    segments: List[RegimeSegmentResult] = []
    errors = []

    segment_candles = max(10, int(segment_candles))
    min_segment_candles = max(5, int(min_segment_candles))

    for coin, candles_raw in candle_sets.items():
        candles = list(candles_raw or [])
        if len(candles) < min_segment_candles:
            errors.append({"coin": coin, "error": "insufficient_candles", "candles": len(candles)})
            continue
        for start in range(0, len(candles), segment_candles):
            segment = candles[start:start + segment_candles]
            if len(segment) < min_segment_candles:
                continue
            regime = infer_candle_regime(segment)
            for strategy in strategies:
                allowed = strategy_allowed_in_regime(strategy, regime)
                try:
                    result = backtester.run(
                        segment,
                        strategy=strategy,
                        coin=str(coin).upper(),
                        experiment_id=f"regime_gate_{coin}_{strategy}_{start}",
                    )
                    segments.append(
                        RegimeSegmentResult(
                            coin=str(coin).upper(),
                            strategy=strategy,
                            regime=regime,
                            allowed=allowed,
                            start_ts=_timestamp(segment[0]),
                            end_ts=_timestamp(segment[-1]),
                            candle_count=len(segment),
                            total_trades=int(result.total_trades or 0),
                            total_pnl=float(result.total_pnl or 0.0),
                            win_rate=float(result.win_rate or 0.0),
                            max_drawdown_pct=float(result.max_drawdown_pct or 0.0),
                            sharpe_ratio=float(result.sharpe_ratio or 0.0),
                        )
                    )
                except Exception as exc:
                    logger.debug("Regime gate backtest failed %s/%s: %s", coin, strategy, exc)
                    segments.append(
                        RegimeSegmentResult(
                            coin=str(coin).upper(),
                            strategy=strategy,
                            regime=regime,
                            allowed=allowed,
                            start_ts=_timestamp(segment[0]),
                            end_ts=_timestamp(segment[-1]),
                            candle_count=len(segment),
                            total_trades=0,
                            total_pnl=0.0,
                            win_rate=0.0,
                            max_drawdown_pct=0.0,
                            sharpe_ratio=0.0,
                            error=str(exc),
                        )
                    )

    by_strategy = _aggregate_segments(segments, "strategy")
    by_coin = _aggregate_segments(segments, "coin")
    always_pnl = sum(float(seg.total_pnl or 0.0) for seg in segments)
    gated_pnl = sum(float(seg.total_pnl or 0.0) for seg in segments if seg.allowed)
    summary = {
        "coins": len(candle_sets),
        "strategies": len(strategies),
        "segments": len(segments),
        "allowed_segments": sum(1 for seg in segments if seg.allowed),
        "always_on_pnl": round(always_pnl, 4),
        "regime_gated_pnl": round(gated_pnl, 4),
        "pnl_delta": round(gated_pnl - always_pnl, 4),
        "beats_always_on": gated_pnl > always_pnl,
        "errors": errors,
    }
    return RegimeGateReport(summary=summary, by_strategy=by_strategy, by_coin=by_coin, segments=segments)

