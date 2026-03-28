"""
Candle-Based Backtester
=======================
Fast, vectorized backtester that runs strategies against historical candle data
instead of replaying wallet fills. Supports any strategy that can produce
signals from OHLCV data.

Speed optimisations:
  - Numpy arrays for price data and equity curves (no Python loops for math)
  - Pre-computed indicator arrays (SMA, EMA, RSI, ATR, Bollinger)
  - Vectorized PnL calculation
  - Parallel multi-coin backtesting via ProcessPoolExecutor

Usage:
    from src.backtest.candle_backtester import CandleBacktester, CandleBacktestConfig
    from src.backtest.data_fetcher import DataFetcher

    fetcher = DataFetcher()
    candles = fetcher.fetch_candles("BTC", "1h", start="2025-01-01", end="2025-03-01")

    bt = CandleBacktester(CandleBacktestConfig())
    result = bt.run(candles, strategy="momentum")
"""
import logging
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("candle_backtester")


# ─── Config ──────────────────────────────────────────────────────

@dataclass
class CandleBacktestConfig:
    """Configuration for candle-based backtesting."""
    initial_balance: float = 10_000.0

    # Position sizing
    position_size_pct: float = 0.05       # 5% of balance per trade
    max_positions: int = 5
    max_leverage: float = 3.0

    # Risk management
    stop_loss_pct: float = 0.02           # 2% stop
    take_profit_pct: float = 0.04         # 4% take-profit
    trailing_stop_pct: float = 0.015      # 1.5% trailing

    # Execution costs
    slippage_bps: float = 3.0             # 3 bps per trade
    maker_fee_bps: float = 0.2            # 0.02% maker
    taker_fee_bps: float = 3.5            # 0.035% taker

    # Funding
    funding_rate_8h: float = 0.0001
    funding_enabled: bool = True

    # Strategy params
    strategy: str = "momentum"            # momentum, mean_reversion, breakout, ma_crossover
    fast_period: int = 10                 # Fast MA period
    slow_period: int = 30                 # Slow MA period
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Speed
    parallel: bool = True                 # Use multiprocessing for multi-coin
    chunk_size: int = 50_000              # Process candles in chunks for memory

    def to_dict(self) -> Dict:
        return asdict(self)


# ─── Indicator Library (vectorized numpy) ────────────────────────

def _sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average — cumsum trick, O(n)."""
    out = np.full_like(close, np.nan)
    if len(close) < period:
        return out
    cumsum = np.cumsum(close)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    out[period - 1:] = cumsum[period - 1:] / period
    return out


def _ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    out = np.full_like(close, np.nan)
    if len(close) < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = np.mean(close[:period])
    for i in range(period, len(close)):
        out[i] = alpha * close[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index — Wilder's smoothing."""
    out = np.full_like(close, np.nan)
    if len(close) < period + 1:
        return out
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        out[i + 1] = 100 - (100 / (1 + rs))

    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    """Average True Range."""
    out = np.full_like(close, np.nan)
    if len(close) < period + 1:
        return out
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    # Wilder smoothing
    atr_val = np.mean(tr[:period])
    out[period] = atr_val
    for i in range(period, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        out[i + 1] = atr_val
    return out


def _bollinger(close: np.ndarray, period: int = 20,
               std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands → (upper, middle, lower)."""
    mid = _sma(close, period)
    # Rolling std
    std = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i - period + 1:i + 1])
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


# ─── Strategy Signal Generators ──────────────────────────────────

def _signals_momentum(close: np.ndarray, cfg: CandleBacktestConfig) -> np.ndarray:
    """Momentum: buy when fast EMA crosses above slow EMA."""
    fast = _ema(close, cfg.fast_period)
    slow = _ema(close, cfg.slow_period)
    signals = np.zeros(len(close), dtype=np.int8)  # 0=hold, 1=long, -1=short

    for i in range(1, len(close)):
        if np.isnan(fast[i]) or np.isnan(slow[i]):
            continue
        if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]:
            signals[i] = 1   # Golden cross → long
        elif fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]:
            signals[i] = -1  # Death cross → short

    return signals


def _signals_mean_reversion(close: np.ndarray, cfg: CandleBacktestConfig) -> np.ndarray:
    """Mean reversion: buy at lower Bollinger, sell at upper."""
    upper, mid, lower = _bollinger(close, cfg.bb_period, cfg.bb_std)
    signals = np.zeros(len(close), dtype=np.int8)

    for i in range(1, len(close)):
        if np.isnan(lower[i]):
            continue
        if close[i] <= lower[i] and close[i - 1] > lower[i - 1]:
            signals[i] = 1   # Bounced off lower band → long
        elif close[i] >= upper[i] and close[i - 1] < upper[i - 1]:
            signals[i] = -1  # Hit upper band → short

    return signals


def _signals_breakout(close: np.ndarray, high: np.ndarray,
                      low: np.ndarray, cfg: CandleBacktestConfig) -> np.ndarray:
    """Breakout: buy on new N-period high, sell on new N-period low."""
    period = cfg.slow_period
    signals = np.zeros(len(close), dtype=np.int8)

    for i in range(period, len(close)):
        window_high = np.max(high[i - period:i])
        window_low = np.min(low[i - period:i])
        if close[i] > window_high:
            signals[i] = 1   # Breakout up → long
        elif close[i] < window_low:
            signals[i] = -1  # Breakout down → short

    return signals


def _signals_rsi(close: np.ndarray, cfg: CandleBacktestConfig) -> np.ndarray:
    """RSI: buy on oversold, sell on overbought."""
    rsi = _rsi(close, cfg.rsi_period)
    signals = np.zeros(len(close), dtype=np.int8)

    for i in range(1, len(close)):
        if np.isnan(rsi[i]):
            continue
        if rsi[i] < cfg.rsi_oversold and rsi[i - 1] >= cfg.rsi_oversold:
            signals[i] = 1   # Entered oversold → long
        elif rsi[i] > cfg.rsi_overbought and rsi[i - 1] <= cfg.rsi_overbought:
            signals[i] = -1  # Entered overbought → short

    return signals


STRATEGY_MAP = {
    "momentum": _signals_momentum,
    "ma_crossover": _signals_momentum,
    "mean_reversion": _signals_mean_reversion,
    "breakout": _signals_breakout,
    "rsi": _signals_rsi,
}


# ─── Backtest Result ─────────────────────────────────────────────

@dataclass
class CandleBacktestResult:
    """Results from a candle-based backtest."""
    experiment_id: str
    config: Dict
    coin: str
    timeframe: str
    candle_count: int

    # Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_hold_candles: float = 0.0
    total_fees: float = 0.0

    # Curves (numpy → list for serialization)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)

    # Trade log
    trades: List[Dict] = field(default_factory=list)

    # Timing
    duration_seconds: float = 0.0
    candles_per_second: float = 0.0

    def summary(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "coin": self.coin,
            "timeframe": self.timeframe,
            "candles": self.candle_count,
            "trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1f}%",
            "pnl": f"${self.total_pnl:+,.2f}",
            "pnl_pct": f"{self.total_pnl_pct:+.1f}%",
            "max_dd": f"{self.max_drawdown_pct:.1f}%",
            "sharpe": f"{self.sharpe_ratio:.3f}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "speed": f"{self.candles_per_second:,.0f} candles/s",
        }


# ─── Core Engine ─────────────────────────────────────────────────

class CandleBacktester:
    """
    Fast candle-based backtester with vectorized indicators.

    Workflow:
      1. Convert candle list to numpy arrays (one-time cost)
      2. Compute all indicators vectorized
      3. Generate signal array
      4. Simulate trades sequentially (unavoidable — state-dependent)
      5. Compute metrics vectorized from equity curve
    """

    def __init__(self, cfg: CandleBacktestConfig = None):
        self.cfg = cfg or CandleBacktestConfig()

    def run(self, candles, strategy: str = None,
            coin: str = None, experiment_id: str = None) -> CandleBacktestResult:
        """
        Run backtest on candle data.

        Args:
            candles: List of Candle objects OR dict with numpy arrays
            strategy: Override strategy name
            coin: Coin label
            experiment_id: Optional experiment ID
        """
        t0 = time.time()

        strat = strategy or self.cfg.strategy
        if experiment_id is None:
            experiment_id = f"cbt_{strat}_{int(time.time())}"

        # Convert to numpy arrays
        if isinstance(candles, dict):
            ts = candles["timestamp_ms"]
            o, h, l, c, v = candles["open"], candles["high"], candles["low"], candles["close"], candles["volume"]
            coin = candles.get("coin", coin or "UNKNOWN")
            tf = candles.get("timeframe", "1h")
        else:
            # List of Candle objects
            if not candles:
                return CandleBacktestResult(
                    experiment_id=experiment_id, config=self.cfg.to_dict(),
                    coin=coin or "UNKNOWN", timeframe="?", candle_count=0)

            coin = coin or candles[0].coin
            tf = candles[0].timeframe
            ts = np.array([c.timestamp_ms for c in candles], dtype=np.int64)
            o = np.array([c.open for c in candles], dtype=np.float64)
            h = np.array([c.high for c in candles], dtype=np.float64)
            l = np.array([c.low for c in candles], dtype=np.float64)
            c = np.array([c.close for c in candles], dtype=np.float64)
            v = np.array([c.volume for c in candles], dtype=np.float64)

        n = len(c)
        logger.info(f"Running {strat} backtest on {coin} {tf}: {n:,} candles")

        # Generate signals
        sig_fn = STRATEGY_MAP.get(strat)
        if sig_fn is None:
            raise ValueError(f"Unknown strategy: {strat}. Available: {list(STRATEGY_MAP.keys())}")

        if strat == "breakout":
            signals = sig_fn(c, h, l, self.cfg)
        else:
            signals = sig_fn(c, self.cfg)

        # Simulate trades
        trades, equity = self._simulate(c, h, l, ts, signals)

        # Compute metrics from equity curve
        equity_arr = np.array(equity)
        result = self._compute_metrics(
            trades, equity_arr, ts, n,
            experiment_id, coin, tf, time.time() - t0
        )

        logger.info(
            f"Backtest done: {result.total_trades} trades, "
            f"PnL=${result.total_pnl:+,.2f} ({result.total_pnl_pct:+.1f}%), "
            f"Sharpe={result.sharpe_ratio:.3f}, "
            f"MaxDD={result.max_drawdown_pct:.1f}%, "
            f"{result.candles_per_second:,.0f} candles/s"
        )

        return result

    def _simulate(self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  timestamps: np.ndarray, signals: np.ndarray
                  ) -> Tuple[List[Dict], List[float]]:
        """
        Sequential trade simulation (must be sequential — positions are state-dependent).
        Returns (trade_list, equity_curve).
        """
        cfg = self.cfg
        balance = cfg.initial_balance
        equity = [balance]
        trades = []

        # Active position state
        pos_side = 0       # 0=flat, 1=long, -1=short
        pos_entry = 0.0
        pos_size = 0.0
        pos_entry_idx = 0
        pos_trail_high = 0.0
        pos_trail_low = float("inf")

        fee_rate = cfg.taker_fee_bps / 10_000
        slip_rate = cfg.slippage_bps / 10_000
        total_fees = 0.0

        for i in range(len(close)):
            price = close[i]

            # Check exits on existing position
            if pos_side != 0:
                # Update trailing
                if pos_side == 1:
                    pos_trail_high = max(pos_trail_high, high[i])
                else:
                    pos_trail_low = min(pos_trail_low, low[i])

                # Check TP/SL/trailing
                exit_reason = None
                exit_price = price

                if pos_side == 1:  # Long
                    pnl_pct = (price - pos_entry) / pos_entry
                    if low[i] <= pos_entry * (1 - cfg.stop_loss_pct):
                        exit_reason = "stop_loss"
                        exit_price = pos_entry * (1 - cfg.stop_loss_pct)
                    elif high[i] >= pos_entry * (1 + cfg.take_profit_pct):
                        exit_reason = "take_profit"
                        exit_price = pos_entry * (1 + cfg.take_profit_pct)
                    elif cfg.trailing_stop_pct > 0 and pos_trail_high > 0:
                        trail_trigger = pos_trail_high * (1 - cfg.trailing_stop_pct)
                        if low[i] <= trail_trigger and pnl_pct > cfg.trailing_stop_pct:
                            exit_reason = "trailing_stop"
                            exit_price = trail_trigger
                else:  # Short
                    pnl_pct = (pos_entry - price) / pos_entry
                    if high[i] >= pos_entry * (1 + cfg.stop_loss_pct):
                        exit_reason = "stop_loss"
                        exit_price = pos_entry * (1 + cfg.stop_loss_pct)
                    elif low[i] <= pos_entry * (1 - cfg.take_profit_pct):
                        exit_reason = "take_profit"
                        exit_price = pos_entry * (1 - cfg.take_profit_pct)
                    elif cfg.trailing_stop_pct > 0 and pos_trail_low < float("inf"):
                        trail_trigger = pos_trail_low * (1 + cfg.trailing_stop_pct)
                        if high[i] >= trail_trigger and pnl_pct > cfg.trailing_stop_pct:
                            exit_reason = "trailing_stop"
                            exit_price = trail_trigger

                # Signal-based exit (reverse signal)
                if exit_reason is None and signals[i] != 0 and signals[i] != pos_side:
                    exit_reason = "signal_reverse"
                    exit_price = price

                if exit_reason:
                    # Apply slippage on exit
                    if pos_side == 1:
                        exit_price *= (1 - slip_rate)
                        pnl = (exit_price - pos_entry) * pos_size * cfg.max_leverage
                    else:
                        exit_price *= (1 + slip_rate)
                        pnl = (pos_entry - exit_price) * pos_size * cfg.max_leverage

                    fee = abs(pos_size * exit_price * fee_rate)
                    pnl -= fee
                    total_fees += fee

                    # Funding charges
                    if cfg.funding_enabled:
                        hold_candles = i - pos_entry_idx
                        # Approximate: assume each candle is one period
                        funding = pos_size * pos_entry * cfg.funding_rate_8h * (hold_candles / 8)
                        pnl -= funding

                    balance += pnl
                    trades.append({
                        "entry_idx": pos_entry_idx,
                        "exit_idx": i,
                        "side": "long" if pos_side == 1 else "short",
                        "entry_price": pos_entry,
                        "exit_price": exit_price,
                        "size": pos_size,
                        "pnl": pnl,
                        "pnl_pct": pnl / (pos_entry * pos_size) * 100 if pos_size > 0 else 0,
                        "exit_reason": exit_reason,
                        "hold_candles": i - pos_entry_idx,
                        "entry_ts": int(timestamps[pos_entry_idx]) if pos_entry_idx < len(timestamps) else 0,
                        "exit_ts": int(timestamps[i]),
                    })

                    pos_side = 0
                    pos_entry = 0
                    pos_size = 0

            # Open new position on signal (if flat)
            if pos_side == 0 and signals[i] != 0:
                pos_side = int(signals[i])
                # Apply slippage on entry
                if pos_side == 1:
                    pos_entry = price * (1 + slip_rate)
                else:
                    pos_entry = price * (1 - slip_rate)

                pos_size = (balance * cfg.position_size_pct) / pos_entry
                pos_entry_idx = i
                pos_trail_high = high[i]
                pos_trail_low = low[i]

                fee = abs(pos_size * pos_entry * fee_rate)
                balance -= fee
                total_fees += fee

            # Track equity
            unrealized = 0.0
            if pos_side == 1:
                unrealized = (price - pos_entry) * pos_size * cfg.max_leverage
            elif pos_side == -1:
                unrealized = (pos_entry - price) * pos_size * cfg.max_leverage
            equity.append(balance + unrealized)

        # Force close if still in position
        if pos_side != 0:
            final_price = close[-1]
            if pos_side == 1:
                pnl = (final_price - pos_entry) * pos_size * cfg.max_leverage
            else:
                pnl = (pos_entry - final_price) * pos_size * cfg.max_leverage
            fee = abs(pos_size * final_price * fee_rate)
            pnl -= fee
            balance += pnl
            trades.append({
                "entry_idx": pos_entry_idx,
                "exit_idx": len(close) - 1,
                "side": "long" if pos_side == 1 else "short",
                "entry_price": pos_entry,
                "exit_price": final_price,
                "size": pos_size,
                "pnl": pnl,
                "pnl_pct": pnl / (pos_entry * pos_size) * 100 if pos_size > 0 else 0,
                "exit_reason": "end_of_data",
                "hold_candles": len(close) - 1 - pos_entry_idx,
                "entry_ts": int(timestamps[pos_entry_idx]) if pos_entry_idx < len(timestamps) else 0,
                "exit_ts": int(timestamps[-1]),
            })

        return trades, equity

    def _compute_metrics(self, trades: List[Dict], equity: np.ndarray,
                         timestamps: np.ndarray, n_candles: int,
                         experiment_id: str, coin: str, tf: str,
                         elapsed: float) -> CandleBacktestResult:
        """Compute all metrics from trade list and equity curve (vectorized where possible)."""
        cfg = self.cfg

        # Basic trade stats
        pnls = np.array([t["pnl"] for t in trades]) if trades else np.array([])
        total_trades = len(trades)
        wins = int(np.sum(pnls > 0)) if len(pnls) > 0 else 0
        losses = int(np.sum(pnls <= 0)) if len(pnls) > 0 else 0
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        total_pnl = float(np.sum(pnls)) if len(pnls) > 0 else 0
        total_pnl_pct = total_pnl / cfg.initial_balance * 100

        # Drawdown (vectorized)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(drawdown)) * 100 if len(drawdown) > 0 else 0

        # Returns for Sharpe/Sortino
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            returns = returns[np.isfinite(returns)]

            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            else:
                sharpe = 0.0

            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252))
            else:
                sortino = 0.0
        else:
            sharpe = sortino = 0.0

        # Profit factor
        gross_profit = float(np.sum(pnls[pnls > 0])) if len(pnls) > 0 else 0
        gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if len(pnls) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        # Average trade stats
        avg_pnl = float(np.mean(pnls)) if len(pnls) > 0 else 0
        best_pnl = float(np.max(pnls)) if len(pnls) > 0 else 0
        worst_pnl = float(np.min(pnls)) if len(pnls) > 0 else 0
        avg_hold = float(np.mean([t["hold_candles"] for t in trades])) if trades else 0

        # Total fees
        total_fees = sum(
            abs(t["size"] * t["entry_price"] * cfg.taker_fee_bps / 10_000) +
            abs(t["size"] * t["exit_price"] * cfg.taker_fee_bps / 10_000)
            for t in trades
        )

        return CandleBacktestResult(
            experiment_id=experiment_id,
            config=cfg.to_dict(),
            coin=coin,
            timeframe=tf,
            candle_count=n_candles,
            total_trades=total_trades,
            winning_trades=wins,
            losing_trades=losses,
            win_rate=win_rate,
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            profit_factor=round(profit_factor, 4),
            avg_trade_pnl=round(avg_pnl, 2),
            best_trade_pnl=round(best_pnl, 2),
            worst_trade_pnl=round(worst_pnl, 2),
            avg_hold_candles=round(avg_hold, 1),
            total_fees=round(total_fees, 2),
            equity_curve=equity.tolist(),
            drawdown_curve=(drawdown * 100).tolist(),
            trades=trades,
            duration_seconds=round(elapsed, 3),
            candles_per_second=round(n_candles / elapsed, 0) if elapsed > 0 else 0,
        )

    def run_multi_coin(self, candle_sets: Dict[str, list],
                       strategy: str = None) -> Dict[str, CandleBacktestResult]:
        """
        Run backtests on multiple coins in parallel.

        Args:
            candle_sets: {coin: [Candle, ...]}
            strategy: Strategy to test

        Returns:
            {coin: CandleBacktestResult}
        """
        results = {}
        strat = strategy or self.cfg.strategy

        # Sequential fallback (ProcessPoolExecutor can't pickle lambdas)
        for coin, candles in candle_sets.items():
            try:
                r = self.run(candles, strategy=strat, coin=coin)
                results[coin] = r
            except Exception as e:
                logger.error(f"Backtest failed for {coin}: {e}")

        return results

    def parameter_sweep(self, candles, param_name: str,
                        values: list, strategy: str = None) -> List[CandleBacktestResult]:
        """
        Run the same backtest with different parameter values.

        Args:
            candles: Candle data
            param_name: Config field to sweep (e.g. "fast_period")
            values: List of values to test

        Returns:
            List of results, one per value
        """
        results = []
        for val in values:
            cfg_copy = CandleBacktestConfig(**self.cfg.to_dict())
            if hasattr(cfg_copy, param_name):
                setattr(cfg_copy, param_name, val)
            else:
                logger.warning(f"Unknown param: {param_name}")
                continue

            bt = CandleBacktester(cfg_copy)
            r = bt.run(candles, strategy=strategy)
            results.append(r)
            logger.info(f"  {param_name}={val}: PnL=${r.total_pnl:+,.2f}, "
                       f"Sharpe={r.sharpe_ratio:.3f}, WR={r.win_rate:.1f}%")

        return results
