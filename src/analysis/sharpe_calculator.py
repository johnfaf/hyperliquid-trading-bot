"""
Sharpe Ratio Calculator
========================
Computes proper Sharpe ratios from actual trade PnL data instead of estimating
from win_rate/profit_factor. Uses individual trade returns to calculate
mean and standard deviation, then annualizes.

Provides:
  - Sharpe Ratio: (annualized_return - risk_free_rate) / annualized_volatility
  - Sortino Ratio: Uses only downside deviation (better for asymmetric returns)
  - Calmar Ratio: Annualized return / max drawdown
  - Rolling Sharpe: Trend detection over time
"""
import logging
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Annualization factor: assume ~240 trading days per year
TRADING_DAYS_PER_YEAR = 240
TRADES_PER_YEAR_ESTIMATE = 250  # Conservative estimate if we don't know trade frequency


@dataclass
class SharpeResult:
    """Result of Sharpe/Sortino/Calmar analysis."""
    sharpe_ratio: float              # Sharpe Ratio (annualized)
    sortino_ratio: float             # Sortino Ratio (downside only)
    calmar_ratio: float              # Calmar Ratio (return / max DD)
    max_drawdown: float              # Maximum drawdown as decimal (e.g. 0.25 = 25%)
    annualized_return: float         # Annualized return as decimal
    annualized_volatility: float     # Annualized return std dev
    trade_count: int                 # Number of closed trades
    avg_return: float                # Average per-trade return
    return_std: float                # Per-trade return std dev
    sortino_denominator: float       # Downside volatility (for reference)
    win_rate: float                  # Percentage of profitable trades
    profit_factor: float             # Gross wins / Gross losses
    largest_win: float               # Largest single trade gain
    largest_loss: float              # Largest single trade loss
    consecutive_losses: int          # Longest losing streak
    return_series: List[float]       # Per-trade returns (for diagnostics)
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _group_fills_into_roundtrips(fills: List[Dict]) -> List[Dict]:
    """
    Group fills into round-trips (open→close sequences).

    A round-trip is a sequence of fills that open and close a position:
      - Buy → Sell (long round-trip)
      - Sell → Buy (short round-trip)

    Returns list of closed trades, each with:
      - entry_price, exit_price, size, entry_time, exit_time, pnl, direction
    """
    if not fills:
        return []

    rounds = []

    # Sort by coin and time
    fills_by_coin = {}
    for f in fills:
        coin = f.get("coin", "UNKNOWN")
        if coin not in fills_by_coin:
            fills_by_coin[coin] = []
        fills_by_coin[coin].append(f)

    # For each coin, match opens and closes
    for coin, coin_fills in fills_by_coin.items():
        coin_fills_sorted = sorted(coin_fills, key=lambda x: x.get("time_ms", 0))

        position_stack = []  # Stack of open trades: (side, price, size, time_ms)

        for fill in coin_fills_sorted:
            side = fill.get("side", "").lower()  # "buy" or "sell"
            price = fill.get("price", 0)
            original_price = fill.get("original_price", price)
            size = abs(fill.get("size", 0))
            time_ms = fill.get("time_ms", 0)
            closed_pnl = fill.get("closed_pnl", 0)
            pnl_adj = fill.get("penalised_pnl", closed_pnl)

            # Only care about fills with actual PnL (closed trades)
            if closed_pnl == 0 and pnl_adj == 0:
                continue

            # If this is a closing fill, we have PnL
            # Match it to most recent open of opposite side
            if position_stack:
                entry = position_stack.pop(0)
                entry_side, entry_price, entry_size, entry_time = entry

                # Ensure sides are opposite
                if (entry_side == "buy" and side == "sell") or \
                   (entry_side == "sell" and side == "buy"):
                    rounds.append({
                        "coin": coin,
                        "direction": f"{'Long' if entry_side == 'buy' else 'Short'}",
                        "entry_price": entry_price,
                        "exit_price": price,
                        "size": min(entry_size, size),
                        "entry_time": entry_time,
                        "exit_time": time_ms,
                        "closed_pnl": closed_pnl,
                        "penalised_pnl": pnl_adj,
                    })
            else:
                # No matching entry, treat as a new open
                position_stack.append((side, price, size, time_ms))

    return rounds


def calculate_sharpe(
    fills: List[Dict],
    risk_free_rate: float = 0.05,
    min_trades: int = 5,
) -> Optional[SharpeResult]:
    """
    Calculate proper Sharpe ratio from actual closed PnL in fills.

    Args:
        fills: List of fill dicts with keys: coin, side, price, size,
               time_ms, closed_pnl, penalised_pnl
        risk_free_rate: Annual risk-free rate (default 5%)
        min_trades: Minimum closed trades to compute ratio (default 5)

    Returns:
        SharpeResult with all metrics, or None if insufficient data
    """
    if not fills:
        logger.debug("calculate_sharpe: no fills provided")
        return None

    # Group into round-trips
    rounds = _group_fills_into_roundtrips(fills)

    if len(rounds) < min_trades:
        logger.debug(f"calculate_sharpe: only {len(rounds)} closed trades, need {min_trades}")
        return None

    # Extract PnL values
    pnl_values = [r["penalised_pnl"] for r in rounds if r["penalised_pnl"] != 0]

    if not pnl_values:
        logger.debug("calculate_sharpe: no closed PnL found")
        return None

    # Convert to returns (normalized by something, e.g., position notional)
    # Simple approach: use raw PnL as proxy for return
    returns = np.array(pnl_values, dtype=float)

    # Compute basic stats
    mean_return = np.mean(returns)
    return_std = np.std(returns, ddof=1) if len(returns) > 1 else 0

    # Annualize: assume we have N trades over some period
    # Conservative: assume ~250 trades per year for most strategies
    trades_per_year = TRADES_PER_YEAR_ESTIMATE

    # Annualized return = mean_return * trades_per_year
    # Annualized vol = return_std * sqrt(trades_per_year)
    annualized_return = mean_return * trades_per_year
    annualized_volatility = return_std * np.sqrt(trades_per_year) if return_std > 0 else 0

    # Sharpe Ratio = (annualized_return - risk_free_rate) / annualized_volatility
    if annualized_volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    elif annualized_return > 0:
        sharpe_ratio = float('inf')
    elif annualized_return < 0:
        sharpe_ratio = float('-inf')
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio: use only downside deviations
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns, ddof=1)
        annualized_downside = downside_std * np.sqrt(trades_per_year)
        if annualized_downside > 0:
            sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside
        else:
            sortino_ratio = float('inf') if annualized_return > 0 else 0.0
    else:
        # No downside, or only one downside trade
        sortino_ratio = sharpe_ratio if annualized_volatility > 0 else 0.0
        annualized_downside = 0.0

    # Calmar Ratio: annualized_return / max_drawdown
    # Compute max drawdown from cumulative returns
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / (np.abs(running_max) + 1e-9)
    max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = float('inf') if annualized_return > 0 else 0.0

    # Win rate
    wins = np.sum(returns > 0)
    win_rate = wins / len(returns) * 100

    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float('inf') if gross_profit > 0 else 1.0
    )

    # Largest win/loss
    largest_win = np.max(returns) if len(returns) > 0 else 0
    largest_loss = np.min(returns) if len(returns) > 0 else 0

    # Consecutive losses
    consecutive_losses = 0
    current_streak = 0
    for r in returns:
        if r < 0:
            current_streak += 1
            consecutive_losses = max(consecutive_losses, current_streak)
        else:
            current_streak = 0

    # Clamp infinities to reasonable bounds
    sharpe_ratio = max(-10.0, min(10.0, sharpe_ratio)) if not np.isinf(sharpe_ratio) else sharpe_ratio
    sortino_ratio = max(-10.0, min(10.0, sortino_ratio)) if not np.isinf(sortino_ratio) else sortino_ratio
    calmar_ratio = max(-10.0, min(10.0, calmar_ratio)) if not np.isinf(calmar_ratio) else calmar_ratio

    return SharpeResult(
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        calmar_ratio=float(calmar_ratio),
        max_drawdown=float(max_drawdown),
        annualized_return=float(annualized_return),
        annualized_volatility=float(annualized_volatility),
        trade_count=len(returns),
        avg_return=float(mean_return),
        return_std=float(return_std),
        sortino_denominator=float(annualized_downside),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        largest_win=float(largest_win),
        largest_loss=float(largest_loss),
        consecutive_losses=int(consecutive_losses),
        return_series=returns.tolist(),
    )


def calculate_rolling_sharpe(
    fills: List[Dict],
    window: int = 30,
    risk_free_rate: float = 0.05,
) -> List[float]:
    """
    Calculate rolling Sharpe ratio using a sliding window of trades.

    Useful for detecting strategy performance degradation or improvement over time.

    Args:
        fills: List of fill dicts
        window: Number of trades in each rolling window (default 30)
        risk_free_rate: Annual risk-free rate

    Returns:
        List of rolling Sharpe ratios (same length as number of windows)
    """
    if not fills or window < 5:
        return []

    rounds = _group_fills_into_roundtrips(fills)
    if len(rounds) < window:
        return []

    pnl_values = np.array([r["penalised_pnl"] for r in rounds if r["penalised_pnl"] != 0])

    rolling_sharpes = []
    for i in range(len(pnl_values) - window + 1):
        window_returns = pnl_values[i:i+window]

        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns, ddof=1) if len(window_returns) > 1 else 0

        # Annualize
        trades_per_year = TRADES_PER_YEAR_ESTIMATE
        ann_ret = mean_ret * trades_per_year
        ann_vol = std_ret * np.sqrt(trades_per_year) if std_ret > 0 else 0

        if ann_vol > 0:
            sharpe = (ann_ret - risk_free_rate) / ann_vol
        elif ann_ret > 0:
            sharpe = 10.0  # Cap at 10 for inf
        else:
            sharpe = -10.0 if ann_ret < 0 else 0.0

        # Clamp
        sharpe = max(-10.0, min(10.0, sharpe))
        rolling_sharpes.append(float(sharpe))

    return rolling_sharpes


def get_sharpe_summary(sharpe_result: Optional[SharpeResult]) -> str:
    """
    Format SharpeResult as a human-readable summary string.

    Args:
        sharpe_result: SharpeResult from calculate_sharpe() or None

    Returns:
        Formatted string for logging/reporting
    """
    if not sharpe_result:
        return "No Sharpe data (insufficient trades)"

    return (
        f"Sharpe={sharpe_result.sharpe_ratio:.2f} | "
        f"Sortino={sharpe_result.sortino_ratio:.2f} | "
        f"Calmar={sharpe_result.calmar_ratio:.2f} | "
        f"MaxDD={sharpe_result.max_drawdown:.2%} | "
        f"AnnReturn={sharpe_result.annualized_return:.2%} | "
        f"Trades={sharpe_result.trade_count} | "
        f"WinRate={sharpe_result.win_rate:.1f}% | "
        f"PF={sharpe_result.profit_factor:.2f}"
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    # Create sample fills for testing
    sample_fills = [
        {
            "coin": "ETH",
            "side": "buy",
            "price": 2000.0,
            "original_price": 2000.0,
            "size": 1.0,
            "time_ms": 1000,
            "closed_pnl": 0,
            "penalised_pnl": 0,
        },
        {
            "coin": "ETH",
            "side": "sell",
            "price": 2050.0,
            "original_price": 2050.0,
            "size": 1.0,
            "time_ms": 2000,
            "closed_pnl": 50.0,
            "penalised_pnl": 48.0,
        },
        {
            "coin": "ETH",
            "side": "buy",
            "price": 2000.0,
            "original_price": 2000.0,
            "size": 1.0,
            "time_ms": 3000,
            "closed_pnl": 0,
            "penalised_pnl": 0,
        },
        {
            "coin": "ETH",
            "side": "sell",
            "price": 1980.0,
            "original_price": 1980.0,
            "size": 1.0,
            "time_ms": 4000,
            "closed_pnl": -20.0,
            "penalised_pnl": -22.0,
        },
    ]

    result = calculate_sharpe(sample_fills)
    if result:
        print(f"Sharpe Result:\n{get_sharpe_summary(result)}")
        print(f"Trade returns: {result.return_series}")

    rolling = calculate_rolling_sharpe(sample_fills, window=2)
    print(f"Rolling Sharpe (window=2): {rolling}")
