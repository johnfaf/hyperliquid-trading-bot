"""
Monte-Carlo Stress Testing
============================
Runs N-path simulations to quantify tail risk and robustness of a strategy.

Given a set of historical trade returns (from candle backtester or live trades),
this module:
  1. Bootstraps random trade sequences with replacement
  2. Adds synthetic crash scenarios (2022-style -20% drawdown events)
  3. Optionally injects correlated funding-rate shocks
  4. Computes per-path equity curves and risk metrics
  5. Reports percentile-based outcomes (5th, 25th, 50th, 75th, 95th)

Usage:
    from src.backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator()
    results = mc.run(trade_returns, n_paths=5000, include_crashes=True)
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte-Carlo simulation."""
    n_paths: int = 5000                     # Number of simulation paths
    initial_balance: float = 10_000.0

    # Crash injection
    include_crashes: bool = True
    crash_probability: float = 0.02         # 2% chance per "day" of crash event
    crash_magnitude_mean: float = -0.15     # Mean crash return (-15%)
    crash_magnitude_std: float = 0.05       # Std dev of crash magnitude

    # Funding rate shocks (correlated with crashes)
    include_funding_shocks: bool = True
    funding_shock_bps: float = -50.0        # -50 bps (deeply negative) during crash
    normal_funding_bps: float = 1.0         # +1 bps normal funding

    # Trade sampling
    trades_per_path: int = 0                # 0 = match historical length
    position_size_pct: float = 0.05         # 5% per trade
    max_leverage: float = 3.0

    # Output
    percentiles: List[float] = field(default_factory=lambda: [5, 25, 50, 75, 95])

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MonteCarloResult:
    """Results from Monte-Carlo simulation."""
    n_paths: int
    trades_per_path: int
    duration_seconds: float

    # Distribution of final returns (%)
    return_percentiles: Dict[str, float]   # e.g. {"p5": -12.3, "p50": 15.6, ...}

    # Distribution of max drawdown (%)
    drawdown_percentiles: Dict[str, float]

    # Distribution of Sharpe ratios
    sharpe_percentiles: Dict[str, float]

    # Distribution of Calmar ratios
    calmar_percentiles: Dict[str, float]

    # Probability of ruin (balance drops below 50%)
    prob_ruin: float

    # Probability of positive return
    prob_positive: float

    # Mean and std of final returns
    mean_return: float
    std_return: float
    mean_max_dd: float

    # Worst path
    worst_return: float
    worst_drawdown: float

    # Best path
    best_return: float

    # Equity curve percentile bands (for plotting)
    equity_bands: Dict[str, List[float]]   # {"p5": [...], "p50": [...], ...}

    config: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        return {
            "paths": self.n_paths,
            "mean_return": f"{self.mean_return:+.2f}%",
            "mean_max_dd": f"{self.mean_max_dd:.2f}%",
            "prob_positive": f"{self.prob_positive:.1f}%",
            "prob_ruin": f"{self.prob_ruin:.2f}%",
            "worst_return": f"{self.worst_return:+.2f}%",
            "best_return": f"{self.best_return:+.2f}%",
            **{f"return_p{int(k[1:])}": f"{v:+.2f}%"
               for k, v in self.return_percentiles.items()},
        }


class MonteCarloSimulator:
    """
    Run Monte-Carlo simulations on trade return distributions.
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self.cfg = config or MonteCarloConfig()

    def run(self, trade_returns: np.ndarray,
            config: Optional[MonteCarloConfig] = None) -> MonteCarloResult:
        """
        Run Monte-Carlo simulation.

        Args:
            trade_returns: Array of per-trade returns (as fractions, e.g. 0.02 = 2%)
            config: Override config for this run

        Returns:
            MonteCarloResult with distribution metrics
        """
        cfg = config or self.cfg
        t0 = time.time()

        n = len(trade_returns)
        if n == 0:
            raise ValueError("No trade returns provided")

        trades_per_path = cfg.trades_per_path if cfg.trades_per_path > 0 else n
        n_paths = cfg.n_paths

        logger.info(f"Monte-Carlo: {n_paths:,} paths × {trades_per_path} trades "
                    f"(from {n} historical trades)")

        # Pre-allocate result arrays
        final_returns = np.zeros(n_paths)
        max_drawdowns = np.zeros(n_paths)
        sharpe_ratios = np.zeros(n_paths)
        calmar_ratios = np.zeros(n_paths)

        # Equity curve storage (sampled paths for bands)
        n_sample_points = min(trades_per_path, 500)
        sample_indices = np.linspace(0, trades_per_path - 1, n_sample_points).astype(int)
        equity_matrix = np.zeros((n_paths, n_sample_points))

        rng = np.random.default_rng(42)

        for path in range(n_paths):
            # Bootstrap trade returns with replacement
            sampled = rng.choice(trade_returns, size=trades_per_path, replace=True)

            # Inject crash events
            if cfg.include_crashes:
                crash_mask = rng.random(trades_per_path) < cfg.crash_probability
                n_crashes = np.sum(crash_mask)
                if n_crashes > 0:
                    crash_returns = rng.normal(
                        cfg.crash_magnitude_mean,
                        cfg.crash_magnitude_std,
                        size=n_crashes
                    )
                    sampled[crash_mask] = crash_returns

            # Inject funding rate effects
            if cfg.include_funding_shocks:
                funding = np.where(
                    sampled < -0.05,  # large loss → correlated funding shock
                    cfg.funding_shock_bps / 10_000 * cfg.max_leverage,
                    cfg.normal_funding_bps / 10_000 * cfg.max_leverage
                )
                sampled = sampled + funding

            # Scale returns by position size and leverage
            scaled = sampled * cfg.position_size_pct * cfg.max_leverage

            # Compute equity curve
            equity = cfg.initial_balance * np.cumprod(1.0 + scaled)

            # Metrics
            final_val = equity[-1]
            final_ret = (final_val - cfg.initial_balance) / cfg.initial_balance * 100
            final_returns[path] = final_ret

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / np.where(peak > 0, peak, 1)
            max_dd = float(np.max(dd)) * 100
            max_drawdowns[path] = max_dd

            # Sharpe
            path_returns = np.diff(equity) / equity[:-1]
            if len(path_returns) > 1 and np.std(path_returns) > 0:
                sharpe_ratios[path] = float(np.mean(path_returns) / np.std(path_returns) * np.sqrt(252))
            else:
                sharpe_ratios[path] = 0.0

            # Calmar
            calmar_ratios[path] = final_ret / max_dd if max_dd > 0 else 0.0

            # Sample equity for bands
            equity_matrix[path] = equity[sample_indices]

        # Compute percentile bands
        equity_bands = {}
        for p in cfg.percentiles:
            band = np.percentile(equity_matrix, p, axis=0)
            equity_bands[f"p{int(p)}"] = band.tolist()

        # Return percentiles
        ret_pcts = {}
        dd_pcts = {}
        sharpe_pcts = {}
        calmar_pcts = {}
        for p in cfg.percentiles:
            ret_pcts[f"p{int(p)}"] = round(float(np.percentile(final_returns, p)), 2)
            dd_pcts[f"p{int(p)}"] = round(float(np.percentile(max_drawdowns, p)), 2)
            sharpe_pcts[f"p{int(p)}"] = round(float(np.percentile(sharpe_ratios, p)), 4)
            calmar_pcts[f"p{int(p)}"] = round(float(np.percentile(calmar_ratios, p)), 4)

        elapsed = time.time() - t0

        result = MonteCarloResult(
            n_paths=n_paths,
            trades_per_path=trades_per_path,
            duration_seconds=round(elapsed, 3),
            return_percentiles=ret_pcts,
            drawdown_percentiles=dd_pcts,
            sharpe_percentiles=sharpe_pcts,
            calmar_percentiles=calmar_pcts,
            prob_ruin=round(float(np.mean(final_returns < -50)) * 100, 2),
            prob_positive=round(float(np.mean(final_returns > 0)) * 100, 1),
            mean_return=round(float(np.mean(final_returns)), 2),
            std_return=round(float(np.std(final_returns)), 2),
            mean_max_dd=round(float(np.mean(max_drawdowns)), 2),
            worst_return=round(float(np.min(final_returns)), 2),
            worst_drawdown=round(float(np.max(max_drawdowns)), 2),
            best_return=round(float(np.max(final_returns)), 2),
            equity_bands=equity_bands,
            config=cfg.to_dict(),
        )

        logger.info(
            f"Monte-Carlo done in {elapsed:.1f}s: "
            f"mean={result.mean_return:+.1f}%, "
            f"p5={ret_pcts.get('p5', 0):+.1f}%, "
            f"p95={ret_pcts.get('p95', 0):+.1f}%, "
            f"mean_DD={result.mean_max_dd:.1f}%, "
            f"prob_positive={result.prob_positive:.0f}%"
        )

        return result

    def run_from_backtest(self, backtest_result,
                          config: Optional[MonteCarloConfig] = None) -> MonteCarloResult:
        """
        Convenience: run Monte-Carlo from a CandleBacktestResult.
        Extracts per-trade returns automatically.
        """
        trades = backtest_result.trades
        if not trades:
            raise ValueError("No trades in backtest result")

        returns = np.array([
            t["pnl"] / (t["entry_price"] * t["size"])
            if t.get("entry_price") and t.get("size") else 0.0
            for t in trades
        ])

        # Filter out zero returns
        returns = returns[returns != 0]
        if len(returns) == 0:
            raise ValueError("No non-zero returns in backtest trades")

        return self.run(returns, config)
