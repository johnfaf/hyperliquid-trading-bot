#!/usr/bin/env python3
"""
Crash Monte-Carlo Simulator
============================
Generates synthetic 2022-style crash trade returns and runs Monte-Carlo
simulations with and without crash injection to quantify tail risk.

Synthetic Data: 200 trades over bear market with:
  - Mean return: -0.3% per trade
  - Std dev: 3%
  - Negative skew (occasional large drawdowns)
  - Mix of winning trades (realistic)

Runs 10,000-path simulations comparing:
  1. WITH crash injection enabled
  2. WITHOUT crash injection (baseline)

Results include comprehensive percentile analysis, ruin probabilities,
and Sharpe/Calmar distributions.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig


def generate_synthetic_2022_crash_returns(n_trades: int = 200, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic trade returns modeling 2022 bear market crash.

    Characteristics:
    - Mean: ~-0.3% per trade
    - Std dev: ~3%
    - Negative skew (tail risk)
    - Some large drawdowns (5-10% single trades)
    - Mix of winners and losers (realistic)

    Args:
        n_trades: Number of trades to generate
        seed: Random seed for reproducibility

    Returns:
        Array of per-trade returns (as fractions, e.g. -0.003 = -0.3%)
    """
    rng = np.random.default_rng(seed)

    # Base distribution: slightly negative mean, high volatility
    base_returns = rng.normal(loc=-0.003, scale=0.03, size=n_trades)

    # Inject occasional large drawdowns (5-10% single trade losses)
    # ~5% of trades are crash events
    crash_mask = rng.random(n_trades) < 0.05
    n_crashes = np.sum(crash_mask)
    if n_crashes > 0:
        crash_returns = rng.uniform(-0.10, -0.05, size=n_crashes)
        base_returns[crash_mask] = crash_returns

    # Add positive skew correction: include some larger winners (3-5%)
    # ~10% of trades are larger winners
    winner_mask = rng.random(n_trades) < 0.10
    n_winners = np.sum(winner_mask)
    if n_winners > 0:
        winner_returns = rng.uniform(0.02, 0.05, size=n_winners)
        base_returns[winner_mask] = winner_returns

    return base_returns


def format_table_row(label: str, *values) -> str:
    """Format a table row with left-aligned label and right-aligned values."""
    label_part = f"{label:<20}"
    value_parts = [f"{str(v):>12}" for v in values]
    return label_part + "".join(value_parts)


def print_results_table(result, scenario_name: str):
    """Print comprehensive results table for a scenario."""
    print(f"\n{'='*80}")
    print(f"{scenario_name:^80}")
    print('='*80)

    # Return percentiles
    print("\nRETURN PERCENTILES (%):")
    print(format_table_row("Metric", "Value"))
    print("-" * 35)
    print(format_table_row("  p5", f"{result.return_percentiles['p5']:+.2f}"))
    print(format_table_row("  p25", f"{result.return_percentiles['p25']:+.2f}"))
    print(format_table_row("  p50 (median)", f"{result.return_percentiles['p50']:+.2f}"))
    print(format_table_row("  p75", f"{result.return_percentiles['p75']:+.2f}"))
    print(format_table_row("  p95", f"{result.return_percentiles['p95']:+.2f}"))
    print(format_table_row("  Mean", f"{result.mean_return:+.2f}"))
    print(format_table_row("  Std Dev", f"{result.std_return:+.2f}"))
    print(format_table_row("  Best", f"{result.best_return:+.2f}"))
    print(format_table_row("  Worst", f"{result.worst_return:+.2f}"))

    # Drawdown percentiles
    print("\nMAX DRAWDOWN PERCENTILES (%):")
    print(format_table_row("Metric", "Value"))
    print("-" * 35)
    print(format_table_row("  p5", f"{result.drawdown_percentiles['p5']:.2f}"))
    print(format_table_row("  p25", f"{result.drawdown_percentiles['p25']:.2f}"))
    print(format_table_row("  p50 (median)", f"{result.drawdown_percentiles['p50']:.2f}"))
    print(format_table_row("  p75", f"{result.drawdown_percentiles['p75']:.2f}"))
    print(format_table_row("  p95", f"{result.drawdown_percentiles['p95']:.2f}"))
    print(format_table_row("  Mean", f"{result.mean_max_dd:.2f}"))
    print(format_table_row("  Worst", f"{result.worst_drawdown:.2f}"))

    # Sharpe ratio distribution
    print("\nSHARPE RATIO DISTRIBUTION:")
    print(format_table_row("Metric", "Value"))
    print("-" * 35)
    print(format_table_row("  p5", f"{result.sharpe_percentiles['p5']:.4f}"))
    print(format_table_row("  p25", f"{result.sharpe_percentiles['p25']:.4f}"))
    print(format_table_row("  p50 (median)", f"{result.sharpe_percentiles['p50']:.4f}"))
    print(format_table_row("  p75", f"{result.sharpe_percentiles['p75']:.4f}"))
    print(format_table_row("  p95", f"{result.sharpe_percentiles['p95']:.4f}"))

    # Calmar ratio distribution
    print("\nCALMAR RATIO DISTRIBUTION:")
    print(format_table_row("Metric", "Value"))
    print("-" * 35)
    print(format_table_row("  p5", f"{result.calmar_percentiles['p5']:.4f}"))
    print(format_table_row("  p25", f"{result.calmar_percentiles['p25']:.4f}"))
    print(format_table_row("  p50 (median)", f"{result.calmar_percentiles['p50']:.4f}"))
    print(format_table_row("  p75", f"{result.calmar_percentiles['p75']:.4f}"))
    print(format_table_row("  p95", f"{result.calmar_percentiles['p95']:.4f}"))

    # Risk metrics
    print("\nRISK & PROBABILITY METRICS:")
    print(format_table_row("Metric", "Value"))
    print("-" * 35)
    print(format_table_row("  Prob Ruin (<-50%)", f"{result.prob_ruin:.2f}%"))
    print(format_table_row("  Prob Positive Return", f"{result.prob_positive:.1f}%"))

    # Summary
    print("\nSIMULATION CONFIG:")
    print(format_table_row("  Paths", f"{result.n_paths:,}"))
    print(format_table_row("  Trades/Path", f"{result.trades_per_path}"))
    print(format_table_row("  Runtime", f"{result.duration_seconds:.2f}s"))


def print_comparison_summary(result_with_crashes, result_without_crashes):
    """Print side-by-side comparison of the two scenarios."""
    print(f"\n{'='*100}")
    print(f"{'COMPARISON: WITH vs WITHOUT CRASH INJECTION':^100}")
    print('='*100)

    # Return comparison
    print("\nRETURN DISTRIBUTION COMPARISON (%):")
    print(f"{'Metric':<25} {'With Crashes':>20} {'Without Crashes':>20} {'Difference':>20}")
    print("-" * 90)

    metrics = ['p5', 'p25', 'p50', 'p75', 'p95', 'mean', 'worst', 'best']
    for metric in metrics:
        if metric == 'mean':
            with_val = result_with_crashes.mean_return
            without_val = result_without_crashes.mean_return
            label = "Mean Return"
        elif metric == 'worst':
            with_val = result_with_crashes.worst_return
            without_val = result_without_crashes.worst_return
            label = "Worst Return"
        elif metric == 'best':
            with_val = result_with_crashes.best_return
            without_val = result_without_crashes.best_return
            label = "Best Return"
        else:
            with_val = result_with_crashes.return_percentiles[metric]
            without_val = result_without_crashes.return_percentiles[metric]
            label = f"p{metric[1:]}" if metric.startswith('p') else metric.upper()

        diff = with_val - without_val
        print(f"{label:<25} {with_val:>20.2f} {without_val:>20.2f} {diff:>+20.2f}")

    # Drawdown comparison
    print("\nMAX DRAWDOWN COMPARISON (%):")
    print(f"{'Metric':<25} {'With Crashes':>20} {'Without Crashes':>20} {'Difference':>20}")
    print("-" * 90)

    metrics = ['p5', 'p25', 'p50', 'p75', 'p95', 'mean', 'worst']
    for metric in metrics:
        if metric == 'mean':
            with_val = result_with_crashes.mean_max_dd
            without_val = result_without_crashes.mean_max_dd
            label = "Mean Max DD"
        elif metric == 'worst':
            with_val = result_with_crashes.worst_drawdown
            without_val = result_without_crashes.worst_drawdown
            label = "Worst Max DD"
        else:
            with_val = result_with_crashes.drawdown_percentiles[metric]
            without_val = result_without_crashes.drawdown_percentiles[metric]
            label = f"p{metric[1:]}" if metric.startswith('p') else metric.upper()

        diff = with_val - without_val
        print(f"{label:<25} {with_val:>20.2f} {without_val:>20.2f} {diff:>+20.2f}")

    # Risk metrics comparison
    print("\nRISK METRICS COMPARISON (%):")
    print(f"{'Metric':<25} {'With Crashes':>20} {'Without Crashes':>20} {'Difference':>20}")
    print("-" * 90)

    prob_ruin_diff = result_with_crashes.prob_ruin - result_without_crashes.prob_ruin
    prob_pos_diff = result_with_crashes.prob_positive - result_without_crashes.prob_positive

    print(f"{'Prob Ruin (< -50%)':<25} {result_with_crashes.prob_ruin:>20.2f} "
          f"{result_without_crashes.prob_ruin:>20.2f} {prob_ruin_diff:>+20.2f}")
    print(f"{'Prob Positive Return':<25} {result_with_crashes.prob_positive:>20.1f} "
          f"{result_without_crashes.prob_positive:>20.1f} {prob_pos_diff:>+20.1f}")

    # Sharpe ratio comparison
    print("\nSHARPE RATIO COMPARISON (Median):")
    print(f"{'Metric':<25} {'With Crashes':>20} {'Without Crashes':>20} {'Difference':>20}")
    print("-" * 90)

    sharpe_with = result_with_crashes.sharpe_percentiles['p50']
    sharpe_without = result_without_crashes.sharpe_percentiles['p50']
    sharpe_diff = sharpe_with - sharpe_without

    print(f"{'Sharpe p50':<25} {sharpe_with:>20.4f} {sharpe_without:>20.4f} {sharpe_diff:>+20.4f}")

    # Summary insights
    print(f"\n{'INSIGHTS':^100}")
    print("-" * 100)

    impact_return = abs(result_with_crashes.mean_return - result_without_crashes.mean_return)
    impact_dd = abs(result_with_crashes.mean_max_dd - result_without_crashes.mean_max_dd)
    impact_ruin = result_with_crashes.prob_ruin - result_without_crashes.prob_ruin

    print(f"  Crash injection reduces mean return by {impact_return:.2f}% (base effect size)")
    print(f"  Crash injection increases mean max drawdown by {impact_dd:.2f}% (risk amplification)")
    print(f"  Probability of ruin increases by {impact_ruin:.2f} percentage points")
    print(f"\n  Impact Assessment: Crashes shift tail risk significantly:")
    print(f"    - Worst-case scenario: {result_with_crashes.worst_return:+.2f}% (vs {result_without_crashes.worst_return:+.2f}% without crashes)")
    print(f"    - 5th percentile:      {result_with_crashes.return_percentiles['p5']:+.2f}% (vs {result_without_crashes.return_percentiles['p5']:+.2f}% without crashes)")


def main():
    """Main entry point."""
    print("="*80)
    print("CRASH MONTE-CARLO SIMULATOR (2022-Style Bear Market)".center(80))
    print("="*80)

    # Generate synthetic 2022 crash returns
    print("\nGenerating synthetic 2022-style crash trade returns...")
    trade_returns = generate_synthetic_2022_crash_returns(n_trades=200, seed=42)

    print(f"  Generated {len(trade_returns)} trade returns")
    print(f"  Mean return per trade: {np.mean(trade_returns)*100:+.3f}%")
    print(f"  Std dev: {np.std(trade_returns)*100:.3f}%")
    print(f"  Min return: {np.min(trade_returns)*100:.3f}%")
    print(f"  Max return: {np.max(trade_returns)*100:+.3f}%")
    print(f"  Skewness: {np.mean((trade_returns - np.mean(trade_returns))**3) / (np.std(trade_returns)**3):.3f}")

    # Initialize simulator
    simulator = MonteCarloSimulator()

    # Scenario 1: WITH crash injection (crashes enabled)
    print("\n" + "="*80)
    print("SCENARIO 1: Monte-Carlo WITH Crash Injection".center(80))
    print("="*80)
    print("Running 10,000-path simulation with crash events enabled...")

    config_with_crashes = MonteCarloConfig(
        n_paths=10_000,
        initial_balance=10_000.0,
        include_crashes=True,
        crash_probability=0.02,
        crash_magnitude_mean=-0.15,
        crash_magnitude_std=0.05,
        include_funding_shocks=True,
        trades_per_path=len(trade_returns),
    )

    t0 = time.time()
    result_with_crashes = simulator.run(trade_returns, config_with_crashes)
    print(f"Completed in {result_with_crashes.duration_seconds:.2f}s")
    print_results_table(result_with_crashes, "WITH CRASH INJECTION ENABLED")

    # Scenario 2: WITHOUT crash injection (crashes disabled)
    print("\n" + "="*80)
    print("SCENARIO 2: Monte-Carlo WITHOUT Crash Injection".center(80))
    print("="*80)
    print("Running 10,000-path simulation with crash events disabled...")

    config_without_crashes = MonteCarloConfig(
        n_paths=10_000,
        initial_balance=10_000.0,
        include_crashes=False,
        include_funding_shocks=True,
        trades_per_path=len(trade_returns),
    )

    result_without_crashes = simulator.run(trade_returns, config_without_crashes)
    print(f"Completed in {result_without_crashes.duration_seconds:.2f}s")
    print_results_table(result_without_crashes, "WITHOUT CRASH INJECTION (BASELINE)")

    # Comparison
    print_comparison_summary(result_with_crashes, result_without_crashes)

    print("\n" + "="*80)
    print("SIMULATION COMPLETE".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
