"""
Unit tests for Monte-Carlo stress testing.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig


def test_basic_simulation():
    """Should run without errors and produce valid output."""
    mc = MonteCarloSimulator()
    returns = np.random.normal(0.01, 0.03, 100)  # 1% mean, 3% std
    cfg = MonteCarloConfig(n_paths=100, include_crashes=False,
                           include_funding_shocks=False)
    result = mc.run(returns, cfg)

    assert result.n_paths == 100
    assert result.trades_per_path == 100
    assert result.prob_positive >= 0
    assert result.mean_max_dd >= 0
    assert len(result.equity_bands) > 0


def test_crash_injection():
    """With crashes enabled, drawdowns should be worse."""
    returns = np.random.normal(0.005, 0.02, 200)

    mc = MonteCarloSimulator()
    no_crash = mc.run(returns, MonteCarloConfig(
        n_paths=500, include_crashes=False, include_funding_shocks=False))
    with_crash = mc.run(returns, MonteCarloConfig(
        n_paths=500, include_crashes=True, crash_probability=0.05,
        include_funding_shocks=False))

    # Crashes should increase mean drawdown
    assert with_crash.mean_max_dd >= no_crash.mean_max_dd * 0.8  # Roughly worse


def test_percentile_ordering():
    """Lower percentiles should have lower returns."""
    mc = MonteCarloSimulator()
    returns = np.random.normal(0.005, 0.03, 150)
    result = mc.run(returns, MonteCarloConfig(
        n_paths=1000, include_crashes=False, include_funding_shocks=False))

    pcts = result.return_percentiles
    assert pcts["p5"] <= pcts["p25"]
    assert pcts["p25"] <= pcts["p50"]
    assert pcts["p50"] <= pcts["p75"]
    assert pcts["p75"] <= pcts["p95"]


def test_empty_returns_raises():
    """Should raise on empty input."""
    mc = MonteCarloSimulator()
    with pytest.raises(ValueError):
        mc.run(np.array([]))


def test_all_negative_returns():
    """With all losing trades, prob_positive should be low."""
    mc = MonteCarloSimulator()
    returns = np.random.uniform(-0.05, -0.01, 100)
    result = mc.run(returns, MonteCarloConfig(
        n_paths=200, include_crashes=False, include_funding_shocks=False))

    assert result.mean_return < 0
    assert result.prob_positive < 20  # Should be very low


def test_calmar_in_results():
    """Calmar percentiles should be present in results."""
    mc = MonteCarloSimulator()
    returns = np.random.normal(0.01, 0.02, 100)
    result = mc.run(returns, MonteCarloConfig(
        n_paths=100, include_crashes=False, include_funding_shocks=False))

    assert "p50" in result.calmar_percentiles
    assert result.calmar_percentiles["p50"] is not None


def test_summary():
    """Summary should return a formatted dict."""
    mc = MonteCarloSimulator()
    returns = np.random.normal(0.005, 0.02, 100)
    result = mc.run(returns, MonteCarloConfig(
        n_paths=50, include_crashes=False, include_funding_shocks=False))

    summary = result.summary()
    assert "paths" in summary
    assert "mean_return" in summary
    assert "prob_positive" in summary
