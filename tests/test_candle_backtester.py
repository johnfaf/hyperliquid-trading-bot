"""
Unit tests for candle-based backtester.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.backtest.candle_backtester import (
    CandleBacktester, CandleBacktestConfig, CandleBacktestResult,
    _sma, _ema, _rsi, _atr, _bollinger, STRATEGY_MAP,
)


# ─── Indicator tests ─────────────────────────────────────────

def test_sma_basic():
    """SMA of constant series should equal that constant."""
    close = np.array([10.0] * 20)
    sma = _sma(close, 5)
    assert sma[4] == pytest.approx(10.0)
    assert sma[19] == pytest.approx(10.0)


def test_sma_nan_prefix():
    """SMA should have NaN for indices < period-1."""
    close = np.arange(1.0, 11.0)
    sma = _sma(close, 5)
    assert np.isnan(sma[0])
    assert np.isnan(sma[3])
    assert not np.isnan(sma[4])


def test_ema_convergence():
    """EMA should converge toward constant value."""
    close = np.array([50.0] * 5 + [100.0] * 50)
    ema = _ema(close, 5)
    # After enough periods at 100, EMA should be close to 100
    assert ema[-1] == pytest.approx(100.0, abs=0.5)


def test_rsi_boundaries():
    """RSI should be between 0 and 100."""
    close = np.random.normal(100, 5, 100).cumsum()
    rsi = _rsi(close, 14)
    valid = rsi[~np.isnan(rsi)]
    assert all(0 <= v <= 100 for v in valid)


def test_atr_positive():
    """ATR should always be positive."""
    high = np.random.uniform(101, 110, 50)
    low = np.random.uniform(90, 99, 50)
    close = (high + low) / 2
    atr = _atr(high, low, close, 14)
    valid = atr[~np.isnan(atr)]
    assert all(v > 0 for v in valid)


def test_bollinger_bands_order():
    """Upper band > middle > lower band always."""
    close = np.random.normal(100, 3, 50)
    upper, mid, lower = _bollinger(close, 20, 2.0)
    for i in range(19, 50):
        if not np.isnan(upper[i]):
            assert upper[i] >= mid[i] >= lower[i]


# ─── Strategy map tests ─────────────────────────────────────

def test_all_strategies_registered():
    """All expected strategies should be in STRATEGY_MAP."""
    expected = [
        "momentum", "ma_crossover", "mean_reversion", "breakout", "rsi",
        "macd", "macd_histogram", "vwap_reversion", "stochastic",
        "adx_trend", "supertrend", "ema_rsi_combo", "volume_breakout", "ichimoku",
    ]
    for name in expected:
        assert name in STRATEGY_MAP, f"Strategy '{name}' missing from STRATEGY_MAP"


# ─── Config tests ────────────────────────────────────────────

def test_config_defaults():
    """Config should have sane defaults."""
    cfg = CandleBacktestConfig()
    assert cfg.initial_balance == 10_000.0
    assert cfg.stop_loss_pct > 0
    assert cfg.take_profit_pct > 0
    assert cfg.fast_period < cfg.slow_period


def test_config_to_dict():
    """Config should be serializable."""
    cfg = CandleBacktestConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert "strategy" in d
    assert "initial_balance" in d


# ─── Result tests ────────────────────────────────────────────

def test_result_has_calmar():
    """CandleBacktestResult should have calmar_ratio field."""
    result = CandleBacktestResult(
        experiment_id="test", config={}, coin="BTC",
        timeframe="1h", candle_count=100
    )
    assert hasattr(result, "calmar_ratio")
    assert result.calmar_ratio == 0.0


def test_result_summary_includes_calmar():
    """Summary dict should include calmar."""
    result = CandleBacktestResult(
        experiment_id="test", config={}, coin="BTC",
        timeframe="1h", candle_count=100,
        calmar_ratio=2.5
    )
    summary = result.summary()
    assert "calmar" in summary
