from math import sin

from src.backtest.data_fetcher import Candle
from src.backtest.research_suite import (
    DEFAULT_10_COIN_UNIVERSE,
    DEFAULT_12_STRATEGIES,
    run_cross_coin_strategy_sweep,
    run_walk_forward_parameter_optimization,
)


def _candles(coin: str, n: int = 220):
    out = []
    price = 100.0
    for i in range(n):
        price += sin(i / 4.0) * 0.8 + (0.03 if i % 30 < 15 else -0.02)
        out.append(Candle(
            timestamp_ms=i * 3_600_000,
            open=price - 0.2,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1_000 + i,
            coin=coin,
            timeframe="1h",
        ))
    return out


def test_default_research_universe_and_strategy_counts():
    assert len(DEFAULT_10_COIN_UNIVERSE) == 10
    assert len(DEFAULT_12_STRATEGIES) == 12
    assert "rsi" in DEFAULT_12_STRATEGIES
    assert "mean_reversion" in DEFAULT_12_STRATEGIES


def test_cross_coin_strategy_sweep_reports_survivorship_shape():
    report = run_cross_coin_strategy_sweep(
        {"BTC": _candles("BTC", 120), "ETH": _candles("ETH", 120)},
        strategies=["rsi", "mean_reversion"],
        min_candles=50,
    )

    assert set(report.per_strategy) == {"rsi", "mean_reversion"}
    assert report.per_strategy["rsi"]["coins_tested"] == 2
    assert "survives_cross_coin" in report.per_strategy["rsi"]
    assert report.skipped == {}


def test_walk_forward_parameter_optimization_uses_train_then_oos_test():
    report = run_walk_forward_parameter_optimization(
        _candles("BTC", 240),
        coin="BTC",
        strategy="rsi",
        param_grid={"rsi_period": [7, 14], "rsi_oversold": [25.0], "rsi_overbought": [75.0]},
        train_days=3,
        test_days=2,
        step_days=2,
        min_train_candles=24,
        min_test_candles=12,
    )

    assert report.strategy == "rsi"
    assert report.coin == "BTC"
    assert report.aggregate["fold_count"] >= 1
    assert report.folds[0]["selected_params"]["rsi_period"] in {7, 14}
    assert "test_total_pnl" in report.aggregate
