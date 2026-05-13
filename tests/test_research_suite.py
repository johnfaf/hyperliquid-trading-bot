from src.backtest.candle_backtester import CandleBacktestConfig
from src.backtest.data_fetcher import Candle
from src.backtest.research_suite import (
    infer_candle_regime,
    run_regime_conditional_strategy_enablement,
    strategy_allowed_in_regime,
)


def _candles(start_price=100.0, step=1.0, n=80):
    rows = []
    price = start_price
    for idx in range(n):
        price += step
        rows.append(
            Candle(
                timestamp_ms=idx * 3_600_000,
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000.0,
                coin="BTC",
                timeframe="1h",
            )
        )
    return rows


def test_strategy_allowed_in_regime_maps_trend_and_range():
    assert strategy_allowed_in_regime("momentum", "trending_up") is True
    assert strategy_allowed_in_regime("mean_reversion", "trending_up") is False
    assert strategy_allowed_in_regime("mean_reversion", "ranging") is True


def test_regime_conditional_strategy_enablement_compares_gated_vs_always_on():
    candles = _candles()
    assert infer_candle_regime(candles) == "trending_up"

    report = run_regime_conditional_strategy_enablement(
        {"BTC": candles},
        strategies=["momentum", "mean_reversion"],
        config=CandleBacktestConfig(initial_balance=1000, position_size_pct=0.1),
        segment_candles=80,
        min_segment_candles=40,
    )

    by_strategy = {row["strategy"]: row for row in report.by_strategy}
    assert by_strategy["momentum"]["allowed_segments"] == 1
    assert by_strategy["mean_reversion"]["allowed_segments"] == 0
    assert report.summary["segments"] == 2

