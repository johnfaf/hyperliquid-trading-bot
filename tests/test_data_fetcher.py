from src.backtest.data_fetcher import Candle, DataFetcher, TIMEFRAME_MS


def _candle(ts: int) -> Candle:
    return Candle(timestamp_ms=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)


def test_cache_coverage_requires_recent_end():
    interval = TIMEFRAME_MS["1h"]
    candles = [_candle(i * interval) for i in range(10)]

    assert DataFetcher._has_complete_coverage(candles, 0, 10 * interval, interval)
    assert not DataFetcher._has_complete_coverage(candles[:-3], 0, 10 * interval, interval)


def test_cache_coverage_rejects_large_holes():
    interval = TIMEFRAME_MS["1h"]
    candles = [_candle(0), _candle(interval), _candle(5 * interval), _candle(6 * interval)]

    assert not DataFetcher._has_complete_coverage(candles, 0, 6 * interval, interval)
