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


def test_fetch_candles_skips_malformed_rows_without_losing_valid_chunk(tmp_path, monkeypatch):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"t": 0, "o": "1", "h": "1", "l": "1", "c": "1", "v": "1"},
                {"t": 60_000, "o": "1", "h": "0.5", "l": "2", "c": "1", "v": "1"},
                {"t": 60_000, "o": "1", "h": "2", "l": "1", "c": "1.5", "v": "3"},
            ]

    monkeypatch.setattr("src.backtest.data_fetcher.requests.post", lambda *a, **k: FakeResponse())
    fetcher = DataFetcher(cache_dir=str(tmp_path))

    candles = fetcher.fetch_candles(
        "BTC",
        "1m",
        start="1970-01-01",
        end="1970-01-01T00:02:00",
        use_cache=False,
    )

    assert len(candles) == 1
    assert candles[0].timestamp_ms == 60_000
    assert candles[0].close == 1.5
