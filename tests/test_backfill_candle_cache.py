"""Tests for the candle-cache backfill script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def mod():
    path = Path(__file__).resolve().parent.parent / "scripts" / "backfill_candle_cache.py"
    spec = importlib.util.spec_from_file_location("backfill_candle_cache", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_default_coins_are_crypto_perps(mod):
    coins = mod.DEFAULT_COINS.split(",")
    assert "SOL" in coins and "HYPE" in coins and "IP" in coins
    # No HL equity/index/spot instruments in the default perp set.
    assert not any(c.startswith(("xyz:", "para:", "@")) or "/" in c for c in coins)


def test_backfill_calls_fetcher_per_coin_and_aggregates(mod, monkeypatch):
    import src.backtest.data_fetcher as df

    calls = []

    class _Fetcher:
        def __init__(self, cache_dir="data"):
            self.cache_dir = cache_dir

        def fetch_candles(self, coin, timeframe, start=None, end=None):
            calls.append((coin, timeframe, start, end))
            return [object()] * 100  # 100 candles

    monkeypatch.setattr(df, "DataFetcher", _Fetcher)
    out = mod.backfill(["SOL", "HYPE"], "1h", "2026-03-01", "2026-04-23", "/data")
    assert out == {"SOL": 100, "HYPE": 100}
    assert [c[0] for c in calls] == ["SOL", "HYPE"]
    assert calls[0][1] == "1h" and calls[0][3] == "2026-04-23"


def test_backfill_one_bad_coin_does_not_abort(mod, monkeypatch):
    import src.backtest.data_fetcher as df

    class _Fetcher:
        def __init__(self, cache_dir="data"):
            pass

        def fetch_candles(self, coin, timeframe, start=None, end=None):
            if coin == "BADCOIN":
                raise RuntimeError("no such coin")
            return [object()] * 5

    monkeypatch.setattr(df, "DataFetcher", _Fetcher)
    out = mod.backfill(["SOL", "BADCOIN", "ETH"], "1h", "a", "b", "data")
    assert out["SOL"] == 5 and out["ETH"] == 5
    assert isinstance(out["BADCOIN"], str) and out["BADCOIN"].startswith("ERR:")


def test_main_runs_with_defaults(mod, monkeypatch):
    import src.backtest.data_fetcher as df

    monkeypatch.setattr(
        df, "DataFetcher",
        type("F", (), {"__init__": lambda self, cache_dir="data": None,
                       "fetch_candles": lambda self, *a, **k: []}),
    )
    # Should parse defaults and complete without error.
    assert mod.main(["--coins", "SOL", "--start", "2026-03-01", "--end", "2026-03-02"]) == 0
