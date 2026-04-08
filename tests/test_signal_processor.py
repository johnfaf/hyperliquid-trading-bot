from src.signals.signal_processor import SignalProcessor


def _strategy(name: str, strategy_type: str, score: float, coins):
    return {
        "name": name,
        "strategy_type": strategy_type,
        "current_score": score,
        "trade_count": 25,
        "parameters": {"coins": coins},
    }


def test_resolve_conflicts_does_not_duplicate_multi_coin_strategy():
    processor = SignalProcessor({"dedup_enabled": False, "max_signals_out": 50})
    strategies = [
        _strategy("s1", "trend_following", 0.88, ["BTC", "ETH"]),
        _strategy("s2", "swing_trading", 0.74, ["SOL"]),
    ]

    resolved = processor._resolve_conflicts(
        strategies,
        regime_data={"overall_regime": "RANGING"},
    )

    assert len(resolved) == 2
    assert resolved[0]["name"] == "s1"
    assert resolved[1]["name"] == "s2"


def test_process_never_expands_strategy_count():
    processor = SignalProcessor({"max_signals_out": 100})
    strategies = [
        _strategy("s1", "trend_following", 0.88, ["BTC", "ETH"]),
        _strategy("s2", "swing_trading", 0.79, ["SOL", "XRP"]),
        _strategy("s3", "breakout", 0.81, ["DOGE", "AVAX"]),
        _strategy("s4", "momentum_short", 0.76, ["LINK", "ARB"]),
        _strategy("s5", "contrarian", 0.73, ["WIF", "SEI"]),
    ]

    out = processor.process(
        strategies,
        regime_data={"overall_regime": "RANGING"},
    )

    assert len(out) <= len(strategies)
