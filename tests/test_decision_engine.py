from src.signals.decision_engine import DecisionEngine


def test_decision_engine_skips_strategy_without_asset():
    engine = DecisionEngine({"min_decision_score": 0.0})

    result = engine.decide(
        [
            {
                "id": 1,
                "name": "generic trend",
                "strategy_type": "trend_following",
                "current_score": 0.9,
                "parameters": {},
                "metrics": {},
            }
        ],
        regime_data={"overall_regime": "trending_up", "overall_confidence": 0.9},
        open_positions=[],
    )

    assert result == []
    assert engine.stats["total_missing_asset"] == 1


def test_decision_engine_prescreens_more_than_execution_cap():
    engine = DecisionEngine(
        {
            "min_decision_score": 0.0,
            "max_trades_per_cycle": 1,
            "max_prescreen_candidates": 3,
            "max_positions": 8,
        }
    )
    strategies = [
        {
            "id": idx,
            "name": f"strategy-{idx}",
            "strategy_type": "momentum_long",
            "current_score": 0.8 - idx * 0.01,
            "parameters": {"coins": [coin]},
        }
        for idx, coin in enumerate(["BTC", "ETH", "SOL", "ARB"])
    ]

    result = engine.decide(strategies, regime_data={}, open_positions=[])

    assert len(result) == 3
    assert [item["_decision_coin"] for item in result] == ["BTC", "ETH", "SOL"]
