from src.learning.dataset_builder import DatasetBuildResult, LearningExample
from src.learning.decision_replay_report import build_decision_replay_report
from src.learning.replay_backtester import DecisionReplayBacktester, ReplayPolicy


def _example(
    idx,
    *,
    coin="BTC",
    pnl=1.0,
    confidence=0.8,
    executed=True,
    features=None,
    metadata=None,
    label_win=None,
):
    return LearningExample(
        decision_id=f"d{idx}",
        coin=coin,
        side="long",
        source="strategy",
        created_at=f"2026-04-21T10:{idx % 60:02d}:00+00:00",
        features=features if features is not None else {"momentum": 1.0, "spread_bps": 2.0},
        confidence=confidence,
        executed=executed,
        label_win=label_win if label_win is not None else (1 if pnl > 0 else 0),
        outcome_pnl=pnl,
        paper_trade_id=idx if executed else None,
        metadata=metadata or {},
        source_key="strategy:momentum",
        strategy_type="momentum",
        final_status="paper_closed" if executed else "firewall_prescreen_rejected",
        rejection_reason="" if executed else "confidence_floor",
        regime={"overall_regime": "trending_up"},
        outcome_return_pct=pnl / 100.0,
    )


def test_decision_replay_report_keeps_why_costs_drift_and_counterfactuals():
    metadata = {
        "decision_metadata": {
            "firewall_decision": "approved",
            "proposed_size_usd": 50.0,
            "proposed_leverage": 5.0,
            "proposed_sl_price": 99.0,
            "proposed_tp_price": 105.0,
            "source_health": {"polymarket": {"status": "degraded"}},
        },
        "outcome_metadata": {
            "paper_metadata": {
                "gross_pnl_before_fees": 12.0,
                "total_fees_paid": 1.0,
                "total_slippage_cost": 0.5,
                "funding_paid": 0.25,
                "fill_ratio": 0.75,
                "live_pnl": 9.0,
                "paper_entry_price": 100.0,
                "live_entry_price": 100.2,
            },
            "close_reason": "take_profit",
        },
    }
    rejected_metadata = {
        "decision_metadata": {"firewall_decision": "rejected", "source_health": {"candles": {"status": "down"}}},
        "outcome_metadata": {"forward_label_metadata": {"data_gap_1h": 1, "data_coverage_1h": 0.25}},
    }
    dataset = DatasetBuildResult(
        "ds_report",
        [
            _example(1, coin="BTC", pnl=10.0, metadata=metadata),
            _example(
                2,
                coin="ETH",
                pnl=4.0,
                executed=False,
                features={"momentum": 1.0},
                metadata=rejected_metadata,
                label_win=1,
            ),
        ],
        ["momentum", "spread_bps"],
        {"rows": 2},
    )

    report = build_decision_replay_report(dataset, ReplayPolicy("all", include_rejected=True))

    assert report["summary"]["accepted_by_policy"] == 2
    assert report["execution_costs"]["known_costs"] == 1.75
    assert report["fills"]["partial_fill_count"] == 1
    assert report["counterfactuals"]["held_to_sl_tp"] == 1
    assert report["counterfactuals"]["rejected_would_win"] == 1
    assert report["live_paper_drift"]["coverage_count"] == 1
    assert report["live_paper_drift"]["total_pnl_drift"] == -1.0
    assert report["portfolio"]["multi_coin"] is True
    why = report["trade_reports"][0]["why_entered"]
    assert "strategy=momentum" in why
    assert "risk(" in why
    assert "source_health=polymarket:degraded" in why


def test_replay_backtester_fails_when_accepted_decisions_have_data_gaps():
    examples = [_example(idx, pnl=2.0) for idx in range(30)]
    examples[20] = _example(20, pnl=2.0, features={"momentum": 1.0})
    dataset = DatasetBuildResult("ds_gap", examples, ["momentum", "spread_bps"], {"rows": len(examples)})

    result = DecisionReplayBacktester(min_trades=5, train_fraction=0.5, min_test_trades=5).run(
        dataset,
        ReplayPolicy("candidate_all", min_confidence=0.0),
        persist=False,
    )

    assert result.metrics["split"]["train_passed"] is True
    assert result.metrics["split"]["test_passed"] is True
    assert result.metrics["split"]["data_quality_passed"] is False
    assert result.metrics["decision_replay_report"]["data_quality"]["accepted_data_gap_count"] == 1
    assert result.passed is False


def test_walk_forward_trained_selects_threshold_on_past_then_tests_future():
    examples = []
    for idx in range(60):
        winner = idx % 4 != 0
        examples.append(
            _example(
                idx,
                pnl=2.0 if winner else -3.0,
                confidence=0.82 if winner else 0.45,
            )
        )
    dataset = DatasetBuildResult("ds_walk_forward", examples, ["momentum", "spread_bps"], {"rows": 60})

    result = DecisionReplayBacktester(min_trades=8, train_fraction=0.6, min_test_trades=3).run(
        dataset,
        ReplayPolicy("candidate_trainable", min_confidence=0.0),
        persist=False,
    )

    windows = result.metrics["walk_forward_trained"]
    assert windows
    assert all(window["selected_min_confidence"] >= 0.5 for window in windows)
    assert all("train" in window and "test" in window for window in windows)
