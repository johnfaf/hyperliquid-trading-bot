from src.analysis.trade_analytics import compute_trade_analytics, evaluate_short_side_policy


def test_compute_trade_analytics_groups_by_side_and_source():
    trades = [
        {
            "side": "long",
            "pnl": 1.2,
            "metadata": {
                "source_key": "strategy:trend",
                "total_fees_paid": 0.1,
                "total_slippage_cost": 0.02,
                "gross_pnl_before_fees": 1.3,
            },
        },
        {
            "side": "short",
            "pnl": -0.6,
            "metadata": {
                "source_key": "copy_trade:0xabc",
                "total_fees_paid": 0.08,
                "total_slippage_cost": 0.01,
                "gross_pnl_before_fees": -0.52,
            },
        },
        {
            "side": "short",
            "pnl": -0.2,
            "metadata": {
                "source_key": "copy_trade:0xabc",
                "total_fees_paid": 0.05,
                "total_slippage_cost": 0.0,
                "gross_pnl_before_fees": -0.15,
            },
        },
    ]

    analytics = compute_trade_analytics(trades, source_limit=5)

    assert analytics["summary"]["count"] == 3
    assert analytics["summary"]["net_pnl"] == 0.4
    short_row = next(row for row in analytics["by_side"] if row["label"] == "short")
    assert short_row["count"] == 2
    assert short_row["net_pnl"] == -0.8
    assert short_row["fees"] == 0.13
    source_row = next(row for row in analytics["by_source"] if row["label"] == "copy_trade:0xabc")
    assert source_row["count"] == 2
    assert source_row["net_pnl"] == -0.8


def test_evaluate_short_side_policy_blocks_bad_short_run():
    trades = [
        {"side": "short", "pnl": -0.7, "metadata": {"source_key": "strategy:a"}},
        {"side": "short", "pnl": -0.4, "metadata": {"source_key": "strategy:b"}},
        {"side": "short", "pnl": -0.2, "metadata": {"source_key": "strategy:c"}},
        {"side": "short", "pnl": 0.1, "metadata": {"source_key": "strategy:d"}},
    ]

    policy = evaluate_short_side_policy(
        trades,
        min_trades=4,
        degrade_win_rate=0.45,
        block_win_rate=0.35,
        block_net_pnl=-1.0,
    )

    assert policy["status"] == "blocked"
    assert policy["metrics"]["count"] == 4
    assert policy["metrics"]["net_pnl"] == -1.2
