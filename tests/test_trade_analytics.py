from src.analysis.trade_analytics import (
    compute_live_paper_drift,
    compute_trade_analytics,
    evaluate_short_side_policy,
    evaluate_source_policy,
)


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
    source_row = next(row for row in analytics["by_source"] if row["label"] == "copy_trade")
    assert source_row["count"] == 2
    assert source_row["net_pnl"] == -0.8
    coin_side_row = next(row for row in analytics["by_coin_side"] if row["label"] == "UNKNOWN short")
    assert coin_side_row["count"] == 2
    assert coin_side_row["net_pnl"] == -0.8


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


def test_evaluate_source_policy_blocks_bad_copy_trades():
    trades = [
        {"side": "short", "pnl": -10.0, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"side": "short", "pnl": -8.0, "metadata": {"source_key": "copy_trade:0xdef"}},
        {"side": "long", "pnl": -9.0, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"side": "short", "pnl": 1.0, "metadata": {"source_key": "strategy:trend"}},
    ]

    policy = evaluate_source_policy(
        trades,
        source_label="copy_trade",
        min_trades=3,
        degrade_win_rate=0.40,
        block_win_rate=0.25,
        block_net_pnl=-25.0,
    )

    assert policy["status"] == "blocked"
    assert policy["metrics"]["count"] == 3
    assert policy["metrics"]["net_pnl"] == -27.0


def test_compute_live_paper_drift_combines_paper_audit_and_live_counts():
    drift = compute_live_paper_drift(
        closed_trades=[
            {"side": "short", "pnl": -12.0, "metadata": {"source_key": "copy_trade:0xabc"}},
            {"side": "long", "pnl": 4.5, "metadata": {"source_key": "strategy:mean_reversion"}},
        ],
        open_trades=[
            {"side": "long", "metadata": {"source": "strategy"}},
            {"side": "short", "metadata": {"source": "copy_trade", "is_copy_trade": True}},
        ],
        audit_rows=[
            {"action": "signal_approved", "source": "copy_trade", "side": "short", "details": {}},
            {"action": "signal_rejected", "source": "options_flow", "side": "short", "details": {"reason": "Cooldown"}},
        ],
        live_source_orders_today={"copy_trade:0xabc": 1, "strategy:mean_reversion": 2},
        source_limit=5,
    )

    assert drift["summary"]["paper_open_positions"] == 2
    assert drift["summary"]["paper_closed_trades"] == 2
    assert drift["summary"]["live_entries_today"] == 3
    assert drift["summary"]["approval_gap"] == -2
    copy_row = next(row for row in drift["by_source"] if row["label"] == "copy_trade")
    assert copy_row["paper_open"] == 1
    assert copy_row["paper_closed"] == 1
    assert copy_row["live_entries_today"] == 1
    options_row = next(row for row in drift["by_source"] if row["label"] == "options_flow")
    assert options_row["top_reject_reason"] == "Cooldown"
