from src.analysis.trade_analytics import (
    compute_live_paper_drift,
    compute_trade_analytics,
    evaluate_short_side_policy,
    evaluate_side_source_policy,
    evaluate_source_policy,
    normalise_hyperliquid_fill_history,
)


# ── Tainted-trade fix: ensure analytics gates ignore artefacts ──


def _losing_trade(idx: int, *, tainted: bool = False, source: str = "strategy:momentum_short") -> dict:
    """Build a single losing-short trade with optional tainted flag."""
    meta = {"source_key": source, "close_reason": "live_reconciled_closed"}
    if tainted:
        meta["tainted"] = True
        meta["taint_reason"] = "reconciler_kill_pre_fix"
    return {
        "side": "short",
        "coin": "BTC",
        "pnl": -5.0 - idx * 0.1,
        "metadata": meta,
    }


def test_tainted_trades_excluded_from_summary():
    trades = [_losing_trade(i, tainted=True) for i in range(20)] + [
        {"side": "short", "coin": "BTC", "pnl": +3.0,
         "metadata": {"source_key": "strategy:momentum_short", "close_reason": "take_profit"}},
    ]
    analytics = compute_trade_analytics(trades, source_limit=5)
    # The 20 tainted -$5 trades should NOT appear in the count.
    assert analytics["summary"]["count"] == 1
    assert analytics["summary"]["net_pnl"] == 3.0


def test_tainted_trades_excluded_from_side_source_policy():
    """Regression: pre-fix, 18 tainted reconciler kills made the
    side-source gate read 28% WR / -$142 and block every
    strategy:momentum_short signal forever.  With the taint filter
    those trades are excluded and the gate returns 'insufficient' /
    'healthy' depending on what's left."""
    tainted = [_losing_trade(i, tainted=True) for i in range(18)]
    result = evaluate_side_source_policy(
        tainted,
        side="short",
        source_key="strategy:momentum_short",
        min_trades=5,
        degrade_win_rate=0.40,
        block_win_rate=0.30,
        block_net_pnl=-50.0,
        exact_source=True,
    )
    # Tainted-only history is filtered to empty → "insufficient" sample.
    assert result["status"] == "insufficient"
    assert result["metrics"]["count"] == 0


def test_tainted_trades_excluded_from_source_policy():
    tainted = [_losing_trade(i, tainted=True, source="strategy") for i in range(15)]
    result = evaluate_source_policy(
        tainted,
        source_label="strategy",
        min_trades=5,
        degrade_win_rate=0.40,
        block_win_rate=0.30,
        block_net_pnl=-50.0,
    )
    assert result["status"] == "insufficient"
    assert result["metrics"]["count"] == 0


def test_analytics_include_tainted_env_opt_in_restores_legacy(monkeypatch):
    """Operators can force analytics to count tainted trades again via
    ANALYTICS_INCLUDE_TAINTED=1 (forensic comparisons)."""
    monkeypatch.setenv("ANALYTICS_INCLUDE_TAINTED", "1")
    trades = [_losing_trade(i, tainted=True) for i in range(5)]
    analytics = compute_trade_analytics(trades, source_limit=5)
    assert analytics["summary"]["count"] == 5
    assert analytics["summary"]["net_pnl"] < 0


# ── Phase 6 expansion: ANY reconciler-closed trade is tainted ──


def test_reconciler_close_with_mirror_success_is_still_tainted():
    """A 'mirrored successfully' trade closed via live_reconciled_closed
    has PnL = mid-snapshot, not real exit fill.  Treat as tainted."""
    trades = [
        # NOT tainted flag, mirror succeeded, but close_reason makes it suspect.
        {
            "side": "short", "coin": "BTC", "pnl": -30.0,
            "metadata": {
                "source_key": "strategy:momentum_short",
                "close_reason": "live_reconciled_closed",
                "live_mirror_status": "success",
            },
        },
        # A clean TP close — must stay counted.
        {
            "side": "short", "coin": "BTC", "pnl": +5.0,
            "metadata": {
                "source_key": "strategy:momentum_short",
                "close_reason": "take_profit",
            },
        },
    ]
    analytics = compute_trade_analytics(trades, source_limit=5)
    assert analytics["summary"]["count"] == 1
    assert analytics["summary"]["net_pnl"] == 5.0


def test_reconciler_close_via_reconciliation_reason_alias():
    """Legacy rows used `reconciliation_reason` instead of
    `close_reason`. Both must trigger the taint."""
    trades = [{
        "side": "long", "coin": "ETH", "pnl": -50.0,
        "metadata": {
            "source_key": "strategy:momentum_long",
            "reconciliation_reason": "live_reconciled_closed",
            "live_mirror_status": "success",
        },
    }]
    analytics = compute_trade_analytics(trades, source_limit=5)
    assert analytics["summary"]["count"] == 0


def test_reconciler_close_taint_excluded_from_side_source_gate():
    """The downstream firewall gate must read this lower count too."""
    trades = [
        {"side": "short", "coin": "BTC", "pnl": -30.0,
         "metadata": {"source_key": "strategy:momentum_short",
                      "close_reason": "live_reconciled_closed",
                      "live_mirror_status": "success"}}
        for _ in range(10)
    ]
    # All 10 are reconciler closes -> filtered to empty -> "insufficient" sample.
    result = evaluate_side_source_policy(
        trades,
        side="short",
        source_key="strategy:momentum_short",
        min_trades=5,
        degrade_win_rate=0.40,
        block_win_rate=0.30,
        block_net_pnl=-50.0,
        exact_source=True,
    )
    assert result["status"] == "insufficient"
    assert result["metrics"]["count"] == 0


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
    exact_row = next(row for row in analytics["by_exact_source"] if row["label"] == "copy_trade:0xabc")
    assert exact_row["count"] == 2
    exact_side_row = next(
        row for row in analytics["by_exact_source_side"]
        if row["label"] == "copy_trade:0xabc short"
    )
    assert exact_side_row["net_pnl"] == -0.8


def test_normalise_hyperliquid_fill_history_maps_closed_side_and_fees():
    rows = normalise_hyperliquid_fill_history(
        [
            {
                "coin": "SOL",
                "dir": "Open Short",
                "side": "sell",
                "time": 1700000000000,
                "closedPnl": "0",
                "fee": "0.10",
            },
            {
                "coin": "SOL",
                "dir": "Close Short",
                "side": "buy",
                "time": 1700000001000,
                "closedPnl": "-2.00",
                "fee": "0.40",
                "hash": "h2",
                "oid": 456,
            },
            {
                "coin": "ETH",
                "dir": "Open Long",
                "side": "buy",
                "time": 1700000002000,
                "sz": "2.0",
                "closedPnl": "-0.10",
                "fee": "0.10",
            },
            {
                "coin": "ETH",
                "dir": "Close Long",
                "side": "sell",
                "time": 1700000003000,
                "sz": "2.0",
                "closedPnl": "1.50",
                "fee": "0.25",
                "hash": "h1",
                "oid": 123,
            },
        ],
        limit=10,
        subtract_fees=True,
    )

    assert [row["coin"] for row in rows] == ["ETH", "SOL"]
    assert rows[0]["side"] == "long"
    assert rows[0]["pnl"] == 1.15
    assert rows[0]["metadata"]["total_fees_paid"] == 0.35
    assert rows[0]["metadata"]["matched_entry_fee_paid"] == 0.1
    assert rows[1]["side"] == "short"
    assert rows[1]["pnl"] == -2.4
    assert rows[1]["metadata"]["gross_pnl_before_fees"] == -2.0


def test_compute_trade_analytics_includes_tp_sl_path_metrics():
    analytics = compute_trade_analytics(
        [
            {
                "coin": "ETH",
                "side": "long",
                "pnl": 1.0,
                "metadata": {
                    "source_key": "strategy:path",
                    "max_r_multiple": 2.5,
                    "min_r_multiple": -0.4,
                    "exit_r_multiple": 1.5,
                    "path_capture_ratio": 0.6,
                },
            },
            {
                "coin": "ETH",
                "side": "long",
                "pnl": -0.5,
                "metadata": {
                    "source_key": "strategy:path",
                    "max_r_multiple": 0.2,
                    "min_r_multiple": -1.0,
                    "exit_r_multiple": -1.0,
                    "path_capture_ratio": 0.0,
                },
            },
        ]
    )

    summary = analytics["summary"]
    assert summary["path_count"] == 2
    assert summary["avg_mfe_r"] == 1.35
    assert summary["avg_mae_r"] == -0.7
    assert summary["avg_exit_r"] == 0.25
    assert summary["avg_path_capture_ratio"] == 0.3


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
    # H28: require enough trades that a 0% win rate is statistically
    # distinguishable from a 50% null (binomial p-value < 0.20).  At
    # n=3 the two-sided p is 0.25 -- too noisy to block on; at n>=5 it
    # drops to 0.0625 and the gate fires correctly.  Bumped sample
    # size accordingly.
    trades = [
        {"side": "short", "pnl": -10.0, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"side": "short", "pnl": -8.0, "metadata": {"source_key": "copy_trade:0xdef"}},
        {"side": "long", "pnl": -9.0, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"side": "short", "pnl": -7.0, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"side": "long", "pnl": -6.0, "metadata": {"source_key": "copy_trade:0xdef"}},
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
    assert policy["metrics"]["count"] == 5
    assert policy["metrics"]["net_pnl"] == -40.0
    # H28: pvalue is exposed in the metrics for downstream auditing.
    assert policy["metrics"]["win_rate_pvalue"] is not None
    assert policy["metrics"]["win_rate_pvalue"] < 0.20


def test_evaluate_side_source_policy_blocks_bad_exact_copy_short_only():
    trades = [
        {"coin": "SOL", "side": "short", "pnl": -0.4, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"coin": "ETH", "side": "short", "pnl": -0.3, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"coin": "BTC", "side": "short", "pnl": -0.2, "metadata": {"source_key": "copy_trade:0xabc"}},
        {"coin": "SOL", "side": "short", "pnl": 0.8, "metadata": {"source_key": "copy_trade:0xdef"}},
        {"coin": "SOL", "side": "long", "pnl": 0.5, "metadata": {"source_key": "copy_trade:0xabc"}},
    ]

    bad_source_short = evaluate_side_source_policy(
        trades,
        side="short",
        source_key="copy_trade:0xabc",
        min_trades=3,
        degrade_win_rate=0.45,
        block_win_rate=0.35,
        block_net_pnl=-0.25,
    )
    good_source_short = evaluate_side_source_policy(
        trades,
        side="short",
        source_key="copy_trade:0xdef",
        min_trades=1,
        degrade_win_rate=0.45,
        block_win_rate=0.35,
        block_net_pnl=-0.25,
    )

    assert bad_source_short["status"] == "blocked"
    assert bad_source_short["metrics"]["count"] == 3
    assert good_source_short["status"] == "healthy"


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
