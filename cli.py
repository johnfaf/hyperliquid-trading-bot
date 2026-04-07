"""
CLI Entrypoints
===============
All command-line parsing, bootstrap, backtest, cache management, and
paper-trade reset logic.  Keeps ``main.py`` focused on orchestration.

Extracted from the bottom ~400 lines of the old monolithic main.py.
"""
import argparse
from datetime import datetime

import config
from src.core.time_utils import utc_from_timestamp_naive
from src.data import database as db
from src.data.database import init_db, backup_to_json


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def bootstrap_seed_data(logger, days: int = 14):
    """Cold-start bootstrap: seed DB with top trader data."""
    from src.discovery.golden_wallet import init_golden_tables
    from src.discovery.trader_discovery import TraderDiscovery

    logger.info("Bootstrap mode: seeding DB with last %d days of top trader data…", days)
    init_db()
    init_golden_tables()

    discovery = TraderDiscovery()
    logger.info("Step 1/3: Discovering top traders…")
    discovery_result = discovery.run_discovery_cycle()
    humans = discovery_result.get("human_like", 0)
    logger.info("Discovery found %d human-like traders", humans)
    if humans == 0:
        logger.warning("No human-like traders found. Try running again later.")
        return

    logger.info("Step 2/3: Running golden wallet evaluation…")
    from src.discovery.golden_wallet import run_golden_scan
    summary = run_golden_scan(max_wallets=30)
    golden = summary.get("golden", 0)
    logger.info("Golden scan: %d golden wallets found", golden)

    logger.info("Step 3/3: Initializing paper trading account…")
    account = db.get_paper_account()
    if not account:
        db.create_paper_account(config.PAPER_TRADING_INITIAL_BALANCE)
        logger.info("Paper account created: $%s", f"{config.PAPER_TRADING_INITIAL_BALANCE:,.0f}")

    backup_to_json()
    logger.info("Bootstrap complete: %d traders, %d golden wallets, DB backed up", humans, golden)


def run_cli_backtest(logger, args):
    """Run backtest on golden wallet data."""
    from src.backtest.backtester import BacktestEngine, BacktestConfig, BacktestFill
    from src.discovery.golden_wallet import _get_db as gw_get_db
    import csv as _csv

    logger.info("Running CLI backtest…")
    cfg = BacktestConfig()
    conn = gw_get_db()
    try:
        query = "SELECT * FROM wallet_fills WHERE 1=1"
        params = []
        if args.bt_coins:
            coins = [c.strip().upper() for c in args.bt_coins.split(",")]
            placeholders = ",".join("?" * len(coins))
            query += f" AND coin IN ({placeholders})"
            params.extend(coins)
        if args.bt_start:
            start_ms = int(datetime.strptime(args.bt_start, "%Y-%m-%d").timestamp() * 1000)
            query += " AND time_ms >= ?"
            params.append(start_ms)
        if args.bt_end:
            end_ms = int(datetime.strptime(args.bt_end, "%Y-%m-%d").timestamp() * 1000)
            query += " AND time_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY time_ms ASC"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.warning("No fills found matching filters")
        print("No fills found. Run a golden wallet scan first.")
        return

    fills = []
    for r in rows:
        r = dict(r)
        fills.append(BacktestFill(
            wallet_address=r.get("wallet_address", ""),
            coin=r.get("coin", ""), side=r.get("side", ""),
            price=float(r.get("price", 0)),
            original_price=float(r.get("original_price", r.get("price", 0))),
            size=float(r.get("size", 0)),
            time_ms=int(r.get("time_ms", 0)),
            closed_pnl=float(r.get("closed_pnl", 0)),
            direction=r.get("direction", ""),
            is_liquidation=bool(r.get("is_liquidation", False)),
        ))

    logger.info("Loaded %d fills for backtest", len(fills))
    engine = BacktestEngine(cfg)
    result = engine.run(fills)

    print(f"\n{'='*60}")
    print("  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Trades:           {result.total_trades}")
    print(f"  Win Rate:         {result.win_rate:.1f}%")
    print(f"  Total PnL:        ${result.total_pnl:+,.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:    {result.sortino_ratio:.3f}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Calmar Ratio:     {result.calmar_ratio:.3f}")
    print(f"  Expectancy:       {result.expectancy:.4f}")
    print(f"  Avg Hold (h):     {result.avg_hold_hours:.1f}")
    print(f"  Max Consec Loss:  {result.max_consecutive_losses}")
    if result.coin_breakdown:
        print("\n  Per-Coin Breakdown:")
        for coin, data in sorted(result.coin_breakdown.items(), key=lambda x: x[1]["pnl"], reverse=True):
            print(f"    {coin:>6}: {data['trades']:>3} trades, "
                  f"PnL=${data['pnl']:+,.2f}, WR={data.get('win_rate', 0):.0f}%")
    print(f"{'='*60}\n")

    if args.bt_export and result.trades:
        import csv as _csv
        with open(args.bt_export, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow([
                "coin", "side", "entry_price", "exit_price", "size", "leverage",
                "entry_time", "exit_time", "pnl", "pnl_pct", "exit_reason",
                "source_wallet", "hold_hours",
            ])
            for t in result.trades:
                writer.writerow([
                    t.coin, t.side, t.entry_price, t.exit_price, t.size, t.leverage,
                    utc_from_timestamp_naive(t.entry_time_ms / 1000).isoformat(),
                    utc_from_timestamp_naive(t.exit_time_ms / 1000).isoformat(),
                    round(t.pnl, 2), round(t.pnl_pct, 6), t.exit_reason,
                    t.source_wallet, round(t.hold_time_hours, 2),
                ])
        print(f"Exported {len(result.trades)} trades to {args.bt_export}")


def run_candle_backtest(logger, args):
    """Run candle-based backtest."""
    from src.backtest.data_fetcher import DataFetcher
    from src.backtest.candle_backtester import CandleBacktester, CandleBacktestConfig

    fetcher = DataFetcher()
    cfg = CandleBacktestConfig(strategy=args.cbt_strategy)
    use_cache = not args.cbt_no_cache

    if args.cbt_import:
        filepath = args.cbt_import
        if filepath.endswith(".json"):
            candles = fetcher.import_json(filepath, coin=args.cbt_coin, timeframe=args.cbt_timeframe)
        else:
            candles = fetcher.import_csv(filepath, coin=args.cbt_coin, timeframe=args.cbt_timeframe)
        if not candles:
            logger.error("No candles imported from %s", filepath)
            return
    elif args.cbt_multi:
        coins = [c.strip().upper() for c in args.cbt_multi.split(",")]
        bt = CandleBacktester(cfg)
        candle_sets = {}
        for coin in coins:
            logger.info("Fetching %s…", coin)
            data = fetcher.fetch_candles(
                coin, args.cbt_timeframe, start=args.cbt_start, end=args.cbt_end, use_cache=use_cache
            )
            if data:
                candle_sets[coin] = data
        results = bt.run_multi_coin(candle_sets, strategy=args.cbt_strategy)
        print(f"\n{'='*70}")
        print(f"  Multi-Coin Backtest: {args.cbt_strategy} | {args.cbt_timeframe}")
        print(f"{'='*70}")
        print(f"{'Coin':<8} {'Trades':>7} {'Win%':>7} {'PnL':>12} {'MaxDD':>8} {'Sharpe':>8} {'PF':>8}")
        print("-" * 70)
        for coin, r in sorted(results.items(), key=lambda x: x[1].total_pnl, reverse=True):
            print(f"{coin:<8} {r.total_trades:>7} {r.win_rate:>6.1f}% "
                  f"${r.total_pnl:>+10,.2f} {r.max_drawdown_pct:>7.1f}% "
                  f"{r.sharpe_ratio:>7.3f} {r.profit_factor:>7.2f}")
        total = sum(r.total_pnl for r in results.values())
        print(f"{'TOTAL':<8} {'':>7} {'':>7} ${total:>+10,.2f}")
        return
    else:
        candles = fetcher.fetch_candles(
            args.cbt_coin, args.cbt_timeframe,
            start=args.cbt_start, end=args.cbt_end, use_cache=use_cache,
        )
        if not candles:
            logger.error("No candle data found for %s", args.cbt_coin)
            return

    # Parameter sweep
    if args.cbt_sweep:
        parts = args.cbt_sweep.split("=")
        param_name = parts[0]
        vals = parts[1].split(",")
        if len(vals) == 3:
            import numpy as _np
            start_val, end_val, step_val = float(vals[0]), float(vals[1]), float(vals[2])
            sweep_values = _np.arange(start_val, end_val + step_val, step_val).tolist()
            if param_name in ("fast_period", "slow_period", "rsi_period", "bb_period", "atr_period"):
                sweep_values = [int(v) for v in sweep_values]
        else:
            sweep_values = [float(v) if "." in v else int(v) for v in vals]
        bt = CandleBacktester(cfg)
        results = bt.parameter_sweep(candles, param_name, sweep_values, strategy=args.cbt_strategy)
        print(f"\n{'='*70}")
        print(f"  Parameter Sweep: {param_name} | {args.cbt_strategy} | {args.cbt_coin} {args.cbt_timeframe}")
        print(f"{'='*70}")
        for r in results:
            val = r.config.get(param_name, "?")
            print(f"{val:>10} {r.total_trades:>7} {r.win_rate:>6.1f}% "
                  f"${r.total_pnl:>+10,.2f} {r.max_drawdown_pct:>7.1f}% "
                  f"{r.sharpe_ratio:>7.3f}")
        return

    # Standard run
    bt = CandleBacktester(cfg)
    result = bt.run(candles, strategy=args.cbt_strategy)
    print(f"\n{'='*60}")
    print(f"  Candle Backtest: {result.coin} {result.timeframe} | {args.cbt_strategy}")
    print(f"{'='*60}")
    for k, v in result.summary().items():
        print(f"  {k:<20} {v}")

    if args.bt_export and result.trades:
        import csv as _csv
        with open(args.bt_export, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=result.trades[0].keys() if result.trades else [])
            writer.writeheader()
            writer.writerows(result.trades)
        print(f"\n  Exported {len(result.trades)} trades to {args.bt_export}")


def run_cache_list(logger):
    from src.backtest.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    cached = fetcher.list_cached()
    stats = fetcher.get_cache_stats()
    if not cached:
        print("No cached candle data. Run --candle-backtest to fetch some.")
        return
    print(f"\nCandle Cache ({stats['db_size_mb']:.1f} MB, {stats['total_candles']:,} candles)")
    print(f"{'Coin':<8} {'TF':<6} {'Start':<12} {'End':<12} {'Candles':>10} {'Days':>8}")
    print("-" * 60)
    for c in cached:
        print(f"{c['coin']:<8} {c['timeframe']:<6} {c['start']:<12} {c['end']:<12} "
              f"{c['candles']:>10,} {c['days']:>7.0f}")


def run_cache_clear(logger):
    from src.backtest.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    stats_before = fetcher.get_cache_stats()
    fetcher.clear_cache()
    print(f"Cleared {stats_before['total_candles']:,} candles ({stats_before['db_size_mb']:.1f} MB)")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperliquid Auto-Research Trading Bot")
    parser.add_argument(
        "--runtime-profile",
        choices=("paper", "shadow", "live"),
        default=None,
        help="Apply the named runtime profile before config loads.",
    )
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--report", action="store_true", help="Generate a report and exit")
    parser.add_argument("--status", action="store_true", help="Print current status and exit")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Cold start: seed DB with top trader data")
    parser.add_argument("--bootstrap-days", type=int, default=14)
    parser.add_argument("--reset-paper", action="store_true",
                        help="Clear all paper trades and reset balance")
    parser.add_argument("--reset-balance", type=float, default=None)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--bt-coins", type=str, default=None)
    parser.add_argument("--bt-start", type=str, default=None)
    parser.add_argument("--bt-end", type=str, default=None)
    parser.add_argument("--bt-strategy", type=str, default=None)
    parser.add_argument("--bt-export", type=str, default=None)
    parser.add_argument("--candle-backtest", action="store_true")
    parser.add_argument("--cbt-coin", type=str, default="BTC")
    parser.add_argument("--cbt-timeframe", type=str, default="1h")
    parser.add_argument("--cbt-start", type=str, default=None)
    parser.add_argument("--cbt-end", type=str, default=None)
    parser.add_argument("--cbt-strategy", type=str, default="momentum")
    parser.add_argument("--cbt-import", type=str, default=None)
    parser.add_argument("--cbt-sweep", type=str, default=None)
    parser.add_argument("--cbt-multi", type=str, default=None)
    parser.add_argument("--cbt-no-cache", action="store_true")
    parser.add_argument("--cache-list", action="store_true")
    parser.add_argument("--cache-clear", action="store_true")
    parser.add_argument("--core-only", action="store_true",
                        help="Run with fundable-core profile only (minimal subsystems)")
    return parser
