#!/usr/bin/env python3
"""
Seed & Replay — One-command reproducible backtest from a fresh checkout
========================================================================

Loads the sample dataset from fixtures/sample_data.json, generates
deterministic wallet fills, seeds the SQLite database, and runs a
full backtester replay.

Usage:
    python scripts/seed_and_replay.py                # seed + backtest
    python scripts/seed_and_replay.py --seed-only    # just populate DB
    python scripts/seed_and_replay.py --replay-only  # backtest existing DB
    python scripts/seed_and_replay.py --sweep         # parameter sweep

Requires only: numpy, pandas, requests (no websocket/eth_account needed).
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ─── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.data import database as db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_replay")

FIXTURES_PATH = ROOT / "fixtures" / "sample_data.json"

# ─── Deterministic price generator ────────────────────────────
# Generates a realistic price series with trends, mean-reversion,
# and vol clusters so the backtester has something to trade against.

_BASE_PRICES = {
    "BTC": 95000, "ETH": 3400, "SOL": 180,
    "DOGE": 0.32, "ARB": 1.15, "LINK": 18.5,
}

_VOLATILITY = {
    "BTC": 0.025, "ETH": 0.030, "SOL": 0.045,
    "DOGE": 0.060, "ARB": 0.050, "LINK": 0.035,
}


def _generate_price_series(coin: str, n_points: int, seed: int = 42) -> list:
    """
    Geometric Brownian Motion with mean-reversion and vol clustering.
    Returns list of (time_ms, price) tuples spanning ~90 days.
    """
    rng = random.Random(seed + hash(coin))
    base = _BASE_PRICES.get(coin, 100)
    vol = _VOLATILITY.get(coin, 0.03)

    start_ms = int(datetime(2025, 12, 20).timestamp() * 1000)
    interval_ms = int(90 * 86400 * 1000 / n_points)  # spread over 90 days

    prices = []
    price = base
    trend = 0.0
    vol_state = vol

    for i in range(n_points):
        # Vol clustering (GARCH-like)
        vol_shock = rng.gauss(0, 0.002)
        vol_state = max(vol * 0.5, min(vol * 2.0, vol_state * 0.95 + vol * 0.05 + vol_shock))

        # Mean-reversion pull toward base
        mr_force = -0.001 * (price - base) / base

        # Trend component (momentum)
        trend = trend * 0.98 + rng.gauss(0, 0.0005)

        # Daily return
        ret = trend + mr_force + rng.gauss(0, vol_state)
        price *= (1 + ret)
        price = max(price * 0.3, price)  # floor at 30% of current

        t = start_ms + i * interval_ms
        prices.append((t, round(price, 6)))

    return prices


# ─── Fill generator ───────────────────────────────────────────

def _generate_fills_for_wallet(wallet: dict, strategies: list, seed: int = 42) -> list:
    """
    Generate realistic fills for a golden wallet based on its stats.
    Each fill has: wallet_address, coin, side, original_price, penalised_price,
    size, time_ms, delayed_time_ms, closed_pnl, penalised_pnl, fee,
    is_liquidation, direction.
    """
    rng = random.Random(seed + hash(wallet["address"]))
    address = wallet["address"]
    total_fills = wallet["total_fills"]
    coins = json.loads(wallet["coins_traded"])
    win_rate = wallet["win_rate"]

    fills = []

    # Generate price series for each coin this wallet trades
    price_series = {}
    for coin in coins:
        price_series[coin] = _generate_price_series(coin, total_fills * 4, seed)

    # Generate open/close fill pairs
    fill_count = 0
    price_idx = {c: 0 for c in coins}

    while fill_count < total_fills and all(price_idx[c] < len(price_series[c]) - 2 for c in coins):
        coin = rng.choice(coins)
        idx = price_idx[coin]
        if idx >= len(price_series[coin]) - 2:
            continue

        t_open, p_open = price_series[coin][idx]
        # Hold for 1-20 price ticks
        hold_ticks = rng.randint(1, min(20, len(price_series[coin]) - idx - 1))
        t_close, p_close = price_series[coin][idx + hold_ticks]
        price_idx[coin] = idx + hold_ticks + 1

        is_long = rng.random() < 0.55  # slight long bias
        side_open = "buy" if is_long else "sell"
        side_close = "sell" if is_long else "buy"
        direction_open = "Open Long" if is_long else "Open Short"
        direction_close = "Close Long" if is_long else "Close Short"

        # Determine if this trade wins (biased by wallet win_rate)
        is_win = rng.random() < win_rate

        # Adjust close price to match win/loss
        if is_long:
            if is_win:
                p_close = p_open * (1 + rng.uniform(0.005, 0.08))
            else:
                p_close = p_open * (1 - rng.uniform(0.005, 0.06))
        else:
            if is_win:
                p_close = p_open * (1 - rng.uniform(0.005, 0.08))
            else:
                p_close = p_open * (1 + rng.uniform(0.005, 0.06))

        # Size in coin units (USD $500–$5000 notional)
        notional = rng.uniform(500, 5000)
        size = round(notional / p_open, 6)

        # Penalised prices (taker fee + slippage = 4.5 bps per leg)
        slippage = 0.00045
        if is_long:
            pen_open = p_open * (1 + slippage)
            pen_close = p_close * (1 - slippage)
        else:
            pen_open = p_open * (1 - slippage)
            pen_close = p_close * (1 + slippage)

        # PnL
        if is_long:
            raw_pnl = (p_close - p_open) * size
            pen_pnl = (pen_close - pen_open) * size
        else:
            raw_pnl = (p_open - p_close) * size
            pen_pnl = (pen_open - pen_close) * size

        fee = abs(notional * slippage * 2)  # both legs

        # OPEN fill
        fills.append({
            "wallet_address": address,
            "coin": coin,
            "side": side_open,
            "original_price": round(p_open, 6),
            "penalised_price": round(pen_open, 6),
            "size": size,
            "time_ms": t_open,
            "delayed_time_ms": t_open + 100,
            "closed_pnl": 0.0,
            "penalised_pnl": 0.0,
            "fee": round(fee / 2, 4),
            "is_liquidation": 0,
            "direction": direction_open,
        })

        # CLOSE fill
        fills.append({
            "wallet_address": address,
            "coin": coin,
            "side": side_close,
            "original_price": round(p_close, 6),
            "penalised_price": round(pen_close, 6),
            "size": size,
            "time_ms": t_close,
            "delayed_time_ms": t_close + 100,
            "closed_pnl": round(raw_pnl, 4),
            "penalised_pnl": round(pen_pnl, 4),
            "fee": round(fee / 2, 4),
            "is_liquidation": 0,
            "direction": direction_close,
        })

        fill_count += 2

    # Sort by time
    fills.sort(key=lambda f: f["time_ms"])
    return fills


# ─── Database seeding ─────────────────────────────────────────

def seed_database():
    """Load fixtures/sample_data.json and populate the database."""
    logger.info("Loading sample data from %s", FIXTURES_PATH)

    with open(FIXTURES_PATH) as f:
        data = json.load(f)

    meta = data["_meta"]
    logger.info("Dataset: %s wallets, %d target fills, %s",
                meta["wallets"], meta["fills"], meta["date_range"])

    # Init DB schema
    db.init_db()

    with db.get_connection() as conn:
        cur = conn.cursor()

        # ─── Traders ───────────────────────────────────────────
        for t in data["traders"]:
            cur.execute("""
                INSERT OR REPLACE INTO traders
                (address, first_seen, last_updated, total_pnl, roi_pct,
                 account_value, win_rate, trade_count, active, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (t["address"], t["first_seen"], t["last_updated"],
                  t["total_pnl"], t["roi_pct"], t["account_value"],
                  t["win_rate"], t["trade_count"], t["active"], t["metadata"]))
        logger.info("  Seeded %d traders", len(data["traders"]))

        # ─── Golden wallets ────────────────────────────────────
        for gw in data["golden_wallets"]:
            cur.execute("""
                INSERT OR REPLACE INTO golden_wallets
                (address, bot_score, total_fills, raw_pnl, penalised_pnl,
                 max_drawdown_pct, penalised_max_drawdown_pct, sharpe_ratio,
                 win_rate, trades_per_day, is_golden, coins_traded, best_coin,
                 worst_coin, evaluated_at, connected_to_live)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (gw["address"], gw["bot_score"], gw["total_fills"],
                  gw["raw_pnl"], gw["penalised_pnl"], gw["max_drawdown_pct"],
                  gw["penalised_max_drawdown_pct"], gw["sharpe_ratio"],
                  gw["win_rate"], gw["trades_per_day"], gw["is_golden"],
                  gw["coins_traded"], gw["best_coin"], gw["worst_coin"],
                  gw["evaluated_at"], gw["connected_to_live"]))
        logger.info("  Seeded %d golden wallets", len(data["golden_wallets"]))

        # ─── Strategies ────────────────────────────────────────
        for s in data["strategies"]:
            cur.execute("""
                INSERT OR REPLACE INTO strategies
                (name, description, strategy_type, parameters, discovered_at,
                 last_scored, current_score, total_pnl, trade_count, win_rate,
                 sharpe_ratio, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (s["name"], s["description"], s["strategy_type"],
                  s["parameters"], s["discovered_at"], s["last_scored"],
                  s["current_score"], s["total_pnl"], s["trade_count"],
                  s["win_rate"], s["sharpe_ratio"], s["active"]))
        logger.info("  Seeded %d strategies", len(data["strategies"]))

        # ─── Generate and insert wallet fills ──────────────────
        total_fills = 0
        for gw in data["golden_wallets"]:
            fills = _generate_fills_for_wallet(gw, data["strategies"], seed=42)
            for fill in fills:
                cur.execute("""
                    INSERT INTO wallet_fills
                    (wallet_address, coin, side, original_price, penalised_price,
                     size, time_ms, delayed_time_ms, closed_pnl, penalised_pnl,
                     fee, is_liquidation, direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (fill["wallet_address"], fill["coin"], fill["side"],
                      fill["original_price"], fill["penalised_price"],
                      fill["size"], fill["time_ms"], fill["delayed_time_ms"],
                      fill["closed_pnl"], fill["penalised_pnl"], fill["fee"],
                      fill["is_liquidation"], fill["direction"]))
                total_fills += 1

        logger.info("  Generated and seeded %d wallet fills (deterministic, seed=42)", total_fills)

    # ─── Verify ────────────────────────────────────────────
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM traders")
        n_traders = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM golden_wallets WHERE is_golden = 1")
        n_golden = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM strategies")
        n_strats = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM wallet_fills")
        n_fills = cur.fetchone()[0]

    logger.info("Database seeded: %d traders, %d golden wallets, "
                "%d strategies, %d fills", n_traders, n_golden, n_strats, n_fills)
    return n_fills


def run_replay(sweep: bool = False):
    """Run the backtester on seeded data."""
    from src.backtest.backtester import (
        BacktestConfig, BacktestEngine, BacktestFill,
        load_fills_from_db, init_experiments_table, save_experiment,
        parameter_sweep,
    )

    fills = load_fills_from_db(golden_only=True)
    if not fills:
        logger.error("No fills in database. Run with --seed-only first.")
        return None

    n_wallets = len(set(f.wallet_address for f in fills))
    logger.info("Loaded %d fills from %d golden wallets", len(fills), n_wallets)

    if sweep:
        logger.info("Running parameter sweep...")
        results = parameter_sweep(fills)

        print(f"\n{'='*90}")
        print(f"  Parameter Sweep — {len(results)} configurations tested")
        print(f"{'='*90}")
        print(f"  {'Rank':>4}  {'Sharpe':>7}  {'PnL':>11}  {'WR%':>6}  "
              f"{'MaxDD':>7}  {'PF':>6}  {'Trades':>7}")
        print(f"  {'─'*60}")

        for i, r in enumerate(results[:10]):
            print(f"  {i+1:>4}  {r.sharpe_ratio:>7.3f}  ${r.total_pnl:>9,.2f}  "
                  f"{r.win_rate:>5.1f}%  {r.max_drawdown_pct:>6.1f}%  "
                  f"{r.profit_factor:>5.2f}  {r.total_trades:>7}")

        init_experiments_table()
        for r in results[:5]:
            save_experiment(r, notes="sample_data_sweep")
        logger.info("Saved top 5 experiments to DB")
        return results[0] if results else None

    else:
        cfg = BacktestConfig(
            initial_balance=10_000,
            max_position_pct=0.08,
            max_positions=5,
            max_per_coin=2,
            max_leverage=5.0,
            max_aggregate_exposure_pct=0.50,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            trailing_stop=True,
            trailing_pct=0.025,
            copy_delay_ms=2000,
            slippage_bps=4.5,
            seed=42,
        )

        engine = BacktestEngine(cfg)
        result = engine.run(fills, experiment_id="sample_replay_001")

        print(f"\n{'='*60}")
        print(f"  Sample Replay Results")
        print(f"{'='*60}")
        print(f"  Wallets:           {len(result.wallets_used)}")
        print(f"  Total Trades:      {result.total_trades}")
        print(f"  Win Rate:          {result.win_rate:.1f}%")
        print(f"  Total PnL:         ${result.total_pnl:+,.2f}")
        print(f"  Max Drawdown:      {result.max_drawdown_pct:.1f}%")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio:     {result.sortino_ratio:.3f}")
        print(f"  Profit Factor:     {result.profit_factor:.3f}")
        print(f"  Calmar Ratio:      {result.calmar_ratio:.3f}")
        print(f"  Expectancy:        {result.expectancy:.4f}")
        print(f"  Avg Hold Time:     {result.avg_hold_hours:.1f}h")
        print(f"  Duration:          {result.duration_seconds:.1f}s")

        if result.coin_breakdown:
            print(f"\n  Coin Breakdown:")
            print(f"    {'Coin':<8}  {'Trades':>7}  {'WR%':>6}  {'PnL':>11}")
            for coin, d in sorted(result.coin_breakdown.items(),
                                    key=lambda x: x[1]["pnl"], reverse=True):
                print(f"    {coin:<8}  {d['trades']:>7}  "
                      f"{d['win_rate']:>5.1f}%  ${d['pnl']:>9,.2f}")

        if engine._rejection_reasons:
            print(f"\n  Signal Rejections ({engine._signals_rejected} total):")
            for reason, count in sorted(engine._rejection_reasons.items(),
                                          key=lambda x: x[1], reverse=True):
                print(f"    {reason:<25}  {count:>5}")

        init_experiments_table()
        save_experiment(result, notes="sample_data_replay")
        logger.info("Saved experiment: %s", result.experiment_id)

        print(f"\n{'='*60}")
        return result


# ─── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed sample data and run a reproducible backtest replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_and_replay.py              # seed DB + run backtest
  python scripts/seed_and_replay.py --seed-only  # just populate the DB
  python scripts/seed_and_replay.py --replay-only # backtest existing DB data
  python scripts/seed_and_replay.py --sweep       # parameter sweep
        """,
    )
    parser.add_argument("--seed-only", action="store_true",
                        help="Only seed the database, don't run backtest")
    parser.add_argument("--replay-only", action="store_true",
                        help="Only run backtest (DB must already be seeded)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep instead of single backtest")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Hyperliquid Trading Bot — Seed & Replay")
    print("=" * 60)

    if not args.replay_only:
        logger.info("Phase 1: Seeding database...")
        n_fills = seed_database()
        if args.seed_only:
            print(f"\nDatabase seeded with {n_fills} fills. "
                  f"Run without --seed-only to backtest.")
            return

    logger.info("Phase 2: Running backtest replay...")
    result = run_replay(sweep=args.sweep)

    if result and hasattr(result, "total_trades") and result.total_trades > 0:
        print("\nReplay complete. The backtester produced trades from the sample data.")
        print("You can now explore the results in the dashboard or run further experiments.")
    else:
        print("\nReplay complete but no trades were generated.")
        print("Run `python scripts/diagnose_rejections.py` to debug.")


if __name__ == "__main__":
    main()
