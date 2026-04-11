#!/usr/bin/env python3
# ruff: noqa: E402
"""
Download Real Market Data & Stress Test
========================================
Downloads real candle data from Hyperliquid for random coins and time
periods, generates realistic wallet fills from the actual price action,
seeds the database, and runs the full stress test suite.

This gives us confidence that the backtester and stress engine handle
real market conditions — not just synthetic GBM data.

Usage:
    python scripts/download_real_data_stress_test.py                  # full run
    python scripts/download_real_data_stress_test.py --download-only  # just fetch data
    python scripts/download_real_data_stress_test.py --coins BTC ETH SOL
    python scripts/download_real_data_stress_test.py --n-coins 12     # random 12 coins
    python scripts/download_real_data_stress_test.py --no-stress      # skip stress test
"""
import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ─── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import database as db
from src.discovery.golden_wallet import init_golden_tables
from src.backtest.data_fetcher import DataFetcher, Candle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("real_data_stress")

# ─── Coin pool (liquid Hyperliquid perps) ─────────────────────
# Curated for liquidity — avoids micro-cap memecoins that may have
# sparse candle data or extreme gaps.
COIN_POOL = [
    "BTC", "ETH", "SOL", "DOGE", "ARB", "LINK", "AVAX", "OP",
    "SUI", "INJ", "XRP", "LTC", "BNB", "APT", "AAVE", "NEAR",
    "FIL", "TIA", "SEI", "RUNE", "FET", "RENDER", "DOT", "ADA",
    "UNI", "PENDLE", "WIF", "JUP", "HBAR", "TON", "STX", "TAO",
    "ORDI", "ENA", "MKR", "COMP", "CRV", "SNX", "IMX", "AR",
]

# Hyperliquid candle API data starts ~2025-09-06.  We define windows
# within that range so every fetch returns real data.
_TODAY = datetime.now(timezone.utc)

DATE_WINDOWS = [
    ("2025-sep-launch",    "2025-09-10", "2025-10-31"),
    ("2025-Q4-fall",       "2025-11-01", "2025-12-31"),
    ("2026-Q1-newYear",    "2026-01-01", "2026-03-31"),
]

# Filter windows to only include ones that have started before today
DATE_WINDOWS = [
    (label, s, e) for label, s, e in DATE_WINDOWS
    if datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc) < _TODAY
]


def pick_random_coins(n: int, seed: int = None) -> List[str]:
    """Pick n random coins from the pool."""
    rng = random.Random(seed)
    return rng.sample(COIN_POOL, min(n, len(COIN_POOL)))


def pick_random_window(seed: int = None) -> Tuple[str, str, str]:
    """Pick a random date window and a 30-90 day sub-window within it."""
    rng = random.Random(seed)
    label, win_start, win_end = rng.choice(DATE_WINDOWS)

    start_dt = datetime.strptime(win_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(win_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Clamp end to today
    if end_dt > _TODAY:
        end_dt = _TODAY - timedelta(days=1)

    span_days = (end_dt - start_dt).days
    if span_days < 14:
        return label, win_start, win_end

    # Pick random sub-window of 30-90 days
    duration = rng.randint(30, min(90, span_days))
    max_offset = span_days - duration
    offset = rng.randint(0, max(0, max_offset))
    sub_start = start_dt + timedelta(days=offset)
    sub_end = sub_start + timedelta(days=duration)

    return label, sub_start.strftime("%Y-%m-%d"), sub_end.strftime("%Y-%m-%d")


# ─── Download candle data ────────────────────────────────────────

def download_candles(coins: List[str], start: str, end: str,
                     timeframe: str = "1h") -> Dict[str, List[Candle]]:
    """
    Download real candle data from Hyperliquid for each coin.
    Returns {coin: [Candle, ...]}.
    """
    fetcher = DataFetcher(cache_dir=str(ROOT / "data"))
    result = {}

    for i, coin in enumerate(coins):
        logger.info("[%d/%d] Fetching %s %s candles (%s -> %s)...",
                    i + 1, len(coins), coin, timeframe, start, end)
        try:
            candles = fetcher.fetch_candles(
                coin=coin, timeframe=timeframe,
                start=start, end=end, use_cache=True,
            )
            if candles:
                result[coin] = candles
                logger.info("  Got %d candles for %s (%.1f days)",
                           len(candles), coin,
                           (candles[-1].timestamp_ms - candles[0].timestamp_ms) / 86_400_000)
            else:
                logger.warning("  No candles returned for %s", coin)
        except Exception as e:
            logger.error("  Failed to fetch %s: %s", coin, e)

        # Brief pause between coins to be nice to the API
        if i < len(coins) - 1:
            time.sleep(0.3)

    return result


# ─── Generate fills from real price data ─────────────────────────

def _moving_average(prices: List[float], window: int) -> List[Optional[float]]:
    """Simple moving average."""
    result = [None] * len(prices)
    for i in range(window - 1, len(prices)):
        result[i] = sum(prices[i - window + 1:i + 1]) / window
    return result


def generate_fills_from_candles(
    candle_data: Dict[str, List[Candle]],
    wallet_address: str,
    seed: int = 42,
    trades_per_coin: int = 40,
) -> List[Dict]:
    """
    Generate realistic wallet fills from real candle data using
    multiple strategy signals (MA crossover, breakout, mean-reversion).

    Each trade is an open/close pair derived from actual price levels.
    """
    rng = random.Random(seed)
    all_fills = []

    for coin, candles in candle_data.items():
        if len(candles) < 50:
            logger.warning("Skipping %s — only %d candles (need 50+)", coin, len(candles))
            continue

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        timestamps = [c.timestamp_ms for c in candles]

        # Compute indicators
        ma_fast = _moving_average(closes, 8)
        ma_slow = _moving_average(closes, 21)

        # Find signal points (MA crossovers + breakouts + reversals)
        signals = []
        for i in range(22, len(candles) - 5):
            if ma_fast[i] is None or ma_slow[i] is None:
                continue
            if ma_fast[i - 1] is None or ma_slow[i - 1] is None:
                continue

            # MA crossover signal
            if ma_fast[i - 1] < ma_slow[i - 1] and ma_fast[i] > ma_slow[i]:
                signals.append((i, "long", "ma_cross"))
            elif ma_fast[i - 1] > ma_slow[i - 1] and ma_fast[i] < ma_slow[i]:
                signals.append((i, "short", "ma_cross"))

            # Breakout: price breaks 20-bar high/low
            recent_high = max(highs[i - 20:i])
            recent_low = min(lows[i - 20:i])
            if closes[i] > recent_high * 1.001:
                signals.append((i, "long", "breakout"))
            elif closes[i] < recent_low * 0.999:
                signals.append((i, "short", "breakout"))

            # Mean-reversion: price deviates >2% from 21-MA
            if ma_slow[i] and closes[i] < ma_slow[i] * 0.98:
                signals.append((i, "long", "mean_rev"))
            elif ma_slow[i] and closes[i] > ma_slow[i] * 1.02:
                signals.append((i, "short", "mean_rev"))

        if not signals:
            logger.warning("No signals found for %s, generating random entries", coin)
            # Fall back to random entries
            for _ in range(trades_per_coin):
                idx = rng.randint(22, len(candles) - 10)
                side = rng.choice(["long", "short"])
                signals.append((idx, side, "random"))

        # Sample trades_per_coin signals
        rng.shuffle(signals)
        # Sort by index so we can deduplicate chronologically
        signals.sort(key=lambda s: s[0])
        # Deduplicate by ensuring entries are at least 2 bars apart
        selected = []
        last_idx = -10
        for sig in signals:
            if sig[0] - last_idx >= 2:
                selected.append(sig)
                last_idx = sig[0]
            if len(selected) >= trades_per_coin:
                break

        # Generate open/close fill pairs from selected signals
        for entry_idx, direction, strategy in selected:
            # Hold for 2-24 bars (2-24 hours for 1h candles)
            max_hold = min(24, len(candles) - entry_idx - 1)
            if max_hold < 2:
                continue
            hold_bars = rng.randint(2, max_hold)
            exit_idx = entry_idx + hold_bars

            entry_price = closes[entry_idx]
            exit_price = closes[exit_idx]

            # Use actual high/low during hold period for realistic exits
            period_high = max(highs[entry_idx:exit_idx + 1])
            period_low = min(lows[entry_idx:exit_idx + 1])

            is_long = direction == "long"
            side_open = "buy" if is_long else "sell"
            side_close = "sell" if is_long else "buy"
            dir_open = "Open Long" if is_long else "Open Short"
            dir_close = "Close Long" if is_long else "Close Short"

            # Check if stop loss would have triggered (5% adverse move)
            stop_pct = 0.05
            if is_long and period_low < entry_price * (1 - stop_pct):
                exit_price = entry_price * (1 - stop_pct)
            elif not is_long and period_high > entry_price * (1 + stop_pct):
                exit_price = entry_price * (1 + stop_pct)

            # Check if take profit would have triggered (10% favorable move)
            tp_pct = 0.10
            if is_long and period_high > entry_price * (1 + tp_pct):
                exit_price = entry_price * (1 + tp_pct)
            elif not is_long and period_low < entry_price * (1 - tp_pct):
                exit_price = entry_price * (1 - tp_pct)

            # Position size: $500-$5000 notional
            notional = rng.uniform(500, 5000)
            size = round(notional / entry_price, 6)

            # Slippage/fees (4.5 bps per leg)
            slippage = 0.00045
            if is_long:
                pen_open = entry_price * (1 + slippage)
                pen_close = exit_price * (1 - slippage)
                raw_pnl = (exit_price - entry_price) * size
                pen_pnl = (pen_close - pen_open) * size
            else:
                pen_open = entry_price * (1 - slippage)
                pen_close = exit_price * (1 + slippage)
                raw_pnl = (entry_price - exit_price) * size
                pen_pnl = (pen_open - pen_close) * size

            fee = abs(notional * slippage * 2)

            # OPEN fill
            all_fills.append({
                "wallet_address": wallet_address,
                "coin": coin,
                "side": side_open,
                "original_price": round(entry_price, 6),
                "penalised_price": round(pen_open, 6),
                "size": size,
                "time_ms": timestamps[entry_idx],
                "delayed_time_ms": timestamps[entry_idx] + 2000,
                "closed_pnl": 0.0,
                "penalised_pnl": 0.0,
                "fee": round(fee / 2, 4),
                "is_liquidation": 0,
                "direction": dir_open,
            })

            # CLOSE fill
            all_fills.append({
                "wallet_address": wallet_address,
                "coin": coin,
                "side": side_close,
                "original_price": round(exit_price, 6),
                "penalised_price": round(pen_close, 6),
                "size": size,
                "time_ms": timestamps[exit_idx],
                "delayed_time_ms": timestamps[exit_idx] + 2000,
                "closed_pnl": round(raw_pnl, 4),
                "penalised_pnl": round(pen_pnl, 4),
                "fee": round(fee / 2, 4),
                "is_liquidation": 0,
                "direction": dir_close,
            })

    all_fills.sort(key=lambda f: f["time_ms"])
    return all_fills


# ─── Database seeding ────────────────────────────────────────────

def seed_real_data(fills: List[Dict], coins: List[str],
                   wallet_label: str, window_label: str):
    """Seed the database with real-data-derived fills."""
    db.init_db()
    init_golden_tables()

    wallet_address = f"0xreal_data_{wallet_label}"

    with db.get_connection() as conn:
        cur = conn.cursor()

        # Clear previous real-data test entries
        cur.execute("DELETE FROM wallet_fills WHERE wallet_address LIKE '0xreal_data_%'")
        cur.execute("DELETE FROM golden_wallets WHERE address LIKE '0xreal_data_%'")
        cur.execute("DELETE FROM traders WHERE address LIKE '0xreal_data_%'")

        # Insert trader
        cur.execute("""
            INSERT OR REPLACE INTO traders
            (address, first_seen, last_updated, total_pnl, roi_pct,
             account_value, win_rate, trade_count, active, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (wallet_address, datetime.now(timezone.utc).isoformat(),
              datetime.now(timezone.utc).isoformat(), 0, 0, 10000, 0.5,
              len(fills) // 2, 1,
              json.dumps({"source": "real_data", "window": window_label})))

        # Insert golden wallet
        cur.execute("""
            INSERT OR REPLACE INTO golden_wallets
            (address, bot_score, total_fills, raw_pnl, penalised_pnl,
             max_drawdown_pct, penalised_max_drawdown_pct, sharpe_ratio,
             win_rate, trades_per_day, is_golden, coins_traded, best_coin,
             worst_coin, evaluated_at, connected_to_live)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (wallet_address, 0, len(fills), 0, 0,
              0, 0, 0, 0.5, 5, 1,
              json.dumps(coins), coins[0] if coins else "",
              coins[-1] if coins else "",
              datetime.now(timezone.utc).isoformat(), 0))

        # Insert fills
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

    logger.info("Seeded %d fills for wallet %s (%s)",
                len(fills), wallet_address, window_label)
    return wallet_address


# ─── Run stress test ─────────────────────────────────────────────

def run_backtest_and_stress(fills_raw: List[Dict]):
    """Run backtester and stress test on the real data fills."""
    from src.backtest.backtester import (
        BacktestConfig, BacktestEngine, BacktestFill,
        init_experiments_table, save_experiment,
    )
    from src.backtest.stress_test import (
        StressTestEngine, generate_html_report,
    )

    # ── Backtest ──
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

    bt_fills = []
    for f in fills_raw:
        bt_fills.append(BacktestFill(
            wallet_address=f["wallet_address"],
            coin=f["coin"],
            side=f["side"],
            price=f["penalised_price"],
            original_price=f["original_price"],
            size=f["size"],
            time_ms=f["time_ms"],
            closed_pnl=f.get("penalised_pnl", 0),
            direction=f.get("direction", ""),
            is_liquidation=bool(f.get("is_liquidation", 0)),
        ))
    bt_fills.sort(key=lambda f: f.time_ms)

    engine = BacktestEngine(cfg)
    result = engine.run(bt_fills, experiment_id="real_data_backtest")

    print(f"\n{'='*65}")
    print("  BACKTEST RESULTS (Real Market Data)")
    print(f"{'='*65}")
    print(f"  Fills:             {len(bt_fills)}")
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    print(f"  Total PnL:         ${result.total_pnl:+,.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:     {result.sortino_ratio:.3f}")
    print(f"  Profit Factor:     {result.profit_factor:.3f}")
    print(f"  Calmar Ratio:      {result.calmar_ratio:.3f}")
    print(f"  Avg Hold Time:     {result.avg_hold_hours:.1f}h")

    if result.coin_breakdown:
        print("\n  Coin Breakdown:")
        print(f"    {'Coin':<8}  {'Trades':>7}  {'WR%':>6}  {'PnL':>11}")
        for coin, d in sorted(result.coin_breakdown.items(),
                              key=lambda x: x[1]["pnl"], reverse=True):
            print(f"    {coin:<8}  {d['trades']:>7}  "
                  f"{d['win_rate']:>5.1f}%  ${d['pnl']:>9,.2f}")

    if engine._rejection_reasons:
        print(f"\n  Signal Rejections ({engine._signals_rejected} total):")
        for reason, count in sorted(engine._rejection_reasons.items(),
                                    key=lambda x: x[1], reverse=True)[:8]:
            print(f"    {reason:<30}  {count:>5}")

    init_experiments_table()
    save_experiment(result, notes="real_data_backtest")
    print(f"{'='*65}")

    # ── Stress Test ──
    print(f"\n{'='*65}")
    print("  STRESS TEST (Real Market Data)")
    print(f"{'='*65}")

    stress_engine = StressTestEngine(cfg)
    report = stress_engine.run(fills_raw)

    # Print stress summary
    score = report.composite_stress_score
    grade = "RESILIENT" if score >= 70 else "MODERATE" if score >= 40 else "FRAGILE"

    print(f"\n  Composite Score: {score:.0f}/100 ({grade})")
    print(f"  Baseline: PnL=${report.baseline_pnl:+,.2f}  "
          f"DD={report.baseline_dd:.1f}%  Sharpe={report.baseline_sharpe:.3f}")
    print(f"  Survived: {report.scenarios_survived}/{len(report.scenarios)} scenarios")

    print(f"\n  {'Scenario':<26} {'PnL':>10} {'MaxDD':>8} {'Liq':>5} "
          f"{'Score':>7} {'Verdict':>10}")
    print(f"  {'-'*70}")
    for s in report.scenarios:
        verdict = "SURVIVED" if s.survived else ("BLOWN" if s.blown else "DAMAGED")
        print(f"  {s.scenario_name:<26} ${s.total_pnl:>+8,.0f} "
              f"{s.max_drawdown_pct:>7.1f}% {s.liquidations:>5} "
              f"{s.severity_score:>6.0f}/100 {verdict:>10}")

    print(f"\n  Worst scenario: {report.worst_scenario}")
    print(f"  Runtime: {report.duration_seconds:.1f}s")

    # Save reports
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"real_data_stress_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    html_path = reports_dir / f"real_data_stress_{ts}.html"
    generate_html_report(report, str(html_path))

    print("\n  Reports saved:")
    print(f"    JSON: {json_path}")
    print(f"    HTML: {html_path}")
    print(f"{'='*65}")

    return result, report


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download real Hyperliquid data and run stress tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_real_data_stress_test.py
  python scripts/download_real_data_stress_test.py --coins BTC ETH SOL DOGE ARB
  python scripts/download_real_data_stress_test.py --n-coins 15 --seed 123
  python scripts/download_real_data_stress_test.py --download-only
  python scripts/download_real_data_stress_test.py --no-stress
  python scripts/download_real_data_stress_test.py --start 2024-06-01 --end 2024-09-01
        """,
    )
    parser.add_argument("--coins", nargs="+",
                        help="Specific coins to download (e.g. BTC ETH SOL)")
    parser.add_argument("--n-coins", type=int, default=8,
                        help="Number of random coins to pick (default: 8)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD). Random if omitted")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD). Random if omitted")
    parser.add_argument("--timeframe", default="1h",
                        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="Candle timeframe (default: 1h)")
    parser.add_argument("--trades-per-coin", type=int, default=40,
                        help="Number of trades to generate per coin (default: 40)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: based on current time)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download and cache data, don't run tests")
    parser.add_argument("--no-stress", action="store_true",
                        help="Run backtest only, skip stress scenarios")
    args = parser.parse_args()

    seed = args.seed or int(time.time()) % 100000

    print()
    print("=" * 65)
    print("  Real Market Data Stress Test")
    print("=" * 65)
    print(f"  Seed: {seed}")

    # ── Pick coins ──
    if args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = pick_random_coins(args.n_coins, seed=seed)
    print(f"  Coins ({len(coins)}): {', '.join(coins)}")

    # ── Pick date window ──
    if args.start and args.end:
        window_label = f"custom_{args.start}_to_{args.end}"
        start_date, end_date = args.start, args.end
    else:
        window_label, start_date, end_date = pick_random_window(seed=seed)
    print(f"  Window: {window_label}")
    print(f"  Period: {start_date} -> {end_date}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Trades/coin: {args.trades_per_coin}")
    print("=" * 65)

    # ── Phase 1: Download ──
    print("\nPhase 1: Downloading candle data from Hyperliquid...")
    candle_data = download_candles(coins, start_date, end_date, args.timeframe)

    if not candle_data:
        print("\nFailed to download any candle data. Check your internet connection.")
        return

    total_candles = sum(len(v) for v in candle_data.values())
    print(f"\nDownloaded {total_candles:,} candles across {len(candle_data)} coins")

    # Show data summary
    print(f"\n  {'Coin':<8} {'Candles':>8} {'Start':>12} {'End':>12} {'Days':>6}")
    print(f"  {'-'*50}")
    for coin, candles in sorted(candle_data.items()):
        start_dt = datetime.fromtimestamp(candles[0].timestamp_ms / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(candles[-1].timestamp_ms / 1000, tz=timezone.utc)
        days = (candles[-1].timestamp_ms - candles[0].timestamp_ms) / 86_400_000
        print(f"  {coin:<8} {len(candles):>8} {start_dt.strftime('%Y-%m-%d'):>12} "
              f"{end_dt.strftime('%Y-%m-%d'):>12} {days:>5.0f}")

    if args.download_only:
        print("\nData cached in data/candle_cache.db. Run without --download-only to test.")
        return

    # ── Phase 2: Generate fills ──
    print("\nPhase 2: Generating wallet fills from real price action...")
    wallet_label = f"seed{seed}"
    wallet_address = f"0xreal_data_{wallet_label}"

    fills = generate_fills_from_candles(
        candle_data,
        wallet_address=wallet_address,
        seed=seed,
        trades_per_coin=args.trades_per_coin,
    )

    if not fills:
        print("\nNo fills generated. The candle data may be too sparse.")
        return

    n_trades = len(fills) // 2
    n_coins = len(set(f["coin"] for f in fills))
    print(f"  Generated {len(fills)} fills ({n_trades} trades) across {n_coins} coins")

    # ── Phase 3: Seed database ──
    print("\nPhase 3: Seeding database...")
    actual_coins = list(set(f["coin"] for f in fills))
    seed_real_data(fills, actual_coins, wallet_label, window_label)

    # ── Phase 4: Run tests ──
    if args.no_stress:
        print("\nPhase 4: Running backtest only (stress test skipped)...")
        from src.backtest.backtester import (
            BacktestConfig, BacktestEngine, BacktestFill,
            init_experiments_table, save_experiment,
        )

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

        bt_fills = []
        for f in fills:
            bt_fills.append(BacktestFill(
                wallet_address=f["wallet_address"],
                coin=f["coin"],
                side=f["side"],
                price=f["penalised_price"],
                original_price=f["original_price"],
                size=f["size"],
                time_ms=f["time_ms"],
                closed_pnl=f.get("penalised_pnl", 0),
                direction=f.get("direction", ""),
                is_liquidation=bool(f.get("is_liquidation", 0)),
            ))
        bt_fills.sort(key=lambda f: f.time_ms)

        engine = BacktestEngine(cfg)
        result = engine.run(bt_fills, experiment_id="real_data_backtest")

        print(f"\n{'='*65}")
        print("  BACKTEST RESULTS (Real Market Data)")
        print(f"{'='*65}")
        print(f"  Total Trades:  {result.total_trades}")
        print(f"  Win Rate:      {result.win_rate:.1f}%")
        print(f"  Total PnL:     ${result.total_pnl:+,.2f}")
        print(f"  Max Drawdown:  {result.max_drawdown_pct:.1f}%")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:.3f}")
        print(f"{'='*65}")

        init_experiments_table()
        save_experiment(result, notes="real_data_backtest_no_stress")
    else:
        print("\nPhase 4: Running backtest + stress test suite...")
        run_backtest_and_stress(fills)

    print(f"\nDone! Seed={seed} -- rerun with --seed {seed} to reproduce.")


if __name__ == "__main__":
    main()
