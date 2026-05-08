"""
Bot-vs-BTC Buy-and-Hold Backtest
=================================
Reads a Hyperliquid trade-history CSV export and compares the bot's realized
PnL against a naive "buy BTC at the start of the period, hold to the end"
strategy at the same average notional.

Usage:
    python scripts/backtest_vs_btc.py path/to/trade_history.csv [bot_capital]

    bot_capital (optional): account size in USD used to scale buy-and-hold
                            so it's apples-to-apples with bot trade size.
                            Defaults to 1000 USD.

Output:
    Markdown-style comparison table with realized PnL, return %, Sharpe,
    max drawdown, win rate, and benchmark return.

Notes:
    - Uses the first and last BTC trade in the CSV to bracket the period.
    - "Bot PnL" is the sum of closedPnl on close-side trades (no fees double-
      counted; the CSV's closedPnl is post-fees).
    - "Buy-and-hold" PnL = (exit_btc_price - entry_btc_price) / entry *
      buy_and_hold_notional, where buy_and_hold_notional defaults to the
      bot's average open notional.
    - Daily PnL series for Sharpe/MDD comes from the bot's close events.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev


def parse_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("ntl", "fee", "closedPnl", "sz", "px"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = 0.0
            rows.append(r)
    return rows


def parse_dt(raw: str) -> datetime:
    # Format: "05/04/2026 - 14:32:44"  (DD/MM/YYYY)
    return datetime.strptime(raw.strip(), "%d/%m/%Y - %H:%M:%S")


def compute_sharpe(daily_pnls: list[float]) -> float:
    if len(daily_pnls) < 2:
        return 0.0
    mu = mean(daily_pnls)
    sd = pstdev(daily_pnls)
    if sd == 0:
        return 0.0
    # Annualized Sharpe (sqrt(365) for daily PnL series).
    return (mu / sd) * (365 ** 0.5)


def max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = min(dd, v - peak)
    return dd


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: backtest_vs_btc.py <trade_history.csv> [bot_capital]")
        return 2
    csv_path = Path(argv[1])
    bot_capital = float(argv[2]) if len(argv) >= 3 else 1000.0

    rows = parse_csv(csv_path)
    if not rows:
        print("No rows in CSV")
        return 1

    btc_rows = [r for r in rows if str(r.get("coin", "")).upper() == "BTC"]
    if not btc_rows:
        print("No BTC rows — cannot benchmark against BTC.")
        return 1

    # Period brackets
    first_btc = btc_rows[0]
    last_btc = btc_rows[-1]
    entry_px = float(first_btc["px"])
    exit_px = float(last_btc["px"])
    start_dt = parse_dt(first_btc["time"])
    end_dt = parse_dt(last_btc["time"])
    days = max(1, (end_dt - start_dt).days)

    # Closes only (Open rows just record the fee charge as -closedPnl).
    closes = [
        r for r in rows
        if r["dir"].startswith("Close") or ">" in r["dir"]
    ]
    opens = [r for r in rows if r["dir"].startswith("Open")]
    realized = sum(r["closedPnl"] for r in closes)
    fees = sum(r["fee"] for r in rows)
    wins = [r for r in closes if r["closedPnl"] > 0]
    losses = [r for r in closes if r["closedPnl"] <= 0]
    wr = (len(wins) / len(closes)) if closes else 0.0
    avg_open_ntl = (
        sum(r["ntl"] for r in opens) / len(opens) if opens else 0.0
    )

    # Daily PnL series for Sharpe / MDD.
    by_day: dict[str, float] = defaultdict(float)
    for r in closes:
        d = parse_dt(r["time"]).date().isoformat()
        by_day[d] += r["closedPnl"]
    daily_pnls = [by_day[d] for d in sorted(by_day.keys())]
    equity_curve: list[float] = []
    cum = 0.0
    for v in daily_pnls:
        cum += v
        equity_curve.append(cum)
    sharpe = compute_sharpe(daily_pnls)
    mdd = max_drawdown(equity_curve)

    # Buy-and-hold benchmark on same notional as the bot's average open.
    buy_and_hold_pct = (exit_px - entry_px) / entry_px
    bnh_pnl_at_avg_ntl = avg_open_ntl * buy_and_hold_pct
    bnh_pnl_at_capital = bot_capital * buy_and_hold_pct

    # Outliers (best/worst close)
    sorted_closes = sorted(closes, key=lambda x: x["closedPnl"])
    best = sorted_closes[-1] if sorted_closes else None
    worst = sorted_closes[0] if sorted_closes else None

    # Top winners/losers excluded view
    pnl_excl_best = realized - (best["closedPnl"] if best else 0.0)

    # Report
    print("=" * 70)
    print(" BOT vs BTC BUY-AND-HOLD")
    print("=" * 70)
    print(f" Period:           {start_dt.date()} -> {end_dt.date()} ({days} days)")
    print(f" CSV:              {csv_path.name}")
    print(f" Trade rows:       {len(rows)}  ({len(opens)} opens / {len(closes)} closes)")
    print(f" Avg open notional: ${avg_open_ntl:.2f}")
    print()
    print(f" === BOT ===")
    print(f"  Realized PnL:    ${realized:+.3f}")
    print(f"  PnL excl. best:  ${pnl_excl_best:+.3f}  "
          f"(best was {best['coin']} {best['dir']} ${best['closedPnl']:+.2f})")
    print(f"  Win rate:        {wr * 100:.1f}%  ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Total fees:      ${fees:.2f}")
    print(f"  Sharpe (annu):   {sharpe:.2f}")
    print(f"  Max drawdown:    ${mdd:+.3f}")
    print()
    print(f" === BTC BUY-AND-HOLD (same period) ===")
    print(f"  Entry / exit:    ${entry_px:,.2f} -> ${exit_px:,.2f}")
    print(f"  Return:          {buy_and_hold_pct * 100:+.2f}%")
    print(f"  PnL on avg ntl ${avg_open_ntl:.2f}: ${bnh_pnl_at_avg_ntl:+.2f}")
    print(f"  PnL on ${bot_capital:.0f} capital:  ${bnh_pnl_at_capital:+.2f}")
    print()
    print(f" === VERDICT ===")
    if realized > bnh_pnl_at_avg_ntl:
        margin = realized - bnh_pnl_at_avg_ntl
        print(f"  Bot beat buy-and-hold by ${margin:+.3f} at matched notional.")
    else:
        margin = bnh_pnl_at_avg_ntl - realized
        print(f"  BTC buy-and-hold beat the bot by ${margin:+.3f} at matched notional.")
    if best and (realized - best["closedPnl"]) <= 0:
        print(f"  WARNING: bot PnL is negative without its single best trade "
              f"({best['coin']} {best['closedPnl']:+.2f}). The 'edge' is one trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
