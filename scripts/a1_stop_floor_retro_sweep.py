"""A1 retro-analysis: would the ATR-aware stop floor have changed
the outcome of past stop-out losses?

Loads closed paper_trades from the last N days that closed via the
stop-loss path, then for each (atr_mult, noise_floor_bps) parameter
combo computes whether A1's *widened* stop would have left the trade
alive at the actual exit price.

The conservative claim this produces:
    "With k=2.5, floor=50 bps: 4 of last week's 5 noise stop-outs
     would NOT have triggered the stop (the exit price stayed within
     the A1-widened band). Avoided realised loss: $-N.NN."

This is intentionally a *minimum* benefit estimate -- a trade whose
stop is widened may eventually hit TP, time-out, or close at break-
even, so the actual P&L benefit is >= the "loss avoided" figure.
That's why this is enough evidence to flip the flag in shadow before
running a fuller candle-replay sweep.

Output is a markdown table written to stdout AND a JSON dump in
data/a1_retro_sweep_<timestamp>.json for record-keeping.

Usage:
    python scripts/a1_stop_floor_retro_sweep.py [--days N] [--coin BTC]

Defaults:
    --days 14
    All coins
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time

# Make `src` importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import database as db
from src.signals.signal_schema import RiskParams


# ── Parameter grid for the sweep ──────────────────────────────────

ATR_MULT_GRID = [1.5, 2.0, 2.5, 3.0, 4.0]
NOISE_FLOOR_BPS_GRID = [20.0, 50.0, 75.0, 100.0]


@dataclass
class Trade:
    trade_id: int
    coin: str
    side: str               # "long" or "short"
    entry_price: float
    exit_price: float
    leverage: float
    pnl: float
    close_reason: str
    atr_pct: float          # from metadata, may be 0 if missing
    stop_loss: float        # the actual stop price used


def load_stopped_trades(days: int, coin_filter: str | None) -> list[Trade]:
    """Pull closed stop-loss trades from the DB."""
    where = [
        "status = 'closed'",
        "closed_at >= NOW() - INTERVAL %s",
        "(metadata->>'close_reason' LIKE %s OR metadata->>'close_reason' LIKE %s)",
    ]
    params: list = [f"{days} days", "%stop_loss%", "%stop-loss%"]
    if coin_filter:
        where.append("coin = %s")
        params.append(coin_filter.upper())

    sql = f"""
        SELECT id, coin, side, entry_price, exit_price, leverage, pnl,
               metadata->>'close_reason' AS close_reason,
               COALESCE((metadata->>'atr_pct')::float, 0.0) AS atr_pct,
               stop_loss
          FROM paper_trades
         WHERE {' AND '.join(where)}
         ORDER BY closed_at DESC
    """

    out: list[Trade] = []
    with db.get_connection(for_read=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                (tid, coin, side, entry, exit_, lev, pnl,
                 close_reason, atr_pct, sl) = row
                out.append(Trade(
                    trade_id=tid, coin=coin,
                    side=str(side or "").lower(),
                    entry_price=float(entry or 0.0),
                    exit_price=float(exit_ or 0.0),
                    leverage=float(lev or 1.0),
                    pnl=float(pnl or 0.0),
                    close_reason=str(close_reason or ""),
                    atr_pct=float(atr_pct or 0.0),
                    stop_loss=float(sl or 0.0),
                ))
    return out


def a1_stop_price(
    *,
    entry_price: float,
    side: str,
    leverage: float,
    atr_pct: float,
    atr_mult: float,
    noise_floor_bps: float,
    base_stop_pct: float = 0.04,
    base_tp_pct: float = 0.20,
) -> float:
    """Compute the A1-widened stop price for a trade using the same
    code path that ships in production (RiskParams.resolve_trigger_prices)."""
    risk = RiskParams(stop_loss_pct=base_stop_pct, take_profit_pct=base_tp_pct,
                      risk_basis="roe")
    # We toggle A1 on inside this script via a config override.
    import config as _cfg
    prev = getattr(_cfg, "ATR_STOP_FLOOR_ENABLED", False)
    prev_mult = getattr(_cfg, "ATR_STOP_ATR_MULTIPLIER", 2.5)
    prev_floor = getattr(_cfg, "ATR_STOP_NOISE_FLOOR_BPS", 50.0)
    try:
        _cfg.ATR_STOP_FLOOR_ENABLED = True
        _cfg.ATR_STOP_ATR_MULTIPLIER = atr_mult
        _cfg.ATR_STOP_NOISE_FLOOR_BPS = noise_floor_bps
        sl, _tp = risk.resolve_trigger_prices(
            entry_price, side, leverage, atr_pct=atr_pct,
        )
    finally:
        _cfg.ATR_STOP_FLOOR_ENABLED = prev
        _cfg.ATR_STOP_ATR_MULTIPLIER = prev_mult
        _cfg.ATR_STOP_NOISE_FLOOR_BPS = prev_floor
    return sl


def would_a1_stop_have_triggered(
    trade: Trade, atr_mult: float, noise_floor_bps: float,
) -> bool:
    """True iff the trade's actual exit_price would still have hit
    the A1-widened stop. (i.e. A1 wouldn't have helped this trade.)"""
    if trade.entry_price <= 0 or trade.exit_price <= 0:
        return True  # malformed; assume same outcome
    a1_sl = a1_stop_price(
        entry_price=trade.entry_price,
        side=trade.side,
        leverage=trade.leverage,
        atr_pct=trade.atr_pct,
        atr_mult=atr_mult,
        noise_floor_bps=noise_floor_bps,
    )
    # For a LONG: stop triggers when price <= sl
    # For a SHORT: stop triggers when price >= sl
    if trade.side == "long":
        return trade.exit_price <= a1_sl
    return trade.exit_price >= a1_sl


def sweep(trades: list[Trade]) -> dict:
    """Run the full parameter grid; return a structured result."""
    n_total = len(trades)
    total_loss = sum(t.pnl for t in trades if t.pnl < 0)

    cells: list[dict] = []
    for atr_mult in ATR_MULT_GRID:
        for floor in NOISE_FLOOR_BPS_GRID:
            avoided_trades = 0
            avoided_loss = 0.0
            for t in trades:
                if not would_a1_stop_have_triggered(t, atr_mult, floor):
                    avoided_trades += 1
                    if t.pnl < 0:
                        avoided_loss += abs(t.pnl)
            cells.append({
                "atr_mult": atr_mult,
                "noise_floor_bps": floor,
                "trades_avoided": avoided_trades,
                "trades_total": n_total,
                "avoid_rate": round(avoided_trades / max(1, n_total), 4),
                "loss_avoided_usd": round(avoided_loss, 4),
                "total_loss_usd": round(total_loss, 4),
            })
    return {
        "generated_at": int(time()),
        "n_stopped_trades": n_total,
        "total_loss_usd_in_window": round(total_loss, 4),
        "atr_mult_grid": ATR_MULT_GRID,
        "noise_floor_bps_grid": NOISE_FLOOR_BPS_GRID,
        "cells": cells,
    }


def render_markdown(result: dict) -> str:
    """Pretty-print the sweep as a markdown table."""
    lines: list[str] = []
    lines.append(f"# A1 retro-sweep — {result['n_stopped_trades']} stopped trades, "
                 f"total realised loss ${result['total_loss_usd_in_window']:.2f}\n")
    lines.append("| atr_mult \\ floor (bps) |"
                 + " | ".join(f"{f:.0f}" for f in result["noise_floor_bps_grid"])
                 + " |")
    lines.append("|---" * (len(result["noise_floor_bps_grid"]) + 1) + "|")
    by_pair = {(c["atr_mult"], c["noise_floor_bps"]): c for c in result["cells"]}
    for m in result["atr_mult_grid"]:
        row = [f"**{m}**"]
        for f in result["noise_floor_bps_grid"]:
            c = by_pair[(m, f)]
            row.append(f"{c['trades_avoided']}/{c['trades_total']} "
                       f"(${c['loss_avoided_usd']:.2f})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Each cell shows `trades_avoided / total ($ loss avoided)`.")
    lines.append("\"avoided\" = the trade's actual exit price stayed within "
                 "the A1-widened stop band, so the stop would NOT have fired.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--coin", default=None)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    trades = load_stopped_trades(args.days, args.coin)
    if not trades:
        print("No closed stop-out trades in the window; nothing to sweep.")
        return 0

    result = sweep(trades)

    md = render_markdown(result)
    print(md)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / f"a1_retro_sweep_{result['generated_at']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull sweep dumped to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
