"""
Trade analytics helpers.

Provides lightweight aggregation over closed trade rows so the dashboard,
firewall, and reporting paths can all reason about side/source performance
without duplicating parsing logic.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable, List


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trade_metadata(trade: Dict) -> Dict:
    raw = trade.get("metadata", {})
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _new_bucket() -> Dict:
    return {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "net_pnl": 0.0,
        "gross_pnl": 0.0,
        "fees": 0.0,
        "slippage": 0.0,
        "avg_pnl": 0.0,
        "win_rate": 0.0,
    }


def _finalize_bucket(label: str, bucket: Dict) -> Dict:
    count = int(bucket.get("count", 0) or 0)
    wins = int(bucket.get("wins", 0) or 0)
    losses = int(bucket.get("losses", 0) or 0)
    net_pnl = round(float(bucket.get("net_pnl", 0.0) or 0.0), 4)
    gross_pnl = round(float(bucket.get("gross_pnl", 0.0) or 0.0), 4)
    fees = round(float(bucket.get("fees", 0.0) or 0.0), 4)
    slippage = round(float(bucket.get("slippage", 0.0) or 0.0), 4)
    return {
        "label": label,
        "count": count,
        "wins": wins,
        "losses": losses,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "slippage": slippage,
        "avg_pnl": round(net_pnl / count, 4) if count else 0.0,
        "win_rate": round(wins / count, 4) if count else 0.0,
    }


def compute_trade_analytics(
    trades: Iterable[Dict],
    *,
    source_limit: int = 12,
) -> Dict:
    summary = _new_bucket()
    by_side = defaultdict(_new_bucket)
    by_source = defaultdict(_new_bucket)

    for trade in trades or []:
        pnl = _coerce_float(trade.get("pnl", 0.0))
        side = str(trade.get("side", "") or "unknown").strip().lower() or "unknown"
        meta = _trade_metadata(trade)
        fees = _coerce_float(meta.get("total_fees_paid", 0.0))
        slippage = _coerce_float(meta.get("total_slippage_cost", 0.0))
        gross_pnl = _coerce_float(meta.get("gross_pnl_before_fees", pnl + fees))
        source_key = str(
            meta.get("source_key")
            or meta.get("source")
            or trade.get("source")
            or trade.get("strategy_type")
            or "unknown"
        ).strip().lower() or "unknown"

        for bucket in (summary, by_side[side], by_source[source_key]):
            bucket["count"] += 1
            bucket["net_pnl"] += pnl
            bucket["gross_pnl"] += gross_pnl
            bucket["fees"] += fees
            bucket["slippage"] += slippage
            if pnl > 0:
                bucket["wins"] += 1
            elif pnl < 0:
                bucket["losses"] += 1

    side_rows: List[Dict] = []
    for side in ("long", "short", "unknown"):
        if by_side.get(side, {}).get("count"):
            side_rows.append(_finalize_bucket(side, by_side[side]))

    source_rows = [
        _finalize_bucket(source_key, bucket)
        for source_key, bucket in by_source.items()
        if bucket.get("count")
    ]
    source_rows.sort(key=lambda row: (row["net_pnl"], row["win_rate"], row["count"]), reverse=True)

    short_row = next((row for row in side_rows if row["label"] == "short"), None)
    long_row = next((row for row in side_rows if row["label"] == "long"), None)

    return {
        "summary": _finalize_bucket("all", summary),
        "by_side": side_rows,
        "by_source": source_rows[:source_limit],
        "short_vs_long": {
            "short_trades": int(short_row["count"]) if short_row else 0,
            "short_net_pnl": float(short_row["net_pnl"]) if short_row else 0.0,
            "short_win_rate": float(short_row["win_rate"]) if short_row else 0.0,
            "long_trades": int(long_row["count"]) if long_row else 0,
            "long_net_pnl": float(long_row["net_pnl"]) if long_row else 0.0,
            "long_win_rate": float(long_row["win_rate"]) if long_row else 0.0,
        },
    }


def evaluate_short_side_policy(
    trades: Iterable[Dict],
    *,
    min_trades: int,
    degrade_win_rate: float,
    block_win_rate: float,
    block_net_pnl: float,
) -> Dict:
    analytics = compute_trade_analytics(trades, source_limit=8)
    short_row = next((row for row in analytics["by_side"] if row["label"] == "short"), None)
    if not short_row:
        return {
            "status": "insufficient",
            "reason": "No closed short trades yet",
            "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
        }

    count = int(short_row["count"])
    win_rate = float(short_row["win_rate"])
    net_pnl = float(short_row["net_pnl"])
    if count < int(min_trades):
        return {
            "status": "insufficient",
            "reason": f"Need {min_trades} closed shorts before policy activates",
            "metrics": {"count": count, "win_rate": win_rate, "net_pnl": net_pnl},
        }
    if win_rate < float(block_win_rate) and net_pnl <= float(block_net_pnl):
        return {
            "status": "blocked",
            "reason": (
                f"Recent shorts are underperforming ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
            ),
            "metrics": {"count": count, "win_rate": win_rate, "net_pnl": net_pnl},
        }
    if win_rate < float(degrade_win_rate) and net_pnl < 0:
        return {
            "status": "degraded",
            "reason": (
                f"Recent shorts need caution ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
            ),
            "metrics": {"count": count, "win_rate": win_rate, "net_pnl": net_pnl},
        }
    return {
        "status": "healthy",
        "reason": (
            f"Short side healthy enough ({count} trades, "
            f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
        ),
        "metrics": {"count": count, "win_rate": win_rate, "net_pnl": net_pnl},
    }
