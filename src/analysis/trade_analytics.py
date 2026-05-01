"""
Trade analytics helpers.

Provides lightweight aggregation over closed trade rows so the dashboard,
firewall, and reporting paths can all reason about side/source performance
without duplicating parsing logic.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping


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


def _normalize_source_label(value) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if raw.startswith("copy_trade"):
        return "copy_trade"
    if raw.startswith("options_flow"):
        return "options_flow"
    if raw.startswith("strategy"):
        return "strategy"
    if ":" in raw:
        return raw.split(":", 1)[0] or "unknown"
    return raw


def _trade_source_label(trade: Dict) -> str:
    meta = _trade_metadata(trade)
    return _normalize_source_label(
        meta.get("source_key")
        or meta.get("source")
        or trade.get("source")
        or trade.get("strategy_type")
        or "unknown"
    )


def _trade_source_key(trade: Dict) -> str:
    """Return the exact source key, preserving trader/strategy identity."""
    meta = _trade_metadata(trade)
    raw = (
        meta.get("source_key")
        or meta.get("source")
        or trade.get("source")
        or trade.get("strategy_type")
        or "unknown"
    )
    key = str(raw or "unknown").strip().lower() or "unknown"
    if key == "copy_trade":
        trader = str(
            meta.get("source_trader")
            or trade.get("trader_address")
            or trade.get("source_trader")
            or ""
        ).strip().lower()
        if trader:
            return f"copy_trade:{trader}"
    if key == "strategy":
        strategy_type = str(
            meta.get("strategy_type")
            or trade.get("strategy_type")
            or ""
        ).strip().lower()
        if strategy_type:
            return f"strategy:{strategy_type}"
    return key


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
        "path_count": 0,
        "mfe_r_sum": 0.0,
        "mae_r_sum": 0.0,
        "exit_r_sum": 0.0,
        "path_capture_sum": 0.0,
    }


def _finalize_bucket(label: str, bucket: Dict) -> Dict:
    count = int(bucket.get("count", 0) or 0)
    wins = int(bucket.get("wins", 0) or 0)
    losses = int(bucket.get("losses", 0) or 0)
    net_pnl = round(float(bucket.get("net_pnl", 0.0) or 0.0), 4)
    gross_pnl = round(float(bucket.get("gross_pnl", 0.0) or 0.0), 4)
    fees = round(float(bucket.get("fees", 0.0) or 0.0), 4)
    slippage = round(float(bucket.get("slippage", 0.0) or 0.0), 4)
    path_count = int(bucket.get("path_count", 0) or 0)
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
        # ★ M31 FIX: previously divided wins by `count` (which includes
        # zero-PnL trades).  Every other win_rate in the codebase
        # (strategy_scorer, agent_scoring, replay_backtester, golden_wallet)
        # uses ``wins / (wins + losses)`` -- excluding break-even fills.
        # Using count made buckets with many zero-PnL closes look worse
        # than they were.  Switch to the project-wide convention.
        "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else 0.0,
        "path_count": path_count,
        "avg_mfe_r": round(float(bucket.get("mfe_r_sum", 0.0) or 0.0) / path_count, 4)
        if path_count else 0.0,
        "avg_mae_r": round(float(bucket.get("mae_r_sum", 0.0) or 0.0) / path_count, 4)
        if path_count else 0.0,
        "avg_exit_r": round(float(bucket.get("exit_r_sum", 0.0) or 0.0) / path_count, 4)
        if path_count else 0.0,
        "avg_path_capture_ratio": round(
            float(bucket.get("path_capture_sum", 0.0) or 0.0) / path_count,
            4,
        ) if path_count else 0.0,
    }


def compute_trade_analytics(
    trades: Iterable[Dict],
    *,
    source_limit: int = 12,
    coin_side_limit: int = 12,
) -> Dict:
    summary = _new_bucket()
    by_side = defaultdict(_new_bucket)
    by_source = defaultdict(_new_bucket)
    by_exact_source = defaultdict(_new_bucket)
    by_source_side = defaultdict(_new_bucket)
    by_exact_source_side = defaultdict(_new_bucket)
    by_coin_side = defaultdict(_new_bucket)

    for trade in trades or []:
        pnl = _coerce_float(trade.get("pnl", 0.0))
        side = str(trade.get("side", "") or "unknown").strip().lower() or "unknown"
        coin = str(trade.get("coin", "") or "unknown").strip().upper() or "UNKNOWN"
        meta = _trade_metadata(trade)
        fees = _coerce_float(meta.get("total_fees_paid", 0.0))
        slippage = _coerce_float(meta.get("total_slippage_cost", 0.0))
        gross_pnl = _coerce_float(meta.get("gross_pnl_before_fees", pnl + fees))
        has_path_metrics = any(
            key in meta
            for key in (
                "max_r_multiple",
                "min_r_multiple",
                "exit_r_multiple",
                "path_capture_ratio",
            )
        )
        max_r = _coerce_float(meta.get("max_r_multiple", meta.get("mfe_r_multiple", 0.0)))
        min_r = _coerce_float(meta.get("min_r_multiple", meta.get("mae_r_multiple", 0.0)))
        exit_r = _coerce_float(meta.get("exit_r_multiple", 0.0))
        capture_ratio = _coerce_float(meta.get("path_capture_ratio", 0.0))
        source_key = _trade_source_label(trade)
        exact_source_key = _trade_source_key(trade)

        for bucket in (
            summary,
            by_side[side],
            by_source[source_key],
            by_exact_source[exact_source_key],
            by_source_side[(source_key, side)],
            by_exact_source_side[(exact_source_key, side)],
            by_coin_side[(coin, side)],
        ):
            bucket["count"] += 1
            bucket["net_pnl"] += pnl
            bucket["gross_pnl"] += gross_pnl
            bucket["fees"] += fees
            bucket["slippage"] += slippage
            if pnl > 0:
                bucket["wins"] += 1
            elif pnl < 0:
                bucket["losses"] += 1
            if has_path_metrics:
                bucket["path_count"] += 1
                bucket["mfe_r_sum"] += max_r
                bucket["mae_r_sum"] += min_r
                bucket["exit_r_sum"] += exit_r
                bucket["path_capture_sum"] += capture_ratio

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
    exact_source_rows = [
        _finalize_bucket(source_key, bucket)
        for source_key, bucket in by_exact_source.items()
        if bucket.get("count")
    ]
    exact_source_rows.sort(
        key=lambda row: (row["net_pnl"], row["win_rate"], row["count"]), reverse=True
    )
    source_side_rows = []
    for (source_key, side), bucket in by_source_side.items():
        if not bucket.get("count"):
            continue
        row = _finalize_bucket(f"{source_key} {side}", bucket)
        row["source"] = source_key
        row["side"] = side
        source_side_rows.append(row)
    source_side_rows.sort(key=lambda row: (row["net_pnl"], -row["count"], row["label"]))
    exact_source_side_rows = []
    for (source_key, side), bucket in by_exact_source_side.items():
        if not bucket.get("count"):
            continue
        row = _finalize_bucket(f"{source_key} {side}", bucket)
        row["source"] = source_key
        row["side"] = side
        exact_source_side_rows.append(row)
    exact_source_side_rows.sort(
        key=lambda row: (row["net_pnl"], -row["count"], row["label"])
    )
    coin_side_rows = []
    for (coin, side), bucket in by_coin_side.items():
        if not bucket.get("count"):
            continue
        row = _finalize_bucket(f"{coin} {side}", bucket)
        row["coin"] = coin
        row["side"] = side
        coin_side_rows.append(row)
    coin_side_rows.sort(key=lambda row: (row["net_pnl"], -row["count"], row["label"]))

    short_row = next((row for row in side_rows if row["label"] == "short"), None)
    long_row = next((row for row in side_rows if row["label"] == "long"), None)

    return {
        "summary": _finalize_bucket("all", summary),
        "by_side": side_rows,
        "by_source": source_rows[:source_limit],
        "by_exact_source": exact_source_rows[:source_limit],
        "by_source_side": source_side_rows[:source_limit],
        "by_exact_source_side": exact_source_side_rows[:source_limit],
        "by_coin_side": coin_side_rows[:coin_side_limit],
        "short_vs_long": {
            "short_trades": int(short_row["count"]) if short_row else 0,
            "short_net_pnl": float(short_row["net_pnl"]) if short_row else 0.0,
            "short_win_rate": float(short_row["win_rate"]) if short_row else 0.0,
            "long_trades": int(long_row["count"]) if long_row else 0,
            "long_net_pnl": float(long_row["net_pnl"]) if long_row else 0.0,
            "long_win_rate": float(long_row["win_rate"]) if long_row else 0.0,
        },
    }


def evaluate_side_source_policy(
    trades: Iterable[Dict],
    *,
    side: str,
    source_key: str | None = None,
    coin: str | None = None,
    min_trades: int,
    degrade_win_rate: float,
    block_win_rate: float,
    block_net_pnl: float,
    exact_source: bool = True,
) -> Dict:
    """Evaluate a performance policy for a side, optionally scoped by source/coin."""
    normalized_side = str(side or "").strip().lower()
    normalized_coin = str(coin or "").strip().upper()
    normalized_source = str(source_key or "").strip().lower()
    filtered: List[Dict] = []

    for trade in trades or []:
        trade_side = str(trade.get("side", "") or "").strip().lower()
        if normalized_side and trade_side != normalized_side:
            continue
        if normalized_coin:
            trade_coin = str(trade.get("coin", "") or "").strip().upper()
            if trade_coin != normalized_coin:
                continue
        if normalized_source:
            trade_source = _trade_source_key(trade) if exact_source else _trade_source_label(trade)
            if trade_source != normalized_source:
                continue
        filtered.append(trade)

    analytics = compute_trade_analytics(filtered, source_limit=8)
    summary = analytics["summary"]
    count = int(summary.get("count", 0) or 0)
    win_rate = float(summary.get("win_rate", 0.0) or 0.0)
    net_pnl = float(summary.get("net_pnl", 0.0) or 0.0)
    metrics = {"count": count, "win_rate": win_rate, "net_pnl": net_pnl}

    scope_parts = []
    if normalized_coin:
        scope_parts.append(normalized_coin)
    if normalized_source:
        scope_parts.append(normalized_source)
    scope_parts.append(normalized_side or "side")
    scope = " ".join(scope_parts)

    if count < int(min_trades):
        return {
            "status": "insufficient",
            "reason": f"Need {min_trades} closed {scope} trades before policy activates",
            "metrics": metrics,
            "source": normalized_source or "all",
            "side": normalized_side,
            "coin": normalized_coin,
            "scope": scope,
        }
    if win_rate < float(block_win_rate) and net_pnl <= float(block_net_pnl):
        return {
            "status": "blocked",
            "reason": (
                f"Recent {scope} trades are underperforming ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
            ),
            "metrics": metrics,
            "source": normalized_source or "all",
            "side": normalized_side,
            "coin": normalized_coin,
            "scope": scope,
        }
    if win_rate < float(degrade_win_rate) and net_pnl < 0:
        return {
            "status": "degraded",
            "reason": (
                f"Recent {scope} trades need caution ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
            ),
            "metrics": metrics,
            "source": normalized_source or "all",
            "side": normalized_side,
            "coin": normalized_coin,
            "scope": scope,
        }
    return {
        "status": "healthy",
        "reason": (
            f"{scope} healthy enough ({count} trades, "
            f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
        ),
        "metrics": metrics,
        "source": normalized_source or "all",
        "side": normalized_side,
        "coin": normalized_coin,
        "scope": scope,
    }


def evaluate_source_policy(
    trades: Iterable[Dict],
    *,
    source_label: str,
    min_trades: int,
    degrade_win_rate: float,
    block_win_rate: float,
    block_net_pnl: float,
) -> Dict:
    normalized = _normalize_source_label(source_label)
    source_trades = [
        trade for trade in (trades or [])
        if _trade_source_label(trade) == normalized
    ]
    analytics = compute_trade_analytics(source_trades, source_limit=8)
    summary = analytics["summary"]
    count = int(summary.get("count", 0) or 0)
    win_rate = float(summary.get("win_rate", 0.0) or 0.0)
    net_pnl = float(summary.get("net_pnl", 0.0) or 0.0)
    metrics = {"count": count, "win_rate": win_rate, "net_pnl": net_pnl}

    if count < int(min_trades):
        return {
            "status": "insufficient",
            "reason": f"Need {min_trades} closed {normalized} trades before policy activates",
            "metrics": metrics,
            "source": normalized,
        }

    # ★ H28 FIX: previously a low win-rate run with negative net PnL
    # automatically tripped the "blocked" / "degraded" gates even when the
    # sample size was small enough that the underperformance was not
    # statistically distinguishable from noise (mirror of the H16 fix in
    # strategy_scorer).  Reuse the same exact two-sided binomial p-value
    # against a 50% null so a 4/12 streak isn't treated identically to a
    # 25/100 sustained underperformance.
    pvalue = None
    try:
        wins = int(round(win_rate * count))
        # Lazy import keeps this module importable in environments that
        # don't have scipy installed; the helper has its own fallback.
        from src.analysis.strategy_scorer import _binomial_pvalue_two_sided
        pvalue = _binomial_pvalue_two_sided(wins, count, 0.5)
    except Exception:
        pvalue = None
    metrics["win_rate_pvalue"] = round(pvalue, 4) if pvalue is not None else None
    significant_underperformance = (
        pvalue is None or pvalue < 0.20
    )

    if (
        win_rate < float(block_win_rate)
        and net_pnl <= float(block_net_pnl)
        and significant_underperformance
    ):
        return {
            "status": "blocked",
            "reason": (
                f"Recent {normalized} trades are underperforming ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f}, "
                f"p={pvalue:.3f})" if pvalue is not None else
                f"Recent {normalized} trades are underperforming ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
            ),
            "metrics": metrics,
            "source": normalized,
        }
    if (
        win_rate < float(degrade_win_rate)
        and net_pnl < 0
        and significant_underperformance
    ):
        return {
            "status": "degraded",
            "reason": (
                f"Recent {normalized} trades need caution ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f}, "
                f"p={pvalue:.3f})" if pvalue is not None else
                f"Recent {normalized} trades need caution ({count} trades, "
                f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
            ),
            "metrics": metrics,
            "source": normalized,
        }
    return {
        "status": "healthy",
        "reason": (
            f"{normalized} healthy enough ({count} trades, "
            f"win rate {win_rate:.0%}, net {net_pnl:.2f})"
        ),
        "metrics": metrics,
        "source": normalized,
    }


def compute_live_paper_drift(
    *,
    closed_trades: Iterable[Dict],
    open_trades: Iterable[Dict],
    audit_rows: Iterable[Mapping],
    live_source_orders_today: Mapping[str, object] | None = None,
    source_limit: int = 8,
) -> Dict:
    by_source = defaultdict(
        lambda: {
            "paper_open": 0,
            "paper_closed": 0,
            "paper_net_pnl": 0.0,
            "audit_approved": 0,
            "audit_rejected": 0,
            "live_entries_today": 0,
            "reject_reasons": Counter(),
        }
    )
    by_side = defaultdict(
        lambda: {
            "paper_open": 0,
            "paper_closed": 0,
            "paper_net_pnl": 0.0,
            "audit_approved": 0,
            "audit_rejected": 0,
        }
    )
    summary = {
        "paper_open_positions": 0,
        "paper_closed_trades": 0,
        "paper_net_pnl": 0.0,
        "audit_approved": 0,
        "audit_rejected": 0,
        "live_entries_today": 0,
        "live_sources_active": 0,
        "paper_sources_realized": 0,
    }

    for trade in open_trades or []:
        source = _trade_source_label(trade)
        side = str(trade.get("side", "") or "unknown").strip().lower() or "unknown"
        by_source[source]["paper_open"] += 1
        by_side[side]["paper_open"] += 1
        summary["paper_open_positions"] += 1

    for trade in closed_trades or []:
        source = _trade_source_label(trade)
        side = str(trade.get("side", "") or "unknown").strip().lower() or "unknown"
        pnl = _coerce_float(trade.get("pnl", 0.0))
        by_source[source]["paper_closed"] += 1
        by_source[source]["paper_net_pnl"] += pnl
        by_side[side]["paper_closed"] += 1
        by_side[side]["paper_net_pnl"] += pnl
        summary["paper_closed_trades"] += 1
        summary["paper_net_pnl"] += pnl

    for row in audit_rows or []:
        action = str(row.get("action", "") or "").strip().lower()
        source = _normalize_source_label(row.get("source"))
        side = str(row.get("side", "") or "unknown").strip().lower() or "unknown"
        if action == "signal_approved":
            by_source[source]["audit_approved"] += 1
            by_side[side]["audit_approved"] += 1
            summary["audit_approved"] += 1
        elif action == "signal_rejected":
            by_source[source]["audit_rejected"] += 1
            by_side[side]["audit_rejected"] += 1
            summary["audit_rejected"] += 1
            details = row.get("details") or {}
            if isinstance(details, str):
                try:
                    details = json.loads(details or "{}")
                except Exception:
                    details = {}
            reason = str((details or {}).get("reason", "") or "").strip()
            if reason:
                by_source[source]["reject_reasons"][reason] += 1

    live_source_orders_today = dict(live_source_orders_today or {})
    for source_key, value in live_source_orders_today.items():
        source = _normalize_source_label(source_key)
        count = int(_coerce_float(value, 0.0))
        by_source[source]["live_entries_today"] += count
        summary["live_entries_today"] += count

    source_rows = []
    for label, bucket in by_source.items():
        if not any(
            bucket[key]
            for key in ("paper_open", "paper_closed", "audit_approved", "audit_rejected", "live_entries_today")
        ):
            continue
        top_reject_reason = ""
        if bucket["reject_reasons"]:
            top_reject_reason = bucket["reject_reasons"].most_common(1)[0][0]
        source_rows.append(
            {
                "label": label,
                "paper_open": int(bucket["paper_open"]),
                "paper_closed": int(bucket["paper_closed"]),
                "paper_net_pnl": round(float(bucket["paper_net_pnl"]), 4),
                "audit_approved": int(bucket["audit_approved"]),
                "audit_rejected": int(bucket["audit_rejected"]),
                "live_entries_today": int(bucket["live_entries_today"]),
                "top_reject_reason": top_reject_reason,
            }
        )
    source_rows.sort(
        key=lambda row: (
            row["live_entries_today"],
            row["paper_closed"],
            row["audit_approved"],
            row["paper_net_pnl"],
        ),
        reverse=True,
    )

    side_rows = []
    for label, bucket in by_side.items():
        if not any(bucket.values()):
            continue
        side_rows.append(
            {
                "label": label,
                "paper_open": int(bucket["paper_open"]),
                "paper_closed": int(bucket["paper_closed"]),
                "paper_net_pnl": round(float(bucket["paper_net_pnl"]), 4),
                "audit_approved": int(bucket["audit_approved"]),
                "audit_rejected": int(bucket["audit_rejected"]),
            }
        )
    side_rows.sort(key=lambda row: row["label"])

    summary["paper_net_pnl"] = round(float(summary["paper_net_pnl"]), 4)
    summary["paper_sources_realized"] = sum(1 for row in source_rows if row["paper_closed"] > 0)
    summary["live_sources_active"] = sum(1 for row in source_rows if row["live_entries_today"] > 0)
    summary["approval_gap"] = int(summary["audit_approved"] - summary["live_entries_today"])

    return {
        "summary": summary,
        "by_source": source_rows[:source_limit],
        "by_side": side_rows,
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
