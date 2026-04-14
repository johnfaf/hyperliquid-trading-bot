"""
Walk-forward validation for dynamic risk policies.

This module evaluates closed trades chronologically, splitting them into
train/test windows by signal source so we can see whether a risk policy is
actually improving outcomes instead of just looking sensible in isolation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional

from src.data import database as db


def _trade_metadata(trade: Dict[str, Any]) -> Dict[str, Any]:
    metadata = trade.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata or "{}")
        except Exception:
            metadata = {}
    return dict(metadata or {})


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _source_label(trade: Dict[str, Any]) -> str:
    metadata = _trade_metadata(trade)
    raw = (
        metadata.get("source")
        or metadata.get("source_key")
        or trade.get("source")
        or trade.get("strategy_type")
        or "unknown"
    )
    text = str(raw or "unknown").strip().lower()
    if ":" in text:
        return text.split(":", 1)[0]
    return text or "unknown"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _trade_return_pct(trade: Dict[str, Any]) -> float:
    entry = _safe_float(trade.get("entry_price"), 0.0)
    size = abs(_safe_float(trade.get("size"), 0.0))
    leverage = max(_safe_float(trade.get("leverage"), 1.0), 1.0)
    if entry <= 0 or size <= 0:
        return 0.0
    notional = entry * size * leverage
    if notional <= 0:
        return 0.0
    return _safe_float(trade.get("pnl"), 0.0) / notional


def _hold_hours(trade: Dict[str, Any]) -> float:
    opened_at = _parse_timestamp(trade.get("opened_at"))
    closed_at = _parse_timestamp(trade.get("closed_at"))
    if not opened_at or not closed_at:
        return 0.0
    return max((closed_at - opened_at).total_seconds() / 3600.0, 0.0)


def _compute_metrics(trades: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(trades)
    pnls = [_safe_float(row.get("pnl"), 0.0) for row in rows]
    returns = [_trade_return_pct(row) for row in rows]
    count = len(rows)
    wins = sum(1 for value in pnls if value > 0)
    losses = sum(1 for value in pnls if value < 0)
    total_pnl = round(sum(pnls), 6)
    avg_return = sum(returns) / count if count else 0.0

    variance = 0.0
    if count > 1:
        variance = sum((value - avg_return) ** 2 for value in returns) / (count - 1)
    std_dev = math.sqrt(max(variance, 0.0))
    t_stat = 0.0
    p_value = 1.0
    significant = False
    if count >= 5 and std_dev > 0:
        t_stat = avg_return / (std_dev / math.sqrt(count))
        p_value = max(0.0, min(1.0, 2.0 * (1.0 - NormalDist().cdf(abs(t_stat)))))
        significant = p_value < 0.05 and avg_return > 0

    avg_hold = sum(_hold_hours(row) for row in rows) / count if count else 0.0
    return {
        "count": count,
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / count) if count else 0.0, 4),
        "net_pnl": total_pnl,
        "avg_pnl": round((total_pnl / count) if count else 0.0, 6),
        "avg_return_pct": round(avg_return * 100.0, 4),
        "std_return_pct": round(std_dev * 100.0, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
        "avg_hold_hours": round(avg_hold, 3),
    }


def _build_windows(trades: List[Dict[str, Any]], min_train: int, min_test: int, max_windows: int) -> List[Dict[str, Any]]:
    total = len(trades)
    if total < (min_train + min_test):
        return []
    max_start = total - min_test
    step = max(1, (max_start - min_train) // max(max_windows, 1))
    boundaries: List[int] = []
    current = min_train
    while current <= max_start and len(boundaries) < max_windows:
        boundaries.append(current)
        current += step
    if boundaries and boundaries[-1] != max_start:
        boundaries[-1] = max_start

    windows = []
    for split_idx in boundaries:
        train_rows = trades[:split_idx]
        test_rows = trades[split_idx:]
        if len(train_rows) < min_train or len(test_rows) < min_test:
            continue
        train_end = _parse_timestamp(train_rows[-1].get("closed_at"))
        test_end = _parse_timestamp(test_rows[-1].get("closed_at"))
        windows.append(
            {
                "train_start": train_rows[0].get("closed_at", ""),
                "train_end": train_end.isoformat() if train_end else train_rows[-1].get("closed_at", ""),
                "test_start": test_rows[0].get("closed_at", ""),
                "test_end": test_end.isoformat() if test_end else test_rows[-1].get("closed_at", ""),
                "train": _compute_metrics(train_rows),
                "test": _compute_metrics(test_rows),
            }
        )
    return windows


def load_closed_trades(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM paper_trades WHERE status = 'closed' ORDER BY closed_at ASC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def validate_risk_policy_walkforward(
    closed_trades: Optional[List[Dict[str, Any]]] = None,
    *,
    min_train_trades: int = 20,
    min_test_trades: int = 10,
    max_windows: int = 4,
) -> Dict[str, Any]:
    rows = list(closed_trades) if closed_trades is not None else load_closed_trades()
    rows = [row for row in rows if str(row.get("status", "closed")).lower() == "closed"]
    rows.sort(key=lambda row: str(row.get("closed_at", "") or ""))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_source_label(row), []).append(row)

    by_source = []
    significant_sources = 0
    for source, source_rows in sorted(grouped.items()):
        windows = _build_windows(source_rows, min_train_trades, min_test_trades, max_windows)
        source_summary = _compute_metrics(source_rows)
        test_windows = [window["test"] for window in windows]
        passing_windows = sum(1 for item in test_windows if item.get("significant"))
        if passing_windows:
            significant_sources += 1
        by_source.append(
            {
                "source": source,
                "total": source_summary,
                "windows": windows,
                "passing_test_windows": passing_windows,
            }
        )

    return {
        "summary": {
            "sources": len(by_source),
            "closed_trades": len(rows),
            "significant_sources": significant_sources,
        },
        "by_source": by_source,
    }
