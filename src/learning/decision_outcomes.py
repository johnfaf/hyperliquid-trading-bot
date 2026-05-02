"""Decision outcome labelling for offline learning.

This module connects the decision journal to the feature store. It is
deliberately offline/best-effort: failures produce missing labels, not live
trading side effects.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.data import decision_journal

logger = logging.getLogger(__name__)

HORIZONS_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "24h": 24 * 60 * 60 * 1000,
}


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_ts_ms(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        return int(raw if raw > 10_000_000_000 else raw * 1000)
    try:
        text = str(value).strip()
        if text.isdigit():
            raw = float(text)
            return int(raw if raw > 10_000_000_000 else raw * 1000)
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _row_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {}


def _table_exists(conn: Any, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _fetch_future_close(
    conn: Any,
    *,
    coin: str,
    target_ts_ms: int,
    timeframes: Iterable[str],
) -> Optional[float]:
    for timeframe in timeframes:
        try:
            row = conn.execute(
                """
                SELECT close
                FROM candles
                WHERE coin = ? AND timeframe = ? AND timestamp_ms >= ?
                ORDER BY timestamp_ms ASC
                LIMIT 1
                """,
                (coin, timeframe, int(target_ts_ms)),
            ).fetchone()
        except Exception:
            row = None
        if row:
            close = _float(_row_dict(row).get("close"), None)
            if close and close > 0:
                return close
    return None


def _fetch_candle_path(
    conn: Any,
    *,
    coin: str,
    start_ts_ms: int,
    target_ts_ms: int,
    timeframes: Iterable[str],
) -> List[Dict[str, float]]:
    for timeframe in timeframes:
        try:
            rows = conn.execute(
                """
                SELECT timestamp_ms, high, low, close
                FROM candles
                WHERE coin = ? AND timeframe = ? AND timestamp_ms > ? AND timestamp_ms <= ?
                ORDER BY timestamp_ms ASC
                """,
                (coin, timeframe, int(start_ts_ms), int(target_ts_ms)),
            ).fetchall()
        except Exception:
            rows = []
        candles: List[Dict[str, float]] = []
        for row in rows or []:
            data = _row_dict(row)
            high = _float(data.get("high"), None)
            low = _float(data.get("low"), None)
            close = _float(data.get("close"), None)
            if high is None or low is None or close is None:
                continue
            if high <= 0 or low <= 0 or close <= 0:
                continue
            candles.append({"high": high, "low": low, "close": close})
        if candles:
            return candles
    return []


def _counterfactual_path_return(
    decision: Dict[str, Any],
    *,
    side: str,
    entry: float,
    candles: List[Dict[str, float]],
    horizon_return: float,
) -> tuple[float, str]:
    """Estimate the rejected trade outcome using stored SL/TP and intrahorizon bars.

    If a wide candle crosses both SL and TP, assume the stop fired first. That is
    deliberately conservative for calibration: it avoids teaching the firewall
    that rejected trades were winners when OHLC data cannot prove the fill order.
    """
    sl = _float(decision.get("proposed_sl_price"), None)
    tp = _float(decision.get("proposed_tp_price"), None)
    if not candles or not sl or not tp or sl <= 0 or tp <= 0:
        return float(horizon_return), "horizon_close"

    for candle in candles:
        high = float(candle.get("high", 0.0) or 0.0)
        low = float(candle.get("low", 0.0) or 0.0)
        if side == "long":
            hit_sl = low <= sl
            hit_tp = high >= tp
            if hit_sl:
                reason = "ambiguous_stop_first" if hit_tp else "stop_loss"
                return (sl - entry) / entry, reason
            if hit_tp:
                return (tp - entry) / entry, "take_profit"
        else:
            hit_sl = high >= sl
            hit_tp = low <= tp
            if hit_sl:
                reason = "ambiguous_stop_first" if hit_tp else "stop_loss"
                return (entry - sl) / entry, reason
            if hit_tp:
                return (entry - tp) / entry, "take_profit"

    return float(horizon_return), "horizon_close"


def compute_forward_labels(
    decision: Dict[str, Any],
    *,
    primary_timeframes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Compute forward side-adjusted returns from stored candles.

    Returns an empty dict when candles are unavailable. ``side_correct`` and
    ``would_have_won`` are based on the longest populated horizon, preferring 4h
    for short-term decisions and falling back to 1h/15m.
    """
    from src.data import database as db

    coin = str(decision.get("coin") or "").upper()
    side = str(decision.get("side") or "").lower()
    if not coin or side not in {"long", "short"}:
        return {}
    signal_ts_ms = _parse_ts_ms(decision.get("signal_timestamp") or decision.get("created_at"))
    if not signal_ts_ms:
        return {}
    entry = _float(decision.get("entry_price"), None)
    if not entry or entry <= 0:
        return {}
    timeframes = list(primary_timeframes or ("5m", "15m", "1h"))
    labels: Dict[str, Any] = {}
    try:
        with db.get_connection(for_read=True) as conn:
            if not _table_exists(conn, "candles"):
                return {}
            for horizon, delta_ms in HORIZONS_MS.items():
                target_ts_ms = signal_ts_ms + delta_ms
                close = _fetch_future_close(
                    conn,
                    coin=coin,
                    target_ts_ms=target_ts_ms,
                    timeframes=timeframes,
                )
                if close is None:
                    continue
                raw_return = (close - entry) / entry
                side_return = raw_return if side == "long" else -raw_return
                labels[f"forward_return_{horizon}"] = side_return
                candle_path = _fetch_candle_path(
                    conn,
                    coin=coin,
                    start_ts_ms=signal_ts_ms,
                    target_ts_ms=target_ts_ms,
                    timeframes=timeframes,
                )
                path_return, exit_reason = _counterfactual_path_return(
                    decision,
                    side=side,
                    entry=entry,
                    candles=candle_path,
                    horizon_return=side_return,
                )
                labels[f"counterfactual_return_{horizon}"] = path_return
                labels[f"counterfactual_exit_{horizon}"] = exit_reason
    except Exception as exc:
        logger.debug("Forward label computation skipped for %s: %s", coin, exc)
        return {}

    preferred = None
    for key in ("forward_return_4h", "forward_return_1h", "forward_return_15m", "forward_return_24h"):
        if labels.get(key) is not None:
            horizon = key.rsplit("_", 1)[-1]
            preferred = labels.get(f"counterfactual_return_{horizon}", labels[key])
            labels["preferred_counterfactual_horizon"] = horizon
            labels["preferred_counterfactual_exit"] = labels.get(f"counterfactual_exit_{horizon}")
            break
    if preferred is not None:
        labels["side_correct"] = 1 if float(preferred) > 0 else 0
        labels["would_have_won"] = 1 if float(preferred) > 0 else 0
        proposed_size = _float(decision.get("proposed_size_usd"), 0.0) or 0.0
        leverage = max(_float(decision.get("proposed_leverage"), 1.0) or 1.0, 1.0)
        labels["missed_profit_usd"] = max(float(preferred), 0.0) * proposed_size * leverage
    return labels


def rebuild_decision_outcomes(
    *,
    limit: int = 5000,
    include_forward: bool = True,
) -> int:
    """Backfill ``decision_outcomes`` rows from existing snapshots."""
    from src.data import database as db

    decision_journal.ensure_schema_ready()
    count = 0
    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM decision_snapshots
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    for row in rows:
        decision = _row_dict(row)
        forward = compute_forward_labels(decision) if include_forward else {}
        if decision_journal.record_decision_outcome(
            str(decision.get("decision_id") or ""),
            forward_labels=forward,
        ):
            count += 1
    return count


def load_decision_outcome_rows(limit: int = 5000) -> List[Dict[str, Any]]:
    """Load compact outcome rows for diagnostics/tests."""
    from src.data import database as db

    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM decision_outcomes
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_row_dict(row) for row in rows]
