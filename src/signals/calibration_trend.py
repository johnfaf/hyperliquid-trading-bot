"""Rolling Brier-score trend monitor.

ECE at a point in time tells you "are predictions miscalibrated *right
now*?". Brier captures both miscalibration AND lack of discrimination,
but it's also a point-in-time score. What matters for live trading is
*deterioration*: a source whose Brier is getting steadily worse should
be de-risked before its calibration crosses the quarantine threshold.

This module reads ``calibration_records`` time-bucketed by day and
returns the slope of Brier over the last N days. ``apply_trend_derisk``
nudges the agent_scorer dynamic_weight down when slope is positive
(getting worse) by enough to matter.

Default N=3 days, threshold = 0.02 Brier units per day (rough rule:
deteriorating by more than 2% per day = bad signal, drop weight).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import config
from src.data import database as db

logger = logging.getLogger(__name__)


def _config_int(name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default) or default)
    except (TypeError, ValueError):
        return default


def _config_float(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default) or default)
    except (TypeError, ValueError):
        return default


def _enabled() -> bool:
    return bool(getattr(config, "CALIBRATION_TREND_ENABLED", True))


def _trend_window_days() -> int:
    return max(2, _config_int("CALIBRATION_TREND_WINDOW_DAYS", 3))


def _deterioration_threshold() -> float:
    return _config_float("CALIBRATION_TREND_DETERIORATION_BRIER_PER_DAY", 0.02)


def _min_samples_per_day() -> int:
    return max(1, _config_int("CALIBRATION_TREND_MIN_SAMPLES_PER_DAY", 5))


def _derisk_multiplier() -> float:
    """How much to multiply dynamic_weight by when a trend is deteriorating."""
    return _config_float("CALIBRATION_TREND_DERISK_MULTIPLIER", 0.75)


def _compute_daily_brier(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Group calibration_records by UTC day and return per-day Brier."""
    bucket: Dict[str, List[Tuple[float, int]]] = {}
    for row in rows:
        ts = row.get("timestamp") or row.get("created_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            day = dt.astimezone(timezone.utc).date().isoformat()
        except Exception:
            continue
        try:
            conf = float(row.get("predicted_confidence", 0.0) or 0.0)
            win = 1 if int(row.get("actual_win", 0) or 0) else 0
        except (TypeError, ValueError):
            continue
        bucket.setdefault(day, []).append((conf, win))

    result: Dict[str, Dict[str, float]] = {}
    for day, samples in bucket.items():
        n = len(samples)
        if n < _min_samples_per_day():
            continue
        sse = sum((c - w) ** 2 for c, w in samples)
        result[day] = {"brier": sse / n, "n": n}
    return result


def get_brier_trend(source_key: str, *, window_days: Optional[int] = None) -> Dict[str, Any]:
    """Return Brier-score slope for a source over the rolling window.

    Returns a dict with keys ``slope_per_day``, ``brier_per_day`` (per-day
    breakdown), ``window_days``, ``deteriorating`` (bool), and the
    ``source_key``. Empty result (slope=None) when there isn't enough
    data on the source.
    """
    window = window_days or _trend_window_days()
    cutoff = datetime.now(timezone.utc) - timedelta(days=window + 1)

    try:
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(
                """
                SELECT timestamp, predicted_confidence, actual_win
                FROM calibration_records
                WHERE source_key = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (source_key, cutoff.isoformat()),
            ).fetchall()
    except Exception as exc:
        logger.debug("calibration_trend: read failed for %s: %s", source_key, exc)
        return {
            "source_key": source_key,
            "window_days": window,
            "slope_per_day": None,
            "brier_per_day": {},
            "deteriorating": False,
            "error": str(exc.__class__.__name__),
        }

    rows = [dict(r) for r in rows or []]
    per_day = _compute_daily_brier(rows)
    if len(per_day) < 2:
        return {
            "source_key": source_key,
            "window_days": window,
            "slope_per_day": None,
            "brier_per_day": per_day,
            "deteriorating": False,
            "reason": f"insufficient_days:{len(per_day)}/2",
        }

    # Simple linear regression of brier ~ day_offset.
    ordered_days = sorted(per_day.keys())
    series = [(i, per_day[d]["brier"]) for i, d in enumerate(ordered_days)]
    n = len(series)
    mean_x = sum(x for x, _ in series) / n
    mean_y = sum(y for _, y in series) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in series)
    den = sum((x - mean_x) ** 2 for x, _ in series) or 1e-9
    slope = num / den
    threshold = _deterioration_threshold()
    deteriorating = slope >= threshold
    return {
        "source_key": source_key,
        "window_days": window,
        "slope_per_day": round(slope, 5),
        "brier_per_day": per_day,
        "ordered_days": ordered_days,
        "deteriorating": bool(deteriorating),
        "threshold": threshold,
    }


def apply_trend_derisk(agent_scorer: Any, source_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """For each source, if Brier is deteriorating, multiply its dynamic_weight
    by ``CALIBRATION_TREND_DERISK_MULTIPLIER`` and return the list of
    sources that were actually adjusted.

    Read-only when disabled or when no calibration data exists.
    """
    if not _enabled():
        return []
    if agent_scorer is None:
        return []

    if source_keys is None:
        try:
            source_keys = list(getattr(agent_scorer, "scores", {}).keys())
        except Exception:
            source_keys = []
    if not source_keys:
        return []

    mult = _derisk_multiplier()
    adjusted: List[Dict[str, Any]] = []
    for key in source_keys:
        trend = get_brier_trend(key)
        if not trend.get("deteriorating"):
            continue
        try:
            score = agent_scorer.scores.get(key)
            if score is None:
                continue
            old_weight = float(score.dynamic_weight)
            new_weight = max(0.05, old_weight * mult)
            if abs(new_weight - old_weight) < 1e-6:
                continue
            score.dynamic_weight = new_weight
            if hasattr(agent_scorer, "_save_score"):
                try:
                    agent_scorer._save_score(key)
                except Exception:
                    pass
            adjusted.append({
                "source_key": key,
                "slope_per_day": trend.get("slope_per_day"),
                "old_weight": old_weight,
                "new_weight": new_weight,
            })
            logger.warning(
                "Calibration trend derisk %s: slope=%.4f Brier/day -> "
                "weight %.3f -> %.3f",
                key, trend.get("slope_per_day") or 0.0, old_weight, new_weight,
            )
        except Exception as exc:
            logger.debug("apply_trend_derisk: failed for %s: %s", key, exc)
    return adjusted
