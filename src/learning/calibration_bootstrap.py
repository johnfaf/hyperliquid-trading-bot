"""Calibration bootstrap from clean trade history.

Problem
-------
On boot (or after a calibration reset / quarantine clear), the in-memory
calibration bins are empty for most ``(source_key, side, regime)``
buckets.  The EV gate's bucketed confidence floor then defaults to a
conservative prior that rejects most signals.  Combined with the
calibration deadlock from src/signals/firewall_shadow.py, this can keep
the firewall clamped for hours after a restart even when the underlying
strategies are healthy.

This module replays clean closed paper_trades through the calibration
tracker so its bins have realistic priors immediately, breaking the
"need outcomes to update calibration, but no outcomes flow because
calibration is conservative" loop on cold start.

Safety
------
- Pure read of paper_trades + writes to calibration_records (the
  existing CalibrationTracker.record() path).
- Skips any (source_key, side, regime) bucket that already has
  ``skip_if_records_exceed`` records -- protects against double-
  counting on repeated boots.
- Excludes trades flagged ``metadata.tainted=true`` so the seed is
  not poisoned by historical reconciler kills.
- Default OFF.  Enable via ``CALIBRATION_BOOTSTRAP_ON_BOOT=1`` or run
  ``scripts/bootstrap_calibration.py`` manually.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from src.data import database as db

logger = logging.getLogger(__name__)


# ── Env helpers ─────────────────────────────────────────────────


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(float(os.environ.get(name, default) or default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def bootstrap_enabled_on_boot() -> bool:
    """True when CALIBRATION_BOOTSTRAP_ON_BOOT is truthy."""
    return _env_bool("CALIBRATION_BOOTSTRAP_ON_BOOT", default=False)


def lookback_days() -> int:
    return _env_int("CALIBRATION_BOOTSTRAP_LOOKBACK_DAYS", 30, 1, 365)


def skip_if_records_exceed() -> int:
    return _env_int("CALIBRATION_BOOTSTRAP_SKIP_THRESHOLD", 100, 10, 10_000)


def max_per_bucket() -> int:
    return _env_int("CALIBRATION_BOOTSTRAP_MAX_PER_BUCKET", 200, 10, 10_000)


# ── Helpers ─────────────────────────────────────────────────────


def _parse_metadata(raw) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _derive_source_key(trade: Dict[str, Any]) -> str:
    meta = _parse_metadata(trade.get("metadata"))
    raw = (
        meta.get("source_key")
        or meta.get("source")
        or trade.get("source")
        or trade.get("strategy_type")
        or "unknown"
    )
    key = str(raw or "unknown").strip().lower() or "unknown"
    if key == "copy_trade":
        trader = str(meta.get("source_trader") or "").strip().lower()
        if trader:
            return f"copy_trade:{trader}"
    elif key == "strategy":
        st = str(meta.get("strategy_type") or trade.get("strategy_type") or "").strip().lower()
        if st:
            return f"strategy:{st}"
    return key


def _derive_regime(trade: Dict[str, Any]) -> Optional[str]:
    meta = _parse_metadata(trade.get("metadata"))
    regime = meta.get("regime") or trade.get("regime")
    if not regime:
        return None
    return str(regime).strip().lower() or None


def _derive_confidence(trade: Dict[str, Any]) -> float:
    meta = _parse_metadata(trade.get("metadata"))
    for key in ("confidence", "predicted_confidence"):
        if key in meta:
            try:
                return max(0.0, min(float(meta[key]), 1.0))
            except (TypeError, ValueError):
                pass
    return 0.5  # neutral prior


def _fetch_clean_closed_trades(
    *,
    lookback_days_v: int,
) -> Iterable[Dict[str, Any]]:
    """Return closed paper_trades from the lookback window, excluding
    tainted ones."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days_v)
    cutoff_iso = cutoff.isoformat()
    try:
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(
                """
                SELECT id, coin, side, pnl, closed_at, metadata
                FROM paper_trades
                WHERE status = 'closed'
                  AND closed_at >= ?
                  AND pnl IS NOT NULL
                """,
                (cutoff_iso,),
            ).fetchall()
    except Exception as exc:
        logger.warning("calibration_bootstrap: clean-trades query failed: %s", exc)
        return []

    out = []
    for row in rows:
        trade = (
            dict(row)
            if hasattr(row, "keys")
            else {
                "id": row[0], "coin": row[1], "side": row[2],
                "pnl": row[3], "closed_at": row[4], "metadata": row[5],
            }
        )
        meta = _parse_metadata(trade.get("metadata"))
        if meta.get("tainted"):
            continue  # reconciler-kill artefact -- exclude
        out.append(trade)
    return out


# ── Public API ──────────────────────────────────────────────────


def bootstrap_calibration_from_history(
    calibration_tracker,
    *,
    lookback_days_v: Optional[int] = None,
    skip_threshold: Optional[int] = None,
    cap_per_bucket: Optional[int] = None,
) -> Dict[str, int]:
    """Seed the calibration tracker's bins from clean closed paper_trades.

    Returns a stats dict::

        {
            "trades_read": N,
            "records_seeded": M,
            "buckets_skipped_full": K,    # already had >threshold records
            "buckets_seeded": B,           # distinct buckets touched
        }

    Idempotent on repeated calls: buckets that already exceed
    ``skip_threshold`` records are skipped.
    """
    if calibration_tracker is None:
        logger.info("calibration_bootstrap: no calibration tracker provided, skipping")
        return {"trades_read": 0, "records_seeded": 0,
                "buckets_skipped_full": 0, "buckets_seeded": 0}

    lookback_v = lookback_days_v if lookback_days_v is not None else lookback_days()
    skip_v = skip_threshold if skip_threshold is not None else skip_if_records_exceed()
    cap_v = cap_per_bucket if cap_per_bucket is not None else max_per_bucket()

    trades = list(_fetch_clean_closed_trades(lookback_days_v=lookback_v))
    if not trades:
        logger.info(
            "calibration_bootstrap: no clean closed trades in last %d days",
            lookback_v,
        )
        return {"trades_read": 0, "records_seeded": 0,
                "buckets_skipped_full": 0, "buckets_seeded": 0}

    # Track per-bucket seed counts to respect cap_v.
    seeded_per_bucket: Dict[tuple, int] = defaultdict(int)
    buckets_skipped_full: set = set()
    records_seeded = 0

    for trade in trades:
        try:
            coin = str(trade.get("coin") or "").strip().upper()
            side = str(trade.get("side") or "").strip().lower()
            if not coin or side not in {"long", "short"}:
                continue
            pnl = float(trade.get("pnl") or 0.0)
            source_key = _derive_source_key(trade)
            regime = _derive_regime(trade)
            confidence = _derive_confidence(trade)
            bucket = (source_key, side, regime or "")

            # Idempotency: skip buckets that already have lots of data.
            if bucket in buckets_skipped_full:
                continue
            try:
                composed = calibration_tracker._resolve_key(  # type: ignore[attr-defined]
                    source_key, side=side, regime=regime,
                )
                current_size = calibration_tracker.get_sample_size(composed)
            except Exception:
                current_size = 0
            if current_size > skip_v:
                buckets_skipped_full.add(bucket)
                continue

            if seeded_per_bucket[bucket] >= cap_v:
                continue

            calibration_tracker.record(
                source_key=source_key,
                predicted_confidence=confidence,
                actual_win=(pnl > 0),
                pnl=pnl,
                coin=coin,
                side=side,
                regime=regime,
            )
            seeded_per_bucket[bucket] += 1
            records_seeded += 1
        except Exception as exc:
            logger.debug(
                "calibration_bootstrap: skipped trade %s: %s",
                trade.get("id"), exc,
            )

    stats = {
        "trades_read": len(trades),
        "records_seeded": records_seeded,
        "buckets_skipped_full": len(buckets_skipped_full),
        "buckets_seeded": len(seeded_per_bucket),
    }
    logger.info(
        "calibration_bootstrap complete: read=%d, seeded=%d, "
        "buckets=%d, skipped_full=%d (lookback=%dd, cap_per_bucket=%d)",
        stats["trades_read"], stats["records_seeded"],
        stats["buckets_seeded"], stats["buckets_skipped_full"],
        lookback_v, cap_v,
    )
    return stats
