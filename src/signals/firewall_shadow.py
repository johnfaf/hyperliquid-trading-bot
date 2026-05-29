"""Firewall shadow mode — calibration data without risk capital.

Problem this solves
-------------------
The bot is in a calibration deadlock: ECE is too high (~0.27) so the
EV gate refuses signals; with 0 signals through, no outcomes are
recorded; without outcomes, calibration never improves.  The firewall
has been blocking 100% of signals for hours.

Shadow mode breaks the loop without risking capital: a configurable
fraction of REJECTED signals are recorded to a side table with their
predicted side+confidence+entry price.  After a hold window
(``FIREWALL_SHADOW_HOLD_MINUTES`` minutes), an evaluator reads
current market prices, computes a simulated win/loss outcome, and
feeds it to the calibration tracker as if the signal had been
executed.  The calibration curves rebuild on this synthetic-but-clean
data, ECE falls, the EV gate relaxes, real signals flow again.

No live capital is ever at risk -- this is pure observability /
back-test wired into the live feature store.

Operator controls
-----------------
- ``FIREWALL_SHADOW_MODE_FRACTION``  (default 0.0; range 0.0-1.0)
  Fraction of rejections to shadow.  0.0 = off, 0.1 = shadow 10%
  of rejected signals.  Sampling is uniform-random per call.

- ``FIREWALL_SHADOW_HOLD_MINUTES`` (default 60; range 5-1440)
  How long to wait before computing the simulated outcome.  The
  outcome is "did the price move favorably for the signal's side
  by N basis points over this window?".  Longer windows yield
  cleaner signals but slower calibration updates.

- ``FIREWALL_SHADOW_WIN_BPS`` (default 20)
  Minimum favorable price move (in basis points) to count as a
  win.  Mirrors the typical short-horizon edge the bot's
  strategies target.

- ``FIREWALL_SHADOW_MAX_EVAL_PER_CYCLE`` (default 20)
  Cap on how many aged shadow signals to evaluate per cycle, so
  one cycle's evaluator pass stays under ~5s even with a backlog.

Safety
------
- Pure observability.  Never touches the live trader / paper trader.
- Default OFF.  Setting any env to 0 / unset is a no-op.
- Failures are fail-open (log and skip); no exception ever bubbles
  to the firewall's hot path.
"""
from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timedelta
from typing import Optional

from src.core.clock_provider import utc_now
from src.data import database as db
from src.data.hyperliquid_client import get_all_mids

logger = logging.getLogger(__name__)


# ── Config helpers (env-driven, never raise) ────────────────────


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, default) or default)
    except (TypeError, ValueError):
        return float(default)
    return max(lo, min(hi, value))


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(float(os.environ.get(name, default) or default))
    except (TypeError, ValueError):
        return int(default)
    return max(lo, min(hi, value))


def shadow_fraction() -> float:
    return _env_float("FIREWALL_SHADOW_MODE_FRACTION", 0.0, 0.0, 1.0)


def shadow_hold_minutes() -> int:
    return _env_int("FIREWALL_SHADOW_HOLD_MINUTES", 60, 5, 1440)


def shadow_win_bps() -> float:
    return _env_float("FIREWALL_SHADOW_WIN_BPS", 20.0, 1.0, 10_000.0)


def shadow_max_eval_per_cycle() -> int:
    return _env_int("FIREWALL_SHADOW_MAX_EVAL_PER_CYCLE", 20, 1, 1000)


# ── Schema (lazy, idempotent) ───────────────────────────────────

# Process-level guard so the CREATE TABLE / CREATE INDEX DDL runs at
# most once per process instead of on every record / evaluate call.
# Without this, an enabled shadow mode (FIREWALL_SHADOW_MODE_FRACTION
# > 0) issues a CREATE TABLE IF NOT EXISTS on the firewall's hot path
# for every sampled rejection -- ~10 redundant DDL statements / cycle,
# each taking a lock on Postgres.  ``force`` lets tests bypass it.
_SCHEMA_READY = False


def _ensure_schema(conn, *, force: bool = False) -> None:
    """Create the firewall_shadow_signals table if missing (once/process).

    Both Postgres and SQLite use compatible-enough syntax that one
    statement works for both via ``executescript`` (SQLite) / direct
    ``execute`` (Postgres).  Indexes are split out for portability.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY and not force:
        return
    backend = db.get_backend_name() if hasattr(db, "get_backend_name") else "sqlite"
    if backend == "postgres":
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS firewall_shadow_signals (
                id BIGSERIAL PRIMARY KEY,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                source_key TEXT,
                entry_price DOUBLE PRECISION NOT NULL,
                rejection_reason TEXT,
                regime TEXT,
                opened_at TIMESTAMPTZ NOT NULL,
                evaluated BOOLEAN NOT NULL DEFAULT FALSE,
                evaluated_at TIMESTAMPTZ,
                simulated_win BOOLEAN,
                simulated_exit_price DOUBLE PRECISION,
                simulated_pnl_pct DOUBLE PRECISION
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_firewall_shadow_pending "
            "ON firewall_shadow_signals (evaluated, opened_at)"
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS firewall_shadow_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_key TEXT,
                entry_price REAL NOT NULL,
                rejection_reason TEXT,
                regime TEXT,
                opened_at TEXT NOT NULL,
                evaluated INTEGER NOT NULL DEFAULT 0,
                evaluated_at TEXT,
                simulated_win INTEGER,
                simulated_exit_price REAL,
                simulated_pnl_pct REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_firewall_shadow_pending "
            "ON firewall_shadow_signals (evaluated, opened_at)"
        )

    _SCHEMA_READY = True


# ── Public API ──────────────────────────────────────────────────


def record_shadow_signal(
    signal,
    rejection_reason: str,
    *,
    mid_price: Optional[float] = None,
    regime: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> bool:
    """Persist a rejected signal as a shadow if sampling allows.

    Returns True when the signal was recorded, False otherwise.
    Never raises.

    Sampling uses a uniform draw against ``shadow_fraction()``.  When
    the env flag is 0.0 (default) this short-circuits in O(1) without
    touching the DB -- safe to call on the firewall's hot path.
    """
    fraction = shadow_fraction()
    if fraction <= 0.0:
        return False

    rng = rng or random
    if rng.random() >= fraction:
        return False

    try:
        coin = str(getattr(signal, "coin", "") or "").strip().upper()
        if not coin:
            return False

        side_obj = getattr(signal, "side", None)
        side = (
            side_obj.value if hasattr(side_obj, "value") else str(side_obj or "")
        ).strip().lower()
        if side not in {"long", "short"}:
            return False

        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)

        # Resolve source key the same way the firewall does.
        source_obj = getattr(signal, "source", None)
        source = (
            source_obj.value if hasattr(source_obj, "value") else str(source_obj or "")
        ).strip().lower() or "unknown"
        strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
        trader_addr = str(getattr(signal, "trader_address", "") or "").strip().lower()
        if source == "copy_trade" and trader_addr:
            source_key = f"{source}:{trader_addr}"
        elif strategy_type:
            source_key = f"{source}:{strategy_type}"
        else:
            source_key = source

        # Entry price: prefer caller-supplied mid, else signal.entry_price.
        entry_price = float(mid_price or 0.0)
        if entry_price <= 0:
            entry_price = float(getattr(signal, "entry_price", 0.0) or 0.0)
        if entry_price <= 0:
            return False

        opened_at = utc_now().isoformat()
        with db.get_connection() as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO firewall_shadow_signals (
                    coin, side, confidence, source_key, entry_price,
                    rejection_reason, regime, opened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    coin, side, confidence, source_key, entry_price,
                    rejection_reason[:200], regime, opened_at,
                ),
            )
        logger.debug(
            "Firewall shadow recorded: %s %s @ %.6f (source=%s, conf=%.2f, "
            "rejected: %s)",
            side.upper(), coin, entry_price, source_key, confidence,
            rejection_reason[:80],
        )
        return True
    except Exception as exc:
        # Fail-open: shadow mode is observability only; never block the
        # firewall's hot path on it.
        logger.debug("Firewall shadow recording skipped (%s)", exc)
        return False


def evaluate_pending_shadow_signals(
    calibration_tracker=None,
    *,
    max_per_call: Optional[int] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Evaluate aged shadow signals and feed outcomes to calibration.

    Returns a stats dict: ``{evaluated: N, wins: W, losses: L, skipped: K}``.
    Safe to call from any cycle phase -- bounded by
    ``FIREWALL_SHADOW_MAX_EVAL_PER_CYCLE``.

    Algorithm
    ---------
    For each pending shadow signal older than the hold window:
      1. Look up the current mid price for its coin.
      2. Compute the price move from entry, signed by side
         (long → favorable when up, short → favorable when down).
      3. ``win = (favorable_move_bps >= shadow_win_bps())``.
      4. Mark the row evaluated and (if calibration_tracker provided)
         feed ``record(source_key, predicted_confidence, win, pnl_pct,
         coin, side)``.

    No partial-fill / slippage / fee modeling.  This produces the
    cleanest "did the bot's directional read materialize over N
    minutes?" signal possible -- which is precisely the prior the
    calibration tracker needs to learn from.
    """
    stats = {"evaluated": 0, "wins": 0, "losses": 0, "skipped": 0}

    fraction = shadow_fraction()
    if fraction <= 0.0:
        # Off entirely.  Even with old rows in the table from a prior
        # run, we don't evaluate them here -- restart the feature.
        return stats

    hold_minutes = shadow_hold_minutes()
    win_bps = shadow_win_bps()
    cap = max_per_call or shadow_max_eval_per_cycle()
    cutoff = (now or utc_now()) - timedelta(minutes=hold_minutes)
    cutoff_iso = cutoff.isoformat()
    eval_now_iso = (now or utc_now()).isoformat()

    try:
        mids = get_all_mids() or {}
    except Exception as exc:
        logger.debug("Shadow evaluator skipped (mid lookup failed): %s", exc)
        return stats

    try:
        with db.get_connection() as conn:
            _ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT id, coin, side, confidence, source_key, entry_price,
                       opened_at, regime
                FROM firewall_shadow_signals
                WHERE evaluated = 0 AND opened_at <= ?
                ORDER BY opened_at
                LIMIT ?
                """,
                (cutoff_iso, cap),
            ).fetchall()
    except Exception as exc:
        logger.debug("Shadow evaluator query failed: %s", exc)
        return stats

    for row in rows:
        try:
            rid = row["id"] if hasattr(row, "keys") else row[0]
            coin = row["coin"] if hasattr(row, "keys") else row[1]
            side = row["side"] if hasattr(row, "keys") else row[2]
            confidence = float(row["confidence"] if hasattr(row, "keys") else row[3])
            source_key = row["source_key"] if hasattr(row, "keys") else row[4]
            entry_price = float(row["entry_price"] if hasattr(row, "keys") else row[5])
            regime = (row["regime"] if hasattr(row, "keys") else row[7]) or None

            current = float(mids.get(coin) or 0.0)
            if current <= 0 or entry_price <= 0:
                # Mid not available -- leave row for next pass instead
                # of marking it evaluated, but count as skipped this
                # round so the stats are honest.
                stats["skipped"] += 1
                continue

            move_bps = ((current - entry_price) / entry_price) * 10_000.0
            if side == "short":
                move_bps = -move_bps
            win = move_bps >= win_bps
            pnl_pct = move_bps / 100.0  # convert bps -> %

            with db.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE firewall_shadow_signals
                    SET evaluated = 1,
                        evaluated_at = ?,
                        simulated_win = ?,
                        simulated_exit_price = ?,
                        simulated_pnl_pct = ?
                    WHERE id = ?
                    """,
                    (eval_now_iso, 1 if win else 0, current, pnl_pct, rid),
                )

            if calibration_tracker is not None:
                try:
                    calibration_tracker.record(
                        source_key=source_key or "unknown",
                        predicted_confidence=confidence,
                        actual_win=bool(win),
                        pnl=pnl_pct,
                        coin=coin,
                        side=side,
                        regime=regime,
                    )
                except Exception as exc:
                    logger.debug(
                        "Shadow outcome -> calibration feed failed for "
                        "%s %s id=%s: %s",
                        side, coin, rid, exc,
                    )

            stats["evaluated"] += 1
            if win:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
        except Exception as exc:
            logger.debug("Shadow row evaluation skipped: %s", exc)
            stats["skipped"] += 1

    if stats["evaluated"] > 0:
        logger.info(
            "Firewall shadow evaluator: %d evaluated (%d wins, %d losses), "
            "%d skipped (hold=%dmin, win_threshold=%.0fbps)",
            stats["evaluated"], stats["wins"], stats["losses"], stats["skipped"],
            hold_minutes, win_bps,
        )
    return stats
