"""Regime-flip exit policy for live positions (default OFF).

Closes a live position via reduce-only market order when the bot's regime
detector + forecaster both flip against the position's direction with high
confidence for a sustained number of cycles.  This is *in addition to*
the SL/TP brackets already attached at entry -- the regime-flip exit is
a higher-conviction "the thesis is broken" exit that fires BEFORE the
fixed-distance SL would.

Layered safety gates (every gate must pass to close):

  1. Module is enabled (REGIME_FLIP_EXIT_ENABLED)
  2. Live trading is actually active for this container
  3. The position has been held at least REGIME_FLIP_EXIT_MIN_HOLD_SECONDS
     (anti-whipsaw on fresh entries while initial volatility settles)
  4. The coin's current regime direction is OPPOSITE to the position side
  5. The coin's regime confidence is >= REGIME_FLIP_EXIT_MIN_CONFIDENCE
  6. (Optional) The forecaster signal also points against the position
     by >= REGIME_FLIP_EXIT_FORECASTER_MIN_SIGNAL magnitude
  7. The against-direction condition has held for
     >= REGIME_FLIP_EXIT_MIN_CONSECUTIVE_CYCLES cycles

When DRY_RUN is True (default), gate 7 still increments the counter and
logs "would close" -- the actual close call is skipped.  This lets
operators observe the trigger pattern in production before flipping
DRY_RUN off.  The counter is keyed per (coin, side) so flipping sides
mid-cycle resets the count.

The module never raises -- any unexpected error is logged at debug and
the cycle continues as if regime-flip exit had been disabled.  Existing
SL/TP brackets remain the safety floor regardless.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import config
from src.core import clock_provider
from src.core.live_execution import get_live_trader, is_live_trading_active
from src.data import database as db

logger = logging.getLogger(__name__)

# Module-level counter: (coin, side) -> consecutive cycles the regime has
# been against this position.  Reset to 0 when the gate fails for any
# reason other than insufficient-consecutive-cycles, OR when the position
# is closed (so a fresh re-entry starts fresh).
_against_cycle_counters: Dict[Tuple[str, str], int] = {}


_BULLISH_TOKENS = {"up", "bullish", "trending_up", "uptrend", "rally"}
_BEARISH_TOKENS = {"down", "bearish", "trending_down", "downtrend", "crash"}


def _coin_direction(coin_data: Dict[str, Any]) -> str:
    """Normalise the per-coin regime read into ``up`` / ``down`` / ``""``.

    Mirrors the logic in ``trading_cycle._direction_from_coin_item`` but
    without the volume/momentum thresholds -- we want the *regime
    classifier's* direction, not the entry filter's stricter rule.
    """
    if not isinstance(coin_data, dict):
        return ""
    explicit = str(
        coin_data.get("direction")
        or coin_data.get("regime")
        or coin_data.get("predicted_regime")
        or ""
    ).strip().lower()
    if explicit in _BULLISH_TOKENS:
        return "up"
    if explicit in _BEARISH_TOKENS:
        return "down"
    # Fall back to a numeric momentum / trend_direction sign with a small
    # tolerance so jitter around 0 doesn't flip us.
    try:
        momentum = float(coin_data.get("momentum", 0) or 0)
        trend = float(coin_data.get("trend_direction", 0) or 0)
        signal = momentum if abs(momentum) >= abs(trend) else trend
    except (TypeError, ValueError):
        signal = 0.0
    if signal > 0.01:
        return "up"
    if signal < -0.01:
        return "down"
    return ""


def _coin_confidence(coin_data: Dict[str, Any]) -> float:
    """Normalise the per-coin confidence to a [0, 1] float."""
    if not isinstance(coin_data, dict):
        return 0.0
    for key in ("confidence", "regime_confidence", "conf"):
        val = coin_data.get(key)
        if val is None:
            continue
        try:
            f = float(val)
            # Some sources express confidence as 0-100 instead of 0-1.
            if f > 1.5:
                f = f / 100.0
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            continue
    return 0.0


def _forecaster_signal(container, coin: str) -> Optional[float]:
    """Return the predictive forecaster's signed signal for ``coin``.

    Positive = bullish lean, negative = bearish lean, magnitude 0..~1.
    Returns None when no forecaster is attached or no read is cached.
    Never raises.
    """
    try:
        forecaster = getattr(container, "predictive_forecaster", None)
        if forecaster is None:
            return None
        # Forecasters cache last reads under different attribute names
        # depending on implementation; try the documented ones in order.
        for attr in ("last_signals", "latest_signals", "_last_signals"):
            store = getattr(forecaster, attr, None)
            if isinstance(store, dict):
                read = store.get(coin) or store.get(coin.upper())
                if isinstance(read, dict):
                    sig = read.get("signal")
                    if sig is not None:
                        return float(sig)
                elif read is not None:
                    return float(read)
    except Exception as exc:
        logger.debug("regime_flip_exit forecaster read failed for %s: %s", coin, exc)
    return None


def _resolve_open_age_seconds(coin: str, now_ts: float) -> Optional[float]:
    """Return how long the current paper trade for ``coin`` has been open.

    Looks up the most recent OPEN paper_trade row for the coin and
    returns ``now_ts - opened_at`` in seconds.  Returns None when the
    row isn't found (e.g. orphan live position with no paper shadow),
    which causes the caller to fail OPEN -- we'd rather let a
    well-protected SL handle an unknown-age position than guess.
    """
    try:
        rows = db.get_open_paper_trades() or []
        for row in rows:
            if str(row.get("coin", "")).upper() != coin.upper():
                continue
            opened_at = row.get("opened_at") or row.get("created_at") or 0
            if not opened_at:
                continue
            # Accept either an ISO string, a unix-seconds float, or ms.
            from datetime import datetime, timezone
            if isinstance(opened_at, (int, float)):
                ts = float(opened_at)
                if ts > 1e12:  # ms
                    ts /= 1000.0
                return max(0.0, now_ts - ts)
            try:
                # Tolerate naive ISO 8601 with or without Z suffix.
                txt = str(opened_at).replace("Z", "+00:00")
                dt = datetime.fromisoformat(txt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return max(0.0, now_ts - dt.timestamp())
            except Exception:
                continue
    except Exception as exc:
        logger.debug("regime_flip_exit age lookup failed for %s: %s", coin, exc)
    return None


def _reset_counter(coin: str, side: str) -> None:
    _against_cycle_counters.pop((coin.upper(), side.lower()), None)


def _bump_counter(coin: str, side: str) -> int:
    key = (coin.upper(), side.lower())
    _against_cycle_counters[key] = _against_cycle_counters.get(key, 0) + 1
    return _against_cycle_counters[key]


def evaluate_regime_flip_exits(container, regime_data: Optional[Dict[str, Any]]) -> None:
    """Scan all open live positions; close ones whose regime has flipped.

    Never raises.  Idempotent within a single cycle (counter only bumps
    when this function is called).  Safe to invoke even when the
    container has no live trader or no regime data.
    """
    if not bool(getattr(config, "REGIME_FLIP_EXIT_ENABLED", False)):
        return
    if not isinstance(regime_data, dict):
        return
    per_coin = regime_data.get("per_coin")
    if not isinstance(per_coin, dict) or not per_coin:
        return

    trader = get_live_trader(container)
    if not trader or not is_live_trading_active(container):
        return

    try:
        positions = trader.get_positions(force_fresh=True) or []
    except Exception as exc:
        logger.debug("regime_flip_exit get_positions failed: %s", exc)
        return

    if not positions:
        return

    # Snapshot config values once per call so the gate behaviour is
    # consistent across positions in this cycle.
    min_conf = float(getattr(config, "REGIME_FLIP_EXIT_MIN_CONFIDENCE", 0.70))
    min_cycles = max(1, int(getattr(
        config, "REGIME_FLIP_EXIT_MIN_CONSECUTIVE_CYCLES", 2,
    )))
    min_hold_s = max(0, int(getattr(config, "REGIME_FLIP_EXIT_MIN_HOLD_SECONDS", 300)))
    require_fc = bool(getattr(
        config, "REGIME_FLIP_EXIT_REQUIRE_FORECASTER_AGREE", True,
    ))
    fc_min_signal = float(getattr(
        config, "REGIME_FLIP_EXIT_FORECASTER_MIN_SIGNAL", 0.20,
    ))
    dry_run = bool(getattr(config, "REGIME_FLIP_EXIT_DRY_RUN", True))

    now_ts = clock_provider.unix_now()

    for pos in positions:
        try:
            coin = str(pos.get("coin", "")).upper()
            if not coin:
                continue
            # szi > 0 = long, < 0 = short.  Use pos["side"] when available
            # (LIVE traders normalise this) and fall back to sign of szi.
            side_raw = str(pos.get("side", "") or "").strip().lower()
            if side_raw not in ("long", "short"):
                try:
                    szi = float(pos.get("szi", 0) or 0)
                except (TypeError, ValueError):
                    szi = 0.0
                if szi == 0:
                    continue
                side_raw = "long" if szi > 0 else "short"

            # Gate 1: minimum hold time
            age_s = _resolve_open_age_seconds(coin, now_ts)
            if age_s is None:
                # Orphan / unresolvable -- let SL handle it, don't reset
                # the counter (next cycle might find the paper row).
                continue
            if age_s < min_hold_s:
                _reset_counter(coin, side_raw)
                continue

            # Gate 2 + 3: regime direction + confidence
            coin_data = per_coin.get(coin) or per_coin.get(coin.lower()) or {}
            direction = _coin_direction(coin_data)
            conf = _coin_confidence(coin_data)
            against = (
                (side_raw == "long" and direction == "down")
                or (side_raw == "short" and direction == "up")
            )
            if not (against and conf >= min_conf):
                _reset_counter(coin, side_raw)
                continue

            # Gate 4 (optional): forecaster must also agree
            if require_fc:
                fc_sig = _forecaster_signal(container, coin)
                if fc_sig is None:
                    # No forecaster read available -- conservative: do
                    # NOT count this cycle (defer to next read).
                    continue
                fc_against = (
                    (side_raw == "long" and fc_sig <= -fc_min_signal)
                    or (side_raw == "short" and fc_sig >= fc_min_signal)
                )
                if not fc_against:
                    _reset_counter(coin, side_raw)
                    continue

            # Gate 5: persistence
            count = _bump_counter(coin, side_raw)
            if count < min_cycles:
                logger.info(
                    "regime_flip_exit: %s %s pending close (cycle %d/%d, "
                    "regime=%s conf=%.0f%% age=%ds)",
                    side_raw.upper(), coin, count, min_cycles,
                    direction, conf * 100.0, int(age_s),
                )
                continue

            # All gates passed.
            if dry_run:
                logger.warning(
                    "[DRY-RUN] regime_flip_exit WOULD close %s %s: "
                    "regime=%s conf=%.0f%% age=%ds cycles_against=%d. "
                    "Set REGIME_FLIP_EXIT_DRY_RUN=false to enable real closes.",
                    side_raw.upper(), coin, direction, conf * 100.0,
                    int(age_s), count,
                )
                continue

            logger.warning(
                "regime_flip_exit: closing %s %s "
                "(regime=%s conf=%.0f%% age=%ds cycles_against=%d)",
                side_raw.upper(), coin, direction, conf * 100.0,
                int(age_s), count,
            )
            try:
                result = trader.close_position(coin)
            except Exception as exc:
                logger.error(
                    "regime_flip_exit close_position failed for %s: %s",
                    coin, exc,
                )
                # Do not reset counter -- retry next cycle.
                continue
            status = (result or {}).get("status", "?")
            if status in ("success", "filled", "ok"):
                logger.warning(
                    "regime_flip_exit: closed %s %s -> %s",
                    side_raw.upper(), coin, status,
                )
                _reset_counter(coin, side_raw)
            else:
                logger.error(
                    "regime_flip_exit close FAILED for %s %s: %s",
                    side_raw.upper(), coin, result,
                )
                # Leave counter intact so next cycle retries.
        except Exception as exc:
            logger.debug(
                "regime_flip_exit per-position eval failed for %s: %s",
                pos.get("coin", "?"), exc,
            )
            continue
