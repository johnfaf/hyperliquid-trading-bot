"""Classify a closed trade by the *cause* of the close, not just by P&L sign.

Why this exists
---------------

The A2 Thompson allocator currently feeds the bandit a binary win/loss
signal derived only from final P&L. That's wrong when the bot's own
stop-loss was too tight and triggered on intra-bar noise -- the source
trader's signal was probably fine, the bot just exited prematurely.
Penalising the source under that scenario is the "stop-loss feedback
loop" structural risk: it punishes traders for *our* mistake, lowers
their bandit weight, and makes future allocations less likely.

A1 attacks the root cause (widen the stop with ATR floor) but A2 also
needs to be aware so it doesn't poison its own posterior with noise
stops while A1 is rolling out (or for any signal where A1's flag is
off / ATR data is missing).

What we classify
----------------

* TAKE_PROFIT      -- TP hit; clean win, feed bandit as win.
* SIGNAL_LOSS      -- A real adverse move beyond the noise band;
                       the source signal was wrong. Feed bandit as loss.
* NOISE_STOP       -- Stop-loss hit but only after a tiny adverse move
                       within typical 5m noise (below max(2*ATR, 50 bps)).
                       Most likely a too-tight stop, NOT a source error.
                       Skip the bandit update (don't penalise the source).
* TIME_OUT         -- Position closed because hold time exceeded a limit;
                       outcome could go either way -- feed bandit by P&L
                       sign, but the operator may also want to study this
                       set separately (long-tail of held-too-long alpha).
* RECONCILED       -- The paper trade was closed because the matching
                       live position vanished (see live_reconciled_closed
                       close_reason). Often book-keeping rather than
                       directional. Skip bandit update by default.
* OTHER            -- Anything else -- feed bandit by P&L sign as a
                       safe default.

The function is *pure* (no DB, no logging). Callers decide whether to
suppress / weight / log the bandit update based on the classification.

Default for the wiring (in agent_scoring.record_outcome) is OFF -- the
bandit gets the legacy P&L-sign-only signal until an operator opts in
via the BANDIT_SKIP_NOISE_STOPS_ENABLED flag. Default-OFF preserves
byte-identity for the current bandit behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CloseClass(str, Enum):
    """How the trade closed (the bandit-relevant grouping)."""
    TAKE_PROFIT = "take_profit"
    SIGNAL_LOSS = "signal_loss"
    NOISE_STOP = "noise_stop"
    TIME_OUT = "time_out"
    RECONCILED = "reconciled"
    OTHER = "other"


@dataclass(frozen=True)
class ClassifyInputs:
    """Bundled inputs for classify_close().

    All fields optional so callers without full context can still classify
    at the coarser level (e.g. just by close_reason). The classifier
    becomes more selective the more context it has.
    """
    pnl: float = 0.0
    close_reason: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    atr_pct: float = 0.0
    side: str = ""                      # "long" or "short"
    leverage: float = 1.0


def _price_move_pct(entry: float, exit_price: float) -> float:
    """Absolute price move as fraction of entry. Returns 0 if inputs invalid."""
    if entry <= 0 or exit_price <= 0:
        return 0.0
    return abs(exit_price - entry) / entry


def _is_take_profit_reason(reason: str) -> bool:
    r = (reason or "").lower()
    return "take_profit" in r or "tp_hit" in r or "tp_reached" in r


def _is_stop_loss_reason(reason: str) -> bool:
    r = (reason or "").lower()
    return "stop_loss" in r or "stop-loss" in r or "sl_hit" in r


def _is_time_out_reason(reason: str) -> bool:
    r = (reason or "").lower()
    return any(t in r for t in ("time_limit", "time_stop", "max_hold", "expiry"))


def _is_reconciled_reason(reason: str) -> bool:
    r = (reason or "").lower()
    return "reconcil" in r  # matches live_reconciled_closed, etc.


def classify_close(
    inputs: ClassifyInputs,
    *,
    noise_atr_multiplier: float = 2.0,
    noise_floor_bps: float = 50.0,
) -> CloseClass:
    """Classify a closed trade by its dominant cause.

    Parameters
    ----------
    inputs
        Bundle of trade close fields.
    noise_atr_multiplier
        A stop-loss exit whose realised price move is below
        ``noise_atr_multiplier * atr_pct`` is classified as a noise
        stop (assuming we have an ATR estimate). Default 2.0 is
        slightly tighter than A1's 2.5 multiplier so the classifier
        is conservative -- only obvious noise stops get marked.
    noise_floor_bps
        If we have no ATR estimate (atr_pct == 0), a stop-loss exit
        whose realised price move is below this many bps is still
        treated as a noise stop. Default 50 bps mirrors A1's default
        ATR_STOP_NOISE_FLOOR_BPS.

    Returns
    -------
    CloseClass enum value. Never raises; malformed inputs collapse
    to OTHER.
    """
    reason = inputs.close_reason or ""

    # Reconciliation: book-keeping, not a source-quality signal.
    if _is_reconciled_reason(reason):
        return CloseClass.RECONCILED

    # Take-profit: by definition a win.
    if _is_take_profit_reason(reason):
        return CloseClass.TAKE_PROFIT

    # Time-out: orthogonal to source quality, classify separately.
    if _is_time_out_reason(reason):
        return CloseClass.TIME_OUT

    # Stop-loss: distinguish noise vs real signal failure.
    if _is_stop_loss_reason(reason):
        move = _price_move_pct(inputs.entry_price, inputs.exit_price)
        if move <= 0:
            return CloseClass.SIGNAL_LOSS  # unknown move; assume real
        # If we have an ATR estimate, compare against k * ATR.
        if inputs.atr_pct > 0:
            noise_threshold = max(
                noise_atr_multiplier * inputs.atr_pct,
                noise_floor_bps / 10000.0,
            )
        else:
            noise_threshold = noise_floor_bps / 10000.0
        if move < noise_threshold:
            return CloseClass.NOISE_STOP
        return CloseClass.SIGNAL_LOSS

    # No close-reason hint -- fall back to P&L sign.
    if inputs.pnl > 0:
        return CloseClass.TAKE_PROFIT
    if inputs.pnl < 0:
        return CloseClass.SIGNAL_LOSS
    return CloseClass.OTHER


# ── Convenience for the wiring layer ────────────────────────────────


def should_feed_bandit(
    cls: CloseClass,
    *,
    skip_noise_stops: bool = True,
    skip_reconciled: bool = True,
) -> bool:
    """Decision helper: given a classification, should we update the bandit?

    Default policy: skip NOISE_STOP and RECONCILED. Real wins/losses
    AND time-outs still feed the posterior.
    """
    if skip_noise_stops and cls == CloseClass.NOISE_STOP:
        return False
    if skip_reconciled and cls == CloseClass.RECONCILED:
        return False
    return True


def bandit_outcome(cls: CloseClass, pnl: float) -> Optional[bool]:
    """Map a classification + P&L into the bandit's binary input.

    Returns
    -------
    True if the bandit should record a win, False for a loss, or
    None if the caller should *not* feed the bandit at all.
    Callers normally pair this with should_feed_bandit() but
    bandit_outcome() handles the same predicate internally for
    convenience.
    """
    if cls in (CloseClass.NOISE_STOP, CloseClass.RECONCILED):
        return None
    if cls == CloseClass.TAKE_PROFIT:
        return True
    if cls == CloseClass.SIGNAL_LOSS:
        return False
    # TIME_OUT / OTHER: fall back to P&L sign.
    return pnl > 0
