"""A6: Maker-first execution policy with adaptive aggressiveness.

The live_trader already has a maker-then-market path
(``LIVE_ENTRY_EXECUTION_MODE=maker_then_market``), but it's a
single-stage timeout: post ALO, wait N seconds, fall back to market.
That's too coarse for a multi-source bot:

- **copy_trade signals** have inherent execution lag relative to the
  leader; paying taker on top is double-dipping on adverse selection.
  Better: wait longer at the BBO, re-post when the book moves,
  abandon if the signal grows stale rather than pay taker.

- **funding_carry signals** (A4) are tight-timing — the funding
  event has a known cadence. A 15s timeout that escalates to taker
  is the right call.

- **alpha_arena / xgboost signals** sit in the middle.

This module provides a pure ``MakerExecutionPolicy`` that takes
per-source configuration and the current book state, and returns
the next action a wrapping executor should take. The policy is
side-effect-free and trivially testable; the side-effectful order
placement is the caller's job.

State machine
-------------
::

    POST_ALO ──(timeout, BBO unchanged)──► REPOST_AT_BBO
        │                                          │
        │                                          ▼
        │                            (signal_age < max_age)
        │                                          │
        ▼                                          ▼
    FILLED                              TAKER_FALLBACK or ABANDON
                                        (per policy.taker_fallback)

If at any point the BBO moves *toward* the order, the policy treats
the order as filled at the BBO-touching price (caller verifies fills
out-of-band). If the BBO moves *away* by more than ``reprice_bps``,
the policy emits a REPOST_AT_BBO action so the wrapping executor
can cancel + repost without paying taker.

If ``signal_age`` exceeds ``max_signal_age_s``, the policy refuses
to escalate to taker — the position is stale and you'd be eating
the leader's adverse selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MakerAction(str, Enum):
    """One of the actions the policy can recommend."""
    POST_ALO = "post_alo"
    HOLD = "hold"                 # order is live, no change needed
    REPOST_AT_BBO = "repost_at_bbo"
    TAKER_FALLBACK = "taker_fallback"
    ABANDON = "abandon"
    FILLED = "filled"


@dataclass(frozen=True)
class BookState:
    """Snapshot of the relevant book at decision time."""
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)


@dataclass(frozen=True)
class OrderState:
    """Current ALO order state, or None if no order live."""
    side: str                      # "buy" or "sell"
    price: float
    age_s: float                   # seconds since placed
    filled: bool = False


@dataclass(frozen=True)
class MakerPolicy:
    """Configuration knobs for one source-class.

    All timeouts in seconds; price tolerances in bps of mid.
    """
    timeout_s: float = 15.0                  # how long to sit at the BBO before reposting
    reprice_threshold_bps: float = 5.0        # mid-move that triggers a repost
    max_signal_age_s: float = 60.0            # never escalate to taker past this
    taker_fallback: bool = False              # whether to fall back to taker at age limit
    abandon_after_s: Optional[float] = None   # hard timeout — abandon if no fill
    offset_bps: float = 0.0                   # how far inside the BBO to post (0 = at BBO)


@dataclass(frozen=True)
class MakerDecision:
    action: MakerAction
    target_price: Optional[float] = None      # for POST_ALO / REPOST_AT_BBO
    reason: str = ""


def decide(
    *,
    side: str,
    book: BookState,
    order: Optional[OrderState],
    policy: MakerPolicy,
    signal_age_s: float,
) -> MakerDecision:
    """Pure state-machine step.

    Inputs
    ------
    side
        Intended direction: ``"buy"`` or ``"sell"``.
    book
        Current top-of-book bid/ask.
    order
        Currently live ALO order, or None if no order is in the book.
    policy
        Per-source-class config.
    signal_age_s
        Seconds since the underlying signal was emitted. Used to
        decide whether escalating to taker is still acceptable.

    Returns
    -------
    MakerDecision with the action to take, optional target price for
    POST_ALO / REPOST_AT_BBO, and a human-readable reason.
    """
    side_norm = (side or "").strip().lower()
    if side_norm not in {"buy", "sell"}:
        return MakerDecision(MakerAction.ABANDON, reason=f"unknown_side:{side!r}")

    # Where would we WANT to post right now?
    target = _target_price(side_norm, book, policy.offset_bps)
    if target <= 0:
        return MakerDecision(MakerAction.ABANDON, reason="invalid_book")

    # Case 1: no live order yet → post initial ALO.
    if order is None:
        return MakerDecision(
            MakerAction.POST_ALO, target_price=target, reason="initial_post",
        )

    # Case 2: order is reported filled.
    if order.filled:
        return MakerDecision(MakerAction.FILLED, reason="order_filled")

    # Case 3: hard abandon timeout.
    if policy.abandon_after_s is not None and order.age_s >= policy.abandon_after_s:
        return MakerDecision(
            MakerAction.ABANDON, reason=f"abandon_after:{order.age_s:.1f}s",
        )

    # Case 4: signal staleness short-circuit. If the underlying signal
    # is too old, abandon regardless of book or order timeout. We don't
    # want to repost OR take liquidity for a position whose alpha has
    # decayed — both paths inherit the leader's adverse selection.
    if signal_age_s >= policy.max_signal_age_s:
        return MakerDecision(
            MakerAction.ABANDON,
            reason=(
                f"signal_stale:{signal_age_s:.1f}s>=max={policy.max_signal_age_s:.1f}s"
            ),
        )

    # Case 5: BBO moved against the order beyond reprice threshold → repost.
    mid = book.mid
    if mid > 0:
        diff_bps = abs(order.price - target) / mid * 10000.0
        if diff_bps >= policy.reprice_threshold_bps:
            return MakerDecision(
                MakerAction.REPOST_AT_BBO, target_price=target,
                reason=f"bbo_moved:{diff_bps:.1f}bps",
            )

    # Case 6: timeout reached. Decide between taker_fallback and repost.
    if order.age_s >= policy.timeout_s:
        if policy.taker_fallback:
            return MakerDecision(
                MakerAction.TAKER_FALLBACK,
                reason=f"timeout_taker_ok:age={order.age_s:.1f}s",
            )
        # Maker-only mode and timed out → repost in the hope the book
        # comes to us.
        return MakerDecision(
            MakerAction.REPOST_AT_BBO, target_price=target,
            reason=f"timeout_repost:age={order.age_s:.1f}s",
        )

    # Case 6: nothing to do — order is live, recent, on-BBO.
    return MakerDecision(MakerAction.HOLD, reason="on_bbo")


def _target_price(side: str, book: BookState, offset_bps: float) -> float:
    """Return the price we'd want to post an ALO at, given side + book.

    Long: post AT the best bid (or `offset_bps` better i.e. inside the
    spread). Short: post AT the best ask (or `offset_bps` better).
    """
    offset = offset_bps / 10000.0
    if side == "buy":
        return book.bid * (1.0 + offset)
    return book.ask * (1.0 - offset)


# ── Convenience factory: default policies per source-class ────────


def policy_for_source(source_key: str) -> MakerPolicy:
    """Return the recommended default policy for a known source class.

    The categories mirror the multi-lane architecture introduced by
    A2 (Thompson allocator) and A4 (funding carry):
    - copy_trade: be patient, never take liquidity (already lagged)
    - funding_carry: tight timing, taker-fallback OK
    - alpha_arena / xgboost: medium patience, taker-fallback OK
    """
    key = (source_key or "").lower()
    if key.startswith("copy_trade"):
        return MakerPolicy(
            timeout_s=30.0, reprice_threshold_bps=3.0,
            max_signal_age_s=45.0, taker_fallback=False,
            abandon_after_s=90.0, offset_bps=0.0,
        )
    if key.startswith("funding_carry"):
        return MakerPolicy(
            timeout_s=10.0, reprice_threshold_bps=5.0,
            max_signal_age_s=30.0, taker_fallback=True,
            abandon_after_s=60.0, offset_bps=0.0,
        )
    # Default (alpha_arena, xgboost, etc.)
    return MakerPolicy(
        timeout_s=15.0, reprice_threshold_bps=5.0,
        max_signal_age_s=60.0, taker_fallback=True,
        abandon_after_s=120.0, offset_bps=0.0,
    )
