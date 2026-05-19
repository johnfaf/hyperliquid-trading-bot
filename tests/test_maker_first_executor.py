"""A6: maker-first execution policy tests.

The decide() function is the entire state machine — pure, no IO.
These tests exercise every transition path:
- initial post
- hold on quiet book
- repost on BBO move
- taker fallback when policy allows
- abandon when signal is stale
- abandon on hard timeout
- filled detection
- malformed inputs

Plus per-source-class defaults.
"""
from __future__ import annotations

import pytest

from src.trading.maker_first_executor import (
    BookState,
    MakerAction,
    MakerDecision,
    MakerPolicy,
    OrderState,
    decide,
    policy_for_source,
)


# ── Initial post ─────────────────────────────────────────────────


def test_initial_post_buy_at_bid():
    """No live order → POST_ALO at the bid."""
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=None,
        policy=MakerPolicy(),
        signal_age_s=2.0,
    )
    assert decision.action == MakerAction.POST_ALO
    assert decision.target_price == pytest.approx(100.0)


def test_initial_post_sell_at_ask():
    decision = decide(
        side="sell",
        book=BookState(bid=100.0, ask=100.5),
        order=None,
        policy=MakerPolicy(),
        signal_age_s=2.0,
    )
    assert decision.action == MakerAction.POST_ALO
    assert decision.target_price == pytest.approx(100.5)


def test_initial_post_with_offset():
    """offset_bps shifts the post price INSIDE the spread."""
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=None,
        policy=MakerPolicy(offset_bps=10.0),  # 10 bps inside
        signal_age_s=2.0,
    )
    # 100.0 * (1 + 0.001) = 100.1
    assert decision.target_price == pytest.approx(100.1)


# ── Hold / filled / no-op ─────────────────────────────────────────


def test_hold_when_order_recent_and_on_bbo():
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=OrderState(side="buy", price=100.0, age_s=5.0),
        policy=MakerPolicy(timeout_s=15.0, reprice_threshold_bps=5.0),
        signal_age_s=10.0,
    )
    assert decision.action == MakerAction.HOLD


def test_filled_short_circuits():
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=OrderState(side="buy", price=100.0, age_s=5.0, filled=True),
        policy=MakerPolicy(),
        signal_age_s=10.0,
    )
    assert decision.action == MakerAction.FILLED


# ── Repost ────────────────────────────────────────────────────────


def test_repost_on_bbo_drift_beyond_threshold():
    """Book moves away by 8 bps; reprice threshold is 5 bps → repost."""
    # bid moved from 100.0 (where order sits) to 100.08 → 8 bps drift
    decision = decide(
        side="buy",
        book=BookState(bid=100.08, ask=100.58),
        order=OrderState(side="buy", price=100.0, age_s=3.0),
        policy=MakerPolicy(timeout_s=15.0, reprice_threshold_bps=5.0),
        signal_age_s=8.0,
    )
    assert decision.action == MakerAction.REPOST_AT_BBO
    assert decision.target_price == pytest.approx(100.08)


def test_no_repost_below_threshold():
    """3 bps drift, threshold 5 bps → HOLD."""
    decision = decide(
        side="buy",
        book=BookState(bid=100.03, ask=100.53),
        order=OrderState(side="buy", price=100.0, age_s=3.0),
        policy=MakerPolicy(timeout_s=15.0, reprice_threshold_bps=5.0),
        signal_age_s=8.0,
    )
    assert decision.action == MakerAction.HOLD


def test_repost_on_timeout_when_taker_disabled():
    """Maker-only policy + timeout → REPOST_AT_BBO (not taker)."""
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=OrderState(side="buy", price=100.0, age_s=20.0),
        policy=MakerPolicy(timeout_s=15.0, taker_fallback=False, max_signal_age_s=60.0),
        signal_age_s=25.0,
    )
    assert decision.action == MakerAction.REPOST_AT_BBO


# ── Taker fallback ────────────────────────────────────────────────


def test_taker_fallback_when_policy_allows_and_signal_fresh():
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=OrderState(side="buy", price=100.0, age_s=20.0),
        policy=MakerPolicy(timeout_s=15.0, taker_fallback=True, max_signal_age_s=60.0),
        signal_age_s=25.0,
    )
    assert decision.action == MakerAction.TAKER_FALLBACK


def test_no_taker_when_signal_too_stale():
    """Critical invariant: never escalate to taker on a stale signal —
    you'd be eating the leader's adverse selection."""
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=OrderState(side="buy", price=100.0, age_s=20.0),
        policy=MakerPolicy(timeout_s=15.0, taker_fallback=True, max_signal_age_s=60.0),
        signal_age_s=120.0,
    )
    assert decision.action == MakerAction.ABANDON
    assert "signal_stale" in decision.reason


# ── Hard abandon ──────────────────────────────────────────────────


def test_hard_abandon_after_timeout():
    decision = decide(
        side="buy",
        book=BookState(bid=100.0, ask=100.5),
        order=OrderState(side="buy", price=100.0, age_s=120.0),
        policy=MakerPolicy(timeout_s=15.0, abandon_after_s=60.0),
        signal_age_s=30.0,
    )
    assert decision.action == MakerAction.ABANDON
    assert "abandon_after" in decision.reason


# ── Malformed inputs ──────────────────────────────────────────────


def test_unknown_side_aborts():
    decision = decide(
        side="diagonal",
        book=BookState(bid=100.0, ask=100.5),
        order=None,
        policy=MakerPolicy(),
        signal_age_s=2.0,
    )
    assert decision.action == MakerAction.ABANDON
    assert "unknown_side" in decision.reason


def test_invalid_book_aborts():
    """Zero/negative book values produce ABANDON, not crash."""
    decision = decide(
        side="buy",
        book=BookState(bid=0.0, ask=0.0),
        order=None,
        policy=MakerPolicy(),
        signal_age_s=2.0,
    )
    assert decision.action == MakerAction.ABANDON
    assert "invalid_book" in decision.reason


# ── Per-source policy defaults ────────────────────────────────────


def test_copy_trade_policy_is_maker_only_and_patient():
    p = policy_for_source("copy_trade:0xabc")
    assert p.taker_fallback is False
    assert p.timeout_s >= 20.0
    # Should refuse taker even when signal is reasonably fresh — that's
    # the whole point of the patient policy for copy_trade.


def test_funding_carry_policy_is_tight_and_taker_ok():
    p = policy_for_source("funding_carry/BTC/hl_binance")
    assert p.taker_fallback is True
    assert p.timeout_s <= 15.0


def test_unknown_source_uses_default_policy():
    p = policy_for_source("some_unknown:xyz")
    assert isinstance(p, MakerPolicy)
    assert p.taker_fallback is True


def test_empty_source_key_uses_default():
    p = policy_for_source("")
    assert isinstance(p, MakerPolicy)


# ── BookState convenience ────────────────────────────────────────


def test_book_mid_and_spread():
    b = BookState(bid=100.0, ask=100.5)
    assert b.mid == pytest.approx(100.25)
    assert b.spread == pytest.approx(0.5)


def test_book_spread_never_negative():
    """Inverted book (shouldn't happen but defensive) → spread=0."""
    b = BookState(bid=100.5, ask=100.0)
    assert b.spread == 0.0


# ── End-to-end: copy_trade signal lifecycle ──────────────────────


def test_copy_trade_lifecycle_does_not_take_liquidity():
    """A full copy_trade signal: post → wait → BBO moves → repost →
    signal grows stale → ABANDON. Never taker."""
    p = policy_for_source("copy_trade:0xabc")
    actions: list[MakerAction] = []

    # T=0: initial post
    d = decide(side="buy", book=BookState(bid=100.0, ask=100.5),
               order=None, policy=p, signal_age_s=1.0)
    actions.append(d.action)

    # T=10: order still live, BBO unchanged → HOLD
    d = decide(side="buy", book=BookState(bid=100.0, ask=100.5),
               order=OrderState(side="buy", price=100.0, age_s=10.0),
               policy=p, signal_age_s=11.0)
    actions.append(d.action)

    # T=20: BBO drifted up 5 bps (above threshold of 3) → REPOST
    d = decide(side="buy", book=BookState(bid=100.05, ask=100.55),
               order=OrderState(side="buy", price=100.0, age_s=20.0),
               policy=p, signal_age_s=21.0)
    actions.append(d.action)

    # T=50: signal is now > 45s old (max_signal_age) → ABANDON
    d = decide(side="buy", book=BookState(bid=100.0, ask=100.5),
               order=OrderState(side="buy", price=100.05, age_s=35.0),
               policy=p, signal_age_s=55.0)
    actions.append(d.action)

    assert actions == [
        MakerAction.POST_ALO,
        MakerAction.HOLD,
        MakerAction.REPOST_AT_BBO,
        MakerAction.ABANDON,
    ]
    # Critical: never TAKER_FALLBACK on copy_trade
    assert MakerAction.TAKER_FALLBACK not in actions
