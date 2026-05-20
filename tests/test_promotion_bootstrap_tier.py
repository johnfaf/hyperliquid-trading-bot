"""Bootstrap-tier promotion path (Option B; default OFF).

The bootstrap tier provides an *alternative* promotion path when the
standard 30-trades / 45%-WR gate rejects a source.  It requires a
smaller sample at a HIGHER accuracy bar (defaults: 5 trades, 60% WR)
and returns ``ok_bootstrap:scale=X`` so the live-mirror path can
apply a fractional size scale (default 0.25×).

Default behaviour MUST be byte-identical to the legacy gate: the
bootstrap path is only consulted when ``PROMOTION_BOOTSTRAP_TIER_ENABLED``
is True.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

import config
import src.learning.promotion_gate as pg
from src.learning.promotion_gate import (
    get_bootstrap_scale,
    is_live_promotable,
)


# ── Helpers ──────────────────────────────────────────────────


def _agent_row(total_signals: int, accuracy: float):
    """Return a fake agent_scores row matching the gate's SELECT."""
    correct = int(total_signals * accuracy)
    return {
        "total_signals": int(total_signals),
        "correct_signals": correct,
        "accuracy": float(accuracy),
    }


def _patch_agent_scores(monkeypatch, row):
    """Patch the agent_scores DB lookup to return ``row`` (or None)."""
    @contextmanager
    def _conn(*a, **k):
        class _C:
            def execute(self, *_a, **_k):
                class _Cur:
                    def fetchone(_self):
                        return row
                return _Cur()
        yield _C()
    monkeypatch.setattr(pg.db, "get_connection", _conn)


@pytest.fixture(autouse=True)
def _gate_defaults(monkeypatch):
    """Force gate ON + canonical thresholds, bootstrap tier OFF by default."""
    monkeypatch.setattr(config, "LIVE_PROMOTION_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LIVE_PROMOTION_MIN_TRADES", 30, raising=False)
    monkeypatch.setattr(config, "LIVE_PROMOTION_MIN_WIN_RATE", 0.45, raising=False)
    monkeypatch.setattr(config, "LIVE_PROMOTION_MIN_SCORE", 0.20, raising=False)
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", False, raising=False,
    )
    monkeypatch.setattr(config, "PROMOTION_BOOTSTRAP_MIN_TRADES", 5, raising=False)
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_MIN_WIN_RATE", 0.60, raising=False,
    )
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_SIZE_SCALE", 0.25, raising=False,
    )


_TRADE_COPY = {
    "coin": "ETH",
    "source": "copy_trade",
    "source_trader": "0xABCDEF" + "0" * 34,
}


# ── Default-off (bootstrap tier disabled) ────────────────────


def test_bootstrap_disabled_blocks_thin_history(monkeypatch):
    """Flag OFF: 4 trades @ 75% WR -> still blocked under standard gate."""
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=4, accuracy=0.75))
    ok, reason = is_live_promotable(_TRADE_COPY)
    assert ok is False
    # Reason from the standard tier, NOT bootstrap.
    assert reason.startswith("insufficient_signals")
    assert "bootstrap" not in reason


def test_bootstrap_disabled_keeps_standard_pass(monkeypatch):
    """Flag OFF: 40 trades @ 50% WR -> standard pass, no bootstrap suffix."""
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=40, accuracy=0.50))
    ok, reason = is_live_promotable(_TRADE_COPY)
    assert ok is True
    assert reason == "ok"


# ── Bootstrap tier enabled ───────────────────────────────────


def test_bootstrap_enabled_promotes_thin_high_accuracy(monkeypatch):
    """Flag ON: 5 trades @ 80% WR -> bootstrap PASS with scale=0.25."""
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", True, raising=False,
    )
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=5, accuracy=0.80))
    ok, reason = is_live_promotable(_TRADE_COPY)
    assert ok is True
    assert reason.startswith("ok_bootstrap:scale=")
    assert get_bootstrap_scale(reason) == pytest.approx(0.25)


def test_bootstrap_enabled_rejects_below_bootstrap_min(monkeypatch):
    """Flag ON: 3 trades is below bootstrap min (5) -> still blocked."""
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", True, raising=False,
    )
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=3, accuracy=0.99))
    ok, reason = is_live_promotable(_TRADE_COPY)
    assert ok is False
    assert "bootstrap" in reason


def test_bootstrap_enabled_rejects_low_bootstrap_accuracy(monkeypatch):
    """Flag ON: 10 trades @ 50% WR -> bootstrap requires 60%, still blocked."""
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", True, raising=False,
    )
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=10, accuracy=0.50))
    ok, reason = is_live_promotable(_TRADE_COPY)
    assert ok is False
    assert reason.startswith("bootstrap_accuracy_too_low")


def test_bootstrap_does_not_override_standard_pass(monkeypatch):
    """Flag ON: source already meets standard tier -> standard reason wins.

    We never want to silently downgrade a fully-qualified source to the
    smaller bootstrap scale.
    """
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", True, raising=False,
    )
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=50, accuracy=0.55))
    ok, reason = is_live_promotable(_TRADE_COPY)
    assert ok is True
    assert reason == "ok"
    assert get_bootstrap_scale(reason) == 1.0


def test_bootstrap_does_not_retry_on_lookup_failure(monkeypatch):
    """Bootstrap tier must NOT retry when the agent_scores lookup failed.

    A transient DB error returning ``agent_score_lookup_failed`` should
    propagate as-is; promoting blindly on a query error is exactly the
    failure mode the gate is designed to prevent.
    """
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", True, raising=False,
    )

    @contextmanager
    def _boom(*a, **k):
        raise RuntimeError("db unavailable")
        yield  # pragma: no cover
    monkeypatch.setattr(pg.db, "get_connection", _boom)

    ok, reason = is_live_promotable(_TRADE_COPY)
    assert ok is False
    assert reason == "agent_score_lookup_failed"


# ── get_bootstrap_scale parser ───────────────────────────────


@pytest.mark.parametrize("reason,expected", [
    ("ok", 1.0),
    ("ok_dsr:dsr=2.30,p=0.003", 1.0),
    ("ok_bootstrap:scale=0.25", 0.25),
    ("ok_bootstrap:scale=0.5", 0.5),
    ("ok_bootstrap:scale=1.0", 1.0),
    # Defensive parser behaviour: garbage / out-of-range -> full size (1.0)
    # since the bootstrap path is ALREADY validated upstream in
    # _bootstrap_agent_score_ok; we never want to mute a real source
    # because of a downstream parse error.
    ("ok_bootstrap:scale=bogus", 1.0),
    ("ok_bootstrap:scale=-0.5", 1.0),
    ("ok_bootstrap:scale=2.0", 1.0),
    ("ok_bootstrap:scale=0.4,extra=foo", 0.4),
])
def test_get_bootstrap_scale_parsing(reason, expected):
    assert get_bootstrap_scale(reason) == pytest.approx(expected)


def test_get_bootstrap_scale_handles_non_string():
    assert get_bootstrap_scale(None) == 1.0  # type: ignore[arg-type]
    assert get_bootstrap_scale(42) == 1.0  # type: ignore[arg-type]


# ── Strategy_type path also uses bootstrap ───────────────────


def test_bootstrap_tier_also_applies_to_strategy_type_path(monkeypatch):
    """Path 3 (strategy:<type>) gets the same bootstrap fallback as Path 2."""
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", True, raising=False,
    )
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=6, accuracy=0.83))

    trade = {
        "coin": "BTC",
        "strategy_type": "momentum_short",
    }
    ok, reason = is_live_promotable(trade)
    assert ok is True
    assert reason.startswith("ok_bootstrap:scale=")
    assert get_bootstrap_scale(reason) == pytest.approx(0.25)


# ── Live-mirror integration (size-scale wiring) ──────────────


def test_live_execution_applies_bootstrap_scale(monkeypatch):
    """End-to-end: when promotion returns bootstrap, the rescale multiplies
    the live size by the bootstrap scale factor.

    Builds a minimal trade + stub trader so ``_rescale_size_for_live``
    runs without touching real DBs/exchange APIs.  Then checks that
    enabling the bootstrap tier shrinks the resulting live size by the
    bootstrap_scale factor relative to a standard-tier promotion.
    """
    from src.core import live_execution

    # Stub paper account + open paper margin so the rescale has clean
    # numbers: $10k paper balance, $0 margin used -> free = $10k.
    monkeypatch.setattr(
        live_execution.db, "get_paper_account",
        lambda: {"balance": 10_000.0},
    )
    monkeypatch.setattr(live_execution, "_paper_open_margin_used", lambda: 0.0)
    monkeypatch.setattr(live_execution, "get_all_mids", lambda: {"ETH": 2000.0})

    class _StubTrader:
        # Cap high enough that NEITHER full nor bootstrap hits it for the
        # test trade below.  We want to measure the bootstrap multiplier
        # in isolation; cap interaction is exercised separately.
        max_order_usd = 100_000.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 10_000.0  # live equity matches paper for a 1.0x base scale

        def get_free_margin(self):
            return 10_000.0  # free = equity for the test

    trader = _StubTrader()

    # Trade: 0.25 ETH @ $2000 = $500 notional.  Well above $11 floor and
    # well below the $100k cap, so the bootstrap scale measures cleanly.
    trade = {
        "coin": "ETH",
        "side": "short",
        "size": 0.25,
        "entry_price": 2000.0,
        "leverage": 1.0,
        "source": "copy_trade",
        "source_trader": "0xABCDEF" + "0" * 34,
    }

    # Standard tier passes -> full 1.0x proportional size.
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=40, accuracy=0.50))
    full = live_execution._rescale_size_for_live(dict(trade), trader)
    assert full is not None
    full_size = float(full["size"])

    # Bootstrap tier: 5 trades @ 80% -> ok_bootstrap:scale=0.25
    monkeypatch.setattr(
        config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", True, raising=False,
    )
    _patch_agent_scores(monkeypatch, _agent_row(total_signals=5, accuracy=0.80))
    boot = live_execution._rescale_size_for_live(dict(trade), trader)
    assert boot is not None
    boot_size = float(boot["size"])

    # Neither tier hits cap or floor: bootstrap should be exactly 0.25x full.
    assert boot_size < full_size, (
        f"bootstrap_size ({boot_size}) should be smaller than full ({full_size})"
    )
    ratio = boot_size / full_size
    assert ratio == pytest.approx(0.25, rel=0.01), (
        f"bootstrap_size / full_size = {ratio:.4f}, expected ~0.25"
    )
