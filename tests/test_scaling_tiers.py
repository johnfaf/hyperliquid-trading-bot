"""Tests for the live-trading scaling tier ladder.

Covers the safety invariants that matter for capital protection:

  * downward-only clamping on every cap
  * T4 = no-op override (operator config wins)
  * name resolution aliases (T0/t0/canary/CANARY)
  * min_order_usd floor respected
  * bogus names don't mutate caps silently
"""
from __future__ import annotations

import pytest

from src.trading.scaling_tiers import (
    apply_tier_caps,
    available_tiers,
    resolve_tier,
    summarize_tier,
)


# ─── Name resolution ───────────────────────────────────────────

@pytest.mark.parametrize("name, expected", [
    ("T0", "T0"),
    ("t0", "T0"),
    ("T4", "T4"),
    ("canary", "T0"),
    ("CANARY", "T0"),
    ("seed", "T1"),
    ("growth", "T2"),
    ("scale", "T3"),
    ("full", "T4"),
    ("  T2  ", "T2"),  # whitespace tolerated
])
def test_resolve_tier_names(name, expected):
    t = resolve_tier(name)
    assert t is not None
    assert t.name == expected


@pytest.mark.parametrize("name", ["", None, "bogus", "T99", "XYZ"])
def test_resolve_tier_invalid_returns_none(name):
    assert resolve_tier(name) is None


def test_available_tiers_snapshot():
    tiers = available_tiers()
    # Ladder must cover canary through full.
    assert set(tiers.keys()) == {"T0", "T1", "T2", "T3", "T4"}
    # T4 = no cap override (sentinel values).
    assert tiers["T4"].max_order_usd is None
    assert tiers["T4"].max_daily_loss_usd is None
    assert tiers["T4"].max_position_size_usd is None
    assert tiers["T4"].max_signals_per_day is None
    # Order caps must be monotonic non-decreasing up the ladder.
    caps = [tiers[k].max_order_usd for k in ("T0", "T1", "T2", "T3")]
    assert caps == sorted(caps), "Tier order caps must be non-decreasing"


# ─── Downward-only clamping ───────────────────────────────────

def test_tier_tightens_oversized_env():
    t0 = resolve_tier("T0")
    caps = apply_tier_caps(
        t0,
        max_order_usd=1000.0,
        max_daily_loss=500.0,
        max_position_size=2000.0,
        max_signals_per_day=0,
        min_order_usd=11.0,
    )
    assert caps["max_order_usd"] == 25.0
    assert caps["max_daily_loss"] == 50.0
    assert caps["max_position_size"] == 50.0
    assert caps["max_signals_per_day"] == 25


def test_tier_never_expands_operator_caps():
    """Tier can only tighten, never expand.  If operator set a $15 cap,
    T2's $100 default must NOT raise it."""
    t2 = resolve_tier("T2")
    caps = apply_tier_caps(
        t2,
        max_order_usd=15.0,
        max_daily_loss=30.0,
        max_position_size=20.0,
        max_signals_per_day=10,
        min_order_usd=11.0,
    )
    assert caps["max_order_usd"] == 15.0
    assert caps["max_daily_loss"] == 30.0
    assert caps["max_position_size"] == 20.0
    assert caps["max_signals_per_day"] == 10


def test_t4_is_passthrough():
    t4 = resolve_tier("T4")
    caps = apply_tier_caps(
        t4,
        max_order_usd=500.0,
        max_daily_loss=2000.0,
        max_position_size=5000.0,
        max_signals_per_day=0,
        min_order_usd=11.0,
    )
    assert caps["max_order_usd"] == 500.0
    assert caps["max_daily_loss"] == 2000.0
    assert caps["max_position_size"] == 5000.0
    assert caps["max_signals_per_day"] == 0  # unlimited stays unlimited


def test_min_order_usd_floor_respected():
    """Tier cannot push per-order cap below exchange minimum."""
    t0 = resolve_tier("T0")  # T0 caps order at $25
    # Operator floor is $30 — exchange rule trumps tier ambition.
    caps = apply_tier_caps(
        t0,
        max_order_usd=1000.0,
        max_daily_loss=500.0,
        max_position_size=2000.0,
        max_signals_per_day=0,
        min_order_usd=30.0,  # ← higher than T0's $25 cap
    )
    # Effective = max(min_order_usd, tier.max_order_usd) then min with operator env
    assert caps["max_order_usd"] == 30.0


def test_signals_per_day_takes_tier_when_operator_has_none():
    """Operator unset (=0) + tier caps → tier cap wins."""
    t0 = resolve_tier("T0")
    caps = apply_tier_caps(
        t0,
        max_order_usd=1000.0,
        max_daily_loss=500.0,
        max_position_size=2000.0,
        max_signals_per_day=0,  # operator disabled
        min_order_usd=11.0,
    )
    assert caps["max_signals_per_day"] == 25  # T0 default


# ─── Summary / JSON shape ─────────────────────────────────────

def test_summarize_tier_json_serializable():
    import json
    for name in ("T0", "T1", "T2", "T3", "T4"):
        summary = summarize_tier(resolve_tier(name))
        assert summary["name"] == name
        # Must round-trip through JSON (no dataclass/Decimal surprises).
        json.dumps(summary)
