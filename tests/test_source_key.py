"""Tests for src.signals.source_key.

The whole point of this module is to prevent the historical
address-truncation bug from recurring. Tests are organised around
that goal: every truncation shape MUST be rejected or coerced to
the safe untagged form.
"""
from __future__ import annotations

import pytest

from src.signals.source_key import (
    COPY_TRADE_PREFIX,
    COPY_TRADE_UNTAGGED,
    copy_trade_source_key,
    is_canonical_source_key,
    is_truncated_address,
    strategy_source_key,
)


# A canonical full ETH address (lower-case 0x + 40 hex)
FULL_ADDR = "0x1ee7a73c5be3b9c1ab1cd9f1d52a7e98b41a4d33"
# Some real-world truncated forms we've seen in DB audit
TRUNC_ADDRS = [
    "0x1ee7a73c",         # 10 chars (the historical [:10] bug)
    "0x1ee7a73c5",        # 11
    "0x1ee7a73c5be",      # 13
    "0x1ee7a73c5be3b9c1", # 18
    "0x",                 # bare prefix
    "0x1ee7a73c5be3b9c1ab1cd9f1d52a7e98b41a4d3",  # 41 (one char short)
]


# ── copy_trade_source_key ──────────────────────────────────────────


def test_full_address_builds_canonical_key():
    sk = copy_trade_source_key(FULL_ADDR)
    assert sk == f"{COPY_TRADE_PREFIX}:{FULL_ADDR}"


def test_full_address_is_lowercased():
    sk = copy_trade_source_key(FULL_ADDR.upper())
    assert sk == f"{COPY_TRADE_PREFIX}:{FULL_ADDR}"


def test_full_address_whitespace_stripped():
    sk = copy_trade_source_key(f"  {FULL_ADDR}\n")
    assert sk == f"{COPY_TRADE_PREFIX}:{FULL_ADDR}"


@pytest.mark.parametrize("trunc", TRUNC_ADDRS)
def test_truncated_address_non_strict_falls_back_to_untagged(trunc):
    """Non-strict: malformed addresses become the canonical untagged
    key, preserving fragmentation safety without crashing trade exec."""
    assert copy_trade_source_key(trunc) == COPY_TRADE_UNTAGGED


@pytest.mark.parametrize("trunc", TRUNC_ADDRS)
def test_truncated_address_strict_raises(trunc):
    """Strict: malformed addresses are loud failures, used at
    construction-time pre-conditions."""
    with pytest.raises(ValueError, match="not a full ETH address|invalid"):
        copy_trade_source_key(trunc, strict=True)


def test_empty_address_non_strict_returns_untagged():
    assert copy_trade_source_key("") == COPY_TRADE_UNTAGGED
    assert copy_trade_source_key(None) == COPY_TRADE_UNTAGGED  # type: ignore[arg-type]
    assert copy_trade_source_key("   ") == COPY_TRADE_UNTAGGED


def test_empty_address_strict_raises():
    with pytest.raises(ValueError, match="empty"):
        copy_trade_source_key("", strict=True)


def test_non_address_tag_passes_through():
    """A 'trader' identifier that isn't a 0x address (e.g. a name)
    is allowed but treated as a custom tag."""
    sk = copy_trade_source_key("alice_bot_v3")
    assert sk == f"{COPY_TRADE_PREFIX}:alice_bot_v3"


# ── strategy_source_key ────────────────────────────────────────────


def test_strategy_key_basic():
    assert strategy_source_key("momentum_long") == "strategy:momentum_long"


def test_strategy_key_lowercases_and_strips():
    assert strategy_source_key("  Momentum_Long  ") == "strategy:momentum_long"


def test_strategy_key_invalid_non_strict_returns_unknown():
    assert strategy_source_key("") == "strategy:unknown"
    assert strategy_source_key("invalid name with spaces") == "strategy:unknown"


def test_strategy_key_invalid_strict_raises():
    with pytest.raises(ValueError):
        strategy_source_key("", strict=True)


# ── is_canonical_source_key (CI structural test helper) ───────────


def test_full_canonical_keys_are_canonical():
    assert is_canonical_source_key(f"{COPY_TRADE_PREFIX}:{FULL_ADDR}")
    assert is_canonical_source_key("strategy:momentum_long")
    assert is_canonical_source_key(COPY_TRADE_UNTAGGED)
    assert is_canonical_source_key("funding_carry:btc/hyperliquid_binance")
    assert is_canonical_source_key("xgboost:base")


@pytest.mark.parametrize("trunc", TRUNC_ADDRS)
def test_truncated_addresses_not_canonical(trunc):
    """The whole point: a truncated address key MUST be detected as
    non-canonical, so audit scripts can flag it for cleanup."""
    bad_key = f"{COPY_TRADE_PREFIX}:{trunc}"
    assert not is_canonical_source_key(bad_key)


def test_empty_and_non_strings_not_canonical():
    assert not is_canonical_source_key("")
    assert not is_canonical_source_key(None)  # type: ignore[arg-type]
    assert not is_canonical_source_key(42)    # type: ignore[arg-type]


# ── is_truncated_address (audit predicate) ────────────────────────


@pytest.mark.parametrize("trunc", TRUNC_ADDRS)
def test_is_truncated_address_flags_truncation(trunc):
    assert is_truncated_address(trunc)
    assert is_truncated_address(f"{COPY_TRADE_PREFIX}:{trunc}")


def test_is_truncated_address_passes_full_address():
    assert not is_truncated_address(FULL_ADDR)
    assert not is_truncated_address(f"{COPY_TRADE_PREFIX}:{FULL_ADDR}")


def test_is_truncated_address_passes_non_address_keys():
    """A 'copy_trade:alice' or 'strategy:foo' shouldn't be flagged."""
    assert not is_truncated_address(f"{COPY_TRADE_PREFIX}:alice_bot")
    assert not is_truncated_address("strategy:momentum_long")
    assert not is_truncated_address(COPY_TRADE_UNTAGGED)


def test_is_truncated_address_handles_garbage_input():
    assert not is_truncated_address(None)  # type: ignore[arg-type]
    assert not is_truncated_address("")
    assert not is_truncated_address(42)    # type: ignore[arg-type]


# ── Property: builder output is canonical ─────────────────────────


def test_property_builder_output_always_canonical_for_valid_input():
    """For every valid input shape, the builder's output is canonical."""
    inputs = [FULL_ADDR, FULL_ADDR.upper(), f"  {FULL_ADDR}  ",
              "alice_bot_v3", "trader_v2", "x"]
    for addr in inputs:
        sk = copy_trade_source_key(addr)
        assert is_canonical_source_key(sk), f"{addr!r} -> {sk!r} not canonical"


def test_property_builder_output_canonical_for_truncated_input():
    """For every truncated input, the builder falls back to the
    canonical untagged key (which is itself canonical)."""
    for trunc in TRUNC_ADDRS:
        sk = copy_trade_source_key(trunc)
        assert is_canonical_source_key(sk)
        # And specifically: it's the safe fragmentation-free key
        assert sk == COPY_TRADE_UNTAGGED
