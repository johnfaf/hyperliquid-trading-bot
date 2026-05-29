"""Tests for the live-only conviction + re-entry-cooldown gate.

Live-wallet CSV audit (May 2026): ~70% of the loss was fees from
over-trading (277 opens churned faster than they cleared the ~5bps
round-trip taker fee).  These gates throttle the LIVE mirror path
only -- paper/learning keeps running at full rate.  Both default OFF.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core import live_execution as le


def _sig(coin="BTC", side="long", confidence=0.65):
    return SimpleNamespace(
        coin=coin,
        side=SimpleNamespace(value=side),
        confidence=confidence,
    )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    # Clear the module-level cooldown cache between tests.
    le._LAST_LIVE_MIRROR_TS.clear()
    for k in ("LIVE_MIRROR_MIN_CONFIDENCE", "LIVE_MIRROR_MIN_REENTRY_SECONDS"):
        monkeypatch.delenv(k, raising=False)
    yield
    le._LAST_LIVE_MIRROR_TS.clear()


# ── Default OFF ─────────────────────────────────────────────────


def test_gate_off_by_default_allows_everything():
    allow, reason = le._live_mirror_conviction_gate(_sig(confidence=0.01))
    assert allow is True
    assert reason == ""


# ── Conviction floor ────────────────────────────────────────────


def test_conviction_floor_blocks_low_confidence(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_CONFIDENCE", "0.60")
    allow, reason = le._live_mirror_conviction_gate(_sig(confidence=0.45))
    assert allow is False
    assert "conviction" in reason


def test_conviction_floor_allows_high_confidence(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_CONFIDENCE", "0.60")
    allow, _ = le._live_mirror_conviction_gate(_sig(confidence=0.72))
    assert allow is True


def test_conviction_floor_clamped_to_unit_range(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_CONFIDENCE", "5")
    assert le._live_mirror_min_confidence() == 1.0
    monkeypatch.setenv("LIVE_MIRROR_MIN_CONFIDENCE", "-1")
    assert le._live_mirror_min_confidence() == 0.0


def test_conviction_floor_invalid_env_is_off(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_CONFIDENCE", "abc")
    assert le._live_mirror_min_confidence() == 0.0


# ── Re-entry cooldown ───────────────────────────────────────────


def test_reentry_cooldown_blocks_recent_coin(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_REENTRY_SECONDS", "600")
    # First mirror of BTC: allowed, and we record the timestamp.
    allow, _ = le._live_mirror_conviction_gate(_sig(coin="BTC"))
    assert allow is True
    le._mark_live_mirror_time(_sig(coin="BTC"))
    # Immediate re-entry of BTC: blocked by cooldown.
    allow, reason = le._live_mirror_conviction_gate(_sig(coin="BTC"))
    assert allow is False
    assert "cooldown" in reason


def test_reentry_cooldown_is_per_coin(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_REENTRY_SECONDS", "600")
    le._mark_live_mirror_time(_sig(coin="BTC"))
    # A DIFFERENT coin is not affected by BTC's cooldown.
    allow, _ = le._live_mirror_conviction_gate(_sig(coin="ETH"))
    assert allow is True


def test_reentry_cooldown_expires(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_REENTRY_SECONDS", "600")
    # Seed a timestamp 700s in the past -> cooldown elapsed.
    le._LAST_LIVE_MIRROR_TS["BTC"] = le._time.time() - 700
    allow, _ = le._live_mirror_conviction_gate(_sig(coin="BTC"))
    assert allow is True


# ── Combined ────────────────────────────────────────────────────


def test_both_gates_must_pass(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_MIN_CONFIDENCE", "0.60")
    monkeypatch.setenv("LIVE_MIRROR_MIN_REENTRY_SECONDS", "600")
    # High conviction, fresh coin -> allowed
    assert le._live_mirror_conviction_gate(_sig(coin="SOL", confidence=0.8))[0] is True
    # High conviction but in cooldown -> blocked
    le._mark_live_mirror_time(_sig(coin="SOL"))
    assert le._live_mirror_conviction_gate(_sig(coin="SOL", confidence=0.8))[0] is False
    # Fresh coin but low conviction -> blocked
    assert le._live_mirror_conviction_gate(_sig(coin="XRP", confidence=0.2))[0] is False
