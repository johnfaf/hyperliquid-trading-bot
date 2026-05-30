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


def _bsig(source="copy_trade", side="short", strategy_type="", regime="", coin="BTC", confidence=0.65):
    return SimpleNamespace(
        coin=coin,
        side=SimpleNamespace(value=side),
        confidence=confidence,
        source=SimpleNamespace(value=source),
        strategy_type=strategy_type,
        regime=regime,
    )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    # Clear the module-level cooldown cache between tests.
    le._LAST_LIVE_MIRROR_TS.clear()
    for k in (
        "LIVE_MIRROR_MIN_CONFIDENCE",
        "LIVE_MIRROR_MIN_REENTRY_SECONDS",
        "LIVE_MIRROR_BUCKET_BLOCKLIST",
    ):
        monkeypatch.delenv(k, raising=False)
    yield
    le._LAST_LIVE_MIRROR_TS.clear()


# ── Edge-bucket blocklist ───────────────────────────────────────


def test_bucket_blocklist_off_by_default():
    allow, reason = le._live_mirror_conviction_gate(_bsig(source="copy_trade", side="short"))
    assert allow is True and reason == ""


def test_bucket_blocklist_blocks_source_side(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_BUCKET_BLOCKLIST", "copy_trade|short")
    allow, reason = le._live_mirror_conviction_gate(_bsig(source="copy_trade", side="short"))
    assert allow is False and "blocklist" in reason


def test_bucket_blocklist_namespace_prefix_matches_specific_trader(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_BUCKET_BLOCKLIST", "copy_trade|short")
    s = _bsig(source="copy_trade:0xabc123", side="short")
    assert le._live_mirror_conviction_gate(s)[0] is False


def test_bucket_blocklist_allows_other_side(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_BUCKET_BLOCKLIST", "copy_trade|short")
    assert le._live_mirror_conviction_gate(_bsig(source="copy_trade", side="long"))[0] is True


def test_bucket_blocklist_strategy_with_type(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_BUCKET_BLOCKLIST", "strategy:momentum_long|long")
    s = _bsig(source="strategy", strategy_type="momentum_long", side="long")
    assert le._live_mirror_conviction_gate(s)[0] is False
    # A different strategy type is unaffected.
    s2 = _bsig(source="strategy", strategy_type="scalping", side="long")
    assert le._live_mirror_conviction_gate(s2)[0] is True


def test_bucket_blocklist_regime_specific(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_BUCKET_BLOCKLIST", "copy_trade|short|bear")
    assert le._live_mirror_conviction_gate(_bsig(side="short", regime="bear"))[0] is False
    assert le._live_mirror_conviction_gate(_bsig(side="short", regime="bull"))[0] is True


def test_bucket_blocklist_multiple_entries(monkeypatch):
    monkeypatch.setenv("LIVE_MIRROR_BUCKET_BLOCKLIST", "copy_trade|short, strategy:momentum_long|long")
    assert le._live_mirror_conviction_gate(_bsig(source="copy_trade", side="short"))[0] is False
    s = _bsig(source="strategy", strategy_type="momentum_long", side="long")
    assert le._live_mirror_conviction_gate(s)[0] is False
    assert le._live_mirror_conviction_gate(_bsig(source="copy_trade", side="long"))[0] is True


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
