"""#1 Recent-side hardening deadlock escape.

The hardening lookback is count-based (last N closed trades). A blocked
side stops trading -> its window never refreshes -> the block is
PERMANENT (observed live: 0 trades in 6h, "Recent longs are
underperforming x35"). After a side is continuously blocked
recent_side_block_max_hours, the hard block downgrades to a reduced-size
probe so the sample can refresh.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import src.signals.decision_firewall as dfw
from src.signals.decision_firewall import DecisionFirewall

_CFG = {
    "enable_predictive_derisk": False,
    "funding_risk_enabled": False,
    "recent_side_block_max_hours": 24.0,
}


def _fw(**over):
    cfg = dict(_CFG)
    cfg.update(over)
    return DecisionFirewall(cfg)


def _clock(monkeypatch, t):
    monkeypatch.setattr(dfw.clock_provider, "unix_now", lambda: float(t))


def test_disabled_when_max_hours_zero(monkeypatch):
    fw = _fw(recent_side_block_max_hours=0.0)
    _clock(monkeypatch, 1_000_000.0)
    # legacy permanent block: never downgrades
    assert fw._recent_side_block_escape("long", "blocked") == "blocked"
    assert fw._recent_side_block_escape("long", "blocked") == "blocked"


def test_block_holds_until_cooldown_then_degrades(monkeypatch):
    fw = _fw(recent_side_block_max_hours=24.0)
    T = 1_000_000.0
    _clock(monkeypatch, T)
    assert fw._recent_side_block_escape("long", "blocked") == "blocked"  # arm timer
    _clock(monkeypatch, T + 3600 * 1)
    assert fw._recent_side_block_escape("long", "blocked") == "blocked"  # 1h < 24h
    _clock(monkeypatch, T + 3600 * 23.9)
    assert fw._recent_side_block_escape("long", "blocked") == "blocked"  # still < 24h
    _clock(monkeypatch, T + 3600 * 24)
    assert fw._recent_side_block_escape("long", "blocked") == "degraded"  # escape


def test_escape_resets_timer_one_probe_per_cooldown(monkeypatch):
    fw = _fw(recent_side_block_max_hours=10.0)
    T = 2_000_000.0
    _clock(monkeypatch, T)
    fw._recent_side_block_escape("long", "blocked")            # arm
    _clock(monkeypatch, T + 3600 * 10)
    assert fw._recent_side_block_escape("long", "blocked") == "degraded"  # 1st probe
    _clock(monkeypatch, T + 3600 * 11)
    assert fw._recent_side_block_escape("long", "blocked") == "blocked"   # cooldown restarted
    _clock(monkeypatch, T + 3600 * 20)
    assert fw._recent_side_block_escape("long", "blocked") == "degraded"  # 2nd probe ~1/cooldown


def test_non_blocked_status_clears_timer(monkeypatch):
    fw = _fw(recent_side_block_max_hours=5.0)
    T = 3_000_000.0
    _clock(monkeypatch, T)
    fw._recent_side_block_escape("long", "blocked")            # arm
    _clock(monkeypatch, T + 3600 * 2)
    assert fw._recent_side_block_escape("long", "healthy") == "healthy"   # clears timer
    # timer cleared -> a fresh block must arm again, not immediately escape
    _clock(monkeypatch, T + 3600 * 3)
    assert fw._recent_side_block_escape("long", "blocked") == "blocked"
    _clock(monkeypatch, T + 3600 * 9)  # only 6h since re-arm, < 5h? -> >5h -> degrade
    assert fw._recent_side_block_escape("long", "blocked") == "degraded"


def test_long_and_short_timers_are_independent(monkeypatch):
    fw = _fw(recent_side_block_max_hours=8.0)
    T = 4_000_000.0
    _clock(monkeypatch, T)
    fw._recent_side_block_escape("long", "blocked")   # arm long only
    _clock(monkeypatch, T + 3600 * 9)
    assert fw._recent_side_block_escape("long", "blocked") == "degraded"   # long escaped
    assert fw._recent_side_block_escape("short", "blocked") == "blocked"   # short just arming


def test_apply_long_hardening_breaks_the_deadlock(monkeypatch):
    """Integration: a permanently-blocked long side becomes a reduced
    probe after the cooldown instead of a hard reject forever."""
    fw = _fw(recent_side_block_max_hours=6.0)
    fw.long_hardening_enabled = True
    monkeypatch.setattr(
        fw, "_get_long_side_policy",
        lambda: {"status": "blocked",
                 "reason": "Recent longs are underperforming (12 trades, win rate 25%, net -41.58)",
                 "metrics": {}},
    )
    sig = MagicMock()
    sig.coin = "BTC"
    sig.confidence = 0.8
    sig.position_pct = 0.05
    sig.size = 0.0
    sig.context = {}
    T = 5_000_000.0
    _clock(monkeypatch, T)
    ok, reason = fw._apply_long_hardening(sig)
    assert ok is False and "underperforming" in reason  # still hard-blocked initially
    _clock(monkeypatch, T + 3600 * 6)
    ok, reason = fw._apply_long_hardening(sig)
    assert ok is True and reason == ""                   # escaped -> reduced probe
    assert sig.confidence < 0.8                           # de-risked, not full size
