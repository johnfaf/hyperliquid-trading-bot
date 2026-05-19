"""A3: tests for the runtime multiplier-cascade trace.

The trace is the *runtime* half of A3 — the linter test
(test_no_multiplier_cascade.py) catches structural cascades at CI time;
this catches them in live data via the dashboard panel that reads
`recent_traces()` / `cascade_stats_by_source()` / `detect_cascade()`.
"""
from __future__ import annotations

import pytest

from src.signals.multiplier_trace import (
    MultiplierEvent,
    cascade_stats_by_source,
    clear_traces,
    detect_cascade,
    recent_traces,
    record_multiplier,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_traces()
    yield
    clear_traces()


def test_record_and_recent_roundtrip():
    record_multiplier(
        signal_id="sig1", source="copy_trade:0xabc",
        stage="firewall.long_hardening",
        multiplier=0.80, confidence_before=0.90, confidence_after=0.72,
    )
    events = recent_traces()
    assert len(events) == 1
    ev = events[0]
    assert ev.signal_id == "sig1"
    assert ev.source == "copy_trade:0xabc"
    assert ev.stage == "firewall.long_hardening"
    assert ev.multiplier == pytest.approx(0.80)
    assert ev.confidence_before == pytest.approx(0.90)
    assert ev.confidence_after == pytest.approx(0.72)


def test_recent_limit():
    for i in range(20):
        record_multiplier("s", "src", f"stage{i}", 0.9, 0.5, 0.45)
    assert len(recent_traces(5)) == 5
    assert len(recent_traces(0)) == 0
    assert len(recent_traces()) == 20


def test_cascade_stats_aggregation():
    # Source A: collapses to 0.43 ± 0.005 (dead-band)
    for i in range(15):
        record_multiplier(
            f"sig_a{i}", "copy_trade:cascade",
            "firewall.synthetic_cap", 0.5, 0.85, 0.43 + 0.003 * ((i % 2) * 2 - 1),
        )
    # Source B: healthy spread
    record_multiplier("sig_b1", "alpha_arena", "filter", 0.9, 0.80, 0.72)
    record_multiplier("sig_b2", "alpha_arena", "filter", 0.9, 0.60, 0.54)
    record_multiplier("sig_b3", "alpha_arena", "filter", 0.9, 0.50, 0.45)

    stats = cascade_stats_by_source()
    assert "copy_trade:cascade" in stats
    assert "alpha_arena" in stats

    a = stats["copy_trade:cascade"]
    assert a.n == 15
    assert a.confidence_after_max - a.confidence_after_min < 0.01

    b = stats["alpha_arena"]
    assert b.n == 3
    assert b.confidence_after_max - b.confidence_after_min > 0.20


def test_detect_cascade_flags_dead_band_collapse():
    # 12 samples all collapsing into [0.428, 0.432]
    for i in range(12):
        offset = 0.002 * ((i % 2) * 2 - 1)
        record_multiplier(
            f"sig_{i}", "copy_trade:bad",
            "cascade.synthetic", 0.5, 0.90, 0.43 + offset,
        )
    stats = cascade_stats_by_source()
    flagged = detect_cascade(stats, min_samples=10, dead_band_width=0.01, floor=0.43)
    assert "copy_trade:bad" in flagged


def test_detect_cascade_does_not_flag_healthy_source():
    for i in range(12):
        # confidence_after spread from 0.30 to 0.85 — clearly healthy
        after = 0.30 + (i / 11.0) * 0.55
        record_multiplier(f"s{i}", "ok_source", "stage", 0.9, 0.95, after)
    flagged = detect_cascade(cascade_stats_by_source(), min_samples=10, dead_band_width=0.01, floor=0.45)
    assert flagged == []


def test_detect_cascade_requires_min_samples():
    # Collapsed but only 5 samples — not enough evidence yet
    for i in range(5):
        record_multiplier(f"s{i}", "early", "stage", 0.5, 0.90, 0.43)
    flagged = detect_cascade(cascade_stats_by_source(), min_samples=10, dead_band_width=0.01, floor=0.43)
    assert flagged == []


def test_disabled_via_env(monkeypatch):
    """When MULTIPLIER_TRACE_ENABLED=false, record_multiplier is a no-op."""
    monkeypatch.setattr("src.signals.multiplier_trace._TRACE_ENABLED", False)
    record_multiplier("sig", "src", "stage", 0.5, 0.9, 0.45)
    assert recent_traces() == []


def test_malformed_arguments_do_not_crash():
    """Trace must never raise — it's not on the critical path."""
    record_multiplier(None, None, None, "not-a-number", "garbage", None)  # type: ignore[arg-type]
    # We don't care what got recorded — only that the call didn't throw.
    assert isinstance(recent_traces(), list)
