"""Tests for the A4/A6 shadow log analyzer.

The analyzer is the operator's main feedback loop once the shadow
flags are flipped in production, so the parser MUST be robust to
realistic log shape: timestamp prefixes, log levels, surrounding
fluff. Tests cover:
- Parsing both MAKER_SHADOW and CARRY_SHADOW lines with realistic
  log prefixes (timestamp, level, logger name).
- Garbage lines silently skipped (no crash).
- Aggregation: action histograms per source class, edge stats,
  veto-reason ranking.
- Render: markdown output contains expected sections.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_shadow_logs import (  # noqa: E402
    parse_carry_line,
    parse_maker_line,
    render,
    summarize,
)


# Realistic-shaped log lines as our app emits via the standard logger
SAMPLE_MAKER_LINES = [
    "2026-05-19 18:42:11 INFO src.trading.live_trader MAKER_SHADOW [BTC buy] "
    "src=copy_trade:0xabc age=2.3s mid=68421.123456 action=post_alo "
    "reason='initial_post' target=68420.83",
    "2026-05-19 18:42:14 INFO src.trading.live_trader MAKER_SHADOW [ETH sell] "
    "src=copy_trade:0xdef age=18.5s mid=3211.450000 action=abandon "
    "reason='signal_stale:18.5s>=max=15.0s' target=-",
    "2026-05-19 18:42:20 INFO src.trading.live_trader MAKER_SHADOW [SOL buy] "
    "src=alpha_arena age=4.0s mid=141.220000 action=taker_fallback "
    "reason='timeout_taker_ok:age=15.0s' target=141.21",
]

SAMPLE_CARRY_LINES = [
    "2026-05-19 18:43:01 INFO src.exchanges.cross_venue CARRY_SHADOW [BTC] "
    "hl↔binance: edge=12.34bps hold=4.0h actionable=True veto='' "
    "long=hyperliquid short=binance",
    "2026-05-19 18:43:01 INFO src.exchanges.cross_venue CARRY_SHADOW [ETH] "
    "hl↔bybit: edge=-3.20bps hold=4.0h actionable=False "
    "veto='below_min_edge' long=hyperliquid short=bybit",
    "2026-05-19 18:43:02 INFO src.exchanges.cross_venue CARRY_SHADOW [BTC] "
    "hl↔bybit: edge=1.50bps hold=4.0h actionable=False "
    "veto='basis_dominates' long=hyperliquid short=bybit",
]


# ── Parsing ─────────────────────────────────────────────────────────


def test_parse_maker_line_extracts_all_fields():
    ev = parse_maker_line(SAMPLE_MAKER_LINES[0])
    assert ev is not None
    assert ev.coin == "BTC"
    assert ev.side == "buy"
    assert ev.src == "copy_trade:0xabc"
    assert ev.age_s == pytest.approx(2.3)
    assert ev.mid == pytest.approx(68421.123456)
    assert ev.action == "post_alo"
    assert ev.reason == "initial_post"


def test_parse_maker_line_handles_stale_abandon():
    ev = parse_maker_line(SAMPLE_MAKER_LINES[1])
    assert ev is not None
    assert ev.action == "abandon"
    assert "signal_stale" in ev.reason
    assert ev.target == "-"  # no target when abandoning


def test_parse_carry_line_extracts_all_fields():
    ev = parse_carry_line(SAMPLE_CARRY_LINES[0])
    assert ev is not None
    assert ev.coin == "BTC"
    assert ev.cex == "binance"  # the colon is the regex separator, stripped from capture
    assert ev.edge_bps == pytest.approx(12.34)
    assert ev.hold_hours == pytest.approx(4.0)
    assert ev.actionable is True
    assert ev.veto == ""


def test_parse_carry_line_with_veto():
    ev = parse_carry_line(SAMPLE_CARRY_LINES[1])
    assert ev is not None
    assert ev.actionable is False
    assert ev.veto == "below_min_edge"
    assert ev.edge_bps == pytest.approx(-3.20)


def test_garbage_lines_return_none():
    assert parse_maker_line("random text with no fields") is None
    assert parse_carry_line("nothing carryshadow-related here") is None
    # Mid-line corruption is also rejected (defends against partial reads)
    assert parse_maker_line("MAKER_SHADOW [BTC b") is None


# ── Aggregation ─────────────────────────────────────────────────────


def test_summarize_classifies_sources_and_counts_actions():
    m, _ = summarize(SAMPLE_MAKER_LINES)
    assert m.n_events == 3
    # Two copy_trade events, one alpha_arena
    assert m.by_src_class["copy_trade"]["post_alo"] == 1
    assert m.by_src_class["copy_trade"]["abandon"] == 1
    assert m.by_src_class["alpha_arena"]["taker_fallback"] == 1
    # ABANDON / TAKER tallies
    assert m.abandons == 1
    assert m.taker_fallbacks == 1


def test_summarize_carry_edge_stats():
    _, c = summarize(SAMPLE_CARRY_LINES)
    assert c.n_events == 3
    assert c.actionable == 1
    assert c.vetoed == 2
    assert len(c.edge_when_actionable) == 1
    assert c.edge_when_actionable[0] == pytest.approx(12.34)


def test_summarize_carry_veto_reasons_ranked():
    extra = SAMPLE_CARRY_LINES + [SAMPLE_CARRY_LINES[1]]  # one more "below_min_edge"
    _, c = summarize(extra)
    # below_min_edge twice, basis_dominates once
    assert c.veto_reasons.most_common(1)[0] == ("below_min_edge", 2)


def test_summarize_silently_ignores_non_shadow_lines():
    mixed = SAMPLE_MAKER_LINES + [
        "2026-05-19 18:42:30 ERROR src.foo Something else entirely",
        "",
        "log line about an order placement that isn't shadow",
    ] + SAMPLE_CARRY_LINES
    m, c = summarize(mixed)
    assert m.n_events == 3
    assert c.n_events == 3


# ── Render ──────────────────────────────────────────────────────────


def test_render_contains_both_sections():
    m, c = summarize(SAMPLE_MAKER_LINES + SAMPLE_CARRY_LINES)
    md = render(m, c)
    assert "A6 — MAKER_SHADOW" in md
    assert "A4 — CARRY_SHADOW" in md
    # Action histogram table header
    assert "post_alo" in md
    assert "abandon" in md
    # Carry section: actionable rate is mentioned
    assert "Actionable:" in md


def test_render_handles_empty_input_gracefully():
    m, c = summarize([])
    md = render(m, c)
    assert "No MAKER_SHADOW events" in md
    assert "No CARRY_SHADOW events" in md
