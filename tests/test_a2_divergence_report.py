"""Tests for the A2 Thompson divergence report.

DB-free; replay logic is tested with hand-built TradeRow lists so the
math is deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from a2_thompson_divergence_report import (  # noqa: E402
    TradeRow,
    build_report,
    render,
    replay,
)


def _make_trades(source: str, wins: int, losses: int) -> list[TradeRow]:
    """Manufacture a chronological win/loss sequence for one source."""
    out: list[TradeRow] = []
    ts = 1_700_000_000.0
    for _ in range(wins):
        out.append(TradeRow(closed_at_ts=ts, source_key=source, pnl=10.0))
        ts += 3600.0
    for _ in range(losses):
        out.append(TradeRow(closed_at_ts=ts, source_key=source, pnl=-10.0))
        ts += 3600.0
    return out


def test_replay_aggregates_counts():
    trades = _make_trades("copy_trade:0xA", wins=3, losses=2)
    alloc, counts = replay(trades)
    assert counts["copy_trade:0xA"] == (5, 3)
    arm = alloc._arms["copy_trade:0xA"]
    # Wins -> alpha increment; losses -> beta increment. Plus prior(1,1).
    # No decay (all updates on the same trading day) -> alpha~4, beta~3
    assert arm.alpha == pytest.approx(4.0, rel=1e-2)
    assert arm.beta == pytest.approx(3.0, rel=1e-2)


def test_build_report_divergence_when_static_disagrees():
    """Source with high win rate (3 wins, 1 loss) should have Thompson μ
    around 0.75 — if static_weight is 0.30, divergence ~ 0.45."""
    trades = _make_trades("copy_trade:0xB", wins=3, losses=1)
    static = {"copy_trade:0xB": 0.30}
    reports = build_report(trades, static)
    assert len(reports) == 1
    r = reports[0]
    assert r.n_trades == 4
    assert r.wins == 3
    assert r.win_rate == pytest.approx(0.75)
    # Beta(4, 2) mean = 4/6 ≈ 0.667
    assert r.thompson_mean == pytest.approx(0.667, abs=0.01)
    assert r.divergence == pytest.approx(abs(0.667 - 0.30), abs=0.01)


def test_build_report_treats_unknown_source_as_neutral():
    """A source with no entry in static_weights gets the 0.5 neutral
    baseline, so divergence reflects only the bandit's posterior."""
    trades = _make_trades("strategy:momentum", wins=5, losses=0)
    static: dict[str, float] = {}
    reports = build_report(trades, static)
    r = reports[0]
    # Beta(6, 1) mean = 6/7 ≈ 0.857 vs static=0.5 → divergence ≈ 0.357
    assert r.thompson_mean == pytest.approx(0.857, abs=0.01)
    assert r.static_weight == pytest.approx(0.5)
    assert r.divergence == pytest.approx(0.357, abs=0.01)


def test_build_report_sorts_by_divergence_desc():
    trades_a = _make_trades("source_a", wins=10, losses=0)
    trades_b = _make_trades("source_b", wins=5, losses=5)
    reports = build_report(
        trades_a + trades_b,
        {"source_a": 0.5, "source_b": 0.5},
    )
    # source_a: μ near 1.0, divergence near 0.5
    # source_b: μ near 0.5, divergence near 0.0
    assert reports[0].source_key == "source_a"
    assert reports[1].source_key == "source_b"
    assert reports[0].divergence > reports[1].divergence


def test_wilson_lower_is_conservative_vs_posterior_mean():
    """Wilson lower bound should be < posterior mean for any source
    with finite N. (Equal in the degenerate prior-only case.)"""
    trades = _make_trades("low_n", wins=2, losses=0)
    reports = build_report(trades, {"low_n": 0.5})
    r = reports[0]
    assert r.thompson_wilson < r.thompson_mean


def test_render_table_shape():
    trades = _make_trades("copy_trade:X", wins=3, losses=1)
    reports = build_report(trades, {"copy_trade:X": 0.40})
    md = render(reports)
    assert "A2 — Thompson vs. static AgentScorer divergence" in md
    # Header row + separator + 1 data row
    table_rows = [line for line in md.splitlines() if line.startswith("|")]
    assert len(table_rows) == 3
    # The source key appears in the data row
    assert "copy_trade:X" in md


def test_render_empty_input():
    md = render([])
    assert "0 sources" in md
