"""Tests for the A1 retro-sweep script.

Covers the pure functions (math/geometry/render). The DB load is
exercised separately by manual run against a real DB; here we drive
the sweep with hand-built `Trade` objects so the test is deterministic
and DB-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the script importable as a module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from a1_stop_floor_retro_sweep import (  # noqa: E402
    Trade,
    a1_stop_price,
    render_markdown,
    sweep,
    would_a1_stop_have_triggered,
)


def _make_trade(
    *,
    side: str = "long", entry: float = 100.0, exit_: float = 99.84,
    leverage: float = 25.0, atr_pct: float = 0.015, pnl: float = -1.0,
) -> Trade:
    return Trade(
        trade_id=1, coin="BTC", side=side,
        entry_price=entry, exit_price=exit_,
        leverage=leverage, pnl=pnl,
        close_reason="stop_loss_hit", atr_pct=atr_pct,
        stop_loss=entry * 0.9984 if side == "long" else entry * 1.0016,
    )


def test_a1_stop_price_widens_high_leverage_long():
    """Same setup as A1 unit test: base 4% ROE on 25x leverage is
    16 bps. With k=2.5 and ATR=1.5%, A1 floor = 3.75% — stop widens."""
    sl = a1_stop_price(
        entry_price=100.0, side="long", leverage=25.0,
        atr_pct=0.015, atr_mult=2.5, noise_floor_bps=50.0,
    )
    assert sl == pytest.approx(96.25, rel=1e-6)


def test_a1_stop_price_widens_high_leverage_short():
    sl = a1_stop_price(
        entry_price=100.0, side="short", leverage=25.0,
        atr_pct=0.015, atr_mult=2.5, noise_floor_bps=50.0,
    )
    assert sl == pytest.approx(103.75, rel=1e-6)


def test_would_a1_stop_have_triggered_long_noise_stop_avoided():
    """A trade that was stopped at -0.16% on a long would NOT have hit
    the A1 floor of -3.75%, so A1 saves it."""
    t = _make_trade(side="long", entry=100.0, exit_=99.84,
                    leverage=25.0, atr_pct=0.015, pnl=-1.50)
    assert would_a1_stop_have_triggered(
        t, atr_mult=2.5, noise_floor_bps=50.0,
    ) is False


def test_would_a1_stop_have_triggered_long_real_loss_not_avoided():
    """A trade that closed at -5% on a long DID hit a 3.75% A1 stop —
    A1 wouldn't have helped here; the trade was a real directional miss."""
    t = _make_trade(side="long", entry=100.0, exit_=95.00,
                    leverage=25.0, atr_pct=0.015, pnl=-12.50)
    assert would_a1_stop_have_triggered(
        t, atr_mult=2.5, noise_floor_bps=50.0,
    ) is True


def test_would_a1_stop_have_triggered_short_noise_avoided():
    t = _make_trade(side="short", entry=100.0, exit_=100.16,
                    leverage=25.0, atr_pct=0.015, pnl=-1.50)
    assert would_a1_stop_have_triggered(
        t, atr_mult=2.5, noise_floor_bps=50.0,
    ) is False


def test_sweep_shape_and_counts():
    """Sweep returns a complete grid with avoided counts that match
    the geometric truth."""
    trades = [
        _make_trade(exit_=99.84, pnl=-1.0),    # noise; widened stop misses
        _make_trade(exit_=99.84, pnl=-1.5),    # noise; widened stop misses
        _make_trade(exit_=95.00, pnl=-12.0),   # real loss; widened still hits
    ]
    result = sweep(trades)
    assert result["n_stopped_trades"] == 3
    # Loss in window
    assert result["total_loss_usd_in_window"] == pytest.approx(-14.5)
    # Grid completeness
    assert len(result["cells"]) == len(result["atr_mult_grid"]) * len(
        result["noise_floor_bps_grid"])
    # The (2.5, 50) cell should avoid the 2 noise trades, not the real loss
    cell_25_50 = next(
        c for c in result["cells"]
        if c["atr_mult"] == 2.5 and c["noise_floor_bps"] == 50.0
    )
    assert cell_25_50["trades_avoided"] == 2
    assert cell_25_50["loss_avoided_usd"] == pytest.approx(2.5, abs=1e-6)


def test_render_markdown_has_grid_shape():
    trades = [_make_trade(exit_=99.84, pnl=-1.0)]
    md = render_markdown(sweep(trades))
    assert "A1 retro-sweep" in md
    assert "atr_mult" in md
    # One header row + separator + one row per atr_mult value
    rows = [line for line in md.splitlines() if line.startswith("|")]
    # header + separator + len(grid)
    assert len(rows) >= 2 + 5  # 5 = len(ATR_MULT_GRID)


def test_tighter_floor_avoids_fewer_trades():
    """Sanity: tightening the noise floor should reduce the avoid count
    (or keep it the same), never increase it."""
    trades = [
        _make_trade(exit_=99.84, pnl=-1.0),
        _make_trade(exit_=99.90, pnl=-1.0),
    ]
    result = sweep(trades)
    cells_at_mult_2_5 = sorted(
        [c for c in result["cells"] if c["atr_mult"] == 2.5],
        key=lambda c: c["noise_floor_bps"],
    )
    counts = [c["trades_avoided"] for c in cells_at_mult_2_5]
    assert counts == sorted(counts), (
        "trades_avoided must be non-decreasing as noise_floor widens"
    )


def test_higher_multiplier_avoids_at_least_as_many():
    """Sanity: increasing atr_mult should not decrease the avoid count."""
    trades = [
        _make_trade(exit_=99.84, pnl=-1.0, atr_pct=0.015),
        _make_trade(exit_=99.92, pnl=-0.8, atr_pct=0.010),
    ]
    result = sweep(trades)
    by_mult = {}
    for c in result["cells"]:
        if c["noise_floor_bps"] == 50.0:
            by_mult[c["atr_mult"]] = c["trades_avoided"]
    mults_sorted = sorted(by_mult.keys())
    avoided_sequence = [by_mult[m] for m in mults_sorted]
    assert avoided_sequence == sorted(avoided_sequence), (
        "trades_avoided must be non-decreasing as atr_mult grows"
    )
