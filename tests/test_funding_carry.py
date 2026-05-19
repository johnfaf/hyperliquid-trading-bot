"""A4: Funding-rate carry HL ↔ CEX tests."""
from __future__ import annotations

import pytest

from src.signals.funding_carry import (
    DEFAULT_BASIS_STOPOUT_SIGMA,
    DEFAULT_MAX_HOLD_HOURS,
    DEFAULT_MIN_EDGE_BPS,
    CarryOpportunity,
    FundingSnapshot,
    annualise_funding,
    best_actionable,
    evaluate_carry,
    per_hour_funding_bps,
    scan_for_carry_opportunities,
)


# ── Pure-helpers ──────────────────────────────────────────────────


def test_annualise_funding_known_values():
    # Binance 0.0001 every 8h = 0.0001 * 1095 = 10.95% APR
    assert annualise_funding(0.0001, 8.0) == pytest.approx(0.1095, rel=1e-3)
    # HL 0.00001 every 1h = 0.00001 * 8760 = 8.76% APR
    assert annualise_funding(0.00001, 1.0) == pytest.approx(0.0876, rel=1e-3)


def test_annualise_zero_interval_safe():
    assert annualise_funding(0.001, 0.0) == 0.0


def test_per_hour_bps_calculation():
    # 0.0001 per 8h = 0.0000125 per hour = 0.125 bps/hour
    assert per_hour_funding_bps(0.0001, 8.0) == pytest.approx(0.125, rel=1e-3)


# ── evaluate_carry ───────────────────────────────────────────────


def _snap(venue: str, symbol: str, rate: float, interval: float) -> FundingSnapshot:
    return FundingSnapshot(venue=venue, symbol=symbol, rate_native=rate, interval_hours=interval)


def test_symbol_mismatch_is_vetoed():
    hl = _snap("hl", "BTC", 0.0001, 1.0)
    cex = _snap("binance", "ETH", 0.0001, 8.0)
    op = evaluate_carry(hl, cex)
    assert op.is_actionable is False
    assert "symbol_mismatch" in op.veto_reason


def test_equal_funding_is_vetoed():
    """Same annualised rate on both venues → no carry."""
    hl = _snap("hl", "BTC", 0.00001, 1.0)        # ~8.76% APR
    cex = _snap("binance", "BTC", 0.00008, 8.0)  # ~8.76% APR
    op = evaluate_carry(hl, cex)
    assert op.is_actionable is False
    assert op.veto_reason == "no_funding_spread"


def test_negative_hl_positive_cex_long_hl_short_cex():
    """Classic carry: HL pays you to be long (rate < 0), CEX charges
    the short side. Long HL, short CEX.

    Uses a funding-squeeze magnitude (~ -0.05%/hr on HL) since realistic
    carry trades only clear the 8 bps round-trip cost in *extreme*
    funding regimes; baseline regimes have spreads measured in
    sub-bp/hour and need much longer hold windows."""
    hl = _snap("hl", "BTC", -0.0005, 1.0)         # -438% APR (squeeze)
    cex = _snap("binance", "BTC", 0.0005, 8.0)    # +54.75% APR
    op = evaluate_carry(hl, cex, hold_hours=4.0, round_trip_cost_bps=8.0)
    assert op.long_venue == "hl"
    assert op.short_venue == "binance"
    # Spread ≈ 5.6 bps/hour; over 4h -> ~22.5 bps; net ~14.5 bps.
    assert op.is_actionable is True
    assert op.net_edge_bps > 10


def test_below_min_edge_is_vetoed():
    """Tiny funding spread -> carry doesn't cover round-trip cost."""
    hl = _snap("hl", "BTC", 0.0000005, 1.0)   # 0.438% APR
    cex = _snap("binance", "BTC", 0.000005, 8.0)  # 0.547% APR
    op = evaluate_carry(hl, cex, hold_hours=4.0, round_trip_cost_bps=8.0)
    # spread ≈ 0.001 APR -> 0.00114 bps/hr -> ~0.005 bps for 4h hold
    assert op.is_actionable is False
    assert "insufficient_edge" in op.veto_reason


def test_basis_vol_veto_when_risk_dominates():
    """High basis volatility kills an otherwise-profitable trade."""
    hl = _snap("hl", "BTC", -0.0005, 1.0)
    cex = _snap("binance", "BTC", 0.0005, 8.0)
    # Carry ~22.5 bps over 4h. Vol 10 bps/hr -> 2σ*sqrt(4)*10 = 40 bps
    # adverse. Carry < adverse → veto.
    op = evaluate_carry(
        hl, cex,
        hold_hours=4.0, round_trip_cost_bps=8.0,
        basis_vol_bps_per_hour=10.0,
    )
    assert op.is_actionable is False
    assert "basis_risk_dominates" in op.veto_reason


def test_basis_vol_pass_when_carry_dominates():
    """High carry survives moderate basis vol."""
    hl = _snap("hl", "BTC", -0.002, 1.0)         # extreme funding squeeze
    cex = _snap("binance", "BTC", 0.002, 8.0)
    # Carry ~90 bps over 4h. Vol 10 bps/hr -> 2σ*sqrt(4)*10 = 40 bps.
    # Carry > adverse → still actionable.
    op = evaluate_carry(
        hl, cex,
        hold_hours=4.0, round_trip_cost_bps=8.0,
        basis_vol_bps_per_hour=10.0,
    )
    assert op.is_actionable is True


def test_round_trip_cost_subtracted():
    hl = _snap("hl", "BTC", -0.00002, 1.0)
    cex = _snap("binance", "BTC", 0.00002, 8.0)
    op_cheap = evaluate_carry(hl, cex, hold_hours=4.0, round_trip_cost_bps=4.0, min_edge_bps=0.5)
    op_pricey = evaluate_carry(hl, cex, hold_hours=4.0, round_trip_cost_bps=40.0, min_edge_bps=0.5)
    assert op_cheap.net_edge_bps > op_pricey.net_edge_bps
    assert op_cheap.expected_carry_bps_for_hold == pytest.approx(
        op_pricey.expected_carry_bps_for_hold, rel=1e-6
    )


# ── scan_for_carry_opportunities ─────────────────────────────────


def test_scan_pairs_by_symbol():
    hl_snaps = [
        _snap("hl", "BTC", -0.00005, 1.0),
        _snap("hl", "ETH", 0.00005, 1.0),
        _snap("hl", "SOL", -0.00001, 1.0),
    ]
    cex_snaps = [
        _snap("binance", "BTC", 0.0001, 8.0),
        _snap("binance", "ETH", -0.00005, 8.0),
        # SOL missing → should not be paired
    ]
    ops = scan_for_carry_opportunities(hl_snaps, cex_snaps,
                                       hold_hours=4.0, round_trip_cost_bps=8.0)
    assert len(ops) == 2
    symbols = {op.symbol for op in ops}
    assert symbols == {"BTC", "ETH"}


def test_scan_handles_empty_inputs():
    assert scan_for_carry_opportunities([], []) == []
    assert scan_for_carry_opportunities([_snap("hl", "BTC", 0.0, 1.0)], []) == []


def test_best_actionable_picks_max_edge():
    op_small = CarryOpportunity(
        symbol="BTC", long_venue="hl", short_venue="binance",
        long_rate_annualised=0.0, short_rate_annualised=0.1,
        expected_carry_bps_per_hour=1.0, expected_carry_bps_for_hold=20.0,
        hold_hours=4.0, round_trip_cost_bps=8.0, net_edge_bps=12.0,
        is_actionable=True, veto_reason="",
    )
    op_big = CarryOpportunity(
        symbol="ETH", long_venue="hl", short_venue="binance",
        long_rate_annualised=0.0, short_rate_annualised=0.2,
        expected_carry_bps_per_hour=2.0, expected_carry_bps_for_hold=80.0,
        hold_hours=4.0, round_trip_cost_bps=8.0, net_edge_bps=72.0,
        is_actionable=True, veto_reason="",
    )
    op_vetoed = CarryOpportunity(
        symbol="SOL", long_venue="", short_venue="",
        long_rate_annualised=0.0, short_rate_annualised=0.0,
        expected_carry_bps_per_hour=0.0, expected_carry_bps_for_hold=0.0,
        hold_hours=4.0, round_trip_cost_bps=8.0, net_edge_bps=0.0,
        is_actionable=False, veto_reason="no_funding_spread",
    )
    best = best_actionable([op_small, op_big, op_vetoed])
    assert best is not None
    assert best.symbol == "ETH"


def test_best_actionable_none_when_all_vetoed():
    op = CarryOpportunity(
        symbol="BTC", long_venue="", short_venue="",
        long_rate_annualised=0.0, short_rate_annualised=0.0,
        expected_carry_bps_per_hour=0.0, expected_carry_bps_for_hold=0.0,
        hold_hours=4.0, round_trip_cost_bps=8.0, net_edge_bps=-8.0,
        is_actionable=False, veto_reason="no_funding_spread",
    )
    assert best_actionable([op]) is None
    assert best_actionable([]) is None


def test_defaults_reasonable():
    """Sanity-check the module defaults are within the documented ranges."""
    assert 5.0 <= DEFAULT_MIN_EDGE_BPS <= 30.0
    assert 1.0 <= DEFAULT_MAX_HOLD_HOURS <= 12.0
    assert 1.5 <= DEFAULT_BASIS_STOPOUT_SIGMA <= 3.5


def test_hold_hours_scales_carry_linearly():
    """For a fixed funding spread, expected_carry_bps_for_hold must
    scale linearly with hold_hours (modulo rounding)."""
    hl = _snap("hl", "BTC", -0.00005, 1.0)
    cex = _snap("binance", "BTC", 0.00005, 8.0)
    op_1h = evaluate_carry(hl, cex, hold_hours=1.0, round_trip_cost_bps=0.0, min_edge_bps=-1000)
    op_4h = evaluate_carry(hl, cex, hold_hours=4.0, round_trip_cost_bps=0.0, min_edge_bps=-1000)
    assert op_4h.expected_carry_bps_for_hold == pytest.approx(
        4 * op_1h.expected_carry_bps_for_hold, rel=1e-6
    )
