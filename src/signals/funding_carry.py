"""A4: Funding-rate carry HL ↔ CEX — delta-neutral second alpha lane.

The copy-trade lane was showing 30% WR / -$27.93 net over 7 days
when this work began. A single-strategy bot is a single-source-of-
failure bot, and the existing infrastructure already supports the
opposite side of a delta-neutral trade: ``src/trading/cross_venue_
hedger.py``, the Binance/Bybit/Crypto.com adapters, and the funding
divergence cache.

This module identifies the *carry* opportunity (not the divergence
brake — that's already done in ``funding_divergence.py``): symbols
where Hyperliquid funding diverges from CEX funding by enough to
cover round-trip fees, slippage, and basis volatility for the hold
window, then proposes a delta-neutral leg pair (long the
underfunded venue, short the overfunded one).

The module is pure: it consumes funding snapshots and venue cost
parameters, returns ``CarryOpportunity`` dataclasses. It does NOT
place orders. Wiring into ``cross_venue_hedger`` and the execution
layer is a separate follow-up; this commit is the math + tests so
the strategy can soak in shadow mode against live funding data
before any capital is allocated.

Key design choices
------------------
- **Hard 4h max hold.** Hyperliquid pays funding hourly; CEX
  (Binance/Bybit) pays every 8h. The carry edge dominates on a
  *short* hold so basis blowouts have less time to bite. Initial
  ceiling: 4 hours. Tunable via ``FUNDING_CARRY_MAX_HOLD_HOURS``.
- **2σ basis stop-out.** If the spread moves against the position
  by more than 2× the recent basis standard deviation, close. The
  exit dominates the entry on this lane.
- **Round-trip cost guard.** ``expected_carry_bps`` MUST exceed
  ``min_edge_bps`` (default 8 bps) for a trade to be actionable.
  HL maker is 1.5 bps, taker is 4.5 bps; CEX makers 1-2 bps; total
  round-trip ~7-9 bps on majors. The default keeps us above noise.
- **Default OFF.** Wiring will be flag-gated via
  ``FUNDING_CARRY_ENABLED`` so no real money flows until backtest
  shows positive expectation across the funding-event population.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ── Constants (defaults; can be overridden by config wiring) ───────


# Funding payment intervals (per spec, not per implementation):
HL_FUNDING_INTERVAL_HOURS = 1
CEX_FUNDING_INTERVAL_HOURS = 8

# Round-trip fee budget the carry must clear before being actionable.
# 8 bps is a conservative pad covering ~3 bps HL maker + ~2 bps CEX
# maker + ~3 bps slippage on majors. Operators tighten this when the
# size grows or when running on more liquid pairs.
DEFAULT_MIN_EDGE_BPS = 8.0

# Maximum hold duration. The basis vol risk grows monotonically with
# hold time; capping at 4 hours covers four HL funding events and is
# well inside one CEX 8h cycle.
DEFAULT_MAX_HOLD_HOURS = 4.0

# 2σ adverse-move stop-out (in bps of spot).
DEFAULT_BASIS_STOPOUT_SIGMA = 2.0


# ── Datatypes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class FundingSnapshot:
    """Frozen-in-time view of one venue's funding rate for one symbol.

    All rates are *per the venue's native interval* (per-hour for HL,
    per-8h for Binance/Bybit). Normalisation to a common cadence is
    done by :func:`annualise_funding` so the carry math is unit-safe.
    """
    venue: str
    symbol: str
    rate_native: float            # rate at venue's native cadence
    interval_hours: float         # hours between funding payments
    timestamp: float = 0.0        # unix seconds; 0 = unknown


@dataclass(frozen=True)
class CarryOpportunity:
    """A proposed delta-neutral carry pair."""
    symbol: str
    long_venue: str               # venue to go long on (the underfunded one)
    short_venue: str              # venue to go short on (the overfunded one)
    long_rate_annualised: float
    short_rate_annualised: float
    expected_carry_bps_per_hour: float
    expected_carry_bps_for_hold: float
    hold_hours: float
    round_trip_cost_bps: float
    net_edge_bps: float           # expected_carry_bps_for_hold - round_trip_cost_bps
    is_actionable: bool           # True iff net_edge >= min_edge AND no veto
    veto_reason: str = ""         # "" if actionable; otherwise the rejection reason


# ── Pure helpers ──────────────────────────────────────────────────


def annualise_funding(rate_native: float, interval_hours: float) -> float:
    """Convert a per-interval funding rate into annualised %.

    Example: Binance pays funding every 8 hours. Rate of 0.0001 means
    1 bp per 8h. Annualised = 0.0001 * (8760 / 8) = 0.1095 = 10.95% APR.
    """
    if interval_hours <= 0:
        return 0.0
    intervals_per_year = 8760.0 / interval_hours
    return float(rate_native) * intervals_per_year


def per_hour_funding_bps(rate_native: float, interval_hours: float) -> float:
    """Express a funding rate as bps-per-hour for hold-window arithmetic."""
    if interval_hours <= 0:
        return 0.0
    return float(rate_native) * 10000.0 / interval_hours


# ── Core ──────────────────────────────────────────────────────────


def evaluate_carry(
    hl: FundingSnapshot,
    cex: FundingSnapshot,
    *,
    hold_hours: float = DEFAULT_MAX_HOLD_HOURS,
    round_trip_cost_bps: float = DEFAULT_MIN_EDGE_BPS,
    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS,
    basis_vol_bps_per_hour: Optional[float] = None,
    basis_stopout_sigma: float = DEFAULT_BASIS_STOPOUT_SIGMA,
) -> CarryOpportunity:
    """Decide whether the HL ↔ CEX funding spread is worth trading.

    Parameters
    ----------
    hl, cex
        :class:`FundingSnapshot` for the same symbol from two venues.
        ``hl.symbol == cex.symbol`` (else the function vetoes).
    hold_hours
        Intended max hold duration. The carry is integrated for this
        many hours assuming both venues pay at their native cadence.
    round_trip_cost_bps
        Conservative estimate of in + out trading cost (maker fees,
        slippage, half-spread) summed across both legs.
    min_edge_bps
        Minimum net edge needed to mark actionable. Acts as a hard
        floor so marginally-positive opportunities are vetoed.
    basis_vol_bps_per_hour
        Optional recent basis-volatility estimate (bps of spot per
        hour). If provided, the function checks that ``hold_hours *
        basis_stopout_sigma * vol`` is comparable to the expected
        carry; otherwise the trade is dominated by basis risk.

    Returns
    -------
    CarryOpportunity dataclass with the full breakdown. ``is_actionable``
    is True iff:
      1. symbols match,
      2. funding spread direction is consistent (one venue clearly
         under, the other clearly over),
      3. net_edge_bps >= min_edge_bps,
      4. (if basis_vol provided) carry > 2σ basis-vol-over-hold.
    """
    # 1. Symbol consistency
    if hl.symbol != cex.symbol:
        return CarryOpportunity(
            symbol=hl.symbol, long_venue="", short_venue="",
            long_rate_annualised=0.0, short_rate_annualised=0.0,
            expected_carry_bps_per_hour=0.0,
            expected_carry_bps_for_hold=0.0,
            hold_hours=hold_hours,
            round_trip_cost_bps=round_trip_cost_bps,
            net_edge_bps=0.0,
            is_actionable=False,
            veto_reason=f"symbol_mismatch:{hl.symbol}!={cex.symbol}",
        )

    hl_annual = annualise_funding(hl.rate_native, hl.interval_hours)
    cex_annual = annualise_funding(cex.rate_native, cex.interval_hours)

    # 2. Funding-rate spread direction:
    # Long the venue with *lower* (or more negative) funding (i.e. you
    # receive funding on the long leg if rate < 0, or pay less). Short
    # the venue with higher funding.
    if hl_annual == cex_annual:
        return CarryOpportunity(
            symbol=hl.symbol, long_venue="", short_venue="",
            long_rate_annualised=hl_annual, short_rate_annualised=cex_annual,
            expected_carry_bps_per_hour=0.0,
            expected_carry_bps_for_hold=0.0,
            hold_hours=hold_hours,
            round_trip_cost_bps=round_trip_cost_bps,
            net_edge_bps=-round_trip_cost_bps,
            is_actionable=False,
            veto_reason="no_funding_spread",
        )

    if hl_annual < cex_annual:
        long_venue, long_rate = hl.venue, hl
        short_venue, short_rate = cex.venue, cex
        long_annual, short_annual = hl_annual, cex_annual
    else:
        long_venue, long_rate = cex.venue, cex
        short_venue, short_rate = hl.venue, hl
        long_annual, short_annual = cex_annual, hl_annual

    # 3. Expected carry over the hold window. Long leg *receives*
    # short_annual - long_annual (delta-neutral). Convert to bps/hr.
    spread_annual = short_annual - long_annual              # positive
    spread_bps_per_hour = spread_annual * 10000.0 / 8760.0
    expected_carry_bps_for_hold = spread_bps_per_hour * hold_hours
    net_edge_bps = expected_carry_bps_for_hold - round_trip_cost_bps

    veto_reason = ""
    is_actionable = True
    if net_edge_bps < min_edge_bps:
        is_actionable = False
        veto_reason = (
            f"insufficient_edge:net={net_edge_bps:.2f}bps<min={min_edge_bps:.2f}bps"
        )

    # 4. Basis-volatility veto (optional but recommended)
    if is_actionable and basis_vol_bps_per_hour is not None:
        sigma_move_bps = basis_stopout_sigma * basis_vol_bps_per_hour * math.sqrt(hold_hours)
        if expected_carry_bps_for_hold < sigma_move_bps:
            is_actionable = False
            veto_reason = (
                f"basis_risk_dominates:carry={expected_carry_bps_for_hold:.2f}bps<"
                f"{basis_stopout_sigma}σ*vol*√hold={sigma_move_bps:.2f}bps"
            )

    return CarryOpportunity(
        symbol=hl.symbol,
        long_venue=long_venue,
        short_venue=short_venue,
        long_rate_annualised=long_annual,
        short_rate_annualised=short_annual,
        expected_carry_bps_per_hour=spread_bps_per_hour,
        expected_carry_bps_for_hold=expected_carry_bps_for_hold,
        hold_hours=hold_hours,
        round_trip_cost_bps=round_trip_cost_bps,
        net_edge_bps=net_edge_bps,
        is_actionable=is_actionable,
        veto_reason=veto_reason,
    )


def scan_for_carry_opportunities(
    hl_snapshots: List[FundingSnapshot],
    cex_snapshots: List[FundingSnapshot],
    **kwargs: Any,
) -> List[CarryOpportunity]:
    """Pair HL and CEX snapshots by symbol and evaluate each."""
    by_symbol_hl: Dict[str, FundingSnapshot] = {s.symbol: s for s in hl_snapshots}
    by_symbol_cex: Dict[str, FundingSnapshot] = {s.symbol: s for s in cex_snapshots}
    common = sorted(set(by_symbol_hl) & set(by_symbol_cex))
    return [
        evaluate_carry(by_symbol_hl[sym], by_symbol_cex[sym], **kwargs)
        for sym in common
    ]


def best_actionable(opportunities: List[CarryOpportunity]) -> Optional[CarryOpportunity]:
    """Return the highest-net-edge actionable opportunity, or None."""
    actionable = [o for o in opportunities if o.is_actionable]
    if not actionable:
        return None
    return max(actionable, key=lambda o: o.net_edge_bps)
