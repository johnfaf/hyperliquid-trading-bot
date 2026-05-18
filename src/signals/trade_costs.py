"""Per-signal trade-cost estimation in basis points.

The EV gate needs a single source of truth for how much each candidate
trade is expected to cost so the "is this still profitable after costs"
question can actually be answered. This module reads the existing
paper-mode fee/slippage/funding configuration and pulls a live funding
rate from the Hyperliquid asset context.

Returned numbers are *one-way* fees/slippage doubled to round-trip:
entering and exiting both cost the spread. Funding is per-hour mid-
estimated cost from a holding-period assumption; callers can override
the assumed holding period if their strategy is faster.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)


def _fee_bps(role: str) -> float:
    role = str(role or "taker").lower()
    if role == "maker":
        return float(getattr(config, "PAPER_TRADING_MAKER_FEE_BPS", 0.2) or 0.0)
    return float(getattr(config, "PAPER_TRADING_TAKER_FEE_BPS", 2.5) or 0.0)


def _slippage_bps_estimate() -> float:
    """Return the per-leg slippage estimate the cost model should assume.

    Uses ``TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS`` when set, otherwise
    falls back to the paper-trader's max-slippage knob. Conservative
    midpoint: prefer over-estimating costs so the EV gate has a buffer.
    """
    explicit = getattr(config, "TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS", None)
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    return max(0.0, float(getattr(config, "PAPER_TRADING_SLIPPAGE_MAX_BPS", 5.0) or 0.0))


def _funding_bps_for_coin(coin: str, *, side: str, holding_hours: float) -> float:
    """Estimate funding cost over the assumed holding period, in bps.

    Hyperliquid funding ticks hourly. Positive funding paid by longs;
    negative by shorts. ``holding_hours`` defaults to the configured
    median holding period (24h is a generic default that's conservative
    for swing-style strategies).
    """
    try:
        from src.data.hyperliquid_client import get_asset_contexts

        contexts = get_asset_contexts() or {}
    except Exception as exc:
        logger.debug("trade_costs: asset context fetch failed: %s", exc)
        return 0.0
    coin_ctx = contexts.get(str(coin or "").upper(), {})
    try:
        hourly_rate = float(coin_ctx.get("funding", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    period_rate = hourly_rate * max(float(holding_hours or 0.0), 0.0)
    cost_bps = period_rate * 10_000.0
    side_norm = str(side or "").strip().lower()
    if side_norm in {"sell", "short"}:
        cost_bps = -cost_bps  # shorts earn (or pay if negative) the opposite sign
    return cost_bps


def estimate_signal_costs_bps(
    signal: Any,
    *,
    role: Optional[str] = None,
    holding_hours: Optional[float] = None,
) -> Dict[str, float]:
    """Return per-signal expected-cost breakdown in bps.

    ``total_bps`` is round-trip (entry + exit) for fees and slippage,
    plus directional funding for the holding window. Funding can be
    *negative* (a benefit, e.g. shorting a coin with positive funding
    earns the long-side premium).

    Parameters
    ----------
    signal:
        Either a TradeSignal-like with ``coin`` / ``side`` / ``leverage``
        attributes, or a dict with the same keys.
    role:
        Execution role for fee schedule. Defaults to
        ``PAPER_TRADING_DEFAULT_EXECUTION_ROLE`` (typically taker).
    holding_hours:
        Hours the position is assumed to be open for funding cost.
        Defaults to ``TRADE_COSTS_DEFAULT_HOLDING_HOURS`` (24h).
    """
    if role is None:
        role = str(
            getattr(config, "PAPER_TRADING_DEFAULT_EXECUTION_ROLE", "taker") or "taker"
        )
    if holding_hours is None:
        holding_hours = float(
            getattr(config, "TRADE_COSTS_DEFAULT_HOLDING_HOURS", 24.0) or 24.0
        )

    def _attr(name: str, default: Any = None) -> Any:
        if isinstance(signal, dict):
            return signal.get(name, default)
        return getattr(signal, name, default)

    coin = str(_attr("coin", "") or "")
    side_obj = _attr("side", "")
    side = side_obj.value if hasattr(side_obj, "value") else str(side_obj or "")

    fee_bps_one_leg = _fee_bps(role)
    slip_bps_one_leg = _slippage_bps_estimate()
    fees_bps = 2.0 * fee_bps_one_leg
    slippage_bps = 2.0 * slip_bps_one_leg
    funding_bps = _funding_bps_for_coin(coin, side=side, holding_hours=holding_hours)
    total_bps = fees_bps + slippage_bps + funding_bps

    return {
        "fees_bps": round(fees_bps, 3),
        "slippage_bps": round(slippage_bps, 3),
        "funding_bps": round(funding_bps, 3),
        "total_bps": round(total_bps, 3),
        "role": role,
        "holding_hours": float(holding_hours),
    }
