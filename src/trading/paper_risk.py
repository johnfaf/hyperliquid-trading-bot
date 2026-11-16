"""Sizing controls for simulated entries; no live order behavior."""
import math

import config


def size_for_planned_stop(
    *, balance: float, side: str, entry: float, stop: float, size: float,
    leverage: float, execution_role: str,
) -> tuple[float, dict[str, float]]:
    """Cap size using planned stop distance, modeled exit slippage and fees."""
    if not all(math.isfinite(v) and v > 0 for v in (balance, entry, stop, size, leverage)):
        raise ValueError("invalid paper position or stop")
    if not ((side == "long" and stop < entry) or (side == "short" and stop > entry)):
        raise ValueError("initial stop is on the wrong side of the paper fill")

    slippage = max(config.PAPER_TRADING_SLIPPAGE_MIN_BPS, config.PAPER_TRADING_SLIPPAGE_MAX_BPS, 0.0) / 10_000
    stop_fill = stop * (1 - slippage if side == "long" else 1 + slippage)
    exit_fee = max(config.PAPER_TRADING_TAKER_FEE_BPS, 0.0) / 10_000
    entry_fee = max(config.PAPER_TRADING_MAKER_FEE_BPS, 0.0) / 10_000 if execution_role == "maker" else exit_fee
    loss_per_unit = leverage * (abs(entry - stop_fill) + entry * entry_fee + stop_fill * exit_fee)
    budget = balance * config.PAPER_MAX_PLANNED_STOP_LOSS_PCT
    capped_size = min(size, budget / loss_per_unit) if budget > 0 else size
    return capped_size, {
        "proposed_notional_usd": entry * size * leverage,
        "entry_notional_usd": entry * capped_size * leverage,
        "planned_stop_loss_usd": loss_per_unit * capped_size,
        "planned_stop_budget_usd": budget,
    }
