"""Paired stock exit study: same entries, independent episodes, no order API."""

from decimal import Decimal
from math import isfinite

from src.backtest.stock_entry_lab import STRESSES
from src.backtest.stock_orb import COSTS
from src.backtest.stock_pullback import HOLD, exit_reason, simulate
from src.trading.stock_orb import positive, stamp

PROTOCOL = "spy_paired_exit_retry_v1"
RETRY_OFFSETS = (60, 180, 300)
WINDOW_END = 30
BPS = Decimal(10000)
REFERENCE_SPREAD_BPS = 100
RETRY_SPREAD_BPS = 35
RETRY_COST_BPS = 65


def scheduled_exit(days, entry, setup):
    for index in range(entry, entry + HOLD):
        reason = exit_reason(days[index], setup, index - entry + 1)
        if reason:
            return index, reason
    raise ValueError("episode has no scheduled exit")


def quote_pairs(rows, slot, case, cutoff, *, reference=False):
    width_limit = REFERENCE_SPREAD_BPS if reference else RETRY_SPREAD_BPS
    depth = 1 if reference else case.depth
    quotes = []
    for row in rows:
        try:
            when = stamp(row["t"])
            if not slot + case.delay <= when <= min(slot + WINDOW_END, cutoff):
                continue
            bid, ask = (positive(row[k]) for k in ("bp", "ap"))
            sizes = tuple(float(row[k]) for k in ("bs", "as"))
            if bid > ask or not all(isfinite(x) and x >= depth for x in sizes):
                continue
        except (AttributeError, ValueError, KeyError, TypeError, ArithmeticError):
            continue
        widen = Decimal(0) if reference else (ask - bid) * (case.spread - 1) / 2
        bid, ask = bid - widen, ask + widen
        if bid > 0 and (ask - bid) * BPS <= bid * width_limit:
            quotes.append((when, bid, ask))
    return sorted(quotes, key=lambda quote: quote[0])


def retry_exit(trigger, exit_day, windows, scenario):
    """Called only after the primary attempt fails. The reference never advances."""
    case = STRESSES[scenario]
    slip, fee = (Decimal(x) / BPS for x in COSTS[case.costs])
    gap = Decimal(case.exit_gap_bps) / BPS
    slot = exit_day.slot
    primary = None if case.missing_exit else windows.get(slot)
    refs = quote_pairs(primary or [], slot, case, exit_day.closed, reference=True)
    if refs:
        ref_at, bid, ask = refs[0]
        reference, basis = (bid + ask) / 2, "primary_midpoint"
    else:
        ref_at = trigger.closed
        reference, basis = trigger.close, "trigger_close"
    meta = {
        "reference": str(reference),
        "reference_source": basis,
        "reference_at": ref_at,
        "cost_cap_bps": RETRY_COST_BPS,
        "attempts": [],
    }
    for offset in RETRY_OFFSETS:
        attempt = slot + offset
        rows = None if case.missing_exit else windows.get(attempt)
        eligible = quote_pairs(rows or [], attempt, case, exit_day.closed)
        retry = {
            "slot": attempt,
            "capture": "forced_outage"
            if case.missing_exit
            else "missing"
            if rows is None
            else "captured",
            "quotes": len(rows or []),
            "eligible_quotes": len(eligible),
            "over_cost_cap": 0,
        }
        meta["attempts"].append(retry)
        for when, bid, _ in eligible:
            price = bid * (1 - slip) * (1 - gap)
            net = price * (1 - fee)
            cost_bps = (reference - net) * BPS / reference
            if cost_bps > RETRY_COST_BPS:
                retry["over_cost_cap"] += 1
                continue
            return {
                **meta,
                "state": "retry_filled",
                "quote_at": when,
                "price": str(price),
                "net_per_share": str(net),
                "cost_bps": str(cost_bps),
                "delay_seconds": when - slot,
            }
    return {
        **meta,
        "state": "still_open",
        "alert_at": slot + RETRY_OFFSETS[-1] + WINDOW_END,
    }


def capture_slots(days, plans):
    """Historical read schedule only, not visible to entry or retry decisions."""
    slots = {}
    for entry, setup in plans.items():
        trigger, _ = scheduled_exit(days, entry, setup)
        opening, leaving = days[entry], days[trigger + 1]
        slots[opening.slot] = opening.day
        for offset in (0, *RETRY_OFFSETS):
            slots[leaving.slot + offset] = leaving.day
    return dict(sorted(slots.items()))


def paired_episodes(days, plans, windows, scenario):
    """Reuse the frozen control per entry. Overlapping samples are not a portfolio."""
    trace = []
    for entry, setup in sorted(plans.items()):
        starting = days[entry]
        trigger, reason = scheduled_exit(days, entry, setup)
        leaving = days[trigger + 1]
        strict_quotes = {
            day.day: windows.get(day.slot) or [] for day in (starting, leaving)
        }
        _, control = simulate(
            days[entry : trigger + 2], starting.day, {0: setup}, strict_quotes, scenario
        )
        opened = control[0]
        episode = {
            "signal": setup.signal,
            "entry_day": starting.day,
            "entry": opened,
            "exit_day": leaving.day,
            "exit_reason": reason,
        }
        if opened["state"] != "entered":
            trace.append(episode)
            continue
        strict = control[-1]
        if strict["state"] == "closed":
            trace.append(
                {**episode, "strict": strict, "exit": {"state": "primary_filled"}}
            )
            continue
        result = retry_exit(days[trigger], leaving, windows, scenario)
        case = STRESSES[scenario]
        fee = Decimal(COSTS[case.costs][1]) / BPS
        qty = Decimal(opened["qty"])
        if result["state"] == "retry_filled":
            entry_cost = Decimal(opened["price"]) * (1 + fee)
            result["trade_net"] = str(
                qty * (Decimal(result["net_per_share"]) - entry_cost)
            )
        else:
            result["alert"] = {
                "kind": "exit_exposure_still_open",
                "qty": str(qty),
                "symbol": "SPY",
                "timestamp": result["alert_at"],
                "signal": setup.signal,
                "reason": reason,
            }
        trace.append({**episode, "strict": strict, "exit": result})
    outcomes = [row["exit"] for row in trace if "exit" in row]
    rescued = [row for row in outcomes if row["state"] == "retry_filled"]
    still_open = sum(row["state"] == "still_open" for row in outcomes)
    return {
        "paired_entries": len(outcomes),
        "primary_fills": len(outcomes) - len(rescued) - still_open,
        "strict_unresolved": len(rescued) + still_open,
        "retry_fills": len(rescued),
        "retry_unresolved": still_open,
        "alert_events": still_open,
        "max_retry_cost_bps": max(
            (float(row["cost_bps"]) for row in rescued), default=None
        ),
        "max_retry_delay_seconds": max(
            (row["delay_seconds"] for row in rescued), default=None
        ),
    }, trace
