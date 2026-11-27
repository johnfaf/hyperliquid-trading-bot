"""Completed-session SPY pullback research. No broker or order imports."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from src.backtest.stock_entry_lab import STRESSES, quotes_under
from src.backtest.stock_orb import CASH, COSTS, SIZE_STEP
from src.backtest.stock_range_reversal import risk_reward
from src.trading.stock_orb import SLEEVE

RULE = "spy_rising_trend_sma10_pullback_v1"
HOLD = 10
WARMUP = 61
CASES = ("base", "stress", "combined", "exit_outage")
CHASE = Decimal(".0035")


@dataclass(frozen=True)
class Day:
    day: str
    opened: int
    closed: int
    low: Decimal
    close: Decimal

    @property
    def slot(self):
        return self.opened + 300


@dataclass(frozen=True)
class Setup:
    signal: str
    side: int
    close: Decimal
    stop: Decimal
    target: Decimal


def signal(prefix):
    if len(prefix) < WARMUP:
        return None
    closes = [d.close for d in prefix[-WARMUP:]]
    last, prior = closes[-1], closes[-2]
    latest50 = sum(closes[-50:]) / 50
    prior50 = sum(closes[-60:-10]) / 50
    if not (
        last > latest50 > prior50
        and prior <= sum(closes[-11:-1]) / 10
        and last > sum(closes[-10:]) / 10
    ):
        return None
    stop = min(d.low for d in prefix[-4:])
    if stop >= last:
        return None
    return Setup(prefix[-1].day, 1, last, stop, last + Decimal("2.5") * (last - stop))


def proposals(days, start):
    return {
        i + 1: setup
        for i in range(len(days) - HOLD - 1)
        if days[i].day >= start and (setup := signal(days[: i + 1])) is not None
    }


def exit_reason(day, setup, elapsed):
    if day.close <= setup.stop:
        return "stop"
    if day.close >= setup.target:
        return "target"
    return "time_limit" if elapsed >= HOLD else None


def quote_days(days, start, plans):
    """Schedule historical reads only. The simulator makes its own chronological decisions."""
    needed = {next(i for i, d in enumerate(days) if d.day >= start), len(days) - 1}
    for entry, setup in plans.items():
        needed.add(entry)
        for i in range(entry, entry + HOLD):
            if exit_reason(days[i], setup, i - entry + 1):
                needed.add(i + 1)
                break
    return sorted(needed)


def markets(day, quotes, case):
    return quotes_under(quotes.get(day.day) or [], day.slot, case, cutoff=day.closed)


def buy(cash, price, fee):
    return (min(SLEEVE, cash) / (price * (1 + fee))).quantize(
        SIZE_STEP, rounding=ROUND_DOWN
    )


def passive_net(first, last, quotes, case, slip, fee):
    entering, leaving = markets(first, quotes, case), markets(last, quotes, case)
    if not entering or not leaving or case.missing_exit:
        return None
    price = entering[0][2] * (1 + slip)
    qty = buy(CASH, price, fee)
    exit_bid = leaving[0][1] * (1 - slip) * (1 - Decimal(case.exit_gap_bps) / 10000)
    return float(qty * (exit_bid * (1 - fee) - price * (1 + fee)))


def simulate(days, start, plans, quotes, scenario):
    case = STRESSES[scenario]
    slip, fee = (Decimal(x) / 10000 for x in COSTS[case.costs])
    gap = Decimal(case.exit_gap_bps) / 10000
    first = next(i for i, d in enumerate(days) if d.day >= start)
    cash = peak = CASH
    drawdown = Decimal(0)
    position = None
    pending_exit = None
    missed = blocked = unresolved = exposure_days = 0
    trace, returns = [], []
    for i in range(first, len(days)):
        day = days[i]
        exiting = position is not None and pending_exit is not None
        if exiting:
            entering, setup, qty, paid = position
            available = [] if case.missing_exit else markets(day, quotes, case)
            if not available:
                unresolved = 1
                trace.append(
                    {"day": day.day, "state": "unresolved_exit", "reason": pending_exit}
                )
                # No later sleeve reuse, and no invented sale.
                blocked += sum(k >= i for k in plans)
                break
            sold_at, bid, _ = available[0]
            out = bid * (1 - slip) * (1 - gap)
            proceeds = qty * out * (1 - fee)
            cash += proceeds
            returns.append(proceeds - paid)
            trace.append(
                {
                    "day": day.day,
                    "state": "closed",
                    "signal": setup.signal,
                    "quote_at": sold_at,
                    "price": str(out),
                    "reason": pending_exit,
                    "net": str(proceeds - paid),
                }
            )
            position = pending_exit = None
        if i in plans:
            setup = plans[i]
            if position is not None or exiting:
                blocked += 1
                trace.append(
                    {
                        "day": day.day,
                        "state": "sleeve_unavailable",
                        "signal": setup.signal,
                    }
                )
            else:
                for at, _, ask in markets(day, quotes, case):
                    price = ask * (1 + slip)
                    if (
                        price > setup.close * (1 + CHASE)
                        or risk_reward(setup, price, slip, fee, Decimal(0)) is None
                    ):
                        continue
                    qty = buy(cash, price, fee)
                    if qty * price < 1:
                        continue
                    paid = qty * price * (1 + fee)
                    cash -= paid
                    position = (i, setup, qty, paid)
                    trace.append(
                        {
                            "day": day.day,
                            "state": "entered",
                            "signal": setup.signal,
                            "quote_at": at,
                            "price": str(price),
                            "qty": str(qty),
                            "stop": str(setup.stop),
                            "target": str(setup.target),
                        }
                    )
                    break
                if position is None:
                    missed += 1
                    trace.append(
                        {
                            "day": day.day,
                            "state": "missed_entry",
                            "signal": setup.signal,
                        }
                    )
        nav = cash
        if position is not None:
            entry, setup, qty, paid = position
            exposure_days += 1
            pending_exit = exit_reason(day, setup, i - entry + 1)
            nav += day.close * (1 - slip) * (1 - fee) * qty
        peak = max(peak, nav)
        drawdown = max(drawdown, peak - nav)
    if position is not None and not unresolved:
        unresolved = 1
    net = sum(returns, Decimal(0))
    return {
        "closed": len(returns),
        "wins": sum(p > 0 for p in returns),
        "missed": missed,
        "sleeve_blocked": blocked,
        "unresolved": unresolved,
        "exposure_sessions": exposure_days,
        "net": None if unresolved else float(net),
        "realized_subtotal": float(net),
        "without_largest": None
        if unresolved
        else float(net - max([Decimal(0), *returns])),
        "close_mark_drawdown": float(drawdown),
        "passive": passive_net(days[first], days[-1], quotes, case, slip, fee),
    }, trace
