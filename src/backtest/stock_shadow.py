"""Deterministic two-quote research execution, with no broker or account imports."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

RULE = "spy_sma20_two_quote_forward_v1"
CAPITAL = Decimal("1000")
SLEEVE = Decimal("75")
DAY_LIMIT = 60
GRACE = 75
COSTS = {"base": (5, 2), "stress": (12, 6)}
QTY_STEP = Decimal(".000001")


@dataclass(frozen=True)
class Frame:
    quote_at: int
    observed_at: int
    bid: Decimal
    ask: Decimal
    entry: bool | None = None
    exit: bool | None = None


def simulate(days, costs):
    """Consume each day's decision and execution quotes in order without backfills."""
    slip, fee = (Decimal(v)/10000 for v in costs)
    cash, qty, paid = CAPITAL, Decimal(0), Decimal(0)
    passive_cash, passive_qty = CAPITAL, Decimal(0)
    peak, drawdown = CAPITAL, Decimal(0)
    nav = passive_nav = CAPITAL
    closes, fees = [], Decimal(0)
    entries = skipped = 0
    trace = []

    for day, first, second in days:
        eligible = (first is not None and second is not None and first.entry is not None and first.exit is not None
                    and first.observed_at < second.quote_at <= second.observed_at)
        if first is not None and first.entry is not None and (first.entry or (qty and first.exit)) and not eligible:
            skipped += 1
        for slot, frame in enumerate((first, second)):
            if frame is None:
                continue
            sell, buy = frame.bid*(1-slip), frame.ask*(1+slip)
            if slot == 1:
                if eligible and not passive_qty:
                    passive_qty = (SLEEVE/(buy*(1+fee))).quantize(QTY_STEP, rounding=ROUND_DOWN)
                    passive_cash -= passive_qty*buy*(1+fee)
                if eligible and qty and first.exit:
                    received = qty*sell*(1-fee)
                    fees += qty*sell*fee
                    cash += received
                    closes.append(received-paid)
                    trace.append({"session":day,"event":"sell","qty":str(qty),"price":str(sell),"net":str(received-paid)})
                    qty = paid = Decimal(0)
                elif eligible and not qty and first.entry:
                    qty = (min(SLEEVE,cash)/(buy*(1+fee))).quantize(QTY_STEP, rounding=ROUND_DOWN)
                    if qty > 0:
                        paid = qty*buy*(1+fee)
                        fees += qty*buy*fee
                        cash -= paid
                        entries += 1
                        trace.append({"session":day,"event":"buy","qty":str(qty),"price":str(buy)})
            nav = cash + qty*sell*(1-fee)
            passive_nav = passive_cash + passive_qty*sell*(1-fee)
            peak = max(peak,nav)
            if peak:
                drawdown = max(drawdown,(peak-nav)/peak*100)

    net = nav-CAPITAL
    return {
        "net":float(net),"cash":0.,"passive":float(passive_nav-CAPITAL),
        "realized":float(sum(closes,Decimal(0))),"fees":float(fees),"closed":len(closes),
        "entries":entries,"open":qty>0,"missed_signal_slots":skipped,"drawdown_pct":float(drawdown),
        "net_without_largest":float(net-max([Decimal(0),*closes])),
    }, trace
