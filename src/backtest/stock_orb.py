"""Point-in-time SPY replay fills using later historical IEX quotes."""
from decimal import Decimal, ROUND_DOWN
import math

from src.trading.stock_orb import MAX_CHASE, MAX_SPREAD, SLEEVE, positive, stamp

COSTS = {"base":(5,2), "stress":(12,6)}
SIZE_STEP = Decimal(".000001")
CASH = Decimal(1000)


def later_quotes(rows, at):
    quotes = []
    for row in rows:
        observed = stamp(row["t"])
        if not at+2 <= observed <= at+30:
            continue
        try:
            bid,ask = positive(row["bp"]),positive(row["ap"])
            bid_size,ask_size = float(row["bs"]),float(row["as"])
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue
        if (bid <= ask <= bid*(1+MAX_SPREAD) and math.isfinite(bid_size) and math.isfinite(ask_size)
                and min(bid_size,ask_size) >= 1):
            quotes.append((observed,bid,ask))
    return sorted(quotes,key=lambda row:row[0])


def simulate(records, scenario):
    slip,fee = (Decimal(x)/10000 for x in COSTS[scenario])
    cash,peak,drawdown,total_fees = CASH,CASH,Decimal(0),Decimal(0)
    exits,trace = [],[]
    missed_entries = unresolved = 0
    for day, signal, quotes, exit_at, exit_reason, exits_quotes in records:
        if signal is None:
            continue
        buy = next(((at,ask*(1+slip)) for at,_,ask in quotes
                    if ask*(1+slip) <= signal.price*(1+MAX_CHASE)),None)
        if buy is None:
            missed_entries += 1
            trace.append({"day":day,"status":"missed_entry","signal_at":signal.at})
            continue
        at,price = buy
        qty=(min(SLEEVE,cash)/(price*(1+fee))).quantize(SIZE_STEP,rounding=ROUND_DOWN)
        if qty <= 0 or not exits_quotes:
            unresolved += 1
            trace.append({"day":day,"status":"unresolved_exit","signal_at":signal.at,"entry_quote_at":at})
            continue
        left,bid,_ = exits_quotes[0]
        if not at < exit_at <= left:
            raise ValueError("overlapping entry and exit")
        sell = bid*(1-slip)
        fees = (price+sell)*qty*fee
        pnl = (sell-price)*qty-fees
        cash += pnl
        peak = max(peak,cash)
        drawdown = max(drawdown,peak-cash)
        total_fees += fees
        exits.append(pnl)
        trace.append({"day":day,"status":"closed","signal_at":signal.at,"entry_quote_at":at,"exit_quote_at":left,
                      "reason":exit_reason,"qty":str(qty),"net":str(pnl)})
    net = cash-CASH
    return {"closed":len(exits),"missed_entries":missed_entries,"unresolved":unresolved,
            "realized_subtotal":float(net),"net":None if unresolved else float(net),
            "without_largest":None if unresolved else float(net-max([Decimal(0),*exits])),
            "drawdown":float(drawdown),"cost_fees":float(total_fees)},trace
