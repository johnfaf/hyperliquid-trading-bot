"""Compare fixed, research-only SPY entries and perturb the historical fill assumptions."""
from dataclasses import dataclass
from decimal import Decimal
import math

from src.backtest.stock_orb import COSTS, simulate
from src.trading.stock_orb import (
    ENTRY_BARS, MAX_CHASE, MAX_SPREAD, OPEN_BARS, STEP,
    Signal, bars_before, first_signal, positive, stamp,
)

LAB = "spy_entry_comparison_v1"
RULES = ("opening_breakout","range_retest","vwap_reclaim")


@dataclass(frozen=True)
class Stress:
    costs: str
    delay: int = 2
    spread: int = 1
    depth: int = 1
    exit_gap_bps: int = 0
    missing_exit: bool = False


STRESSES = {
    "base":Stress("base"),
    "stress":Stress("stress"),
    "delayed":Stress("stress",delay=20),
    "wide":Stress("stress",spread=3),
    "thin":Stress("stress",depth=5),
    "exit_gap":Stress("stress",exit_gap_bps=35),
    "exit_outage":Stress("stress",missing_exit=True),
    "combined":Stress("stress",delay=20,spread=3,depth=5,exit_gap_bps=35),
}


def compare_signals(rows, opened, closed, cutoff):
    """Only the prefix of completed bars can affect the first entry for each rule."""
    bars = bars_before(rows,opened,closed,cutoff)
    results = {}
    try:
        breakout = first_signal(bars,opened,cutoff)
        results["opening_breakout"] = (breakout, None)
    except ValueError:
        breakout = None
        results["opening_breakout"] = (None,"missing_bars")
    n = min(ENTRY_BARS, int(cutoff-opened)//STEP)
    if n <= OPEN_BARS:
        return bars,{name:(None,"entry_window_not_started") for name in RULES}
    if breakout is None:
        results["range_retest"] = (None,results["opening_breakout"][1] or "no_initial_breakout")
    else:
        retest, why = None,"no_confirmed_retest"
        last = min(opened+ENTRY_BARS*STEP, int(cutoff), breakout.at+3*STEP)
        for at in range(breakout.at,last,STEP):
            if at+STEP>last:
                break
            bar = bars.get(at)
            if bar is None:
                why="missing_bars"
                break
            if bar.close <= breakout.stop:
                why="range_failed_before_retest"
                break
            if bar.low <= breakout.range_high and bar.close > breakout.range_high:
                retest=Signal(at+STEP,bar.close,breakout.stop,breakout.range_high)
                why=None
                break
        results["range_retest"]=(retest,why)
    raw = {int(stamp(row["t"])):row for row in rows if stamp(row["t"])+STEP <= cutoff}
    volume,turnover=Decimal(0),Decimal(0)
    prev=None
    reclaim,why=None,"no_vwap_reclaim_above_range_mid"
    opening = [bars.get(opened+i*STEP) for i in range(OPEN_BARS)]
    if None in opening:
        results["vwap_reclaim"]=(None,"missing_bars")
        return bars,results
    high,low=max(b.high for b in opening),min(b.low for b in opening)
    stop=(high+low)/2
    for i in range(n):
        at=opened+i*STEP
        bar=bars.get(at)
        if bar is None:
            why="missing_bars"
            break
        try:
            p=positive(raw[at]["vw"])
            qty=Decimal(str(raw[at]["v"]))
            if not qty.is_finite() or qty<=0:
                raise ValueError("missing VWAP volume")
        except (ValueError,KeyError,TypeError,ArithmeticError):
            why="missing_vwap_data"
            break
        volume+=qty
        turnover+=qty*p
        above=bar.close>turnover/volume
        if i>=OPEN_BARS and prev is False and above and bar.close>stop:
            reclaim=Signal(at+STEP,bar.close,stop,high)
            why=None
            break
        prev=above
    results["vwap_reclaim"]=(reclaim,why)
    return bars,results


def quotes_under(rows, at, case, *, cutoff, exit_leg=False):
    if exit_leg and case.missing_exit:
        return []
    result=[]
    for row in rows:
        try:
            observed=stamp(row["t"])
            if not at+case.delay<=observed<=min(at+30,cutoff):
                continue
            bid,ask=positive(row["bp"]),positive(row["ap"])
            bids,asks=float(row["bs"]),float(row["as"])
            if bid>ask or not all(math.isfinite(q) and q>=case.depth for q in (bids,asks)):
                continue
        except (ValueError,KeyError,TypeError,AttributeError,ArithmeticError):
            continue
        width=(ask-bid)*(case.spread-1)/2
        bid,ask=bid-width,ask+width
        if not 0<bid<=ask<=bid*(1+MAX_SPREAD):
            continue
        if exit_leg:
            bid*=1-Decimal(case.exit_gap_bps)/10000
        result.append((observed,bid,ask))
    return sorted(result,key=lambda row:row[0])


def stressed_replays(records, *, cutoff):
    """Reuse the original quantity, fee, slippage, and exit model after perturbing quotes."""
    results, receipts = {},{}
    for name,case in STRESSES.items():
        entries=[]
        for day,signal,quotes,when,reason,out in records:
            entering=quotes_under(quotes,signal.at,case,cutoff=cutoff)
            leaving=quotes_under(out,when,case,cutoff=cutoff,exit_leg=True) if when is not None else []
            entries.append((day,signal,entering,when,reason,leaving))
        aggregate,trace=simulate(entries,case.costs)
        slip=Decimal(COSTS[case.costs][0])/10000
        for item,(_,signal,entering,_,_,_) in zip(trace,entries):
            filled=next(((at,ask*(1+slip)) for at,_,ask in entering if ask*(1+slip)<=signal.price*(1+MAX_CHASE)),None)
            item["entry_price"]=float(filled[1]) if filled else None
        results[name]=aggregate
        receipts[name]=trace
    return results,receipts
