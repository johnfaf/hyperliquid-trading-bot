"""Independent SPY range-reversal experiments. No broker or order dependencies."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from src.backtest.stock_entry_lab import STRESSES, quotes_under
from src.backtest.stock_orb import CASH, COSTS, SIZE_STEP
from src.trading.stock_orb import ENTRY_BARS, HOLD, MAX_CHASE, OPEN_BARS, SLEEVE, STEP, bars_before

PROTOCOL = "spy_range_reversal_pair_v1"
RULES = ("long_rebound","short_failed_retest")
CASES = ("base","stress","combined","exit_outage")
MIN_RR = Decimal("1.5")
BORROW_HAIRCUT_BPS = {"base":2,"stress":5}


@dataclass(frozen=True)
class Plan:
    at: int
    side: int
    close: Decimal
    stop: Decimal
    target: Decimal


@dataclass(frozen=True)
class Sample:
    day: str
    plan: Plan
    entry_quotes: list
    exit_at: int
    exit_reason: str
    exit_quotes: list


def decisions(rows, opened, closed, through):
    bars=bars_before(rows,opened,closed,through)
    n=min(ENTRY_BARS,(int(through)-opened)//STEP)
    if n<OPEN_BARS:
        return bars,dict.fromkeys(RULES)
    opening=[bars.get(opened+i*STEP) for i in range(OPEN_BARS)]
    if None in opening:
        raise ValueError("missing opening range")
    high,low=max(b.high for b in opening),min(b.low for b in opening)
    width=high-low
    lower=low+width/4
    middle=low+width/2
    if width<=0 or low-width<=0:
        return bars,dict.fromkeys(RULES)
    short=None
    long=None
    below_at=None
    short_failed=False
    prev=opening[-1].close
    for at in range(opened+OPEN_BARS*STEP,opened+n*STEP,STEP):
        bar=bars.get(at)
        if bar is None:
            raise ValueError("missing entry-window bar")
        if long is None and below_at is None and lower<bar.close<middle and bar.low<=lower and bar.close>prev:
            long=Plan(at+STEP,1,bar.close,bar.low,high)
        if below_at is None and prev>=low>bar.close:
            below_at=at+STEP
        elif below_at is not None and short is None and not short_failed:
            if at+STEP>below_at+3*STEP or bar.close>=low:
                short_failed=True
            elif bar.high>=low and low-width<bar.close<low:
                short=Plan(at+STEP,-1,bar.close,bar.high,low-width)
        prev=bar.close
        if long is not None and short is not None:
            break
    return bars,{"long_rebound":long,"short_failed_retest":short}


def exit_decision(bars, plan, through):
    for opening in range(plan.at,min(plan.at+HOLD,int(through)),STEP):
        at=opening+STEP
        if at>through:
            break
        bar=bars.get(opening)
        if bar is None:
            raise ValueError("missing exit-window bar")
        if plan.side*(bar.close-plan.stop)<=0:
            return at,"stop"
        if plan.side*(bar.close-plan.target)>=0:
            return at,"target"
        if at==plan.at+HOLD:
            return at,"time_limit"
    return None


def fees(entry, exit_price, rate, carry):
    return (entry+exit_price)*rate + entry*carry


def risk_reward(plan, entry, slip, rate, carry):
    target=plan.target*(1-plan.side*slip)
    stop=plan.stop*(1-plan.side*slip)
    if not plan.side*(plan.target-entry)>0 or not plan.side*(entry-plan.stop)>0:
        return None
    reward=plan.side*(target-entry)-fees(entry,target,rate,carry)
    risk=plan.side*(entry-stop)+fees(entry,stop,rate,carry)
    if risk<=0 or reward/risk<MIN_RR:
        return None
    return reward/risk


def simulate(samples, case_name, cutoff):
    case=STRESSES[case_name]
    slip,rate=(Decimal(x)/10000 for x in COSTS[case.costs])
    gap=Decimal(case.exit_gap_bps)/10000
    cash,peak,drawdown=CASH,CASH,Decimal(0)
    closed=[]
    trace=[]
    unresolved=missed=blocked=0
    for sample in samples:
        plan=sample.plan
        if unresolved:
            blocked+=1
            trace.append({"day":sample.day,"state":"blocked_by_prior_unresolved_exit"})
            continue
        carry=Decimal(BORROW_HAIRCUT_BPS[case.costs] if plan.side<0 else 0)/10000
        entered=None
        for at,bid,ask in quotes_under(sample.entry_quotes,plan.at,case,cutoff=cutoff):
            price=(ask if plan.side==1 else bid)*(1+plan.side*slip)
            edge=risk_reward(plan,price,slip,rate,carry)
            if plan.side*(price-plan.close)<=plan.close*MAX_CHASE and edge is not None:
                qty=(min(SLEEVE,cash)/(price*(1+rate+carry))).quantize(SIZE_STEP,rounding=ROUND_DOWN)
                if qty*price>=1:
                    entered=(at,price,qty,edge)
                    break
        if entered is None:
            missed+=1
            trace.append({"day":sample.day,"state":"missed_entry","signal_at":plan.at})
            continue
        at,price,qty,edge=entered
        exits=[] if case.missing_exit else quotes_under(sample.exit_quotes,sample.exit_at,case,cutoff=cutoff)
        record={"day":sample.day,"signal_at":plan.at,"entry_at":at,"entry_price":str(price),
                "qty":str(qty),"reward_risk":str(edge)}
        if not exits:
            unresolved+=1
            trace.append({**record,"state":"unresolved_exit"})
            continue
        left,bid,ask=exits[0]
        if not at<sample.exit_at<left:
            raise ValueError("invalid quote chronology")
        exit_price=(bid if plan.side==1 else ask)*(1-plan.side*slip)*(1-plan.side*gap)
        trade_net=(plan.side*(exit_price-price)-fees(price,exit_price,rate,carry))*qty
        cash+=trade_net
        peak=max(peak,cash)
        drawdown=max(drawdown,peak-cash)
        closed.append(trade_net)
        trace.append({**record,"state":"closed","exit_at":left,"exit_price":str(exit_price),"reason":sample.exit_reason,"net":str(trade_net)})
    net=cash-CASH
    return {"closed":len(closed),"wins":sum(p>0 for p in closed),"missed":missed,"unresolved":unresolved,
            "blocked_after_unresolved":blocked,"net":None if unresolved else float(net),"realized_subtotal":float(net),
            "without_largest":None if unresolved else float(net-max([Decimal(0),*closed])),
            "realized_drawdown":float(drawdown)},trace


def clears_screen(sessions, covered, results):
    return (sessions>=40 and covered>=sessions*.9
            and all(results[name]["closed"]>=20 and not results[name]["unresolved"]
                    and results[name]["without_largest"] is not None and results[name]["without_largest"]>0
                    and results[name]["realized_drawdown"]<=10 for name in ("base","stress","combined")))
