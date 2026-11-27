from decimal import Decimal

from scripts import run_stock_entry_lab as lab
from scripts.run_stock_orb_replay import utc
from src.backtest.stock_entry_lab import STRESSES, compare_signals, quotes_under, stressed_replays
from src.backtest.stock_orb import later_quotes, simulate
from src.trading.stock_orb import STEP, Signal


OPEN=1700000000
CLOSE=OPEN+390*60


def bars(n=12):
    records=[]
    for i in range(n):
        close="100" if i<6 else "102"
        low="99" if i<6 else "101.5" if i==6 else "100.9"
        high="101" if i<6 else "103"
        records.append({"t":utc(OPEN+i*STEP),"h":high,"l":low,"c":close,"v":500,"vw":"100.5"})
    return records


def quote(at,bid="101.99",ask="102",depth=2):
    return {"t":utc(at),"bp":bid,"ap":ask,"bs":depth,"as":depth}


def test_alternatives_only_see_completed_prefix_and_retest_is_later():
    records=bars()
    before=OPEN+7*STEP-1
    assert all(sig is None for sig,_ in compare_signals(records,OPEN,CLOSE,before)[1].values())
    breakout=OPEN+7*STEP
    _,first=compare_signals(records,OPEN,CLOSE,breakout)
    assert first["opening_breakout"][0].at==first["vwap_reclaim"][0].at==breakout
    assert first["range_retest"][0] is None
    _,second=compare_signals(records,OPEN,CLOSE,breakout+STEP)
    assert second["range_retest"][0].at==breakout+STEP
    changed=records[:8]+[{**r,"c":"99","h":"102","l":"95","vw":"98"} for r in records[8:]]
    assert compare_signals(changed,OPEN,CLOSE,breakout+STEP)[1]==second


def test_missing_vwap_inputs_do_not_become_a_reclaim_or_discard_orb():
    rows=bars()
    rows[0].pop("vw")
    _,out=compare_signals(rows,OPEN,CLOSE,OPEN+8*STEP)
    assert out["opening_breakout"][0] is not None
    assert out["vwap_reclaim"]==(None,"missing_vwap_data")


def test_stresses_reject_bad_quotes_and_match_frozen_baseline():
    entry=OPEN+7*STEP
    leave=entry+1800
    sig=Signal(entry,Decimal("102"),Decimal("100"),Decimal("101"))
    valid=[quote(entry+4),quote(entry+20,"101.95","102")]
    earlier=quote(entry-1)
    after=quote(entry+35)
    assert len(quotes_under([earlier,*valid,after],entry,STRESSES["base"],cutoff=entry+10))==1
    assert len(quotes_under(valid,entry,STRESSES["delayed"],cutoff=leave))==1
    assert quotes_under(valid,entry,STRESSES["thin"],cutoff=leave)==[]
    assert quotes_under([quote(entry+4,"101.94","102")],entry,STRESSES["wide"],cutoff=leave)==[]
    out=[quote(leave+4,bid="103",ask="103.01")]
    originals=[("day",sig,later_quotes(valid,entry),leave,"time_limit",later_quotes(out,leave))]
    result,trace=stressed_replays([("day",sig,valid,leave,"time_limit",out)],cutoff=leave+30)
    assert result["base"]==simulate(originals,"base")[0]
    assert result["stress"]==simulate(originals,"stress")[0]
    assert 102<trace["base"][0]["entry_price"]<103
    assert result["exit_gap"]["net"]<result["stress"]["net"]
    assert result["thin"]["missed_entries"]==1
    assert result["exit_outage"]["unresolved"]==1 and result["exit_outage"]["net"] is None


def test_asof_report_distinguishes_past_candidate_from_pending_fill():
    entry=OPEN+7*STEP
    data={"opened":OPEN,"closed":CLOSE,"cutoff":entry+15,"bars":bars(),"quotes":{}}
    result,_=lab.review_day("test",data)
    assert result["live_approval"] is result["paper_approval"] is False
    assert result["plans"]["opening_breakout"]["state"]=="Awaiting entry quote window"
    assert result["plans"]["opening_breakout"]["results"] is None
    assert result["plans"]["range_retest"]["state"]=="No signal yet"
