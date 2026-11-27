from datetime import UTC, date, datetime
from decimal import Decimal
import sqlite3

import pytest

from scripts import run_stock_range_holdout as runner
from scripts.run_stock_orb_replay import utc
from src.backtest.stock_range_reversal import Plan, Sample, clears_screen, decisions, exit_decision, risk_reward, simulate
from src.trading.stock_orb import STEP

OPEN=1700000000
CLOSE=OPEN+390*60
D=Decimal


def bar(i,high,low,close):
    return {"t":utc(OPEN+i*STEP),"h":high,"l":low,"c":close}


def opening():
    return [bar(i,"110","100","102") for i in range(6)]


def quote(at,bid,ask):
    return {"t":utc(at),"bp":bid,"ap":ask,"bs":6,"as":6}


def test_bounce_breakdown_and_retest_only_use_completed_bars():
    market=opening()+[bar(6,"104","101","103"),bar(7,"102","98","99"),
                     bar(8,"100.3","98","99.3"),bar(9,"99","89","90")]
    before=OPEN+7*STEP-1
    assert decisions(market,OPEN,CLOSE,before)[1]=={"long_rebound":None,"short_failed_retest":None}
    _,first=decisions(market,OPEN,CLOSE,OPEN+7*STEP)
    assert first["long_rebound"]==Plan(OPEN+7*STEP,1,D(103),D(101),D(110))
    assert first["short_failed_retest"] is None
    assert decisions(market,OPEN,CLOSE,OPEN+8*STEP)[1]["short_failed_retest"] is None
    _,after=decisions(market,OPEN,CLOSE,OPEN+9*STEP)
    assert after["short_failed_retest"]==Plan(OPEN+9*STEP,-1,D("99.3"),D("100.3"),D(90))
    changed=market[:-1]+[bar(9,"130","81","130")]
    assert decisions(changed,OPEN,CLOSE,OPEN+9*STEP)[1]==after


def test_retest_that_closes_inside_range_invalidates_short():
    market=opening()+[bar(6,"102","98","99"),bar(7,"103","99","101"),
                     bar(8,"101","98","99"),bar(9,"102","97","98")]
    _,plans=decisions(market,OPEN,CLOSE,OPEN+10*STEP)
    assert plans=={"long_rebound":None,"short_failed_retest":None}


def test_exits_require_completed_bars_and_honor_side():
    market=opening()+[bar(6,"104","101","103"),bar(7,"110.2","103","110")]
    bars,plans=decisions(market,OPEN,CLOSE,OPEN+8*STEP)
    plan=plans["long_rebound"]
    assert exit_decision(bars,plan,OPEN+8*STEP-1) is None
    assert exit_decision(bars,plan,OPEN+8*STEP)==(OPEN+8*STEP,"target")
    bars.pop(OPEN+7*STEP)
    with pytest.raises(ValueError,match="missing exit-window"):
        exit_decision(bars,plan,OPEN+8*STEP)


def test_quotes_fees_and_gap_are_adverse_for_both_directions():
    at=OPEN+7*STEP
    leave=at+STEP
    long=Plan(at,1,D(103),D(101),D(110))
    short=Plan(at,-1,D(99),D("100.3"),D(90))
    cases=[
        Sample("day",long,[quote(at+22,"102.99","103")],leave,"target",[quote(leave+22,"109.99","110")]),
        Sample("day",short,[quote(at+22,"99","99.01")],leave,"target",[quote(leave+22,"90","90.01")]),
    ]
    for sample in cases:
        base,trace=simulate([sample],"base",CLOSE)
        stress,_=simulate([sample],"stress",CLOSE)
        combined,_=simulate([sample],"combined",CLOSE)
        assert base["net"]>stress["net"]>combined["net"]>0
        actual=D(trace[0]["entry_price"])
        assert actual>sample.plan.close if sample.plan.side==1 else actual<sample.plan.close
        assert D(trace[0]["qty"])*actual<=75
    weak=Plan(at,1,D(103),D(101),D("104.3"))
    assert risk_reward(weak,D("103.05"),D(".0005"),D(".0002"),D(0)) is None


def test_outage_does_not_release_shared_sleeve():
    at=OPEN+7*STEP
    plan=Plan(at,1,D(103),D(101),D(110))
    fill=[quote(at+6,"102.99","103")]
    row=Sample("day",plan,fill,at+STEP,"time_limit",[])
    report,trace=simulate([row,row],"exit_outage",CLOSE)
    assert report["net"] is None and report["unresolved"]==report["blocked_after_unresolved"]==1
    assert [r["state"] for r in trace]==["unresolved_exit","blocked_by_prior_unresolved_exit"]
    assert not clears_screen(62,62,dict.fromkeys(("base","stress","combined"),report))


def test_holdout_registration_cannot_be_moved_after_capture():
    now=datetime(2026,11,27,18,tzinfo=UTC)
    with sqlite3.connect(":memory:") as db:
        initial=runner.preregister(db,date(2026,6,1),date(2026,9,1),now)
        assert initial==runner.preregister(db,date(2026,6,1),date(2026,9,1),now)
        with pytest.raises(ValueError,match="already pinned"):
            runner.preregister(db,date(2026,6,2),date(2026,9,1),now)
