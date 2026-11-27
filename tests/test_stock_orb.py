from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest

from scripts import run_stock_orb_replay as replay
from scripts import run_stock_paper as worker
from scripts.publish_dashboard_status import stock_orb_status
from src.backtest.stock_orb import later_quotes, simulate
from src.trading.order_journal import OrderJournal
from src.trading.stock_evidence import StockEvidence, read_summary
from src.trading.stock_intraday_pilot import PURPOSE, Pilot
from src.trading.stock_orb import STEP, Signal, bars_before, exit_decision, first_signal, qualifies, rule_digest
from tests.test_stock_paper_activation import Market


def window(opened):
    return [{"t":replay.utc(opened+i*STEP),"h":"101" if i<6 else "104","l":"99",
             "c":"100" if i<6 else "102" if i==6 else "99.5"} for i in range(9)]


def quote(at,bid="101.99",ask="102",size=1):
    return {"t":replay.utc(at),"bp":bid,"ap":ask,"bs":size,"as":size}


def test_opening_range_refuses_partial_bars_and_executes_after_signal():
    opening=1700000000
    end=opening+390*60
    rows=window(opening)
    through=opening+7*STEP
    before=bars_before(rows,opening,end,through-1)
    assert first_signal(before,opening,through-1) is None
    complete=bars_before(rows,opening,end,through)
    signal=first_signal(complete,opening,through)
    assert signal.at==through and signal.stop==100 and signal.range_high==101
    assert exit_decision(complete,signal.at,signal.stop,through) is None
    assert exit_decision(complete,signal.at,signal.stop,through+STEP)==(through+STEP,"missing_protection_data")
    stop=exit_decision(bars_before(rows,opening,end,through+STEP),signal.at,signal.stop,through+STEP)
    assert stop==(through+STEP,"range_stop")
    offers=later_quotes([quote(through-5),quote(through),quote(through+3,size=0),quote(through+4)],through)
    assert len(offers)==1 and offers[0][0]>through
    exits=later_quotes([quote(stop[0]+3,"99.49","99.50")],stop[0])
    record=("session",signal,offers,stop[0],stop[1],exits)
    base,_=simulate([record],"base")
    stress,_=simulate([record],"stress")
    assert base["closed"]==stress["closed"]==1 and stress["net"]<base["net"]<0
    miss,_=simulate([(*record[:2],(),*record[3:])],"stress")
    assert miss["missed_entries"]==1 and miss["closed"]==0
    unresolved,_=simulate([(*record[:-1],())],"stress")
    assert unresolved["unresolved"]==1 and unresolved["net"] is None


def test_replay_holdout_cannot_change_and_gate_needs_complete_stress_evidence(tmp_path):
    now=datetime(2026,11,27,22,tzinfo=UTC)
    start,end=datetime(2026,9,1).date(),now.date()
    with sqlite3.connect(":memory:") as db:
        manifest=replay.preregister(db,start,end,now)
        with pytest.raises(ValueError,match="already registered"):
            replay.preregister(db,start+timedelta(days=1),end,now)
    row={"closed":25,"unresolved":0,"net":3.,"without_largest":2.,"drawdown":1.,"missed_entries":0,"realized_subtotal":3.}
    report={**manifest,"strategy_sha256":rule_digest(),"published_at":now.isoformat(),
            "live_approval":False,"window_closed":True,"feed":"iex","capital":1000,"sleeve":75,
            "sessions":60,"covered":56,"signals":25,"results":{"base":row,"stress":{**row}},
            "state":"Eligible for supervised paper pilot"}
    assert qualifies(report,digest=rule_digest())
    path=tmp_path/"report.json"
    replay.publish(path,report)
    assert stock_orb_status(path,now)["signals"]==25
    report["results"]["stress"]["without_largest"]=-1
    assert not qualifies(report,digest=rule_digest())
    report["results"]["stress"]["without_largest"]=1
    assert not qualifies(report,digest="not-the-approved-code")
    report["results"]["stress"]["closed"]=5
    assert not qualifies(report,digest=rule_digest())


def test_quote_capture_follows_pages_and_rejects_a_loop():
    class Feed:
        loop=False

        def get(self,name,**query):
            assert name=="quotes" and query["sort"]=="asc"
            if not query.get("page_token"):
                return {"quotes":[{"n":1}],"next_page_token":"one"}
            return {"quotes":[{"n":2}],"next_page_token":"one" if self.loop else None}

    feed=Feed()
    assert replay.quote_window(feed,1700000000)==[{"n":1},{"n":2}]
    feed.loop=True
    with pytest.raises(ValueError,match="partial historical quote window"):
        replay.quote_window(feed,1700000000)


def test_paper_stock_journal_explains_no_order_from_exact_daily_inputs(tmp_path,monkeypatch):
    class Below(Market):
        def request(self,path,**kwargs):
            r=super().request(path,**kwargs)
            if path=="/v2/stocks/SPY/bars":
                r["bars"][-1]["c"]=90
            return r

    market=Below()

    class Frozen(datetime):
        @classmethod
        def now(cls,tz=None):
            return market.now

    monkeypatch.setattr(worker,"datetime",Frozen)
    path=tmp_path/"equity.db"
    journal=OrderJournal(path,"paper")
    evidence=StockEvidence(path,journal.account)
    result=worker.tick(market,journal,symbol="SPY",orders_enabled=True)
    assert result["state"]=="no_order" and not result["daily_signal"]["entry"]
    assert result["daily_signal"]["last_close"]=="90" and result["daily_signal"]["sma20"]=="99.5"
    for i in range(2):
        t=market.now+timedelta(minutes=5*i)
        evidence.record(str(i),{**result,"quote_at":t.isoformat()},symbol="SPY",mode="paper",
                        purpose="strategy",observed_at=t)
    summary=read_summary(path)
    assert summary["daily_signal"]["entry"]==0 and summary["daily_signal"]["last_close"]=="90"
    assert "No SMA20 crossover" in summary["scorecard"]["state"]
    assert market.orders=={}


def test_pilot_uses_shared_inventory_and_exits_when_new_entries_disabled(tmp_path,monkeypatch):
    market=Market()

    class Frozen(datetime):
        @classmethod
        def now(cls,tz=None):
            return market.now

    monkeypatch.setattr(worker,"datetime",Frozen)
    monkeypatch.setattr(worker,"sma20_from_closes",lambda _: (False,True))
    path=tmp_path/"shared.db"
    journal=OrderJournal(path,f"alpaca-paper:{market.id}")
    plans=Path(tmp_path/"pilot.db")
    pilot=Pilot(plans,enabled=True)
    signal=Signal(int(market.now.timestamp())-5,Decimal("100"),Decimal("99"),Decimal("99.98"))
    monkeypatch.setattr(pilot,"candidate",lambda *_:signal)
    order=worker.tick(market,journal,symbol="SPY",orders_enabled=True,expected_account=market.id,pilot=pilot)
    assert order["route"]==PURPOSE and order["state"]=="intent_sent"
    assert order["intent"]["qty"]=="0.750000" and plans.is_file()
    assert not journal.reserve("parallel-daily-entry",{},entry=True,exclusive_entry=True)
    market.fill(order["intent"])
    pilot.enabled=False
    market.now+=timedelta(minutes=30)
    market.bars_available=False
    sale=worker.tick(market,journal,symbol="SPY",orders_enabled=True,expected_account=market.id,pilot=pilot)
    assert sale["intent"]["side"]=="sell" and sale["intraday_reason"]=="time_limit"
    assert sale["intent"]["qty"]==order["intent"]["qty"]
    market.fill(sale["intent"])
    assert worker.reconcile(market,journal)==({},None)
