from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256

import pytest

from scripts.isolated_stock_drill import CAP, Blocked, act
from src.trading.order_journal import Fill

NOW = datetime(2026, 11, 23, 17, tzinfo=timezone.utc)
EXCLUDED = {sha256(b"production").hexdigest()}


class Paper:
    def __init__(self, identity="isolated"):
        self.identity = identity
        self.cash = Decimal(1000)
        self.qty = Decimal(0)
        self.orders, self.observed_fills = [], []
        self.delayed_fills = False
        self.post_count = 0

    def request(self, path, **kwargs):
        assert "payload" not in kwargs
        return {
            "/v2/account": {"id":self.identity,"status":"ACTIVE","trading_blocked":False,"cash":str(self.cash)},
            "/v2/positions": [{"symbol":"SPY","qty":str(self.qty)}] if self.qty else [],
            "/v2/orders": [] if kwargs.get("query",{}).get("status")=="open" else self.orders,
            "/v2/clock": {"timestamp":NOW.isoformat(),"is_open":True,"next_close":(NOW+timedelta(hours=1)).isoformat()},
            "/v2/assets/SPY": {"class":"us_equity","tradable":True,"fractionable":True},
            "/v2/stocks/SPY/quotes/latest": {"quote":{"bp":99.99,"ap":100,"t":NOW.isoformat()}},
        }[path]

    def fills(self):
        return [] if self.delayed_fills else self.observed_fills

    def lookup(self,cid):
        return next((order for order in self.orders if order["client_order_id"]==cid),None)

    def submit(self,intent):
        self.post_count += 1
        qty,price = Decimal(intent["qty"]),Decimal(intent["limit_price"])
        sign = 1 if intent["side"]=="buy" else -1
        if sign==1:
            assert qty*price <= CAP
        else:
            assert qty == self.qty
        self.qty += sign*qty
        self.cash -= sign*qty*price
        oid = str(self.post_count)
        response = {"id":oid,"client_order_id":intent["client_order_id"],"status":"filled"}
        self.orders.append(response)
        self.observed_fills.append(Fill(oid+"-fill",oid,intent["symbol"],intent["side"],qty,price))
        return response


def run(broker,path,action,*,confirm=True,excluded=EXCLUDED):
    return act(broker,path,action,expected=broker.identity,excluded=excluded,now=NOW,confirm=confirm)


def test_exclusion_and_operator_confirmation_fail_closed(tmp_path):
    file = tmp_path/"drill.db"
    current = Paper("production")
    with pytest.raises(Blocked,match="cannot be used"):
        run(current,file,"entry")
    assert not file.exists() and current.post_count == 0
    separate = Paper()
    with pytest.raises(Blocked,match="exclusion"):
        run(separate,file,"entry",excluded=set())
    with pytest.raises(Blocked,match="explicit"):
        run(separate,file,"entry",confirm=False)
    assert separate.post_count == 0


def test_lost_ack_then_process_restart_delayed_fill_and_cash_clear(tmp_path):
    file=tmp_path/"drill.db"
    broker=Paper()
    assert run(broker,file,"preflight",confirm=False)["state"].startswith("Ready")
    assert run(broker,file,"entry")["state"] == "submission_unknown"
    assert not run(broker,file,"entry")["passed"] and broker.post_count == 1
    broker.delayed_fills=True
    with pytest.raises(Blocked,match="pending"):
        run(broker,file,"recover",confirm=False)
    broker.delayed_fills=False
    assert "without resubmission" in run(broker,file,"recover",confirm=False)["state"]
    assert broker.post_count == 1
    assert run(broker,file,"exit")["state"] == "intent_sent"
    assert run(broker,file,"verify",confirm=False)["passed"]
    assert broker.post_count == 2
    broker.cash += Decimal(".02")
    with pytest.raises(Blocked,match="cash"):
        run(broker,file,"verify",confirm=False)


def test_nonempty_venue_detected_before_submission(tmp_path):
    broker=Paper()
    broker.orders.append({"id":"outside","client_order_id":"unmanaged","status":"canceled"})
    with pytest.raises(Blocked,match="without order"):
        run(broker,tmp_path/"drill.db","entry")
    assert broker.post_count == 0
