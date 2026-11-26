from datetime import UTC, datetime, timedelta
from decimal import Decimal
import sys

import pytest

from scripts import run_stock_paper as worker
from scripts.check_stock_paper_readonly import arm
from src.trading.alpaca_paper import paper_account_pin
from src.trading.order_journal import Fill, OrderJournal
from src.trading.stock_evidence import StockEvidence


class Market:
    now = datetime(2026, 11, 16, 16, tzinfo=UTC)
    id = "strategy-paper"

    def __init__(self):
        self.equity = "1000"
        self.inventory = Decimal(0)
        self.orders = {}
        self.filled = []
        self.bars_available = True

    def request(self, path, **kwargs):
        if path == "/v2/account":
            return {"id": self.id, "status": "ACTIVE", "trading_blocked": False, "equity": self.equity,
                    "last_equity": "1000", "cash": "1000", "buying_power": "2000"}
        if path == "/v2/positions":
            return [{"symbol": "SPY", "qty": str(self.inventory)}] if self.inventory else []
        if path == "/v2/orders":
            return [r for r in self.orders.values() if r["status"] == "new"]
        if path == "/v2/clock":
            return {"is_open": True, "timestamp": self.now.isoformat(),
                    "next_close": (self.now+timedelta(hours=2)).isoformat()}
        if path == "/v2/stocks/SPY/quotes/latest":
            return {"quote": {"bp": "99.99", "ap": "100", "t": self.now.isoformat()}}
        dates = [(self.now.date()-timedelta(days=i)).isoformat() for i in range(21, -1, -1)]
        if path == "/v2/calendar":
            return [{"date": day} for day in dates]
        if path == "/v2/assets/SPY":
            return {"class": "us_equity", "tradable": True, "fractionable": True}
        assert path == "/v2/stocks/SPY/bars"
        if not self.bars_available:
            pytest.fail("risk exit must not fetch the unavailable bars")
        return {"bars": [{"t": day+"T05:00:00Z", "c": 101 if day == dates[-2] else 100} for day in dates[:-1]]}

    def submit(self, intent):
        client = intent["client_order_id"]
        assert client not in self.orders
        order = {"id": f"order-{len(self.orders)}", "client_order_id": client, "status": "new"}
        self.orders[client] = order
        return order

    def fill(self, intent):
        order = self.orders[intent["client_order_id"]]
        order["status"] = "filled"
        qty = Decimal(intent["qty"])
        side = intent["side"]
        self.inventory += qty if side == "buy" else -qty
        self.filled.append(Fill("fill-" + order["id"], order["id"], "SPY", side, qty, Decimal(intent["limit_price"])))

    def fills(self):
        return self.filled

    def lookup(self, cid):
        return self.orders.get(cid)


def test_flat_verified_account_can_be_armed_but_drill_account_cannot(tmp_path):
    market = Market()
    path = tmp_path/"ledger.db"
    receipt = tmp_path/"execution.env"
    scope = f"alpaca-paper:{market.id}"
    OrderJournal(path, scope)
    evidence = StockEvidence(path, scope)
    with pytest.raises(ValueError, match="not bound"):
        arm(market, path, receipt, drill_account="isolated-drill")
    evidence.record("observed", {"state": "outside_entry_or_exit_window"}, symbol="SPY", mode="shadow", purpose="strategy")
    with pytest.raises(ValueError, match="distinct"):
        arm(market, path, receipt, drill_account=market.id)
    assert not receipt.exists()
    proof = arm(market, path, receipt, drill_account="isolated-drill")
    assert proof["status"] == "Pass"
    assert receipt.read_text().splitlines() == [
        "STOCK_PAPER_SEND_ORDERS=true",
        f"STOCK_PAPER_ACCOUNT_SHA256={paper_account_pin(market.id)}",
    ]
    assert market.orders == {}
    market.inventory = Decimal("0.5")
    with pytest.raises(ValueError, match="position quantities"):
        arm(market, path, receipt, drill_account="isolated-drill")


@pytest.mark.parametrize("bad_pin", [None, "0"*64])
def test_stock_worker_rejects_an_unpinned_account(tmp_path, monkeypatch, capsys, bad_pin):
    market = Market()
    monkeypatch.setenv("STOCK_PAPER_KEY", "unused")
    monkeypatch.setenv("STOCK_PAPER_SECRET", "unused")
    monkeypatch.setenv("STOCK_PAPER_SEND_ORDERS", "true")
    if bad_pin is None:
        monkeypatch.delenv("STOCK_PAPER_ACCOUNT_SHA256", raising=False)
    else:
        monkeypatch.setenv("STOCK_PAPER_ACCOUNT_SHA256", bad_pin)
    monkeypatch.setattr(worker, "AlpacaPaper", lambda *_: market)
    monkeypatch.setattr(sys, "argv", ["run_stock_paper.py", "--db", str(tmp_path/"ledger.db")])
    worker.main()
    assert '"state": "paper_account_unverified"' in capsys.readouterr().out
    assert market.orders == {}


def test_automatic_stock_paper_fill_then_risk_exit_without_bars(tmp_path, monkeypatch):
    market = Market()

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return market.now

    monkeypatch.setattr(worker, "datetime", Frozen)
    path = tmp_path/"ledger.db"
    scope = f"alpaca-paper:{market.id}"
    journal = OrderJournal(path, scope)

    def cycle(bound=market.id):
        return worker.tick(market, journal, symbol="SPY", orders_enabled=True, expected_account=bound)

    assert cycle("wrong-account")["state"] == "paper_account_unverified"
    sent = cycle()
    assert sent["state"] == "intent_sent"
    assert sent["intent"]["side"] == "buy" and sent["intent"]["qty"] == "0.750000"
    assert cycle()["state"] == "order_pending"
    assert len(market.orders) == 1
    market.fill(sent["intent"])
    market.now += timedelta(minutes=5)
    market.equity, market.bars_available = "989", False
    risk = cycle()
    assert risk["state"] == "intent_sent"
    assert risk["intent"]["side"] == "sell" and Decimal(risk["intent"]["qty"]) == Decimal(".75")
    market.fill(risk["intent"])
    assert worker.reconcile(market, journal) == ({}, None)
    assert set(row["status"] for row in journal.orders()) == {"filled"}
    assert len(market.orders) == 2
