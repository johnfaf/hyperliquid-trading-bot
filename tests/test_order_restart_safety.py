from decimal import Decimal
from datetime import datetime, timedelta, UTC
from threading import Lock
from types import SimpleNamespace

import pytest

import config
from scripts import run_stock_paper as stocks
from src.trading.alpaca_paper import AlpacaPaper
from src.trading.live_trader import LiveTrader
from src.trading.order_journal import Fill, OrderJournal


def make_live(api):
    trader = LiveTrader.__new__(LiveTrader)
    trader.dry_run = False
    trader._nonce_lock = Lock()
    trader._order_dedup_lock = Lock()
    trader._last_nonce = 0
    trader._ORDER_DEDUP_WINDOW = 30
    trader._recent_order_hashes = {}
    trader._persist_order_dedup_cache = lambda: None
    trader.signer = SimpleNamespace(sign_action=lambda *a, **kw: {})
    trader.info_url = "https://example.invalid/info"
    trader.exchange_url = "https://example.invalid/exchange"
    trader.public_address = "0x123"
    trader.api_manager = SimpleNamespace(post=api)
    return trader


def order(cid, reduce_only=False):
    return {"type": "order", "orders": [{"c": cid, "a": 0, "r": reduce_only}]}


def test_hyperliquid_does_not_resubmit_after_lost_response_and_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "bot.db"))
    original = "0x" + "a" * 32
    later = "0x" + "b" * 32
    reduction = "0x" + "c" * 32
    accepted, discovered = [], set()

    def api(payload, **kwargs):
        if payload.get("type") == "orderStatus":
            cid = payload["oid"]
            if cid in discovered:
                return {"status": "order", "order": {"status": "filled", "order": {
                    "cloid": cid, "oid": 12, "coin": "BTC", "side": "B", "origSz": "0.25", "sz": "0",
                }}}
            return {"status": "unknownOid"}
        if payload.get("type") == "userFills":
            first = {"tid": 23, "oid": 12, "coin": "BTC", "side": "B", "sz": ".1", "px": "64000"}
            return [first, first, {**first, "tid": 24, "sz": ".15"}]
        assert kwargs["retries"] == 1
        cid = payload["action"]["orders"][0]["c"]
        accepted.append(cid)
        if cid == original:
            return None  # Venue has accepted it, but the client never gets a response.
        return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 17}}]}}}

    assert make_live(api)._post_order(order(original))["reason"] == "submission_unknown"
    restarted = make_live(api)
    assert restarted._post_order(order(later))["reason"] == "durable_order_blocked"
    assert restarted._post_order(order(reduction, True))["status"] == "ok"
    discovered.add(original)
    assert restarted._post_order(order(original))["reason"] == "durable_order_blocked"
    assert restarted._durable_order_journal().unresolved() == []
    assert restarted._durable_order_journal().inventory() == {"BTC": Decimal(".25")}
    assert restarted._post_order(order(later))["status"] == "ok"
    assert accepted == [original, reduction, later]


class TestBroker:
    __test__ = False

    def __init__(self):
        self.posts = 0
        self.remote = None
        self.venue_fills = []
        self.position = []
        self.open_orders = []

    def submit(self, intent):
        self.posts += 1
        self.remote = {"client_order_id": intent["client_order_id"], "id": "broker-order", "status": "partially_filled"}
        raise TimeoutError

    def lookup(self, cid):
        return self.remote

    def request(self, path, **kwargs):
        return {"/v2/positions": self.position, "/v2/orders": self.open_orders}[path]

    def fills(self):
        return self.venue_fills


def test_stock_restart_and_broker_fill_replay_are_idempotent(tmp_path):
    path = tmp_path / "stock.sqlite3"
    broker = TestBroker()
    journal = OrderJournal(path, "alpaca:paper")
    intent = {"client_order_id": "key", "symbol": "SPY", "side": "buy"}
    assert stocks.submit_once(broker, journal, intent) == "submission_unknown"

    restarted = OrderJournal(path, "alpaca:paper")
    assert stocks.submit_once(broker, restarted, intent) == "intent_already_recorded_or_paused"
    assert broker.posts == 1
    one = Fill("fill-one", "broker-order", "SPY", "buy", Decimal(".1"), Decimal("678.90"))
    two = Fill("fill-two", "broker-order", "SPY", "buy", Decimal(".2"), Decimal("679.15"))
    broker.venue_fills = [one, one]
    broker.position = [{"symbol": "SPY", "qty": ".1"}]
    assert stocks.reconcile(broker, restarted) == ({"SPY": Decimal(".1")}, "order_pending")
    broker.remote["status"] = "filled"
    broker.venue_fills = [one, one, two]
    broker.position[0]["qty"] = ".3"
    assert stocks.reconcile(broker, restarted) == ({"SPY": Decimal(".3")}, None)
    assert stocks.reconcile(broker, restarted) == ({"SPY": Decimal(".3")}, None)
    assert broker.posts == 1
    broker.position[0]["qty"] = ".4"
    assert stocks.reconcile(broker, restarted)[1] == "broker_position_mismatch"


def test_entry_pause_and_account_isolation(tmp_path):
    path = tmp_path / "orders.sqlite3"
    first = OrderJournal(path, "account-one")
    second = OrderJournal(path, "account-two")
    first.set_pause("operator")
    assert not first.reserve("entry", {}, entry=True)
    assert first.reserve("exit", {}, entry=False)
    assert second.reserve("entry", {}, entry=True)
    first.set_pause(None)
    assert not first.reserve("entry", {}, entry=True)   # Exit submission remains unresolved.
    assert first.unresolved() == ["exit"]


def test_paper_adapter_uses_only_paper_host(monkeypatch):
    captured = []
    broker = AlpacaPaper("fake-paper-key", "fake-paper-secret")
    from io import StringIO

    monkeypatch.setattr(broker.opener, "open", lambda request, **kw: captured.append(request) or StringIO("{}"))
    broker.submit({"symbol": "SPY", "qty": "0.1", "side": "buy"})
    assert captured[0].full_url == "https://paper-api.alpaca.markets/v2/orders"
    assert captured[0].get_method() == "POST"


def test_ambiguous_maker_ack_never_falls_back_to_market():
    trader = make_live(lambda *a, **k: None)
    trader.entry_execution_mode = "maker_then_market"
    trader.maker_entry_fallback_to_market = True
    trader._maker_entry_price = lambda *a: 100
    trader._incr_entry_metric = lambda *a: None
    trader.place_limit_order = lambda *a, **k: {"status": "error", "reason": "submission_unknown"}
    trader.place_market_order = lambda *a, **k: pytest.fail("maker response is ambiguous")
    assert trader._place_entry_order("BTC", "buy", .01, leverage=1)["reason"] == "submission_unknown"


def test_stock_shadow_caps_fractional_order_and_refuses_stale_quotes(tmp_path, monkeypatch):
    now = datetime(2026, 11, 16, 16, tzinfo=UTC)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(stocks, "datetime", Frozen)
    dates = [(now.date() - timedelta(days=i)).isoformat() for i in range(21, -1, -1)]
    bars = [
        {"t": day + "T05:00:00Z", "c": 101 if day == dates[-2] else 100} for day in dates[:-1]
    ]
    quote = {"bp": 99.99, "ap": 100, "t": now.isoformat()}

    class Market(TestBroker):
        def request(self, path, **kwargs):
            return {
                "/v2/account": {"status": "ACTIVE", "equity": "1000", "last_equity": "1000",
                                "cash": "1000", "buying_power": "2000"},
                "/v2/clock": {"is_open": True, "timestamp": now.isoformat(),
                              "next_close": (now + timedelta(hours=5)).isoformat()},
                "/v2/calendar": [{"date": d} for d in dates],
                "/v2/assets/SPY": {"class": "us_equity", "tradable": True, "fractionable": True},
                "/v2/stocks/SPY/bars": {"bars": bars}, "/v2/stocks/SPY/quotes/latest": {"quote": quote},
                "/v2/positions": [], "/v2/orders": [],
            }[path]

    broker, journal = Market(), OrderJournal(tmp_path / "stock.sqlite3", "paper")
    result = stocks.tick(broker, journal, symbol="SPY")
    assert result["state"] == "shadow_intent"
    assert result["intent"]["qty"] == "0.750000"
    assert result["intent"]["extended_hours"] is False
    assert not journal.orders() and not broker.posts
    quote["t"] = (now - timedelta(minutes=5)).isoformat()
    assert stocks.tick(broker, journal, symbol="SPY", orders_enabled=True)["state"] == "quote_not_usable"
    assert not broker.posts
