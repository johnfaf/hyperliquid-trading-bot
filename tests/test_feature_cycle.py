from contextlib import contextmanager

from src.core import live_execution
from src.core.cycles import feature_cycle
from src.data import database as db
from src.data import hyperliquid_client as hl


def test_watched_coins_bootstrap_top_coins_is_capped(monkeypatch):
    coins = ["BTC", "ETH", "SOL", "DOGE", "XRP", "SUI", "LINK", "AVAX", "ARB", "OP"]

    @contextmanager
    def empty_db(*, for_read=False):
        raise RuntimeError("no local strategy/open-position DB in this unit test")
        yield

    monkeypatch.setattr(feature_cycle.config, "FEATURE_STORE_COINS", "")
    monkeypatch.setattr(feature_cycle, "_MAX_COINS", 30)
    monkeypatch.setattr(feature_cycle, "_BOOTSTRAP_TOP_COINS", 8)
    monkeypatch.setattr(db, "get_connection", empty_db)
    monkeypatch.setattr(hl, "get_all_coins", lambda: list(coins))

    watched = feature_cycle._get_watched_coins()

    assert len(watched) == 8
    assert {"BTC", "ETH"}.issubset(set(watched))


def test_watched_coins_include_execution_positions(monkeypatch):
    @contextmanager
    def empty_db(*, for_read=False):
        raise RuntimeError("no local strategy/open-position DB in this unit test")
        yield

    monkeypatch.setattr(feature_cycle.config, "FEATURE_STORE_COINS", "")
    monkeypatch.setattr(feature_cycle, "_MAX_COINS", 30)
    monkeypatch.setattr(feature_cycle, "_BOOTSTRAP_TOP_COINS", 8)
    monkeypatch.setattr(db, "get_connection", empty_db)
    monkeypatch.setattr(
        live_execution,
        "get_execution_open_positions",
        lambda container: [{"coin": "SOL"}, {"symbol": "DOGE"}],
    )
    monkeypatch.setattr(
        hl,
        "get_all_coins",
        lambda: ["BTC", "ETH", "XRP", "SUI", "LINK", "AVAX", "ARB", "OP"],
    )

    watched = feature_cycle._get_watched_coins(container=object())

    assert {"BTC", "ETH", "SOL", "DOGE"}.issubset(set(watched))
