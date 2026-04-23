import contextlib
import sqlite3

import requests

from src.data import database as db
from src.data.polymarket_history import (
    CLOB_API,
    DATA_API,
    PolymarketHistoricalDownloader,
    store_markets,
)
from src.learning.schema import ensure_sqlite_schema


@contextlib.contextmanager
def _sqlite_ctx(conn):
    yield conn
    conn.commit()


def _memory_db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_sqlite_schema(conn)
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    return conn


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}", response=self)

    def json(self):
        return self._payload


def test_backfill_recent_trades_uses_public_data_api_and_preserves_internal_market_id(monkeypatch):
    conn = _memory_db(monkeypatch)
    store_markets(
        [
            {
                "id": "2023505",
                "conditionId": "0x791412c365a0543d318a503db67cb6310218c691d835a0cf108caeb863a6c7fe",
                "question": "Total Kills Over/Under 30.5 in Game 2?",
                "active": True,
                "closed": False,
            }
        ],
        observed_at_ms=1_000,
    )

    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        assert url == f"{DATA_API}/trades"
        return _FakeResponse(
            [
                {
                    "proxyWallet": "0xaaa",
                    "side": "BUY",
                    "asset": "asset-1",
                    "conditionId": "0x791412c365a0543d318a503db67cb6310218c691d835a0cf108caeb863a6c7fe",
                    "size": 1.0,
                    "price": 0.59,
                    "timestamp": 1776948146,
                    "transactionHash": "0xhash",
                },
                {
                    "proxyWallet": "0xaaa",
                    "side": "BUY",
                    "asset": "asset-2",
                    "conditionId": "0x791412c365a0543d318a503db67cb6310218c691d835a0cf108caeb863a6c7fe",
                    "size": 2.0,
                    "price": 0.61,
                    "timestamp": 1776948147,
                    "transactionHash": "0xhash",
                },
            ]
        )

    monkeypatch.setattr("src.data.polymarket_history.requests.get", _fake_get)

    downloader = PolymarketHistoricalDownloader(trade_source="data_api", max_retries=1)
    inserted = downloader.backfill_recent_trades(["2023505"], per_market=5, max_markets=1)

    assert inserted == 2
    assert calls[0][1]["market"] == "0x791412c365a0543d318a503db67cb6310218c691d835a0cf108caeb863a6c7fe"

    rows = conn.execute(
        "SELECT trade_id, market_id, token_id FROM polymarket_trades ORDER BY token_id"
    ).fetchall()
    assert [row["market_id"] for row in rows] == ["2023505", "2023505"]
    assert [row["token_id"] for row in rows] == ["asset-1", "asset-2"]
    assert rows[0]["trade_id"] != rows[1]["trade_id"]


def test_fetch_market_trades_falls_back_from_clob_401_to_public_data_api(monkeypatch):
    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        if url == f"{CLOB_API}/trades":
            return _FakeResponse({"error": "unauthorized"}, status_code=401)
        if url == f"{DATA_API}/trades":
            return _FakeResponse(
                [
                    {
                        "proxyWallet": "0xbbb",
                        "side": "SELL",
                        "asset": "asset-3",
                        "conditionId": "0x16a2b3df1bf89e800eaf6951e6eed12234aa3d0e77b299523612773d34af1db2",
                        "size": 4.0,
                        "price": 0.42,
                        "timestamp": 1776948200,
                        "transactionHash": "0xother",
                    }
                ]
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("src.data.polymarket_history.requests.get", _fake_get)

    downloader = PolymarketHistoricalDownloader(trade_source="clob", max_retries=1)
    trades = downloader.fetch_market_trades(
        {
            "id": "1294692",
            "conditionId": "0x16a2b3df1bf89e800eaf6951e6eed12234aa3d0e77b299523612773d34af1db2",
        },
        limit=3,
    )

    assert len(trades) == 1
    assert trades[0]["market_id"] == "1294692"
    assert calls[0][0] == f"{CLOB_API}/trades"
    assert calls[1][0] == f"{DATA_API}/trades"
