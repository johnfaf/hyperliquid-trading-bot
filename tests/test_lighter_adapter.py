from src.exchanges.lighter_adapter import LighterAdapter, VenueState


def test_get_market_data_parses_bulk_order_book_details_once(monkeypatch):
    adapter = LighterAdapter()
    adapter.state = VenueState.HEALTHY
    adapter._symbol_map = {"1": "BTC", "2": "ETH"}
    monkeypatch.setattr(adapter, "_ensure_markets_loaded", lambda: None)

    calls = []

    def _fake_get(path, params=None, retries=2, quiet=False):
        calls.append((path, params, retries, quiet))
        assert path == "/orderBookDetails"
        return {
            "code": 0,
            "order_book_details": [
                {
                    "market_id": 1,
                    "symbol": "BTC",
                    "last_trade_price": "72260.5",
                    "daily_quote_token_volume": "937386224.286584",
                    "open_interest": "1613.79791",
                },
                {
                    "market_id": 2,
                    "symbol": "ETH",
                    "last_trade_price": "3520.25",
                    "daily_quote_token_volume": "123456789.0",
                    "open_interest": "987.65",
                },
            ],
        }

    monkeypatch.setattr(adapter, "_get", _fake_get)

    markets = adapter.get_market_data(coins=["BTC", "ETH"])

    assert len(calls) == 1
    assert [market.coin for market in markets] == ["BTC", "ETH"]

    btc_market = markets[0]
    assert btc_market.exchange == "lighter"
    assert btc_market.coin == "BTC"
    assert btc_market.mid_price == 72260.5
    assert btc_market.mark_price == 72260.5
    assert btc_market.index_price == 72260.5
    assert btc_market.volume_24h == 937386224.286584
    assert btc_market.open_interest == 1613.79791
    assert btc_market.spread_bps == 0.0
