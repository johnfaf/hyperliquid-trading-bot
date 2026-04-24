from src.exchanges.hyperliquid_adapter import HyperliquidAdapter


def test_get_top_traders_keeps_explicit_zero_account_value(monkeypatch):
    adapter = HyperliquidAdapter()
    monkeypatch.setattr(
        "src.exchanges.hyperliquid_adapter.hl.get_leaderboard",
        lambda: [
            {
                "ethAddress": "0x1234567890abcdef1234567890abcdef12345678",
                "displayName": "Alice Trader",
                "accountValue": "0",
                "totalPnl": "9999",
            }
        ],
    )

    traders = adapter.get_top_traders(limit=10)

    assert len(traders) == 1
    assert traders[0].pnl_total == 0.0
    assert traders[0].display_name == "A***r"
    assert traders[0].raw_data["displayName"] == "A***r"


def test_get_market_data_tolerates_missing_mark_price_without_falling_back_to_mid(monkeypatch):
    adapter = HyperliquidAdapter()
    monkeypatch.setattr(
        "src.exchanges.hyperliquid_adapter.hl.get_asset_contexts",
        lambda: {
            "BTC": {
                "funding": "0.0001",
                "openInterest": "12345",
                "dayNtlVlm": "999999",
                "markPx": None,
                "oraclePx": "72010.5",
            }
        },
    )
    monkeypatch.setattr(
        "src.exchanges.hyperliquid_adapter.hl.get_all_mids",
        lambda: {"BTC": "72000.0"},
    )

    markets = adapter.get_market_data(["BTC"])

    assert len(markets) == 1
    assert markets[0].mid_price == 72000.0
    assert markets[0].mark_price == 0.0
    assert markets[0].index_price == 72010.5
    assert markets[0].funding_rate == 0.0001
    assert markets[0].open_interest == 12345.0
