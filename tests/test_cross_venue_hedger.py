from src.trading.cross_venue_hedger import CrossVenueHedger


def test_check_and_hedge_filters_positions_without_coin(monkeypatch):
    hedger = CrossVenueHedger(
        {
            "dry_run": True,
            "binance_enabled": True,
            "bybit_enabled": False,
            "crash_confidence": 0.5,
            "rate_limit_ms": 0,
        }
    )
    placed = []
    monkeypatch.setattr(
        hedger,
        "_place_hedges",
        lambda coin, position: placed.append((coin, position["size"])) or True,
    )

    result = hedger.check_and_hedge(
        {"regime": "crash", "confidence": 0.9},
        [
            {"size": 1.0, "side": "long"},
            {"coin": "", "size": 2.0, "side": "long"},
            {"coin": " btc ", "size": 3.0, "side": "long"},
        ],
    )

    assert placed == [("BTC", 3.0)]
    assert result["coins_affected"] == ["BTC"]
    assert result["hedges_placed"] == 1
