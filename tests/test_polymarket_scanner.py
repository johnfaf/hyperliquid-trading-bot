from src.data.polymarket_scanner import PolymarketScanner


def _raw_market(mid: str, title: str, volume: float, liquidity: float):
    return {
        "condition_id": mid,
        "question": title,
        "description": "",
        "outcomes": ["Yes", "No"],
        "tokens": [
            {"token_id": f"{mid}-yes", "price": 0.55},
            {"token_id": f"{mid}-no", "price": 0.45},
        ],
        "volume_24hr": volume,
        "liquidity": liquidity,
        "category": "crypto",
    }


def test_scan_markets_filters_and_caps_top_markets(monkeypatch):
    scanner = PolymarketScanner(
        config={
            "min_volume_threshold": 1000.0,
            "min_liquidity_threshold": 500.0,
            "max_markets_per_scan": 2,
        }
    )
    monkeypatch.setattr(
        scanner,
        "_fetch_raw_markets",
        lambda: [
            _raw_market("m1", "Will Bitcoin close above 120k?", 4000.0, 3000.0),
            _raw_market("m2", "Will Ethereum ETF be approved?", 3000.0, 900.0),
            _raw_market("m3", "Will Solana hit new ATH?", 2000.0, 700.0),
            _raw_market("m4", "Will BTC drop below 40k?", 800.0, 900.0),  # below volume threshold
        ],
    )

    markets = scanner.scan_markets()

    assert len(markets) == 2
    assert [m.market_id for m in markets] == ["m1", "m2"]
    assert scanner._markets_tracked == 2
    assert len(scanner._market_cache) == 2
    assert scanner._crypto_markets_found == 2


def test_fetch_raw_markets_page_budget_scales_with_market_cap(monkeypatch):
    scanner = PolymarketScanner(config={"max_markets_per_scan": 50})
    calls = {"n": 0}

    def _fake_fetch_json(_url: str):
        calls["n"] += 1
        return {"data": [{"condition_id": f"m{calls['n']}"}], "next_cursor": "NEXT"}

    monkeypatch.setattr(scanner, "_fetch_json", _fake_fetch_json)
    scanner._fetch_raw_markets()
    assert calls["n"] == 2  # (50 + 99) // 100 + 1

    scanner.max_markets_per_scan = 250
    calls["n"] = 0
    scanner._fetch_raw_markets()
    assert calls["n"] == 4  # (250 + 99) // 100 + 1
