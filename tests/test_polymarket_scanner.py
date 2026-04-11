import logging

from src.data.polymarket_scanner import OddsMovement, PolymarketScanner


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


def _gamma_market(mid: str, title: str, volume: str, liquidity: str):
    return {
        "conditionId": mid,
        "question": title,
        "description": "",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.61","0.39"]',
        "clobTokenIds": f'["{mid}-yes","{mid}-no"]',
        "volumeNum": volume,
        "liquidityNum": liquidity,
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
        return [
            {"condition_id": f"m{calls['n']}-{i}"}
            for i in range(100)
        ]

    monkeypatch.setattr(scanner, "_fetch_json", _fake_fetch_json)
    scanner._fetch_raw_markets()
    assert calls["n"] == 2  # (50 + 99) // 100 + 1

    scanner.max_markets_per_scan = 250
    calls["n"] = 0
    scanner._fetch_raw_markets()
    assert calls["n"] == 4  # (250 + 99) // 100 + 1


def test_scan_markets_parses_gamma_market_string_fields(monkeypatch):
    scanner = PolymarketScanner(
        config={
            "min_volume_threshold": 1000.0,
            "min_liquidity_threshold": 500.0,
            "max_markets_per_scan": 5,
        }
    )
    monkeypatch.setattr(
        scanner,
        "_fetch_raw_markets",
        lambda: [_gamma_market("m1", "Will Bitcoin close above 120k?", "12000.5", "7500.25")],
    )

    markets = scanner.scan_markets()

    assert len(markets) == 1
    market = markets[0]
    assert market.market_id == "m1"
    assert market.token_id == "m1-yes"
    assert market.current_prices == [0.61, 0.39]
    assert market.outcomes == ["Yes", "No"]
    assert market.volume_24h == 12000.5
    assert market.liquidity == 7500.25


def test_get_market_sentiment_reuses_recent_empty_scan_without_refetch(monkeypatch):
    scanner = PolymarketScanner(
        config={
            "min_volume_threshold": 10_000.0,
            "min_liquidity_threshold": 5_000.0,
            "scan_interval_seconds": 180,
        }
    )
    calls = {"n": 0}

    def _fake_fetch():
        calls["n"] += 1
        return [
            _raw_market("m1", "Will Bitcoin close above 120k?", 0.0, 0.0),
        ]

    monkeypatch.setattr(scanner, "_fetch_raw_markets", _fake_fetch)

    markets = scanner.scan_markets()
    sentiment = scanner.get_market_sentiment()

    assert markets == []
    assert calls["n"] == 1
    assert sentiment["sentiment"] == "neutral"
    assert sentiment["markets_analyzed"] == 0


def test_generate_signals_logs_info_when_markets_filtered_out(monkeypatch, caplog):
    scanner = PolymarketScanner(
        config={
            "min_volume_threshold": 10_000.0,
            "min_liquidity_threshold": 5_000.0,
            "scan_interval_seconds": 180,
        }
    )
    monkeypatch.setattr(
        scanner,
        "_fetch_raw_markets",
        lambda: [_raw_market("m1", "Will Bitcoin close above 120k?", 0.0, 0.0)],
    )

    with caplog.at_level(logging.INFO):
        signals = scanner.generate_signals()

    assert signals == []
    assert "No Polymarket markets passed filters" in caplog.text
    assert "No markets found in scan" not in caplog.text


def test_scan_markets_falls_back_to_active_crypto_markets_when_strict_filters_zero_out(monkeypatch, caplog):
    scanner = PolymarketScanner(
        config={
            "min_volume_threshold": 10_000.0,
            "min_liquidity_threshold": 5_000.0,
            "max_markets_per_scan": 2,
        }
    )
    monkeypatch.setattr(
        scanner,
        "_fetch_raw_markets",
        lambda: [
            _raw_market("m1", "Will Bitcoin close above 120k?", 4000.0, 3000.0),
            _raw_market("m2", "Will Ethereum ETF be approved?", 2500.0, 500.0),
            _raw_market("m3", "Will Solana hit new ATH?", 1500.0, 250.0),
        ],
    )

    with caplog.at_level(logging.INFO):
        markets = scanner.scan_markets()

    assert [market.market_id for market in markets] == ["m1", "m2"]
    assert scanner._last_filtered_market_count == 0
    assert scanner._markets_tracked == 2
    assert "falling back to 2 active crypto markets" in caplog.text


def test_generate_signals_inverts_bearish_market_odds_moves(monkeypatch):
    scanner = PolymarketScanner(
        config={
            "min_volume_threshold": 1000.0,
            "min_liquidity_threshold": 500.0,
        }
    )
    monkeypatch.setattr(
        scanner,
        "_fetch_raw_markets",
        lambda: [_raw_market("m1", "Will Bitcoin crash below 50k?", 4000.0, 3000.0)],
    )

    markets = scanner.scan_markets()
    assert len(markets) == 1
    movement = OddsMovement(
        market_id="m1",
        title="Will Bitcoin crash below 50k?",
        direction="up",
        magnitude=0.15,
        timeframe="1h",
        current_probability=0.35,
        volume_move=4000.0,
        smart_money_score=0.5,
    )
    monkeypatch.setattr(scanner, "detect_odds_movements", lambda markets: [movement])

    signals = scanner.generate_signals()

    assert len(signals) == 1
    assert signals[0]["coin"] == "BTC"
    assert signals[0]["side"] == "short"
