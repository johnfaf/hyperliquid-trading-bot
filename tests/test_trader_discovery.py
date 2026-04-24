from src.discovery.trader_discovery import TraderDiscovery
import src.discovery.trader_discovery as trader_discovery


def _fill(side, closed_pnl, time_ms, *, size=1.0, price=1000.0, coin="BTC"):
    return {
        "coin": coin,
        "side": side,
        "closed_pnl": closed_pnl,
        "time": time_ms,
        "size": size,
        "price": price,
    }


def test_arb_detector_ignores_tiny_scalper_round_trips():
    fills = []
    for idx in range(12):
        fills.append(_fill("buy" if idx % 2 == 0 else "sell", 1.0, idx * 1000, size=0.01, price=100.0))

    assert TraderDiscovery._detect_arb_pattern(fills) is False


def test_arb_detector_requires_repeated_meaningful_pairs():
    fills = []
    for idx in range(6):
        base_ts = idx * 10_000
        fills.append(_fill("buy", 0.0, base_ts, size=1.0, price=1000.0))
        fills.append(_fill("sell", 2.0, base_ts + 1_000, size=1.0, price=1002.0))

    assert TraderDiscovery._detect_arb_pattern(fills) is True


def test_detect_leaderboard_schema_cache_is_thread_safe():
    trader_discovery._leaderboard_schema_key = None
    payload = {"leaderboardRows": [{"address": "0xabc"}]}

    results = []

    def _call():
        results.append(trader_discovery._detect_leaderboard_schema(payload))

    import threading

    threads = [threading.Thread(target=_call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(entries == payload["leaderboardRows"] for entries, _ in results)
    assert trader_discovery._leaderboard_schema_key == "leaderboardRows"


def test_parse_leaderboard_masks_display_names():
    payload = {
        "leaderboardRows": [
            {
                "ethAddress": "0x1234567890abcdef1234567890abcdef12345678",
                "displayName": "Alice Trader",
                "accountValue": "123.4",
            }
        ]
    }

    discovery = TraderDiscovery.__new__(TraderDiscovery)
    traders = discovery._parse_leaderboard(payload)

    assert len(traders) == 1
    assert traders[0]["metadata"]["display_name"] == "A***r"
