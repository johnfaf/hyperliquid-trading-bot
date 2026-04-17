from src.discovery.trader_discovery import TraderDiscovery


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
