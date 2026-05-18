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


def test_analyze_fills_scores_closed_outcomes_not_raw_hl_fill_cap():
    discovery = TraderDiscovery.__new__(TraderDiscovery)
    fills = [
        _fill("buy", 0.0, idx * 1_000, size=0.1, price=1000.0)
        for idx in range(1997)
    ]
    fills.extend(
        [
            _fill("sell", 1.0, 1_997_000, size=0.1, price=1000.0),
            _fill("sell", 2.0, 1_998_000, size=0.1, price=1000.0),
            _fill("sell", 3.0, 1_999_000, size=0.1, price=1000.0),
        ]
    )

    metrics = discovery._analyze_fills(fills)

    assert metrics["raw_fill_count"] == 2000
    assert metrics["sample_is_capped"] is True
    assert metrics["closed_trade_count"] == 3
    assert metrics["total_trades"] == 3
    assert metrics["win_rate"] == 1.0
    assert metrics["profit_factor"] is None


def _ta(*, closed, raw=None, win_rate=0.0, losing=0):
    """Minimal trade_analysis dict for the anomaly scorer."""
    return {
        "closed_trade_count": closed,
        "raw_fill_count": raw if raw is not None else closed,
        "win_rate": win_rate,
        "losing_trades": losing,
    }


def test_anomaly_scorer_flags_perfect_winrate_over_real_sample():
    d = TraderDiscovery.__new__(TraderDiscovery)
    score, reason = d._statistical_anomaly_score(
        _ta(closed=40, raw=120, win_rate=1.0, losing=0)
    )
    assert score >= 3  # >= config.BOT_THRESHOLD -> excluded from humans
    assert "perfect_winrate" in reason


def test_anomaly_scorer_flags_never_realizes_loss():
    d = TraderDiscovery.__new__(TraderDiscovery)
    # 0.90 win rate is below the perfect threshold, but 0 losing trades
    # over a real sample means losses are held open forever (uncopyable).
    score, reason = d._statistical_anomaly_score(
        _ta(closed=25, raw=80, win_rate=0.90, losing=0)
    )
    assert score >= 3
    assert "never_realizes_loss" in reason


def test_anomaly_scorer_flags_zero_evidence_junk():
    d = TraderDiscovery.__new__(TraderDiscovery)
    score, reason = d._statistical_anomaly_score(
        _ta(closed=0, raw=50, win_rate=0.0, losing=0)
    )
    assert score >= 3
    assert "no_evidence" in reason


def test_anomaly_scorer_ignores_thin_perfect_record():
    """A 3-trade 100% record is thin data, not a proven bot -- it must NOT
    be flagged (upstream 'insufficient' handling owns this case; this also
    guards test_analyze_fills_* which legitimately yields win_rate=1.0)."""
    d = TraderDiscovery.__new__(TraderDiscovery)
    score, reason = d._statistical_anomaly_score(
        _ta(closed=3, raw=2000, win_rate=1.0, losing=0)
    )
    assert score == 0
    assert reason == ""


def test_anomaly_scorer_passes_normal_human():
    d = TraderDiscovery.__new__(TraderDiscovery)
    score, reason = d._statistical_anomaly_score(
        _ta(closed=60, raw=140, win_rate=0.55, losing=27)
    )
    assert score == 0
    assert reason == ""


def test_discovery_cycle_refuses_to_promote_bot_like_sources(monkeypatch):
    discovery = TraderDiscovery.__new__(TraderDiscovery)
    discovery.known_traders = {}
    bot_address = "0x" + "a" * 40
    inactive_updates = []

    monkeypatch.setattr(
        discovery,
        "discover_top_traders",
        lambda: [{"address": bot_address, "metadata": {}}],
    )
    monkeypatch.setattr(discovery, "_fast_prescreen", lambda _address: True)
    monkeypatch.setattr(
        discovery,
        "analyze_trader",
        lambda address: {
            "address": address,
            "account_value": 10_000,
            "bot_score": 3,
        },
    )
    monkeypatch.setattr(trader_discovery.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(trader_discovery.db, "get_known_bot_addresses", lambda: {bot_address})
    monkeypatch.setattr(trader_discovery.db, "log_research_cycle", lambda **_kwargs: None)
    monkeypatch.setattr(
        trader_discovery.db,
        "upsert_trader",
        lambda **kwargs: inactive_updates.append(kwargs),
    )

    result = discovery.run_discovery_cycle()

    assert result["human_traders"] == 0
    assert result["bot_like_traders"] == 1
    assert result["promoted_bots"] == 0
    assert result["final_pool"] == 0
    assert result["bots_marked_inactive"] == 1
    assert inactive_updates[0]["is_active"] is False
