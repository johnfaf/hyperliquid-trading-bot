import threading
from types import SimpleNamespace

import src.notifications.telegram_bot as telegram_bot
from src.trading.live_trader import LiveTrader


def test_snapshot_balance_alerts_when_free_margin_stays_zero(monkeypatch, caplog):
    trader = LiveTrader.__new__(LiveTrader)
    trader.live_requested = True
    trader.dry_run = False
    trader.public_address = "0x123"
    trader.max_order_usd = 35.0
    trader.kill_switch_active = False
    trader.status_reason = "live_ready"
    trader._last_balance_snapshot = {
        "perps_margin": None,
        "free_margin": None,
        "spot_usdc": None,
        "total": None,
        "timestamp": None,
    }
    trader._balance_log_interval_s = 999999.0
    trader._last_balance_log_ts = 0.0
    trader._last_known_free_margin = None
    trader._free_margin_zero_since_ts = 0.0
    trader._last_free_margin_alert_ts = 0.0
    trader._free_margin_alert_cooldown_s = 300.0
    trader._recent_order_hashes = {}
    trader._order_dedup_lock = threading.Lock()
    trader._last_hash_cleanup_ts = 0.0
    trader._HASH_CLEANUP_INTERVAL = 60.0
    trader._ORDER_DEDUP_WINDOW = 30.0
    # S3 drawdown tracker attrs — defaults disable the check.
    trader._max_drawdown_usd = 0.0
    trader._drawdown_window_s = 24 * 3600.0
    trader._equity_samples = __import__("collections").deque()
    trader._equity_samples_lock = threading.Lock()
    trader._peak_equity_since_start = 0.0
    trader.is_deployable = lambda: True

    def _post(payload, **kwargs):
        req_type = payload.get("type")
        if req_type == "clearinghouseState":
            return {
                "withdrawable": "0",
                "marginSummary": {
                    "accountValue": "50",
                    "totalMarginUsed": "50",
                },
            }
        if req_type == "spotClearinghouseState":
            return {"balances": [{"coin": "USDC", "total": "0"}]}
        raise AssertionError(f"unexpected payload: {payload}")

    trader.api_manager = SimpleNamespace(post=_post)

    alerts = []
    monkeypatch.setattr(telegram_bot, "is_configured", lambda: True)
    monkeypatch.setattr(
        telegram_bot,
        "notify_live_margin_blocked",
        lambda **kwargs: alerts.append(kwargs),
    )

    with caplog.at_level("WARNING", logger="src.trading.live_trader"):
        snapshot = trader.snapshot_balance(log=False)

    assert snapshot["free_margin"] == 0.0
    assert trader.status_reason == "no_free_margin_available"
    assert len(alerts) == 1
    assert alerts[0].get("resolved", False) is False
    assert "Live free margin is $0.00" in caplog.text

    trader.snapshot_balance(log=False)
    assert len(alerts) == 1  # cooldown suppresses duplicates


def test_snapshot_balance_alerts_when_free_margin_recovers(monkeypatch):
    trader = LiveTrader.__new__(LiveTrader)
    trader.live_requested = True
    trader.dry_run = False
    trader.public_address = "0x123"
    trader.max_order_usd = 35.0
    trader.kill_switch_active = False
    trader.status_reason = "no_free_margin_available"
    trader._last_balance_snapshot = {
        "perps_margin": None,
        "free_margin": None,
        "spot_usdc": None,
        "total": None,
        "timestamp": None,
    }
    trader._balance_log_interval_s = 999999.0
    trader._last_balance_log_ts = 0.0
    trader._last_known_free_margin = 0.0
    trader._free_margin_zero_since_ts = 1.0
    trader._last_free_margin_alert_ts = 0.0
    trader._free_margin_alert_cooldown_s = 0.0
    trader._recent_order_hashes = {}
    trader._order_dedup_lock = threading.Lock()
    trader._last_hash_cleanup_ts = 0.0
    trader._HASH_CLEANUP_INTERVAL = 60.0
    trader._ORDER_DEDUP_WINDOW = 30.0
    # S3 drawdown tracker attrs — defaults disable the check.
    trader._max_drawdown_usd = 0.0
    trader._drawdown_window_s = 24 * 3600.0
    trader._equity_samples = __import__("collections").deque()
    trader._equity_samples_lock = threading.Lock()
    trader._peak_equity_since_start = 0.0
    trader.is_deployable = lambda: True

    def _post(payload, **kwargs):
        req_type = payload.get("type")
        if req_type == "clearinghouseState":
            return {
                "withdrawable": "12.5",
                "marginSummary": {
                    "accountValue": "50",
                    "totalMarginUsed": "37.5",
                },
            }
        if req_type == "spotClearinghouseState":
            return {"balances": [{"coin": "USDC", "total": "0"}]}
        raise AssertionError(f"unexpected payload: {payload}")

    trader.api_manager = SimpleNamespace(post=_post)

    alerts = []
    monkeypatch.setattr(telegram_bot, "is_configured", lambda: True)
    monkeypatch.setattr(
        telegram_bot,
        "notify_live_margin_blocked",
        lambda **kwargs: alerts.append(kwargs),
    )

    snapshot = trader.snapshot_balance(log=False)

    assert snapshot["free_margin"] == 12.5
    assert trader.status_reason == "live_ready"
    assert trader._free_margin_zero_since_ts == 0.0
    assert len(alerts) == 1
    assert alerts[0]["resolved"] is True
