import src.core.api_manager as api_manager


def test_ws_gap_warn_threshold_defaults_to_30s(monkeypatch):
    monkeypatch.delattr(api_manager.config, "WS_FEED_GAP_WARN_MS", raising=False)
    ws = api_manager.HyperliquidWebSocket()
    assert ws._gap_warn_threshold_ms == 30000.0


def test_transient_disconnect_classifier_matches_remote_host_lost():
    assert api_manager.HyperliquidWebSocket._is_transient_disconnect_error(
        "Connection to remote host was lost. - goodbye"
    )
    assert api_manager.HyperliquidWebSocket._is_transient_disconnect_error(
        "fin=1 opcode=8 data=b'\\x03\\xe8Expired'"
    )
    assert not api_manager.HyperliquidWebSocket._is_transient_disconnect_error(
        "invalid subscription payload"
    )


def test_on_error_transient_disconnect_logs_info(monkeypatch):
    ws = api_manager.HyperliquidWebSocket()
    ws._running = True
    ws._connected = True
    infos = []
    warnings = []
    monkeypatch.setattr(api_manager.logger, "info", lambda msg, *args: infos.append(msg % args if args else msg))
    monkeypatch.setattr(api_manager.logger, "warning", lambda msg, *args: warnings.append(msg % args if args else msg))

    ws._on_error(None, "Connection to remote host was lost.")

    assert ws._connected is False
    assert any("transient disconnect" in m for m in infos)
    assert warnings == []


def test_on_error_expired_disconnect_requests_fast_reconnect(monkeypatch):
    ws = api_manager.HyperliquidWebSocket()
    ws._running = True
    infos = []
    warnings = []
    monkeypatch.setattr(api_manager.logger, "info", lambda msg, *args: infos.append(msg % args if args else msg))
    monkeypatch.setattr(api_manager.logger, "warning", lambda msg, *args: warnings.append(msg % args if args else msg))

    ws._on_error(None, "fin=1 opcode=8 data=b'\\x03\\xe8Expired'")

    wait, reason = ws._consume_reconnect_wait()
    assert wait == 1.0
    assert reason == "expired WebSocket session"
    assert any("session expired" in m.lower() for m in infos)
    assert warnings == []


def test_gap_under_default_threshold_does_not_warn(monkeypatch):
    ws = api_manager.HyperliquidWebSocket()
    ws._last_msg_time = 100.0
    ws._gap_warn_threshold_ms = 30000.0
    ws._gap_warn_grace_until = 0.0
    warned = []
    monkeypatch.setattr(api_manager.logger, "warning", lambda msg, *args: warned.append(msg % args if args else msg))
    monkeypatch.setattr(api_manager.time, "time", lambda: 113.0)

    ws._on_message(None, "{}")

    assert warned == []


def test_candle_snapshot_server_errors_open_request_type_cooldown(monkeypatch):
    mgr = api_manager.APIManager()
    api_calls = []
    acquire_calls = []

    monkeypatch.setattr(
        mgr.bucket,
        "acquire",
        lambda priority, timeout=30: acquire_calls.append((priority, timeout)) or True,
    )

    def fake_do_request(*args, **kwargs):
        api_calls.append(kwargs["req_type"])
        return None, "server_error"

    monkeypatch.setattr(mgr, "_do_request", fake_do_request)

    payload = {"type": "candleSnapshot", "req": {"coin": "BTC", "interval": "1h"}}
    assert mgr.post(payload, priority=api_manager.Priority.LOW, cache_response=False) is None

    cooldown_until = mgr._req_type_cooldown_until.get("candleSnapshot", 0.0)
    assert cooldown_until > 0.0

    # Subsequent calls should short-circuit locally instead of hitting the upstream again.
    assert mgr.post(payload, priority=api_manager.Priority.LOW, cache_response=False) is None
    assert mgr.post(payload, priority=api_manager.Priority.LOW, cache_response=False) is None
    assert api_calls == ["candleSnapshot"]
    assert len(acquire_calls) == 1


def test_candle_snapshot_server_errors_fail_fast_without_retry_sleep(monkeypatch):
    mgr = api_manager.APIManager()
    warns = []
    sleeps = []

    class FakeResponse:
        status_code = 500
        text = "boom"

    monkeypatch.setattr(mgr.bucket, "acquire", lambda priority, timeout=30: True)
    monkeypatch.setattr(
        api_manager.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr(api_manager.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        api_manager.logger,
        "warning",
        lambda msg, *args: warns.append(msg % args if args else msg),
    )

    payload = {"type": "candleSnapshot", "req": {"coin": "BTC", "interval": "1h"}}
    assert mgr.post(payload, priority=api_manager.Priority.LOW, cache_response=False) is None
    assert sleeps == []
    assert any("fail-fast cooldown trigger" in msg for msg in warns)
    assert mgr._req_type_cooldown_until.get("candleSnapshot", 0.0) > 0.0


def test_candle_snapshot_success_resets_request_type_failure_streak(monkeypatch):
    mgr = api_manager.APIManager()
    monkeypatch.setattr(mgr.bucket, "acquire", lambda priority, timeout=30: True)

    responses = iter(
        [
            ([{"t": 1, "c": "1"}], None),
            (None, "server_error"),
        ]
    )

    monkeypatch.setattr(mgr, "_do_request", lambda *args, **kwargs: next(responses))
    payload = {"type": "candleSnapshot", "req": {"coin": "ETH", "interval": "1h"}}

    assert mgr.post(payload, priority=api_manager.Priority.LOW, cache_response=False) == [{"t": 1, "c": "1"}]
    assert mgr._req_type_failures.get("candleSnapshot") == 0
    assert mgr._req_type_cooldown_until.get("candleSnapshot", 0.0) == 0.0

    assert mgr.post(payload, priority=api_manager.Priority.LOW, cache_response=False) is None
    assert mgr._req_type_cooldown_until.get("candleSnapshot", 0.0) > 0.0
