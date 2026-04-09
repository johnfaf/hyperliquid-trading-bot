import src.core.api_manager as api_manager


def test_ws_gap_warn_threshold_defaults_to_30s(monkeypatch):
    monkeypatch.delattr(api_manager.config, "WS_FEED_GAP_WARN_MS", raising=False)
    ws = api_manager.HyperliquidWebSocket()
    assert ws._gap_warn_threshold_ms == 30000.0


def test_transient_disconnect_classifier_matches_remote_host_lost():
    assert api_manager.HyperliquidWebSocket._is_transient_disconnect_error(
        "Connection to remote host was lost. - goodbye"
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

