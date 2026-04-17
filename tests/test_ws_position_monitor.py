import logging

from src.notifications.ws_position_monitor import (
    PositionMonitor,
    _TransientWebSocketLibraryLogFilter,
)


def test_default_gap_warn_threshold_is_30s_when_not_overridden(monkeypatch):
    import src.notifications.ws_position_monitor as monitor_module

    monkeypatch.delattr(monitor_module.config, "POSITION_MONITOR_GAP_WARN_S", raising=False)
    monkeypatch.setattr(
        monitor_module.config,
        "POSITION_MONITOR_WATCHDOG_TIMEOUT_S",
        30.0,
        raising=False,
    )

    monitor = monitor_module.PositionMonitor()
    assert monitor._gap_warn_threshold_s == 30.0


def test_watchdog_trigger_uses_transport_activity_and_cooldown():
    monitor = PositionMonitor()
    monitor._connected = True
    monitor._watchdog_timeout_s = 30.0
    monitor._watchdog_reconnect_cooldown_s = 120.0
    monitor._last_ws_activity_time = 80.0
    monitor._last_msg_time = 0.0
    monitor._last_watchdog_reconnect_time = 0.0

    with monitor._lock:
        first_idle = monitor._consume_watchdog_trigger_locked(120.0)
    assert first_idle == 40.0
    assert monitor._last_watchdog_reconnect_time == 120.0

    with monitor._lock:
        second_idle = monitor._consume_watchdog_trigger_locked(150.0)
    assert second_idle is None

    with monitor._lock:
        third_idle = monitor._consume_watchdog_trigger_locked(241.0)
    assert third_idle == 161.0
    assert monitor._last_watchdog_reconnect_time == 241.0


def test_watchdog_uses_last_pong_timestamp_as_activity():
    class _FakeWs:
        def __init__(self, last_pong_tm: float):
            self.last_pong_tm = last_pong_tm

    monitor = PositionMonitor()
    monitor._connected = True
    monitor._watchdog_timeout_s = 30.0
    monitor._watchdog_reconnect_cooldown_s = 120.0
    monitor._last_ws_activity_time = 0.0
    monitor._last_msg_time = 0.0
    monitor._last_watchdog_reconnect_time = 0.0

    ws = _FakeWs(last_pong_tm=200.0)
    with monitor._lock:
        idle = monitor._consume_watchdog_trigger_locked(220.0, ws=ws)
    assert idle is None

    with monitor._lock:
        idle = monitor._consume_watchdog_trigger_locked(241.0, ws=ws)
    assert idle == 41.0


def test_watchdog_respects_startup_grace_window():
    monitor = PositionMonitor()
    monitor._connected = True
    monitor._watchdog_timeout_s = 30.0
    monitor._watchdog_reconnect_cooldown_s = 120.0
    monitor._last_ws_activity_time = 100.0
    monitor._last_msg_time = 100.0
    monitor._last_watchdog_reconnect_time = 0.0
    monitor._watchdog_grace_until = 145.0

    with monitor._lock:
        idle = monitor._consume_watchdog_trigger_locked(140.0)
    assert idle is None

    with monitor._lock:
        idle = monitor._consume_watchdog_trigger_locked(146.0)
    assert idle == 46.0


def test_on_pong_updates_transport_activity(monkeypatch):
    monitor = PositionMonitor()
    monkeypatch.setattr("src.notifications.ws_position_monitor.time.time", lambda: 123.45)
    monitor._on_pong(None, b"")
    assert monitor._last_ws_activity_time == 123.45


def test_on_message_updates_transport_activity_on_non_json_frame(monkeypatch):
    monitor = PositionMonitor()
    monitor._last_msg_time = 100.0
    monitor._gap_warn_threshold_s = 9999.0
    monkeypatch.setattr("src.notifications.ws_position_monitor.time.time", lambda: 130.0)

    monitor._on_message(None, "not-json")

    assert monitor._last_msg_time == 130.0
    assert monitor._last_ws_activity_time == 130.0


def test_transient_ws_close_error_detection_recognizes_inactive():
    assert PositionMonitor._is_transient_ws_close_error(
        "fin=1 opcode=8 data=b'\\x03\\xe8Inactive' - goodbye"
    )


def test_transient_ws_close_error_detection_recognizes_expired():
    assert PositionMonitor._is_transient_ws_close_error(
        "fin=1 opcode=8 data=b'\\x03\\xe8Expired'"
    )


def test_transient_ws_close_error_detection_rejects_generic_error():
    assert not PositionMonitor._is_transient_ws_close_error("ssl cert verify failed")


def test_on_error_logs_info_for_transient_inactive_close(caplog):
    monitor = PositionMonitor()
    monitor._running = True
    monitor._inactive_rest_only_interval_s = 123.0

    with caplog.at_level(logging.INFO):
        monitor._on_error(None, "fin=1 opcode=8 data=b'\\x03\\xe8Inactive' - goodbye")

    wait, reason = monitor._consume_reconnect_wait()
    assert wait == 123.0
    assert reason == "inactive userEvents stream"
    assert "rest-only mode" in caplog.text.lower()


def test_on_error_logs_warning_for_non_transient_error(caplog):
    monitor = PositionMonitor()
    monitor._running = True

    with caplog.at_level(logging.WARNING):
        monitor._on_error(None, "ssl cert verify failed")

    assert "websocket error" in caplog.text.lower()


def test_on_error_expired_requests_fast_reconnect(caplog):
    monitor = PositionMonitor()
    monitor._running = True

    with caplog.at_level(logging.INFO):
        monitor._on_error(None, "fin=1 opcode=8 data=b'\\x03\\xe8Expired'")

    wait, reason = monitor._consume_reconnect_wait()
    assert wait == 1.0
    assert reason == "expired WebSocket session"
    assert "session expired" in caplog.text.lower()


def test_on_close_inactive_requests_idle_rest_only_wait(caplog):
    monitor = PositionMonitor()
    monitor._running = True
    monitor._inactive_rest_only_interval_s = 321.0
    monitor._subscribed_addresses = {"0x" + "1" * 40}

    with caplog.at_level(logging.INFO):
        monitor._on_close(None, None, "Inactive")

    wait, reason = monitor._consume_reconnect_wait()
    assert wait == 321.0
    assert reason == "inactive userEvents stream"
    assert monitor._subscribed_addresses == set()
    assert monitor._reconnect_wake_event.is_set() is False
    assert "rest-only mode" in caplog.text.lower()


def test_note_disconnect_preserves_backoff_for_short_lived_flaps(monkeypatch):
    monitor = PositionMonitor()
    monitor._connected = True
    monitor._connected_since = 100.0
    monitor._reconnect_count = 3
    monitor._stable_connection_reset_s = 180.0

    monkeypatch.setattr("src.notifications.ws_position_monitor.time.time", lambda: 150.0)
    monitor._note_disconnect(transient=True)

    assert monitor._connected is False
    assert monitor._reconnect_count == 3
    assert monitor._transient_disconnects == 1


def test_note_disconnect_resets_backoff_after_stable_connection(monkeypatch):
    monitor = PositionMonitor()
    monitor._connected = True
    monitor._connected_since = 100.0
    monitor._reconnect_count = 3
    monitor._stable_connection_reset_s = 120.0

    monkeypatch.setattr("src.notifications.ws_position_monitor.time.time", lambda: 260.0)
    monitor._note_disconnect(transient=True)

    assert monitor._reconnect_count == 0


def test_rest_reconcile_once_detects_changes_while_disconnected(monkeypatch):
    monitor = PositionMonitor()
    monitor._connected = False
    monitor._tracked_addresses = {"0xabc"}
    monitor._position_cache = {
        "0xabc": {
            "BTC": {
                "size": 1.0,
                "side": "long",
                "entry_price": 100.0,
                "leverage": 2.0,
                "unrealized_pnl": 0.0,
            }
        }
    }
    monitor._mids_cache = {"ETH": 200.0}

    monkeypatch.setattr(
        monitor,
        "_fetch_positions_snapshot",
        lambda address: {
            "BTC": {
                "size": 1.0,
                "side": "long",
                "entry_price": 100.0,
                "leverage": 2.0,
                "unrealized_pnl": 0.0,
            },
            "ETH": {
                "size": 2.0,
                "side": "long",
                "entry_price": 200.0,
                "leverage": 3.0,
                "unrealized_pnl": 0.0,
            },
        },
    )

    emitted = monitor._rest_reconcile_once()

    assert emitted == 1
    signal = monitor.drain_signals()[0]
    assert signal["type"] == "copy_open"
    assert signal["coin"] == "ETH"
    assert monitor.get_stats()["rest_fallback_cycles"] == 1


def test_websocket_library_filter_suppresses_transient_close_frame_errors():
    filt = _TransientWebSocketLibraryLogFilter(PositionMonitor._TRANSIENT_CLOSE_MARKERS)
    transient = logging.LogRecord(
        name="websocket",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="fin=1 opcode=8 data=b'\\x03\\xe8Inactive' - goodbye",
        args=(),
        exc_info=None,
    )
    non_transient = logging.LogRecord(
        name="websocket",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="ssl cert verify failed",
        args=(),
        exc_info=None,
    )

    assert filt.filter(transient) is False
    assert filt.filter(non_transient) is True


def test_on_open_without_active_positions_switches_to_rest_only_mode(monkeypatch):
    monitor = PositionMonitor()
    address = "0x" + "1" * 40
    monitor._tracked_addresses = {address}

    reasons = []
    monkeypatch.setattr(monitor, "_bootstrap_positions", lambda _address: {})
    monkeypatch.setattr(monitor, "_enter_idle_rest_only_mode", lambda reason: reasons.append(reason))

    monitor._on_open(object())

    assert reasons == ["no active tracked positions"]
    assert monitor._subscribed_addresses == set()


def test_rest_reconcile_subscribes_watch_only_trader_when_position_appears(monkeypatch):
    monitor = PositionMonitor()
    address = "0x" + "2" * 40
    monitor._connected = True
    monitor._tracked_addresses = {address}
    monitor._subscribed_addresses = set()
    monitor._mids_cache = {"BTC": 100.0}

    subscribed = []
    monkeypatch.setattr(
        monitor,
        "_fetch_positions_snapshot",
        lambda _address: {
            "BTC": {
                "size": 1.0,
                "side": "long",
                "entry_price": 100.0,
                "leverage": 2.0,
                "unrealized_pnl": 0.0,
            }
        },
    )
    monkeypatch.setattr(monitor, "_subscribe_to_address", lambda addr: subscribed.append(addr))

    emitted = monitor._rest_reconcile_once()

    assert emitted == 1
    assert subscribed == [address]


def test_process_user_events_empty_positions_marks_trader_flat_and_idles(monkeypatch):
    monitor = PositionMonitor()
    address = "0x" + "3" * 40
    monitor._connected = True
    monitor._tracked_addresses = {address}
    monitor._subscribed_addresses = {address}
    monitor._active_position_addresses = {address}
    monitor._position_cache = {
        address: {
            "BTC": {
                "size": 1.0,
                "side": "long",
                "entry_price": 100.0,
                "leverage": 2.0,
                "unrealized_pnl": 0.0,
            }
        }
    }

    reasons = []
    monkeypatch.setattr(monitor, "_enter_idle_rest_only_mode", lambda reason: reasons.append(reason))

    monitor._process_user_events({"user": address, "positions": []})

    assert address not in monitor._subscribed_addresses
    assert address not in monitor._active_position_addresses
    assert reasons == ["all tracked positions are flat"]
