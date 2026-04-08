from src.notifications.ws_position_monitor import PositionMonitor


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
