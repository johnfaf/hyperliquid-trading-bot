from types import SimpleNamespace

import main


class _FakeLogger:
    def __init__(self):
        self.warnings = []
        self.criticals = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def critical(self, msg, *args):
        self.criticals.append(msg % args if args else msg)


def test_restore_last_discovery_time_logs_and_returns_zero_on_failure(monkeypatch):
    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    bot.logger = _FakeLogger()

    class _BrokenCtx:
        def __enter__(self):
            raise RuntimeError("db locked")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main.db, "get_connection", lambda: _BrokenCtx())

    assert bot._restore_last_discovery_time() == 0.0
    assert any("Could not restore last discovery timestamp" in msg for msg in bot.logger.warnings)


def test_sleep_with_kill_switch_checks_survives_poll_and_check_errors(monkeypatch):
    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    bot.logger = _FakeLogger()
    bot.container = object()
    bot.running = True

    clock = {"t": 0.0}

    def _sleep(seconds):
        clock["t"] += seconds

    poll_calls = {"count": 0}
    check_calls = {"count": 0}

    def _poll(_container):
        poll_calls["count"] += 1
        raise RuntimeError("poll failed")

    def _check(_container):
        check_calls["count"] += 1
        raise RuntimeError("stat failed")

    bot.runtime_config = SimpleNamespace(poll=_poll)
    monkeypatch.setattr(main.time, "time", lambda: clock["t"])
    monkeypatch.setattr(main.time, "sleep", _sleep)
    monkeypatch.setattr(main, "check_file_kill_switch", _check)

    bot._sleep_with_kill_switch_checks(1.0)

    assert bot.running is True
    assert poll_calls["count"] >= 1
    assert check_calls["count"] >= 1
    assert any("Runtime config poll failed during sleep window" in msg for msg in bot.logger.warnings)
    assert any("Kill-switch check failed during sleep window" in msg for msg in bot.logger.warnings)
