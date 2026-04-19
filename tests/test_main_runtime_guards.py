import threading
import time
from types import SimpleNamespace

import main


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.criticals = []
        self.errors = []

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def critical(self, msg, *args):
        self.criticals.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else msg)


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


def test_trading_cycle_checks_kill_switch_before_work(monkeypatch):
    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    bot.logger = _FakeLogger()
    bot.container = object()
    bot.runtime_config = SimpleNamespace(poll=lambda _container: None)
    bot.running = True
    bot._cycle_count = 0

    monkeypatch.setattr(main, "check_file_kill_switch", lambda _container: True)
    monkeypatch.setattr(
        main,
        "run_feature_cycle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    bot._run_trading_cycle()

    assert bot.running is False
    assert bot._cycle_count == 0
    assert any("KILL_SWITCH triggered before trading cycle" in msg for msg in bot.logger.criticals)


def test_run_with_timeout_continues_after_hung_shutdown_work():
    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    bot.logger = _FakeLogger()
    release = threading.Event()

    ok = bot._run_with_timeout("hung_step", lambda: release.wait(1.0), timeout_s=0.01)

    release.set()
    assert ok is False
    assert any("hung_step timed out" in msg for msg in bot.logger.errors)


def test_start_discovery_async_runs_in_background(monkeypatch):
    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    bot.logger = _FakeLogger()
    bot.container = object()
    bot.runtime_monitor = SimpleNamespace(evaluate_and_alert=lambda **kwargs: None)
    bot._discovery_state_lock = threading.Lock()
    bot._discovery_thread = None
    bot._discovery_retry_after_ts = 0.0

    started = threading.Event()
    release = threading.Event()

    def _run_discovery():
        started.set()
        release.wait(1.0)

    bot._run_discovery = _run_discovery

    assert bot._start_discovery_async("unit_test") is True
    assert started.wait(1.0) is True
    assert bot._discovery_running() is True

    release.set()
    deadline = time.time() + 1.0
    while bot._discovery_running() and time.time() < deadline:
        time.sleep(0.01)

    assert bot._discovery_running() is False
    assert any("Starting background discovery" in msg for msg in bot.logger.infos)


def test_run_loop_schedules_initial_discovery_without_blocking_trading(monkeypatch):
    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    bot.logger = _FakeLogger()
    bot.container = SimpleNamespace(reporter=None, position_monitor=None, dashboard=None)
    bot.runtime_config = SimpleNamespace(poll=lambda _container: None)
    bot.runtime_monitor = SimpleNamespace(evaluate_and_alert=lambda **kwargs: None)
    bot.task_runner = SimpleNamespace(start_all=lambda: None, stop_all=lambda timeout=10: None)
    bot._last_research = 0.0
    bot._last_discovery = 0.0
    bot._last_report = 0.0
    bot._cycle_count = 0
    bot._fast_cycle_count = 0
    bot._shutdown_orders_cancelled = False
    bot._discovery_state_lock = threading.Lock()
    bot._discovery_thread = None
    bot._discovery_retry_after_ts = 0.0
    bot._cancel_live_orders_for_shutdown = lambda reason: None

    scheduled = []
    monkeypatch.setattr(main.db, "get_active_traders", lambda: [])
    def _start_discovery_async(reason):
        scheduled.append(reason)
        bot._discovery_thread = SimpleNamespace(is_alive=lambda: True)
        return True

    monkeypatch.setattr(bot, "_start_discovery_async", _start_discovery_async)
    monkeypatch.setattr(bot, "_fast_cycle", lambda: None)
    monkeypatch.setattr(bot, "_sleep_with_kill_switch_checks", lambda interval_s: None)
    monkeypatch.setattr(main.signal, "signal", lambda *args, **kwargs: None)

    def _run_trading_cycle():
        bot.running = False

    bot._run_trading_cycle = _run_trading_cycle

    bot.run_loop()

    assert scheduled == ["startup_empty_trader_pool"]
    assert any("scheduling initial discovery in background" in msg for msg in bot.logger.infos)


def test_register_background_tasks_includes_heartbeat_supervisor(monkeypatch):
    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    registered = []
    bot.task_runner = SimpleNamespace(
        register=lambda name, target, interval_seconds, max_retries=5: registered.append(
            (name, interval_seconds, max_retries, target)
        )
    )
    bot.container = SimpleNamespace(polymarket=None, options_scanner=None)

    calls = []
    monkeypatch.setattr(main.health_registry, "register", lambda name, affects_trading=False: calls.append((name, affects_trading)))

    bot._register_background_tasks()

    assert registered[0][0] == "bg-heartbeat"
    assert registered[0][1] >= 15.0
    assert calls[0] == ("bg-heartbeat", False)
