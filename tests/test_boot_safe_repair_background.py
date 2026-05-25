"""Boot safe-repair can run in background so a slow repair doesn't hang boot.

Background
----------
On a large ``/data/bot.db`` the boot-time ``run_startup_safe_repair``
call walks several large tables and can take 10+ minutes.  PR #27
moved the boot DB audit to a daemon thread; this PR applies the same
pattern to safe-repair.

After this fix
--------------
With ``BOOT_SAFE_REPAIR_BACKGROUND=true`` (the new default), the
safe-repair runs on a daemon thread.  Boot continues immediately
into subsystem initialisation; the repair completes whenever the DB
is available, by which point the bot is already trading.

Operator can flip ``BOOT_SAFE_REPAIR_BACKGROUND=false`` to restore
the legacy blocking behaviour, or
``DB_SAFE_AUTO_REPAIR_ON_BOOT=false`` to skip safe-repair entirely.
"""
from __future__ import annotations

import logging
import threading
import time
from unittest.mock import patch

import config

from src.core.boot import _run_safe_repair_inline, init_database


# ── Background mode is the default ──────────────────────────


def test_background_mode_returns_immediately(monkeypatch, caplog):
    """init_database returns without waiting for safe-repair to finish."""
    monkeypatch.setattr(
        config, "DB_SAFE_AUTO_REPAIR_ON_BOOT", True, raising=False,
    )
    monkeypatch.setattr(
        config, "BOOT_SAFE_REPAIR_BACKGROUND", True, raising=False,
    )

    # Stub init_db + restore_from_json so we don't touch real DB state.
    monkeypatch.setattr("src.data.database.init_db", lambda: None)
    monkeypatch.setattr("src.data.database.restore_from_json", lambda: False)

    # Stub run_startup_safe_repair to sleep 2s -- enough to clearly
    # blow past any normal boot timing.
    repair_started = threading.Event()
    repair_finished = threading.Event()

    def _slow_repair():
        repair_started.set()
        time.sleep(2.0)
        repair_finished.set()
        return []  # no actions applied

    with patch("src.data.db_audit.run_startup_safe_repair", _slow_repair):
        logger = logging.getLogger("test_boot_safe_repair_bg")
        with caplog.at_level(logging.INFO, logger="test_boot_safe_repair_bg"):
            start = time.time()
            init_database(logger)
            elapsed = time.time() - start

    # init_database must return almost instantly (well under 2s).
    assert elapsed < 1.0, (
        f"init_database blocked for {elapsed:.2f}s; "
        f"background mode should return immediately"
    )
    # The daemon thread should have started.
    assert repair_started.wait(timeout=3), (
        "background safe-repair did not start within 3s"
    )

    messages = " ".join(rec.message for rec in caplog.records)
    assert "Scheduling startup DB safe-repair in background" in messages, (
        f"expected scheduling log line; saw: {messages}"
    )

    # Let it finish so it doesn't leak into later tests.
    repair_finished.wait(timeout=5)


def test_background_thread_is_daemon(monkeypatch):
    """The repair thread is daemonic so it doesn't keep the process alive."""
    monkeypatch.setattr(
        config, "DB_SAFE_AUTO_REPAIR_ON_BOOT", True, raising=False,
    )
    monkeypatch.setattr(
        config, "BOOT_SAFE_REPAIR_BACKGROUND", True, raising=False,
    )
    monkeypatch.setattr("src.data.database.init_db", lambda: None)
    monkeypatch.setattr("src.data.database.restore_from_json", lambda: False)

    # Block the repair thread for long enough that we can capture it
    # from the enumeration of live threads before it exits.
    repair_blocker = threading.Event()
    repair_running = threading.Event()

    def _slow_repair():
        repair_running.set()
        repair_blocker.wait(timeout=5)
        return []

    with patch("src.data.db_audit.run_startup_safe_repair", _slow_repair):
        logger = logging.getLogger("test_boot_safe_repair_daemon")
        init_database(logger)
        assert repair_running.wait(timeout=3), (
            "background repair did not start"
        )
        try:
            # Find the thread by name from the live thread list.
            target_thread = next(
                (t for t in threading.enumerate() if t.name == "boot-safe-repair"),
                None,
            )
            assert target_thread is not None, (
                f"expected thread named 'boot-safe-repair'; "
                f"got {[t.name for t in threading.enumerate()]}"
            )
            assert target_thread.daemon is True, "must be daemon=True"
        finally:
            repair_blocker.set()


def test_background_safe_repair_logs_applied_actions(monkeypatch, caplog):
    """When repairs are applied, the post-thread log line names them."""
    from dataclasses import dataclass

    @dataclass
    class _FakeAction:
        action: str
        status: str
        details: dict

    monkeypatch.setattr(
        config, "DB_SAFE_AUTO_REPAIR_ON_BOOT", True, raising=False,
    )
    monkeypatch.setattr(
        config, "BOOT_SAFE_REPAIR_BACKGROUND", True, raising=False,
    )
    monkeypatch.setattr("src.data.database.init_db", lambda: None)
    monkeypatch.setattr("src.data.database.restore_from_json", lambda: False)

    fake_actions = [
        _FakeAction("paper_account", "applied", {}),
        _FakeAction("stale_pending_decisions", "applied", {}),
    ]

    with patch("src.data.db_audit.run_startup_safe_repair",
               return_value=fake_actions):
        logger = logging.getLogger("test_safe_repair_applied")
        with caplog.at_level(logging.INFO, logger="test_safe_repair_applied"):
            init_database(logger)
            # Wait briefly for the daemon thread to log.
            time.sleep(0.5)

    messages = " ".join(rec.message for rec in caplog.records)
    assert "Startup DB safe-repair applied" in messages
    assert "paper_account" in messages or "stale_pending_decisions" in messages


# ── Legacy blocking mode for opt-out ────────────────────────


def test_legacy_blocking_mode(monkeypatch, caplog):
    """BOOT_SAFE_REPAIR_BACKGROUND=false runs synchronously on boot thread."""
    monkeypatch.setattr(
        config, "DB_SAFE_AUTO_REPAIR_ON_BOOT", True, raising=False,
    )
    monkeypatch.setattr(
        config, "BOOT_SAFE_REPAIR_BACKGROUND", False, raising=False,
    )
    monkeypatch.setattr("src.data.database.init_db", lambda: None)
    monkeypatch.setattr("src.data.database.restore_from_json", lambda: False)

    repair_ran = []

    def _fast_repair():
        repair_ran.append(True)
        return []

    with patch("src.data.db_audit.run_startup_safe_repair", _fast_repair):
        logger = logging.getLogger("test_safe_repair_blocking")
        with caplog.at_level(logging.INFO, logger="test_safe_repair_blocking"):
            init_database(logger)

    # In blocking mode the repair ran by the time init_database returned.
    assert repair_ran == [True]
    messages = " ".join(rec.message for rec in caplog.records)
    assert "Running startup DB safe-repair" in messages, (
        f"expected legacy 'Running startup DB safe-repair' log; "
        f"saw: {messages}"
    )


def test_skip_when_safe_auto_repair_disabled(monkeypatch, caplog):
    """DB_SAFE_AUTO_REPAIR_ON_BOOT=false short-circuits the entire path."""
    monkeypatch.setattr(
        config, "DB_SAFE_AUTO_REPAIR_ON_BOOT", False, raising=False,
    )
    monkeypatch.setattr("src.data.database.init_db", lambda: None)
    monkeypatch.setattr("src.data.database.restore_from_json", lambda: False)

    called = []

    def _should_not_run():
        called.append(True)
        return []

    with patch("src.data.db_audit.run_startup_safe_repair", _should_not_run):
        logger = logging.getLogger("test_safe_repair_skipped")
        init_database(logger)

    # Neither blocking nor background path ran -- safe-repair is fully
    # disabled.
    assert called == [], (
        "safe-repair should not run when DB_SAFE_AUTO_REPAIR_ON_BOOT=false"
    )


# ── Inline helper handles errors ────────────────────────────


def test_inline_helper_swallows_repair_errors(caplog):
    """An exception in run_startup_safe_repair must not crash boot."""
    def _boom():
        raise RuntimeError("simulated repair failure")

    with patch("src.data.db_audit.run_startup_safe_repair", _boom):
        logger = logging.getLogger("test_safe_repair_err")
        with caplog.at_level(logging.WARNING, logger="test_safe_repair_err"):
            _run_safe_repair_inline(logger)

    messages = " ".join(rec.message for rec in caplog.records)
    assert "Startup DB safe-repair failed" in messages
