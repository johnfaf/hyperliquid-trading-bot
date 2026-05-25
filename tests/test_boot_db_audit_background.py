"""Boot DB audit can run in background so a slow audit doesn't hang boot.

Background
----------
On a large ``/data/bot.db`` (1800+ strategies + 1300+ traders, with a
bloated freelist from prior deletes) the boot-time ``run_db_audit``
call can take 10+ minutes -- mostly PRAGMA integrity_check scanning
every page.  The audit is purely informational (logs findings, never
mutates trades) so blocking the boot on it makes no functional sense.

After this fix
--------------
With ``BOOT_DB_AUDIT_BACKGROUND=true`` (the new default), the audit
runs on a daemon thread.  Boot continues immediately into subsystem
initialisation and trading cycles; the audit logs its findings when
it finishes (after the bot is already trading).

Operator can flip ``BOOT_DB_AUDIT_BACKGROUND=false`` to restore the
legacy blocking behaviour, or ``BOOT_DB_AUDIT_SKIP=true`` to skip
the audit entirely.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import config


class _FakeMain:
    """Minimal stand-in for the main bot class with just the audit helpers.

    The real bot class has a lot of unrelated initialisation, but
    ``_launch_background_db_audit`` and ``_run_blocking_db_audit`` are
    self-contained so we can test them without bringing up the full
    bot machinery.
    """

    def __init__(self):
        import logging
        self.logger = logging.getLogger("test_main")
        self._boot_db_audit_thread = None

    # Bind the real methods onto the fake by import.  We reach into
    # ``main`` module-level only for these two helper methods.
    from main import (  # noqa: E402 - intentional late import
        HyperliquidResearchBot,
    )
    _run_blocking_db_audit = HyperliquidResearchBot._run_blocking_db_audit
    _launch_background_db_audit = (
        HyperliquidResearchBot._launch_background_db_audit
    )


# ── Background mode is the default ───────────────────────────


def test_background_audit_returns_immediately(monkeypatch):
    """The launcher must return without waiting for the audit to finish."""
    monkeypatch.setattr(config, "BOOT_DB_AUDIT_BACKGROUND", True, raising=False)
    monkeypatch.setattr(
        config, "BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE", False, raising=False,
    )

    # Stub run_db_audit to sleep for 2 seconds -- enough that a
    # blocking call would clearly stall the test.
    audit_started = threading.Event()
    audit_finished = threading.Event()
    fake_report = MagicMock()
    fake_report.findings = []
    fake_report.findings_at_or_above = MagicMock(return_value=False)

    def _slow_audit(**_kwargs):
        audit_started.set()
        time.sleep(2.0)
        audit_finished.set()
        return fake_report

    with patch("src.data.db_audit.run_db_audit", _slow_audit), \
         patch("src.data.db_audit.format_db_audit_report", lambda *a, **k: ""):
        main = _FakeMain()
        start = time.time()
        main._launch_background_db_audit()
        elapsed = time.time() - start

    # Launcher must have returned almost instantly (well under the
    # 2-second audit duration).
    assert elapsed < 1.0, (
        f"_launch_background_db_audit blocked for {elapsed:.2f}s; "
        f"should return immediately"
    )

    # The daemon thread should have started (or be about to).
    assert audit_started.wait(timeout=3), (
        "background audit thread did not start within 3s"
    )

    # Clean up by waiting for the thread to finish; daemon=True means
    # we don't NEED to wait, but doing so keeps test output clean.
    audit_finished.wait(timeout=5)


def test_background_audit_thread_is_daemon(monkeypatch):
    """The audit thread is daemonic so it doesn't keep the process alive."""
    monkeypatch.setattr(config, "BOOT_DB_AUDIT_BACKGROUND", True, raising=False)
    monkeypatch.setattr(
        config, "BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE", False, raising=False,
    )

    fake_report = MagicMock()
    fake_report.findings = []
    fake_report.findings_at_or_above = MagicMock(return_value=False)

    with patch("src.data.db_audit.run_db_audit", lambda **k: fake_report), \
         patch("src.data.db_audit.format_db_audit_report", lambda *a, **k: ""):
        main = _FakeMain()
        main._launch_background_db_audit()
        thread = main._boot_db_audit_thread

    assert thread is not None, "background thread reference was not stored"
    assert thread.daemon is True, "background audit thread must be a daemon"
    assert thread.name == "boot-db-audit", (
        f"unexpected thread name {thread.name!r}"
    )

    # Let the thread finish so it doesn't leak into other tests.
    thread.join(timeout=5)


def test_background_audit_logs_findings_when_complete(monkeypatch, caplog):
    """When the audit eventually finishes, its findings get logged."""
    import logging
    monkeypatch.setattr(config, "BOOT_DB_AUDIT_BACKGROUND", True, raising=False)
    monkeypatch.setattr(
        config, "BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE", False, raising=False,
    )

    fake_report = MagicMock()
    fake_report.findings = []
    fake_report.findings_at_or_above = MagicMock(return_value=False)

    with patch("src.data.db_audit.run_db_audit", lambda **k: fake_report), \
         patch("src.data.db_audit.format_db_audit_report", lambda *a, **k: ""):
        main = _FakeMain()
        with caplog.at_level(logging.INFO, logger="test_main"):
            main._launch_background_db_audit()
            # Wait for the thread to finish so its log lands
            main._boot_db_audit_thread.join(timeout=5)

    messages = " ".join(rec.message for rec in caplog.records)
    assert "Scheduling startup DB audit in background" in messages, (
        f"Expected scheduling log line; saw: {messages}"
    )
    assert "Background DB audit passed readiness threshold" in messages, (
        f"Expected post-audit success log; saw: {messages}"
    )


def test_background_audit_handles_audit_failure(monkeypatch, caplog):
    """If the audit itself raises, the thread logs a warning and exits cleanly."""
    import logging
    monkeypatch.setattr(config, "BOOT_DB_AUDIT_BACKGROUND", True, raising=False)
    monkeypatch.setattr(
        config, "BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE", False, raising=False,
    )

    def _boom(**_kwargs):
        raise RuntimeError("simulated DB audit failure")

    with patch("src.data.db_audit.run_db_audit", _boom):
        main = _FakeMain()
        with caplog.at_level(logging.WARNING, logger="test_main"):
            main._launch_background_db_audit()
            main._boot_db_audit_thread.join(timeout=5)

    messages = " ".join(rec.message for rec in caplog.records)
    assert "Background DB audit failed" in messages, (
        f"Expected post-audit failure log; saw: {messages}"
    )


# ── Blocking mode for opt-out ────────────────────────────────


def test_blocking_audit_runs_synchronously(monkeypatch):
    """_run_blocking_db_audit runs the audit in the caller's thread."""
    monkeypatch.setattr(
        config, "BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE", False, raising=False,
    )

    fake_report = MagicMock()
    fake_report.findings = []
    fake_report.findings_at_or_above = MagicMock(return_value=False)
    audit_ran = []

    def _audit(**_kwargs):
        audit_ran.append(True)
        return fake_report

    with patch("src.data.db_audit.run_db_audit", _audit), \
         patch("src.data.db_audit.format_db_audit_report", lambda *a, **k: ""):
        main = _FakeMain()
        main._run_blocking_db_audit()

    # The audit ran by the time the call returned (synchronous).
    assert audit_ran == [True]


def test_blocking_audit_swallows_errors(monkeypatch, caplog):
    """A failure in the blocking audit must not crash boot."""
    import logging
    monkeypatch.setattr(
        config, "BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE", False, raising=False,
    )

    def _boom(**_kwargs):
        raise RuntimeError("simulated DB audit failure")

    with patch("src.data.db_audit.run_db_audit", _boom):
        main = _FakeMain()
        with caplog.at_level(logging.WARNING, logger="test_main"):
            main._run_blocking_db_audit()

    messages = " ".join(rec.message for rec in caplog.records)
    assert "Database audit skipped during boot" in messages
