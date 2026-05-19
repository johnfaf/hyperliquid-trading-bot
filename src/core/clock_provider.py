"""Process-wide swappable time source.

Production code that needs to ask "what time is it" should import from here
rather than calling `datetime.now(timezone.utc)` or `time.time()` directly.
The replay harness installs a fake backend before booting subsystems so all
time-dependent reads slide back to the replay window.

In production the default backend is `LiveClock`, so behavior is identical
to calling the stdlib directly. The harness calls `install(clock)` to swap
in a `ReplayClock`, and `restore()` on teardown.

Why module-level globals instead of a DI container field:
- Time is needed in places that don't take a container reference (helper
  functions, dataclass __post_init__, scoring loops).
- The "ambient" pattern matches how `datetime.now()` is already used today,
  so converting a call site is a one-line change.
- A single install/restore on the harness's boot/teardown is harder to get
  wrong than threading the clock through five constructor signatures.

Thread-safety: install/restore are not thread-safe. Only the harness
orchestrator should call them, before threads are started.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.backtest.replay.clock import Clock, LiveClock


_backend: Clock = LiveClock()


def utc_now() -> datetime:
    """tz-aware UTC datetime. Replaces `datetime.now(timezone.utc)`."""
    return _backend.now()


def unix_now() -> float:
    """Epoch seconds float. Replaces `time.time()`."""
    return _backend.now_unix()


def unix_ms() -> int:
    """Epoch milliseconds int. Replaces `int(time.time() * 1000)`."""
    return _backend.now_ms()


def iso_now() -> str:
    """ISO-8601 UTC string with 'Z' suffix."""
    return _backend.now_iso()


def install(clock: Clock) -> Clock:
    """Swap the backend. Returns the previous backend so callers can restore."""
    global _backend
    prev = _backend
    _backend = clock
    return prev


def restore(previous: Optional[Clock] = None) -> None:
    """Restore the previous backend. Pass the value returned by install()."""
    global _backend
    _backend = previous if previous is not None else LiveClock()


def current() -> Clock:
    """Return the active backend. For introspection / tests."""
    return _backend
