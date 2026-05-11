"""Controllable wall clock for historical replay.

Production code reaches `datetime.now(timezone.utc)` and `time.time()` directly
in dozens of places. For replay we route those reads through `Clock` instances
on the SubsystemContainer so the harness can advance time deterministically.

Two concrete implementations:
  - LiveClock: thin wrapper that delegates to the system clock. The default in
    production -- existing call sites become `container.clock.now()` instead of
    `datetime.now(timezone.utc)`.
  - ReplayClock: holds a fixed t and only changes when explicitly advanced.
    Raises if read before being set, so we catch any code path that boots
    without a clock instead of silently using whatever t the harness was
    initialised with.

Thread-safety: ReplayClock reads are atomic (a single int load); writes happen
only from the harness orchestrator on the main thread. If we later let cycles
run concurrently we'll need a lock, but the production loop is single-threaded
per cycle today.
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Anything that can answer "what time is it" in the units the bot uses."""

    def now(self) -> datetime: ...        # tz-aware UTC datetime
    def now_ms(self) -> int: ...          # epoch milliseconds
    def now_unix(self) -> float: ...      # epoch seconds (float, like time.time)
    def now_iso(self) -> str: ...         # ISO-8601 string with 'Z'


class LiveClock:
    """Production clock. Delegates to the system."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def now_ms(self) -> int:
        return int(_time.time() * 1000)

    def now_unix(self) -> float:
        return _time.time()

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ReplayClock:
    """Replay clock. Time advances only when `set` or `advance` is called.

    Reads before `set` raise `RuntimeError` -- silent fallback to a default
    timestamp would mask bugs in the harness wiring.
    """

    __slots__ = ("_t_ms", "_set", "_label")

    def __init__(self, start_ts_ms: int | None = None, *, label: str = "replay"):
        self._set = False
        self._t_ms = 0
        self._label = label
        if start_ts_ms is not None:
            self.set(start_ts_ms)

    def set(self, ts_ms: int) -> None:
        if not isinstance(ts_ms, int):
            raise TypeError(f"ReplayClock.set: expected int ms, got {type(ts_ms).__name__}")
        if ts_ms < 0:
            raise ValueError(f"ReplayClock.set: timestamp must be non-negative, got {ts_ms}")
        self._t_ms = ts_ms
        self._set = True

    def advance(self, delta_ms: int) -> None:
        if delta_ms < 0:
            raise ValueError(f"ReplayClock.advance: delta must be non-negative, got {delta_ms}")
        if not self._set:
            raise RuntimeError("ReplayClock.advance called before set()")
        self._t_ms += delta_ms

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._read_ms() / 1000.0, tz=timezone.utc)

    def now_ms(self) -> int:
        return self._read_ms()

    def now_unix(self) -> float:
        return self._read_ms() / 1000.0

    def now_iso(self) -> str:
        return self.now().isoformat().replace("+00:00", "Z")

    def is_set(self) -> bool:
        return self._set

    def _read_ms(self) -> int:
        if not self._set:
            raise RuntimeError(
                f"{self._label} ReplayClock read before set() -- "
                "harness wiring bug; would have silently fabricated a timestamp"
            )
        return self._t_ms

    def __repr__(self) -> str:
        if not self._set:
            return f"<ReplayClock {self._label} unset>"
        return f"<ReplayClock {self._label} @ {self.now_iso()}>"
