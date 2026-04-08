"""
Supervised Task Runner
======================
Manages background daemon threads (scanners, refreshers) with proper lifecycle
control: stop events, exponential-backoff retries, max-retry ceilings, and
health-registry integration.

Every "run forever unless the universe ends" loop in the codebase should be
converted to a SupervisedTask managed by this runner.

Usage::

    from src.core.task_runner import runner

    def polymarket_scan():
        scanner.scan_markets()
        scanner.get_market_sentiment()

    runner.register("bg-polymarket", polymarket_scan, interval_seconds=180, max_retries=10)
    runner.start_all()
    # ...
    runner.stop_all()   # graceful shutdown with thread join
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SupervisedTask:
    name: str
    target: Callable
    interval_seconds: float
    max_retries: int = 5
    auto_recover_cooldown_s: float = 300.0
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    retry_count: int = 0
    consecutive_failures: int = 0
    last_success_ts: float = 0.0
    last_error: str = ""
    state: str = "stopped"          # stopped | running | failed | retrying


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SupervisedTaskRunner:
    """
    Registry + supervisor for background daemon threads.

    Integrates with ``SubsystemHealthRegistry`` when one is provided: each
    task heartbeats on success and reports failures.
    """

    def __init__(self, health_registry=None):
        self._lock = threading.Lock()
        self._tasks: Dict[str, SupervisedTask] = {}
        self._health = health_registry  # optional SubsystemHealthRegistry

    # ── Registration ──────────────────────────────────────────────

    def register(self, name: str, target: Callable,
                 interval_seconds: float, max_retries: int = 5,
                 auto_recover_cooldown_s: float = 300.0) -> None:
        """Register a background task.  Does NOT start it yet."""
        with self._lock:
            if name in self._tasks:
                logger.warning("Task '%s' already registered — overwriting", name)
            self._tasks[name] = SupervisedTask(
                name=name,
                target=target,
                interval_seconds=interval_seconds,
                max_retries=max_retries,
                auto_recover_cooldown_s=max(0.0, float(auto_recover_cooldown_s)),
            )
        logger.debug(
            "Registered supervised task: %s (interval=%ss, max_retries=%d, auto_recover=%ss)",
            name,
            interval_seconds,
            max_retries,
            auto_recover_cooldown_s,
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self, name: str) -> None:
        """Start a single registered task in its own daemon thread."""
        with self._lock:
            task = self._tasks.get(name)
            if task is None:
                raise KeyError(f"Task '{name}' is not registered")
            if task.state == "running":
                logger.info("Task '%s' already running — skipping start", name)
                return
            task.stop_event.clear()
            task.retry_count = 0
            task.consecutive_failures = 0
            task.state = "running"
            t = threading.Thread(
                target=self._run_loop,
                args=(task,),
                daemon=True,
                name=f"supervised-{name}",
            )
            task.thread = t
            t.start()
        logger.info("Started supervised task '%s'", name)

    def start_all(self) -> None:
        """Start every registered task."""
        with self._lock:
            names = list(self._tasks.keys())
        for n in names:
            self.start(n)

    def stop(self, name: str, timeout: float = 10) -> None:
        """
        Signal a task to stop and wait up to *timeout* seconds for its
        thread to exit.
        """
        with self._lock:
            task = self._tasks.get(name)
            if task is None:
                return
            task.stop_event.set()
            thread = task.thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Task '%s' did not stop within %.1fs timeout", name, timeout)

        with self._lock:
            task = self._tasks.get(name)
            if task:
                task.state = "stopped"
        logger.info("Stopped supervised task '%s'", name)

    def stop_all(self, timeout: float = 10) -> None:
        """Signal all tasks to stop, then join them."""
        with self._lock:
            names = list(self._tasks.keys())
            for task in self._tasks.values():
                task.stop_event.set()

        for n in names:
            with self._lock:
                thread = self._tasks[n].thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)

        with self._lock:
            for task in self._tasks.values():
                task.state = "stopped"
        logger.info("All supervised tasks stopped")

    def restart(self, name: str) -> None:
        """Stop then start a task."""
        self.stop(name)
        time.sleep(0.1)
        self.start(name)

    # ── Query ─────────────────────────────────────────────────────

    def is_running(self, name: str) -> bool:
        with self._lock:
            task = self._tasks.get(name)
            return task is not None and task.state == "running"

    def get_status(self, name: str) -> Optional[dict]:
        with self._lock:
            task = self._tasks.get(name)
            if task is None:
                return None
            return self._task_to_dict(task)

    def get_all_status(self) -> Dict[str, dict]:
        with self._lock:
            return {
                name: self._task_to_dict(task)
                for name, task in self._tasks.items()
            }

    @staticmethod
    def _task_to_dict(task: SupervisedTask) -> dict:
        return {
            "name": task.name,
            "state": task.state,
            "interval_seconds": task.interval_seconds,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "auto_recover_cooldown_s": task.auto_recover_cooldown_s,
            "consecutive_failures": task.consecutive_failures,
            "last_success_ts": task.last_success_ts,
            "last_error": task.last_error,
        }

    # ── Internal loop ─────────────────────────────────────────────

    def _run_loop(self, task: SupervisedTask) -> None:
        """
        Core supervised loop.

        * Uses ``stop_event.wait(interval)`` instead of ``time.sleep()``
          so stop requests are honoured within one interval.
        * On failure: exponential back-off up to 60 s, capped at
          ``max_retries`` total consecutive failures before giving up.
        * On success: resets failure counters, heartbeats the health
          registry.
        """
        logger.info("Supervised loop started: %s (interval=%ss)",
                     task.name, task.interval_seconds)
        while not task.stop_event.is_set():
            try:
                task.target()

                # ── success ──
                task.last_success_ts = time.time()
                task.consecutive_failures = 0
                task.retry_count = 0
                task.state = "running"

                if self._health:
                    try:
                        self._health.heartbeat(task.name)
                    except Exception:
                        pass  # registry may not have this subsystem registered

            except Exception as exc:
                task.consecutive_failures += 1
                task.retry_count += 1
                task.last_error = str(exc)[:200]
                logger.warning(
                    "Task '%s' failed (attempt %d/%d): %s",
                    task.name, task.retry_count, task.max_retries, exc,
                )

                if task.retry_count >= task.max_retries:
                    task.state = "failed"
                    logger.error(
                        "Task '%s' reached max retries (%d).",
                        task.name, task.max_retries,
                    )
                    if self._health:
                        try:
                            from src.core.health_registry import SubsystemState
                            self._health.set_status(
                                task.name, SubsystemState.FAILED,
                                reason=f"max retries exceeded: {task.last_error}",
                            )
                        except Exception:
                            pass
                    cooldown = max(0.0, float(task.auto_recover_cooldown_s))
                    if cooldown <= 0:
                        logger.error(
                            "Task '%s' auto-recovery disabled; stopping supervised loop.",
                            task.name,
                        )
                        break  # explicit permanent stop
                    logger.error(
                        "Task '%s' entering recovery cooldown for %.0fs before restart.",
                        task.name,
                        cooldown,
                    )
                    if task.stop_event.wait(cooldown):
                        break
                    task.retry_count = 0
                    task.consecutive_failures = 0
                    task.state = "running"
                    if self._health:
                        try:
                            from src.core.health_registry import SubsystemState
                            self._health.set_status(
                                task.name,
                                SubsystemState.DEGRADED,
                                reason="auto-recovered after max retries",
                                dependency_ready=True,
                            )
                        except Exception:
                            pass
                    continue

                # Exponential back-off: 2, 4, 8, 16, 32, 60, 60, …
                task.state = "retrying"
                backoff = min(2 ** task.consecutive_failures, 60)
                logger.info("Task '%s' retrying in %ds", task.name, backoff)
                if task.stop_event.wait(backoff):
                    break  # stop was requested during backoff
                continue  # skip the normal interval wait

            # ── normal interval sleep (interruptible) ──
            if task.stop_event.wait(task.interval_seconds):
                break

        logger.info("Supervised loop exited: %s (state=%s)", task.name, task.state)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
runner = SupervisedTaskRunner()
