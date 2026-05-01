from src.core.task_runner import SupervisedTask, SupervisedTaskRunner
import threading


class _CountingLock:
    def __init__(self):
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_loop_updates_task_state_under_runner_lock():
    runner = SupervisedTaskRunner()
    runner._lock = _CountingLock()

    calls = {"count": 0}

    def _target():
        calls["count"] += 1
        task.stop_event.set()

    task = SupervisedTask(name="demo", target=_target, interval_seconds=0.01)

    runner._run_loop(task)

    assert calls["count"] == 1
    assert runner._lock.enter_count >= 1
    assert task.last_success_ts > 0


def test_run_loop_honors_initial_delay_before_first_run():
    runner = SupervisedTaskRunner()
    called = threading.Event()

    def _target():
        called.set()

    task = SupervisedTask(
        name="delayed",
        target=_target,
        interval_seconds=1.0,
        initial_delay_seconds=0.2,
    )
    thread = threading.Thread(target=runner._run_loop, args=(task,), daemon=True)
    thread.start()

    assert called.wait(0.05) is False
    assert called.wait(0.5) is True

    task.stop_event.set()
    thread.join(timeout=1.0)
