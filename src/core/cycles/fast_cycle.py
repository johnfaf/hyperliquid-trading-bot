"""
Fast Cycle (Tier 1)
===================
Quick position check + copy-trade scan + whale detection.
Runs every 60s between trading cycles.

Extracted from ``HyperliquidResearchBot._fast_cycle``.
"""
import logging
import os
import threading
import time

from src.core.live_execution import (
    is_live_trading_active,
    mirror_executed_trades_to_live,
    sync_shadow_book_to_live,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_KILL_SWITCH_FILE = "/data/KILL_SWITCH"


def _live_order_hygiene_audit_interval_cycles() -> int:
    try:
        return max(1, int(float(os.environ.get("LIVE_ORDER_HYGIENE_AUDIT_INTERVAL_CYCLES", "5"))))
    except (TypeError, ValueError):
        return 5


# CRIT-FIX C5: module-level lock + flag used by BOTH the file-kill-switch
# path (fast cycle thread) and the shutdown signal handler (main thread) so
# that live-order cancellation is performed at most once, regardless of which
# trigger fires first or whether they race.  Without this, SIGINT arriving at
# the same moment as a KILL_SWITCH file appearing could schedule two parallel
# `cancel_all_orders()` calls against the exchange.
_order_cancel_lock = threading.Lock()


# P0-1 (audit): independent watchdog thread.  The legacy design only checked
# the kill-switch file at fast-cycle boundaries and inside
# ``_sleep_with_kill_switch_checks`` — which meant a long trading cycle
# (scoring + options + arena + API calls) could run for minutes after the
# operator dropped the file, with resting orders still live.  The watchdog
# runs at its own 1s cadence regardless of what the main loop is doing.
_watchdog_thread: threading.Thread | None = None
_watchdog_stop_event = threading.Event()


def _kill_switch_watchdog_loop(container, interval_s: float) -> None:
    """Poll the kill-switch file on an independent cadence.

    Uses ``check_file_kill_switch`` directly — that already serialises with
    the SIGINT shutdown handler via ``_order_cancel_lock``, and the
    P0-2 structured-cancel semantics ensure a fetch failure doesn't
    mark cancellation complete.
    """
    while not _watchdog_stop_event.is_set():
        try:
            check_file_kill_switch(container)
        except Exception as exc:
            # The watchdog MUST keep polling even after exceptions — the
            # whole point is a runaway trading cycle can't suppress it.
            logger.error("[kill-switch watchdog] poll failed: %s", exc)
        # Use Event.wait so we exit promptly when stop_kill_switch_watchdog()
        # is called during shutdown (no 1s tail-latency).
        if _watchdog_stop_event.wait(timeout=max(0.1, interval_s)):
            break
    logger.info("[kill-switch watchdog] stopped")


def start_kill_switch_watchdog(container, interval_s: float = 1.0) -> bool:
    """Start the independent kill-switch watchdog thread.

    Idempotent: repeated calls are no-ops.  Returns True if this call
    started the thread, False if one was already running.
    """
    global _watchdog_thread
    if _watchdog_thread is not None and _watchdog_thread.is_alive():
        return False
    _watchdog_stop_event.clear()
    _watchdog_thread = threading.Thread(
        target=_kill_switch_watchdog_loop,
        args=(container, interval_s),
        name="kill-switch-watchdog",
        daemon=True,
    )
    _watchdog_thread.start()
    logger.info(
        "[kill-switch watchdog] started (interval=%.1fs)", interval_s,
    )
    return True


def stop_kill_switch_watchdog(timeout: float = 2.0) -> None:
    """Signal the watchdog to stop and join briefly.

    Called from the shutdown path.  Never blocks the shutdown flow for
    more than ``timeout`` seconds — the thread is daemon so the process
    can exit even if the join times out.
    """
    global _watchdog_thread
    _watchdog_stop_event.set()
    thread = _watchdog_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    _watchdog_thread = None


def _snapshot_state_lock(container) -> threading.Lock:
    lock = getattr(container, "_snapshot_balance_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(container, "_snapshot_balance_lock", lock)
    return lock


def _get_snapshot_backoff_state(container) -> tuple[float, int]:
    with _snapshot_state_lock(container):
        next_try = float(getattr(container, "_snapshot_balance_next_try_ts", 0.0) or 0.0)
        failures = int(getattr(container, "_snapshot_balance_failures", 0) or 0)
        return next_try, failures


def _set_snapshot_backoff_state(container, *, next_try: float, failures: int) -> None:
    with _snapshot_state_lock(container):
        container._snapshot_balance_next_try_ts = float(next_try)
        container._snapshot_balance_failures = int(failures)


def _mark_kill_switch_state(container, reason: str) -> bool:
    """Atomically set the sticky kill-switch flag on the container.

    Returns True the first time this is called for a given container (the
    caller is then responsible for performing the cancellation work), and
    False on every subsequent call.
    """
    with _order_cancel_lock:
        if getattr(container, "_file_kill_switch_triggered", False):
            return False
        setattr(container, "_file_kill_switch_triggered", True)
        setattr(container, "_stop_requested", True)
        setattr(container, "_kill_switch_reason", reason)
        return True


def cancel_live_orders_once(container, reason: str, *, force: bool = False) -> bool:
    """Cancel all live orders exactly once, serialised across threads.

    Used both by the fast-cycle file-kill-switch path and by the main-thread
    shutdown signal handler.  After the first call the container's
    ``_shutdown_orders_cancelled`` flag is set and subsequent calls are
    no-ops.

    Parameters
    ----------
    force : bool
        When True the dry-run guard is bypassed — used by the KILL_SWITCH
        emergency path where we want to cancel regardless of the live_trader's
        configured dry-run state.

    Returns True if this call performed the cancellation, False if it was
    skipped because another thread had already done it or the trader is
    dry-run and force is False.
    """
    with _order_cancel_lock:
        if getattr(container, "_shutdown_orders_cancelled", False):
            return False
        live_trader = getattr(container, "live_trader", None)
        if not live_trader:
            setattr(container, "_shutdown_orders_cancelled", True)
            return False
        if not force and getattr(live_trader, "dry_run", True):
            setattr(container, "_shutdown_orders_cancelled", True)
            return False
        # P0-2 (audit): do NOT flip _shutdown_orders_cancelled pre-emptively.
        # Only mark cancellation done when cancel_all_orders_detailed()
        # confirms that open-order fetch succeeded AND every cancel attempt
        # returned success.  Otherwise we could log "Cancelled 0 live orders"
        # on a fetch failure and block any further retries while resting
        # orders stay live.  An un-cancelled order during a kill-switch
        # event is a much worse outcome than a duplicate cancel call.
        try:
            detail_fn = getattr(live_trader, "cancel_all_orders_detailed", None)
            if callable(detail_fn):
                result = detail_fn()
                cancelled = int(result.get("cancelled_count", 0))
                success = bool(result.get("success"))
                fetch_ok = bool(result.get("fetch_succeeded"))
                seen = int(result.get("open_orders_seen", 0))
                logger.critical(
                    "Cancel sweep (%s): fetch=%s, seen=%d, cancelled=%d, "
                    "failed=%d, reason=%s",
                    reason,
                    "ok" if fetch_ok else "FAILED",
                    seen,
                    cancelled,
                    int(result.get("failed_count", 0)),
                    result.get("reason"),
                )
                if success:
                    setattr(container, "_shutdown_orders_cancelled", True)
                    return True
                # Fetch failed or partial cancel: leave the flag False so
                # the SIGINT handler / watchdog / next loop iteration can
                # retry.  The caller decides whether to escalate.
                return False
            # Fallback for older live_trader shims that only expose the
            # legacy int-returning cancel_all_orders(): we still have no
            # structured signal, so we leave the pre-P0-2 behaviour
            # (set-then-attempt) rather than leaking half-cancelled state.
            setattr(container, "_shutdown_orders_cancelled", True)
            cancelled = live_trader.cancel_all_orders()
            logger.critical(
                "Cancelled %d live orders during shutdown (%s) [legacy path]",
                cancelled, reason,
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to cancel live orders during shutdown (%s): %s -- "
                "leaving shutdown_orders_cancelled=False so a retry can run",
                reason, exc,
            )
            setattr(container, "_shutdown_orders_cancelled", False)
            return False


def check_file_kill_switch(container) -> bool:
    """Emergency stop via file presence.

    If the configured kill-switch file exists, cancel all live orders, mark the
    live trader as killed, and request the bot loop to stop.  The check is
    idempotent and thread-safe — the kill-switch flag is set atomically and
    order cancellation is serialised with any concurrent shutdown signal.
    """
    path = os.environ.get("LIVE_EXTERNAL_KILL_SWITCH_FILE", "").strip() or DEFAULT_KILL_SWITCH_FILE
    if not path or not os.path.exists(path):
        return False

    # Atomic check-and-set: only the first caller does the cancellation work.
    first = _mark_kill_switch_state(container, reason=f"file:{path}")
    if not first:
        return True

    logger.critical("[fast] KILL_SWITCH file detected at %s", path)

    live_trader = getattr(container, "live_trader", None)
    if live_trader:
        try:
            reason = f"file:{path}"
            activate = getattr(live_trader, "activate_kill_switch", None)
            if callable(activate):
                activate(reason, status_reason="external_kill_switch")
            else:
                live_trader.kill_switch_active = True
                live_trader._kill_switch_reason = reason
                live_trader.status_reason = "external_kill_switch"
        except Exception as exc:
            # Log but still fall through to cancel_live_orders_once below —
            # a broken activate_kill_switch() must not block the emergency
            # order-cancel path.  C10.
            logger.error(
                "[fast] Failed to flip sticky kill switch on live_trader (%s) -- "
                "continuing to cancel orders regardless",
                exc,
            )
        # Serialised single-shot cancellation — if the SIGINT handler also
        # fires, only one of the two will actually hit the exchange.  KILL
        # SWITCH is an emergency path, so we force-cancel regardless of the
        # live_trader's dry-run flag (the file being present is the operator's
        # explicit signal to purge).
        cancel_live_orders_once(
            container, reason=f"kill_switch_file:{path}", force=True,
        )

    return True


def _fast_cycle_failure_threshold() -> int:
    """Consecutive fast-cycle failures before we force-kill live trading.  S7."""
    try:
        return max(1, int(os.environ.get("FAST_CYCLE_MAX_CONSECUTIVE_FAILURES", "5")))
    except (TypeError, ValueError):
        return 5


def _record_fast_cycle_success(container) -> None:
    setattr(container, "_fast_cycle_consecutive_failures", 0)


def _record_fast_cycle_failure(container, exc: Exception) -> None:
    """Bump the consecutive-failure counter; trip kill switch on threshold.  S7.

    A fast cycle that keeps throwing means the bot cannot enforce SL/TP,
    reconcile fills, or see the file kill switch — continuing to trade while
    the risk loop is blind is exactly how a canary turns into a blow-up.
    We escalate with both a task-runner FAILED flag (so the supervisor notices)
    and the live-trader kill switch (so no new orders ship).
    """
    count = int(getattr(container, "_fast_cycle_consecutive_failures", 0) or 0) + 1
    setattr(container, "_fast_cycle_consecutive_failures", count)
    threshold = _fast_cycle_failure_threshold()
    if count >= threshold:
        logger.critical(
            "[fast] %d consecutive fast-cycle failures (>= %d) -- escalating: "
            "marking bg-fast FAILED and activating live kill switch.  Last error: %s",
            count, threshold, exc,
        )
        # Mark the task as FAILED so the supervisor / health checks surface it.
        task_runner = getattr(container, "task_runner", None)
        if task_runner is not None:
            try:
                task_runner.mark_failed("bg-fast", f"consecutive_failures={count}")
            except Exception as mark_exc:
                logger.error("[fast] mark_failed('bg-fast') failed: %s", mark_exc)
        # Trip the live kill switch so no new orders ship while the risk
        # loop is broken.  activate_kill_switch is idempotent.
        live_trader = getattr(container, "live_trader", None)
        if live_trader is not None:
            try:
                live_trader.activate_kill_switch(
                    f"fast_cycle_consecutive_failures:{count}",
                    status_reason="fast_cycle_failed",
                )
            except Exception as ks_exc:
                logger.error("[fast] activate_kill_switch failed during escalation: %s", ks_exc)


def run_fast_cycle(container, cycle_count: int) -> None:
    """
    Check SL/TP on open positions, scan for copy-trade signals,
    and monitor crypto.com for whale trades.
    """
    try:
        if check_file_kill_switch(container):
            logger.critical("[fast] Emergency stop requested by KILL_SWITCH file")
            return

        # Always run paper SL/TP monitoring — even when live trading is active,
        # paper positions still need stop-loss/take-profit/time-exit checks.
        # Previously this was skipped when live was active, leaving paper
        # positions unmonitored and without SL/TP enforcement.
        if container.paper_trader:
            closed = container.paper_trader.check_open_positions()
            if closed:
                logger.info("[fast] Closed %d positions (SL/TP)", len(closed))

        if is_live_trading_active(container):
            # Refresh the cached wallet snapshot (perps + spot + total) so the
            # dashboard and logs always see fresh numbers.  snapshot_balance
            # self-rate-limits INFO logs to once every 5 min to avoid spam.
            now_ts = time.time()
            next_allowed, prior_failures = _get_snapshot_backoff_state(container)
            if now_ts >= next_allowed:
                try:
                    container.live_trader.snapshot_balance()
                    _set_snapshot_backoff_state(container, next_try=0.0, failures=0)
                except Exception as exc:
                    failures = prior_failures + 1
                    backoff = min(300.0, float(2 ** min(failures, 8)))
                    _set_snapshot_backoff_state(
                        container,
                        next_try=now_ts + backoff,
                        failures=failures,
                    )
                    logger.warning(
                        "[fast] snapshot_balance error (failure #%d): %s; backing off %.0fs",
                        failures,
                        exc,
                        backoff,
                    )
            container.live_trader.update_daily_pnl_from_fills()
            reconciled = sync_shadow_book_to_live(container)
            if reconciled:
                logger.info("[fast] Reconciled %d shadow trades to exchange", len(reconciled))
            manage_summary = container.live_trader.manage_open_positions()
            if manage_summary.get("updated") or manage_summary.get("closed") or manage_summary.get("failed"):
                logger.info("[fast] Live risk management summary: %s", manage_summary)
            audit_fn = getattr(container.live_trader, "audit_live_order_hygiene", None)
            if callable(audit_fn):
                interval = _live_order_hygiene_audit_interval_cycles()
                if cycle_count % interval == 0:
                    audit_summary = audit_fn(repair=True)
                    failed = int(audit_summary.get("failed", 0) or 0)
                    cancelled = int(audit_summary.get("stale_cancelled", 0) or 0)
                    cancel_failed = int(audit_summary.get("cancel_failed", 0) or 0)
                    if failed or cancelled or cancel_failed:
                        logger.warning("[fast] Live order hygiene audit summary: %s", audit_summary)

        if container.copy_trader:
            copy_signals = container.copy_trader.scan_top_traders(top_n=10)
            if copy_signals:
                copy_executed = container.copy_trader.execute_copy_signals(copy_signals)
                if copy_executed:
                    logger.info("[fast] Copy-traded %d positions", len(copy_executed))
                    mirror_executed_trades_to_live(
                        container,
                        copy_executed,
                        success_label="[fast] LIVE COPY",
                        skip_label="[fast] Live trader requested but not deployable; skipping copy mirroring",
                    )

        # Scan for whale trades on crypto.com and feed into signal pipeline
        _scan_whale_trades(container)

        if check_file_kill_switch(container):
            logger.critical("[fast] Emergency stop requested after fast cycle work")
            return

        # Clean pass — reset the consecutive-failure counter.
        _record_fast_cycle_success(container)

    except Exception as exc:
        logger.error("Error in fast cycle: %s", exc)
        _record_fast_cycle_failure(container, exc)


def _scan_whale_trades(container) -> None:
    """
    Detect whale trades on crypto.com and feed into signal processor.

    Process:
      1. Get whale_scanner from container (or create if absent)
      2. Scan crypto.com for trades >$100K notional
      3. Convert whales to TradeSignal objects
      4. Feed into signal_processor if available
      5. Log whale activity at INFO level
    """
    try:
        # Get or create whale scanner
        if not hasattr(container, "whale_scanner") or container.whale_scanner is None:
            try:
                from src.exchanges.whale_scanner import WhaleScanner
                container.whale_scanner = WhaleScanner(
                    coins=["BTC", "ETH", "SOL"],
                    min_notional=100000.0,
                    cache_ttl=30.0
                )
                logger.info("[whale] WhaleScanner initialized")
            except Exception as e:
                logger.warning(f"[whale] Failed to initialize WhaleScanner: {e}")
                return

        # Scan for whales
        whales = container.whale_scanner.scan_whales()
        if not whales:
            return

        # Convert whales to signals
        try:
            from src.signals.signal_schema import signal_from_whale_trade
            whale_signals = []

            for whale in whales:
                # Adjust confidence based on notional size
                # Larger trades = higher confidence
                notional = whale.get("notional", 0)
                base_confidence = 0.55
                size_bonus = min((notional - 100000) / 1000000, 0.25)  # Up to +0.25
                confidence = min(base_confidence + size_bonus, 0.95)

                signal = signal_from_whale_trade(
                    coin=whale["coin"],
                    whale_side=whale["side"],
                    notional=whale["notional"],
                    exchange=whale.get("exchange", "crypto.com"),
                    confidence=confidence,
                )
                whale_signals.append(signal)

            if whale_signals:
                logger.info(f"[whale] Generated {len(whale_signals)} signals from whale trades")

                # Feed into signal processor if available
                if hasattr(container, "signal_processor") and container.signal_processor:
                    try:
                        # Convert signals back to dict format for signal processor
                        whale_signal_dicts = [
                            {
                                "id": None,
                                "name": f"whale_{sig.coin}_{sig.side.value}",
                                "type": "whale_trade",
                                "coin": sig.coin,
                                "side": sig.side.value,
                                "direction": sig.side.value,
                                "confidence": sig.confidence,
                                "strategy_type": "whale_detection",
                                "current_score": sig.confidence,
                                "source": "whale_trade",
                                "reason": sig.reason,
                                "parameters": {
                                    "coins": [sig.coin],
                                },
                                "metadata": {
                                    "notional": next(
                                        (w["notional"] for w in whales if w["coin"] == sig.coin), 0
                                    ),
                                },
                            }
                            for sig in whale_signals
                        ]
                        # BUG-3 FIX: actually dispatch whale signals to the
                        # processor.  Previously the dicts were built and logged
                        # but never sent, making whale detection purely cosmetic.
                        # Queue them for the next trading cycle via container
                        # attribute so they get injected into top_strategies.
                        if not hasattr(container, "_whale_strategy_queue"):
                            container._whale_strategy_queue = []
                        container._whale_strategy_queue.extend(whale_signal_dicts)
                        logger.info(f"[whale] Queued {len(whale_signal_dicts)} whale signals for next trading cycle")
                    except Exception as e:
                        logger.debug(f"[whale] Could not feed signals to processor: {e}")

                # Log whale activity at INFO level
                for signal in whale_signals:
                    logger.info(
                        f"[whale] {signal.reason} | confidence={signal.confidence:.2f}"
                    )

        except Exception as e:
            logger.warning(f"[whale] Error converting whale trades to signals: {e}")

    except Exception as e:
        logger.debug(f"[whale] Whale scan error: {e}")
