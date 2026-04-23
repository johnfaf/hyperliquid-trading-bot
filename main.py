#!/usr/bin/env python3
"""
Hyperliquid Auto-Research Trading Bot
======================================
Slim orchestrator.  All heavy logic lives in domain modules:

  src/core/boot.py               — logging, dependency validation, DB init
  src/core/subsystem_registry.py — instantiate + wire subsystems
  src/core/health_registry.py    — per-subsystem health states
  src/core/task_runner.py        — supervised background threads
  src/core/cycles/               — research, trading, fast, reporting cycles
  cli.py                         — CLI entrypoints (backtest, bootstrap, etc.)

Usage:
    python main.py              # Run the full bot loop
    python main.py --once       # Run a single cycle then exit
    python main.py --core-only  # Run with fundable-core profile only
    python main.py --report     # Generate a report and exit
    python main.py --status     # Print current status and exit
    python main.py --bootstrap  # Cold-start DB seeding
"""
import os
import signal
import sys
import threading
import time

# Force unbuffered stdout/stderr for Docker/Railway log visibility
# Belt-and-suspenders: also set via PYTHONUNBUFFERED=1 in Dockerfile
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, os.path.dirname(__file__))

# Load .env BEFORE importing config so os.environ.get() calls pick up values.
# On Railway/Docker the vars come from the platform; load_dotenv() is a no-op
# when variables are already set, so this is always safe to call.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

import config

from src.core.boot import (
    setup_logging,
    validate_dependencies,
    init_database,
    log_persistence_info,
)
from src.core.health_registry import registry as health_registry
from src.core.readiness import RuntimeIncidentMonitor
from src.core.runtime_config import RuntimeConfigManager
from src.core.task_runner import SupervisedTaskRunner
from src.core.subsystem_registry import (
    build_subsystems,
    heartbeat_active,
    FUNDABLE_CORE,
    FULL_PROFILE,
)
from src.core.cycles.research_cycle import run_discovery
from src.core.cycles.trading_cycle import run_trading_cycle
from src.core.cycles.fast_cycle import (
    run_fast_cycle,
    check_file_kill_switch,
    cancel_live_orders_once,
    start_kill_switch_watchdog,
    stop_kill_switch_watchdog,
    DEFAULT_KILL_SWITCH_FILE,
)
from src.core.cycles.reporting_cycle import run_reporting
from src.core.cycles.feature_cycle import run_feature_cycle, backfill_all as backfill_features, feature_store_is_empty
from src.data import database as db
from src.data.database import backup_to_json

from cli import (
    build_parser,
    bootstrap_seed_data,
    run_cli_backtest,
    run_candle_backtest,
    run_cache_list,
    run_cache_clear,
    run_db_audit_cli,
    run_db_repair_cli,
)


# ─── Bot Engine ────────────────────────────────────────────────

class HyperliquidResearchBot:
    """
    Orchestrates the 3-tier scheduling loop.  Delegates all work to cycle
    modules and uses the health registry to track subsystem state.
    """

    def __init__(self, profile=None):
        self.logger = setup_logging()
        self.running = False
        self._last_research = 0
        self._last_discovery = 0
        self._last_report = 0
        self._cycle_count = 0
        self._fast_cycle_count = 0
        self._shutdown_orders_cancelled = False
        self._discovery_thread = None
        self._discovery_state_lock = threading.Lock()
        self._discovery_retry_after_ts = 0.0

        # ── Boot sequence ──
        log_persistence_info(self.logger)
        validate_dependencies(self.logger)
        init_database(self.logger)
        try:
            from src.data.db_audit import format_db_audit_report, run_db_audit

            audit_report = run_db_audit(include_candle_cache=True, include_code_scan=False)
            block_severity = getattr(config, "READINESS_DB_AUDIT_BLOCK_SEVERITY", "high")
            if audit_report.findings_at_or_above(block_severity):
                self.logger.warning(
                    "Database audit found readiness blockers:\n%s",
                    format_db_audit_report(audit_report, block_severity=block_severity),
                )
            else:
                self.logger.info(
                    "Database audit passed readiness threshold (%s): %d finding(s)",
                    block_severity,
                    len(audit_report.findings),
                )
        except Exception as exc:
            self.logger.warning("Database audit skipped during boot: %s", exc)

        # ── Build subsystems ──
        effective_profile = profile or FULL_PROFILE
        self.container = build_subsystems(health_registry, effective_profile)
        self.runtime_config = RuntimeConfigManager()
        self.runtime_config.poll(self.container, force=True)
        self.runtime_monitor = RuntimeIncidentMonitor()

        # Wire Telegram critical alert for any subsystem that transitions
        # to FAILED — operator is notified within seconds, not on next log read.
        try:
            from src.notifications.telegram_alerts import send_subsystem_failure_alert
            health_registry.set_failure_callback(send_subsystem_failure_alert)
        except Exception as exc:
            self.logger.warning("Could not wire failure alert callback: %s", exc)

        # ── Feature store initial backfill (Postgres-only, non-blocking) ──
        if getattr(config, "POSTGRES_DSN", ""):
            try:
                from src.data.db.router import init_postgres_schema
                init_postgres_schema()
                if feature_store_is_empty():
                    self.logger.info("Feature store empty — running initial backfill…")
                    backfill_features(self.container)
            except Exception as exc:
                self.logger.warning("Feature store init skipped: %s", exc)

        # ── Supervised background tasks ──
        self.task_runner = SupervisedTaskRunner(health_registry=health_registry)
        self._register_background_tasks()

        self.logger.info("Bot initialized.")
        sys.stdout.flush()
        try:
            self.logger.info(health_registry.get_health_report())
        except Exception as exc:
            self.logger.warning("Health report failed: %s", exc)
        sys.stdout.flush()

    # ── Background tasks (replaces raw daemon threads) ────────

    def _register_background_tasks(self):
        """Register background scanner tasks with the supervised runner."""
        heartbeat_interval = max(
            15.0,
            min(60.0, float(getattr(config, "READINESS_STALE_SECONDS", 600)) / 4.0),
        )
        heartbeat_startup_delay = max(
            0.0,
            float(os.environ.get("BG_HEARTBEAT_STARTUP_DELAY_S", "5")),
        )
        polymarket_startup_delay = max(
            0.0,
            float(os.environ.get("BG_POLYMARKET_STARTUP_DELAY_S", "20")),
        )
        options_startup_delay = max(
            0.0,
            float(os.environ.get("BG_OPTIONS_FLOW_STARTUP_DELAY_S", "40")),
        )
        self.task_runner.register(
            "bg-heartbeat",
            self._background_heartbeat,
            interval_seconds=heartbeat_interval,
            initial_delay_seconds=heartbeat_startup_delay,
            max_retries=3,
        )
        health_registry.register("bg-heartbeat", affects_trading=False)

        if self.container.polymarket:
            self.task_runner.register(
                "bg-polymarket",
                self._polymarket_scan,
                interval_seconds=config.POLYMARKET_SCAN_INTERVAL,
                initial_delay_seconds=polymarket_startup_delay,
                max_retries=10,
            )
            health_registry.register("bg-polymarket", affects_trading=False)

        if self.container.options_scanner:
            self.task_runner.register(
                "bg-options-flow",
                self._options_scan,
                interval_seconds=config.OPTIONS_FLOW_SCAN_INTERVAL,
                initial_delay_seconds=options_startup_delay,
                max_retries=10,
            )
            health_registry.register("bg-options-flow", affects_trading=False)

    def _polymarket_scan(self):
        self.container.polymarket.scan_markets()
        self.container.polymarket.get_market_sentiment()

    def _options_scan(self):
        self.container.options_scanner.scan_flow()

    def _background_heartbeat(self):
        heartbeat_active(self.container, health_registry)
        try:
            from src.learning.source_inventory import persist_source_health_snapshot

            persist_source_health_snapshot(getattr(self.container, "data_source_registry", None))
        except Exception as exc:
            self.logger.debug("Source-health snapshot skipped: %s", exc)

    # ── Discovery timer persistence ───────────────────────────

    def _save_last_discovery_time(self):
        try:
            with db.get_connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS bot_state "
                    "(key TEXT PRIMARY KEY, value TEXT)"
                )
                conn.execute(
                    "INSERT INTO bot_state (key, value) VALUES (?, ?) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    ("last_discovery_ts", str(self._last_discovery)),
                )
        except Exception:
            pass

    def _restore_last_discovery_time(self) -> float:
        try:
            with db.get_connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS bot_state "
                    "(key TEXT PRIMARY KEY, value TEXT)"
                )
                row = conn.execute(
                    "SELECT value FROM bot_state WHERE key = ?",
                    ("last_discovery_ts",),
                ).fetchone()
            return float(row["value"]) if row else 0.0
        except Exception as exc:
            self.logger.warning("Could not restore last discovery timestamp: %s", exc)
            return 0.0

    # ── Scheduling loops ──────────────────────────────────────

    def _run_discovery(self):
        self.runtime_config.poll(self.container)
        run_feature_cycle(self.container, tier="daily")
        run_discovery(self.container)
        heartbeat_active(self.container, health_registry)
        self.runtime_monitor.evaluate_and_alert(
            container=self.container,
            health_registry=health_registry,
        )
        self._last_discovery = time.time()
        self._save_last_discovery_time()

    def _discovery_running(self) -> bool:
        with self._discovery_state_lock:
            return bool(self._discovery_thread and self._discovery_thread.is_alive())

    def _start_discovery_async(self, reason: str) -> bool:
        now = time.time()
        with self._discovery_state_lock:
            if self._discovery_thread and self._discovery_thread.is_alive():
                self.logger.info(
                    "Discovery already running in background — skipping %s request",
                    reason,
                )
                return False
            if now < self._discovery_retry_after_ts:
                retry_in = max(0.0, self._discovery_retry_after_ts - now)
                self.logger.info(
                    "Discovery backoff active — next background retry in %.0fs (%s)",
                    retry_in,
                    reason,
                )
                return False

            def _runner():
                started = time.time()
                failed = False
                self.logger.info(
                    "Starting background discovery (%s) — trading loop remains active",
                    reason,
                )
                try:
                    self._run_discovery()
                except Exception as exc:
                    failed = True
                    backoff = min(900.0, max(60.0, config.FAST_CYCLE_INTERVAL * 5.0))
                    with self._discovery_state_lock:
                        self._discovery_retry_after_ts = time.time() + backoff
                    self.logger.error(
                        "Background discovery failed (%s): %s",
                        reason,
                        exc,
                        exc_info=True,
                    )
                    try:
                        self.runtime_monitor.evaluate_and_alert(
                            container=self.container,
                            health_registry=health_registry,
                        )
                    except Exception:
                        pass
                finally:
                    elapsed = time.time() - started
                    with self._discovery_state_lock:
                        self._discovery_thread = None
                        if not failed:
                            self._discovery_retry_after_ts = 0.0
                    if not failed:
                        self.logger.info(
                            "Background discovery complete (%.1fs, reason=%s)",
                            elapsed,
                            reason,
                        )

            self._discovery_thread = threading.Thread(
                target=_runner,
                daemon=True,
                name=f"bg-discovery-{reason}",
            )
            self._discovery_thread.start()
        return True

    def _run_trading_cycle(self):
        self.runtime_config.poll(self.container)
        try:
            if check_file_kill_switch(self.container):
                self.logger.critical("KILL_SWITCH triggered before trading cycle execution")
                self.running = False
                return
        except Exception as exc:
            self.logger.warning("Kill-switch check failed before trading cycle: %s", exc)
        self._cycle_count += 1
        run_feature_cycle(self.container, tier="trading")
        run_trading_cycle(self.container, self._cycle_count)
        run_reporting(self.container, self._cycle_count, health_registry)
        heartbeat_active(self.container, health_registry)
        self.runtime_monitor.evaluate_and_alert(
            container=self.container,
            health_registry=health_registry,
        )

    def _fast_cycle(self):
        self.runtime_config.poll(self.container)
        self._fast_cycle_count += 1
        if check_file_kill_switch(self.container):
            self.logger.critical("KILL_SWITCH triggered before fast cycle execution")
            self.running = False
            return
        run_fast_cycle(self.container, self._fast_cycle_count)
        if getattr(self.container, "_stop_requested", False):
            self.logger.critical("Stop requested during fast cycle; exiting main loop")
            self.running = False
            return
        heartbeat_active(self.container, health_registry)
        self.runtime_monitor.evaluate_and_alert(
            container=self.container,
            health_registry=health_registry,
        )

    def _cancel_live_orders_for_shutdown(self, reason: str) -> None:
        # CRIT-FIX C5: delegate to the shared, lock-guarded single-shot helper
        # in fast_cycle so the signal-handler path and the file-kill-switch
        # path cannot both issue cancel_all_orders() concurrently.  The helper
        # is idempotent — subsequent calls are no-ops.
        cancel_live_orders_once(self.container, reason=reason)
        self._shutdown_orders_cancelled = True

    def _run_with_timeout(self, name: str, func, timeout_s: float) -> bool:
        """Run shutdown work with a hard wait bound so SIGTERM cannot hang forever."""
        done = threading.Event()
        errors = []

        def _target():
            try:
                func()
            except Exception as exc:  # pragma: no cover - defensive shutdown path
                errors.append(exc)
            finally:
                done.set()

        worker = threading.Thread(target=_target, daemon=True, name=f"shutdown-{name}")
        worker.start()
        worker.join(max(0.1, float(timeout_s)))
        if not done.is_set():
            self.logger.error("%s timed out after %.1fs during shutdown; continuing", name, timeout_s)
            return False
        if errors:
            self.logger.error("%s failed during shutdown: %s", name, errors[0])
            return False
        return True

    def _sleep_with_kill_switch_checks(self, interval_s: float) -> None:
        deadline = time.time() + max(0.0, float(interval_s))
        while self.running and time.time() < deadline:
            try:
                self.runtime_config.poll(self.container)
            except Exception as exc:
                self.logger.warning("Runtime config poll failed during sleep window: %s", exc)

            try:
                if check_file_kill_switch(self.container):
                    self.logger.critical("KILL_SWITCH detected during sleep window")
                    self.running = False
                    return
            except Exception as exc:
                self.logger.warning("Kill-switch check failed during sleep window: %s", exc)

            remaining = deadline - time.time()
            time.sleep(min(1.0, max(0.0, remaining)))

    @staticmethod
    def _kill_switch_file_path() -> str:
        return (
            os.environ.get("LIVE_EXTERNAL_KILL_SWITCH_FILE", "").strip()
            or DEFAULT_KILL_SWITCH_FILE
        )

    def _wait_for_startup_kill_switch_clear(self) -> None:
        """Pause startup while the external kill-switch file is present."""
        path = self._kill_switch_file_path()
        if not path:
            return

        warned = False
        while self.running and os.path.exists(path):
            try:
                check_file_kill_switch(self.container)
            except Exception as exc:
                self.logger.warning("Startup kill-switch check failed: %s", exc)

            if not warned:
                self.logger.critical(
                    "KILL_SWITCH active at startup (%s). Background tasks and "
                    "trading loops are paused until the file is removed.",
                    path,
                )
                warned = True

            try:
                self.runtime_config.poll(self.container)
            except Exception as exc:
                self.logger.warning(
                    "Runtime config poll failed while startup kill switch is active: %s",
                    exc,
                )
            time.sleep(1.0)

        if warned and self.running:
            # Reset the edge-trigger bookkeeping now that the file itself is
            # gone, so a later real shutdown or future file-trigger can still
            # dispatch order cancels exactly once.
            setattr(self.container, "_file_kill_switch_triggered", False)
            setattr(self.container, "_stop_requested", False)
            setattr(self.container, "_shutdown_orders_cancelled", False)
            self.logger.info(
                "Startup KILL_SWITCH file cleared; resuming bot startup. "
                "Persisted live-trader kill-switch state, if any, still applies."
            )

    def run_once(self):
        """Run discovery + trading cycle (CLI --once)."""
        self._run_discovery()
        self._run_trading_cycle()

    def generate_report(self):
        if self.container.reporter:
            print(self.container.reporter.generate_daily_report())

    def print_status(self):
        if self.container.reporter:
            print(self.container.reporter.print_live_status())
        print(health_registry.get_health_report())

    def run_loop(self):
        """
        3-tier continuous loop:
          Tier 1 — Fast (60s):     position SL/TP, copy-trade scan
          Tier 2 — Trading (5m):   regime, scoring, paper+live trading
          Tier 3 — Discovery (24h): leaderboard scan, strategy ID
        """
        self.running = True
        self.logger.info("Entering run_loop()…")
        sys.stdout.flush()

        # ── Graceful shutdown handler ──
        # Second SIGINT/SIGTERM forces immediate exit so operators can never be
        # held hostage by a hung shutdown path (exchange call wedged, etc.).
        shutdown_in_progress = {"flag": False}

        def signal_handler(sig, frame):
            if shutdown_in_progress["flag"]:
                # Second signal → force exit immediately.
                self.logger.critical(
                    "Second shutdown signal received — forcing immediate exit."
                )
                os._exit(1)

            shutdown_in_progress["flag"] = True
            # CRITICAL: set self.running = False FIRST so the main loop and any
            # worker threads observing it stop scheduling new work even if the
            # cancel/backup calls below hang inside their watchdog windows.
            self.running = False
            self.logger.info("Shutdown signal received. Stopping background tasks…")
            # Stop the independent kill-switch watchdog BEFORE the synchronous
            # shutdown cancel — otherwise the watchdog thread could race with
            # our cancel call and double-dispatch cancellations.
            try:
                stop_kill_switch_watchdog(timeout=2.0)
            except Exception as exc:
                self.logger.warning("stop_kill_switch_watchdog failed: %s", exc)
            self._run_with_timeout(
                "cancel_live_orders",
                lambda: self._cancel_live_orders_for_shutdown(reason=f"signal:{sig}"),
                float(os.environ.get("SHUTDOWN_CANCEL_TIMEOUT_S", "8")),
            )
            self.task_runner.stop_all(timeout=10)
            if self._run_with_timeout(
                "backup_to_json",
                backup_to_json,
                float(os.environ.get("SHUTDOWN_BACKUP_TIMEOUT_S", "15")),
            ):
                self.logger.info("DB backup complete.")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self._wait_for_startup_kill_switch_clear()

        if self.running:
            # Start supervised background tasks
            self.task_runner.start_all()

            self.logger.info("Bot starting continuous operation…")
            self.logger.info("  Fast cycle:      every %ds", config.FAST_CYCLE_INTERVAL)
            self.logger.info("  Trading cycle:   every %ds", config.TRADING_CYCLE_INTERVAL)
            self.logger.info("  Discovery cycle: every %ds", config.DISCOVERY_CYCLE_INTERVAL)
            sys.stdout.flush()

            # Initial discovery if needed
            trader_count = 0
            try:
                trader_count = len(db.get_active_traders())
            except Exception:
                pass

            if trader_count == 0:
                self.logger.info(
                    "No traders in DB — scheduling initial discovery in background "
                    "so trading cycles can continue."
                )
                self._start_discovery_async("startup_empty_trader_pool")
            else:
                restored_ts = self._restore_last_discovery_time()
                if restored_ts and (time.time() - restored_ts) < config.DISCOVERY_CYCLE_INTERVAL:
                    self._last_discovery = restored_ts
                    remaining_h = (config.DISCOVERY_CYCLE_INTERVAL - (time.time() - restored_ts)) / 3600
                    self.logger.info("Restored discovery timer, next in %.1fh", remaining_h)
                else:
                    self._last_discovery = time.time()
                    self.logger.info(
                        "DB has %d traders — next discovery in %.0fh",
                        trader_count, config.DISCOVERY_CYCLE_INTERVAL / 3600,
                    )

            # Independent kill-switch watchdog (P0-1)
            # Polls /data/KILL_SWITCH on a 1s cadence regardless of what the
            # main loop is doing. Catches the case where a long trading cycle
            # is mid-flight when the operator drops the file.
            try:
                watchdog_interval_s = float(
                    os.environ.get("LIVE_KILL_SWITCH_WATCHDOG_INTERVAL_S", "1.0")
                )
            except (TypeError, ValueError):
                watchdog_interval_s = 1.0
            start_kill_switch_watchdog(self.container, interval_s=watchdog_interval_s)

        # ── Main loop ──
        while self.running:
            now = time.time()
            try:
                # Tier 3: Discovery (daily)
                if (
                    not self._discovery_running()
                    and now - self._last_discovery >= config.DISCOVERY_CYCLE_INTERVAL
                ):
                    self._start_discovery_async("scheduled_cycle")

                # Tier 2: Trading (5 min)
                if now - self._last_research >= config.TRADING_CYCLE_INTERVAL:
                    self._run_trading_cycle()
                    self._last_research = now
                else:
                    # Tier 1: Fast (60s)
                    self._fast_cycle()

                # Daily report
                if now - self._last_report >= 86400:
                    self.logger.info("Generating daily report…")
                    if self.container.reporter:
                        self.container.reporter.generate_daily_report()
                    self._last_report = now

            except Exception as exc:
                self.logger.error("Error in main loop: %s", exc, exc_info=True)
                try:
                    self.runtime_monitor.evaluate_and_alert(
                        container=self.container,
                        health_registry=health_registry,
                    )
                except Exception:
                    pass

            # Status heartbeat every 10 fast cycles
            if self._fast_cycle_count % 10 == 0 and self._fast_cycle_count > 0:
                if self.container.reporter:
                    self.logger.info(self.container.reporter.print_live_status())

            sys.stdout.flush()
            if self.running:
                self._sleep_with_kill_switch_checks(config.FAST_CYCLE_INTERVAL)

        # ── Shutdown ──
        # Stop the watchdog BEFORE the shutdown cancel path so only one thread
        # is driving the exchange cancel RPC.
        try:
            stop_kill_switch_watchdog(timeout=2.0)
        except Exception as exc:
            self.logger.warning("stop_kill_switch_watchdog failed: %s", exc)
        self._cancel_live_orders_for_shutdown(reason="run_loop_exit")
        self.task_runner.stop_all(timeout=10)
        if self._discovery_running():
            self.logger.info(
                "Background discovery is still running during shutdown — not waiting for completion."
            )
        try:
            if getattr(self.container, "position_monitor", None):
                self.container.position_monitor.stop()
        except Exception as exc:
            self.logger.warning("Position monitor stop failed: %s", exc)
        try:
            from src.data.hyperliquid_client import stop_websocket
            stop_websocket()
        except Exception as exc:
            self.logger.warning("Market websocket stop failed: %s", exc)
        try:
            if getattr(self.container, "dashboard", None):
                self.container.dashboard.shutdown()
                self.container.dashboard.server_close()
        except Exception as exc:
            self.logger.warning("Dashboard shutdown failed: %s", exc)
        self.logger.info("Bot stopped.")
        if self.container.reporter:
            self.container.reporter.generate_daily_report()


# ─── CLI ───────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Commands that don't need the full bot
    if args.bootstrap:
        from src.core.boot import setup_logging
        logger = setup_logging()
        bootstrap_seed_data(logger, days=args.bootstrap_days)
        return

    if args.reset_paper:
        init_database(setup_logging())
        balance = args.reset_balance or config.PAPER_TRADING_INITIAL_BALANCE
        result = db.reset_paper_trades(balance)
        print(f"Paper trades cleared ({result['open_deleted']} open + "
              f"{result['closed_deleted']} closed). Balance reset to ${balance:,.2f}")
        return

    if args.cache_list:
        run_cache_list(setup_logging())
        return

    if args.cache_clear:
        run_cache_clear(setup_logging())
        return

    if args.db_repair:
        sys.exit(run_db_repair_cli(args))

    if args.db_audit:
        sys.exit(run_db_audit_cli(args))

    if args.candle_backtest:
        run_candle_backtest(setup_logging(), args)
        return

    if args.backtest:
        run_cli_backtest(setup_logging(), args)
        return

    # Commands that need the bot
    profile = FUNDABLE_CORE if args.core_only else None
    bot = HyperliquidResearchBot(profile=profile)

    if args.status:
        bot.print_status()
    elif args.report:
        bot.generate_report()
    elif args.once:
        bot.run_once()
    else:
        bot.run_loop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import logging
        logging.getLogger("boot").critical("FATAL: unhandled exception — %s", exc, exc_info=True)
        sys.exit(1)
