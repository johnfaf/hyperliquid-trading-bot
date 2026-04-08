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
    python main.py --runtime-profile shadow
    python main.py --report     # Generate a report and exit
    python main.py --status     # Print current status and exit
    python main.py --bootstrap  # Cold-start DB seeding
"""
import os
import signal
import sys
import time

# Force unbuffered stdout/stderr for Docker/Railway log visibility
# Belt-and-suspenders: also set via PYTHONUNBUFFERED=1 in Dockerfile
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))


def _extract_runtime_profile(argv):
    for index, arg in enumerate(argv):
        if arg.startswith("--runtime-profile="):
            return arg.split("=", 1)[1].strip()
        if arg == "--runtime-profile" and index + 1 < len(argv):
            return str(argv[index + 1]).strip()
    return ""


_runtime_profile_override = _extract_runtime_profile(sys.argv[1:])
if _runtime_profile_override:
    os.environ["BOT_RUNTIME_PROFILE"] = _runtime_profile_override

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
from src.core.task_runner import SupervisedTaskRunner
from src.core.subsystem_registry import (
    build_subsystems,
    heartbeat_active,
    FUNDABLE_CORE,
    FULL_PROFILE,
)
from src.core.cycles.research_cycle import run_discovery
from src.core.cycles.trading_cycle import run_trading_cycle
from src.core.cycles.fast_cycle import run_fast_cycle
from src.core.cycles.reporting_cycle import run_reporting
from src.data import database as db
from src.data.database import backup_to_json

from cli import (
    build_parser,
    bootstrap_seed_data,
    run_cli_backtest,
    run_candle_backtest,
    run_cache_list,
    run_cache_clear,
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

        # ── Boot sequence ──
        log_persistence_info(self.logger)
        validate_dependencies(self.logger)
        init_database(self.logger)

        # ── Build subsystems ──
        effective_profile = profile or FULL_PROFILE
        self.container = build_subsystems(health_registry, effective_profile)

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
        if self.container.polymarket:
            self.task_runner.register(
                "bg-polymarket",
                self._polymarket_scan,
                interval_seconds=config.POLYMARKET_SCAN_INTERVAL,
                max_retries=10,
            )
            health_registry.register("bg-polymarket", affects_trading=False)

        if self.container.options_scanner:
            self.task_runner.register(
                "bg-options-flow",
                self._options_scan,
                interval_seconds=config.OPTIONS_FLOW_SCAN_INTERVAL,
                max_retries=10,
            )
            health_registry.register("bg-options-flow", affects_trading=False)

    def _polymarket_scan(self):
        self.container.polymarket.scan_markets()
        self.container.polymarket.get_market_sentiment()

    def _options_scan(self):
        self.container.options_scanner.scan_flow()

    # ── Discovery timer persistence ───────────────────────────

    def _save_last_discovery_time(self):
        try:
            with db.get_connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS bot_state "
                    "(key TEXT PRIMARY KEY, value TEXT)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
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
        except Exception:
            return 0.0

    def _get_inventory_snapshot(self) -> dict:
        try:
            trader_count = len(db.get_active_traders())
        except Exception:
            trader_count = 0
        try:
            active_strategy_count = len(db.get_active_strategies())
        except Exception:
            active_strategy_count = 0
        try:
            total_strategy_count = len(db.get_all_strategies())
        except Exception:
            total_strategy_count = active_strategy_count
        return {
            "traders": trader_count,
            "active_strategies": active_strategy_count,
            "total_strategies": total_strategy_count,
        }

    def _strategy_pool_is_starved(self) -> bool:
        inventory = self._get_inventory_snapshot()
        return (
            inventory["traders"] < config.DISCOVERY_RECOVERY_MIN_ACTIVE_TRADERS
            or inventory["active_strategies"] < config.DISCOVERY_RECOVERY_MIN_ACTIVE_STRATEGIES
        )

    def _should_run_recovery_discovery(self, now: float) -> bool:
        if not self._strategy_pool_is_starved():
            return False
        elapsed = now - self._last_discovery if self._last_discovery else float("inf")
        return elapsed >= config.DISCOVERY_STARVATION_RECOVERY_INTERVAL

    def _should_force_startup_discovery(self, inventory: dict) -> bool:
        """
        Only block startup on discovery when the bot has no usable inventory.

        If tracked strategies already exist, let the trading cycle run first so
        the scorer can reactivate them immediately instead of sitting behind a
        long discovery pass.
        """
        if inventory.get("traders", 0) == 0:
            return True
        if inventory.get("total_strategies", 0) == 0 and self._strategy_pool_is_starved():
            return True
        return False

    # ── Scheduling loops ──────────────────────────────────────

    def _run_discovery(self):
        run_discovery(self.container)
        heartbeat_active(self.container, health_registry)
        self._last_discovery = time.time()
        self._save_last_discovery_time()

    def _run_trading_cycle(self):
        self._cycle_count += 1
        run_trading_cycle(self.container, self._cycle_count)
        run_reporting(self.container, self._cycle_count, health_registry)
        heartbeat_active(self.container, health_registry)

    def _fast_cycle(self):
        self._fast_cycle_count += 1
        run_fast_cycle(self.container, self._fast_cycle_count)
        heartbeat_active(self.container, health_registry)

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
        def signal_handler(sig, frame):
            self.logger.info("Shutdown signal received. Stopping background tasks…")
            self.task_runner.stop_all(timeout=10)
            try:
                backup_to_json()
                self.logger.info("DB backup complete.")
            except Exception as exc:
                self.logger.error("DB backup failed on shutdown: %s", exc)
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # ── Start supervised background tasks ──
        self.task_runner.start_all()

        self.logger.info("Bot starting continuous operation…")
        self.logger.info("  Fast cycle:      every %ds", config.FAST_CYCLE_INTERVAL)
        self.logger.info("  Trading cycle:   every %ds", config.TRADING_CYCLE_INTERVAL)
        self.logger.info("  Discovery cycle: every %ds", config.DISCOVERY_CYCLE_INTERVAL)
        sys.stdout.flush()

        # Initial discovery if needed
        inventory = self._get_inventory_snapshot()
        trader_count = inventory["traders"]

        if self._should_force_startup_discovery(inventory):
            if trader_count == 0:
                self.logger.info("No traders in DB — running initial discovery…")
            else:
                self.logger.warning(
                    "Strategy pool starved on startup with no tracked strategies — forcing recovery discovery "
                    "(traders=%d, active_strategies=%d, total_strategies=%d)",
                    inventory["traders"],
                    inventory["active_strategies"],
                    inventory["total_strategies"],
                )
            self._run_discovery()
        elif self._strategy_pool_is_starved():
            self.logger.warning(
                "Strategy pool is thin on startup, but tracked strategies exist — "
                "allowing scoring/trading before recovery discovery "
                "(traders=%d, active_strategies=%d, total_strategies=%d)",
                inventory["traders"],
                inventory["active_strategies"],
                inventory["total_strategies"],
            )
            restored_ts = self._restore_last_discovery_time()
            if restored_ts and (time.time() - restored_ts) < config.DISCOVERY_CYCLE_INTERVAL:
                self._last_discovery = restored_ts
                remaining_h = (config.DISCOVERY_CYCLE_INTERVAL - (time.time() - restored_ts)) / 3600
                self.logger.info("Restored discovery timer, next in %.1fh", remaining_h)
            else:
                self._last_discovery = time.time()
                self.logger.info(
                    "DB has %d traders and %d tracked strategies — next recovery discovery check in %.1fh",
                    trader_count,
                    inventory["total_strategies"],
                    config.DISCOVERY_STARVATION_RECOVERY_INTERVAL / 3600,
                )
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

        # ── Main loop ──
        while self.running:
            now = time.time()
            try:
                # Tier 3: Discovery (daily)
                if now - self._last_discovery >= config.DISCOVERY_CYCLE_INTERVAL:
                    self._run_discovery()
                elif self._should_run_recovery_discovery(now):
                    inventory = self._get_inventory_snapshot()
                    self.logger.warning(
                        "Strategy pool still starved â€” running recovery discovery "
                        "(traders=%d, active_strategies=%d, total_strategies=%d)",
                        inventory["traders"],
                        inventory["active_strategies"],
                        inventory["total_strategies"],
                    )
                    self._run_discovery()

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

            # Status heartbeat every 10 fast cycles
            if self._fast_cycle_count % 10 == 0 and self._fast_cycle_count > 0:
                if self.container.reporter:
                    self.logger.info(self.container.reporter.print_live_status())

            sys.stdout.flush()
            if self.running:
                time.sleep(config.FAST_CYCLE_INTERVAL)

        # ── Shutdown ──
        self.task_runner.stop_all(timeout=10)
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
