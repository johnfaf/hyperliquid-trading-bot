#!/usr/bin/env python3
"""
Hyperliquid Auto-Research Trading Bot
======================================
Main orchestrator that runs the continuous research, strategy identification,
scoring, and paper trading loop.

Usage:
    python main.py              # Run the full bot loop
    python main.py --once       # Run a single cycle then exit
    python main.py --report     # Generate a report and exit
    python main.py --status     # Print current status and exit
"""
import sys
import os
import time
import signal
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.database import init_db, restore_from_json, backup_to_json
from src.trader_discovery import TraderDiscovery
from src.strategy_identifier import StrategyIdentifier
from src.strategy_scorer import StrategyScorer
from src.paper_trader import PaperTrader
from src.copy_trader import CopyTrader
from src.reporter import Reporter
from src.dashboard import start_dashboard

# ─── Logging Setup ─────────────────────────────────────────────

def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, f"bot_{datetime.utcnow().strftime('%Y%m%d')}.log")

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config.LOG_LEVEL))
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    return logging.getLogger(__name__)


# ─── Bot Engine ────────────────────────────────────────────────

class HyperliquidResearchBot:
    """
    Main bot that orchestrates:
    1. Trader discovery (find top performers)
    2. Strategy identification (classify what they're doing)
    3. Strategy scoring (rank strategies, decay bad ones)
    4. Paper trading (simulate trades from top strategies)
    5. Copy trading (mirror top traders' live position changes)
    6. Reporting (generate insights)
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = False
        self._last_research = 0
        self._last_scoring = 0
        self._last_report = 0
        self._cycle_count = 0
        self._fast_cycle_count = 0

        # Initialize components
        self.logger.info("Initializing bot components...")
        init_db()

        # Restore from backup if DB is empty (e.g. after Railway redeploy)
        if restore_from_json():
            self.logger.info("Restored DB from backup (post-deploy recovery)")
        self.discovery = TraderDiscovery()
        self.identifier = StrategyIdentifier()
        self.scorer = StrategyScorer()
        self.paper_trader = PaperTrader()
        self.copy_trader = CopyTrader()
        self.reporter = Reporter()

        # Start the web dashboard
        try:
            self.dashboard = start_dashboard()
            self.logger.info("Web dashboard started.")
        except Exception as e:
            self.logger.warning(f"Dashboard failed to start: {e}")

        self.logger.info("Bot initialized successfully.")

    def run_once(self):
        """Run a single complete research + trading cycle."""
        self._cycle_count += 1
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting cycle #{self._cycle_count}")
        self.logger.info(f"{'='*60}")

        try:
            # Phase 1: Discover and analyze traders
            self.logger.info("Phase 1: Trader Discovery")
            discovery_result = self.discovery.run_discovery_cycle()
            self.logger.info(f"  Discovered: {discovery_result.get('traders_discovered', 0)} traders")
            self.logger.info(f"  Analyzed: {discovery_result.get('traders_analyzed', 0)} traders")

            # Phase 2: Identify strategies from analyzed traders
            self.logger.info("Phase 2: Strategy Identification")
            from src.database import get_active_traders
            traders = get_active_traders()
            all_strategies = []

            for trader in traders:
                # Build a minimal profile for strategy identification
                from src import hyperliquid_client as hl
                state = hl.get_user_state(trader["address"])
                if state:
                    profile = {
                        "address": trader["address"],
                        "positions": state["positions"],
                        "position_analysis": self.discovery._analyze_positions(state["positions"]),
                        "trade_analysis": {
                            "total_trades": trader["trade_count"],
                            "win_rate": trader["win_rate"],
                            "total_closed_pnl": trader["total_pnl"],
                            "trading_frequency": "unknown",
                            "profit_factor": 1.5,
                            "coins_traded": [p["coin"] for p in state["positions"] if p["size"] > 0],
                        },
                    }
                    strategies = self.identifier.identify_strategies(profile)
                    all_strategies.extend(strategies)
                time.sleep(0.3)

            if all_strategies:
                saved_ids = self.identifier.save_identified_strategies(all_strategies)
                self.logger.info(f"  Identified and saved {len(saved_ids)} strategies")

            # Phase 3: Score all strategies
            self.logger.info("Phase 3: Strategy Scoring")
            score_results = self.scorer.score_all_strategies()
            self.logger.info(f"  Scored {len(score_results)} strategies")

            # Phase 4: Paper trade top strategies (increased from 5 to 15)
            self.logger.info("Phase 4: Paper Trading")
            top_strategies = self.scorer.get_top_strategies(n=15)

            # Check existing positions first
            closed = self.paper_trader.check_open_positions()
            if closed:
                self.logger.info(f"  Closed {len(closed)} positions")

            # Execute new signals from strategies
            if top_strategies:
                executed = self.paper_trader.execute_strategy_signals(top_strategies)
                self.logger.info(f"  Executed {len(executed)} new paper trades")

            # Phase 4b: Copy trading - mirror top trader positions
            self.logger.info("Phase 4b: Copy Trading")
            copy_signals = self.copy_trader.scan_top_traders(top_n=10)
            if copy_signals:
                copy_executed = self.copy_trader.execute_copy_signals(copy_signals)
                self.logger.info(f"  Executed {len(copy_executed)} copy trades")

            # Phase 5: Report
            self.logger.info("Phase 5: Status Update")
            status = self.reporter.print_live_status()
            print(status)

            # Generate improvement report
            improvement = self.scorer.generate_improvement_report()
            health = improvement.get("health", "unknown")
            self.logger.info(f"  Bot health: {health}")
            self.logger.info(f"  Improving strategies: {improvement.get('improving', 0)}")
            self.logger.info(f"  Declining strategies: {improvement.get('declining', 0)}")

            # Backup DB state (survives Railway redeploys)
            backup_to_json()

            self.logger.info(f"Cycle #{self._cycle_count} complete.")

        except Exception as e:
            self.logger.error(f"Error in cycle #{self._cycle_count}: {e}", exc_info=True)

    def _fast_cycle(self):
        """
        Fast cycle: check positions + copy-trade scan.
        Runs every 60s between full research cycles.
        """
        self._fast_cycle_count += 1
        try:
            # Check SL/TP on open positions
            closed = self.paper_trader.check_open_positions()
            if closed:
                self.logger.info(f"[fast] Closed {len(closed)} positions (SL/TP)")

            # Scan top traders for position changes
            copy_signals = self.copy_trader.scan_top_traders(top_n=10)
            if copy_signals:
                copy_executed = self.copy_trader.execute_copy_signals(copy_signals)
                if copy_executed:
                    self.logger.info(f"[fast] Copy-traded {len(copy_executed)} positions")

        except Exception as e:
            self.logger.error(f"Error in fast cycle: {e}")

    def run_loop(self):
        """Run the bot in a continuous loop."""
        self.running = True

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            self.logger.info("Shutdown signal received. Stopping...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.logger.info("Bot starting continuous operation...")
        self.logger.info(f"  Fast cycle interval: 60s (position checks + copy trading)")
        self.logger.info(f"  Full research interval: {config.RESEARCH_CYCLE_INTERVAL}s")

        while self.running:
            now = time.time()

            try:
                # Run full research cycle periodically (every hour)
                if now - self._last_research >= config.RESEARCH_CYCLE_INTERVAL:
                    self.run_once()
                    self._last_research = now
                else:
                    # Fast cycle: check positions + copy trades (every 60s)
                    self._fast_cycle()

                # Generate daily report
                if now - self._last_report >= 86400:
                    self.logger.info("Generating daily report...")
                    self.reporter.generate_daily_report()
                    self._last_report = now

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

            # Status heartbeat every 10 fast cycles
            if self._fast_cycle_count % 10 == 0 and self._fast_cycle_count > 0:
                print(self.reporter.print_live_status())

            # Sleep 60s between fast cycles
            if self.running:
                time.sleep(60)

        self.logger.info("Bot stopped.")
        # Final report
        self.reporter.generate_daily_report()

    def generate_report(self):
        """Generate a one-off report."""
        report = self.reporter.generate_daily_report()
        print(report)

    def print_status(self):
        """Print current status."""
        print(self.reporter.print_live_status())


# ─── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hyperliquid Auto-Research Trading Bot"
    )
    parser.add_argument("--once", action="store_true",
                        help="Run a single research cycle then exit")
    parser.add_argument("--report", action="store_true",
                        help="Generate a report and exit")
    parser.add_argument("--status", action="store_true",
                        help="Print current status and exit")
    args = parser.parse_args()

    logger = setup_logging()
    bot = HyperliquidResearchBot()

    if args.status:
        bot.print_status()
    elif args.report:
        bot.generate_report()
    elif args.once:
        bot.run_once()
    else:
        bot.run_loop()


if __name__ == "__main__":
    main()
