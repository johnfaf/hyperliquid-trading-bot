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
from src.database import init_db
from src.trader_discovery import TraderDiscovery
from src.strategy_identifier import StrategyIdentifier
from src.strategy_scorer import StrategyScorer
from src.paper_trader import PaperTrader
from src.reporter import Reporter

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
    5. Reporting (generate insights)
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = False
        self._last_research = 0
        self._last_scoring = 0
        self._last_report = 0
        self._cycle_count = 0

        # Initialize components
        self.logger.info("Initializing bot components...")
        init_db()
        self.discovery = TraderDiscovery()
        self.identifier = StrategyIdentifier()
        self.scorer = StrategyScorer()
        self.paper_trader = PaperTrader()
        self.reporter = Reporter()
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

            # Phase 4: Paper trade top strategies
            self.logger.info("Phase 4: Paper Trading")
            top_strategies = self.scorer.get_top_strategies(n=5)

            # Check existing positions first
            closed = self.paper_trader.check_open_positions()
            if closed:
                self.logger.info(f"  Closed {len(closed)} positions")

            # Execute new signals
            if top_strategies:
                executed = self.paper_trader.execute_strategy_signals(top_strategies)
                self.logger.info(f"  Executed {len(executed)} new paper trades")

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

            self.logger.info(f"Cycle #{self._cycle_count} complete.")

        except Exception as e:
            self.logger.error(f"Error in cycle #{self._cycle_count}: {e}", exc_info=True)

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
        self.logger.info(f"  Main loop interval: {config.MAIN_LOOP_INTERVAL}s")
        self.logger.info(f"  Research interval: {config.RESEARCH_CYCLE_INTERVAL}s")
        self.logger.info(f"  Scoring interval: {config.SCORING_INTERVAL}s")

        while self.running:
            now = time.time()

            try:
                # Always check paper positions
                closed = self.paper_trader.check_open_positions()
                if closed:
                    self.logger.info(f"Closed {len(closed)} paper positions")

                # Run full research cycle periodically
                if now - self._last_research >= config.RESEARCH_CYCLE_INTERVAL:
                    self.run_once()
                    self._last_research = now

                # Run scoring more frequently than full research
                elif now - self._last_scoring >= config.SCORING_INTERVAL:
                    self.logger.info("Running scoring cycle...")
                    self.scorer.score_all_strategies()
                    top = self.scorer.get_top_strategies(n=5)
                    self.paper_trader.execute_strategy_signals(top)
                    self._last_scoring = now

                # Generate daily report
                if now - self._last_report >= 86400:
                    self.logger.info("Generating daily report...")
                    self.reporter.generate_daily_report()
                    self._last_report = now

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)

            # Status heartbeat
            if self._cycle_count % 12 == 0:
                print(self.reporter.print_live_status())

            # Sleep until next check
            if self.running:
                time.sleep(config.MAIN_LOOP_INTERVAL)

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
