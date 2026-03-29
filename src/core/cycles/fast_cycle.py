"""
Fast Cycle (Tier 1)
===================
Quick position check + copy-trade scan.
Runs every 60s between trading cycles.

Extracted from ``HyperliquidResearchBot._fast_cycle``.
"""
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_fast_cycle(container, cycle_count: int) -> None:
    """Check SL/TP on open positions and scan for copy-trade signals."""
    try:
        if container.paper_trader:
            closed = container.paper_trader.check_open_positions()
            if closed:
                logger.info("[fast] Closed %d positions (SL/TP)", len(closed))

        if container.copy_trader:
            copy_signals = container.copy_trader.scan_top_traders(top_n=10)
            if copy_signals:
                copy_executed = container.copy_trader.execute_copy_signals(copy_signals)
                if copy_executed:
                    logger.info("[fast] Copy-traded %d positions", len(copy_executed))

    except Exception as exc:
        logger.error("Error in fast cycle: %s", exc)
