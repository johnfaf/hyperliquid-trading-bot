"""
Research / Discovery Cycle (Tier 3)
====================================
Heavy discovery cycle: scan leaderboard, bot-detect, identify strategies.
Runs once per day to refresh the trader/strategy pool.
API-intensive (~3000+ calls) — not needed every trading cycle.

Extracted from ``HyperliquidResearchBot._run_discovery``.
"""
import logging
import time

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_discovery(container) -> None:
    """
    Execute a full discovery cycle using the subsystems in *container*.

    Parameters
    ----------
    container : SubsystemContainer
        Holds all subsystem instances (.discovery, .identifier, etc.)
    """
    logger.info("=" * 60)
    logger.info("DISCOVERY CYCLE — refreshing trader pool")
    logger.info("=" * 60)

    try:
        # Phase 0: Purge non-golden wallets from previous scans
        try:
            from src.discovery.golden_wallet import purge_non_golden_wallets
            purged = purge_non_golden_wallets()
            if purged:
                logger.info("Purged %d non-golden wallets before new scan", purged)
        except Exception as exc:
            logger.debug("Golden wallet purge skipped: %s", exc)

        # Phase 1: Discover and analyze traders
        logger.info("Phase 1: Trader Discovery")
        if container.discovery is None:
            logger.warning("  Discovery subsystem not available — skipping")
            return
        discovery_result = container.discovery.run_discovery_cycle()
        logger.info("  Discovered: %d traders", discovery_result.get("traders_discovered", 0))
        logger.info("  Analyzed: %d traders", discovery_result.get("traders_analyzed", 0))

        # Phase 2: Identify strategies from analyzed traders
        logger.info("Phase 2: Strategy Identification")
        if container.identifier:
            from src.data.database import get_active_traders
            from src.data import hyperliquid_client as hl
            traders = get_active_traders()
            all_strategies = []

            for trader in traders:
                state = hl.get_user_state(trader["address"])
                if state:
                    profile = {
                        "address": trader["address"],
                        "positions": state["positions"],
                        "position_analysis": container.discovery._analyze_positions(state["positions"]),
                        "trade_analysis": {
                            "total_trades": trader["trade_count"],
                            "win_rate": trader["win_rate"],
                            "total_closed_pnl": trader["total_pnl"],
                            "trading_frequency": "unknown",
                            "profit_factor": 1.5,
                            "coins_traded": [
                                p["coin"] for p in state["positions"] if p["size"] > 0
                            ],
                        },
                    }
                    strategies = container.identifier.identify_strategies(profile)
                    all_strategies.extend(strategies)
                time.sleep(0.3)

            if all_strategies:
                saved_ids = container.identifier.save_identified_strategies(all_strategies)
                logger.info("  Identified and saved %d strategies", len(saved_ids))

        # Phase 2b: Golden wallet scan
        try:
            from src.discovery.golden_wallet import run_golden_scan
            golden_summary = run_golden_scan(max_wallets=200)
            golden_count = int(
                golden_summary.get("golden_count", golden_summary.get("golden", 0)) or 0
            )
            total_evaluated = int(
                golden_summary.get("total_evaluated", golden_summary.get("scanned", 0)) or 0
            )
            logger.info(
                "  Golden scan: %d golden wallets out of %d evaluated",
                golden_count,
                total_evaluated,
            )
        except Exception as exc:
            logger.warning("Golden wallet scan failed: %s", exc)

        logger.info("Discovery cycle complete.")
    except Exception as exc:
        logger.error("Discovery cycle error: %s", exc, exc_info=True)
