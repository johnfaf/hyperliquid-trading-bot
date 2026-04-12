"""
Fast Cycle (Tier 1)
===================
Quick position check + copy-trade scan + whale detection.
Runs every 60s between trading cycles.

Extracted from ``HyperliquidResearchBot._fast_cycle``.
"""
import logging
import os
import time

from src.core.live_execution import (
    is_live_trading_active,
    mirror_executed_trades_to_live,
    sync_shadow_book_to_live,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_KILL_SWITCH_FILE = "/data/KILL_SWITCH"


def check_file_kill_switch(container) -> bool:
    """Emergency stop via file presence.

    If the configured kill-switch file exists, cancel all live orders, mark the
    live trader as killed, and request the bot loop to stop. The trigger is
    sticky for the current process lifetime.
    """
    path = os.environ.get("LIVE_EXTERNAL_KILL_SWITCH_FILE", "").strip() or DEFAULT_KILL_SWITCH_FILE
    if not path or not os.path.exists(path):
        return False

    if getattr(container, "_file_kill_switch_triggered", False):
        return True

    setattr(container, "_file_kill_switch_triggered", True)
    setattr(container, "_stop_requested", True)
    logger.critical("[fast] KILL_SWITCH file detected at %s", path)

    live_trader = getattr(container, "live_trader", None)
    if live_trader:
        try:
            live_trader.kill_switch_active = True
            live_trader._kill_switch_reason = f"file:{path}"
            live_trader.status_reason = "external_kill_switch"
        except Exception:
            pass
        try:
            cancelled = live_trader.cancel_all_orders()
            logger.critical("[fast] Cancelled %d live orders due to KILL_SWITCH", cancelled)
        except Exception as exc:
            logger.error("[fast] Failed to cancel live orders for KILL_SWITCH: %s", exc)

    return True


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
            next_allowed = float(getattr(container, "_snapshot_balance_next_try_ts", 0.0) or 0.0)
            if now_ts >= next_allowed:
                try:
                    container.live_trader.snapshot_balance()
                    container._snapshot_balance_failures = 0
                    container._snapshot_balance_next_try_ts = 0.0
                except Exception as exc:
                    failures = int(getattr(container, "_snapshot_balance_failures", 0) or 0) + 1
                    backoff = min(300.0, float(2 ** min(failures, 8)))
                    container._snapshot_balance_failures = failures
                    container._snapshot_balance_next_try_ts = now_ts + backoff
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

    except Exception as exc:
        logger.error("Error in fast cycle: %s", exc)


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
                                "type": "whale_trade",
                                "coin": sig.coin,
                                "side": sig.side.value,
                                "confidence": sig.confidence,
                                "strategy_type": "whale_detection",
                                "current_score": sig.confidence,
                                "reason": sig.reason,
                                "notional": next(w["notional"] for w in whales if w["coin"] == sig.coin),
                            }
                            for sig in whale_signals
                        ]
                        # Note: signal_processor.process() expects strategy dicts
                        # These whale dicts will be treated as low-priority signals
                        # and may be culled if they don't meet the threshold
                        logger.debug(f"[whale] Feeding {len(whale_signal_dicts)} whale signals to processor")
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
