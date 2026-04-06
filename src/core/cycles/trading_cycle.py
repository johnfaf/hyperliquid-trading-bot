"""
Trading Cycle (Tier 2)
======================
Lightweight trading cycle: score strategies, detect regime, trade.
Uses existing DB data from last discovery — no leaderboard scanning.
Runs every ~5 minutes to react to market changes quickly.

Extracted from ``HyperliquidResearchBot._run_trading_cycle``.
"""
import logging
from datetime import datetime

import config
from src.data import database as db
from src.core.live_execution import (
    get_execution_open_positions,
    is_live_trading_active,
    mirror_executed_trades_to_live,
    sync_shadow_book_to_live,
)
from src.signals.signal_schema import build_source_key

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_forecaster_signals(container, regime_data):
    """Feed options flow + polymarket into the predictive forecaster."""
    forecaster = container.predictive_forecaster
    if not forecaster:
        return

    # Options flow
    try:
        convictions = getattr(container.options_scanner, "top_convictions", None)
        if convictions:
            forecaster.update_options_flow(convictions)
            logger.debug("  Forecaster ← %d options convictions", len(convictions))
    except Exception as exc:
        logger.debug("  Forecaster options injection error: %s", exc)

    # Polymarket
    try:
        if container.polymarket:
            pm_sentiment = container.polymarket.get_market_sentiment()
            forecaster.update_polymarket_sentiment(pm_sentiment)
            logger.debug("  Forecaster ← Polymarket sentiment: %s", pm_sentiment.get("sentiment", "?"))
    except Exception as exc:
        logger.debug("  Forecaster polymarket injection error: %s", exc)


def _run_hedger(container, regime_data):
    """Cross-venue hedging — auto-hedge on crash regime."""
    hedger = container.cross_venue_hedger
    if not hedger or not regime_data:
        return
    try:
        pred_regime = {}
        if container.predictive_forecaster:
            pred_regime = container.predictive_forecaster.predict_regime("BTC")
        else:
            pred_regime = {
                "regime": regime_data.get("overall_regime", "neutral"),
                "confidence": regime_data.get("overall_confidence", 0),
            }
        open_trades = get_execution_open_positions(container)
        hedge_result = hedger.check_and_hedge(pred_regime, open_trades)
        if hedge_result.get("action") != "idle":
            logger.info(
                "  Hedger: %s | placed=%d, closed=%d, coins=%s",
                hedge_result["action"], hedge_result["hedges_placed"],
                hedge_result["hedges_closed"], hedge_result["coins_affected"],
            )
    except Exception as exc:
        logger.debug("  Cross-venue hedger error: %s", exc)


def _record_shadow_trade(container, closed_trade, pnl, return_pct, entry):
    """Record a closed trade in the shadow tracker."""
    tracker = container.shadow_tracker
    if not tracker:
        return
    try:
        meta = closed_trade.get("metadata") or {}
        stype = closed_trade.get("strategy_type", "unknown")
        source = meta.get("source_key") or meta.get("source", f"strategy:{stype}")
        tracker.record_trade({
            "signal_source": source,
            "coin": closed_trade.get("coin", "UNK"),
            "side": closed_trade.get("side", "long"),
            "entry_price": entry,
            "exit_price": closed_trade.get("exit_price", entry),
            "size": closed_trade.get("size", 0),
            "pnl": pnl,
            "pnl_pct": return_pct * 100,
            "entry_ts": closed_trade.get("entry_ts", ""),
            "exit_ts": closed_trade.get("exit_ts", ""),
            "regime_at_entry": closed_trade.get("regime", ""),
            "confidence": float(meta.get("confidence", 0.5)),
        })
    except Exception as exc:
        logger.debug("  ShadowTracker record error: %s", exc)


def _coerce_metadata(metadata):
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str):
        try:
            import json

            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _strategy_source_key(strategy: dict) -> str:
    metadata = _coerce_metadata(strategy.get("metadata", {}))
    source_key = str(strategy.get("source_key", "")).strip() or str(metadata.get("source_key", "")).strip()
    if source_key:
        return source_key

    params = strategy.get("parameters", {})
    if isinstance(params, dict):
        coins = params.get("coins", params.get("coins_traded", []))
    else:
        coins = []
    if isinstance(coins, str):
        coins = [coins]
    coin = strategy.get("_decision_coin") or (coins[0] if coins else "")
    return build_source_key(
        strategy.get("source", metadata.get("source", "strategy")),
        strategy_type=strategy.get("strategy_type", strategy.get("type", "")),
        trader_address=str(strategy.get("trader_address", metadata.get("source_trader", ""))),
        coin=str(coin),
        agent_id=str(strategy.get("agent_id", metadata.get("agent_id", ""))),
    )


def _apply_calibration_adjustments(container, top_strategies):
    """Adjust confidence using realized calibration data before ranking."""
    if not top_strategies or not getattr(container, "calibration", None):
        return
    try:
        for strategy in top_strategies:
            source_key = _strategy_source_key(strategy)
            original_conf = float(strategy.get("confidence", strategy.get("current_score", 0.5)) or 0.5)
            adjusted_conf = container.calibration.get_adjustment_factor(source_key, original_conf)
            strategy["confidence"] = round(adjusted_conf, 3)
            if "current_score" in strategy:
                strategy["current_score"] = round(
                    max(0.0, min((float(strategy.get("current_score", 0.5)) * 0.6) + (adjusted_conf * 0.4), 1.0)),
                    3,
                )
            strategy["source_key"] = source_key
    except Exception as exc:
        logger.debug("  Calibration weight apply error: %s", exc)


def _execute_signal_live(container, trade_signal, source_label: str):
    """Execute a TradeSignal directly on the live trader and log the outcome."""
    trader = getattr(container, "live_trader", None)
    if not trader or not is_live_trading_active(container):
        return None

    try:
        result = trader.execute_signal(trade_signal, bypass_firewall=True)
    except Exception as exc:
        logger.error(
            "  LIVE %s execution error for %s %s: %s",
            source_label,
            trade_signal.side.value.upper(),
            trade_signal.coin,
            exc,
        )
        return None

    if result and result.get("status") not in ("error", "rejected"):
        logger.info(
            "  LIVE %s executed: %s %s (%s)",
            source_label,
            trade_signal.side.value.upper(),
            trade_signal.coin,
            result.get("status", "ok"),
        )
        return result

    logger.error(
        "  LIVE %s failed: %s %s -> %s",
        source_label,
        trade_signal.side.value.upper(),
        trade_signal.coin,
        result,
    )
    return None


def _decision_route(strategy):
    """Return the execution route selected for a decision candidate."""
    return str(strategy.get("_decision_route", "paper_strategy") or "paper_strategy").strip()


def _build_lcrs_decision_candidates(container, lcrs_signals, regime_data):
    """Convert liquidation-reversal signals into decision-engine candidates."""
    if not lcrs_signals:
        return []

    from src.signals.signal_schema import RiskParams, SignalSide, SignalSource, TradeSignal

    candidates = []
    open_positions = get_execution_open_positions(container)
    for sig in lcrs_signals:
        try:
            trade_signal = TradeSignal(
                coin=sig["coin"],
                side=SignalSide(sig["side"]),
                confidence=float(sig.get("confidence", 0.0) or 0.0),
                source=SignalSource.STRATEGY,
                reason=f"LCRS: {sig.get('features', {}).get('setup_type', 'unknown')}",
                strategy_type="liquidation_reversal",
                entry_price=float(sig.get("price", 0) or 0),
                leverage=float(sig.get("leverage", 1) or 1),
                position_pct=float(sig.get("position_pct", 0.06) or 0.06),
                risk=RiskParams(stop_loss_pct=0.025, take_profit_pct=0.05),
                regime=regime_data.get("overall_regime", "") if regime_data else "",
            )

            if container.firewall:
                passed, reason = container.firewall.validate(
                    trade_signal,
                    regime_data=regime_data,
                    open_positions=open_positions,
                )
                if not passed:
                    logger.info("  LCRS firewall rejected %s before decision engine: %s", sig["coin"], reason)
                    continue

            source_key = build_source_key(
                "liquidation_strategy",
                strategy_type="liquidation_reversal",
                coin=sig["coin"],
            )
            candidates.append(
                {
                    "id": None,
                    "name": f"lcrs_{sig['coin']}_{sig['side']}",
                    "strategy_type": "liquidation_reversal",
                    "trader_address": "liquidation_strategy",
                    "current_score": float(sig.get("confidence", 0.0) or 0.0),
                    "confidence": float(sig.get("confidence", 0.0) or 0.0),
                    "direction": sig["side"],
                    "side": sig["side"],
                    "source": "liquidation_strategy",
                    "source_key": source_key,
                    "parameters": {"coins": [sig["coin"]]},
                    "metrics": {},
                    "metadata": {
                        "source": "liquidation_strategy",
                        "source_key": source_key,
                        "reason": trade_signal.reason,
                        "setup_type": sig.get("features", {}).get("setup_type", ""),
                        "features": sig.get("features", {}),
                    },
                    "_decision_route": "lcrs",
                    "_lcrs_signal": dict(sig),
                }
            )
        except Exception as exc:
            logger.debug("  LCRS candidate build error %s: %s", sig.get("coin"), exc)
    return candidates


def _gather_copy_trade_signals(container):
    """Gather copy-trade signals without executing them yet."""
    if not container.copy_trader:
        return []

    signals = []
    if container.position_monitor:
        ws_signals = container.position_monitor.drain_signals()
        if ws_signals:
            logger.info("  WebSocket: %d real-time copy signals", len(ws_signals))
            signals.extend(ws_signals)

    try:
        rest_signals = container.copy_trader.scan_top_traders(top_n=10)
        if rest_signals:
            signals.extend(rest_signals)
    except Exception as exc:
        logger.debug("  Copy-trader scan skipped: %s", exc)

    try:
        from src.discovery.golden_bridge import get_golden_copy_signals, auto_connect_golden_wallets

        auto_connect_golden_wallets()
        golden_signals = get_golden_copy_signals()
        if golden_signals:
            logger.info("  Golden bridge: %d signals", len(golden_signals))
            signals = golden_signals + signals
    except Exception as exc:
        logger.debug("  Golden bridge skipped: %s", exc)

    return signals


def _process_copy_trade_closures(container, copy_signals, regime_data):
    """Process copy-trader exit signals before ranking new entries."""
    if not container.copy_trader or not copy_signals:
        return

    close_signals = [dict(signal) for signal in copy_signals if signal.get("type") == "copy_close"]
    if not close_signals:
        return

    container.copy_trader.execute_copy_signals(close_signals, regime_data=regime_data)
    logger.info("  Processed %d copy-trade exits before decision engine", len(close_signals))


def _build_copy_decision_candidates(container, copy_signals, regime_data):
    """Convert copy-entry signals into decision-engine candidates."""
    if not container.copy_trader or not copy_signals:
        return []

    from src.signals.signal_schema import signal_from_copy_trade

    open_positions = get_execution_open_positions(container)
    candidates = []
    for raw_signal in copy_signals:
        if raw_signal.get("type") not in ("copy_open", "copy_scale_in", "copy_flip", "golden_copy"):
            continue

        try:
            signal = dict(raw_signal)
            signal = container.copy_trader._apply_regime_weight(signal, signal["coin"])

            trade_signal = signal_from_copy_trade(
                trader_address=signal.get("source_trader", ""),
                coin=signal["coin"],
                side=signal["side"],
                entry_price=float(signal.get("price", 0) or 0),
                confidence=float(signal.get("confidence", 0.5) or 0.5),
            )
            trade_signal.leverage = float(signal.get("leverage", 2) or 2)
            trade_signal.regime = regime_data.get("overall_regime", "") if regime_data else ""

            if container.firewall:
                passed, reason = container.firewall.validate(
                    trade_signal,
                    regime_data=regime_data,
                    open_positions=open_positions,
                    ignore_position_limit=True,
                    dry_run=True,
                )
                if not passed:
                    logger.info(
                        "  Firewall rejected copy %s %s before decision engine: %s",
                        signal["side"],
                        signal["coin"],
                        reason,
                    )
                    continue

            source_trader = str(signal.get("source_trader", "unknown")).strip() or "unknown"
            source_key = f"copy_trade:{source_trader}"
            signal["confidence"] = trade_signal.confidence
            signal["source_accuracy"] = getattr(trade_signal, "source_accuracy", 0.0)
            signal["regime"] = trade_signal.regime

            candidates.append(
                {
                    "id": None,
                    "name": f"copy_{source_trader}_{signal['coin']}_{signal['side']}",
                    "strategy_type": "copy_trade",
                    "trader_address": source_trader,
                    "current_score": float(signal.get("confidence", 0.5) or 0.5),
                    "confidence": float(signal.get("confidence", 0.5) or 0.5),
                    "direction": signal["side"],
                    "side": signal["side"],
                    "source": "copy_trade",
                    "source_key": source_key,
                    "parameters": {"coins": [signal["coin"]]},
                    "metrics": {},
                    "metadata": {
                        "source": "copy_trade",
                        "source_key": source_key,
                        "source_trader": source_trader,
                        "type": signal.get("type", ""),
                        "source_pnl": signal.get("source_pnl", 0),
                        "reason": f"Copy {source_trader}: {signal['side']} {signal['coin']}",
                    },
                    "_decision_route": "copy_trade",
                    "_copy_signal": signal,
                }
            )
        except Exception as exc:
            logger.debug("  Copy candidate build error %s: %s", raw_signal.get("coin"), exc)

    return candidates


def _execute_selected_decisions(container, selected_strategies, regime_data):
    """Execute only the candidates selected by the main decision engine."""
    from src.notifications import telegram_bot as tg

    if not selected_strategies:
        return

    standard = []
    lcrs = []
    copy_entries = []
    for strategy in selected_strategies:
        route = _decision_route(strategy)
        if route == "lcrs":
            lcrs.append(strategy)
        elif route == "copy_trade":
            copy_entries.append(strategy)
        else:
            standard.append(strategy)

    if standard and container.paper_trader:
        executed = container.paper_trader.execute_strategy_signals(
            standard,
            exchange_agg=container.exchange_agg,
            options_scanner=container.options_scanner,
            regime_data=regime_data,
            arena=container.arena,
        )
        logger.info("  Executed %d decision-engine paper trades", len(executed))
        mirror_executed_trades_to_live(
            container,
            executed,
            success_label="  LIVE",
            skip_label="  Live trader requested but not deployable; skipping strategy mirroring",
        )
        if tg.is_configured():
            for trade in executed:
                tg.notify_trade_opened(trade, source="strategy")

    if lcrs:
        selected_lcrs = [dict(strategy["_lcrs_signal"]) for strategy in lcrs if strategy.get("_lcrs_signal")]
        if selected_lcrs:
            _execute_lcrs_signals(container, selected_lcrs, regime_data)

    if copy_entries and container.copy_trader:
        selected_copy_signals = [dict(strategy["_copy_signal"]) for strategy in copy_entries if strategy.get("_copy_signal")]
        if selected_copy_signals:
            copy_executed = container.copy_trader.execute_copy_signals(selected_copy_signals, regime_data=regime_data)
            logger.info("  Executed %d decision-engine copy trades", len(copy_executed))
            if tg.is_configured():
                for trade in copy_executed:
                    tg.notify_trade_opened(trade, source="copy")
            mirror_executed_trades_to_live(
                container,
                copy_executed,
                success_label="  LIVE COPY",
                skip_label="  Live trader requested but not deployable; skipping copy mirroring",
            )


# ---------------------------------------------------------------------------
# Main trading cycle
# ---------------------------------------------------------------------------

def run_trading_cycle(container, cycle_count: int) -> None:
    """
    Execute one trading cycle using the subsystems in *container*.
    """
    from src.notifications import telegram_bot as tg

    # Kill switch check
    if container.live_trader and not container.live_trader.dry_run:
        container.live_trader.update_daily_pnl_from_fills()
        if container.live_trader.check_daily_loss():
            logger.error("KILL SWITCH ACTIVE — daily loss limit hit, skipping live trades")

        # Sweep for orphaned positions (opened successfully but SL/TP
        # placement was skipped due to an upstream error like the
        # get_positions float(dict) crash).  Safe to call every cycle —
        # protect_orphaned_positions checks for existing reduce-only
        # orders and no-ops for protected positions.
        try:
            container.live_trader.protect_orphaned_positions()
        except Exception as exc:
            logger.warning("Orphan protection sweep failed: %s", exc)

    logger.info("=" * 60)
    logger.info("Starting trading cycle #%d", cycle_count)
    logger.info("=" * 60)

    try:
        # ── Phase 3: Score all strategies ──
        logger.info("Phase 3: Strategy Scoring")
        score_results = container.scorer.score_all_strategies() if container.scorer else []
        logger.info("  Scored %d strategies", len(score_results))

        # ── Phase 3b: Multi-Exchange Volume Analysis ──
        logger.info("Phase 3b: Multi-Exchange Volume Analysis")
        market_overview = {}
        try:
            if container.exchange_agg:
                market_overview = container.exchange_agg.get_market_overview()
                logger.info(
                    "  Market bias: %s (score: %+.4f)",
                    market_overview.get("overall_bias", "?"),
                    market_overview.get("overall_bias_score", 0),
                )
                if tg.is_configured():
                    tg.notify_market_bias(market_overview)
        except Exception as exc:
            logger.warning("  Exchange aggregator error: %s", exc)

        # ── Phase 3c: Options Flow Scan ──
        logger.info("Phase 3c: Options Flow Scan")
        try:
            if container.options_scanner:
                flow_result = container.options_scanner.scan_flow()
                logger.info(
                    "  Unusual prints: %d  Top convictions: %d",
                    flow_result.get("unusual_prints", 0),
                    flow_result.get("top_convictions", 0),
                )
                if tg.is_configured() and container.options_scanner.top_convictions:
                    for conv in container.options_scanner.top_convictions[:3]:
                        if conv.get("conviction_pct", 0) > 60:
                            tg.notify_strong_signal(
                                coin=conv["ticker"],
                                side=conv["direction"],
                                reasons=[
                                    f"Options flow: {conv.get('total_prints', 0)} unusual prints",
                                    f"Net flow: ${conv.get('net_flow', 0):,.0f}",
                                    f"Conviction: {conv.get('conviction_pct', 0):.0f}%",
                                ],
                                confidence=conv.get("conviction_pct", 0) / 100.0,
                            )
        except Exception as exc:
            logger.warning("  Options flow scan error: %s", exc)

        # ── Phase 3d: Regime Detection ──
        logger.info("Phase 3d: Market Regime Detection")
        regime_data = {}
        try:
            if container.regime_detector:
                regime_data = container.regime_detector.get_market_regime()
                container._last_regime_data = regime_data  # Expose for health reporter
                logger.info(
                    "  Regime: %s (confidence=%s)",
                    regime_data.get("overall_regime", "?"),
                    f"{regime_data.get('overall_confidence', 0):.0%}",
                )
        except Exception as exc:
            logger.warning("  Regime detection error: %s", exc)

        # ── Phase 3d2: Polymarket Scan ──
        polymarket_signals = []
        if container.polymarket:
            logger.info("Phase 3d2: Polymarket Scan")
            try:
                polymarket_signals = container.polymarket.generate_signals(hl_regime=regime_data)
                sentiment = container.polymarket.get_market_sentiment()
                logger.info(
                    "  Polymarket: %d signals, sentiment=%s (conf=%s, markets=%d)",
                    len(polymarket_signals),
                    sentiment.get("sentiment", "?"),
                    f"{sentiment.get('confidence', 0):.0%}",
                    sentiment.get("markets_analyzed", 0),
                )
            except Exception as exc:
                logger.warning("  Polymarket scan error: %s", exc)

        # Inject into forecaster
        _inject_forecaster_signals(container, regime_data)

        # Cross-venue hedging
        _run_hedger(container, regime_data)

        # ── Phase 3e: Multi-Exchange Scan ──
        cross_venue_data, funding_arbs = _run_multi_exchange_scan(container)

        # ── Phase 3f: Liquidation Strategy ──
        lcrs_signals = _run_liquidation_scan(container, regime_data)

        # ── Phase 3f2: Copy-trade signal harvest ──
        copy_signals = _gather_copy_trade_signals(container)

        # ── Phase 4: Paper Trading (regime-aware) ──
        logger.info("Phase 4: Paper Trading (regime-aware)")
        top_strategies = container.scorer.get_top_strategies() if container.scorer else []

        # Regime-aware filtering
        if regime_data and container.regime_strategy_filter:
            try:
                top_strategies = container.regime_strategy_filter.filter(top_strategies, regime_data)
            except Exception:
                if container.regime_detector:
                    top_strategies = container.regime_detector.filter_strategies_by_regime(
                        top_strategies, regime_data
                    )
            logger.info("  Post-regime filter: %d strategies active", len(top_strategies))

        # Signal processing
        if container.signal_processor:
            top_strategies = container.signal_processor.process(top_strategies, regime_data=regime_data)
            logger.info("  Post-signal-processor: %d strategies", len(top_strategies))

        # Inject Polymarket signals as synthetic strategies
        if polymarket_signals:
            injected_polymarket = 0
            for pm in polymarket_signals:
                confidence = float(pm.get("confidence", 0.5) or 0.5)
                if confidence < config.POLYMARKET_MIN_DECISION_CONFIDENCE:
                    continue
                source_key = build_source_key(
                    "polymarket",
                    strategy_type="event_driven",
                    coin=pm.get("coin", "BTC"),
                )
                synthetic = {
                    "id": None,
                    "name": f"polymarket_{pm.get('coin', 'UNK')}_{pm.get('side', '?')}",
                    "strategy_type": "event_driven",
                    "trader_address": "polymarket",
                    "current_score": confidence,
                    "confidence": confidence,
                    "direction": pm.get("side", "long"),
                    "side": pm.get("side", "long"),
                    "source": "polymarket",
                    "source_key": source_key,
                    "parameters": {
                        "coins": [pm.get("coin", "BTC")],
                        "market": pm.get("polymarket_market", ""),
                        "probability": pm.get("polymarket_probability", 0),
                    },
                    "metrics": {},
                    "metadata": {
                        "source": "polymarket",
                        "source_key": source_key,
                        "polymarket_volume_24h": pm.get("polymarket_volume_24h", 0),
                        "reason": pm.get("reason", ""),
                    },
                }
                top_strategies.append(synthetic)
                injected_polymarket += 1
            logger.info("  Injected %d Polymarket signals", injected_polymarket)

        # Inject high-conviction options flow as synthetic strategies
        if container.options_scanner:
            convictions = getattr(container.options_scanner, "top_convictions", None) or []
            injected_options = 0
            for conv in convictions:
                if conv.get("conviction_pct", 0) < config.OPTIONS_FLOW_INJECTION_MIN_CONVICTION:
                    continue
                direction = conv.get("direction", "bullish")
                side = "long" if direction == "bullish" else "short"
                source_key = build_source_key(
                    "options_flow",
                    strategy_type="options_momentum",
                    coin=conv.get("ticker", "BTC"),
                )
                synthetic = {
                    "id": None,
                    "name": f"options_flow_{conv.get('ticker', 'UNK')}_{side}",
                    "strategy_type": "options_momentum",
                    "trader_address": "options_flow",
                    "current_score": conv.get("conviction_pct", 70) / 100.0,
                    "confidence": conv.get("conviction_pct", 70) / 100.0,
                    "direction": side,
                    "side": side,
                    "source": "options_flow",
                    "source_key": source_key,
                    "parameters": {
                        "coins": [conv.get("ticker", "BTC")],
                    },
                    "metrics": {},
                    "metadata": {
                        "source": "options_flow",
                        "source_key": source_key,
                        "net_flow": conv.get("net_flow", 0),
                        "total_prints": conv.get("total_prints", 0),
                        "conviction_pct": conv.get("conviction_pct", 0),
                    },
                }
                top_strategies.append(synthetic)
                injected_options += 1
            if injected_options:
                logger.info("  Injected %d options flow signals into decision engine", injected_options)

        # Keep the shadow ledger synced to exchange truth when live mode is active.
        if is_live_trading_active(container):
            closed = sync_shadow_book_to_live(container)
        else:
            closed = container.paper_trader.check_open_positions() if container.paper_trader else []
        if closed:
            logger.info("  Closed %d positions", len(closed))

        closed = _collect_closed_trade_events(container, closed)
        if closed and tg.is_configured():
            for c_trade in closed:
                tg.notify_trade_closed(
                    c_trade, c_trade.get("exit_price", 0),
                    c_trade.get("pnl", 0), c_trade.get("reason", ""),
                )
        if closed:
            _process_closed_trades(container, closed)

        # Copy exits are risk-management events; process them before ranking new entries.
        _process_copy_trade_closures(container, copy_signals, regime_data)

        lcrs_candidates = _build_lcrs_decision_candidates(container, lcrs_signals, regime_data)
        if lcrs_candidates:
            top_strategies.extend(lcrs_candidates)
            logger.info("  Injected %d LCRS signals into decision engine", len(lcrs_candidates))

        arena_candidates = _run_alpha_arena(container, regime_data)
        if arena_candidates:
            top_strategies.extend(arena_candidates)
            logger.info("  Injected %d arena champion signals into decision engine", len(arena_candidates))

        copy_candidates = _build_copy_decision_candidates(container, copy_signals, regime_data)
        if copy_candidates:
            top_strategies.extend(copy_candidates)
            logger.info("  Injected %d copy-trade signals into decision engine", len(copy_candidates))

        # Cross-venue signal confirmation
        _run_cross_venue_confirmation(container, top_strategies)

        # Source-quality and calibration adjustments must happen BEFORE ranking
        # so the decision engine sees which systems are improving over time.
        _apply_agent_scorer_weights(container, top_strategies)
        _apply_calibration_adjustments(container, top_strategies)

        # Decision engine
        open_trades = get_execution_open_positions(container)
        kelly_stats = None
        if container.kelly_sizer:
            try:
                kelly_stats = container.kelly_sizer.get_all_sizing_stats()
            except Exception:
                pass
        selected_strategies = top_strategies
        if container.decision_engine:
            selected_strategies = container.decision_engine.decide(
                top_strategies, regime_data=regime_data,
                open_positions=open_trades, kelly_stats=kelly_stats,
            )

        # Execute only what the main decision engine selected.
        _execute_selected_decisions(container, selected_strategies, regime_data)

        closed = _collect_closed_trade_events(container, [])
        if closed and tg.is_configured():
            for c_trade in closed:
                tg.notify_trade_closed(
                    c_trade, c_trade.get("exit_price", 0),
                    c_trade.get("pnl", 0), c_trade.get("reason", ""),
                )

        # Phase 4c: Feed closed trade outcomes
        if closed:
            _process_closed_trades(container, closed)

        logger.info("Trading cycle #%d complete.", cycle_count)

    except Exception as exc:
        logger.error("Error in cycle #%d: %s", cycle_count, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Sub-phases (keep the main function readable)
# ---------------------------------------------------------------------------

def _run_multi_exchange_scan(container):
    """Phase 3e: multi-exchange scan."""
    cross_venue_data = {}
    funding_arbs = []
    logger.info("Phase 3e: Multi-Exchange Scanner")
    try:
        if container.multi_scanner:
            venue_health = container.multi_scanner.check_health()
            logger.info("  Venue health: %s", venue_health)
            common_markets = container.multi_scanner.get_common_markets()
            if common_markets:
                logger.info("  Common markets: %s…", common_markets[:15])
            funding_arbs = container.multi_scanner.scan_funding_arb()
            if funding_arbs:
                for arb in funding_arbs[:3]:
                    logger.info(
                        "  Funding arb: %s long@%s(%+.4f%%) / short@%s(%+.4f%%) = %.1f%% ann.",
                        arb.coin, arb.long_venue, arb.long_funding_rate,
                        arb.short_venue, arb.short_funding_rate,
                        arb.funding_spread_annualized,
                    )
            cross_venue_data = {
                "health": venue_health,
                "common_markets": common_markets,
                "funding_arbs": funding_arbs,
            }
        else:
            logger.info("  Multi-exchange scanner not available")
    except Exception as exc:
        logger.warning("  Multi-exchange scanner error: %s", exc)
    return cross_venue_data, funding_arbs


def _run_liquidation_scan(container, regime_data):
    """Phase 3f: liquidation cascade reversal scan."""
    lcrs_signals = []
    if not container.liquidation_strategy:
        return lcrs_signals
    logger.info("Phase 3f: Liquidation Strategy Scan")
    try:
        from src.data import hyperliquid_client as hl_client
        import requests
        mids = hl_client.get_all_mids() or {}
        coins = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "ARB",
                 "OP", "SUI", "APT", "INJ", "SEI"]
        for coin in coins:
            price = float(mids.get(coin, 0))
            if price <= 0:
                continue
            try:
                lcrs_features = {}
                if regime_data and "per_coin" in regime_data:
                    coin_regime = regime_data["per_coin"].get(coin, {})
                    lcrs_features["trend_strength"] = coin_regime.get("trend_strength", 0.5)
                    lcrs_features["volatility"] = coin_regime.get("atr_pct", 0.02)
                    lcrs_features["volume_ratio"] = coin_regime.get("volume_ratio", 1.0)

                # Funding rate
                try:
                    meta_resp = requests.post(
                        "https://api.hyperliquid.xyz/info",
                        json={"type": "metaAndAssetCtxs"}, timeout=10,
                    )
                    if meta_resp.status_code == 200:
                        meta_data = meta_resp.json()
                        if isinstance(meta_data, list) and len(meta_data) > 1:
                            for asset_ctx in meta_data[1]:
                                if isinstance(asset_ctx, dict) and asset_ctx.get("coin") == coin:
                                    lcrs_features["funding_rate"] = float(asset_ctx.get("funding", 0))
                                    lcrs_features["oi_change"] = float(asset_ctx.get("openInterest", 0)) * 0.01
                                    break
                except Exception:
                    pass

                # Feature engine enrichment
                if container.feature_engine:
                    try:
                        payload = {
                            "type": "candleSnapshot",
                            "req": {
                                "coin": coin, "interval": "1h",
                                "startTime": int((datetime.utcnow().timestamp() - 100 * 3600) * 1000),
                                "endTime": int(datetime.utcnow().timestamp() * 1000),
                            },
                        }
                        resp = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=10)
                        if resp.status_code == 200:
                            raw = resp.json()
                            if isinstance(raw, list) and len(raw) >= 20:
                                candles = [
                                    {"open": float(c.get("o", 0)), "high": float(c.get("h", 0)),
                                     "low": float(c.get("l", 0)), "close": float(c.get("c", 0)),
                                     "volume": float(c.get("v", 0))} for c in raw
                                ]
                                feat = container.feature_engine.compute(coin, candles)
                                lcrs_features.setdefault("rsi", feat.rsi)
                                lcrs_features.setdefault("momentum_score", feat.momentum_score)
                                lcrs_features.setdefault("trend_strength", feat.trend_strength)
                                lcrs_features.setdefault("volatility", feat.volatility)
                                lcrs_features.setdefault("volume_ratio", feat.volume_ratio)
                                lcrs_features.setdefault("overall_score", feat.overall_score)
                                lcrs_features.setdefault("bollinger_position", feat.bollinger_position)
                                if len(candles) >= 8:
                                    lcrs_features["price_change"] = (
                                        (candles[-1]["close"] - candles[-8]["close"]) / candles[-8]["close"]
                                    )
                    except Exception:
                        pass

                sig = container.liquidation_strategy.generate_signal(coin, lcrs_features, price)
                if sig:
                    lcrs_signals.append(sig)
                    logger.info(
                        "  LCRS: %s %s (conf=%s, type=%s)",
                        sig["side"].upper(), coin,
                        f"{sig['confidence']:.0%}",
                        sig["features"].get("setup_type", ""),
                    )
            except Exception as exc:
                logger.debug("  LCRS scan error %s: %s", coin, exc)

        if lcrs_signals:
            logger.info("  LCRS found %d setups", len(lcrs_signals))
        else:
            logger.info("  LCRS: no setups detected")
    except Exception as exc:
        logger.warning("  Liquidation strategy error: %s", exc)
    return lcrs_signals


def _run_cross_venue_confirmation(container, top_strategies):
    """Enrich strategies with cross-venue confirmation scores."""
    if not (container.multi_scanner and getattr(container.multi_scanner, "cross_venue", None) and top_strategies):
        return
    logger.info("Phase 4 cross-venue: Signal Confirmation")
    try:
        import json
        signals_to_confirm = []
        for s in top_strategies:
            params = s.get("parameters", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            coins = params.get("coins", params.get("coins_traded", []))
            if isinstance(coins, str):
                coins = [coins]
            coin = coins[0] if coins else ""
            direction = s.get("direction", "long")
            score = s.get("score", 0.5)
            if coin and coin != "unknown":
                signals_to_confirm.append({"coin": coin, "direction": direction, "score": score})

        if signals_to_confirm:
            confirmed = container.multi_scanner.confirm_signals(signals_to_confirm)
            confirm_map = {f"{c.coin}:{c.direction}": c.confirmation_score for c in confirmed}
            for s in top_strategies:
                params = s.get("parameters", {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except (json.JSONDecodeError, TypeError):
                        params = {}
                coins = params.get("coins", params.get("coins_traded", []))
                if isinstance(coins, str):
                    coins = [coins]
                coin = coins[0] if coins else ""
                direction = s.get("direction", "long")
                key = f"{coin}:{direction}"
                cv_score = confirm_map.get(key, 0.0)
                if "metadata" not in s:
                    s["metadata"] = {}
                s["metadata"]["cross_venue_score"] = cv_score
                if cv_score > 0.15:
                    original = s.get("current_score", s.get("score", 0.5))
                    s["current_score"] = min(1.0, original + cv_score * 0.15)

            boosted = sum(
                1 for s in top_strategies
                if s.get("metadata", {}).get("cross_venue_score", 0) > 0.15
            )
            logger.info("  Cross-venue: confirmed %d signals, %d boosted", len(signals_to_confirm), boosted)
    except Exception as exc:
        logger.warning("  Cross-venue confirmation error: %s", exc)


def _apply_agent_scorer_weights(container, top_strategies):
    """Apply AgentScorer dynamic weights to strategy confidences."""
    if not top_strategies or not container.agent_scorer:
        return
    try:
        for s in top_strategies:
            source_key = _strategy_source_key(s)
            weight = container.agent_scorer.get_weight(source_key)
            accuracy = container.agent_scorer.get_accuracy(source_key)
            orig_conf = float(s.get("confidence", 0.5))
            s["confidence"] = round(orig_conf * 0.7 + weight * 0.3, 3)
            s["agent_scorer_weight"] = round(weight, 3)
            s["source_accuracy"] = round(accuracy, 3)
            s["source_key"] = source_key
    except Exception as exc:
        logger.debug("  AgentScorer weight apply error: %s", exc)


def _execute_lcrs_signals(container, lcrs_signals, regime_data):
    """Phase 4a: execute liquidation reversal signals."""
    from src.notifications import telegram_bot as tg
    logger.info("Phase 4a: Liquidation Strategy Execution")
    try:
        from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams
        lcrs_executed = []
        open_trades = get_execution_open_positions(container)
        account = None

        for sig in lcrs_signals:
            try:
                trade_signal = TradeSignal(
                    coin=sig["coin"], side=SignalSide(sig["side"]),
                    confidence=sig["confidence"], source=SignalSource.STRATEGY,
                    reason=f"LCRS: {sig['features'].get('setup_type', 'unknown')}",
                    strategy_type="liquidation_reversal",
                    entry_price=sig["price"], leverage=sig["leverage"],
                    position_pct=sig.get("position_pct", 0.06),
                    risk=RiskParams(stop_loss_pct=0.025, take_profit_pct=0.05),
                    regime=regime_data.get("overall_regime", "") if regime_data else "",
                )

                if container.firewall:
                    passed, reason = container.firewall.validate(
                        trade_signal, regime_data=regime_data, open_positions=open_trades
                    )
                    if not passed:
                        logger.info("  LCRS firewall rejected %s: %s", sig["coin"], reason)
                        continue

                if container.kelly_sizer and account:
                    sizing = container.kelly_sizer.get_sizing(
                        "liquidation_reversal", account["balance"], trade_signal.confidence
                    )
                    trade_signal.position_pct = sizing.position_pct

                if container.trade_memory:
                    mem = container.trade_memory.find_similar(
                        sig.get("features", {}), coin=sig["coin"], side=sig["side"]
                    )
                    if mem.recommendation == "avoid":
                        logger.info("  LCRS memory blocked %s: %s", sig["coin"], mem.reason)
                        continue

                if container.llm_filter:
                    ctx = {"regime_data": regime_data, "open_positions": open_trades}
                    approved, adj_conf, reason = container.llm_filter.filter(sig, ctx)
                    if not approved:
                        logger.info("  LCRS LLM filter blocked %s: %s", sig["coin"], reason)
                        continue
                    trade_signal.confidence = adj_conf

                if is_live_trading_active(container):
                    live_result = _execute_signal_live(container, trade_signal, "LCRS")
                    if live_result:
                        lcrs_executed.append(live_result)
                        if tg.is_configured():
                            tg.notify_trade_opened(
                                {"coin": sig["coin"], "side": sig["side"], "entry_price": sig["price"]},
                                source="liquidation_strategy",
                            )
                    continue

                if account is None:
                    account = db.get_paper_account()
                if not account:
                    continue
                size_usd = account["balance"] * trade_signal.effective_size
                size = size_usd / sig["price"]
                trade_id = db.open_paper_trade(
                    strategy_id=None, coin=sig["coin"], side=sig["side"],
                    entry_price=sig["price"], size=size, leverage=sig["leverage"],
                    stop_loss=sig["stop_loss"], take_profit=sig["take_profit"],
                    metadata={
                        "source": "liquidation_strategy",
                        "strategy_type": "liquidation_reversal",
                        "confidence": trade_signal.confidence,
                        "setup_type": sig["features"].get("setup_type", ""),
                        "features": sig["features"],
                    },
                )
                lcrs_executed.append({"id": trade_id, "coin": sig["coin"], "side": sig["side"]})
                logger.info(
                    "  LCRS executed: %s %s @ $%s (conf=%s)",
                    sig["side"].upper(), sig["coin"],
                    f"{sig['price']:,.2f}", f"{trade_signal.confidence:.0%}",
                )
                if tg.is_configured():
                    tg.notify_trade_opened(
                        {"coin": sig["coin"], "side": sig["side"], "entry_price": sig["price"]},
                        source="liquidation_strategy",
                    )
            except Exception as exc:
                logger.debug("  LCRS execution error %s: %s", sig.get("coin"), exc)

        if lcrs_executed:
            logger.info("  Executed %d LCRS trades", len(lcrs_executed))
    except Exception as exc:
        logger.warning("  LCRS execution phase error: %s", exc)


def _execute_options_flow_trades(container, regime_data):
    """Phase 4a2: high-conviction options flow → direct trade."""
    from src.notifications import telegram_bot as tg
    logger.info("Phase 4a2: Options Flow Trades")
    try:
        from src.signals.signal_schema import signal_from_options_flow
        from src.data import hyperliquid_client as hl_client
        mids = hl_client.get_all_mids() or {}
        options_executed = []

        convictions = getattr(container.options_scanner, "top_convictions", None) or []
        for conv in convictions:
            if conv.get("conviction_pct", 0) < config.OPTIONS_FLOW_DIRECT_MIN_CONVICTION:
                continue
            flow_signal = signal_from_options_flow(
                ticker=conv["ticker"], direction=conv["direction"],
                net_flow=conv["net_flow"], prints=conv["total_prints"],
                conviction_pct=conv["conviction_pct"],
            )
            flow_signal.strategy_type = "options_momentum"
            flow_signal.position_pct = 0.04
            flow_signal.leverage = 2.0
            price = float(mids.get(conv["ticker"], 0))
            if price <= 0:
                continue
            flow_signal.entry_price = price
            source_key = build_source_key(
                flow_signal.source,
                strategy_type=flow_signal.strategy_type,
                coin=conv["ticker"],
            )
            flow_signal.source_key = source_key

            if container.agent_scorer:
                weight = container.agent_scorer.get_weight(source_key)
                flow_signal.confidence = min(flow_signal.confidence * 0.7 + weight * 0.3, 1.0)
                flow_signal.source_accuracy = container.agent_scorer.get_accuracy(source_key)

            if getattr(container, "calibration", None):
                flow_signal.confidence = container.calibration.get_adjustment_factor(
                    source_key,
                    flow_signal.confidence,
                )

            if container.firewall:
                passed, reason = container.firewall.validate(
                    flow_signal, regime_data=regime_data,
                    open_positions=get_execution_open_positions(container),
                )
                if not passed:
                    logger.info("  Firewall rejected options flow %s: %s", conv["ticker"], reason)
                    continue

            if getattr(container, "arena", None):
                try:
                    approved, consensus_conf = container.arena.get_consensus_on_signal(
                        flow_signal,
                        features={
                            "net_flow": conv.get("net_flow", 0),
                            "total_prints": conv.get("total_prints", 0),
                            "conviction_pct": conv.get("conviction_pct", 0),
                        },
                    )
                    if not approved:
                        logger.info("  Arena consensus rejected options flow %s", conv["ticker"])
                        continue
                    flow_signal.confidence = consensus_conf
                except Exception as exc:
                    logger.debug("  Arena consensus error for options flow %s: %s", conv["ticker"], exc)

            signal_id = ""
            if container.agent_scorer:
                signal_id = container.agent_scorer.record_signal(source_key, {
                    "coin": conv["ticker"], "side": flow_signal.side.value,
                    "confidence": flow_signal.confidence,
                })

            if is_live_trading_active(container):
                live_result = _execute_signal_live(container, flow_signal, "OPTIONS FLOW")
                if live_result:
                    live_result.update(
                        {
                            "source": "options_flow",
                            "source_key": source_key,
                            "signal_id": signal_id,
                            "strategy_type": "options_momentum",
                            "confidence": flow_signal.confidence,
                        }
                    )
                    options_executed.append(live_result)
                    if tg.is_configured():
                        tg.notify_trade_opened(
                            {"coin": conv["ticker"], "side": flow_signal.side.value, "entry_price": price},
                            source="options_flow",
                        )
                continue

            account = db.get_paper_account()
            if account:
                size_usd = account["balance"] * flow_signal.effective_size
                size = size_usd / price
                side = flow_signal.side.value
                sl = price * (1 - 0.05) if side == "long" else price * (1 + 0.05)
                tp = price * (1 + 0.10) if side == "long" else price * (1 - 0.10)
                trade_id = db.open_paper_trade(
                    strategy_id=None, coin=conv["ticker"], side=side,
                    entry_price=price, size=size, leverage=2,
                    stop_loss=sl, take_profit=tp,
                    metadata={
                        "source": "options_flow",
                        "source_key": source_key,
                        "signal_id": signal_id,
                        "strategy_type": "options_momentum",
                        "confidence": flow_signal.confidence,
                        "source_accuracy": flow_signal.source_accuracy,
                        "conviction": conv["conviction_pct"],
                        "net_flow": conv["net_flow"],
                        "prints": conv["total_prints"],
                    },
                )
                logger.info(
                    "  Options flow trade: %s %s @ $%s (conviction: %d%%)",
                    side.upper(), conv["ticker"], f"{price:,.2f}", conv["conviction_pct"],
                )
                options_executed.append(
                    {
                        "id": trade_id,
                        "coin": conv["ticker"],
                        "side": side,
                        "strategy_type": "options_momentum",
                        "source": "options_flow",
                        "source_key": source_key,
                        "signal_id": signal_id,
                        "confidence": flow_signal.confidence,
                        "entry_price": price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "metadata": {
                            "source": "options_flow",
                            "source_key": source_key,
                            "signal_id": signal_id,
                            "strategy_type": "options_momentum",
                            "confidence": flow_signal.confidence,
                            "source_accuracy": flow_signal.source_accuracy,
                        },
                    }
                )
                if tg.is_configured():
                    tg.notify_trade_opened(
                        {"coin": conv["ticker"], "side": side, "entry_price": price},
                        source="options_flow",
                    )
        if options_executed:
            logger.info("  Executed %d options flow trades", len(options_executed))
    except Exception as exc:
        logger.warning("  Options flow trading error: %s", exc)


def _run_copy_trading(container, regime_data):
    """Phase 4b: copy trading — WebSocket + REST."""
    from src.notifications import telegram_bot as tg
    logger.info("Phase 4b: Copy Trading")
    ws_signals = []
    if container.position_monitor:
        ws_signals = container.position_monitor.drain_signals()
        if ws_signals:
            logger.info("  WebSocket: %d real-time signals", len(ws_signals))
    copy_signals = ws_signals + (
        container.copy_trader.scan_top_traders(top_n=10) if container.copy_trader else []
    )

    try:
        from src.discovery.golden_bridge import get_golden_copy_signals, auto_connect_golden_wallets
        auto_connect_golden_wallets()
        golden_signals = get_golden_copy_signals()
        if golden_signals:
            logger.info("  Golden bridge: %d signals", len(golden_signals))
            copy_signals = golden_signals + copy_signals
    except Exception as exc:
        logger.debug("  Golden bridge skipped: %s", exc)

    if copy_signals and container.copy_trader:
        copy_executed = container.copy_trader.execute_copy_signals(copy_signals, regime_data=regime_data)
        logger.info("  Executed %d copy trades", len(copy_executed))
        if tg.is_configured():
            for t in copy_executed:
                tg.notify_trade_opened(t, source="copy")
        mirror_executed_trades_to_live(
            container,
            copy_executed,
            success_label="  LIVE COPY",
            skip_label="  Live trader requested but not deployable; skipping copy mirroring",
        )


def _collect_closed_trade_events(container, initial_closed):
    """Merge in-memory close events from paper/copy traders, deduping by trade id."""
    merged = list(initial_closed or [])
    seen = set()
    for trade in merged:
        trade_id = trade.get("trade_id")
        if trade_id is not None:
            seen.add(trade_id)

    for trader_name in ("paper_trader", "copy_trader"):
        trader = getattr(container, trader_name, None)
        if not trader or not hasattr(trader, "drain_closed_events"):
            continue
        try:
            for event in trader.drain_closed_events() or []:
                trade_id = event.get("trade_id")
                if trade_id is not None and trade_id in seen:
                    continue
                merged.append(event)
                if trade_id is not None:
                    seen.add(trade_id)
        except Exception as exc:
            logger.debug("  Failed draining %s close events: %s", trader_name, exc)

    return merged


def _process_closed_trades(container, closed):
    """Phase 4c: feed outcomes to arena, agent scorer, shadow tracker, AND Kelly sizer."""
    for c_trade in closed:
        try:
            meta = _coerce_metadata(c_trade.get("metadata") or {})
            if meta.get("synthetic_reconciliation") or c_trade.get("reason") == "live_reconciled_closed":
                continue
            stype = c_trade.get("strategy_type", "unknown")
            source_name = str(meta.get("source", c_trade.get("source", "strategy"))).strip() or "strategy"
            source_key = str(meta.get("source_key", c_trade.get("source_key", ""))).strip() or build_source_key(
                source_name,
                strategy_type=stype,
                trader_address=str(meta.get("source_trader", c_trade.get("trader_address", ""))),
                coin=c_trade.get("coin", ""),
                agent_id=str(meta.get("agent_id", "")),
            )
            pnl = c_trade.get("pnl", 0)
            entry = c_trade.get("entry_price", 1)
            size = c_trade.get("size", 0)
            leverage = c_trade.get("leverage", 1)
            notional = entry * max(size, 1e-8)
            return_pct = pnl / max(notional, 1)

            if container.arena:
                agent_id = str(meta.get("agent_id", "")).strip()
                if source_name == "arena_champion" and agent_id:
                    container.arena.record_trade_result(agent_id, pnl, return_pct)
                else:
                    container.arena.record_trade_for_strategy(stype, pnl, return_pct)

            _record_shadow_trade(container, c_trade, pnl, return_pct, entry)

            # Kelly sizer: feed trade outcomes so it can compute win_rate + reward/risk
            if container.kelly_sizer:
                try:
                    container.kelly_sizer.record_outcome(
                        strategy_key=source_key or stype or "unknown",
                        pnl=pnl,
                        entry_price=entry,
                        size=max(size, 1e-8),
                        leverage=max(leverage, 1),
                    )
                except Exception:
                    pass

            # AgentScorer outcome
            if container.agent_scorer:
                try:
                    signal_id = meta.get("signal_id", c_trade.get("signal_id", ""))
                    if signal_id:
                        container.agent_scorer.record_outcome(source_key, signal_id, pnl, return_pct)
                except Exception:
                    pass
        except Exception:
            pass


def _run_alpha_arena(container, regime_data):
    """Run the arena maintenance cycle and return champion candidates for ranking."""
    if not container.arena:
        return []
    logger.info("Phase 3d3: Alpha Arena")
    try:
        import requests
        arena_candles = None
        try:
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": "BTC", "interval": "1h",
                    "startTime": int((datetime.utcnow().timestamp() - 720 * 3600) * 1000),
                    "endTime": int(datetime.utcnow().timestamp() * 1000),
                },
            }
            resp = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=15)
            if resp.status_code == 200:
                raw = resp.json()
                if isinstance(raw, list) and len(raw) >= 50:
                    arena_candles = [
                        {"open": float(c.get("o", 0)), "high": float(c.get("h", 0)),
                         "low": float(c.get("l", 0)), "close": float(c.get("c", 0)),
                         "volume": float(c.get("v", 0))} for c in raw
                    ]
        except Exception:
            pass

        container.arena.run_cycle(historical_candles=arena_candles)
        stats = container.arena.get_stats()
        logger.info(
            "  Arena: %d active, %d champions, PnL=$%.2f",
            stats["active_agents"], stats["champions"], stats["total_arena_pnl"],
        )

        candidates = []
        if arena_candles and len(arena_candles) >= 30:
            try:
                champion_signals = container.arena.get_champion_signals(
                    current_candles=arena_candles[-100:],
                    min_fitness=config.ARENA_MIN_FITNESS,
                    min_trades=config.ARENA_MIN_TRADES,
                )
                if champion_signals:
                    logger.info("  Arena champions: %d signals", len(champion_signals))
                    for sig in champion_signals:
                        try:
                            price = sig.get("price", 0)
                            if price <= 0:
                                continue
                            source_key = build_source_key(
                                "arena_champion",
                                strategy_type=sig["strategy_type"],
                                coin=sig.get("coin", "BTC"),
                                agent_id=sig.get("agent_id", ""),
                            )
                            fitness = float(sig.get("agent_fitness", 0.0) or 0.0)
                            conf = float(sig.get("confidence", 0.0) or 0.0)
                            composite = min(0.98, max(conf * 0.7 + min(max(fitness, 0.0), 1.0) * 0.3, 0.0))
                            candidates.append(
                                {
                                    "id": None,
                                    "name": f"arena_{sig.get('agent_name', 'champion')}_{sig.get('side', 'long')}",
                                    "strategy_type": sig["strategy_type"],
                                    "current_score": round(composite, 3),
                                    "confidence": round(conf, 3),
                                    "direction": sig.get("side", "long"),
                                    "side": sig.get("side", "long"),
                                    "source": "arena_champion",
                                    "source_key": source_key,
                                    "agent_id": sig.get("agent_id", ""),
                                    "agent_name": sig.get("agent_name", ""),
                                    "parameters": {
                                        "coins": [sig.get("coin", "BTC")],
                                    },
                                    "metrics": {},
                                    "metadata": {
                                        "source": "arena_champion",
                                        "source_key": source_key,
                                        "reason": f"Arena champion: {sig.get('agent_name', 'unknown')}",
                                        "agent_id": sig.get("agent_id", ""),
                                        "agent_name": sig.get("agent_name", ""),
                                        "agent_fitness": sig.get("agent_fitness", 0),
                                        "agent_elo": sig.get("agent_elo", 0),
                                        "atr_pct": sig.get("atr_pct", 0.02),
                                    },
                                }
                            )
                        except Exception as exc:
                            logger.debug("  Champion candidate error: %s", exc)
            except Exception as exc:
                logger.debug("  Champion signals error: %s", exc)
        return candidates
    except Exception as exc:
        logger.warning("  Arena error: %s", exc)
        return []
