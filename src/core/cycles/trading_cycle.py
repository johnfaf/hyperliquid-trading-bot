"""
Trading Cycle (Tier 2)
======================
Lightweight trading cycle: score strategies, detect regime, trade.
Uses existing DB data from last discovery — no leaderboard scanning.
Runs every ~5 minutes to react to market changes quickly.

Extracted from ``HyperliquidResearchBot._run_trading_cycle``.
"""
import logging
import time
from datetime import datetime

import config
from src.data import database as db

logger = logging.getLogger(__name__)


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
        open_trades = db.get_open_paper_trades()
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
        source = meta.get("source", f"strategy:{stype}")
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
        if container.live_trader.check_daily_loss():
            logger.error("KILL SWITCH ACTIVE — daily loss limit hit, skipping live trades")

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
            for pm in polymarket_signals:
                synthetic = {
                    "id": None,
                    "name": f"polymarket_{pm.get('coin', 'UNK')}_{pm.get('side', '?')}",
                    "strategy_type": "event_driven",
                    "trader_address": "polymarket",
                    "current_score": pm.get("confidence", 0.5),
                    "confidence": pm.get("confidence", 0.5),
                    "direction": pm.get("side", "long"),
                    "side": pm.get("side", "long"),
                    "source": "polymarket",
                    "parameters": {
                        "coins": [pm.get("coin", "BTC")],
                        "market": pm.get("polymarket_market", ""),
                        "probability": pm.get("polymarket_probability", 0),
                    },
                    "metrics": {},
                    "metadata": {
                        "polymarket_volume_24h": pm.get("polymarket_volume_24h", 0),
                        "reason": pm.get("reason", ""),
                    },
                }
                top_strategies.append(synthetic)
            logger.info("  Injected %d Polymarket signals", len(polymarket_signals))

        # Check existing positions
        closed = container.paper_trader.check_open_positions() if container.paper_trader else []
        if closed:
            logger.info("  Closed %d positions", len(closed))
            if tg.is_configured():
                for c_trade in closed:
                    tg.notify_trade_closed(
                        c_trade, c_trade.get("exit_price", 0),
                        c_trade.get("pnl", 0), c_trade.get("reason", ""),
                    )

        # Cross-venue signal confirmation
        _run_cross_venue_confirmation(container, top_strategies)

        # Decision engine
        open_trades = db.get_open_paper_trades()
        kelly_stats = None
        if container.kelly_sizer:
            try:
                kelly_stats = container.kelly_sizer.get_all_sizing_stats()
            except Exception:
                pass
        if container.decision_engine:
            top_strategies = container.decision_engine.decide(
                top_strategies, regime_data=regime_data,
                open_positions=open_trades, kelly_stats=kelly_stats,
            )

        # AgentScorer dynamic weights
        _apply_agent_scorer_weights(container, top_strategies)

        # Execute new signals
        if top_strategies and container.paper_trader:
            executed = container.paper_trader.execute_strategy_signals(
                top_strategies, exchange_agg=container.exchange_agg,
                options_scanner=container.options_scanner,
                regime_data=regime_data, arena=container.arena,
            )
            logger.info("  Executed %d new paper trades", len(executed))

            # Mirror to live
            if container.live_trader and executed:
                for t in executed:
                    try:
                        live_result = container.live_trader.execute_signal(t)
                        if live_result:
                            logger.info(
                                "  LIVE: %s %s %s",
                                live_result.get("status", "?"),
                                t.get("coin", "?"), t.get("side", "?"),
                            )
                    except Exception as exc:
                        logger.warning("  Live execution error: %s", exc)

            if tg.is_configured():
                for t in executed:
                    tg.notify_trade_opened(t, source="strategy")

        # Phase 4a: LCRS execution
        if lcrs_signals:
            _execute_lcrs_signals(container, lcrs_signals, regime_data)

        # Phase 4a2: Options flow standalone trades
        _execute_options_flow_trades(container, regime_data)

        # Phase 4b: Copy trading
        _run_copy_trading(container, regime_data)

        # Phase 4c: Feed closed trade outcomes
        if closed:
            _process_closed_trades(container, closed)

        # Phase 5: Alpha Arena
        _run_alpha_arena(container, regime_data)

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
        if hasattr(container.agent_scorer, "apply_weights_to_signals"):
            for s in top_strategies:
                stype = s.get("strategy_type", "unknown")
                source_key = f"strategy:{stype}"
                weight = container.agent_scorer.get_weight(source_key)
                orig_conf = float(s.get("confidence", 0.5))
                s["confidence"] = round(orig_conf * 0.6 + weight * 0.4, 3)
                s["agent_scorer_weight"] = round(weight, 3)
    except Exception as exc:
        logger.debug("  AgentScorer weight apply error: %s", exc)


def _execute_lcrs_signals(container, lcrs_signals, regime_data):
    """Phase 4a: execute liquidation reversal signals."""
    from src.notifications import telegram_bot as tg
    logger.info("Phase 4a: Liquidation Strategy Execution")
    try:
        from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams
        lcrs_executed = []
        open_trades = db.get_open_paper_trades()
        account = db.get_paper_account()

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

                if account:
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
            if conv.get("conviction_pct", 0) < 70:
                continue
            flow_signal = signal_from_options_flow(
                ticker=conv["ticker"], direction=conv["direction"],
                net_flow=conv["net_flow"], prints=conv["total_prints"],
                conviction_pct=conv["conviction_pct"],
            )
            flow_signal.position_pct = 0.04
            flow_signal.leverage = 2.0
            price = float(mids.get(conv["ticker"], 0))
            if price <= 0:
                continue
            flow_signal.entry_price = price

            if container.firewall:
                passed, reason = container.firewall.validate(
                    flow_signal, regime_data=regime_data,
                    open_positions=db.get_open_paper_trades(),
                )
                if not passed:
                    logger.info("  Firewall rejected options flow %s: %s", conv["ticker"], reason)
                    continue

            if container.agent_scorer:
                container.agent_scorer.record_signal("options_flow", {
                    "coin": conv["ticker"], "side": flow_signal.side.value,
                    "confidence": flow_signal.confidence,
                })

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
                    stop_loss=round(sl, 2), take_profit=round(tp, 2),
                    metadata={
                        "source": "options_flow",
                        "conviction": conv["conviction_pct"],
                        "net_flow": conv["net_flow"],
                        "prints": conv["total_prints"],
                    },
                )
                logger.info(
                    "  Options flow trade: %s %s @ $%s (conviction: %d%%)",
                    side.upper(), conv["ticker"], f"{price:,.2f}", conv["conviction_pct"],
                )
                options_executed.append({"id": trade_id, "coin": conv["ticker"], "side": side})
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
        if container.live_trader and copy_executed:
            for t in copy_executed:
                try:
                    live_result = container.live_trader.execute_signal(t)
                    if live_result:
                        logger.info(
                            "  LIVE COPY: %s %s %s",
                            live_result.get("status", "?"),
                            t.get("coin", "?"), t.get("side", "?"),
                        )
                except Exception as exc:
                    logger.warning("  Live copy execution error: %s", exc)


def _process_closed_trades(container, closed):
    """Phase 4c: feed outcomes to arena, agent scorer, shadow tracker."""
    for c_trade in closed:
        try:
            stype = c_trade.get("strategy_type", "unknown")
            pnl = c_trade.get("pnl", 0)
            entry = c_trade.get("entry_price", 1)
            size = c_trade.get("size", 0)
            notional = entry * max(size, 1e-8)
            return_pct = pnl / max(notional, 1)

            if container.arena:
                container.arena.record_trade_for_strategy(stype, pnl, return_pct)

            _record_shadow_trade(container, c_trade, pnl, return_pct, entry)

            # AgentScorer outcome
            if container.agent_scorer:
                try:
                    meta = c_trade.get("metadata") or {}
                    signal_id = meta.get("signal_id", "")
                    source_key = f"strategy:{stype}"
                    if signal_id:
                        container.agent_scorer.record_outcome(source_key, signal_id, pnl, return_pct)
                except Exception:
                    pass
        except Exception:
            pass


def _run_alpha_arena(container, regime_data):
    """Phase 5: Alpha Arena cycle."""
    if not container.arena:
        return
    logger.info("Phase 5: Alpha Arena")
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

        # Champion signals → paper trading
        if arena_candles and len(arena_candles) >= 30:
            try:
                champion_signals = container.arena.get_champion_signals(
                    current_candles=arena_candles[-100:], min_fitness=0.15, min_trades=10,
                )
                if champion_signals:
                    logger.info("  Arena champions: %d signals", len(champion_signals))
                    account = db.get_paper_account()
                    for sig in champion_signals:
                        try:
                            price = sig.get("price", 0)
                            if price <= 0:
                                continue
                            side = sig["side"]
                            conf = sig["confidence"]
                            sl = price * (0.95 if side == "long" else 1.05)
                            tp = price * (1.10 if side == "long" else 0.90)
                            if account:
                                size_usd = account["balance"] * 0.05 * conf
                                size = size_usd / price
                                db.open_paper_trade(
                                    strategy_id=None, coin=sig["coin"], side=side,
                                    entry_price=price, size=size, leverage=2,
                                    stop_loss=round(sl, 2), take_profit=round(tp, 2),
                                    metadata={
                                        "source": "arena_champion",
                                        "agent_id": sig["agent_id"],
                                        "agent_name": sig["agent_name"],
                                        "strategy_type": sig["strategy_type"],
                                        "agent_fitness": sig["agent_fitness"],
                                        "agent_elo": sig["agent_elo"],
                                        "confidence": conf,
                                    },
                                )
                                logger.info(
                                    "  Champion trade: %s %s @ $%s | agent=%s",
                                    side.upper(), sig["coin"], f"{price:,.2f}", sig["agent_name"],
                                )
                        except Exception as exc:
                            logger.debug("  Champion trade exec error: %s", exc)
            except Exception as exc:
                logger.debug("  Champion signals error: %s", exc)
    except Exception as exc:
        logger.warning("  Arena error: %s", exc)
