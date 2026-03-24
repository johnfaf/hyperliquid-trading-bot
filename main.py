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
from src import database as db
from src.database import init_db, restore_from_json, backup_to_json
from src.trader_discovery import TraderDiscovery
from src.strategy_identifier import StrategyIdentifier
from src.strategy_scorer import StrategyScorer
from src.paper_trader import PaperTrader
from src.copy_trader import CopyTrader
from src.reporter import Reporter
from src.dashboard import start_dashboard, set_v2_components
from src.exchange_aggregator import ExchangeAggregator
from src.options_flow import OptionsFlowScanner
from src.regime_detector import RegimeDetector
from src.decision_firewall import DecisionFirewall
from src.agent_scoring import AgentScorer
from src.features import FeatureEngine
from src.alpha_arena import AlphaArena
from src.liquidation_strategy import LiquidationStrategy
from src.kelly_sizing import KellySizer
from src.trade_memory import TradeMemory
from src.calibration import CalibrationTracker
from src.llm_filter import LLMFilter
from src.signal_processor import SignalProcessor, ArenaIncubator
from src.decision_engine import DecisionEngine
from src.exchanges.scanner import MultiExchangeScanner
from src import telegram_bot as tg

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

    # Console handler — use stdout (not stderr) so Railway doesn't tag INFO as "error"
    ch = logging.StreamHandler(sys.stdout)
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
        self.exchange_agg = ExchangeAggregator()
        self.options_scanner = OptionsFlowScanner()
        self.regime_detector = RegimeDetector(exchange_agg=self.exchange_agg)
        self.firewall = DecisionFirewall()
        self.agent_scorer = AgentScorer()
        self.feature_engine = FeatureEngine()

        # V2.5: New modules — liquidation strategy, Kelly sizing, trade memory, calibration, LLM filter
        self.liquidation_strategy = LiquidationStrategy()
        self.kelly_sizer = KellySizer()
        self.trade_memory = TradeMemory()
        self.calibration = CalibrationTracker()
        self.llm_filter = LLMFilter()
        self.signal_processor = SignalProcessor()
        self.arena_incubator = ArenaIncubator()
        self.decision_engine = DecisionEngine()

        # V4: Multi-exchange scanner (Hyperliquid + Lighter cross-venue confirmation)
        try:
            self.multi_scanner = MultiExchangeScanner(config={
                "lighter_enabled": config.LIGHTER_ENABLED,
            })
            self.logger.info(f"Multi-exchange scanner initialized: {list(self.multi_scanner.adapters.keys())}")
        except Exception as e:
            self.logger.warning(f"Multi-exchange scanner init failed (continuing single-venue): {e}")
            self.multi_scanner = None

        # Bootstrap Kelly from existing agent scorer history
        self.kelly_sizer.load_from_agent_scorer(self.agent_scorer)

        self.copy_trader = CopyTrader(
            firewall=self.firewall,
            agent_scorer=self.agent_scorer,
            kelly_sizer=self.kelly_sizer,
            trade_memory=self.trade_memory,
            calibration=self.calibration,
        )
        self.reporter = Reporter()
        self.arena = AlphaArena()

        # Paper trader with ALL V2 + V2.5 components wired in
        self.paper_trader = PaperTrader(
            firewall=self.firewall,
            agent_scorer=self.agent_scorer,
            feature_engine=self.feature_engine,
            kelly_sizer=self.kelly_sizer,
            trade_memory=self.trade_memory,
            calibration=self.calibration,
            llm_filter=self.llm_filter,
        )

        # Start the unified web dashboard (main + options on same port)
        try:
            set_v2_components(
                firewall=self.firewall,
                regime_detector=self.regime_detector,
                arena=self.arena,
                kelly_sizer=self.kelly_sizer,
                trade_memory=self.trade_memory,
                calibration=self.calibration,
                llm_filter=self.llm_filter,
                liquidation_strategy=self.liquidation_strategy,
                signal_processor=self.signal_processor,
                arena_incubator=self.arena_incubator,
                decision_engine=self.decision_engine,
                multi_scanner=self.multi_scanner,
            )
            self.dashboard = start_dashboard(options_scanner=self.options_scanner)
            self.logger.info("Unified dashboard started (main + options flow on same port).")
        except Exception as e:
            self.logger.warning(f"Dashboard failed to start: {e}")

        self.logger.info("Bot initialized successfully.")

        # Send Telegram startup notification
        if tg.is_configured():
            tg.send_startup_message()
            self.logger.info("Telegram notifications enabled.")
        else:
            self.logger.info("Telegram not configured — set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID to enable.")

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

            # Phase 3b: Multi-exchange market overview
            self.logger.info("Phase 3b: Multi-Exchange Volume Analysis")
            try:
                market_overview = self.exchange_agg.get_market_overview()
                self.logger.info(f"  Market bias: {market_overview.get('overall_bias', '?')} "
                               f"(score: {market_overview.get('overall_bias_score', 0):+.4f})")
                self.logger.info(f"  Bullish: {market_overview.get('bullish_coins', [])}")
                self.logger.info(f"  Bearish: {market_overview.get('bearish_coins', [])}")

                # Telegram: notify if strong market bias
                if tg.is_configured():
                    tg.notify_market_bias(market_overview)
            except Exception as e:
                self.logger.warning(f"  Exchange aggregator error: {e}")
                market_overview = {}

            # Phase 3c: Options Flow Scan (Deribit)
            self.logger.info("Phase 3c: Options Flow Scan")
            try:
                flow_result = self.options_scanner.scan_flow()
                self.logger.info(f"  Unusual prints: {flow_result.get('unusual_prints', 0)}")
                self.logger.info(f"  Top convictions: {flow_result.get('top_convictions', 0)}")

                # Telegram: notify strong flow signals
                if tg.is_configured() and self.options_scanner.top_convictions:
                    for conv in self.options_scanner.top_convictions[:3]:
                        if conv.get("conviction_pct", 0) > 60:
                            tg.notify_strong_signal({
                                "source": "options_flow",
                                "ticker": conv["ticker"],
                                "direction": conv["direction"],
                                "net_flow": conv["net_flow"],
                                "prints": conv["total_prints"],
                                "conviction": conv["conviction_pct"],
                            })
            except Exception as e:
                self.logger.warning(f"  Options flow scan error: {e}")

            # Phase 3d: Regime Detection
            self.logger.info("Phase 3d: Market Regime Detection")
            regime_data = {}
            try:
                regime_data = self.regime_detector.get_market_regime()
                self.logger.info(f"  Regime: {regime_data.get('overall_regime', '?')} "
                               f"(confidence={regime_data.get('overall_confidence', 0):.0%})")
                guidance = regime_data.get("strategy_guidance", {})
                self.logger.info(f"  Activate: {guidance.get('activate', [])}")
                self.logger.info(f"  Pause: {guidance.get('pause', [])}")
                self.logger.info(f"  Size modifier: {guidance.get('size_modifier', 1.0):.0%}")
            except Exception as e:
                self.logger.warning(f"  Regime detection error: {e}")

            # Phase 3e: Multi-Exchange Scan + Cross-Venue Confirmation
            self.logger.info("Phase 3e: Multi-Exchange Scanner")
            cross_venue_data = {}
            funding_arbs = []
            try:
                if self.multi_scanner:
                    # Health check all venues
                    venue_health = self.multi_scanner.check_health()
                    self.logger.info(f"  Venue health: {venue_health}")

                    # Get aggregated market data for cross-venue comparison
                    common_markets = self.multi_scanner.get_common_markets()
                    if common_markets:
                        self.logger.info(f"  Common markets across venues: {common_markets[:15]}...")

                    # Scan for funding rate arbitrage opportunities
                    funding_arbs = self.multi_scanner.scan_funding_arb()
                    if funding_arbs:
                        for arb in funding_arbs[:3]:
                            self.logger.info(
                                f"  Funding arb: {arb.coin} "
                                f"long@{arb.long_venue}({arb.long_funding_rate:+.4%}) / "
                                f"short@{arb.short_venue}({arb.short_funding_rate:+.4%}) "
                                f"= {arb.funding_spread_annualized:.1f}% ann."
                            )
                    else:
                        self.logger.info("  No funding arb opportunities found")

                    # Store cross-venue data for signal enrichment later
                    cross_venue_data = {
                        "health": venue_health,
                        "common_markets": common_markets,
                        "funding_arbs": funding_arbs,
                    }
                else:
                    self.logger.info("  Multi-exchange scanner not available (single venue mode)")
            except Exception as e:
                self.logger.warning(f"  Multi-exchange scanner error: {e}")

            # Phase 3f: Liquidation Cascade Reversal Strategy
            self.logger.info("Phase 3e: Liquidation Strategy Scan")
            lcrs_signals = []
            try:
                from src import hyperliquid_client as hl_client
                mids = hl_client.get_all_mids() or {}
                lcrs_coins = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "ARB",
                              "OP", "SUI", "APT", "INJ", "SEI"]

                for coin in lcrs_coins:
                    price = float(mids.get(coin, 0))
                    if price <= 0:
                        continue
                    try:
                        # Build features for LCRS from regime data + feature engine
                        lcrs_features = {}
                        if regime_data and "per_coin" in regime_data:
                            coin_regime = regime_data["per_coin"].get(coin, {})
                            lcrs_features["trend_strength"] = coin_regime.get("trend_strength", 0.5)
                            lcrs_features["volatility"] = coin_regime.get("atr_pct", 0.02)
                            lcrs_features["volume_ratio"] = coin_regime.get("volume_ratio", 1.0)

                        # Get funding rate from Hyperliquid
                        try:
                            import requests
                            meta_resp = requests.post("https://api.hyperliquid.xyz/info",
                                json={"type": "metaAndAssetCtxs"}, timeout=10)
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

                        # Enrich with feature engine data
                        try:
                            payload = {
                                "type": "candleSnapshot",
                                "req": {
                                    "coin": coin,
                                    "interval": "1h",
                                    "startTime": int((datetime.utcnow().timestamp() - 100 * 3600) * 1000),
                                    "endTime": int(datetime.utcnow().timestamp() * 1000),
                                }
                            }
                            resp = requests.post("https://api.hyperliquid.xyz/info",
                                                 json=payload, timeout=10)
                            if resp.status_code == 200:
                                raw = resp.json()
                                if isinstance(raw, list) and len(raw) >= 20:
                                    candles = [{"open": float(c.get("o", 0)), "high": float(c.get("h", 0)),
                                                "low": float(c.get("l", 0)), "close": float(c.get("c", 0)),
                                                "volume": float(c.get("v", 0))} for c in raw]
                                    feat = self.feature_engine.compute(coin, candles)
                                    lcrs_features.setdefault("rsi", feat.rsi)
                                    lcrs_features.setdefault("momentum_score", feat.momentum_score)
                                    lcrs_features.setdefault("trend_strength", feat.trend_strength)
                                    lcrs_features.setdefault("volatility", feat.volatility)
                                    lcrs_features.setdefault("volume_ratio", feat.volume_ratio)
                                    lcrs_features.setdefault("overall_score", feat.overall_score)
                                    lcrs_features.setdefault("bollinger_position", feat.bollinger_position)
                                    # Price change from last 8 candles
                                    if len(candles) >= 8:
                                        lcrs_features["price_change"] = (candles[-1]["close"] - candles[-8]["close"]) / candles[-8]["close"]
                        except Exception:
                            pass

                        sig = self.liquidation_strategy.generate_signal(coin, lcrs_features, price)
                        if sig:
                            lcrs_signals.append(sig)
                            self.logger.info(f"  LCRS: {sig['side'].upper()} {coin} "
                                           f"(conf={sig['confidence']:.0%}, type={sig['features'].get('setup_type', '')})")
                    except Exception as e:
                        self.logger.debug(f"  LCRS scan error {coin}: {e}")

                if lcrs_signals:
                    self.logger.info(f"  LCRS found {len(lcrs_signals)} setups")
                else:
                    self.logger.info("  LCRS: no setups detected")
            except Exception as e:
                self.logger.warning(f"  Liquidation strategy error: {e}")

            # Phase 4: Paper trade top strategies (regime-filtered)
            self.logger.info("Phase 4: Paper Trading (regime-aware)")
            top_strategies = self.scorer.get_top_strategies(n=50)

            # Filter strategies by regime — pause those that don't fit
            if regime_data:
                top_strategies = self.regime_detector.filter_strategies_by_regime(
                    top_strategies, regime_data
                )
                self.logger.info(f"  Post-regime filter: {len(top_strategies)} strategies active")

            # V3: Signal Processing — dedup, conflict resolution, compression
            self.logger.info("Phase 4 pre-process: Signal Processor")
            top_strategies = self.signal_processor.process(
                top_strategies, regime_data=regime_data
            )
            self.logger.info(f"  Post-signal-processor: {len(top_strategies)} strategies")

            # Check existing positions first (V2: outcomes auto-feed to agent scorer + firewall)
            closed = self.paper_trader.check_open_positions()
            if closed:
                self.logger.info(f"  Closed {len(closed)} positions")
                for c in closed:
                    # Telegram: notify closed trades
                    if tg.is_configured():
                        tg.notify_trade_closed(c, c.get("exit_price", 0), c.get("pnl", 0), c.get("reason", ""))

            # V4: Cross-venue signal confirmation (enriches strategies before decision engine)
            if self.multi_scanner and self.multi_scanner.cross_venue and top_strategies:
                self.logger.info("Phase 4 cross-venue: Signal Confirmation")
                try:
                    # Extract coin + direction from each strategy for cross-venue check
                    signals_to_confirm = []
                    for s in top_strategies:
                        params = s.get("parameters", {})
                        if isinstance(params, str):
                            import json
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
                            signals_to_confirm.append({
                                "coin": coin,
                                "direction": direction,
                                "score": score,
                            })

                    if signals_to_confirm:
                        confirmed = self.multi_scanner.confirm_signals(signals_to_confirm)

                        # Inject confirmation scores back into strategies
                        confirm_map = {
                            f"{c.coin}:{c.direction}": c.confirmation_score
                            for c in confirmed
                        }
                        for s in top_strategies:
                            params = s.get("parameters", {})
                            if isinstance(params, str):
                                import json
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
                            # Store cross-venue data in metadata for decision engine
                            if "metadata" not in s:
                                s["metadata"] = {}
                            s["metadata"]["cross_venue_score"] = cv_score

                            # Boost score by up to 15% based on cross-venue confirmation
                            if cv_score > 0.15:
                                original = s.get("score", 0.5)
                                boost = cv_score * 0.15  # Max 15% boost
                                s["score"] = min(1.0, original + boost)

                        boosted = sum(1 for s in top_strategies if s.get("metadata", {}).get("cross_venue_score", 0) > 0.15)
                        self.logger.info(f"  Cross-venue: confirmed {len(signals_to_confirm)} signals, "
                                        f"{boosted} boosted")

                except Exception as e:
                    self.logger.warning(f"  Cross-venue confirmation error: {e}")

            # V3: Final Decision Engine — rank, score, and log decisions
            open_trades = db.get_open_paper_trades()
            kelly_stats = None
            try:
                kelly_stats = self.kelly_sizer.get_all_sizing_stats()
            except Exception:
                pass
            top_strategies = self.decision_engine.decide(
                top_strategies,
                regime_data=regime_data,
                open_positions=open_trades,
                kelly_stats=kelly_stats,
            )

            # Execute new signals from strategies (V2 pipeline: features → scoring → firewall → consensus)
            if top_strategies:
                executed = self.paper_trader.execute_strategy_signals(
                    top_strategies, exchange_agg=self.exchange_agg,
                    options_scanner=self.options_scanner,
                    regime_data=regime_data,
                    arena=self.arena,
                )
                self.logger.info(f"  Executed {len(executed)} new paper trades")
                # Telegram: notify new trades
                if tg.is_configured():
                    for t in executed:
                        tg.notify_trade_opened(t, source="strategy")

            # Phase 4a: Execute LCRS signals through full V2 pipeline
            if lcrs_signals:
                self.logger.info("Phase 4a: Liquidation Strategy Execution")
                try:
                    # Package LCRS signals as strategies for the paper_trader pipeline
                    lcrs_executed = []
                    open_trades = db.get_open_paper_trades()
                    account = db.get_paper_account()

                    for sig in lcrs_signals:
                        try:
                            from src.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams
                            trade_signal = TradeSignal(
                                coin=sig["coin"],
                                side=SignalSide(sig["side"]),
                                confidence=sig["confidence"],
                                source=SignalSource.STRATEGY,
                                reason=f"LCRS: {sig['features'].get('setup_type', 'unknown')}",
                                strategy_type="liquidation_reversal",
                                entry_price=sig["price"],
                                leverage=sig["leverage"],
                                position_pct=sig.get("position_pct", 0.06),
                                risk=RiskParams(
                                    stop_loss_pct=0.025,
                                    take_profit_pct=0.05,
                                ),
                                regime=regime_data.get("overall_regime", "") if regime_data else "",
                            )

                            # Full V2 pipeline: firewall → Kelly → memory → LLM filter
                            if self.firewall:
                                passed, reason = self.firewall.validate(
                                    trade_signal, regime_data=regime_data, open_positions=open_trades)
                                if not passed:
                                    self.logger.info(f"  LCRS firewall rejected {sig['coin']}: {reason}")
                                    continue

                            # Kelly sizing
                            if self.kelly_sizer and account:
                                sizing = self.kelly_sizer.get_sizing(
                                    "liquidation_reversal", account["balance"], trade_signal.confidence)
                                trade_signal.position_pct = sizing.position_pct

                            # Trade memory check
                            if self.trade_memory:
                                mem = self.trade_memory.find_similar(
                                    sig.get("features", {}), coin=sig["coin"], side=sig["side"])
                                if mem.recommendation == "avoid":
                                    self.logger.info(f"  LCRS memory blocked {sig['coin']}: {mem.reason}")
                                    continue

                            # LLM filter
                            if self.llm_filter:
                                ctx = {"regime_data": regime_data, "open_positions": open_trades}
                                approved, adj_conf, reason = self.llm_filter.filter(sig, ctx)
                                if not approved:
                                    self.logger.info(f"  LCRS LLM filter blocked {sig['coin']}: {reason}")
                                    continue
                                trade_signal.confidence = adj_conf

                            # Execute
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
                                self.logger.info(f"  LCRS executed: {sig['side'].upper()} {sig['coin']} "
                                               f"@ ${sig['price']:,.2f} (conf={trade_signal.confidence:.0%})")

                                if tg.is_configured():
                                    tg.notify_trade_opened(
                                        {"coin": sig["coin"], "side": sig["side"], "entry_price": sig["price"]},
                                        source="liquidation_strategy"
                                    )

                        except Exception as e:
                            self.logger.debug(f"  LCRS execution error {sig.get('coin')}: {e}")

                    if lcrs_executed:
                        self.logger.info(f"  Executed {len(lcrs_executed)} LCRS trades")
                except Exception as e:
                    self.logger.warning(f"  LCRS execution phase error: {e}")

            # Phase 4a2: Options flow standalone trades (high conviction → direct trade)
            self.logger.info("Phase 4a2: Options Flow Trades")
            try:
                from src.signal_schema import signal_from_options_flow
                options_executed = []
                for conv in (getattr(self.options_scanner, 'top_convictions', None) or []):
                    if conv.get("conviction_pct", 0) >= 70:  # Only very high conviction
                        flow_signal = signal_from_options_flow(
                            ticker=conv["ticker"],
                            direction=conv["direction"],
                            net_flow=conv["net_flow"],
                            prints=conv["total_prints"],
                            conviction_pct=conv["conviction_pct"],
                        )
                        # Set reasonable sizing
                        flow_signal.position_pct = 0.04  # 4% per options flow trade (conservative)
                        flow_signal.leverage = 2.0

                        # Get entry price
                        price = float(mids.get(conv["ticker"], 0))
                        if price <= 0:
                            continue
                        flow_signal.entry_price = price

                        # Route through firewall
                        passed, reason = self.firewall.validate(
                            flow_signal, regime_data=regime_data,
                            open_positions=db.get_open_paper_trades(),
                        )
                        if not passed:
                            self.logger.info(f"  Firewall rejected options flow {conv['ticker']}: {reason}")
                            continue

                        # Record with agent scorer
                        source_key = "options_flow"
                        signal_id = self.agent_scorer.record_signal(source_key, {
                            "coin": conv["ticker"],
                            "side": flow_signal.side.value,
                            "confidence": flow_signal.confidence,
                        })

                        # Execute as paper trade
                        account = db.get_paper_account()
                        if account:
                            size_usd = account["balance"] * flow_signal.effective_size
                            size = size_usd / price
                            side = flow_signal.side.value

                            if side == "long":
                                sl = price * (1 - 0.05)
                                tp = price * (1 + 0.10)
                            else:
                                sl = price * (1 + 0.05)
                                tp = price * (1 - 0.10)

                            trade_id = db.open_paper_trade(
                                strategy_id=None,
                                coin=conv["ticker"],
                                side=side,
                                entry_price=price,
                                size=size,
                                leverage=2,
                                stop_loss=round(sl, 2),
                                take_profit=round(tp, 2),
                                metadata={
                                    "source": "options_flow",
                                    "conviction": conv["conviction_pct"],
                                    "net_flow": conv["net_flow"],
                                    "prints": conv["total_prints"],
                                    "signal_id": signal_id,
                                },
                            )
                            self.logger.info(f"  Options flow trade: {side.upper()} {conv['ticker']} "
                                           f"@ ${price:,.2f} (conviction: {conv['conviction_pct']}%)")
                            options_executed.append({"id": trade_id, "coin": conv["ticker"], "side": side})

                            if tg.is_configured():
                                tg.notify_trade_opened(
                                    {"coin": conv["ticker"], "side": side, "entry_price": price},
                                    source="options_flow"
                                )

                if options_executed:
                    self.logger.info(f"  Executed {len(options_executed)} options flow trades")
            except Exception as e:
                self.logger.warning(f"  Options flow trading error: {e}")

            # Phase 4b: Copy trading - mirror top trader positions (V2: firewall gated)
            self.logger.info("Phase 4b: Copy Trading (V2)")
            copy_signals = self.copy_trader.scan_top_traders(top_n=10)
            if copy_signals:
                copy_executed = self.copy_trader.execute_copy_signals(
                    copy_signals, regime_data=regime_data
                )
                self.logger.info(f"  Executed {len(copy_executed)} copy trades")
                if tg.is_configured():
                    for t in copy_executed:
                        tg.notify_trade_opened(t, source="copy")

            # Phase 4c: Feed closed trade outcomes to Alpha Arena
            if closed:
                for c in closed:
                    try:
                        stype = c.get("strategy_type", "unknown")
                        pnl = c.get("pnl", 0)
                        entry = c.get("entry_price", 1)
                        size = 1
                        return_pct = pnl / max(entry * size, 1)
                        self.arena.record_trade_for_strategy(stype, pnl, return_pct)
                    except Exception:
                        pass

            # Phase 5: Alpha Arena Cycle
            self.logger.info("Phase 5: Alpha Arena")
            try:
                # Fetch historical candles for backtesting new agents
                arena_candles = None
                try:
                    import requests
                    payload = {
                        "type": "candleSnapshot",
                        "req": {
                            "coin": "BTC",
                            "interval": "1h",
                            "startTime": int((datetime.utcnow().timestamp() - 720 * 3600) * 1000),
                            "endTime": int(datetime.utcnow().timestamp() * 1000),
                        }
                    }
                    resp = requests.post("https://api.hyperliquid.xyz/info",
                                         json=payload, timeout=15)
                    if resp.status_code == 200:
                        raw = resp.json()
                        if isinstance(raw, list) and len(raw) >= 50:
                            arena_candles = [
                                {"open": float(c.get("o", 0)), "high": float(c.get("h", 0)),
                                 "low": float(c.get("l", 0)), "close": float(c.get("c", 0)),
                                 "volume": float(c.get("v", 0))}
                                for c in raw
                            ]
                except Exception:
                    pass

                self.arena.run_cycle(historical_candles=arena_candles)
                stats = self.arena.get_stats()
                self.logger.info(f"  Arena: {stats['active_agents']} active, "
                               f"{stats['champions']} champions, "
                               f"PnL=${stats['total_arena_pnl']:.2f}")
            except Exception as e:
                self.logger.warning(f"  Arena error: {e}")

            # Phase 6: Report
            self.logger.info("Phase 6: Status Update")
            status = self.reporter.print_live_status()
            print(status)

            # Telegram: send cycle summary
            if tg.is_configured():
                summary = self.paper_trader.get_account_summary()
                summary["market_bias"] = market_overview.get("overall_bias", "unknown")
                tg.notify_cycle_summary(summary)

            # V2.5 status: Kelly, Memory, Calibration, LLM Filter, LCRS
            self.logger.info("V2.5 Module Status:")
            try:
                lcrs_stats = self.liquidation_strategy.get_stats()
                self.logger.info(f"  LCRS: {lcrs_stats['setups_detected']} setups, "
                               f"{lcrs_stats['signals_generated']} signals")
            except Exception:
                pass
            try:
                kelly_stats = self.kelly_sizer.get_all_sizing_stats()
                edge_count = sum(1 for v in kelly_stats.values() if v.get("has_edge"))
                self.logger.info(f"  Kelly: {len(kelly_stats)} strategies tracked, "
                               f"{edge_count} with proven edge")
            except Exception:
                pass
            try:
                mem_stats = self.trade_memory.get_stats()
                self.logger.info(f"  Memory: {mem_stats['total_trades']} trades stored, "
                               f"{mem_stats['unique_coins']} coins")
            except Exception:
                pass
            try:
                cal_stats = self.calibration.get_all_stats()
                global_ece = self.calibration.get_ece("global")
                ece_str = f"{global_ece:.3f}" if global_ece is not None else "N/A"
                self.logger.info(f"  Calibration: ECE={ece_str} ({self.calibration._quality_label(global_ece)}), "
                               f"{len(cal_stats)} sources tracked")
            except Exception:
                pass
            try:
                llm_stats = self.llm_filter.get_stats()
                self.logger.info(f"  LLM Filter: {llm_stats['total_filtered']} filtered, "
                               f"pass rate={llm_stats['pass_rate']:.0%}")
            except Exception:
                pass
            try:
                sp_stats = self.signal_processor.get_stats()
                self.logger.info(f"  SignalProcessor: {sp_stats['total_in']} in → {sp_stats['total_out']} out "
                               f"(reduction={sp_stats['reduction_rate']:.0%}, "
                               f"culled={sp_stats['culled']}, deduped={sp_stats['deduped']}, "
                               f"conflicts={sp_stats['conflicts_resolved']})")
            except Exception:
                pass
            try:
                inc_stats = self.arena_incubator.get_stats()
                self.logger.info(f"  Incubator: {inc_stats['currently_incubating']} incubating, "
                               f"{inc_stats['total_promoted']} promoted, "
                               f"{inc_stats['total_rejected']} rejected")
            except Exception:
                pass
            try:
                de_stats = self.decision_engine.get_stats()
                self.logger.info(f"  DecisionEngine: {de_stats['total_decisions']} decisions, "
                               f"{de_stats['total_executions']} executions, "
                               f"no-trade rate={de_stats['no_trade_rate']:.0%}")
            except Exception:
                pass
            try:
                if self.multi_scanner:
                    ms_stats = self.multi_scanner.get_stats()
                    cv_stats = ms_stats.get("cross_venue", {})
                    last = ms_stats.get("last_scan", {})
                    self.logger.info(f"  MultiExchange: {ms_stats['venue_count']} venues "
                                   f"({', '.join(ms_stats['venues'])}), "
                                   f"{ms_stats['scan_count']} scans, "
                                   f"{ms_stats['cached_traders']} cached traders")
                    if cv_stats:
                        self.logger.info(f"  CrossVenue: {cv_stats.get('confirmations_checked', 0)} checked, "
                                       f"{cv_stats.get('confirmations_found', 0)} confirmed, "
                                       f"avg score={cv_stats.get('avg_confirmation_score', 0):.3f}, "
                                       f"arbs={cv_stats.get('funding_arbs_found', 0)}")
            except Exception:
                pass

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
