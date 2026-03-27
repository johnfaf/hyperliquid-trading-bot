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
from src.golden_bridge import get_golden_copy_signals, auto_connect_golden_wallets, get_stats as golden_stats
from src.hyperliquid_client import start_websocket, get_api_stats
from src.ws_position_monitor import PositionMonitor
from src.adaptive_bot_detector import AdaptiveBotDetector
from src.sharpe_calculator import calculate_sharpe
from src.regime_strategy_filter import RegimeStrategyFilter
from src import telegram_alerts as tg_alerts
from src.report_exporter import ReportExporter
from src.polymarket_scanner import PolymarketScanner

# ─── Logging Setup ─────────────────────────────────────────────

import re as _re

# Patterns that should NEVER appear in logs — matches common secret formats
_SECRET_PATTERNS = [
    _re.compile(r'(api[_-]?key|api[_-]?secret|private[_-]?key|secret[_-]?key|password|token|authorization)\s*[=:]\s*\S+', _re.IGNORECASE),
    _re.compile(r'0x[a-fA-F0-9]{64}'),  # Ethereum private keys (64 hex chars)
    _re.compile(r'(Bearer|Basic)\s+[A-Za-z0-9+/=_-]{20,}', _re.IGNORECASE),  # Auth headers
]

def _scrub_secrets(text: str) -> str:
    """Remove any secrets that might accidentally appear in log messages."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production.
    Railway, Datadog, ELK, and most log aggregators parse JSON natively.
    Each line is a self-contained JSON object — no multi-line parsing needed.

    Includes secret-scrubbing: API keys, private keys, and auth tokens
    are redacted before they reach the log output.
    """
    def format(self, record):
        import json as _json
        msg = _scrub_secrets(record.getMessage())
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = _scrub_secrets(self.formatException(record.exc_info))
        # Add extra fields if present (e.g. logger.info("msg", extra={"wallet": "0x..."})
        for key in ("wallet", "coin", "action", "latency_ms", "status_code"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return _json.dumps(log_entry, default=str)


def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, f"bot_{datetime.utcnow().strftime('%Y%m%d')}.log")

    # Structured JSON for stdout (Railway / log aggregators)
    json_formatter = JSONFormatter()

    # Human-readable for local file debugging
    text_formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — human-readable for local debugging
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(text_formatter)

    # Console handler — structured JSON for Railway / production
    # Falls back to text if LOG_FORMAT=text (for local dev)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, config.LOG_LEVEL))
    use_json = os.environ.get("LOG_FORMAT", "json").lower() != "text"
    ch.setFormatter(json_formatter if use_json else text_formatter)

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

        # V5: Start WebSocket feed for real-time market data (reduces REST polling)
        try:
            start_websocket(coins=["BTC", "ETH", "SOL", "DOGE", "ARB", "OP"])
            self.logger.info("WebSocket feed started for real-time price data")
        except Exception as e:
            self.logger.warning(f"WebSocket start failed (REST fallback active): {e}")

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

        # V6: Phase 1 signal quality upgrades
        self.adaptive_bot_detector = AdaptiveBotDetector()
        self.regime_strategy_filter = RegimeStrategyFilter()
        self.report_exporter = ReportExporter()

        # V6: Polymarket prediction market scanner
        try:
            self.polymarket = PolymarketScanner()
            self.logger.info("Polymarket scanner initialized")
        except Exception as e:
            self.polymarket = None
            self.logger.warning(f"Polymarket scanner init failed (continuing without): {e}")

        # V6: Real-time WebSocket position monitor for copy trading
        try:
            top_traders = db.get_active_traders()[:20]
            if top_traders:
                self.position_monitor = PositionMonitor(max_signal_queue=500)
                self.position_monitor.start([t["address"] for t in top_traders])
                self.logger.info(f"WebSocket position monitor started for {len(top_traders)} traders")
            else:
                self.position_monitor = None
                self.logger.info("No active traders yet — position monitor will start after first discovery")
        except Exception as e:
            self.position_monitor = None
            self.logger.warning(f"Position monitor init failed (REST copy trading as fallback): {e}")

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

            # Phase 3d2: Polymarket Prediction Market Scan
            polymarket_signals = []
            if self.polymarket:
                self.logger.info("Phase 3d2: Polymarket Scan")
                try:
                    polymarket_signals = self.polymarket.generate_signals(hl_regime=regime_data)
                    sentiment = self.polymarket.get_market_sentiment()
                    self.logger.info(f"  Polymarket: {len(polymarket_signals)} signals, "
                                   f"sentiment={sentiment.get('sentiment', '?')} "
                                   f"(conf={sentiment.get('confidence', 0):.0%}, "
                                   f"markets={sentiment.get('markets_analyzed', 0)})")
                    if polymarket_signals:
                        for sig in polymarket_signals[:3]:
                            self.logger.info(f"  PM signal: {sig['side'].upper()} {sig['coin']} "
                                           f"(conf={sig['confidence']:.0%}) — {sig.get('reason', '')[:60]}")
                except Exception as e:
                    self.logger.warning(f"  Polymarket scan error: {e}")

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

            # V6: Enhanced regime-aware strategy filtering (compatibility matrix)
            if regime_data:
                try:
                    top_strategies = self.regime_strategy_filter.filter(
                        top_strategies, regime_data
                    )
                    report = self.regime_strategy_filter.get_regime_report(
                        top_strategies, regime_data
                    )
                    self.logger.info(f"  Regime filter (V6): {report}")
                except Exception as e:
                    self.logger.debug(f"  V6 regime filter error, using legacy: {e}")
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

            # Phase 4b: Copy trading - mirror top trader positions
            # V6: Drain real-time WebSocket signals first, fall back to REST polling
            self.logger.info("Phase 4b: Copy Trading (V6 — WebSocket + REST)")
            ws_signals = []
            if self.position_monitor:
                ws_signals = self.position_monitor.drain_signals()
                if ws_signals:
                    self.logger.info(f"  WebSocket position monitor: {len(ws_signals)} real-time signals")
            copy_signals = ws_signals + self.copy_trader.scan_top_traders(top_n=10)

            # Inject golden wallet signals (higher confidence, proven profitable)
            try:
                auto_connect_golden_wallets()
                golden_signals = get_golden_copy_signals()
                if golden_signals:
                    self.logger.info(f"  Golden bridge: {len(golden_signals)} signals from verified wallets")
                    copy_signals = golden_signals + copy_signals  # golden first = higher priority
            except Exception as e:
                self.logger.debug(f"  Golden bridge skipped: {e}")

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

            try:
                gs = golden_stats()
                if gs["total_evaluated"] > 0:
                    self.logger.info(f"  Golden Wallets: {gs['golden_wallets']} golden / "
                                   f"{gs['total_evaluated']} evaluated, "
                                   f"{gs['live_connected']} connected to live")
            except Exception:
                pass
            try:
                api_s = get_api_stats()
                self.logger.info(
                    f"  API Manager: {api_s['rest_requests']} REST, "
                    f"{api_s['cache_served']} cached ({api_s['cache_hit_pct']}% hit), "
                    f"{api_s['ws_served']} from WS | "
                    f"bucket: {api_s['bucket']['tokens_available']:.0f} tokens, "
                    f"429s={api_s['bucket']['consecutive_429s']}"
                )
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

            # V6: Enhanced Telegram alerts — daily P&L + top mover detection
            try:
                if tg.is_configured() and self._cycle_count % 24 == 0:  # ~daily with 1hr cycles
                    tg_alerts.send_daily_pnl_summary()
                    self.logger.info("  Sent daily P&L Telegram summary")
                if tg.is_configured() and self._cycle_count % 168 == 0:  # ~weekly
                    tg_alerts.send_weekly_digest()
                    self.logger.info("  Sent weekly Telegram digest")
            except Exception as e:
                self.logger.debug(f"  Enhanced alerts error: {e}")

            # V6: Export HTML report every 24 cycles (~daily)
            try:
                if self._cycle_count % 24 == 0:
                    report_path = self.report_exporter.export_html_report()
                    self.logger.info(f"  HTML report exported: {report_path}")
            except Exception as e:
                self.logger.debug(f"  Report export error: {e}")

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

        # Handle graceful shutdown — backup DB before exit
        def signal_handler(sig, frame):
            self.logger.info("Shutdown signal received. Backing up DB before exit...")
            try:
                backup_to_json()
                self.logger.info("DB backup complete.")
            except Exception as e:
                self.logger.error(f"DB backup failed on shutdown: {e}")
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

def bootstrap_seed_data(logger, days: int = 14):
    """
    Cold-start bootstrap: pull recent fills from top traders to seed the DB
    so Kelly/Scoring/Calibration have real data from day one.
    """
    from src.golden_wallet import init_golden_tables, evaluate_wallet, save_wallet_report, save_wallet_fills

    logger.info(f"Bootstrap mode: seeding DB with last {days} days of top trader data...")
    init_db()
    init_golden_tables()

    discovery = TraderDiscovery()

    # Step 1: Discover traders (abbreviated scan)
    logger.info("Step 1/3: Discovering top traders...")
    discovery_result = discovery.run_discovery_cycle()
    humans = discovery_result.get("human_like", 0)
    logger.info(f"Discovery found {humans} human-like traders")

    if humans == 0:
        logger.warning("No human-like traders found. Try running again later.")
        return

    # Step 2: Evaluate top wallets (golden scan)
    logger.info("Step 2/3: Running golden wallet evaluation...")
    from src.golden_wallet import run_golden_scan
    summary = run_golden_scan(max_wallets=30)
    golden = summary.get("golden", 0)
    logger.info(f"Golden scan: {golden} golden wallets found")

    # Step 3: Initialize paper account if needed
    logger.info("Step 3/3: Initializing paper trading account...")
    account = db.get_paper_account()
    if not account:
        db.create_paper_account(config.PAPER_TRADING_INITIAL_BALANCE)
        logger.info(f"Paper account created: ${config.PAPER_TRADING_INITIAL_BALANCE:,.0f}")

    # Backup
    backup_to_json()
    logger.info(f"Bootstrap complete: {humans} traders, {golden} golden wallets, DB backed up")


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
    parser.add_argument("--bootstrap", action="store_true",
                        help="Cold start: seed DB with top trader data (run once on first deploy)")
    parser.add_argument("--bootstrap-days", type=int, default=14,
                        help="Days of history to bootstrap (default: 14)")
    args = parser.parse_args()

    logger = setup_logging()

    # Log persistence paths so we can verify in Railway logs
    import config as _cfg
    logger.info(f"[PERSISTENCE] persistent_volume={_cfg._HAS_PERSISTENT_VOLUME} DB_PATH={_cfg.DB_PATH} "
                f"HL_BOT_DB_env={os.environ.get('HL_BOT_DB', 'NOT SET')} "
                f"uid={os.getuid()} /data_exists={os.path.isdir('/data')}")

    if args.bootstrap:
        bootstrap_seed_data(logger, days=args.bootstrap_days)
        return

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
