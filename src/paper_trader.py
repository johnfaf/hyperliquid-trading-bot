"""
Paper Trading Simulator (V2)
============================
Simulates trades based on the top-scoring strategies without risking real funds.
Tracks performance to validate strategies before any real deployment.

V2 integration:
  - Signals routed through DecisionFirewall before execution
  - Feature engine enriches signals with market context
  - Agent scoring weights signals by source reliability
  - All signals go through TradeSignal schema validation
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import database as db
from src import hyperliquid_client as hl
from src.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams, signal_from_strategy
from src.decision_firewall import DecisionFirewall
from src.agent_scoring import AgentScorer
from src.features import FeatureEngine
from src.kelly_sizing import KellySizer
from src.trade_memory import TradeMemory
from src.calibration import CalibrationTracker
from src.llm_filter import LLMFilter
import random

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates trading based on identified strategies."""

    def __init__(self, firewall: Optional[DecisionFirewall] = None,
                 agent_scorer: Optional[AgentScorer] = None,
                 feature_engine: Optional[FeatureEngine] = None,
                 kelly_sizer: Optional[KellySizer] = None,
                 trade_memory: Optional[TradeMemory] = None,
                 calibration: Optional[CalibrationTracker] = None,
                 llm_filter: Optional[LLMFilter] = None):
        self.firewall = firewall
        self.agent_scorer = agent_scorer
        self.feature_engine = feature_engine
        self.kelly_sizer = kelly_sizer
        self.trade_memory = trade_memory
        self.calibration = calibration
        self.llm_filter = llm_filter
        self._ensure_account()

    def _ensure_account(self):
        """Initialize paper trading account if it doesn't exist."""
        account = db.get_paper_account()
        if not account:
            db.init_paper_account(config.PAPER_TRADING_INITIAL_BALANCE)
            logger.info(f"Paper account initialized with ${config.PAPER_TRADING_INITIAL_BALANCE:,.2f}")

    def get_account_summary(self) -> Dict:
        """Get current paper trading account summary."""
        account = db.get_paper_account()
        open_trades = db.get_open_paper_trades()
        closed_trades = db.get_paper_trade_history(limit=100)

        # Calculate unrealized PnL from open trades
        unrealized_pnl = 0
        mids = hl.get_all_mids() or {}
        for trade in open_trades:
            current_price = float(mids.get(trade["coin"], trade["entry_price"]))
            if trade["side"] == "long":
                trade_pnl = (current_price - trade["entry_price"]) * trade["size"] * trade["leverage"]
            else:
                trade_pnl = (trade["entry_price"] - current_price) * trade["size"] * trade["leverage"]
            unrealized_pnl += trade_pnl

        total_equity = account["balance"] + unrealized_pnl if account else 0

        return {
            "balance": account["balance"] if account else 0,
            "total_pnl": account["total_pnl"] if account else 0,
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_equity": round(total_equity, 2),
            "total_trades": account["total_trades"] if account else 0,
            "winning_trades": account["winning_trades"] if account else 0,
            "win_rate": (account["winning_trades"] / account["total_trades"]
                        if account and account["total_trades"] > 0 else 0),
            "open_positions": len(open_trades),
            "roi_pct": round(((total_equity / config.PAPER_TRADING_INITIAL_BALANCE) - 1) * 100, 2),
        }

    def execute_strategy_signals(self, strategies: List[Dict], exchange_agg=None,
                                  options_scanner=None,
                                  regime_data: Optional[Dict] = None,
                                  arena=None) -> List[Dict]:
        """
        Generate and execute paper trades based on top strategies.

        V2 Pipeline:
        1. Generate raw signal dict from strategy
        2. Enrich with feature engine context
        3. Apply multi-exchange volume confirmation
        4. Apply options flow confirmation
        5. Convert to TradeSignal schema
        6. Apply agent scoring weights
        7. Route through DecisionFirewall
        8. Execute approved signals
        """
        account = db.get_paper_account()
        if not account:
            return []

        open_trades = db.get_open_paper_trades()
        executed = []

        # Get current prices
        mids = hl.get_all_mids() or {}

        # Pre-compute features for relevant coins if feature engine available
        coin_features = {}
        if self.feature_engine:
            try:
                from src.regime_detector import RegimeDetector
                for coin in set(["BTC", "ETH", "SOL"]):
                    try:
                        # Fetch candles for feature computation
                        import requests
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
                                features = self.feature_engine.compute(coin, candles)
                                coin_features[coin] = features
                                logger.debug(f"Features {coin}: score={features.overall_score:+.2f}, "
                                           f"rsi={features.rsi:.0f}, vol={features.volatility:.3f}")
                    except Exception as e:
                        logger.debug(f"Feature computation for {coin}: {e}")
            except Exception as e:
                logger.debug(f"Feature engine error: {e}")

        # Build trade signals from strategies
        raw_signals = []
        for strategy in strategies:
            try:
                # Check if we already have a position from this strategy
                existing = [t for t in open_trades if t["strategy_id"] == strategy.get("id")]
                if existing:
                    continue

                # Generate signal from strategy
                signal = self._generate_signal(strategy, mids, regime_data=regime_data)
                if not signal:
                    continue

                # Enrich with feature engine data
                coin = signal["coin"]
                if coin in coin_features:
                    feat = coin_features[coin]
                    # If features strongly oppose the signal direction, reduce confidence
                    if signal["side"] == "long" and feat.overall_score < -0.3:
                        signal["confidence"] *= 0.7
                        logger.debug(f"Features bearish for long {coin} (score={feat.overall_score:+.2f})")
                    elif signal["side"] == "short" and feat.overall_score > 0.3:
                        signal["confidence"] *= 0.7
                        logger.debug(f"Features bullish for short {coin} (score={feat.overall_score:+.2f})")
                    elif (signal["side"] == "long" and feat.overall_score > 0.3) or \
                         (signal["side"] == "short" and feat.overall_score < -0.3):
                        signal["confidence"] = min(signal["confidence"] * 1.15, 1.0)
                        logger.debug(f"Features confirm {signal['side']} {coin}")

                    # Add feature context to signal
                    signal["features"] = {
                        "overall_score": feat.overall_score,
                        "rsi": feat.rsi,
                        "rsi_signal": feat.rsi_signal,
                        "volume_trend": feat.volume_trend,
                        "funding_signal": feat.funding_signal,
                    }

                # Multi-exchange volume confirmation
                if exchange_agg:
                    try:
                        confirmed, vol_confidence = exchange_agg.get_volume_confirmation(
                            signal["coin"], signal["side"]
                        )
                        if not confirmed:
                            logger.debug(f"Volume rejects {signal['side']} {signal['coin']} "
                                        f"(confidence={vol_confidence:.2f})")
                            continue
                        signal["confidence"] = signal.get("confidence", 0.5) * (0.5 + vol_confidence * 0.5)
                        signal["volume_confirmed"] = True
                    except Exception:
                        pass

                # Options flow confirmation
                if options_scanner:
                    try:
                        flow_signal = options_scanner.get_flow_signal(signal["coin"])
                        if flow_signal:
                            if flow_signal["side"] == signal["side"]:
                                boost = 1.0 + flow_signal["confidence"] * 0.3
                                signal["confidence"] = min(signal.get("confidence", 0.5) * boost, 1.0)
                                signal["options_flow_aligned"] = True
                                logger.debug(f"Options flow confirms {signal['side']} {signal['coin']}")
                            else:
                                signal["confidence"] = signal.get("confidence", 0.5) * 0.7
                                signal["options_flow_aligned"] = False
                    except Exception:
                        pass

                # Attach strategy reference
                signal["strategy"] = strategy
                raw_signals.append(signal)

            except Exception as e:
                logger.error(f"Error generating signal for {strategy.get('name', '?')}: {e}")

        # Convert to TradeSignal objects and apply agent scoring
        trade_signals = []
        for sig in raw_signals:
            try:
                trade_signal = TradeSignal(
                    coin=sig["coin"],
                    side=SignalSide(sig["side"]),
                    confidence=sig.get("confidence", 0.5),
                    source=SignalSource.STRATEGY,
                    reason=f"Strategy: {sig['strategy'].get('name', '?')} ({sig.get('strategy_type', '')})",
                    strategy_id=sig["strategy"].get("id"),
                    strategy_type=sig.get("strategy_type", ""),
                    entry_price=sig["price"],
                    leverage=sig["leverage"],
                    position_pct=sig["size"] * sig["price"] / (db.get_paper_account() or {}).get("balance", 10000),
                    risk=RiskParams(
                        stop_loss_pct=config.PAPER_TRADING_STOP_LOSS_PCT,
                        take_profit_pct=config.PAPER_TRADING_TAKE_PROFIT_PCT,
                        max_leverage=config.PAPER_TRADING_MAX_LEVERAGE,
                    ),
                    regime=regime_data.get("overall_regime", "") if regime_data else "",
                    regime_size_modifier=sig["strategy"].get("regime_size_modifier", 1.0),
                    options_flow_aligned=sig.get("options_flow_aligned"),
                    volume_confirmed=sig.get("volume_confirmed"),
                )
                trade_signals.append((trade_signal, sig))
            except Exception as e:
                logger.debug(f"Error creating TradeSignal: {e}")

        # Apply agent scoring weights (higher-performing sources get boosted)
        if self.agent_scorer and trade_signals:
            signals_only = [ts for ts, _ in trade_signals]
            self.agent_scorer.apply_weights_to_signals(signals_only)
            logger.info(f"Agent scoring applied to {len(signals_only)} signals")

        # Route through Decision Firewall
        for trade_signal, sig in trade_signals:
            try:
                # Use firewall if available, else fall back to legacy risk checks
                if self.firewall:
                    passed, reason = self.firewall.validate(
                        trade_signal,
                        regime_data=regime_data,
                        open_positions=open_trades,
                    )
                    if not passed:
                        logger.debug(f"Firewall rejected {sig['side']} {sig['coin']}: {reason}")
                        continue
                    logger.debug(f"Firewall approved {sig['side']} {sig['coin']} "
                                f"(confidence={trade_signal.confidence:.0%})")
                else:
                    # Legacy fallback
                    if not self._check_risk_limits(account, sig, open_trades):
                        logger.debug(f"Risk limit hit, skipping {sig['coin']}")
                        continue

                # Arena consensus vote (multi-agent debate)
                if arena:
                    try:
                        feature_ctx = sig.get("features", {})
                        approved, consensus_conf = arena.get_consensus_on_signal(
                            trade_signal, features=feature_ctx
                        )
                        if not approved:
                            logger.debug(f"Arena consensus REJECTED {sig['side']} {sig['coin']}")
                            continue
                        # Use consensus-adjusted confidence
                        trade_signal.confidence = consensus_conf
                    except Exception as e:
                        logger.debug(f"Arena consensus error: {e}")

                # Calibration adjustment — correct miscalibrated confidence
                if self.calibration:
                    try:
                        source_key = f"strategy:{sig.get('strategy_type', 'unknown')}"
                        adjusted = self.calibration.get_adjustment_factor(
                            source_key, trade_signal.confidence
                        )
                        if abs(adjusted - trade_signal.confidence) > 0.05:
                            logger.debug(f"Calibration adjust {sig['coin']}: "
                                       f"{trade_signal.confidence:.2f} → {adjusted:.2f}")
                        trade_signal.confidence = adjusted
                    except Exception as e:
                        logger.debug(f"Calibration error: {e}")

                # Trade memory lookup — check similar past trades
                if self.trade_memory:
                    try:
                        memory_result = self.trade_memory.find_similar(
                            features=sig.get("features", {}),
                            coin=sig["coin"],
                            side=sig["side"],
                            top_k=8,
                        )
                        if memory_result.recommendation == "avoid":
                            logger.debug(f"Memory BLOCKED {sig['side']} {sig['coin']}: {memory_result.reason}")
                            continue
                        elif memory_result.recommendation == "caution":
                            trade_signal.confidence *= 0.8
                            logger.debug(f"Memory caution for {sig['coin']}: {memory_result.reason}")
                    except Exception as e:
                        logger.debug(f"Trade memory error: {e}")
                        memory_result = None

                # LLM Filter — final contextual check
                if self.llm_filter:
                    try:
                        llm_context = {
                            "regime_data": regime_data,
                            "memory_result": memory_result.to_dict() if hasattr(memory_result, 'to_dict') and memory_result else None,
                            "open_positions": open_trades,
                            "all_signals": raw_signals,
                        }
                        llm_approved, llm_conf, llm_reason = self.llm_filter.filter(sig, llm_context)
                        if not llm_approved:
                            logger.debug(f"LLM filter BLOCKED {sig['side']} {sig['coin']}: {llm_reason}")
                            continue
                        trade_signal.confidence = llm_conf
                    except Exception as e:
                        logger.debug(f"LLM filter error: {e}")

                # Kelly sizing — mathematically optimal position size
                if self.kelly_sizer:
                    try:
                        sizing = self.kelly_sizer.get_sizing(
                            strategy_key=sig.get("strategy_type", "unknown"),
                            account_balance=account["balance"],
                            signal_confidence=trade_signal.confidence,
                        )
                        trade_signal.position_pct = sizing.position_pct
                        if sizing.has_edge:
                            logger.debug(f"Kelly [{sig.get('strategy_type')}]: {sizing.position_pct:.1%} "
                                       f"(WR={sizing.win_rate:.0%}, R:R={sizing.reward_risk_ratio:.2f})")
                    except Exception as e:
                        logger.debug(f"Kelly sizing error: {e}")

                # Record signal with agent scorer
                signal_id = ""
                if self.agent_scorer:
                    source_key = self.agent_scorer.get_source_key(trade_signal)
                    signal_id = self.agent_scorer.record_signal(source_key, {
                        "coin": sig["coin"],
                        "side": sig["side"],
                        "confidence": trade_signal.confidence,
                    })

                # Execute the paper trade
                trade = self._execute_paper_trade(account, sig["strategy"], sig)
                if trade:
                    trade["signal_id"] = signal_id
                    trade["strategy_type"] = sig.get("strategy_type", "")
                    executed.append(trade)
                    open_trades.append(trade)

            except Exception as e:
                logger.error(f"Error in V2 pipeline for {sig.get('coin', '?')}: {e}")

        if executed:
            logger.info(f"Executed {len(executed)} paper trades (V2 pipeline)")

        return executed

    def _generate_signal(self, strategy: Dict, mids: Dict,
                         regime_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate a trading signal from a strategy.
        Maps strategy types to concrete trade parameters.
        """
        import json
        params = strategy.get("parameters", "{}")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                params = {}

        strategy_type = strategy.get("strategy_type", strategy.get("type", ""))
        score = strategy.get("current_score", 0)

        # Only trade strategies with decent scores
        if score < 0.3:
            return None

        # Get target coins from strategy parameters
        coins = params.get("coins", [])
        if not coins:
            coins = params.get("coins_traded", [])
        if isinstance(coins, str):
            coins = [coins]

        # If strategy has no specific coins, pick from top liquid coins
        # to diversify across many assets instead of all piling into BTC
        if not coins:
            import random
            TOP_COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "ARB",
                         "OP", "SUI", "APT", "INJ", "SEI", "TIA", "JUP",
                         "WIF", "PEPE", "ONDO", "RENDER", "FET", "NEAR"]
            available = [c for c in TOP_COINS if c in mids]
            if available:
                # Use strategy ID as seed for deterministic but diverse selection
                strat_id = hash(str(strategy.get("id", random.random())))
                coins = [available[strat_id % len(available)]]
            else:
                coins = ["BTC", "ETH"]

        # Pick the first available coin with a price
        target_coin = None
        target_price = 0
        for coin in coins:
            if coin in mids:
                target_coin = coin
                target_price = float(mids[coin])
                break

        if not target_coin or target_price == 0:
            # Fallback to BTC
            target_coin = "BTC"
            target_price = float(mids.get("BTC", 0))
            if target_price == 0:
                return None

        # Determine direction — regime-aware for ambiguous types.
        # Use pre-computed side from decision engine when available.
        pre_decided = strategy.get("side", strategy.get("_decision_side", ""))

        # Regime direction bias: when regime is confident use it as the default
        # for undirected strategies instead of blindly defaulting to "long".
        _regime_str  = (regime_data or {}).get("overall_regime", "unknown")
        _regime_conf = (regime_data or {}).get("overall_confidence", 0.0)
        if _regime_str == "trending_down" and _regime_conf >= 0.6:
            regime_default = "short"
        elif _regime_str == "trending_up" and _regime_conf >= 0.6:
            regime_default = "long"
        else:
            regime_default = params.get("direction", params.get("bias", "long"))

        if pre_decided in ("long", "short"):
            side = pre_decided
        elif strategy_type == "momentum_long":
            side = "long"
        elif strategy_type in ("momentum_short", "contrarian"):
            side = "short"
        elif strategy_type == "funding_arb":
            side = "short"  # Typically short to earn positive funding
        else:
            # breakout, trend_following, swing_trading, concentrated_bet,
            # mean_reversion, scalping, delta_neutral, etc.
            # — follow regime when confident (e.g. downside breakout = short in trending_down),
            # else use stored param
            side = params.get("direction") or regime_default

        # Determine leverage (capped by config)
        leverage = min(
            params.get("avg_leverage", 2),
            config.PAPER_TRADING_MAX_LEVERAGE
        )
        leverage = max(1, leverage)

        # Calculate position size
        account = db.get_paper_account()
        max_position = account["balance"] * config.PAPER_TRADING_MAX_POSITION_PCT
        size_usd = max_position * (score / 1.0)  # Scale by confidence
        size = size_usd / target_price

        # Stop loss and take profit
        if side == "long":
            stop_loss = target_price * (1 - config.PAPER_TRADING_STOP_LOSS_PCT / leverage)
            take_profit = target_price * (1 + config.PAPER_TRADING_TAKE_PROFIT_PCT / leverage)
        else:
            stop_loss = target_price * (1 + config.PAPER_TRADING_STOP_LOSS_PCT / leverage)
            take_profit = target_price * (1 - config.PAPER_TRADING_TAKE_PROFIT_PCT / leverage)

        return {
            "coin": target_coin,
            "side": side,
            "price": target_price,
            "size": size,
            "leverage": leverage,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "strategy_type": strategy_type,
            "confidence": score,
        }

    def _check_risk_limits(self, account: Dict, signal: Dict, open_trades: List) -> bool:
        """Check if a new trade passes risk management rules."""
        # CRITICAL: No conflicting sides on same asset (no long+short on same coin)
        for t in open_trades:
            if t["coin"] == signal["coin"] and t.get("side") != signal["side"]:
                logger.debug(f"Risk: conflicting side for {signal['coin']} "
                            f"(have {t.get('side')}, want {signal['side']})")
                return False

        # Max position count — allow up to 20 simultaneous positions
        if len(open_trades) >= 20:
            logger.debug(f"Risk: max positions ({len(open_trades)}/20)")
            return False

        # Max positions per coin — allow up to 3 per coin (same direction only)
        coin_positions = sum(1 for t in open_trades if t["coin"] == signal["coin"])
        if coin_positions >= 3:
            logger.debug(f"Risk: max positions for {signal['coin']} ({coin_positions}/3)")
            return False

        # Max exposure per coin — 50% of balance
        coin_exposure = sum(
            t["size"] * t["entry_price"] * t["leverage"]
            for t in open_trades if t["coin"] == signal["coin"]
        )
        max_coin_exposure = account["balance"] * 0.50
        new_exposure = signal["size"] * signal["price"] * signal["leverage"]
        if coin_exposure + new_exposure > max_coin_exposure:
            logger.debug(f"Risk: coin exposure for {signal['coin']} ${coin_exposure+new_exposure:,.0f} > ${max_coin_exposure:,.0f}")
            return False

        # Max total exposure — 500% of balance (leveraged)
        total_exposure = sum(
            t["size"] * t["entry_price"] * t["leverage"]
            for t in open_trades
        )
        max_total_exposure = account["balance"] * 5.0
        if total_exposure + new_exposure > max_total_exposure:
            logger.info(f"Risk: total exposure ${total_exposure+new_exposure:,.0f} > ${max_total_exposure:,.0f}")
            return False

        # Minimum balance remaining
        if account["balance"] < config.PAPER_TRADING_INITIAL_BALANCE * 0.05:
            return False

        return True

    @staticmethod
    def _apply_slippage(price: float, side: str, is_entry: bool = True) -> float:
        """
        Simulate realistic slippage for paper trading.

        Real market orders experience slippage from:
        - Spread crossing (taker fee already modeled elsewhere)
        - Orderbook depth (larger orders move price more)
        - Latency (price moves between signal and fill)

        We apply 0.01% to 0.05% adverse slippage, randomized.
        Entry longs get a higher price, entry shorts get a lower price.
        Exit is the reverse.
        """
        slippage_bps = random.uniform(1, 5)  # 0.01% to 0.05%
        slippage_pct = slippage_bps / 10_000

        if is_entry:
            # Entry: you pay more (long) or receive less (short)
            if side == "long":
                return price * (1 + slippage_pct)
            else:
                return price * (1 - slippage_pct)
        else:
            # Exit: you receive less (closing long) or pay more (closing short)
            if side == "long":
                return price * (1 - slippage_pct)
            else:
                return price * (1 + slippage_pct)

    def _execute_paper_trade(self, account: Dict, strategy: Dict, signal: Dict) -> Optional[Dict]:
        """Execute a paper trade and record it (with slippage simulation)."""
        try:
            # Apply entry slippage — paper trades should reflect realistic fills
            slipped_price = self._apply_slippage(signal["price"], signal["side"], is_entry=True)
            logger.debug(f"Slippage: {signal['coin']} entry {signal['price']:.2f} → {slipped_price:.2f} "
                        f"({signal['side']})")

            trade_id = db.open_paper_trade(
                strategy_id=strategy.get("id"),
                coin=signal["coin"],
                side=signal["side"],
                entry_price=slipped_price,
                size=signal["size"],
                leverage=signal["leverage"],
                stop_loss=signal["stop_loss"],
                take_profit=signal["take_profit"],
                metadata={
                    "strategy_type": signal.get("strategy_type", ""),
                    "confidence": signal.get("confidence", 0),
                    "signal_id": signal.get("signal_id", ""),
                    "features": signal.get("features", {}),
                    "options_flow_aligned": signal.get("options_flow_aligned"),
                    "volume_confirmed": signal.get("volume_confirmed"),
                },
            )

            logger.info(
                f"Paper trade opened: {signal['side'].upper()} {signal['coin']} "
                f"@ ${signal['price']:,.2f} | size={signal['size']:.4f} | "
                f"leverage={signal['leverage']}x | SL=${signal['stop_loss']:,.2f} | "
                f"TP=${signal['take_profit']:,.2f}"
            )

            return {
                "id": trade_id,
                "coin": signal["coin"],
                "side": signal["side"],
                "entry_price": signal["price"],
                "size": signal["size"],
                "leverage": signal["leverage"],
                "strategy_id": strategy.get("id"),
            }
        except Exception as e:
            logger.error(f"Error opening paper trade: {e}")
            return None

    def check_open_positions(self) -> List[Dict]:
        """
        Check all open paper positions against current prices.
        Close any that hit stop-loss, take-profit, or trailing stop.
        Also implements trailing stop: if price moves 3%+ in our favor,
        ratchet the stop-loss to lock in at least breakeven.
        """
        open_trades = db.get_open_paper_trades()
        if not open_trades:
            return []

        mids = hl.get_all_mids() or {}
        closed = []

        for trade in open_trades:
            current_price = float(mids.get(trade["coin"], 0))
            if current_price == 0:
                continue

            should_close = False
            close_reason = ""

            # --- Trailing stop logic ---
            # If price has moved 3%+ in our favor, tighten the stop to lock in profits
            entry = trade["entry_price"]
            leverage = trade.get("leverage", 1)
            sl = trade["stop_loss"]

            if not entry or entry <= 0:
                continue

            if trade["side"] == "long":
                move_pct = (current_price - entry) / entry
                if move_pct >= 0.03 and sl and sl < entry:
                    # Move SL to breakeven + 0.5%
                    new_sl = entry * 1.005
                    if new_sl > sl:
                        self._update_stop_loss(trade["id"], new_sl)
                        sl = new_sl
                elif move_pct >= 0.06 and sl:
                    # Trail SL at 2.5% below current price
                    trail_sl = current_price * (1 - 0.025 / max(leverage, 1))
                    if trail_sl > sl:
                        self._update_stop_loss(trade["id"], trail_sl)
                        sl = trail_sl
            else:
                move_pct = (entry - current_price) / entry
                if move_pct >= 0.03 and sl and sl > entry:
                    new_sl = entry * 0.995
                    if new_sl < sl:
                        self._update_stop_loss(trade["id"], new_sl)
                        sl = new_sl
                elif move_pct >= 0.06 and sl:
                    trail_sl = current_price * (1 + 0.025 / max(leverage, 1))
                    if trail_sl < sl:
                        self._update_stop_loss(trade["id"], trail_sl)
                        sl = trail_sl

            # Check stop loss (using potentially updated SL)
            if sl:
                if trade["side"] == "long" and current_price <= sl:
                    should_close = True
                    close_reason = "trailing_stop" if sl > trade["stop_loss"] else "stop_loss"
                elif trade["side"] == "short" and current_price >= sl:
                    should_close = True
                    close_reason = "trailing_stop" if sl < trade["stop_loss"] else "stop_loss"

            # Check take profit
            if trade["take_profit"] and not should_close:
                if trade["side"] == "long" and current_price >= trade["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"
                elif trade["side"] == "short" and current_price <= trade["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"

            # Time-based exit: close positions older than 24 hours
            if not should_close and trade.get("opened_at"):
                try:
                    opened = datetime.fromisoformat(trade["opened_at"])
                    age_hours = (datetime.utcnow() - opened).total_seconds() / 3600
                    if age_hours > 24:
                        should_close = True
                        close_reason = "time_exit_24h"
                except (ValueError, TypeError):
                    pass

            if should_close:
                # Apply exit slippage for realistic PnL calculation
                slipped_exit = self._apply_slippage(current_price, trade["side"], is_entry=False)
                pnl = self._calculate_pnl(trade, slipped_exit)
                db.close_paper_trade(trade["id"], slipped_exit, pnl)

                # Update account
                account = db.get_paper_account()
                new_balance = account["balance"] + pnl
                new_total_pnl = account["total_pnl"] + pnl
                new_total_trades = account["total_trades"] + 1
                new_winning = account["winning_trades"] + (1 if pnl > 0 else 0)
                db.update_paper_account(new_balance, new_total_pnl, new_total_trades, new_winning)

                logger.info(
                    f"Paper trade closed ({close_reason}): {trade['side'].upper()} {trade['coin']} "
                    f"entry=${trade['entry_price']:,.2f} exit=${current_price:,.2f} "
                    f"PnL=${pnl:,.2f}"
                )

                # Record outcome with V2 scoring systems
                trade_meta = {}
                try:
                    import json
                    trade_meta = json.loads(trade.get("metadata", "{}") or "{}")
                except Exception:
                    pass

                strategy_type = trade_meta.get("strategy_type", "unknown")
                signal_id = trade_meta.get("signal_id", "")

                # Feed outcome to agent scorer
                if self.agent_scorer:
                    try:
                        source_key = f"strategy:{strategy_type}"
                        return_pct = pnl / max(trade["entry_price"] * max(trade["size"], 1e-8) * max(trade.get("leverage", 1), 1), 1e-8)
                        self.agent_scorer.record_outcome(source_key, signal_id, pnl, return_pct)
                    except Exception as e:
                        logger.debug(f"Agent scorer outcome error: {e}")

                # Feed outcome to decision firewall daily tracker
                if self.firewall:
                    try:
                        self.firewall.record_trade_outcome(trade["coin"], pnl)
                    except Exception:
                        pass

                # Feed outcome to Kelly sizer
                if self.kelly_sizer:
                    try:
                        self.kelly_sizer.record_outcome(
                            strategy_key=strategy_type,
                            pnl=pnl,
                            entry_price=trade["entry_price"],
                            size=trade["size"],
                            leverage=trade.get("leverage", 1),
                        )
                    except Exception:
                        pass

                # Feed outcome to calibration tracker
                if self.calibration:
                    try:
                        source_key = f"strategy:{strategy_type}"
                        predicted_conf = trade_meta.get("confidence", 0.5)
                        self.calibration.record(
                            source_key=source_key,
                            predicted_confidence=predicted_conf,
                            actual_win=pnl > 0,
                            pnl=pnl,
                            coin=trade["coin"],
                            side=trade["side"],
                        )
                    except Exception:
                        pass

                # Feed outcome to trade memory
                if self.trade_memory:
                    try:
                        return_pct = pnl / (trade["entry_price"] * trade["size"] * trade.get("leverage", 1)) if trade["entry_price"] > 0 else 0
                        self.trade_memory.record_trade(
                            trade_id=str(trade["id"]),
                            coin=trade["coin"],
                            side=trade["side"],
                            strategy_type=strategy_type,
                            entry_price=trade["entry_price"],
                            exit_price=current_price,
                            pnl=pnl,
                            return_pct=return_pct,
                            opened_at=trade.get("opened_at", ""),
                            closed_at=datetime.utcnow().isoformat(),
                            confidence=trade_meta.get("confidence", 0),
                            source="strategy",
                            regime=trade_meta.get("regime", ""),
                            setup_type=trade_meta.get("setup_type", strategy_type),
                            features=trade_meta.get("features", {}),
                        )
                    except Exception:
                        pass

                closed.append({
                    "trade_id": trade["id"],
                    "coin": trade["coin"],
                    "side": trade["side"],
                    "pnl": pnl,
                    "reason": close_reason,
                    "strategy_type": strategy_type,
                    "signal_id": signal_id,
                    "exit_price": current_price,
                })

        return closed

    def _update_stop_loss(self, trade_id: int, new_sl: float):
        """Update stop loss for a trade (trailing stop)."""
        try:
            with db.get_connection() as conn:
                conn.execute(
                    "UPDATE paper_trades SET stop_loss = ? WHERE id = ? AND status = 'open'",
                    (round(new_sl, 2), trade_id)
                )
            logger.info(f"Trailing stop updated for trade {trade_id}: new SL=${new_sl:,.2f}")
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")

    def _calculate_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate PnL for a trade."""
        if trade["side"] == "long":
            pnl = (exit_price - trade["entry_price"]) * trade["size"] * trade["leverage"]
        else:
            pnl = (trade["entry_price"] - exit_price) * trade["size"] * trade["leverage"]
        return round(pnl, 2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    db.init_db()
    trader = PaperTrader()
    summary = trader.get_account_summary()
    print(f"\nPaper Account Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
