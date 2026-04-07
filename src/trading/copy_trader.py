"""
Copy Trading Engine (V2)
========================
Monitors top traders' live positions on Hyperliquid and mirrors their new trades
as paper trades. Detects when top traders open/close positions and generates signals.

V2: Signals routed through DecisionFirewall and tracked by AgentScorer.
"""
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import database as db
from src.data import hyperliquid_client as hl
from src.signals.signal_schema import TradeSignal, signal_from_copy_trade
from src.signals.decision_firewall import DecisionFirewall
from src.signals.agent_scoring import AgentScorer
from src.signals.kelly_sizing import KellySizer
from src.signals.portfolio_sizer import PortfolioSizer
from src.signals.source_allocator import SourceBudgetAllocator
from src.trading.trade_memory import TradeMemory
from src.trading.portfolio_rotation import PortfolioRotationManager
from src.signals.calibration import CalibrationTracker

logger = logging.getLogger(__name__)


class CopyTrader:
    """Monitors top traders and mirrors their position changes."""

    def __init__(self, firewall: Optional[DecisionFirewall] = None,
                 agent_scorer: Optional[AgentScorer] = None,
                 kelly_sizer: Optional[KellySizer] = None,
                 portfolio_sizer: Optional[PortfolioSizer] = None,
                 source_allocator: Optional[SourceBudgetAllocator] = None,
                 trade_memory: Optional[TradeMemory] = None,
                 calibration: Optional[CalibrationTracker] = None,
                 regime_forecaster: Optional[object] = None):
        # Cache of last-known positions per trader: {address: {coin: position_dict}}
        self._position_cache: Dict[str, Dict[str, Dict]] = {}
        self._copy_count = 0
        self.firewall = firewall
        self.agent_scorer = agent_scorer
        self.kelly_sizer = kelly_sizer
        self.portfolio_sizer = portfolio_sizer
        self.source_allocator = source_allocator
        self.trade_memory = trade_memory
        self.calibration = calibration
        self.regime_forecaster = regime_forecaster
        self.rotation_manager = PortfolioRotationManager()
        self._closed_events: List[Dict] = []

    def scan_top_traders(self, top_n: int = 10) -> List[Dict]:
        """
        Scan the top N traders by PnL, detect new/changed positions,
        and return copy-trade signals.
        """
        traders = db.get_active_traders()[:top_n]
        if not traders:
            return []

        signals = []
        mids = hl.get_all_mids() or {}

        for trader in traders:
            try:
                addr = trader["address"]
                state = hl.get_user_state(addr)
                if not state:
                    continue

                current_positions = {}
                for pos in state["positions"]:
                    if pos["size"] > 0:
                        current_positions[pos["coin"]] = pos

                # Compare with cached positions to find changes
                cached = self._position_cache.get(addr, {})
                new_signals = self._detect_position_changes(
                    addr, cached, current_positions, trader, mids
                )
                signals.extend(new_signals)

                # Update cache
                self._position_cache[addr] = current_positions

            except Exception as e:
                logger.debug(f"Error scanning trader {trader.get('address', '?')[:10]}: {e}")

        if signals:
            logger.info(f"Copy-trader detected {len(signals)} signals from top {top_n} traders")

        return signals

    def _detect_position_changes(
        self,
        address: str,
        old_positions: Dict[str, Dict],
        new_positions: Dict[str, Dict],
        trader: Dict,
        mids: Dict,
    ) -> List[Dict]:
        """Detect new, closed, or significantly changed positions."""
        signals = []
        old_coins = set(old_positions.keys())
        new_coins = set(new_positions.keys())

        # New positions opened by the trader
        for coin in new_coins - old_coins:
            pos = new_positions[coin]
            price = float(mids.get(coin, pos["entry_price"]))
            if price <= 0:
                continue

            signals.append({
                "type": "copy_open",
                "coin": coin,
                "side": pos["side"],
                "price": price,
                "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                "source_trader": address[:10],
                "source_pnl": trader.get("total_pnl", 0),
                "confidence": min(0.9, 0.5 + trader.get("win_rate", 0) * 0.5),
            })

        # Positions closed by the trader (they exited)
        for coin in old_coins - new_coins:
            signals.append({
                "type": "copy_close",
                "coin": coin,
                "source_trader": address[:10],
            })

        # Significantly increased positions (scaling in)
        for coin in old_coins & new_coins:
            old_size = old_positions[coin]["size"]
            new_size = new_positions[coin]["size"]
            if new_size > old_size * 1.5:  # 50%+ increase
                pos = new_positions[coin]
                price = float(mids.get(coin, pos["entry_price"]))
                if price <= 0:
                    continue
                signals.append({
                    "type": "copy_scale_in",
                    "coin": coin,
                    "side": pos["side"],
                    "price": price,
                    "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                    "source_trader": address[:10],
                    "source_pnl": trader.get("total_pnl", 0),
                    "confidence": min(0.85, 0.4 + trader.get("win_rate", 0) * 0.5),
                })

        # Side flips (trader reversed position)
        for coin in old_coins & new_coins:
            if old_positions[coin]["side"] != new_positions[coin]["side"]:
                pos = new_positions[coin]
                price = float(mids.get(coin, pos["entry_price"]))
                if price <= 0:
                    continue
                signals.append({
                    "type": "copy_flip",
                    "coin": coin,
                    "side": pos["side"],
                    "price": price,
                    "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                    "source_trader": address[:10],
                    "source_pnl": trader.get("total_pnl", 0),
                    "confidence": min(0.95, 0.6 + trader.get("win_rate", 0) * 0.5),
                })

        return signals

    def _apply_regime_weight(self, signal: Dict, coin: str) -> Dict:
        """
        Apply regime-based weighting to copy signal confidence.

        Reduces copy aggressiveness during crash regimes and increases during bullish regimes.
        Gracefully degrades if regime_forecaster is unavailable.

        Args:
            signal: Copy signal dict
            coin: Coin symbol

        Returns:
            Modified signal dict with adjusted confidence
        """
        if not self.regime_forecaster:
            return signal

        try:
            regime_info = self.regime_forecaster.predict_regime(coin)
        except Exception as e:
            logger.debug(f"Failed to fetch regime data for {coin}: {e}")
            return signal

        regime = regime_info.get("regime", "neutral")
        original_confidence = signal.get("confidence", 0.5)

        # Apply regime multiplier to confidence
        if regime == "crash":
            # Copy much less aggressively during crashes (30% of normal confidence)
            adjusted_confidence = original_confidence * 0.3
            logger.info(
                f"REGIME WEIGHT: crash detected for {coin}, "
                f"reducing copy confidence {original_confidence:.2f} → {adjusted_confidence:.2f}"
            )
        elif regime == "neutral":
            # Slightly reduce during neutral regimes (70% of normal confidence)
            adjusted_confidence = original_confidence * 0.7
            logger.debug(
                f"REGIME WEIGHT: neutral regime for {coin}, "
                f"reducing copy confidence {original_confidence:.2f} → {adjusted_confidence:.2f}"
            )
        elif regime == "bullish":
            # Increase aggressiveness during bullish regimes (120% of normal, capped at 1.0)
            adjusted_confidence = min(original_confidence * 1.2, 1.0)
            if adjusted_confidence > original_confidence:
                logger.info(
                    f"REGIME WEIGHT: bullish detected for {coin}, "
                    f"boosting copy confidence {original_confidence:.2f} → {adjusted_confidence:.2f}"
                )
        else:
            adjusted_confidence = original_confidence

        signal["confidence"] = adjusted_confidence
        return signal

    @staticmethod
    def _sync_signal_from_trade_signal(
        trade_signal: TradeSignal,
        signal: Dict,
        *,
        account_balance: float,
    ) -> bool:
        price = float(signal.get("price", trade_signal.entry_price) or trade_signal.entry_price or 0.0)
        if price <= 0 or account_balance <= 0:
            return False

        position_pct = max(float(trade_signal.position_pct or 0.0), 0.0)
        if position_pct <= 0:
            return False

        signal["position_pct"] = position_pct
        signal["size"] = (account_balance * position_pct) / price
        signal["stop_loss_pct"] = float(trade_signal.risk.stop_loss_pct or 0.0)
        signal["take_profit_pct"] = float(trade_signal.risk.take_profit_pct or 0.0)
        signal["time_limit_hours"] = float(trade_signal.risk.time_limit_hours or 24.0)
        if signal.get("side") == "long":
            signal["stop_loss"] = price * (1 - signal["stop_loss_pct"])
            signal["take_profit"] = price * (1 + signal["take_profit_pct"])
        else:
            signal["stop_loss"] = price * (1 + signal["stop_loss_pct"])
            signal["take_profit"] = price * (1 - signal["take_profit_pct"])
        return True

    def _apply_portfolio_sizing(
        self,
        trade_signal: TradeSignal,
        signal: Dict,
        *,
        open_positions: List[Dict],
        account_balance: float,
        regime_data: Optional[Dict],
    ) -> bool:
        if self.kelly_sizer:
            try:
                source_key = signal.get("_source_key") or f"copy_trade:{signal.get('source_trader', 'unknown')}"
                sizing = self.kelly_sizer.get_sizing(
                    strategy_key=source_key,
                    account_balance=account_balance,
                    signal_confidence=trade_signal.confidence,
                )
                trade_signal.position_pct = sizing.position_pct
            except Exception as exc:
                logger.debug("Copy Kelly sizing error for %s: %s", signal.get("coin"), exc)

        sizing_context = None
        if self.portfolio_sizer:
            try:
                sizing_context = self.portfolio_sizer.apply_to_signal(
                    trade_signal,
                    open_positions=open_positions,
                    account_balance=account_balance,
                    regime_data=regime_data,
                    features=signal.get("features", {}),
                )
            except Exception as exc:
                logger.debug("Copy portfolio sizing error for %s: %s", signal.get("coin"), exc)

        if sizing_context and sizing_context.blocked:
            logger.info(
                "  Portfolio sizing blocked copy %s %s: %s",
                signal.get("side"),
                signal.get("coin"),
                sizing_context.block_reason,
            )
            return False

        source_budget_context = None
        if self.source_allocator:
            try:
                source_budget_context = self.source_allocator.apply_to_signal(
                    trade_signal,
                    signal=signal,
                    open_positions=open_positions,
                    account_balance=account_balance,
                )
            except Exception as exc:
                logger.debug("Copy source budget error for %s: %s", signal.get("coin"), exc)

        if source_budget_context and source_budget_context.blocked:
            signal["source_budget"] = source_budget_context.to_dict()
            signal["source_budget_status"] = source_budget_context.status
            signal["source_budget_reason"] = source_budget_context.block_reason
            logger.info(
                "  Source budget blocked copy %s %s: %s",
                signal.get("side"),
                signal.get("coin"),
                source_budget_context.block_reason,
            )
            return False

        if not self._sync_signal_from_trade_signal(
            trade_signal,
            signal,
            account_balance=account_balance,
        ):
            return False

        if sizing_context:
            signal["portfolio_sizing"] = sizing_context.to_dict()
            signal["cluster"] = sizing_context.cluster
        if source_budget_context:
            signal["source_budget"] = source_budget_context.to_dict()
            signal["source_budget_status"] = source_budget_context.status
            signal["source_budget_reason"] = source_budget_context.block_reason
        return True

    def execute_copy_signals(self, signals: List[Dict],
                              regime_data: Optional[Dict] = None) -> List[Dict]:
        """
        Execute copy-trade signals as paper trades.
        V2: Routes open signals through DecisionFirewall before execution.
        Returns list of executed trades.
        """
        if not signals:
            return []

        account = db.get_paper_account()
        if not account:
            return []

        open_trades = db.get_open_paper_trades()
        mids = hl.get_all_mids() or {}
        executed = []
        pending_entries = []
        self._annotate_open_trades(open_trades, mids)

        for signal in signals:
            try:
                if signal["type"] == "copy_close":
                    closed = self._close_copy_trades(signal, open_trades, mids)
                    closed_ids = {trade["trade_id"] for trade in closed}
                    open_trades = [trade for trade in open_trades if trade.get("id") not in closed_ids]
                    continue

                if signal["type"] in ("copy_open", "copy_scale_in", "copy_flip", "golden_copy"):
                    signal = self._apply_regime_weight(signal, signal["coin"])
                    trade_signal = None

                    if self.firewall and signal.get("price", 0) > 0:
                        trade_signal = signal_from_copy_trade(
                            trader_address=signal.get("source_trader", ""),
                            coin=signal["coin"],
                            side=signal["side"],
                            entry_price=signal["price"],
                            confidence=signal.get("confidence", 0.5),
                        )
                        trade_signal.leverage = signal.get("leverage", 2)
                        trade_signal.regime = regime_data.get("overall_regime", "") if regime_data else ""

                        if self.agent_scorer:
                            source_key = f"copy_trade:{signal.get('source_trader', 'unknown')}"
                            weight = self.agent_scorer.get_weight(source_key)
                            trade_signal.confidence = trade_signal.confidence * 0.6 + weight * 0.4

                        if self.kelly_sizer:
                            try:
                                source_key = f"copy_trade:{signal.get('source_trader', 'unknown')}"
                                sizing = self.kelly_sizer.get_sizing(
                                    strategy_key=source_key,
                                    account_balance=account["balance"],
                                    signal_confidence=trade_signal.confidence,
                                )
                                trade_signal.position_pct = sizing.position_pct
                            except Exception as exc:
                                logger.debug("Copy Kelly pre-screen sizing error for %s: %s", signal.get("coin"), exc)

                        passed, reason = self.firewall.validate(
                            trade_signal,
                            regime_data=regime_data,
                            open_positions=open_trades,
                            ignore_position_limit=True,
                            dry_run=True,
                        )
                        if not passed:
                            logger.info(f"  Firewall rejected copy {signal['side']} {signal['coin']}: {reason}")
                            continue

                    if trade_signal is None:
                        trade_signal = signal_from_copy_trade(
                            trader_address=signal.get("source_trader", ""),
                            coin=signal["coin"],
                            side=signal["side"],
                            entry_price=signal["price"],
                            confidence=signal.get("confidence", 0.5),
                        )
                        trade_signal.leverage = signal.get("leverage", 2)
                        trade_signal.regime = regime_data.get("overall_regime", "") if regime_data else ""

                    signal["confidence"] = trade_signal.confidence
                    signal["source_accuracy"] = trade_signal.source_accuracy
                    signal["regime"] = trade_signal.regime
                    pending_entries.append({
                        "signal": signal,
                        "trade_signal": trade_signal,
                    })

            except Exception as e:
                logger.error(f"Error executing copy signal: {e}")

        replacements_used = 0
        rotation_enabled = bool(config.ROTATION_ENGINE_ENABLED)
        rotation_dry_run = bool(config.ROTATION_DRY_RUN_TELEMETRY)
        shadow_mode = rotation_enabled and rotation_dry_run
        for candidate in sorted(
            pending_entries,
            key=lambda item: self.rotation_manager.candidate_score(
                item["trade_signal"], regime_data=regime_data
            ),
            reverse=True,
        ):
            signal = candidate["signal"]
            trade_signal = candidate["trade_signal"]
            decision = self.rotation_manager.decide(
                trade_signal,
                open_trades,
                regime_data=regime_data,
                replacements_used=replacements_used,
            )
            victim = None
            candidate_open_positions = open_trades
            shadow_bypass_open = (
                shadow_mode
                and self.rotation_manager.should_bypass_reject_in_shadow_mode(
                    decision,
                    len(open_trades),
                )
            )

            if decision.action == "reject" and not shadow_bypass_open:
                logger.info(
                    "  Rotation skipped copy %s %s: %s",
                    signal["side"],
                    signal["coin"],
                    decision.reason,
                )
                continue
            if shadow_bypass_open:
                logger.info(
                    "  Rotation shadow mode: bypassing rotation reject for copy %s %s (%s)",
                    signal["side"],
                    signal["coin"],
                    decision.reason,
                )

            if decision.action == "replace":
                if not rotation_enabled:
                    if rotation_dry_run:
                        self.rotation_manager.record_dry_run_replacement_skip(
                            decision, reason_key="dry_run_rotation_disabled"
                        )
                        logger.info(
                            "  Rotation dry-run: would replace %s with copy %s (%s)",
                            decision.replacement_trade_id,
                            signal["coin"],
                            decision.reason,
                        )
                    logger.info(
                        "  Rotation disabled by config; skipped replacement for copy %s %s",
                        signal["side"],
                        signal["coin"],
                    )
                    continue
                if rotation_dry_run:
                    self.rotation_manager.record_dry_run_replacement_skip(
                        decision, reason_key="dry_run_shadow_mode"
                    )
                    logger.info(
                        "  Rotation shadow mode: simulated replacement of %s with copy %s (%s)",
                        decision.replacement_trade_id,
                        signal["coin"],
                        decision.reason,
                    )
                    continue

            if decision.action == "replace":
                victim = next(
                    (trade for trade in open_trades if trade.get("id") == decision.replacement_trade_id),
                    None,
                )
                if not victim:
                    logger.info("  Rotation skipped copy %s: incumbent not found", signal["coin"])
                    continue

                candidate_open_positions = [
                    trade for trade in open_trades if trade.get("id") != victim.get("id")
                ]

            if not self._apply_portfolio_sizing(
                trade_signal,
                signal,
                open_positions=candidate_open_positions,
                account_balance=float(account.get("balance", 0) or 0),
                regime_data=regime_data,
            ):
                logger.info(
                    "  Copy sizing skipped %s %s before execution",
                    signal["side"],
                    signal["coin"],
                )
                continue

            if self.firewall:
                passed, reason = self.firewall.validate(
                    trade_signal,
                    regime_data=regime_data,
                    open_positions=candidate_open_positions,
                )
                if not passed:
                    logger.info(
                        "  Firewall rejected copy %s %s at execution time: %s",
                        signal["side"],
                        signal["coin"],
                        reason,
                    )
                    continue

            if self.agent_scorer:
                source_key = f"copy_trade:{signal.get('source_trader', 'unknown')}"
                signal_id = self.agent_scorer.record_signal(source_key, {
                    "coin": signal["coin"],
                    "side": signal["side"],
                    "confidence": trade_signal.confidence,
                })
                signal["_signal_id"] = signal_id
                signal["_source_key"] = source_key

            account = db.get_paper_account() or account
            trade = self._open_copy_trade(account, signal, candidate_open_positions)
            if trade:
                executed.append(trade)
                self._annotate_open_trades([trade], mids)
                open_trades.append(trade)
                if victim:
                    current_price = float(
                        mids.get(victim["coin"], victim.get("entry_price", 0)) or victim.get("entry_price", 0)
                    )
                    closed_trade = self._close_trade(
                        victim,
                        exit_price=current_price,
                        close_reason=f"rotation_out:{signal['coin']}",
                    )
                    if not closed_trade:
                        logger.warning(
                            "  Rotation close failed after opening copy %s; rolling back trade %s",
                            signal["coin"],
                            trade.get("id"),
                        )
                        rollback = self._close_trade(
                            trade,
                            exit_price=float(
                                signal.get("price", trade.get("entry_price", 0))
                                or trade.get("entry_price", 0)
                            ),
                            close_reason=f"rotation_rollback:{victim.get('coin', 'unknown')}",
                        )
                        open_trades = [t for t in open_trades if t.get("id") != trade.get("id")]
                        executed = [t for t in executed if t.get("id") != trade.get("id")]
                        if not rollback:
                            logger.error("  Rollback close also failed for copy trade %s", trade.get("id"))
                        continue

                    open_trades = [t for t in open_trades if t.get("id") != victim.get("id")]
                    replacements_used += 1
                    self.rotation_manager.register_replacement(
                        replaced_trade=victim,
                        new_coin=signal.get("coin", ""),
                        new_side=signal.get("side", ""),
                    )
                    logger.info(
                        "  Rotation replaced %s with copy %s (%s)",
                        victim.get("coin"),
                        signal["coin"],
                        decision.reason,
                    )

        if executed:
            self._copy_count += len(executed)
            logger.info(f"Copy-trader executed {len(executed)} trades via V2 (total: {self._copy_count})")

        return executed

    def _open_copy_trade(self, account: Dict, signal: Dict, open_trades: List) -> Optional[Dict]:
        """Open a paper trade based on a copy signal."""
        price = signal["price"]
        if price <= 0:
            return None

        position_pct = float(signal.get("position_pct", 0.0) or 0.0)
        if position_pct > 0:
            size_usd = account["balance"] * position_pct
        elif self.kelly_sizer:
            try:
                source_key = f"copy_trade:{signal.get('source_trader', 'unknown')}"
                sizing = self.kelly_sizer.get_sizing(
                    strategy_key=source_key,
                    account_balance=account["balance"],
                    signal_confidence=signal.get("confidence", 0.5),
                )
                size_usd = sizing.position_usd
                position_pct = sizing.position_pct
            except Exception:
                size_usd = account["balance"] * 0.05 * signal.get("confidence", 0.5)
        else:
            size_usd = account["balance"] * 0.05 * signal.get("confidence", 0.5)
        size = signal.get("size") or (size_usd / price)
        leverage = signal.get("leverage", 2)
        side = signal["side"]

        # CRITICAL: No conflicting sides on same asset
        for t in open_trades:
            if t.get("coin") == signal["coin"] and t.get("side") != side:
                logger.debug(f"Copy skip: conflicting side for {signal['coin']}")
                return None

        # Check basic risk: max 5 copy trades per coin (same direction only)
        coin_copies = sum(1 for t in open_trades if t.get("coin") == signal["coin"])
        if coin_copies >= 5:
            return None

        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        if not stop_loss or not take_profit:
            if side == "long":
                stop_loss = price * (1 - 0.04 / leverage)
                take_profit = price * (1 + 0.08 / leverage)
            else:
                stop_loss = price * (1 + 0.04 / leverage)
                take_profit = price * (1 - 0.08 / leverage)

        try:
            execution_role = signal.get("execution_role", config.PAPER_TRADING_DEFAULT_EXECUTION_ROLE)
            source_key = signal.get("_source_key") or f"copy_trade:{signal.get('source_trader', 'unknown')}"
            trade_id = db.open_paper_trade(
                strategy_id=None,
                coin=signal["coin"],
                side=side,
                entry_price=price,
                size=size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "type": signal["type"],
                    "source_trader": signal.get("source_trader", ""),
                    "confidence": signal.get("confidence", 0),
                    "signal_id": signal.get("_signal_id", ""),
                    "source_accuracy": signal.get("source_accuracy", 0),
                    "regime": signal.get("regime", ""),
                    "is_copy_trade": True,
                    "is_golden": signal.get("is_golden", False),
                    "golden_wallet": signal.get("is_golden", False),
                    "source": "copy_trade",
                    "source_key": source_key,
                    "execution_role": execution_role,
                    "maker_fee_bps": config.PAPER_TRADING_MAKER_FEE_BPS,
                    "taker_fee_bps": config.PAPER_TRADING_TAKER_FEE_BPS,
                    "intended_entry_price": price,
                    "entry_slipped_price": price,
                    "position_pct": position_pct,
                    "time_limit_hours": signal.get("time_limit_hours", 24.0),
                    "portfolio_sizing": signal.get("portfolio_sizing", {}),
                    "source_budget": signal.get("source_budget", {}),
                    "source_budget_status": signal.get("source_budget_status", ""),
                    "source_budget_reason": signal.get("source_budget_reason", ""),
                    "cluster": signal.get("cluster", ""),
                },
            )
            logger.info(
                f"Copy trade: {side.upper()} {signal['coin']} @ ${price:,.2f} "
                f"(from trader {signal.get('source_trader', '?')})"
            )
            return {
                "id": trade_id,
                "coin": signal["coin"],
                "side": side,
                "entry_price": price,
                "size": size,
                "position_pct": position_pct,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "time_limit_hours": signal.get("time_limit_hours", 24.0),
                "strategy_id": None,
                "strategy_type": "copy_trade",
                "confidence": signal.get("confidence", 0),
                "trader_address": signal.get("source_trader", ""),
                "source": "copy_trade",
                "source_key": source_key,
                "opened_at": datetime.utcnow().isoformat(),
                "source_budget": signal.get("source_budget", {}),
                "metadata": {
                    "type": signal["type"],
                    "source_trader": signal.get("source_trader", ""),
                    "confidence": signal.get("confidence", 0),
                    "signal_id": signal.get("_signal_id", ""),
                    "source_accuracy": signal.get("source_accuracy", 0),
                    "regime": signal.get("regime", ""),
                    "is_copy_trade": True,
                    "is_golden": signal.get("is_golden", False),
                    "golden_wallet": signal.get("is_golden", False),
                    "source": "copy_trade",
                    "source_key": source_key,
                    "execution_role": execution_role,
                    "maker_fee_bps": config.PAPER_TRADING_MAKER_FEE_BPS,
                    "taker_fee_bps": config.PAPER_TRADING_TAKER_FEE_BPS,
                    "intended_entry_price": price,
                    "entry_slipped_price": price,
                    "position_pct": position_pct,
                    "time_limit_hours": signal.get("time_limit_hours", 24.0),
                    "portfolio_sizing": signal.get("portfolio_sizing", {}),
                    "source_budget": signal.get("source_budget", {}),
                    "source_budget_status": signal.get("source_budget_status", ""),
                    "source_budget_reason": signal.get("source_budget_reason", ""),
                    "cluster": signal.get("cluster", ""),
                },
            }
        except Exception as e:
            logger.error(f"Error opening copy trade: {e}")
            return None

    @staticmethod
    def _annotate_open_trades(open_trades: List[Dict], mids: Dict):
        for trade in open_trades:
            fallback_price = float(trade.get("entry_price", 0) or 0)
            trade["current_price"] = float(mids.get(trade.get("coin", ""), fallback_price) or fallback_price)

    def _close_trade(self, trade: Dict, exit_price: float, close_reason: str) -> Optional[Dict]:
        """Close a copy trade and feed the result into the paper-trading scorekeepers."""
        meta = trade.get("metadata", "{}")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        else:
            meta = dict(meta or {})

        if trade["side"] == "long":
            gross_pnl = (exit_price - trade["entry_price"]) * trade["size"] * trade["leverage"]
        else:
            gross_pnl = (trade["entry_price"] - exit_price) * trade["size"] * trade["leverage"]
        gross_pnl = round(gross_pnl, 2)

        execution_role = meta.get("execution_role", config.PAPER_TRADING_DEFAULT_EXECUTION_ROLE)
        fee_rate = (config.PAPER_TRADING_MAKER_FEE_BPS if execution_role == "maker" else config.PAPER_TRADING_TAKER_FEE_BPS) / 10_000
        entry_notional = max(float(trade.get("entry_price", 0)), 0.0) * max(float(trade.get("size", 0)), 0.0) * max(float(trade.get("leverage", 1)), 1.0)
        exit_notional = max(float(exit_price), 0.0) * max(float(trade.get("size", 0)), 0.0) * max(float(trade.get("leverage", 1)), 1.0)
        entry_fee = entry_notional * fee_rate
        exit_fee = exit_notional * fee_rate
        total_fees = round(entry_fee + exit_fee, 4)
        pnl = round(gross_pnl - total_fees, 2)

        db.update_paper_trade_metadata(
            trade["id"],
            {
                "execution_role": execution_role,
                "maker_fee_bps": config.PAPER_TRADING_MAKER_FEE_BPS,
                "taker_fee_bps": config.PAPER_TRADING_TAKER_FEE_BPS,
                "entry_fee_paid": round(entry_fee, 4),
                "exit_fee_paid": round(exit_fee, 4),
                "total_fees_paid": total_fees,
                "entry_slippage_cost": 0.0,
                "exit_slippage_cost": 0.0,
                "total_slippage_cost": 0.0,
                "gross_pnl_before_fees": round(gross_pnl, 4),
                "net_pnl_after_fees": pnl,
                "intended_exit_price": float(exit_price),
                "exit_slipped_price": float(exit_price),
            },
        )
        meta.update(
            {
                "execution_role": execution_role,
                "maker_fee_bps": config.PAPER_TRADING_MAKER_FEE_BPS,
                "taker_fee_bps": config.PAPER_TRADING_TAKER_FEE_BPS,
                "entry_fee_paid": round(entry_fee, 4),
                "exit_fee_paid": round(exit_fee, 4),
                "total_fees_paid": total_fees,
                "entry_slippage_cost": 0.0,
                "exit_slippage_cost": 0.0,
                "total_slippage_cost": 0.0,
                "gross_pnl_before_fees": round(gross_pnl, 4),
                "net_pnl_after_fees": pnl,
                "intended_exit_price": float(exit_price),
                "exit_slipped_price": float(exit_price),
            }
        )

        db.close_paper_trade(trade["id"], exit_price, pnl)

        account = db.get_paper_account()
        if account:
            db.update_paper_account(
                account["balance"] + pnl,
                account["total_pnl"] + pnl,
                account["total_trades"] + 1,
                account["winning_trades"] + (1 if pnl > 0 else 0),
            )

        logger.info(
            "Copy trade closed (%s): %s %s entry=$%s exit=$%s gross=$%s fees=$%s net=$%s",
            close_reason,
            trade["side"].upper(),
            trade["coin"],
            f"{trade['entry_price']:,.2f}",
            f"{exit_price:,.2f}",
            f"{gross_pnl:,.2f}",
            f"{total_fees:,.2f}",
            f"{pnl:,.2f}",
        )

        return_pct = pnl / max(
            trade["entry_price"] * max(trade["size"], 1e-8) * max(trade.get("leverage", 1), 1),
            1e-8,
        )
        source_key = f"copy_trade:{meta.get('source_trader', 'unknown')}"

        if self.agent_scorer and meta.get("source_trader"):
            try:
                self.agent_scorer.record_outcome(source_key, meta.get("signal_id", ""), pnl, return_pct)
            except Exception:
                pass
        if self.firewall:
            try:
                self.firewall.record_trade_outcome(trade["coin"], pnl)
            except Exception:
                pass

        if self.kelly_sizer:
            try:
                self.kelly_sizer.record_outcome(
                    strategy_key=source_key,
                    pnl=pnl,
                    entry_price=trade["entry_price"],
                    size=trade["size"],
                    leverage=trade.get("leverage", 1),
                )
            except Exception:
                pass

        if self.calibration:
            try:
                self.calibration.record(
                    source_key=source_key,
                    predicted_confidence=meta.get("confidence", 0.5),
                    actual_win=pnl > 0,
                    pnl=pnl,
                    coin=trade["coin"],
                    side=trade.get("side", ""),
                )
            except Exception:
                pass

        if self.trade_memory:
            try:
                self.trade_memory.record_trade(
                    trade_id=str(trade["id"]),
                    coin=trade["coin"],
                    side=trade.get("side", ""),
                    strategy_type="copy_trade",
                    entry_price=trade["entry_price"],
                    exit_price=exit_price,
                    pnl=pnl,
                    return_pct=return_pct,
                    opened_at=trade.get("opened_at", ""),
                    closed_at=datetime.utcnow().isoformat(),
                    confidence=meta.get("confidence", 0),
                    source="copy_trade",
                )
            except Exception:
                pass

        closed_event = {
            "trade_id": trade["id"],
            "coin": trade["coin"],
            "side": trade.get("side", ""),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": exit_price,
            "size": trade.get("size", 0),
            "leverage": trade.get("leverage", 1),
            "pnl": pnl,
            "gross_pnl": round(gross_pnl, 4),
            "fees_paid": total_fees,
            "slippage_cost": 0.0,
            "reason": close_reason,
            "metadata": meta,
            "strategy_type": "copy_trade",
            "trader_address": meta.get("source_trader", ""),
            "opened_at": trade.get("opened_at", ""),
            "closed_at": datetime.utcnow().isoformat(),
        }
        self._closed_events.append(closed_event)
        return closed_event

    def _close_copy_trades(self, signal: Dict, open_trades: List, mids: Dict) -> List[Dict]:
        """Close open copy trades for a coin when the source trader exits."""
        closed = []
        for trade in open_trades:
            meta = trade.get("metadata", "{}")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            if (trade["coin"] == signal["coin"] and
                meta.get("is_copy_trade") and
                meta.get("source_trader") == signal.get("source_trader")):

                current_price = float(mids.get(trade["coin"], 0))
                if current_price <= 0:
                    continue
                closed_trade = self._close_trade(
                    trade,
                    exit_price=current_price,
                    close_reason="source_exit",
                )
                if closed_trade:
                    closed.append(closed_trade)

        return closed

    def drain_closed_events(self) -> List[Dict]:
        """Drain closed-trade events produced during this cycle."""
        events = list(self._closed_events)
        self._closed_events.clear()
        return events

    @property
    def total_copy_trades(self) -> int:
        return self._copy_count
