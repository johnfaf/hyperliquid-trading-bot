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
from typing import List, Dict, Optional, Set, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import database as db
from src.data import hyperliquid_client as hl
from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams, signal_from_copy_trade
from src.signals.decision_firewall import DecisionFirewall
from src.signals.agent_scoring import AgentScorer
from src.signals.kelly_sizing import KellySizer
from src.trading.trade_memory import TradeMemory
from src.trading.portfolio_rotation import PortfolioRotationManager
from src.signals.calibration import CalibrationTracker

logger = logging.getLogger(__name__)


class CopyTrader:
    """Monitors top traders and mirrors their position changes."""

    def __init__(self, firewall: Optional[DecisionFirewall] = None,
                 agent_scorer: Optional[AgentScorer] = None,
                 kelly_sizer: Optional[KellySizer] = None,
                 trade_memory: Optional[TradeMemory] = None,
                 calibration: Optional[CalibrationTracker] = None,
                 regime_forecaster: Optional[object] = None):
        # Cache of last-known positions per trader: {address: {coin: position_dict}}
        self._position_cache: Dict[str, Dict[str, Dict]] = {}
        self._copy_count = 0
        self.firewall = firewall
        self.agent_scorer = agent_scorer
        self.kelly_sizer = kelly_sizer
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

            if decision.action == "reject":
                logger.info(
                    "  Rotation skipped copy %s %s: %s",
                    signal["side"],
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
        # Kelly-based position sizing if available, else default 5%
        if self.kelly_sizer:
            try:
                source_key = f"copy_trade:{signal.get('source_trader', 'unknown')}"
                sizing = self.kelly_sizer.get_sizing(
                    strategy_key=source_key,
                    account_balance=account["balance"],
                    signal_confidence=signal.get("confidence", 0.5),
                )
                size_usd = sizing.position_usd
            except Exception:
                size_usd = account["balance"] * 0.05 * signal.get("confidence", 0.5)
        else:
            size_usd = account["balance"] * 0.05 * signal.get("confidence", 0.5)
        price = signal["price"]
        if price <= 0:
            return None

        size = size_usd / price
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

        # SL/TP
        if side == "long":
            stop_loss = price * (1 - 0.04 / leverage)
            take_profit = price * (1 + 0.08 / leverage)
        else:
            stop_loss = price * (1 + 0.04 / leverage)
            take_profit = price * (1 - 0.08 / leverage)

        try:
            execution_role = signal.get("execution_role", config.PAPER_TRADING_DEFAULT_EXECUTION_ROLE)
            trade_id = db.open_paper_trade(
                strategy_id=None,
                coin=signal["coin"],
                side=side,
                entry_price=price,
                size=size,
                leverage=leverage,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
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
                    "execution_role": execution_role,
                    "maker_fee_bps": config.PAPER_TRADING_MAKER_FEE_BPS,
                    "taker_fee_bps": config.PAPER_TRADING_TAKER_FEE_BPS,
                    "intended_entry_price": price,
                    "entry_slipped_price": price,
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
                "leverage": leverage,
                "strategy_id": None,
                "confidence": signal.get("confidence", 0),
                "trader_address": signal.get("source_trader", ""),
                "opened_at": datetime.utcnow().isoformat(),
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
                    "execution_role": execution_role,
                    "maker_fee_bps": config.PAPER_TRADING_MAKER_FEE_BPS,
                    "taker_fee_bps": config.PAPER_TRADING_TAKER_FEE_BPS,
                    "intended_entry_price": price,
                    "entry_slipped_price": price,
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
