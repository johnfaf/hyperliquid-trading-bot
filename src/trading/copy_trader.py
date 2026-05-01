"""
Copy Trading Engine (V2)
========================
Monitors top traders' live positions on Hyperliquid and mirrors their new trades
as paper trades. Detects when top traders open/close positions and generates signals.

V2: Signals routed through DecisionFirewall and tracked by AgentScorer.
"""
import logging
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import database as db
from src.data import hyperliquid_client as hl
from src.analysis.trade_analytics import evaluate_source_policy, evaluate_side_source_policy
from src.signals.signal_schema import RiskParams, SignalSide, SignalSource, TradeSignal, signal_from_copy_trade
from src.signals.decision_firewall import DecisionFirewall
from src.signals.agent_scoring import AgentScorer
from src.signals.kelly_sizing import KellySizer
from src.trading.trade_memory import TradeMemory
from src.trading.portfolio_rotation import PortfolioRotationManager
from src.signals.calibration import CalibrationTracker
from src.signals.risk_policy import RiskPolicyEngine

logger = logging.getLogger(__name__)

_CRASH_COPY_CONFIDENCE_MULTIPLIER = 0.60
_NEUTRAL_COPY_CONFIDENCE_MULTIPLIER = 0.70
_BULLISH_COPY_CONFIDENCE_MULTIPLIER = 1.20

# Signal confidence model — explicit baselines and caps per signal type.
# Rationale:
#   copy_open:    base 0.50 — new unconfirmed position, moderate conviction
#   copy_scale_in: base 0.40 — adding to existing position is cautious (less info)
#   copy_flip:    base 0.60 — reversal implies strong directional conviction
#   golden_copy:  base 0.55 — golden wallet signal, slightly elevated base
# Win-rate contribution: +0 to +0.50 depending on trader quality.
_SIGNAL_CONFIDENCE_MODEL: Dict[str, Dict[str, float]] = {
    "copy_open":    {"base": 0.50, "max": 0.90},
    "copy_scale_in": {"base": 0.40, "max": 0.85},
    "copy_scale_out": {"base": 1.00, "max": 1.00},
    "copy_flip":    {"base": 0.60, "max": 0.95},
    "golden_copy":  {"base": 0.55, "max": 0.90},
    "copy_close":   {"base": 1.00, "max": 1.00},   # Exit is always full conviction
}

# How long to keep cached positions for a trader that is no longer in the top N.
_POSITION_CACHE_TTL_SECONDS = 3600  # 1 hour


class CopyTrader:
    """Monitors top traders and mirrors their position changes."""

    def __init__(self, firewall: Optional[DecisionFirewall] = None,
                 agent_scorer: Optional[AgentScorer] = None,
                 kelly_sizer: Optional[KellySizer] = None,
                 rl_sizer: Optional[object] = None,
                 trade_memory: Optional[TradeMemory] = None,
                 calibration: Optional[CalibrationTracker] = None,
                 regime_forecaster: Optional[object] = None,
                 risk_policy_engine: Optional[RiskPolicyEngine] = None,
                 shadow_tracker: Optional[object] = None):
        # Cache of last-known positions per trader: {address: {coin: position_dict}}
        self._position_cache: Dict[str, Dict[str, Dict]] = {}
        # TTL tracking so stale entries for dropped traders don't accumulate forever
        self._position_cache_ts: Dict[str, float] = {}
        self._copy_count = 0
        self.firewall = firewall
        self.agent_scorer = agent_scorer
        self.kelly_sizer = kelly_sizer
        self.rl_sizer = rl_sizer
        self.trade_memory = trade_memory
        self.calibration = calibration
        self.regime_forecaster = regime_forecaster
        self.risk_policy_engine = risk_policy_engine
        self.shadow_tracker = shadow_tracker
        self.rotation_manager = PortfolioRotationManager()
        self._closed_events: List[Dict] = []
        self._sizer_fallback_count = 0
        self.enabled = bool(getattr(config, "COPY_TRADER_ENABLED", True))
        self.max_concurrent_trades = max(
            0, int(getattr(config, "COPY_TRADER_MAX_CONCURRENT_TRADES", 2))
        )
        self.max_new_trades_per_cycle = max(
            0, int(getattr(config, "COPY_TRADER_MAX_NEW_TRADES_PER_CYCLE", 1))
        )
        self.auto_pause_min_closed_trades = max(
            1, int(getattr(config, "COPY_TRADER_AUTO_PAUSE_MIN_CLOSED_TRADES", 6))
        )
        self.auto_pause_degrade_win_rate = float(
            getattr(config, "COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE", 0.40)
        )
        self.auto_pause_block_win_rate = float(
            getattr(config, "COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE", 0.25)
        )
        self.auto_pause_block_net_pnl = float(
            getattr(config, "COPY_TRADER_AUTO_PAUSE_BLOCK_NET_PNL", -25.0)
        )
        self.source_side_guard_enabled = bool(
            getattr(config, "COPY_TRADER_SOURCE_SIDE_GUARD_ENABLED", True)
        )
        self.source_side_min_closed_trades = max(
            1, int(getattr(config, "COPY_TRADER_SOURCE_SIDE_MIN_CLOSED_TRADES", 3))
        )
        self.source_side_degrade_win_rate = float(
            getattr(config, "COPY_TRADER_SOURCE_SIDE_DEGRADE_WIN_RATE", 0.45)
        )
        self.source_side_block_win_rate = float(
            getattr(config, "COPY_TRADER_SOURCE_SIDE_BLOCK_WIN_RATE", 0.35)
        )
        self.source_side_block_net_pnl = float(
            getattr(config, "COPY_TRADER_SOURCE_SIDE_BLOCK_NET_PNL", -0.25)
        )
        self.source_side_confidence_multiplier = float(
            getattr(config, "COPY_TRADER_SOURCE_SIDE_CONFIDENCE_MULTIPLIER", 0.75)
        )
        self.source_side_size_multiplier = float(
            getattr(config, "COPY_TRADER_SOURCE_SIDE_SIZE_MULTIPLIER", 0.50)
        )
        self._source_side_policy_cache: Dict[str, object] = {"ts": 0.0, "closed": [], "policies": {}}
        self._source_side_policy_cache_ttl_s = 300.0
        self._copy_guardrail_status: Dict[str, object] = {
            "status": "healthy" if self.enabled else "paused",
            "reason": (
                "Copy trader enabled by config"
                if self.enabled
                else "Copy trader disabled by config"
            ),
            "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
            "source": "copy_trade",
            "max_concurrent_trades": self.max_concurrent_trades,
            "max_new_trades_per_cycle": self.max_new_trades_per_cycle,
        }

    @staticmethod
    def _source_key(payload: Dict) -> str:
        trader = str(payload.get("source_trader", "") or "").strip().lower()
        return f"copy_trade:{trader}" if trader else "copy_trade"

    def _get_source_side_closed_trades(self) -> List[Dict]:
        now = time.time()
        cache_ts = float(self._source_side_policy_cache.get("ts", 0.0) or 0.0)
        if (now - cache_ts) < self._source_side_policy_cache_ttl_s:
            return list(self._source_side_policy_cache.get("closed") or [])
        try:
            closed = db.get_paper_trade_history(limit=250)
        except Exception as exc:
            logger.debug("Copy source/side policy lookup failed: %s", exc)
            closed = []
        self._source_side_policy_cache = {
            "ts": now,
            "closed": list(closed or []),
            "policies": {},
        }
        return list(closed or [])

    def _apply_source_side_guard(self, signal: Dict) -> tuple[bool, str]:
        if not self.source_side_guard_enabled:
            return True, ""
        side = str(signal.get("side", "") or "").strip().lower()
        if not side:
            return True, ""
        source_key = self._source_key(signal)
        cache_key = f"{source_key}:{side}"
        policies = dict(self._source_side_policy_cache.get("policies") or {})
        policy = policies.get(cache_key)
        if not policy:
            policy = evaluate_side_source_policy(
                self._get_source_side_closed_trades(),
                side=side,
                source_key=source_key,
                min_trades=self.source_side_min_closed_trades,
                degrade_win_rate=self.source_side_degrade_win_rate,
                block_win_rate=self.source_side_block_win_rate,
                block_net_pnl=self.source_side_block_net_pnl,
            )
            policies[cache_key] = dict(policy)
            self._source_side_policy_cache["policies"] = policies

        status = str(policy.get("status", "") or "").lower()
        if status == "blocked":
            return False, policy.get("reason", "copy source/side policy blocked signal")
        if status == "degraded":
            original_confidence = float(signal.get("confidence", 0.5) or 0.5)
            signal["confidence"] = original_confidence * self.source_side_confidence_multiplier
            signal["copy_source_size_multiplier"] = min(
                float(signal.get("copy_source_size_multiplier", 1.0) or 1.0),
                self.source_side_size_multiplier,
            )
            signal["copy_source_side_policy"] = dict(policy)
            logger.warning(
                "Copy source/side guard de-risked %s %s from %s: confidence %.0f%% -> %.0f%%, size *= %.2f (%s)",
                signal.get("coin", "?"),
                side,
                source_key,
                original_confidence * 100,
                float(signal.get("confidence", 0.0) or 0.0) * 100,
                self.source_side_size_multiplier,
                policy.get("reason", "recent source/side underperformance"),
            )
        return True, ""

    @staticmethod
    def _calculate_signal_confidence(signal_type: str, trader_win_rate: float) -> float:
        """
        Calculate copy signal confidence using the explicit model in
        _SIGNAL_CONFIDENCE_MODEL.

        The win-rate contribution is capped at 0.5 (a perfect trader at 100% WR
        adds 0.5 to the base). This avoids inflating confidence to near-1.0 based
        solely on historical data that may not generalise.
        """
        model = _SIGNAL_CONFIDENCE_MODEL.get(signal_type, {"base": 0.50, "max": 0.90})
        base = model["base"]
        cap = model["max"]
        win_contribution = min(float(trader_win_rate), 1.0) * 0.5
        return min(cap, base + win_contribution)

    def _resolve_copy_trade_risk(
        self,
        signal: Dict,
        leverage: float,
        regime_data: Optional[Dict] = None,
    ) -> tuple[float, float, Dict[str, object]]:
        price = float(signal.get("price", 0.0) or 0.0)
        side = str(signal.get("side", "long") or "long")
        source_trader = str(signal.get("source_trader", "") or "").strip().lower()
        trade_signal = TradeSignal(
            coin=str(signal.get("coin", "")),
            side=SignalSide(side),
            confidence=float(signal.get("confidence", 0.5) or 0.5),
            source=SignalSource.COPY_TRADE,
            reason=f"Copy trade from {source_trader or '?'}",
            trader_address=source_trader,
            entry_price=price,
            leverage=leverage,
            risk=RiskParams(stop_loss_pct=0.04, take_profit_pct=0.20, risk_basis="roe"),
            context={
                "features": {},
                "atr_pct": signal.get("atr_pct"),
                "volatility": signal.get("volatility"),
            },
            regime=str(signal.get("regime", "")),
            source_accuracy=float(signal.get("source_accuracy", 0.0) or 0.0),
        )
        if self.risk_policy_engine:
            try:
                modified = self.risk_policy_engine.apply(trade_signal, regime_data=regime_data)
                if modified is not None:
                    trade_signal = modified
                else:
                    logger.warning("risk_policy_engine.apply() returned None; using original signal")
            except Exception as e:
                logger.warning("Risk policy apply failed (%s); proceeding with original signal", e)
        stop_loss, take_profit = trade_signal.risk.resolve_trigger_prices(price, side, leverage)
        return stop_loss, take_profit, dict((trade_signal.context or {}).get("risk_policy", {}) or {})

    @staticmethod
    def _is_copy_trade(trade: Dict) -> bool:
        meta = trade.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta or "{}")
            except (json.JSONDecodeError, TypeError):
                meta = {}
        meta = dict(meta or {})
        source = str(meta.get("source") or trade.get("source") or "").strip().lower()
        return bool(meta.get("is_copy_trade")) or source.startswith("copy_trade")

    def _open_copy_trade_count(self, trades: List[Dict]) -> int:
        return sum(1 for trade in (trades or []) if self._is_copy_trade(trade))

    @staticmethod
    def _drawdown_from_initial_balance(account_balance: float) -> float:
        baseline = max(float(getattr(config, "PAPER_TRADING_INITIAL_BALANCE", 10_000.0)), 1.0)
        return max(0.0, (baseline - max(float(account_balance), 0.0)) / baseline)

    def _get_position_sizing(self, signal: Dict, account_balance: float):
        strategy_key = self._source_key(signal)
        if self.rl_sizer:
            regime = "unknown"
            if self.regime_forecaster:
                try:
                    pred = self.regime_forecaster.predict_regime(signal.get("coin", "BTC"))
                    regime = str(pred.get("regime", "unknown"))
                except Exception:
                    regime = "unknown"
            return self.rl_sizer.get_sizing(
                strategy_key=strategy_key,
                account_balance=account_balance,
                signal_confidence=signal.get("confidence", 0.5),
                regime=regime,
                recent_volatility=float(signal.get("volatility", 0.02) or 0.02),
                drawdown_from_peak=self._drawdown_from_initial_balance(account_balance),
            )
        if self.kelly_sizer:
            return self.kelly_sizer.get_sizing(
                strategy_key=strategy_key,
                account_balance=account_balance,
                signal_confidence=signal.get("confidence", 0.5),
            )
        return None

    def _refresh_copy_guardrail_status(self) -> Dict[str, object]:
        if not self.enabled:
            self._copy_guardrail_status = {
                "status": "paused",
                "reason": "Copy trader disabled by config",
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
                "source": "copy_trade",
                "max_concurrent_trades": self.max_concurrent_trades,
                "max_new_trades_per_cycle": self.max_new_trades_per_cycle,
            }
            return dict(self._copy_guardrail_status)

        try:
            closed = db.get_paper_trade_history(limit=250)
            policy = evaluate_source_policy(
                closed,
                source_label="copy_trade",
                min_trades=self.auto_pause_min_closed_trades,
                degrade_win_rate=self.auto_pause_degrade_win_rate,
                block_win_rate=self.auto_pause_block_win_rate,
                block_net_pnl=self.auto_pause_block_net_pnl,
            )
        except Exception as exc:
            policy = {
                "status": "unknown",
                "reason": f"Copy guardrail unavailable: {exc}",
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
                "source": "copy_trade",
            }

        policy["max_concurrent_trades"] = self.max_concurrent_trades
        policy["max_new_trades_per_cycle"] = self.max_new_trades_per_cycle
        self._copy_guardrail_status = dict(policy)
        return dict(self._copy_guardrail_status)

    def get_stats(self) -> Dict:
        status = self._refresh_copy_guardrail_status()
        try:
            open_copy_trades = self._open_copy_trade_count(db.get_open_paper_trades())
        except Exception:
            open_copy_trades = 0
        return {
            "enabled": self.enabled,
            "total_executed": self._copy_count,
            "open_copy_trades": open_copy_trades,
            "guardrail": status,
            "max_concurrent_trades": self.max_concurrent_trades,
            "max_new_trades_per_cycle": self.max_new_trades_per_cycle,
            "sizer_fallback_count": self._sizer_fallback_count,
        }

    def scan_top_traders(self, top_n: int = 10) -> List[Dict]:
        """
        Scan the top N traders by PnL, detect new/changed positions,
        and return copy-trade signals.

        Stale position-cache entries for traders that have dropped out of the
        top-N are evicted after _POSITION_CACHE_TTL_SECONDS so memory doesn't
        grow unboundedly over long run times.
        """
        traders = db.get_active_traders(valid_only=True, quarantine_invalid=True)[:top_n]
        if not traders:
            return []

        # Evict stale cache entries
        now = time.time()
        active_addrs = {t["address"] for t in traders}
        stale = [
            addr for addr, ts in self._position_cache_ts.items()
            if now - ts > _POSITION_CACHE_TTL_SECONDS and addr not in active_addrs
        ]
        for addr in stale:
            self._position_cache.pop(addr, None)
            self._position_cache_ts.pop(addr, None)
        if stale:
            logger.debug("Evicted %d stale position-cache entries", len(stale))

        signals = []
        mids = hl.get_all_mids() or {}

        for trader in traders:
            try:
                addr = trader["address"]
                if not hl._is_valid_eth_address(addr):
                    logger.debug("CopyTrader skipping invalid trader address %s", addr)
                    continue
                state = hl.get_user_state(addr)
                if not state:
                    continue

                current_positions = {}
                for pos in state["positions"]:
                    if float(pos.get("size", 0)) > 0:
                        current_positions[pos["coin"]] = pos

                # Compare with cached positions to find changes
                cached = self._position_cache.get(addr, {})
                new_signals = self._detect_position_changes(
                    addr, cached, current_positions, trader, mids
                )
                signals.extend(new_signals)

                # Update cache + timestamp
                self._position_cache[addr] = current_positions
                self._position_cache_ts[addr] = now

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

        win_rate = trader.get("win_rate", 0)
        normalized_address = str(address or "").strip().lower()

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
                "source_trader": normalized_address,
                "source_pnl": trader.get("total_pnl", 0),
                "confidence": self._calculate_signal_confidence("copy_open", win_rate),
            })

        # Positions closed by the trader (they exited)
        for coin in old_coins - new_coins:
            signals.append({
                "type": "copy_close",
                "coin": coin,
                "source_trader": normalized_address,
            })

        # Existing-position deltas. Emit exactly one signal per coin: flips take
        # precedence because they already imply a full directional re-think.
        for coin in old_coins & new_coins:
            old_size = float(old_positions[coin].get("size", 0))
            new_size = float(new_positions[coin].get("size", 0))
            old_side = old_positions[coin].get("side")
            new_side = new_positions[coin].get("side")
            if old_side != new_side:
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
                    "source_trader": normalized_address,
                    "source_pnl": trader.get("total_pnl", 0),
                    "confidence": self._calculate_signal_confidence("copy_flip", win_rate),
                })
            elif old_size > 0 and new_size > old_size * 1.5:  # 50%+ increase
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
                    "source_trader": normalized_address,
                    "source_pnl": trader.get("total_pnl", 0),
                    "confidence": self._calculate_signal_confidence("copy_scale_in", win_rate),
                })
            elif old_size > 0 and new_size <= old_size * 0.5:  # 50%+ decrease
                signals.append({
                    "type": "copy_scale_out",
                    "coin": coin,
                    "source_trader": normalized_address,
                    "old_size": old_size,
                    "new_size": new_size,
                    "reduction_pct": 1.0 - (new_size / old_size),
                    "confidence": self._calculate_signal_confidence("copy_scale_out", win_rate),
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
            # Keep copy trading de-risked during crashes without locking out
            # genuinely strong candidates before the firewall/rotation layers.
            adjusted_confidence = original_confidence * _CRASH_COPY_CONFIDENCE_MULTIPLIER
            logger.info(
                f"REGIME WEIGHT: crash detected for {coin}, "
                f"reducing copy confidence {original_confidence:.2f} -> {adjusted_confidence:.2f}"
            )
        elif regime == "neutral":
            # Slightly reduce during neutral regimes.
            adjusted_confidence = original_confidence * _NEUTRAL_COPY_CONFIDENCE_MULTIPLIER
            logger.debug(
                f"REGIME WEIGHT: neutral regime for {coin}, "
                f"reducing copy confidence {original_confidence:.2f} -> {adjusted_confidence:.2f}"
            )
        elif regime == "bullish":
            # Increase aggressiveness during bullish regimes, capped at 1.0.
            adjusted_confidence = min(
                original_confidence * _BULLISH_COPY_CONFIDENCE_MULTIPLIER,
                1.0,
            )
            if adjusted_confidence > original_confidence:
                logger.info(
                    f"REGIME WEIGHT: bullish detected for {coin}, "
                    f"boosting copy confidence {original_confidence:.2f} -> {adjusted_confidence:.2f}"
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
        guardrail = self._refresh_copy_guardrail_status()
        allow_new_entries = guardrail.get("status") not in {"blocked", "paused"}
        if not allow_new_entries:
            logger.warning(
                "Copy trader open signals paused: %s",
                guardrail.get("reason", "copy guardrail blocked"),
            )
        new_entries_seen = 0

        for signal in signals:
            try:
                if signal["type"] in ("copy_close", "copy_scale_out"):
                    reason = "source_reduce" if signal["type"] == "copy_scale_out" else "source_exit"
                    closed = self._close_copy_trades(signal, open_trades, mids, close_reason=reason)
                    closed_ids = {trade["trade_id"] for trade in closed}
                    open_trades = [trade for trade in open_trades if trade.get("id") not in closed_ids]
                    continue

                if signal["type"] in ("copy_open", "copy_scale_in", "copy_flip", "golden_copy"):
                    if not allow_new_entries:
                        continue
                    if (
                        self.max_new_trades_per_cycle > 0
                        and new_entries_seen >= self.max_new_trades_per_cycle
                    ):
                        logger.info(
                            "  Copy-trader cycle cap reached (%d/%d); skipping %s %s",
                            new_entries_seen,
                            self.max_new_trades_per_cycle,
                            signal["side"],
                            signal["coin"],
                        )
                        continue
                    signal = self._apply_regime_weight(signal, signal["coin"])
                    source_side_ok, source_side_reason = self._apply_source_side_guard(signal)
                    if not source_side_ok:
                        logger.info(
                            "  Copy source/side guard rejected %s %s: %s",
                            signal["side"],
                            signal["coin"],
                            source_side_reason,
                        )
                        continue
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
                            source_key = self._source_key(signal)
                            weight = self.agent_scorer.get_weight(source_key)
                            trade_signal.confidence = trade_signal.confidence * 0.6 + weight * 0.4

                        # Rotation needs to see candidates even when the book is
                        # full.  Pre-screen only signal-level checks here; final
                        # execution-time validation still runs against the real
                        # post-rotation portfolio state.
                        passed, reason = self.firewall.validate(
                            trade_signal,
                            regime_data=regime_data,
                            open_positions=[],
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
                    new_entries_seen += 1

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

            if self.max_concurrent_trades > 0:
                projected_copy_count = self._open_copy_trade_count(candidate_open_positions)
                if projected_copy_count >= self.max_concurrent_trades:
                    logger.info(
                        "  Copy-trader cap reached (%d/%d open copy trades); skipping %s %s",
                        projected_copy_count,
                        self.max_concurrent_trades,
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

            # CRIT-FIX C4: propagate the firewall's regime_size_modifier onto
            # the dict-shaped signal so _open_copy_trade can scale final size.
            # Without this the crash/neutral-regime size reduction set on
            # `trade_signal.regime_size_modifier` was silently dropped at
            # execution time, letting full-size copies through in crash
            # regimes despite the firewall correctly computing the modifier.
            try:
                signal["regime_size_modifier"] = float(
                    getattr(trade_signal, "regime_size_modifier", 1.0) or 1.0
                )
            except (TypeError, ValueError):
                signal["regime_size_modifier"] = 1.0

            if self.agent_scorer:
                source_key = self._source_key(signal)
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
                        candidate_score=decision.candidate_score,
                        incumbent_score=decision.incumbent_score,
                        reason=decision.reason,
                        new_trade_id=trade.get("id"),
                        closed_trade_event=closed_trade,
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

    @staticmethod
    def _build_trade_metadata(signal: Dict, risk_policy: Dict, execution_role: str) -> Dict:
        """
        Build the trade metadata dict once.

        Previously this was constructed twice inside _open_copy_trade (once for
        the DB call and once for the returned dict), which was a maintenance
        hazard — any new field had to be added in two places.  Now it's built
        here and shared between both usages.
        """
        return {
            "type": signal["type"],
            "source_trader": str(signal.get("source_trader", "") or "").strip().lower(),
            "confidence": signal.get("confidence", 0),
            "signal_id": signal.get("_signal_id", ""),
            "source_key": signal.get("_source_key", ""),
            "source_accuracy": signal.get("source_accuracy", 0),
            "copy_source_size_multiplier": signal.get("copy_source_size_multiplier", 1.0),
            "copy_source_side_policy": signal.get("copy_source_side_policy", {}),
            "regime": signal.get("regime", ""),
            "risk_policy": risk_policy,
            "is_copy_trade": True,
            "is_golden": signal.get("is_golden", False),
            "golden_wallet": signal.get("is_golden", False),
            "source": "copy_trade",
            "execution_role": execution_role,
            "maker_fee_bps": config.PAPER_TRADING_MAKER_FEE_BPS,
            "taker_fee_bps": config.PAPER_TRADING_TAKER_FEE_BPS,
            "intended_entry_price": signal.get("price", 0),
            "entry_slipped_price": signal.get("price", 0),
        }

    def _open_copy_trade(self, account: Dict, signal: Dict, open_trades: List) -> Optional[Dict]:
        """Open a paper trade based on a copy signal."""
        # Kelly-based position sizing if available, else default 5%
        if self.kelly_sizer or self.rl_sizer:
            try:
                sizing = self._get_position_sizing(signal, account["balance"])
                size_usd = sizing.position_usd
            except Exception as exc:
                self._sizer_fallback_count += 1
                logger.warning(
                    "Copy-trader sizing failed for %s %s; using 5%% confidence fallback "
                    "(fallback_count=%d): %s",
                    signal.get("side", "?"),
                    signal.get("coin", "?"),
                    self._sizer_fallback_count,
                    exc,
                )
                size_usd = account["balance"] * 0.05 * signal.get("confidence", 0.5)
        else:
            size_usd = account["balance"] * 0.05 * signal.get("confidence", 0.5)

        # CRIT-FIX C4: apply the firewall's regime size modifier (e.g. 0.60x
        # in crash regimes, 1.20x in bullish) to the final notional.  Neither
        # the Kelly sizer nor the default 5% path is regime-aware, so without
        # this multiplication the firewall's regime protection was effectively
        # bypassed at execution time.  Clamp to [0, 2] to guard against a
        # malformed upstream modifier inflating size uncontrollably.
        regime_mod = float(signal.get("regime_size_modifier", 1.0) or 1.0)
        if regime_mod < 0.0:
            regime_mod = 0.0
        elif regime_mod > 2.0:
            regime_mod = 2.0
        if regime_mod != 1.0:
            logger.debug(
                "Copy trade %s %s: applying regime_size_modifier=%.3f to size_usd $%.2f",
                signal.get("side", "?"), signal.get("coin", "?"), regime_mod, size_usd,
            )
        size_usd = size_usd * regime_mod
        source_side_mod = float(signal.get("copy_source_size_multiplier", 1.0) or 1.0)
        if source_side_mod < 0.0:
            source_side_mod = 0.0
        elif source_side_mod > 1.0:
            source_side_mod = 1.0
        if source_side_mod != 1.0:
            logger.debug(
                "Copy trade %s %s: applying source/side multiplier %.3f to size_usd $%.2f",
                signal.get("side", "?"), signal.get("coin", "?"), source_side_mod, size_usd,
            )
        size_usd = size_usd * source_side_mod

        price = signal["price"]
        if price <= 0:
            return None

        if size_usd <= 0:
            # Regime modifier can legitimately zero out sizing (e.g. extreme
            # crash regime).  Skip silently rather than opening a $0 position.
            logger.info(
                "Copy trade %s %s skipped: regime modifier zeroed size",
                signal.get("side", "?"), signal.get("coin", "?"),
            )
            return None

        size = size_usd / price
        leverage = signal.get("leverage", 2)
        side = signal["side"]
        # Normalize both sides of the conflict comparison.  Upstream producers
        # mix Enum, "LONG"/"long"/"Long" etc., and a case-sensitive "!=" would
        # silently clear a real conflict (letting us open a short next to a
        # long on the same coin).  H12.
        side_norm = (
            side.value if hasattr(side, "value") else str(side)
        ).strip().lower()

        # CRITICAL: No conflicting sides on same asset
        for t in open_trades:
            if t.get("coin") != signal["coin"]:
                continue
            t_side = t.get("side")
            t_side_norm = (
                t_side.value if hasattr(t_side, "value") else str(t_side or "")
            ).strip().lower()
            if t_side_norm and t_side_norm != side_norm:
                logger.debug(f"Copy skip: conflicting side for {signal['coin']}")
                return None

        # Check basic risk: max 5 copy trades per coin (same direction only)
        coin_copies = sum(1 for t in open_trades if t.get("coin") == signal["coin"])
        if coin_copies >= 5:
            return None

        stop_loss, take_profit, risk_policy = self._resolve_copy_trade_risk(
            signal,
            leverage,
            regime_data=signal.get("regime_data"),
        )

        try:
            execution_role = signal.get("execution_role", config.PAPER_TRADING_DEFAULT_EXECUTION_ROLE)
            # Build metadata once, reuse for both the DB record and the return value
            metadata = self._build_trade_metadata(signal, risk_policy, execution_role)

            trade_id = db.open_paper_trade(
                strategy_id=None,
                coin=signal["coin"],
                side=side,
                entry_price=price,
                size=size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
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
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_id": None,
                "strategy_type": "copy_trade",
                "confidence": signal.get("confidence", 0),
                "trader_address": signal.get("source_trader", ""),
                "source": "copy_trade",
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
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

        # CRIT-FIX C2: atomic close + account credit in one transaction
        # (see src/data/database.py::close_paper_trade_and_credit_account).
        if not db.close_paper_trade_and_credit_account(trade["id"], exit_price, pnl):
            logger.error(
                "Copy trade close failed for trade %s (%s %s) -- "
                "already closed or account row missing; no credit applied.",
                trade["id"], trade.get("side", "?"), trade.get("coin", "?"),
            )
            return None

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
        source_key = str(meta.get("source_key", "") or "").strip().lower()
        if not source_key:
            source_key = self._source_key(meta)

        # C8: every post-close hook is best-effort but failures used to be
        # swallowed silently; when the scorer or kelly sizer drifts into an
        # unusable state the operator had no breadcrumb.  Log at warning level
        # with the specific subsystem name so the root cause is greppable.
        if self.agent_scorer and meta.get("source_trader"):
            try:
                self.agent_scorer.record_outcome(source_key, meta.get("signal_id", ""), pnl, return_pct)
            except Exception as hook_exc:
                logger.warning("copy_trader post-close hook 'agent_scorer' failed: %s", hook_exc)
        if self.firewall:
            try:
                self.firewall.record_trade_outcome(trade["coin"], pnl)
            except Exception as hook_exc:
                logger.warning("copy_trader post-close hook 'firewall' failed: %s", hook_exc)

        if self.kelly_sizer:
            try:
                self.kelly_sizer.record_outcome(
                    strategy_key=source_key,
                    pnl=pnl,
                    entry_price=trade["entry_price"],
                    size=trade["size"],
                    leverage=trade.get("leverage", 1),
                )
            except Exception as hook_exc:
                logger.warning("copy_trader post-close hook 'kelly_sizer' failed: %s", hook_exc)

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
            except Exception as hook_exc:
                logger.warning("copy_trader post-close hook 'calibration' failed: %s", hook_exc)

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
                    closed_at=datetime.now(timezone.utc).isoformat(),
                    confidence=meta.get("confidence", 0),
                    source=source_key,
                )
            except Exception as hook_exc:
                logger.warning("copy_trader post-close hook 'trade_memory' failed: %s", hook_exc)

        if self.shadow_tracker:
            try:
                self.shadow_tracker.record_trade(
                    {
                        "signal_source": source_key,
                        "coin": trade["coin"],
                        "side": trade.get("side", ""),
                        "entry_price": trade["entry_price"],
                        "exit_price": exit_price,
                        "size": trade["size"],
                        "entry_ts": trade.get("opened_at") or None,
                        "exit_ts": datetime.now(timezone.utc).isoformat(),
                        "pnl": pnl,
                        "pnl_pct": return_pct * 100,
                        "regime_at_entry": meta.get("regime") or None,
                        "confidence": meta.get("confidence", 0),
                        "metadata": {
                            "source_trader": meta.get("source_trader", ""),
                            "signal_id": meta.get("signal_id", ""),
                        },
                    }
                )
            except Exception as hook_exc:
                logger.warning("copy_trader post-close hook 'shadow_tracker' failed: %s", hook_exc)

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
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if self.rotation_manager:
                self.rotation_manager.record_trade_close(closed_event)
        except Exception as hook_exc:
            logger.warning("copy_trader post-close hook 'rotation_manager' failed: %s", hook_exc)
        self._closed_events.append(closed_event)
        return closed_event

    def _close_copy_trades(
        self,
        signal: Dict,
        open_trades: List,
        mids: Dict,
        close_reason: str = "source_exit",
    ) -> List[Dict]:
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
                str(meta.get("source_trader", "") or "").strip().lower()
                == str(signal.get("source_trader", "") or "").strip().lower()):

                current_price = float(mids.get(trade["coin"], 0))
                if current_price <= 0:
                    continue
                closed_trade = self._close_trade(
                    trade,
                    exit_price=current_price,
                    close_reason=close_reason,
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
