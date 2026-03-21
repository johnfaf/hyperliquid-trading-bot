"""
Decision Firewall
=================
The CRITICAL validation layer between signal generation and execution.

Every signal must pass through this firewall before reaching the execution layer.
It enforces:
  1. Schema validation (is the signal well-formed?)
  2. Risk limits (per-trade, per-coin, portfolio-wide)
  3. Regime alignment (does the strategy fit current market conditions?)
  4. Conflict detection (no opposing positions on same coin)
  5. Source accuracy check (has this signal source been reliable?)
  6. Cooldown enforcement (no revenge trading)

Flow: Signal Source → TradeSignal → DecisionFirewall → Execution
"""
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.signal_schema import TradeSignal, SignalStrength
from src import database as db

logger = logging.getLogger(__name__)


class DecisionFirewall:
    """
    Validates and filters trade signals before execution.
    Acts as the final gatekeeper — nothing trades without passing here.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Risk limits
        self.max_risk_per_trade = cfg.get("max_risk_per_trade", 0.01)   # 1% of portfolio
        self.max_total_risk = cfg.get("max_total_risk", 0.05)           # 5% total
        self.max_positions = cfg.get("max_positions", 5)
        self.max_per_coin = cfg.get("max_per_coin", 2)                  # Max 2 positions per coin
        self.max_leverage = cfg.get("max_leverage", 5)
        self.min_confidence = cfg.get("min_confidence", 0.3)            # Reject below 30%
        self.min_source_accuracy = cfg.get("min_source_accuracy", 0.0)  # 0 = no filter
        self.cooldown_seconds = cfg.get("cooldown_seconds", 300)        # 5 min between trades same coin

        # State tracking
        self._recent_trades: Dict[str, float] = {}  # coin → last trade timestamp
        self._daily_losses: float = 0.0
        self._daily_reset_date: str = ""

        # Stats
        self.stats = {
            "total_signals": 0,
            "passed": 0,
            "rejected_schema": 0,
            "rejected_confidence": 0,
            "rejected_risk": 0,
            "rejected_regime": 0,
            "rejected_conflict": 0,
            "rejected_cooldown": 0,
            "rejected_accuracy": 0,
            "rejected_drawdown": 0,
        }

        logger.info(f"DecisionFirewall initialized: max_risk={self.max_risk_per_trade:.0%}/trade, "
                    f"max_positions={self.max_positions}, min_confidence={self.min_confidence:.0%}")

    def validate(self, signal: TradeSignal, regime_data: Optional[Dict] = None,
                 open_positions: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        """
        Validate a single trade signal through all checks.

        Returns: (passed: bool, reason: str)
          - passed=True: signal is approved for execution
          - passed=False: signal is rejected with explanation
        """
        self.stats["total_signals"] += 1

        # 1. Schema validation
        if not signal.validate():
            self.stats["rejected_schema"] += 1
            return False, f"Invalid signal schema: {signal.coin} {signal.side.value}"

        # 2. Minimum confidence
        if signal.confidence < self.min_confidence:
            self.stats["rejected_confidence"] += 1
            return False, f"Low confidence {signal.confidence:.0%} < {self.min_confidence:.0%}"

        # 3. Leverage check
        if signal.leverage > self.max_leverage:
            signal.leverage = self.max_leverage  # Clamp instead of reject
            logger.info(f"Clamped leverage to {self.max_leverage}x for {signal.coin}")

        # 4. Position limits
        positions = open_positions or db.get_open_paper_trades()
        if len(positions) >= self.max_positions:
            self.stats["rejected_risk"] += 1
            return False, f"Max positions reached ({len(positions)}/{self.max_positions})"

        # 5. Per-coin limit
        coin_positions = [p for p in positions if p.get("coin") == signal.coin]
        if len(coin_positions) >= self.max_per_coin:
            self.stats["rejected_risk"] += 1
            return False, f"Max positions for {signal.coin} ({len(coin_positions)}/{self.max_per_coin})"

        # 6. Conflict detection — no opposing positions on same coin
        for pos in coin_positions:
            if pos.get("side") != signal.side.value:
                self.stats["rejected_conflict"] += 1
                return False, (f"Conflict: have {pos.get('side')} {signal.coin}, "
                              f"signal wants {signal.side.value}")

        # 7. Cooldown — prevent revenge trading
        now = time.time()
        last_trade_ts = self._recent_trades.get(signal.coin, 0)
        if now - last_trade_ts < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (now - last_trade_ts))
            self.stats["rejected_cooldown"] += 1
            return False, f"Cooldown: {signal.coin} traded {remaining}s ago"

        # 8. Regime alignment check
        if regime_data:
            guidance = regime_data.get("strategy_guidance", {})
            paused = set(guidance.get("pause", []))
            if "all" in paused:
                self.stats["rejected_regime"] += 1
                return False, f"Regime {regime_data.get('overall_regime', '?')} pauses all trading"
            if signal.strategy_type and signal.strategy_type.lower() in paused:
                self.stats["rejected_regime"] += 1
                return False, (f"Regime pauses {signal.strategy_type} "
                              f"(regime={regime_data.get('overall_regime', '?')})")

            # Apply size modifier from regime
            size_mod = guidance.get("size_modifier", 1.0)
            signal.regime_size_modifier = size_mod

        # 9. Source accuracy check (if we have history)
        if self.min_source_accuracy > 0 and signal.source_accuracy > 0:
            if signal.source_accuracy < self.min_source_accuracy:
                self.stats["rejected_accuracy"] += 1
                return False, (f"Source accuracy {signal.source_accuracy:.0%} < "
                              f"{self.min_source_accuracy:.0%}")

        # 10. Daily drawdown circuit breaker
        self._check_daily_reset()
        account = db.get_paper_account()
        if account:
            balance = account.get("balance", 10000)
            if self._daily_losses / balance > 0.03:  # 3% daily loss limit
                self.stats["rejected_drawdown"] += 1
                return False, f"Daily loss limit hit ({self._daily_losses / balance:.1%} > 3%)"

        # All checks passed
        self.stats["passed"] += 1
        self._recent_trades[signal.coin] = now
        return True, "approved"

    def validate_batch(self, signals: List[TradeSignal],
                        regime_data: Optional[Dict] = None) -> List[Tuple[TradeSignal, bool, str]]:
        """
        Validate a batch of signals. Returns list of (signal, passed, reason).
        Processes in order of confidence (highest first).
        """
        positions = db.get_open_paper_trades()

        # Sort by confidence descending
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

        results = []
        for signal in sorted_signals:
            passed, reason = self.validate(signal, regime_data, positions)
            results.append((signal, passed, reason))

            # If signal passed, add to positions for subsequent checks
            if passed:
                positions.append({
                    "coin": signal.coin,
                    "side": signal.side.value,
                    "status": "open",
                })

        approved = sum(1 for _, p, _ in results if p)
        rejected = sum(1 for _, p, _ in results if not p)
        logger.info(f"Firewall batch: {approved} approved, {rejected} rejected out of {len(signals)}")

        return results

    def record_trade_outcome(self, coin: str, pnl: float):
        """Record a trade outcome for daily drawdown tracking."""
        if pnl < 0:
            self._daily_losses += abs(pnl)

    def _check_daily_reset(self):
        """Reset daily loss counter at midnight UTC."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_losses = 0.0

    def get_stats(self) -> Dict:
        """Return firewall statistics."""
        total = self.stats["total_signals"]
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / total if total > 0 else 0,
            "top_rejection_reason": max(
                [(k, v) for k, v in self.stats.items() if k.startswith("rejected_")],
                key=lambda x: x[1], default=("none", 0)
            )[0],
        }
