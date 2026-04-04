"""
Kelly Criterion Position Sizing
================================
Dynamic, mathematically optimal position sizing based on each strategy's
actual historical win rate and reward-to-risk ratio.

Replaces the static 8% position sizing with adaptive sizing that:
  - Bets more on strategies with proven edge
  - Bets less (or nothing) on strategies with no proven edge
  - Uses Half-Kelly for safety (halves the optimal fraction)
  - Hard caps at configurable maximum per trade

The Kelly Formula:
  f* = W - [(1 - W) / R]
  Where:
    W = win rate (probability of winning)
    R = reward-to-risk ratio (avg_win / avg_loss)
    f* = optimal fraction of bankroll to risk

We use Half-Kelly (f* × 0.5) because:
  - Full Kelly assumes perfect knowledge of probabilities
  - In practice, our estimates have uncertainty
  - Half-Kelly dramatically reduces drawdown risk
"""
import logging
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of Kelly Criterion calculation."""
    strategy_key: str
    kelly_fraction: float      # Raw Kelly fraction
    safe_fraction: float       # After Half-Kelly + caps
    position_pct: float        # Final position % of portfolio
    position_usd: float        # USD amount to risk
    win_rate: float
    reward_risk_ratio: float
    has_edge: bool             # Whether Kelly says there's an edge
    trades_used: int           # How many trades went into the calculation
    confidence: str            # "high", "medium", "low", "insufficient"

    def to_dict(self) -> Dict:
        return asdict(self)


class KellySizer:
    """
    Computes position sizes using the Kelly Criterion.

    Uses historical trade data per strategy/source to determine
    mathematically optimal bet sizes.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Kelly multiplier — controls aggressiveness of sizing:
        #   1.0  = Full Kelly  (theoretical max growth, very volatile)
        #   0.5  = Half-Kelly  (classic conservative choice)
        #   0.25 = Quarter-Kelly (very conservative, recommended for crypto)
        # Default: Quarter-Kelly for crypto perps (high-variance environment)
        self.kelly_multiplier = cfg.get("kelly_multiplier", 0.25)

        # Volatility-adjusted Kelly: dynamically reduce multiplier when
        # recent returns have high variance (set False for fixed fraction)
        self.vol_adjusted_kelly = cfg.get("vol_adjusted_kelly", True)
        self.vol_lookback = cfg.get("vol_lookback", 30)  # trades for vol calc
        self.vol_target = cfg.get("vol_target", 0.02)    # target per-trade vol

        # Position size caps
        self.max_position_pct = cfg.get("max_position_pct", 0.08)    # 8% max per trade (was 10%)
        self.min_position_pct = cfg.get("min_position_pct", 0.01)    # 1% min (if trading at all)
        self.default_position_pct = cfg.get("default_position_pct", 0.03)  # 3% for unknown (was 4%)

        # Minimum trades needed for Kelly to be meaningful
        self.min_trades_for_kelly = cfg.get("min_trades_for_kelly", 15)
        self.min_trades_for_high_confidence = cfg.get("min_trades_for_high_confidence", 50)

        # Confidence scaling: reduce size based on data confidence
        self.confidence_scaling = cfg.get("confidence_scaling", True)

        # Track per-strategy trade outcomes for Kelly computation
        self._strategy_outcomes: Dict[str, list] = {}

        logger.info(f"KellySizer initialized: multiplier={self.kelly_multiplier}, "
                    f"max={self.max_position_pct:.0%}, min_trades={self.min_trades_for_kelly}")

    def record_outcome(self, strategy_key: str, pnl: float,
                        entry_price: float, size: float, leverage: float):
        """
        Record a trade outcome for Kelly calculation.

        Args:
            strategy_key: e.g. "momentum_long", "copy_trade:0xabc", "liquidation_reversal"
            pnl: Absolute PnL in USD
            entry_price: Entry price
            size: Position size (units)
            leverage: Leverage used
        """
        if strategy_key not in self._strategy_outcomes:
            self._strategy_outcomes[strategy_key] = []

        notional = entry_price * size * leverage
        return_pct = pnl / notional if notional > 0 else 0

        self._strategy_outcomes[strategy_key].append({
            "pnl": pnl,
            "return_pct": return_pct,
            "win": pnl > 0,
        })

        # Keep last 200 trades
        if len(self._strategy_outcomes[strategy_key]) > 200:
            self._strategy_outcomes[strategy_key] = self._strategy_outcomes[strategy_key][-200:]

    def calculate_kelly(self, win_rate: float,
                         reward_risk_ratio: float) -> float:
        """
        Core Kelly Criterion calculation.

        Args:
            win_rate: Probability of winning (0-1)
            reward_risk_ratio: avg_win / avg_loss (positive number)

        Returns:
            Kelly fraction (0-1), or 0 if no edge.
        """
        if win_rate <= 0 or win_rate >= 1 or reward_risk_ratio <= 0:
            return 0.0

        # Kelly Formula: f* = W - [(1 - W) / R]
        kelly = win_rate - ((1.0 - win_rate) / reward_risk_ratio)

        if kelly <= 0:
            return 0.0  # No edge — don't bet

        # Apply fractional Kelly multiplier
        safe_kelly = kelly * self.kelly_multiplier

        return safe_kelly

    def _vol_adjusted_multiplier(self, strategy_key: str) -> float:
        """
        Dynamically adjust Kelly multiplier based on recent return volatility.
        When volatility is high, reduce sizing; when low, allow more.
        """
        outcomes = self._strategy_outcomes.get(strategy_key, [])
        recent = outcomes[-self.vol_lookback:] if len(outcomes) > self.vol_lookback else outcomes
        if len(recent) < 5:
            return self.kelly_multiplier

        returns = [o["return_pct"] for o in recent]
        vol = float(np.std(returns)) if len(returns) > 1 else 0.02
        if vol <= 0:
            return self.kelly_multiplier

        # Scale: if actual vol > target, reduce multiplier proportionally
        adjustment = min(self.vol_target / vol, 1.5)  # cap at 1.5x
        adjusted = self.kelly_multiplier * adjustment
        return max(0.05, min(adjusted, 0.75))  # bounds: 5% to 75% of full Kelly

    def get_sizing(self, strategy_key: str,
                    account_balance: float,
                    signal_confidence: float = 0.5) -> SizingResult:
        """
        Get the optimal position size for a strategy.

        Args:
            strategy_key: Strategy identifier
            account_balance: Current account balance
            signal_confidence: Current signal's confidence (0-1)

        Returns:
            SizingResult with the recommended position size.
        """
        outcomes = self._strategy_outcomes.get(strategy_key, [])
        n_trades = len(outcomes)

        # Not enough data — use default sizing scaled by confidence
        if n_trades < self.min_trades_for_kelly:
            position_pct = self.default_position_pct * signal_confidence
            position_pct = max(self.min_position_pct, min(position_pct, self.max_position_pct))
            return SizingResult(
                strategy_key=strategy_key,
                kelly_fraction=0.0,
                safe_fraction=0.0,
                position_pct=position_pct,
                position_usd=account_balance * position_pct,
                win_rate=0.0,
                reward_risk_ratio=0.0,
                has_edge=False,
                trades_used=n_trades,
                confidence="insufficient",
            )

        # Calculate win rate and reward-to-risk from historical trades
        wins = [o for o in outcomes if o["win"]]
        losses = [o for o in outcomes if not o["win"]]

        win_rate = len(wins) / n_trades
        avg_win = sum(abs(o["return_pct"]) for o in wins) / len(wins) if wins else 0
        # HIGH-FIX HIGH-1: when there are no losses yet (100% win rate history),
        # do NOT fall back to 0.001 — that inflates reward_risk_ratio to thousands
        # and causes Kelly to return max fraction despite no real edge data on
        # downside.  Instead, mark has_edge=False and use default sizing so we
        # wait for actual loss data before trusting the R:R estimate.
        if not losses:
            logger.info(
                f"Kelly [{strategy_key}]: no loss history yet (all {n_trades} trades won) — "
                f"using default sizing until loss data is available."
            )
            position_pct = self.default_position_pct * signal_confidence
            position_pct = max(self.min_position_pct, min(position_pct, self.max_position_pct))
            return SizingResult(
                strategy_key=strategy_key,
                kelly_fraction=0.0,
                safe_fraction=0.0,
                position_pct=position_pct,
                position_usd=account_balance * position_pct,
                win_rate=win_rate,
                reward_risk_ratio=0.0,
                has_edge=False,
                trades_used=n_trades,
                confidence="insufficient_loss_history",
            )
        avg_loss = sum(abs(o["return_pct"]) for o in losses) / len(losses)

        reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Compute Kelly
        kelly_fraction = self.calculate_kelly(win_rate, reward_risk_ratio)
        has_edge = kelly_fraction > 0

        # Determine confidence level
        if n_trades >= self.min_trades_for_high_confidence:
            confidence_level = "high"
            confidence_factor = 1.0
        elif n_trades >= self.min_trades_for_kelly:
            confidence_level = "medium"
            confidence_factor = 0.7  # Reduce size for medium confidence
        else:
            confidence_level = "low"
            confidence_factor = 0.4

        # Apply volatility-adjusted Kelly if enabled
        if self.vol_adjusted_kelly and has_edge:
            effective_multiplier = self._vol_adjusted_multiplier(strategy_key)
            # Recompute kelly with adjusted multiplier
            raw_kelly = win_rate - ((1.0 - win_rate) / reward_risk_ratio)
            kelly_fraction = max(0, raw_kelly * effective_multiplier)

        # Calculate final position size
        if has_edge:
            safe_fraction = kelly_fraction * confidence_factor
            # Also scale by current signal confidence
            position_pct = safe_fraction * (0.5 + signal_confidence * 0.5)
        else:
            safe_fraction = 0.0
            # No proven edge — use minimum or skip
            position_pct = self.min_position_pct * signal_confidence

        # Apply caps
        position_pct = max(self.min_position_pct, min(position_pct, self.max_position_pct))
        position_usd = account_balance * position_pct

        logger.debug(f"Kelly [{strategy_key}]: WR={win_rate:.0%}, R:R={reward_risk_ratio:.2f}, "
                     f"kelly={kelly_fraction:.3f}, final={position_pct:.1%} (${position_usd:.0f})")

        return SizingResult(
            strategy_key=strategy_key,
            kelly_fraction=kelly_fraction,
            safe_fraction=safe_fraction,
            position_pct=position_pct,
            position_usd=position_usd,
            win_rate=win_rate,
            reward_risk_ratio=reward_risk_ratio,
            has_edge=has_edge,
            trades_used=n_trades,
            confidence=confidence_level,
        )

    def get_all_sizing_stats(self) -> Dict[str, Dict]:
        """Get sizing stats for all tracked strategies."""
        stats = {}
        for key in self._strategy_outcomes:
            outcomes = self._strategy_outcomes[key]
            n = len(outcomes)
            if n == 0:
                continue
            wins = sum(1 for o in outcomes if o["win"])
            total_pnl = sum(o["pnl"] for o in outcomes)
            stats[key] = {
                "trades": n,
                "win_rate": wins / n,
                "total_pnl": round(total_pnl, 2),
                # HIGH-FIX HIGH-1: use 0 (no edge) when no losses yet, same as get_sizing
                "has_edge": (
                    self.calculate_kelly(
                        wins / n,
                        (sum(abs(o["return_pct"]) for o in outcomes if o["win"]) / max(wins, 1)) /
                        max(sum(abs(o["return_pct"]) for o in outcomes if not o["win"]) / max(n - wins, 1), 1e-9)
                    ) > 0
                    if (n >= self.min_trades_for_kelly and (n - wins) > 0)
                    else False
                ),
            }
        return stats

    def load_from_agent_scorer(self, agent_scorer) -> None:
        """
        Bootstrap Kelly data from existing AgentScorer trade history.
        Call this on startup to populate historical outcomes.
        """
        try:
            for source_key, history in agent_scorer._trade_history.items():
                completed = [t for t in history if t.get("pnl") is not None]
                for trade in completed:
                    pnl = trade.get("pnl", 0)
                    return_pct = trade.get("return_pct", 0)
                    if source_key not in self._strategy_outcomes:
                        self._strategy_outcomes[source_key] = []
                    self._strategy_outcomes[source_key].append({
                        "pnl": pnl,
                        "return_pct": return_pct,
                        "win": pnl > 0,
                    })
            logger.info(f"KellySizer bootstrapped from AgentScorer: "
                        f"{len(self._strategy_outcomes)} strategies loaded")
        except Exception as e:
            logger.debug(f"Could not bootstrap from AgentScorer: {e}")
