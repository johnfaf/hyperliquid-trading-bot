"""
LLM Filter Layer (Hybrid Quant + LLM)
=======================================
The final confirmation layer in the V2 pipeline.

Philosophy:
  - Quant models generate probabilities (math)
  - LLM interprets the complex environment (reasoning)
  - LLM does NOT predict price — it FILTERS bad setups

The LLM asks: "Given everything I know about this setup, is there a reason
NOT to take this trade?"

This is a stub/rule-based implementation that can be upgraded to actual
LLM calls (OpenAI, DeepSeek, Claude) when ready. The rule-based version
captures the same logic patterns an LLM would use.

Filter checks:
  1. Regime contradiction — signal opposes current regime
  2. Memory warning — similar past trades lost money
  3. Multi-signal conflict — different sources disagree
  4. Exhaustion trap — signal is chasing an already-extended move
  5. Risk cluster — too many positions in correlated assets
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMFilter:
    """
    Rule-based "LLM-like" filter that catches bad setups
    the quantitative models might miss.

    Upgradeable to actual LLM API calls in the future.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Enable/disable individual checks
        self.check_regime = cfg.get("check_regime", True)
        self.check_memory = cfg.get("check_memory", True)
        self.check_conflicts = cfg.get("check_conflicts", True)
        self.check_exhaustion = cfg.get("check_exhaustion", True)
        self.check_correlation = cfg.get("check_correlation", True)

        # Thresholds
        self.memory_avoid_threshold = cfg.get("memory_avoid_threshold", 0.30)  # WR below 30% = block
        self.exhaustion_rsi_long = cfg.get("exhaustion_rsi_long", 78)  # Don't long above this RSI
        self.exhaustion_rsi_short = cfg.get("exhaustion_rsi_short", 22)  # Don't short below this RSI
        self.max_correlated_positions = cfg.get("max_correlated_positions", 3)

        # Correlated asset groups
        self.correlation_groups = {
            "l1": {"BTC", "ETH", "SOL", "AVAX", "NEAR", "APT", "SUI"},
            "defi": {"LINK", "UNI", "AAVE", "MKR", "SNX"},
            "l2": {"ARB", "OP", "MATIC", "MANTA", "STRK"},
            "meme": {"DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI"},
            "ai": {"FET", "RENDER", "TAO", "NEAR"},
        }

        # Stats
        self.stats = {
            "total_filtered": 0,
            "passed": 0,
            "blocked_regime": 0,
            "blocked_memory": 0,
            "blocked_conflict": 0,
            "blocked_exhaustion": 0,
            "blocked_correlation": 0,
        }

        logger.info("LLMFilter initialized (rule-based mode)")

    def filter(self, signal: Dict, context: Dict) -> Tuple[bool, float, str]:
        """
        Filter a trade signal through LLM-like reasoning.

        Args:
            signal: The trade signal dict with coin, side, confidence, features, etc.
            context: Additional context:
                - regime_data: Current market regime
                - memory_result: SimilarityResult from TradeMemory
                - open_positions: List of current open positions
                - all_signals: All signals being considered this cycle

        Returns:
            (approved: bool, adjusted_confidence: float, reason: str)
        """
        self.stats["total_filtered"] += 1
        coin = signal.get("coin", "")
        side = signal.get("side", "")
        confidence = signal.get("confidence", 0.5)
        features = signal.get("features", {})

        reasons = []

        # ─── Check 1: Regime Contradiction ───────────────────────
        if self.check_regime:
            regime_data = context.get("regime_data", {})
            regime = regime_data.get("overall_regime", "").upper()

            if regime == "TRENDING_UP" and side == "short":
                confidence *= 0.6
                reasons.append(f"contra-regime: shorting in TRENDING_UP")
            elif regime == "TRENDING_DOWN" and side == "long":
                confidence *= 0.6
                reasons.append(f"contra-regime: longing in TRENDING_DOWN")
            elif regime == "VOLATILE":
                confidence *= 0.8
                reasons.append("volatile regime: reduced confidence")

        # ─── Check 2: Memory Warning ────────────────────────────
        if self.check_memory:
            memory_result = context.get("memory_result")
            if memory_result:
                rec = memory_result.get("recommendation", "proceed") if isinstance(memory_result, dict) else getattr(memory_result, "recommendation", "proceed")

                if rec == "avoid":
                    self.stats["blocked_memory"] += 1
                    reason_text = memory_result.get("reason", "") if isinstance(memory_result, dict) else getattr(memory_result, "reason", "")
                    return False, 0, f"Memory block: {reason_text}"
                elif rec == "caution":
                    confidence *= 0.75
                    reasons.append("memory caution: similar trades had mixed results")

        # ─── Check 3: Multi-Signal Conflict ──────────────────────
        if self.check_conflicts:
            all_signals = context.get("all_signals", [])
            opposing = [
                s for s in all_signals
                if s.get("coin") == coin and s.get("side") != side
            ]
            if opposing:
                # Different sources disagree on this coin
                confidence *= 0.7
                reasons.append(f"signal conflict: {len(opposing)} opposing signals for {coin}")

        # ─── Check 4: Exhaustion Trap ────────────────────────────
        if self.check_exhaustion:
            rsi = features.get("rsi", 50)
            bb_pos = features.get("bollinger_position", 0)

            if side == "long" and rsi > self.exhaustion_rsi_long:
                self.stats["blocked_exhaustion"] += 1
                return False, 0, f"Exhaustion block: longing with RSI={rsi:.0f} (>{self.exhaustion_rsi_long})"

            if side == "short" and rsi < self.exhaustion_rsi_short:
                self.stats["blocked_exhaustion"] += 1
                return False, 0, f"Exhaustion block: shorting with RSI={rsi:.0f} (<{self.exhaustion_rsi_short})"

            # Bollinger extreme warning
            if side == "long" and bb_pos > 0.9:
                confidence *= 0.75
                reasons.append(f"near upper Bollinger (pos={bb_pos:.2f})")
            elif side == "short" and bb_pos < -0.9:
                confidence *= 0.75
                reasons.append(f"near lower Bollinger (pos={bb_pos:.2f})")

        # ─── Check 5: Correlation Cluster ────────────────────────
        if self.check_correlation:
            open_positions = context.get("open_positions", [])
            coin_group = self._get_correlation_group(coin)

            if coin_group:
                same_group_positions = [
                    p for p in open_positions
                    if self._get_correlation_group(p.get("coin", "")) == coin_group
                    and p.get("side") == side
                ]
                if len(same_group_positions) >= self.max_correlated_positions:
                    self.stats["blocked_correlation"] += 1
                    return False, 0, (f"Correlation block: {len(same_group_positions)} "
                                      f"{side} positions in {coin_group} group already")

        # ─── Final Decision ──────────────────────────────────────

        # If confidence was reduced too much, block
        if confidence < 0.20:
            reason_str = " | ".join(reasons) if reasons else "combined filters"
            return False, confidence, f"Confidence too low after filters: {reason_str}"

        self.stats["passed"] += 1

        if reasons:
            logger.debug(f"LLMFilter [{coin} {side}]: passed with adjustments — {', '.join(reasons)}")

        return True, confidence, "approved" + (f" ({', '.join(reasons)})" if reasons else "")

    def _get_correlation_group(self, coin: str) -> Optional[str]:
        """Get the correlation group for a coin."""
        for group_name, coins in self.correlation_groups.items():
            if coin in coins:
                return group_name
        return None

    def get_stats(self) -> Dict:
        """Return filter statistics."""
        total = self.stats["total_filtered"]
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / total if total > 0 else 0,
            "block_rate": 1 - (self.stats["passed"] / total) if total > 0 else 0,
        }
