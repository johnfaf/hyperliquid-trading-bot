"""
Adaptive Bot Detector
=====================
Replaces hardcoded bot detection with a continuous probability scoring system.
Each feature contributes a weighted probability rather than a binary signal.

The final bot_probability (0.0 to 1.0) allows nuanced decisions:
- < 0.3: Likely human
- 0.3 - 0.6: Uncertain (monitor closely)
- 0.6 - 0.85: Likely bot (exclude from strategy analysis, keep in DB)
- > 0.85: Almost certainly bot (mark inactive)

This enables:
1. Continuous scoring without binary thresholds
2. Contextual interpretation (e.g., 49 vs 51 trades/day)
3. Configurable thresholds per use case
4. Better false positive/negative balance
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics

import sys
import os as os_module
sys.path.insert(0, os_module.path.join(os_module.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


@dataclass
class BotResult:
    """Result of bot detection analysis."""
    bot_probability: float  # 0.0 to 1.0
    is_bot: bool            # True if probability > threshold
    confidence: float       # How confident we are in the classification (0.0-1.0)
    signals: Dict[str, float]  # Individual feature scores (each 0.0-1.0)
    reason: str            # Human-readable explanation


class AdaptiveBotDetector:
    """
    Adaptive bot detection using continuous probability scoring.
    Replaces binary signal counting with weighted feature contributions.
    """

    def __init__(self, threshold: Optional[float] = None):
        """
        Initialize the detector.

        Args:
            threshold: Bot probability threshold (default from env or 0.45).
                      Probabilities >= threshold are classified as bots.

        ★ AUDIT FIX (May 2026): the original threshold of 0.60 and even
        weights produced a 5% bot rate on the 816-trader pool (41 bots)
        -- far below the 20-40% industry norm for crypto perps DEXs.
        Observed cases like 0x261af68f (932 trades/day, classified
        "Likely human" with prob=0.27) showed that the previous weights
        let a maxed-out ``trade_frequency`` signal contribute only 0.25
        to the total, below threshold even though 932 trades/day is
        physically impossible for a human.

        Retuning:

          * Threshold 0.60 -> 0.45.  A continuous-prob score with weighted
            features doesn't get to 0.60 unless multiple feature scorers
            agree at high values; 0.45 captures the "uncertain" band
            which observation showed contained many obvious bots.
          * trade_frequency 0.25 -> 0.35 (strongest signal).
          * pnl_pattern     0.20 -> 0.25 (MM / arb detection).
          * session_pattern 0.10 -> 0.05 (24/7 crypto trading is normal
            for humans following Telegram/TradingView signals).
          * size_uniformity 0.15 -> 0.10 (its midpoint=0.05 is too tight
            in practice; reduced weight here + relaxed midpoint in
            ``_score_size_uniformity``).
          * timing_regularity 0.20 -> 0.15.
          * liquidation_rate unchanged at 0.10.  Sum: 1.00.

        Every tunable is env-var overridable so the operator can
        re-tighten without redeploying.
        """
        if threshold is None:
            threshold = float(os.environ.get("BOT_PROB_THRESHOLD", "0.45"))
        self.threshold = threshold
        self.weights = {
            "trade_frequency":   float(os.environ.get("BOT_WEIGHT_TRADE_FREQUENCY",   "0.35")),
            "timing_regularity": float(os.environ.get("BOT_WEIGHT_TIMING_REGULARITY", "0.15")),
            "size_uniformity":   float(os.environ.get("BOT_WEIGHT_SIZE_UNIFORMITY",   "0.10")),
            "pnl_pattern":       float(os.environ.get("BOT_WEIGHT_PNL_PATTERN",       "0.25")),
            "liquidation_rate":  float(os.environ.get("BOT_WEIGHT_LIQUIDATION_RATE",  "0.10")),
            "session_pattern":   float(os.environ.get("BOT_WEIGHT_SESSION_PATTERN",   "0.05")),
        }
        logger.info(
            f"AdaptiveBotDetector initialized with threshold={self.threshold:.2f}, "
            f"weights={ {k: round(v, 2) for k, v in self.weights.items()} }"
        )

    def detect(
        self,
        fills: List[Dict],
        positions: List[Dict],
        trade_analysis: Dict,
        address: str
    ) -> BotResult:
        """
        Detect if an account is likely a bot using adaptive probability scoring.

        Args:
            fills: List of trade fill dictionaries
            positions: List of open positions
            trade_analysis: Pre-computed trade analysis dict
            address: Trader's wallet address (for logging)

        Returns:
            BotResult with probability, classification, and breakdown
        """
        if not fills:
            return BotResult(
                bot_probability=0.0,
                is_bot=False,
                confidence=0.5,
                signals={},
                reason="No trade data available"
            )

        addr_short = address[:10] if address else "unknown"
        signals = {}

        # ★ AUDIT FIX: hard-cutoff short-circuit.  Previously the
        # ``BOT_HARD_CUTOFF_TRADES`` config only capped the
        # ``trade_frequency`` feature score at 1.0, but the weighted
        # contribution was still bounded by the feature weight (0.25 of
        # the total).  A trader doing 932 trades/day could therefore
        # score prob=0.25 and pass as "Likely human" if no other
        # feature fired -- which is exactly what we observed on
        # 0x261af68f.  The hard cutoff is meant to be a *certainty
        # gate*: above it, no further evidence is needed.  Force the
        # whole verdict to is_bot=True before the weighted aggregation.
        tpd = self._compute_trades_per_day(fills)
        if tpd > config.BOT_HARD_CUTOFF_TRADES:
            reason = (
                f"Hard cutoff: {tpd:.0f} trades/day > "
                f"{config.BOT_HARD_CUTOFF_TRADES} (BOT_HARD_CUTOFF_TRADES) -- "
                f"physically implausible for a human"
            )
            logger.info(
                f"Bot detection for {addr_short}: prob=1.00 (True, hard-cutoff) "
                f"-- {tpd:.0f} trades/day exceeds BOT_HARD_CUTOFF_TRADES="
                f"{config.BOT_HARD_CUTOFF_TRADES}"
            )
            return BotResult(
                bot_probability=1.0,
                is_bot=True,
                confidence=0.95,
                signals={"trade_frequency": 1.0},
                reason=reason,
            )

        # Compute each feature's bot probability (0.0-1.0)
        signals["trade_frequency"] = self._score_trade_frequency(fills)
        signals["timing_regularity"] = self._score_timing_regularity(fills)
        signals["size_uniformity"] = self._score_size_uniformity(fills)
        signals["pnl_pattern"] = self._score_pnl_pattern(fills, trade_analysis)
        signals["liquidation_rate"] = self._score_liquidation_rate(fills)
        signals["session_pattern"] = self._score_session_pattern(fills)

        # Weighted combination
        bot_probability = sum(
            signals[key] * self.weights[key]
            for key in self.weights.keys()
        )
        bot_probability = min(max(bot_probability, 0.0), 1.0)  # Clamp to [0, 1]

        # Classification
        is_bot = bot_probability >= self.threshold

        # Confidence: how many signals agree?
        # High agreement = high confidence
        strong_signals = sum(1 for v in signals.values() if v > 0.6)
        weak_signals = sum(1 for v in signals.values() if 0.3 < v <= 0.6)
        confidence = min(0.3 * strong_signals + 0.15 * weak_signals, 1.0)

        # Reason string
        reason = self._build_reason(bot_probability, is_bot, signals, fills)

        logger.info(
            f"Bot detection for {addr_short}: "
            f"prob={bot_probability:.2f} ({is_bot}), confidence={confidence:.2f} -- {reason}"
        )

        return BotResult(
            bot_probability=bot_probability,
            is_bot=is_bot,
            confidence=confidence,
            signals=signals,
            reason=reason
        )

    def _score_trade_frequency(self, fills: List[Dict]) -> float:
        """
        Score trade frequency using a continuous sigmoid curve.

        Returns:
            Bot probability (0.0-1.0) based on trades per day.
            - 10 trades/day → 0.1 (very human)
            - 30 trades/day → 0.3 (active human)
            - 50 trades/day → 0.6 (suspicious)
            - 100+ trades/day → 0.95 (almost bot)
        """
        trades_per_day = self._compute_trades_per_day(fills)

        # Hard ceiling: >100 trades/day is essentially 1.0 bot
        if trades_per_day > config.BOT_HARD_CUTOFF_TRADES:
            return 1.0

        # Sigmoid curve: gentle rise from 0 to 1 over 10-100 trades/day
        # Using logistic function: 1 / (1 + exp(-(x - midpoint) / steepness))
        # Midpoint at 50, steepness of 15 gives nice S-curve
        midpoint = 50.0
        steepness = 15.0
        import math
        sigmoid = 1.0 / (1.0 + math.exp(-(trades_per_day - midpoint) / steepness))
        return sigmoid

    def _score_timing_regularity(self, fills: List[Dict]) -> float:
        """
        Score timing regularity: measure variance in inter-trade intervals.

        Humans have irregular spacing (high variance); bots trade regularly.

        Returns:
            Bot probability (0.0-1.0).
            - High variance (irregular) → low score
            - Low variance (regular) → high score
        """
        if len(fills) < 10:
            return 0.0  # Need enough data

        times = sorted([f.get("time", 0) for f in fills if f.get("time")])
        if len(times) < 2:
            return 0.0

        # Compute inter-trade intervals (milliseconds)
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        if not intervals:
            return 0.0

        # Coefficient of variation (CV) of intervals
        # CV = std / mean
        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.0

        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        cv = std_interval / mean_interval

        # Map CV to bot probability
        # CV < 0.2: very regular (bot-like) → ~0.8
        # CV > 1.0: very irregular (human-like) → ~0.1
        # Use inverse sigmoid
        import math
        # If cv < 0.2 → high score, cv > 1.0 → low score
        # Use: 1 - sigmoid(cv, midpoint=0.5, steepness=0.3)
        midpoint = 0.5
        steepness = 0.3
        regularity_score = 1.0 / (1.0 + math.exp(-(midpoint - cv) / steepness))
        return regularity_score

    def _score_size_uniformity(self, fills: List[Dict]) -> float:
        """
        Score size uniformity: measure variance in trade sizes.

        Humans vary sizes; bots often use identical sizes.

        Returns:
            Bot probability (0.0-1.0).
            - High variance (varied sizes) → low score
            - Low variance (uniform sizes) → high score
        """
        if len(fills) < 10:
            return 0.0

        sizes = [f.get("size", 0) * f.get("price", 1) for f in fills if f.get("size", 0) > 0]
        if not sizes or len(sizes) < 5:
            return 0.0

        mean_size = statistics.mean(sizes)
        if mean_size == 0:
            return 0.0

        std_size = statistics.stdev(sizes) if len(sizes) > 1 else 0
        cv = std_size / mean_size

        # Map CV to bot probability (same as timing_regularity).
        # ★ AUDIT FIX: midpoint widened from 0.05 -> 0.15 because the old
        # value only caught CV<0.07 (very tight uniformity), missing
        # bots with CV=0.1-0.2 which are still suspiciously uniform.
        # Steepness widened from 0.02 -> 0.05 so the transition is less
        # binary and partial uniformity still contributes.
        import math
        midpoint = float(os.environ.get("BOT_SIZE_UNIFORMITY_MIDPOINT", "0.15"))
        steepness = float(os.environ.get("BOT_SIZE_UNIFORMITY_STEEPNESS", "0.05"))
        uniformity_score = 1.0 / (1.0 + math.exp(-(midpoint - cv) / steepness))
        return uniformity_score

    def _score_pnl_pattern(self, fills: List[Dict], trade_analysis: Dict) -> float:
        """
        Score PnL pattern: check for market maker / arbitrage patterns.

        Bot indicators:
        - Very small, consistent PnL = market maker
        - Zero-sum round trips = arbitrage
        - High frequency + near-zero median PnL = funding farm

        Returns:
            Bot probability (0.0-1.0).
        """
        if len(fills) < 20:
            return 0.0

        trades_per_day = self._compute_trades_per_day(fills)
        closed_pnls = [f.get("closed_pnl", 0) for f in fills if f.get("closed_pnl", 0) != 0]
        if not closed_pnls:
            return 0.0

        median_pnl = sorted(closed_pnls)[len(closed_pnls) // 2]
        avg_size = trade_analysis.get("avg_trade_size", 0)

        # Pattern 1: Near-zero PnL at high frequency = MM/funding bot
        # If trades_per_day > 20 AND |median PnL| < $1, suspicious
        if trades_per_day > 20 and abs(median_pnl) < 1.0:
            return 0.8  # Strong indicator

        # Pattern 2: Very small PnL per trade with large sizes = arb bot
        pnl_per_trade = abs(trade_analysis.get("total_closed_pnl", 0)) / len(fills) if fills else 0
        if pnl_per_trade < 0.50 and avg_size > 1000:
            return 0.75  # Strong indicator

        # Pattern 3: Highly positive win rate with tiny average wins = scalp bot
        win_rate = trade_analysis.get("win_rate", 0)
        avg_win = trade_analysis.get("avg_win", 0)
        if win_rate > 0.55 and avg_win < 5 and trades_per_day > 30:
            return 0.6  # Medium indicator

        # No clear pattern
        return 0.0

    def _score_liquidation_rate(self, fills: List[Dict]) -> float:
        """
        Score liquidation rate: high liquidations suggest reckless strategy.

        Not profitable, so not worth copying. But also not necessarily a bot.

        Returns:
            Bot probability (0.0-1.0).
            - > 20% liquidation rate → 0.7 (reckless)
            - > 10% liquidation rate → 0.4 (concerning)
            - < 5% liquidation rate → 0.1 (acceptable)
        """
        if len(fills) < 10:
            return 0.0

        liquidations = sum(1 for f in fills if f.get("is_liquidation", False))
        liq_rate = liquidations / len(fills)

        import math
        # Map to bot probability with sigmoid
        # liq_rate > 0.2 → high score, liq_rate < 0.02 → low score
        midpoint = 0.10
        steepness = 0.05
        liq_score = 1.0 / (1.0 + math.exp(-(liq_rate - midpoint) / steepness))
        return liq_score

    def _score_session_pattern(self, fills: List[Dict]) -> float:
        """
        Score session pattern: humans sleep, bots don't.

        Check if trades happen 24/7 with no gaps, vs. clusters with rest periods.

        Returns:
            Bot probability (0.0-1.0).
            - Trades spread across 24h with no gaps → 0.8 (bot)
            - Trades in 6-10h windows with rest periods → 0.1 (human)
        """
        if len(fills) < 20:
            return 0.0

        times = sorted([f.get("time", 0) for f in fills if f.get("time")])
        if len(times) < 2:
            return 0.0

        # Convert to hours (assuming milliseconds)
        times_hours = [t / (1000 * 3600) for t in times]

        # Find gaps in trading
        intervals = [times_hours[i + 1] - times_hours[i] for i in range(len(times_hours) - 1)]

        # Humans typically have 6-16 hour gaps (sleep)
        # Bots trade continuously
        large_gaps = sum(1 for interval in intervals if interval > 6)  # 6+ hour gaps

        if large_gaps == 0:
            # No 6+ hour gaps → bot-like
            return 0.8
        elif large_gaps >= 1:
            # At least one long gap → human-like
            return 0.1
        else:
            return 0.3

    def evaluate_accuracy(
        self,
        known_bots: List[str],
        known_humans: List[str],
        get_fills_fn,
        get_positions_fn,
        get_trade_analysis_fn
    ) -> Dict:
        """
        Evaluate detector accuracy against a labelled set.

        ★ M18 FIX: previously named ``calibrate`` but it ONLY measured
        accuracy — it never adjusted weights, so the name was misleading.
        Renamed to ``evaluate_accuracy`` to reflect what it actually does.
        ``calibrate`` is kept as a deprecated alias below so existing
        callers don't break.  Implement true weight calibration in a
        future PR using these metrics as the optimization target.

        Args:
            known_bots: List of addresses known to be bots
            known_humans: List of addresses known to be humans
            get_fills_fn: Function(address) -> List[Dict] of fills
            get_positions_fn: Function(address) -> List[Dict] of positions
            get_trade_analysis_fn: Function(address) -> Dict of trade analysis

        Returns:
            Accuracy metrics dict (bot_accuracy, human_accuracy, overall_accuracy).
        """
        logger.info(f"Calibrating on {len(known_bots)} bots, {len(known_humans)} humans...")

        correct_bots = 0
        correct_humans = 0

        for addr in known_bots:
            fills = get_fills_fn(addr)
            positions = get_positions_fn(addr)
            trade_analysis = get_trade_analysis_fn(addr)
            result = self.detect(fills, positions, trade_analysis, addr)
            if result.is_bot:
                correct_bots += 1

        for addr in known_humans:
            fills = get_fills_fn(addr)
            positions = get_positions_fn(addr)
            trade_analysis = get_trade_analysis_fn(addr)
            result = self.detect(fills, positions, trade_analysis, addr)
            if not result.is_bot:
                correct_humans += 1

        bot_accuracy = correct_bots / len(known_bots) if known_bots else 0.0
        human_accuracy = correct_humans / len(known_humans) if known_humans else 0.0
        overall_accuracy = (correct_bots + correct_humans) / (len(known_bots) + len(known_humans))

        metrics = {
            "bot_accuracy": bot_accuracy,
            "human_accuracy": human_accuracy,
            "overall_accuracy": overall_accuracy,
            "correct_bots": correct_bots,
            "correct_humans": correct_humans,
        }

        logger.info(
            f"Accuracy evaluation: bot_acc={bot_accuracy:.2%}, "
            f"human_acc={human_accuracy:.2%}, overall={overall_accuracy:.2%}"
        )

        return metrics

    def calibrate(self, *args, **kwargs) -> Dict:
        """Deprecated: use ``evaluate_accuracy`` instead. ★ M18.

        The original name implied weight optimization; this method only
        measures accuracy.  Kept as a thin alias to avoid breaking older
        callers.  New code should call ``evaluate_accuracy`` directly.
        """
        logger.warning(
            "AdaptiveBotDetector.calibrate() is deprecated -- use evaluate_accuracy()"
        )
        return self.evaluate_accuracy(*args, **kwargs)

    # ─── Helper Methods ────────────────────────────────────────────

    def _compute_trades_per_day(self, fills: List[Dict]) -> float:
        """Compute actual trades-per-day from fill timestamps."""
        if len(fills) < 2:
            return 0.0

        times = sorted([f.get("time", 0) for f in fills if f.get("time")])
        if len(times) < 2:
            return 0.0

        span_ms = times[-1] - times[0]
        if span_ms == 0:
            return 0.0

        span_days = span_ms / (1000 * 3600 * 24)
        trades_per_day = len(fills) / max(span_days, 0.01)  # Avoid division by 0

        return trades_per_day

    def _build_reason(
        self,
        bot_probability: float,
        is_bot: bool,
        signals: Dict[str, float],
        fills: List[Dict]
    ) -> str:
        """Build a human-readable explanation of the classification."""
        parts = []

        # Probability category
        if bot_probability < 0.3:
            category = "Likely human"
        elif bot_probability < 0.6:
            category = "Uncertain"
        elif bot_probability < 0.85:
            category = "Likely bot"
        else:
            category = "Almost certainly bot"

        parts.append(f"{category} (prob={bot_probability:.2f})")

        # Top signals
        top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
        signal_strs = [f"{k}={v:.2f}" for k, v in top_signals if v > 0.2]
        if signal_strs:
            parts.append(f"Top signals: {', '.join(signal_strs)}")

        # Trade info
        trades_per_day = self._compute_trades_per_day(fills)
        parts.append(f"{trades_per_day:.0f} trades/day")

        return " | ".join(parts)
