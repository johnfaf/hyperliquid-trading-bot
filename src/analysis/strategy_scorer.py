"""
Strategy Scoring & Self-Improvement System
Tracks strategy performance over time and automatically adjusts confidence
scores, applying time-decay to outdated strategies and boosting consistently
profitable ones. This is the "learning" core of the bot.
"""
import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import database as db

logger = logging.getLogger(__name__)


# ★ H16 FIX: exact two-sided binomial p-value for significance gate.
# Mirrors src.signals.feature_store_alpha._exact_binomial_pvalue but kept
# local so strategy_scorer has no cross-layer dependency.
def _binomial_pvalue_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value.

    H0: the true success probability is p (default 0.5 = no edge).
    Returns the probability of observing a result at least as extreme
    as k successes in n trials under H0.
    """
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        try:
            return float(binomtest(k, n, p, alternative="two-sided").pvalue)
        except TypeError:
            # scipy < 1.7 uses `binom_test`
            from scipy.stats import binom_test
            return float(binom_test(k, n, p, alternative="two-sided"))
    except Exception:
        # Fall back to a normal approximation if scipy isn't available
        # (valid once np >= 10 and n(1-p) >= 10 — otherwise conservative)
        import math
        mean = n * p
        sd = math.sqrt(n * p * (1 - p))
        if sd <= 0:
            return 1.0
        z = abs(k - mean) / sd
        # two-sided p from z via erf
        return math.erfc(z / math.sqrt(2))


class StrategyScorer:
    """
    Scores and ranks strategies based on multiple performance dimensions.
    Implements time-decay so older results matter less, and strategies that
    keep performing well rise to the top over time.
    """

    def __init__(self):
        self.weights = config.SCORING_WEIGHTS
        self.decay_rate = config.SCORE_DECAY_RATE

    def _persist_scoring_results(
        self,
        results: List[Dict],
        desired_active_ids: set[int],
    ) -> None:
        """Persist one scoring cycle in a single DB transaction.

        The old implementation opened a new SQLite write transaction for every
        score insert, every current-score update, and every active-flag update.
        During startup that multiplied lock contention with background writers.
        Persisting the whole cycle in one short transaction keeps the scoring
        window tight and materially reduces `database is locked` failures.
        """
        if not results:
            return

        scored_at = datetime.now(timezone.utc).isoformat()
        note = f"Auto-scored at {scored_at}"
        with db.get_connection() as conn:
            for result in results:
                breakdown = result["breakdown"]
                strategy_id = int(result["strategy_id"])
                should_be_active = strategy_id in desired_active_ids
                result["active"] = bool(should_be_active)
                conn.execute(
                    """
                    INSERT INTO strategy_scores
                    (strategy_id, timestamp, score, pnl_score, win_rate_score,
                     sharpe_score, consistency_score, risk_adj_score, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        strategy_id,
                        scored_at,
                        float(result["score"]),
                        float(breakdown["pnl_score"]),
                        float(breakdown["win_rate_score"]),
                        float(breakdown["sharpe_score"]),
                        float(breakdown["consistency_score"]),
                        float(breakdown["risk_adj_score"]),
                        note,
                    ),
                )
                conn.execute(
                    """
                    UPDATE strategies
                    SET current_score = ?, last_scored = ?, active = ?
                    WHERE id = ?
                    """,
                    (
                        float(result["score"]),
                        scored_at,
                        bool(should_be_active),
                        strategy_id,
                    ),
                )

    def score_strategy(self, strategy: Dict) -> Dict:
        """
        Compute a composite score for a strategy across multiple dimensions.
        Returns a score breakdown dict.

        ★ H16 FIX: wired in the exact-binomial p-value test from
        feature_store_alpha._exact_binomial_pvalue as the author's own
        AUDIT M4 comment suggested. A strategy's win rate is now tested
        against a 50% null hypothesis. Strategies whose win-rate advantage
        is not statistically distinguishable from noise get a significance
        penalty instead of being ranked purely on raw numerics.

        Gate logic:
          * trade_count < 10:  no p-value test (too small), soft sample penalty only
          * trade_count >= 10: p-value against 0.5 null; penalty scales with p-value
              - p < 0.05:  no significance penalty
              - p < 0.10:  0.85x multiplier
              - p < 0.20:  0.65x multiplier
              - p >= 0.20: 0.40x multiplier (likely noise)
        """
        # Individual dimension scores (each 0-1)
        pnl_score = self._score_pnl(strategy)
        win_rate_score = self._score_win_rate(strategy)
        sharpe_score = self._score_sharpe(strategy)
        consistency_score = self._score_consistency(strategy)
        risk_adj_score = self._score_risk_adjusted(strategy)

        # Weighted composite
        composite = (
            self.weights["pnl"] * pnl_score +
            self.weights["win_rate"] * win_rate_score +
            self.weights["sharpe_ratio"] * sharpe_score +
            self.weights["consistency"] * consistency_score +
            self.weights["risk_adjusted_return"] * risk_adj_score
        )

        # Sample-size penalty: strategies with fewer than 10 trades get
        # a multiplicative discount that prevents high scores from thin data.
        trade_count = strategy.get("trade_count", 0)
        if trade_count < 10:
            # Softened from exponent 1.5 → 1.0: old penalty was too harsh
            # during bootstrap (5 trades = 0.35x killed everything).
            # New: 5 trades = 0.50x, 3 trades = 0.30x, 8 trades = 0.80x
            sample_penalty = (trade_count / 10) ** 1.0
            composite *= sample_penalty
            logger.debug(f"Sample-size penalty: {trade_count} trades -> {sample_penalty:.2f}x")

        # ★ H16 FIX: statistical significance gate
        # Apply once we have enough data (>= 10 trades) to run a real test.
        pvalue = None
        significance_penalty = 1.0
        if trade_count >= 10:
            win_rate = float(strategy.get("win_rate", 0) or 0)
            # Accept win_rate as either fraction (0.58) or percent (58.0)
            if win_rate > 1.5:
                win_rate = win_rate / 100.0
            wins = int(round(win_rate * trade_count))
            try:
                pvalue = _binomial_pvalue_two_sided(wins, trade_count, 0.5)
            except Exception:
                pvalue = None

            if pvalue is not None:
                # Only apply significance penalty to strategies with edge
                # ABOVE 50% — losing strategies (win_rate < 0.5) already
                # lose via win_rate_score and we don't want to double-penalize.
                if win_rate > 0.5:
                    if pvalue < 0.05:
                        significance_penalty = 1.0
                    elif pvalue < 0.10:
                        significance_penalty = 0.85
                    elif pvalue < 0.20:
                        significance_penalty = 0.65
                    else:
                        significance_penalty = 0.40
                    if significance_penalty < 1.0:
                        logger.debug(
                            "Significance penalty: %d/%d wins (wr=%.2f, p=%.3f) -> %.2fx",
                            wins, trade_count, win_rate, pvalue, significance_penalty,
                        )
                    composite *= significance_penalty

        # Apply time decay based on when strategy was last scored
        last_scored = strategy.get("last_scored")
        if last_scored:
            try:
                last_dt = datetime.fromisoformat(last_scored)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                days_since = (datetime.now(timezone.utc) - last_dt).days
                # Accelerated decay: gentle for first 7 days, aggressive after
                if days_since <= 7:
                    decay_factor = self.decay_rate ** days_since
                else:
                    # After 7 days: base decay + accelerated component
                    decay_factor = (self.decay_rate ** 7) * (0.85 ** (days_since - 7))
                composite *= decay_factor
            except (ValueError, TypeError):
                pass

        score_breakdown = {
            "composite": round(composite, 4),
            "pnl_score": round(pnl_score, 4),
            "win_rate_score": round(win_rate_score, 4),
            "sharpe_score": round(sharpe_score, 4),
            "consistency_score": round(consistency_score, 4),
            "risk_adj_score": round(risk_adj_score, 4),
            # H16: expose significance data in breakdown for dashboards/debugging
            "win_rate_pvalue": round(pvalue, 4) if pvalue is not None else None,
            "significance_penalty": round(significance_penalty, 3),
        }

        return score_breakdown

    def _score_pnl(self, strategy: Dict) -> float:
        """Score based on total PnL. Uses sigmoid to normalize."""
        pnl = strategy.get("total_pnl", 0)
        # Sigmoid normalization: maps any PnL to 0-1
        # $10k PnL -> ~0.73, $50k -> ~0.95, -$10k -> ~0.27
        # Clamp exponent to avoid RuntimeWarning: overflow encountered in exp
        x = np.clip(-pnl / 10000, -500, 500)
        return float(1 / (1 + np.exp(x)))

    def _score_win_rate(self, strategy: Dict) -> float:
        """Score based on win rate. Minimum trades required for reliability.

        AUDIT M4 — the ``trade_count < MIN_TRADES_FOR_STRATEGY`` guard
        below is a *quadratic sample penalty*, not a significance
        test.  It shrinks the score for thin data but does not check
        whether the observed win-rate is distinguishable from a
        neutral-edge null.  A win-rate point estimate without a
        confidence interval can rank a lucky 4-out-of-5 strategy (WR
        = 0.80, p ~ 0.19 vs 50%) above a well-sampled 55-out-of-100
        strategy (WR = 0.55, p ~ 0.18 vs 50%) despite the latter
        being meaningfully more reliable.

        Future work: replace the raw ``win_rate`` feed with a Wilson
        lower-confidence-bound (see the binomial helper already
        present at ``src.signals.feature_store_alpha``) so the score
        reflects *conservative* performance estimates rather than
        point estimates.
        """
        win_rate = strategy.get("win_rate", 0)
        trade_count = strategy.get("trade_count", 0)

        if trade_count < config.MIN_TRADES_FOR_STRATEGY:
            # Geometric penalty: 5 trades = 0.25x, 3 trades = 0.09x, 1 trade = 0.01x
            penalty = (trade_count / config.MIN_TRADES_FOR_STRATEGY) ** 2
            return win_rate * penalty

        # MED-FIX MED-1: replace linear *1.3 clip with a sigmoid that rewards
        # all win-rate levels proportionally.  Old formula hit 1.0 at 77% WR
        # making 78%, 90%, and 99% indistinguishable.
        # New: WR=0.50→0.50, WR=0.65→0.77, WR=0.80→0.93, WR=0.95→0.99
        # LOW-FIX LOW-3: guard against None from DB (dict.get returns None when
        # the column exists but holds NULL, not the default value).
        win_rate = float(win_rate or 0)
        if win_rate <= 0:
            return 0.0
        return float(1 / (1 + np.exp(np.clip(-8 * (win_rate - 0.5), -500, 500))))

    def _score_sharpe(self, strategy: Dict) -> float:
        """Score based on Sharpe ratio estimate."""
        sharpe = strategy.get("sharpe_ratio", 0)
        # Normalize: Sharpe of 1 -> 0.5, Sharpe of 2 -> 0.73, Sharpe of 3 -> 0.88
        # HIGH-FIX HIGH-8: zero score for non-positive Sharpe.  The old formula
        # returned 0.1–0.19 for Sharpe in (-1, 0], allowing bad strategies to
        # persist active too long.  A non-positive Sharpe means risk-adjusted
        # returns are at or below the risk-free rate — score 0.0 so that the
        # composite score drives the strategy toward deactivation.
        if sharpe <= 0:
            return 0.0
        return float(1 / (1 + np.exp(np.clip(-sharpe + 1, -500, 500))))

    def _score_consistency(self, strategy: Dict) -> float:
        """
        Score based on consistency of returns over time.
        Uses historical score data to measure variance.
        """
        strategy_id = strategy.get("id")
        if not strategy_id:
            return 0.3  # Default for new strategies

        history = db.get_strategy_score_history(strategy_id, limit=30)
        if len(history) < 3:
            return 0.3  # Not enough data

        scores = [h["score"] for h in history]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if mean_score == 0:
            return 0.3

        # Coefficient of variation (lower is more consistent)
        cv = std_score / abs(mean_score) if mean_score != 0 else float('inf')

        # Invert: low CV -> high consistency score
        consistency = max(0, 1 - cv)
        return min(1.0, consistency)

    def _score_risk_adjusted(self, strategy: Dict) -> float:
        """Score based on risk-adjusted returns (PnL relative to drawdown potential)."""
        pnl = strategy.get("total_pnl", 0)
        trade_count = strategy.get("trade_count", 0)
        win_rate = strategy.get("win_rate", 0)

        if trade_count == 0:
            return 0.3

        # Estimate risk based on losing trade frequency and potential loss
        loss_rate = 1 - win_rate
        # Higher PnL with lower loss rate = better risk-adjusted return
        if pnl > 0:
            risk_factor = max(0.1, loss_rate)
            risk_adj = (pnl / 10000) / risk_factor
            return min(1.0, float(1 / (1 + np.exp(np.clip(-risk_adj + 1, -500, 500)))))
        else:
            return max(0, 0.3 * loss_rate)

    def score_all_strategies(self) -> List[Dict]:
        """
        Score all active strategies and update the database.
        This is the main "learning" loop - called periodically.
        """
        strategies = db.get_active_strategies()
        results = []

        logger.info(f"Scoring {len(strategies)} active strategies...")

        for strategy in strategies:
            try:
                score_breakdown = self.score_strategy(strategy)
                composite = score_breakdown["composite"]

                results.append({
                    "strategy_id": strategy["id"],
                    "name": strategy["name"],
                    "type": strategy["strategy_type"],
                    "score": float(composite),
                    "breakdown": {k: float(v) for k, v in score_breakdown.items()},
                    "active": bool(composite >= config.MIN_STRATEGY_SCORE),
                })

            except Exception as e:
                logger.error(f"Error scoring strategy {strategy.get('id')}: {e}")

        # Sort by score
        results.sort(key=lambda r: r["score"], reverse=True)

        # Log summary
        if results:
            top = results[0]
            logger.info(f"Top strategy: {top['name']} (type={top['type']}, score={top['score']:.4f})")

        # Keep all above-threshold strategies, but ensure at least top-N stay
        # active during weak-score phases (prevents strategy "death spiral").
        keep_top_n = max(1, int(getattr(config, "MIN_ACTIVE_STRATEGIES", 5)))
        threshold = float(config.MIN_STRATEGY_SCORE)
        desired_active_ids = {
            r["strategy_id"] for r in results if r["score"] >= threshold
        }
        for r in results[:keep_top_n]:
            desired_active_ids.add(r["strategy_id"])

        # Hard cap for operational safety.
        if len(desired_active_ids) > config.MAX_ACTIVE_STRATEGIES:
            capped_ids = {
                r["strategy_id"] for r in results[:config.MAX_ACTIVE_STRATEGIES]
            }
            desired_active_ids = capped_ids

        for r in results:
            r["active"] = r["strategy_id"] in desired_active_ids

        try:
            self._persist_scoring_results(results, desired_active_ids)
        except Exception as exc:
            logger.error("Failed to persist scoring cycle: %s", exc)

        logger.info(
            "Strategy activation policy: threshold=%.3f, keep_top_n=%d, active=%d/%d",
            threshold,
            keep_top_n,
            sum(1 for r in results if r["active"]),
            len(results),
        )

        # Log the scoring cycle
        active_count = sum(1 for r in results if r["active"])
        try:
            db.log_research_cycle(
                cycle_type="scoring",
                summary=f"Scored {len(results)} strategies, {active_count} active",
                details={"top_strategies": results[:5]},
                strategies_updated=len(results),
            )
        except Exception as exc:
            logger.error("Failed to log scoring cycle: %s", exc)

        return results

    def get_top_strategies(self, n: int = None) -> List[Dict]:
        """Get the top N strategies by current score.
        Defaults to config.MAX_STRATEGIES_PER_CYCLE."""
        if n is None:
            n = config.MAX_STRATEGIES_PER_CYCLE
        strategies = db.get_active_strategies()
        return strategies[:n]

    def get_strategy_trend(self, strategy_id: int) -> Dict:
        """
        Analyze the scoring trend for a strategy.
        Returns trend direction and momentum.
        """
        history = db.get_strategy_score_history(strategy_id, limit=14)
        if len(history) < 2:
            return {"trend": "insufficient_data", "momentum": 0}

        scores = [h["score"] for h in reversed(history)]  # chronological

        # Simple linear regression for trend
        x = np.arange(len(scores))
        if len(scores) > 1:
            slope = np.polyfit(x, scores, 1)[0]
        else:
            slope = 0

        # Recent momentum (last 3 vs previous 3)
        if len(scores) >= 6:
            recent_avg = np.mean(scores[-3:])
            prev_avg = np.mean(scores[-6:-3])
            momentum = (recent_avg - prev_avg) / prev_avg if prev_avg != 0 else 0
        else:
            momentum = slope

        if slope > 0.01:
            trend = "improving"
        elif slope < -0.01:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(slope, 6),
            "momentum": round(momentum, 4),
            "current_score": scores[-1] if scores else 0,
            "avg_score": round(np.mean(scores), 4),
            "data_points": len(scores),
        }

    def generate_improvement_report(self) -> Dict:
        """
        Generate a report on how the bot's strategy selection is improving over time.
        This is the key self-improvement metric.
        """
        strategies = db.get_active_strategies()

        if not strategies:
            return {"status": "no_strategies", "message": "No strategies tracked yet"}

        # Analyze trends for all strategies
        trends = {}
        for s in strategies:
            trends[s["id"]] = self.get_strategy_trend(s["id"])

        improving = sum(1 for t in trends.values() if t["trend"] == "improving")
        declining = sum(1 for t in trends.values() if t["trend"] == "declining")
        stable = sum(1 for t in trends.values() if t["trend"] == "stable")

        # Overall portfolio score trend
        all_scores = [s["current_score"] for s in strategies]
        avg_score = np.mean(all_scores) if all_scores else 0

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_strategies": len(strategies),
            "improving": improving,
            "declining": declining,
            "stable": stable,
            "avg_score": round(avg_score, 4),
            "top_strategy": strategies[0] if strategies else None,
            "health": "good" if improving > declining else "needs_attention" if declining > improving else "neutral",
            "trends": {s["id"]: trends.get(s["id"], {}) for s in strategies[:10]},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    scorer = StrategyScorer()
    results = scorer.score_all_strategies()
    print(f"\nScored {len(results)} strategies")
    for r in results[:5]:
        print(f"  {r['name']}: score={r['score']:.4f} (active={r['active']})")

    report = scorer.generate_improvement_report()
    print(f"\nImprovement report: {report}")
