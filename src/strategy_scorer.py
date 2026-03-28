"""
Strategy Scoring & Self-Improvement System
Tracks strategy performance over time and automatically adjusts confidence
scores, applying time-decay to outdated strategies and boosting consistently
profitable ones. This is the "learning" core of the bot.
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import database as db

logger = logging.getLogger(__name__)


class StrategyScorer:
    """
    Scores and ranks strategies based on multiple performance dimensions.
    Implements time-decay so older results matter less, and strategies that
    keep performing well rise to the top over time.
    """

    def __init__(self):
        self.weights = config.SCORING_WEIGHTS
        self.decay_rate = config.SCORE_DECAY_RATE

    def score_strategy(self, strategy: Dict) -> Dict:
        """
        Compute a composite score for a strategy across multiple dimensions.
        Returns a score breakdown dict.
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

        # Apply time decay based on when strategy was last scored
        last_scored = strategy.get("last_scored")
        if last_scored:
            try:
                last_dt = datetime.fromisoformat(last_scored)
                days_since = (datetime.utcnow() - last_dt).days
                decay_factor = self.decay_rate ** days_since
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
        """Score based on win rate. Minimum trades required for reliability."""
        win_rate = strategy.get("win_rate", 0)
        trade_count = strategy.get("trade_count", 0)

        if trade_count < config.MIN_TRADES_FOR_STRATEGY:
            # Penalize strategies with too few trades (low confidence)
            return win_rate * (trade_count / config.MIN_TRADES_FOR_STRATEGY)

        # Win rate above 50% is good, above 65% is great
        return min(1.0, win_rate * 1.3)

    def _score_sharpe(self, strategy: Dict) -> float:
        """Score based on Sharpe ratio estimate."""
        sharpe = strategy.get("sharpe_ratio", 0)
        # Normalize: Sharpe of 1 -> 0.5, Sharpe of 2 -> 0.73, Sharpe of 3 -> 0.88
        if sharpe <= 0:
            return max(0, 0.2 + sharpe * 0.1)  # Negative sharpe still gets some score
        return float(1 / (1 + np.exp(np.clip(-sharpe + 1, -500, 500))))

    def _score_consistency(self, strategy: Dict) -> float:
        """
        Score based on consistency of returns over time.
        Uses historical score data to measure variance.
        """
        strategy_id = strategy.get("id")
        if not strategy_id:
            return 0.5  # Default for new strategies

        history = db.get_strategy_score_history(strategy_id, limit=30)
        if len(history) < 3:
            return 0.5  # Not enough data

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
            return max(0, 0.3 * (1 - loss_rate))

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

                # Save the score to history
                db.save_strategy_score(
                    strategy_id=strategy["id"],
                    score=composite,
                    pnl_score=score_breakdown["pnl_score"],
                    win_rate_score=score_breakdown["win_rate_score"],
                    sharpe_score=score_breakdown["sharpe_score"],
                    consistency_score=score_breakdown["consistency_score"],
                    risk_adj_score=score_breakdown["risk_adj_score"],
                    notes=f"Auto-scored at {datetime.utcnow().isoformat()}"
                )

                # Update the strategy's current score
                db.update_strategy_score(strategy["id"], composite)

                # Deactivate strategies below threshold
                if composite < config.MIN_STRATEGY_SCORE:
                    logger.info(f"Deactivating low-scoring strategy {strategy['id']} "
                              f"({strategy['name']}): score={composite:.4f}")
                    with db.get_connection() as conn:
                        conn.execute(
                            "UPDATE strategies SET active = 0 WHERE id = ?",
                            (strategy["id"],)
                        )

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
            active_count = sum(1 for r in results if r["active"])
            logger.info(f"Active strategies: {active_count}/{len(results)}")

        # Prune: if active strategies exceed MAX_ACTIVE_STRATEGIES,
        # deactivate the lowest-scoring ones to prevent unbounded growth
        active_results = [r for r in results if r["active"]]
        if len(active_results) > config.MAX_ACTIVE_STRATEGIES:
            # Keep top MAX_ACTIVE_STRATEGIES, deactivate the rest
            to_deactivate = active_results[config.MAX_ACTIVE_STRATEGIES:]
            for r in to_deactivate:
                try:
                    with db.get_connection() as conn:
                        conn.execute(
                            "UPDATE strategies SET active = 0 WHERE id = ?",
                            (r["strategy_id"],)
                        )
                    r["active"] = False
                except Exception:
                    pass
            logger.info(f"Pruned {len(to_deactivate)} excess strategies "
                       f"(keeping top {config.MAX_ACTIVE_STRATEGIES})")

        # Log the scoring cycle
        active_count = sum(1 for r in results if r["active"])
        db.log_research_cycle(
            cycle_type="scoring",
            summary=f"Scored {len(results)} strategies, {active_count} active",
            details={"top_strategies": results[:5]},
            strategies_updated=len(results),
        )

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
            "timestamp": datetime.utcnow().isoformat(),
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
