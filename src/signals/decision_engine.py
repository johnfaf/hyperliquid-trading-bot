"""
Final Decision Engine
=====================
The missing layer identified from production log analysis.

Problem: SignalProcessor outputs 5 clean strategies, but they go into the
paper trader loop with no priority ordering, no composite ranking, and no
clear "FINAL_DECISION" log. This makes it impossible to audit what the
system actually decided and why.

Solution: Rank the 5 survivors on a composite score, execute in priority
order up to available position slots, and produce clear decision logs.

Pipeline position:
  SignalProcessor (5 strategies) → DecisionEngine (ranked + logged) → Paper Trader (execute)

NOT forcing to 1 trade (ChatGPT's suggestion) because:
  - Portfolio diversification across uncorrelated positions is better risk mgmt
  - We have 5 position slots, forcing to 1 wastes capacity
  - The 5 signals are already on different coins after dedup/conflict resolution

Instead: ranked execution with clear priority + decision logging.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Takes cleaned strategies from SignalProcessor and produces a ranked
    decision list with clear logging of what the system decided and why.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Weights for composite ranking score
        self.w_score = cfg.get("w_score", 0.35)          # Strategy score from scorer
        self.w_regime = cfg.get("w_regime", 0.25)         # Regime alignment bonus
        self.w_diversity = cfg.get("w_diversity", 0.20)   # Diversification bonus
        self.w_freshness = cfg.get("w_freshness", 0.10)   # Prefer strategies with recent activity
        self.w_consensus = cfg.get("w_consensus", 0.10)   # Dedup consensus boost

        # Minimum composite score to even consider executing
        # Lowered from 0.30 → 0.20: with thin trade history the composite
        # scores are suppressed by sample-size penalties. 0.20 lets
        # promising-but-young strategies through for paper validation.
        self.min_decision_score = cfg.get("min_decision_score", 0.20)

        # Max trades to execute per cycle (independent of position slots)
        self.max_trades_per_cycle = cfg.get("max_trades_per_cycle", 3)

        # Track decisions for audit
        self._decision_history: deque = deque(maxlen=100)
        self._cycle_count = 0

        # Stats
        self.stats = {
            "total_decisions": 0,
            "total_executions": 0,
            "total_no_trade": 0,
            "total_candidates": 0,
        }

    def decide(self, strategies: List[Dict],
               regime_data: Optional[Dict] = None,
               open_positions: Optional[List[Dict]] = None,
               kelly_stats: Optional[Dict] = None) -> List[Dict]:
        """
        Rank strategies and produce a prioritized execution list.

        Args:
            strategies: Cleaned strategies from SignalProcessor (max ~5)
            regime_data: Current market regime
            open_positions: Currently open paper trades
            kelly_stats: Kelly sizing stats per strategy type

        Returns:
            Ranked list of strategies with composite scores and decision metadata.
            Only includes strategies that should be executed (above threshold,
            within cycle trade limit).
        """
        self._cycle_count += 1
        open_positions = open_positions or []
        open_coins = {t["coin"] for t in open_positions}
        available_slots = max(0, 8 - len(open_positions))

        self.stats["total_candidates"] += len(strategies)

        if not strategies:
            self._log_decision([], regime_data, available_slots)
            self.stats["total_no_trade"] += 1
            return []

        # Extract and validate coin field — fallback to positions if missing
        TOP_COINS = ["BTC", "ETH", "SOL", "DOGE", "ARB", "OP", "AVAX", "MATIC",
                     "LINK", "WLD", "SUI", "TIA", "SEI", "INJ", "NEAR"]
        valid_strategies = []
        for s in strategies:
            params = s.get("parameters", {})
            if isinstance(params, str):
                import json
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
                # Persist parsed version so downstream doesn't re-parse
                s["parameters"] = params

            coins = params.get("coins", params.get("coins_traded", []))
            if isinstance(coins, str):
                coins = [coins]

            # Fallback 1: extract from trader's current positions in metrics
            if not coins or (coins and coins[0] == "unknown"):
                metrics = s.get("metrics", {})
                traded_coins = metrics.get("coins", metrics.get("coins_traded", []))
                if traded_coins and isinstance(traded_coins, list):
                    coins = traded_coins
                    params["coins"] = coins  # Persist so _compute_composite_score sees it

            # Fallback 2: infer from strategy type — use top liquid coins
            if not coins or (coins and coins[0] == "unknown"):
                import random
                strategy_type = s.get("strategy_type", s.get("type", ""))
                # Pick a coin based on strategy: momentum → BTC/ETH, mean_reversion → alts
                if "momentum" in strategy_type or "trend" in strategy_type:
                    coins = [random.choice(TOP_COINS[:3])]  # BTC, ETH, SOL
                else:
                    coins = [random.choice(TOP_COINS[:7])]  # Top 7 liquid
                logger.info(f"Inferred coin {coins[0]} for strategy "
                           f"{strategy_type} (no asset in parameters)")

            # Always persist resolved coins back into params
            params["coins"] = coins
            s["parameters"] = params

            valid_strategies.append(s)

        strategies = valid_strategies
        if not strategies:
            self._log_decision([], regime_data, available_slots)
            self.stats["total_no_trade"] += 1
            return []

        # ─── Score each strategy ─────────────────────────
        scored = []
        for s in strategies:
            composite = self._compute_composite_score(
                s, regime_data, open_coins, kelly_stats
            )
            scored.append({
                **s,
                "_composite_score": composite["total"],
                "_score_breakdown": composite,
            })

        # ─── Rank by composite score ─────────────────────
        scored.sort(key=lambda x: x["_composite_score"], reverse=True)

        # ─── Filter by minimum threshold ─────────────────
        qualified = [s for s in scored if s["_composite_score"] >= self.min_decision_score]
        disqualified = [s for s in scored if s["_composite_score"] < self.min_decision_score]

        # ─── Limit by cycle trade cap AND available slots ─
        max_this_cycle = min(self.max_trades_per_cycle, available_slots)
        executions = qualified[:max_this_cycle]
        overflow = qualified[max_this_cycle:]

        # ─── Aggregate directional analysis (ChatGPT's insight) ───
        long_score, short_score = self._aggregate_directional_scores(scored)

        # ─── LOG THE DECISION (this is the key output) ────
        self._log_decision(scored, regime_data, available_slots,
                           executions=executions, disqualified=disqualified,
                           overflow=overflow, long_score=long_score,
                           short_score=short_score)

        # ─── Update stats ────────────────────────────────
        self.stats["total_decisions"] += 1
        if executions:
            self.stats["total_executions"] += len(executions)
        else:
            self.stats["total_no_trade"] += 1

        # ─── Store decision for audit trail ──────────────
        self._decision_history.append({
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidates": len(strategies),
            "qualified": len(qualified),
            "executed": len(executions),
            "long_score": long_score,
            "short_score": short_score,
            "market_bias": "long" if long_score > short_score else "short" if short_score > long_score else "neutral",
            "regime": regime_data.get("overall_regime", "unknown") if regime_data else "unknown",
            "decisions": [
                {"coin": e.get("_decision_coin", "?"),
                 "side": e.get("_decision_side", "?"),
                 "composite": e["_composite_score"]}
                for e in executions
            ],
        })

        # deque(maxlen=100) auto-trims old entries

        return executions

    def _compute_composite_score(self, strategy: Dict,
                                  regime_data: Optional[Dict],
                                  open_coins: set,
                                  kelly_stats: Optional[Dict]) -> Dict:
        """
        Compute a composite ranking score from multiple factors.
        Returns breakdown dict with individual component scores + total.
        """
        import json

        strategy_type = strategy.get("strategy_type", strategy.get("type", ""))
        raw_score = strategy.get("current_score", 0)

        # 1. Base score (normalized 0-1)
        base = min(raw_score, 1.0)

        # 2. Regime alignment bonus
        regime_bonus = 0.0
        if regime_data:
            guidance = regime_data.get("strategy_guidance", {})
            activate_list = guidance.get("activate", [])
            pause_list = guidance.get("pause", [])

            if strategy_type in activate_list:
                regime_bonus = 1.0
            elif strategy_type in pause_list:
                regime_bonus = -0.5  # Penalize (should've been filtered, but safety net)
            else:
                regime_bonus = 0.3  # Neutral — neither activated nor paused

        # 3. Diversification bonus — prefer coins we DON'T already have
        params = strategy.get("parameters", "{}")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                params = {}
        coins = params.get("coins", params.get("coins_traded", []))
        if isinstance(coins, str):
            coins = [coins]
        target_coin = coins[0] if coins else "unknown"

        # Store for logging
        strategy["_decision_coin"] = target_coin

        # Infer direction — regime-aware for all non-explicitly-directional types
        # Only momentum_long/short are unconditionally directional.
        # breakout, trend_following, swing_trading follow the dominant trend direction
        # (downside breakout = short in trending_down; upside breakout = long in trending_up).
        long_types  = {"momentum_long"}
        short_types = {"momentum_short", "contrarian"}

        # Derive regime direction bias (default for all non-explicit strategies)
        overall_regime = (regime_data or {}).get("overall_regime", "unknown")
        regime_conf    = (regime_data or {}).get("overall_confidence", 0.0)
        if overall_regime == "trending_down" and regime_conf >= 0.6:
            regime_default = "short"
        elif overall_regime == "trending_up" and regime_conf >= 0.6:
            regime_default = "long"
        else:
            regime_default = params.get("direction", "long")

        if strategy_type in long_types:
            direction = "long"
        elif strategy_type in short_types:
            direction = "short"
        else:
            # breakout, trend_following, swing_trading, concentrated_bet,
            # mean_reversion, scalping, funding_arb, delta_neutral, etc.
            # — follow the regime when confident, else use stored param
            direction = params.get("direction") or regime_default
        strategy["_decision_side"] = direction

        diversity_bonus = 1.0 if target_coin not in open_coins else 0.0

        # 4. Freshness — prefer recently active strategies
        freshness = 0.5  # Default middle value
        discovered = strategy.get("discovered_at", "")
        if discovered:
            try:
                disc_dt = datetime.fromisoformat(discovered.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - disc_dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
                if age_hours < 24:
                    freshness = 1.0
                elif age_hours < 72:
                    freshness = 0.7
                else:
                    freshness = 0.3
            except (ValueError, TypeError):
                pass

        # 5. Consensus boost (from dedup)
        consensus = min(strategy.get("_dedup_count", 1) / 5.0, 1.0)

        # Weighted composite
        total = (
            base * self.w_score +
            regime_bonus * self.w_regime +
            diversity_bonus * self.w_diversity +
            freshness * self.w_freshness +
            consensus * self.w_consensus
        )

        return {
            "total": round(total, 4),
            "base_score": round(base, 3),
            "regime_alignment": round(regime_bonus, 3),
            "diversity": round(diversity_bonus, 3),
            "freshness": round(freshness, 3),
            "consensus": round(consensus, 3),
        }

    def _aggregate_directional_scores(self, scored: List[Dict]) -> Tuple[float, float]:
        """
        Aggregate long vs short conviction across all candidates.
        This is the "forced convergence" insight from ChatGPT — we compute
        an overall directional bias even though we execute multiple trades.
        """
        long_score = 0.0
        short_score = 0.0

        for s in scored:
            direction = s.get("_decision_side", "long")
            composite = s.get("_composite_score", 0)

            if direction == "long":
                long_score += composite
            elif direction == "short":
                short_score += composite

        return round(long_score, 4), round(short_score, 4)

    def _log_decision(self, scored: List[Dict],
                      regime_data: Optional[Dict],
                      available_slots: int,
                      executions: Optional[List] = None,
                      disqualified: Optional[List] = None,
                      overflow: Optional[List] = None,
                      long_score: float = 0,
                      short_score: float = 0):
        """
        Produce the FINAL_DECISION log block that ChatGPT identified as missing.
        This is the key audit artifact.
        """
        regime = regime_data.get("overall_regime", "unknown") if regime_data else "unknown"
        executions = executions or []
        disqualified = disqualified or []
        overflow = overflow or []

        # Compact decision log — one line per candidate, not 20+ lines
        bias_str = ""
        if long_score > 0 or short_score > 0:
            bias_str = "LONG" if long_score > short_score else "SHORT" if short_score > long_score else "NEUTRAL"

        logger.info("DECISION #%d | regime=%s slots=%d/%d candidates=%d bias=%s",
                    self._cycle_count, regime, available_slots, 5, len(scored), bias_str or "N/A")

        # Log only top 5 candidates on one line each
        for i, s in enumerate(scored[:5]):
            coin = s.get("_decision_coin", "?")
            side = s.get("_decision_side", "?")
            composite = s.get("_composite_score", 0)
            marker = " ← EXEC" if s in executions else ""
            logger.info("  #%d %s %s composite=%.4f%s", i + 1, side.upper(), coin, composite, marker)

        if executions:
            logger.info("-> EXECUTING %d trade(s) this cycle", len(executions))
        else:
            reason = "no candidates" if not scored else \
                     "below threshold" if not [s for s in scored if s.get("_composite_score", 0) >= self.min_decision_score] else \
                     "no slots" if available_slots == 0 else "unknown"
            logger.info("-> NO TRADE this cycle (%s)", reason)

    def get_stats(self) -> Dict:
        """Return decision engine statistics."""
        return {
            **self.stats,
            "cycles": self._cycle_count,
            "execution_rate": (self.stats["total_executions"] /
                              max(self.stats["total_candidates"], 1)),
            "no_trade_rate": (self.stats["total_no_trade"] /
                             max(self.stats["total_decisions"], 1)),
            "recent_decisions": list(self._decision_history)[-10:],
        }

    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        """Return recent decision history for dashboard."""
        return list(self._decision_history)[-limit:]
