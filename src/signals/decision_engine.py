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
from datetime import datetime
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
        self.w_score = cfg.get("w_score", 0.25)             # Strategy score from scorer
        self.w_regime = cfg.get("w_regime", 0.20)           # Regime alignment bonus
        self.w_diversity = cfg.get("w_diversity", 0.15)     # Diversification bonus
        self.w_freshness = cfg.get("w_freshness", 0.05)     # Prefer strategies with recent activity
        self.w_consensus = cfg.get("w_consensus", 0.05)     # Dedup consensus boost
        self.w_confidence = cfg.get("w_confidence", 0.15)   # Calibrated trade confidence
        self.w_source_quality = cfg.get("w_source_quality", 0.10)  # Agent-scorer / accuracy quality
        self.w_confirmation = cfg.get("w_confirmation", 0.05)      # External confirmations
        self.w_expected_value = cfg.get("w_expected_value", 0.20)  # Net expectancy after costs

        # Minimum composite score to even consider executing
        self.min_decision_score = cfg.get("min_decision_score", 0.34)
        self.min_signal_confidence = cfg.get("min_signal_confidence", 0.58)
        self.min_source_weight = cfg.get("min_source_weight", 0.35)
        self.min_expected_value_pct = cfg.get("min_expected_value_pct", 0.0015)

        # Max trades to execute per cycle (independent of position slots)
        self.max_trades_per_cycle = cfg.get("max_trades_per_cycle", 2)

        # Economic quality defaults used when the signal does not explicitly
        # carry enough execution metadata to estimate trade expectancy.
        self.maker_fee_bps = cfg.get("maker_fee_bps", 1.5)
        self.taker_fee_bps = cfg.get("taker_fee_bps", 4.5)
        self.expected_slippage_bps = cfg.get("expected_slippage_bps", 3.0)
        self.churn_penalty_bps = cfg.get("churn_penalty_bps", 2.0)
        self.default_execution_role = str(cfg.get("default_execution_role", "taker") or "taker").lower()

        # Track decisions for audit
        self._decision_history: deque = deque(maxlen=100)
        self._cycle_count = 0

        # Stats
        self.stats = {
            "total_decisions": 0,
            "total_executions": 0,
            "total_no_trade": 0,
            "total_candidates": 0,
            "total_filtered": 0,
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
        qualified = []
        disqualified = []
        for strategy in scored:
            blockers = self._decision_blockers(strategy)
            strategy["_decision_blockers"] = blockers
            if blockers or strategy["_composite_score"] < self.min_decision_score:
                disqualified.append(strategy)
            else:
                qualified.append(strategy)

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
        self.stats["total_filtered"] += len(disqualified)
        if executions:
            self.stats["total_executions"] += len(executions)
        else:
            self.stats["total_no_trade"] += 1

        # ─── Store decision for audit trail ──────────────
        self._decision_history.append({
            "cycle": self._cycle_count,
            "timestamp": datetime.utcnow().isoformat(),
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
                 "composite": e["_composite_score"],
                 "net_expectancy_pct": round(float(e.get("_expected_value_pct", 0.0) or 0.0), 4)}
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
        confidence = min(max(float(strategy.get("confidence", raw_score) or raw_score), 0.0), 1.0)

        # 1. Base score (normalized 0-1)
        base = min(raw_score, 1.0)

        # 2. Regime alignment
        regime_alignment = 0.55
        if regime_data:
            guidance = regime_data.get("strategy_guidance", {})
            activate_list = guidance.get("activate", [])
            pause_list = guidance.get("pause", [])

            if strategy_type in activate_list:
                regime_alignment = 1.0
            elif strategy_type in pause_list:
                regime_alignment = 0.0
            else:
                regime_alignment = 0.55

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
                age_hours = (datetime.utcnow() - disc_dt.replace(tzinfo=None)).total_seconds() / 3600
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

        source_quality = min(
            max(
                float(
                    strategy.get("agent_scorer_weight", strategy.get("source_accuracy", 0.5)) or 0.5
                ),
                0.0,
            ),
            1.0,
        )
        confirmation = self._confirmation_score(strategy)
        expected_value = self._expected_value_score(
            strategy,
            confidence=confidence,
            source_quality=source_quality,
            confirmation=confirmation,
            direction=direction,
            target_coin=target_coin,
            regime_alignment=regime_alignment,
            open_coins=open_coins,
            kelly_stats=kelly_stats,
        )

        components = [
            (base, self.w_score),
            (regime_alignment, self.w_regime),
            (diversity_bonus, self.w_diversity),
            (freshness, self.w_freshness),
            (consensus, self.w_consensus),
            (confidence, self.w_confidence),
            (source_quality, self.w_source_quality),
            (confirmation, self.w_confirmation),
            (expected_value["score"], self.w_expected_value),
        ]
        weight_total = sum(weight for _, weight in components if weight > 0)
        total = (
            sum(score * weight for score, weight in components) / weight_total
            if weight_total > 0
            else 0.0
        )

        strategy["_expected_value_pct"] = expected_value["net_expectancy_pct"]
        strategy["_execution_cost_pct"] = expected_value["execution_cost_pct"]
        strategy["_estimated_win_prob"] = expected_value["win_probability"]
        strategy["_risk_reward_ratio"] = expected_value["reward_risk_ratio"]

        return {
            "total": round(total, 4),
            "base_score": round(base, 3),
            "regime_alignment": round(regime_alignment, 3),
            "diversity": round(diversity_bonus, 3),
            "freshness": round(freshness, 3),
            "consensus": round(consensus, 3),
            "confidence": round(confidence, 3),
            "source_quality": round(source_quality, 3),
            "confirmation": round(confirmation, 3),
            "expected_value": round(expected_value["score"], 3),
            "win_probability": round(expected_value["win_probability"], 3),
            "reward_risk_ratio": round(expected_value["reward_risk_ratio"], 3),
            "gross_expectancy_pct": round(expected_value["gross_expectancy_pct"], 4),
            "net_expectancy_pct": round(expected_value["net_expectancy_pct"], 4),
            "execution_cost_pct": round(expected_value["execution_cost_pct"], 4),
            "kelly_has_edge": expected_value["kelly_has_edge"],
        }

    def _decision_blockers(self, strategy: Dict) -> List[str]:
        """Hard floors that keep low-quality candidates out before execution."""
        blockers: List[str] = []
        confidence = float(strategy.get("confidence", 0.0) or 0.0)
        source_quality = float(
            strategy.get("agent_scorer_weight", strategy.get("source_accuracy", 0.5)) or 0.5
        )

        if confidence < self.min_signal_confidence:
            blockers.append(f"confidence<{self.min_signal_confidence:.2f}")
        if source_quality < self.min_source_weight:
            blockers.append(f"source_weight<{self.min_source_weight:.2f}")
        breakdown = strategy.get("_score_breakdown", {})
        net_expectancy_pct = float(
            breakdown.get(
                "net_expectancy_pct",
                strategy.get("_expected_value_pct", 0.0),
            )
            or 0.0
        )
        if net_expectancy_pct < self.min_expected_value_pct:
            blockers.append(f"net_ev<{self.min_expected_value_pct:.4f}")

        return blockers

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(value, upper))

    def _get_metadata(self, strategy: Dict) -> Dict:
        metadata = strategy.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _extract_risk_geometry(self, strategy: Dict, direction: str) -> Dict[str, float]:
        metadata = self._get_metadata(strategy)
        params = strategy.get("parameters", {})
        if not isinstance(params, dict):
            params = {}

        entry_price = float(
            strategy.get("entry_price", metadata.get("entry_price", params.get("entry_price", 0.0))) or 0.0
        )
        stop_loss = float(
            strategy.get("stop_loss", metadata.get("stop_loss", params.get("stop_loss", 0.0))) or 0.0
        )
        take_profit = float(
            strategy.get("take_profit", metadata.get("take_profit", params.get("take_profit", 0.0))) or 0.0
        )

        risk_pct = 0.0
        reward_pct = 0.0
        if entry_price > 0:
            if direction == "short":
                if stop_loss > entry_price:
                    risk_pct = (stop_loss - entry_price) / entry_price
                if 0 < take_profit < entry_price:
                    reward_pct = (entry_price - take_profit) / entry_price
            else:
                if 0 < stop_loss < entry_price:
                    risk_pct = (entry_price - stop_loss) / entry_price
                if take_profit > entry_price:
                    reward_pct = (take_profit - entry_price) / entry_price

        if risk_pct <= 0:
            risk_pct = float(
                metadata.get(
                    "stop_loss_pct",
                    metadata.get("risk_pct", params.get("stop_loss_pct", 0.0)),
                )
                or 0.0
            )
        if reward_pct <= 0:
            reward_pct = float(
                metadata.get(
                    "take_profit_pct",
                    metadata.get("reward_pct", params.get("take_profit_pct", 0.0)),
                )
                or 0.0
            )

        source = str(strategy.get("source", metadata.get("source", "strategy"))).strip().lower()
        if risk_pct <= 0 or reward_pct <= 0:
            atr_pct = float(metadata.get("atr_pct", 0.0) or 0.0)
            if atr_pct > 0:
                risk_pct = risk_pct or max(0.0125, min(atr_pct * 1.2, 0.06))
                reward_pct = reward_pct or max(risk_pct * 2.0, atr_pct * 1.8)

        if risk_pct <= 0 or reward_pct <= 0:
            if source == "copy_trade":
                risk_pct = risk_pct or 0.04
                reward_pct = reward_pct or 0.08
            elif source in ("options_flow", "liquidation_strategy", "arena_champion"):
                risk_pct = risk_pct or 0.025
                reward_pct = reward_pct or 0.05
            else:
                risk_pct = risk_pct or 0.03
                reward_pct = reward_pct or 0.06

        risk_pct = max(risk_pct, 0.005)
        reward_pct = max(reward_pct, 0.005)
        return {
            "risk_pct": risk_pct,
            "reward_pct": reward_pct,
            "reward_risk_ratio": reward_pct / risk_pct if risk_pct > 0 else 0.0,
        }

    def _lookup_kelly_stats(self, strategy: Dict, kelly_stats: Optional[Dict]) -> Dict:
        if not isinstance(kelly_stats, dict):
            return {}
        keys = [
            strategy.get("source_key"),
            strategy.get("strategy_type"),
            strategy.get("name"),
        ]
        for key in keys:
            if key and key in kelly_stats:
                stats = kelly_stats.get(key)
                if isinstance(stats, dict):
                    return stats
        return {}

    def _expected_value_score(
        self,
        strategy: Dict,
        *,
        confidence: float,
        source_quality: float,
        confirmation: float,
        direction: str,
        target_coin: str,
        regime_alignment: float,
        open_coins: set,
        kelly_stats: Optional[Dict],
    ) -> Dict[str, float]:
        metadata = self._get_metadata(strategy)
        geometry = self._extract_risk_geometry(strategy, direction)
        kelly = self._lookup_kelly_stats(strategy, kelly_stats)
        kelly_win_rate = None
        if kelly:
            try:
                kelly_win_rate = float(kelly.get("win_rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                kelly_win_rate = None

        weighted_inputs = [
            (confidence, 0.50),
            (source_quality, 0.25),
            (confirmation, 0.15),
        ]
        if kelly_win_rate is not None:
            weighted_inputs.append((self._clamp(kelly_win_rate, 0.0, 1.0), 0.10))

        weight_sum = sum(weight for _, weight in weighted_inputs) or 1.0
        win_probability = sum(value * weight for value, weight in weighted_inputs) / weight_sum
        win_probability += (regime_alignment - 0.5) * 0.06
        if kelly and kelly.get("has_edge") is True:
            win_probability += 0.03
        elif kelly and int(kelly.get("trades", 0) or 0) >= 15:
            win_probability -= 0.02
        win_probability = self._clamp(win_probability, 0.01, 0.99)

        execution_role = str(
            metadata.get("execution_role", self.default_execution_role) or self.default_execution_role
        ).lower()
        if execution_role == "maker":
            fee_bps = float(metadata.get("maker_fee_bps", self.maker_fee_bps) or self.maker_fee_bps)
        else:
            fee_bps = float(metadata.get("taker_fee_bps", self.taker_fee_bps) or self.taker_fee_bps)

        slippage_bps = float(
            metadata.get(
                "expected_slippage_bps",
                metadata.get("slippage_bps", self.expected_slippage_bps),
            )
            or self.expected_slippage_bps
        )
        churn_penalty_bps = float(metadata.get("churn_penalty_bps", self.churn_penalty_bps) or self.churn_penalty_bps)
        if target_coin in open_coins:
            churn_penalty_bps *= 1.5

        execution_cost_pct = ((fee_bps + slippage_bps) * 2.0 + churn_penalty_bps) / 10_000.0
        gross_expectancy_pct = (
            win_probability * geometry["reward_pct"]
            - (1.0 - win_probability) * geometry["risk_pct"]
        )
        net_expectancy_pct = gross_expectancy_pct - execution_cost_pct
        score_bound = max(geometry["reward_pct"] + geometry["risk_pct"], 0.04)
        normalized_score = self._clamp(0.5 + (net_expectancy_pct / score_bound), 0.0, 1.0)

        return {
            "score": normalized_score,
            "win_probability": win_probability,
            "reward_risk_ratio": geometry["reward_risk_ratio"],
            "gross_expectancy_pct": gross_expectancy_pct,
            "net_expectancy_pct": net_expectancy_pct,
            "execution_cost_pct": execution_cost_pct,
            "kelly_has_edge": bool(kelly.get("has_edge", False)) if kelly else False,
        }

    def _confirmation_score(self, strategy: Dict) -> float:
        """Score how much independent confirmation a candidate has."""
        metadata = self._get_metadata(strategy)

        confirmations = []
        if strategy.get("options_flow_aligned") is True:
            confirmations.append(1.0)
        elif strategy.get("options_flow_aligned") is False:
            confirmations.append(0.0)

        if strategy.get("volume_confirmed") is True:
            confirmations.append(1.0)

        cross_venue = float(metadata.get("cross_venue_score", 0.0) or 0.0)
        if cross_venue:
            confirmations.append(min(max(cross_venue, 0.0), 1.0))

        source = str(strategy.get("source", metadata.get("source", "strategy"))).strip().lower()
        if source in ("polymarket", "options_flow", "arena_champion"):
            confirmations.append(0.8)

        if not confirmations:
            return 0.25
        return round(sum(confirmations) / len(confirmations), 3)

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
            expected_value_pct = float(s.get("_expected_value_pct", 0.0) or 0.0) * 100.0
            execution_cost_pct = float(s.get("_execution_cost_pct", 0.0) or 0.0) * 100.0
            marker = " ← EXEC" if s in executions else ""
            logger.info(
                "  #%d %s %s composite=%.4f ev=%+.2f%% cost=%.2f%%%s",
                i + 1,
                side.upper(),
                coin,
                composite,
                expected_value_pct,
                execution_cost_pct,
                marker,
            )

        if executions:
            logger.info("→ EXECUTING %d trade(s) this cycle", len(executions))
        else:
            reason = "no candidates" if not scored else \
                     "below threshold" if not [s for s in scored if s.get("_composite_score", 0) >= self.min_decision_score] else \
                     "no slots" if available_slots == 0 else "unknown"
            logger.info("→ NO TRADE this cycle (%s)", reason)

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
