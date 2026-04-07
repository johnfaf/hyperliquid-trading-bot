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
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

from src.data import database as db

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
        self.w_confluence = cfg.get("w_confluence", 0.10)          # Cross-source same-coin agreement
        self.w_context = cfg.get("w_context", 0.08)                # Source x coin x side x regime fitness
        self.w_calibration = cfg.get("w_calibration", 0.06)        # Confidence honesty / ECE quality
        self.w_memory = cfg.get("w_memory", 0.07)                  # Similar historical setup fitness
        self.w_divergence = cfg.get("w_divergence", 0.08)          # Paper/shadow/live alignment quality

        # Minimum composite score to even consider executing
        self.min_decision_score = cfg.get("min_decision_score", 0.34)
        self.min_signal_confidence = cfg.get("min_signal_confidence", 0.58)
        self.min_source_weight = cfg.get("min_source_weight", 0.35)
        self.min_expected_value_pct = cfg.get("min_expected_value_pct", 0.0015)
        self.confluence_enabled = bool(cfg.get("confluence_enabled", True))
        self.confluence_baseline = float(cfg.get("confluence_baseline", 0.30))
        self.confluence_full_weight = float(cfg.get("confluence_full_weight", 1.50))
        self.confluence_target_support_sources = int(cfg.get("confluence_target_support_sources", 2))
        self.confluence_conflict_block_threshold = float(
            cfg.get("confluence_conflict_block_threshold", 0.65)
        )
        self.confluence_conflict_floor = float(cfg.get("confluence_conflict_floor", 0.35))
        self.context_performance_enabled = bool(cfg.get("context_performance_enabled", True))
        self.context_performance_lookback_hours = float(
            cfg.get("context_performance_lookback_hours", 24.0 * 30)
        )
        self.context_performance_min_trades = int(cfg.get("context_performance_min_trades", 3))
        self.context_performance_return_scale = float(cfg.get("context_performance_return_scale", 0.03))
        self.context_performance_block_win_rate = float(
            cfg.get("context_performance_block_win_rate", 0.30)
        )
        self.context_performance_block_avg_return_pct = float(
            cfg.get("context_performance_block_avg_return_pct", -0.015)
        )
        self.context_performance_boost_win_rate = float(
            cfg.get("context_performance_boost_win_rate", 0.60)
        )
        self.calibration = cfg.get("calibration")
        self.calibration_enabled = bool(
            cfg.get("calibration_enabled", self.calibration is not None)
        )
        self.calibration_min_records = int(cfg.get("calibration_min_records", 20))
        self.calibration_target_ece = float(cfg.get("calibration_target_ece", 0.05))
        self.calibration_max_ece = float(cfg.get("calibration_max_ece", 0.20))
        self.trade_memory = cfg.get("trade_memory")
        self.memory_enabled = bool(
            cfg.get("memory_enabled", self.trade_memory is not None)
        )
        self.memory_min_trades = int(cfg.get("memory_min_trades", 3))
        self.memory_min_similarity = float(cfg.get("memory_min_similarity", 0.55))
        self.memory_top_k = int(cfg.get("memory_top_k", 8))
        self.memory_block_on_avoid = bool(cfg.get("memory_block_on_avoid", True))
        self.divergence_controller = cfg.get("divergence_controller")
        self.divergence_enabled = bool(
            cfg.get("divergence_enabled", self.divergence_controller is not None)
        )
        self.divergence_block_on_status = bool(cfg.get("divergence_block_on_status", True))

        # Max trades to execute per cycle (independent of position slots)
        self.max_trades_per_cycle = cfg.get("max_trades_per_cycle", 2)

        # Economic quality defaults used when the signal does not explicitly
        # carry enough execution metadata to estimate trade expectancy.
        self.maker_fee_bps = cfg.get("maker_fee_bps", 1.5)
        self.taker_fee_bps = cfg.get("taker_fee_bps", 4.5)
        self.expected_slippage_bps = cfg.get("expected_slippage_bps", 3.0)
        self.churn_penalty_bps = cfg.get("churn_penalty_bps", 2.0)
        self.default_execution_role = str(cfg.get("default_execution_role", "taker") or "taker").lower()
        self.persist_research = bool(cfg.get("persist_research", False))
        self._last_research_cycle_id: Optional[int] = None
        self.execution_quality_enabled = bool(cfg.get("execution_quality_enabled", True))
        self.execution_quality_lookback_hours = float(cfg.get("execution_quality_lookback_hours", 24.0 * 7))
        self.execution_quality_min_events = int(cfg.get("execution_quality_min_events", 3))
        self.execution_rejection_penalty_bps = float(cfg.get("execution_rejection_penalty_bps", 12.0))
        self.execution_fill_gap_penalty_bps = float(cfg.get("execution_fill_gap_penalty_bps", 8.0))
        self.execution_protective_failure_penalty_bps = float(
            cfg.get("execution_protective_failure_penalty_bps", 18.0)
        )
        self.execution_policy = cfg.get("execution_policy")
        self.execution_policy_enabled = bool(
            cfg.get("execution_policy_enabled", self.execution_policy is not None)
        )
        self.adaptive_learning = cfg.get("adaptive_learning")
        self.adaptive_learning_enabled = bool(cfg.get("adaptive_learning_enabled", True))
        self.adaptive_learning_block_on_status = bool(
            cfg.get("adaptive_learning_block_on_status", True)
        )
        self.adaptive_learning_min_health_score = float(
            cfg.get("adaptive_learning_min_health_score", 0.42)
        )

        # Track decisions for audit
        self._decision_history: deque = deque(maxlen=100)
        self._cycle_count = 0
        self._context_profile_cache: Dict[Tuple, Dict] = {}
        self._calibration_profile_cache: Dict[Tuple, Dict] = {}
        self._calibration_stats_snapshot: Optional[Dict[str, Dict]] = None
        self._memory_profile_cache: Dict[Tuple, Dict] = {}
        self._divergence_profile_cache: Dict[Tuple, Dict] = {}

        # Stats
        self.stats = {
            "total_decisions": 0,
            "total_executions": 0,
            "total_no_trade": 0,
            "total_candidates": 0,
            "total_filtered": 0,
            "total_source_conflict_blocks": 0,
            "total_context_blocks": 0,
            "total_calibration_blocks": 0,
            "total_memory_blocks": 0,
            "total_divergence_blocks": 0,
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
        self._context_profile_cache = {}
        self._calibration_profile_cache = {}
        self._calibration_stats_snapshot = None
        self._memory_profile_cache = {}
        self._divergence_profile_cache = {}
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
        confluence_map = self._build_source_confluence_map(strategies, regime_data)
        for s in strategies:
            composite = self._compute_composite_score(
                s, regime_data, open_coins, kelly_stats, confluence_map=confluence_map
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
        self.stats["total_source_conflict_blocks"] += sum(
            1 for item in disqualified if "source_conflict" in item.get("_decision_blockers", [])
        )
        self.stats["total_context_blocks"] += sum(
            1 for item in disqualified if "context_underperforming" in item.get("_decision_blockers", [])
        )
        self.stats["total_calibration_blocks"] += sum(
            1 for item in disqualified if "calibration_poor" in item.get("_decision_blockers", [])
        )
        self.stats["total_memory_blocks"] += sum(
            1 for item in disqualified if "memory_avoid" in item.get("_decision_blockers", [])
        )
        self.stats["total_divergence_blocks"] += sum(
            1 for item in disqualified if "divergence_guard" in item.get("_decision_blockers", [])
        )
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
                 "net_expectancy_pct": round(float(e.get("_expected_value_pct", 0.0) or 0.0), 4),
                 "confluence_score": round(
                     float((e.get("_source_confluence", {}) or {}).get("confluence_score", 0.0) or 0.0),
                     4,
                 )}
                for e in executions
            ],
        })
        if self.persist_research:
            self._persist_decision_research(
                scored=scored,
                executions=executions,
                disqualified=disqualified,
                overflow=overflow,
                regime_data=regime_data,
                open_positions=open_positions,
                available_slots=available_slots,
                long_score=long_score,
                short_score=short_score,
                kelly_stats=kelly_stats,
            )

        # deque(maxlen=100) auto-trims old entries

        return executions

    def _persist_decision_research(
        self,
        *,
        scored: List[Dict],
        executions: List[Dict],
        disqualified: List[Dict],
        overflow: List[Dict],
        regime_data: Optional[Dict],
        open_positions: Optional[List[Dict]],
        available_slots: int,
        long_score: float,
        short_score: float,
        kelly_stats: Optional[Dict],
    ) -> None:
        execution_ids = {id(item) for item in executions}
        disqualified_ids = {id(item) for item in disqualified}
        overflow_ids = {id(item) for item in overflow}

        serialized_candidates = []
        for rank, candidate in enumerate(scored, start=1):
            if id(candidate) in execution_ids:
                status = "selected"
            elif id(candidate) in disqualified_ids:
                status = "blocked"
            elif id(candidate) in overflow_ids:
                status = "overflow"
            else:
                status = "ranked"

            serialized_candidates.append(
                {
                    "rank": rank,
                    "status": status,
                    "name": candidate.get("name"),
                    "source": str(candidate.get("source", "")),
                    "source_key": candidate.get("source_key"),
                    "strategy_type": candidate.get("strategy_type"),
                    "coin": candidate.get("_decision_coin"),
                    "side": candidate.get("_decision_side"),
                    "route": str(candidate.get("_decision_route", "paper_strategy") or "paper_strategy"),
                    "composite_score": candidate.get("_composite_score", 0.0),
                    "confidence": candidate.get("confidence", 0.0),
                    "expected_value_pct": candidate.get("_expected_value_pct", 0.0),
                    "execution_cost_pct": candidate.get("_execution_cost_pct", 0.0),
                    "blockers": candidate.get("_decision_blockers", []),
                    "score_breakdown": candidate.get("_score_breakdown", {}),
                    "raw_candidate": self._sanitize_candidate_for_research(candidate),
                }
            )

        tracked_keys = {
            candidate.get("source_key")
            for candidate in scored
            if candidate.get("source_key")
        }
        kelly_summary = {}
        if isinstance(kelly_stats, dict):
            for key in tracked_keys:
                if key in kelly_stats and isinstance(kelly_stats[key], dict):
                    kelly_summary[key] = dict(kelly_stats[key])

        market_bias = (
            "long" if long_score > short_score else "short" if short_score > long_score else "neutral"
        )
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle_number": self._cycle_count,
            "regime": regime_data.get("overall_regime", "unknown") if regime_data else "unknown",
            "available_slots": available_slots,
            "candidate_count": len(scored),
            "qualified_count": len(scored) - len(disqualified),
            "executed_count": len(executions),
            "long_score": long_score,
            "short_score": short_score,
            "market_bias": market_bias,
            "context": {
                "regime_data": regime_data or {},
                "open_positions": open_positions or [],
                "kelly_summary": kelly_summary,
            },
            "candidates": serialized_candidates,
        }

        try:
            self._last_research_cycle_id = db.save_decision_research_snapshot(snapshot)
        except Exception as exc:
            logger.debug("Decision research persistence error: %s", exc)

    def _sanitize_candidate_for_research(self, candidate: Dict) -> Dict:
        """Strip transient fields that are noisy or unserializable for research storage."""
        payload = {}
        for key, value in candidate.items():
            if key in {"_lcrs_signal", "_copy_signal"}:
                payload[key] = value
                continue
            if key == "raw":
                continue
            if key.startswith("_") and key not in {
                "_decision_coin",
                "_decision_side",
                "_decision_route",
                "_composite_score",
                "_expected_value_pct",
                "_execution_cost_pct",
                "_decision_blockers",
                "_score_breakdown",
            }:
                continue
            payload[key] = value
        # Round-trip through JSON so datetimes/enums become stable strings.
        return json.loads(json.dumps(payload, default=str))

    def _resolve_candidate_context(self, strategy: Dict, regime_data: Optional[Dict]) -> Dict:
        import json

        strategy_type = strategy.get("strategy_type", strategy.get("type", ""))
        raw_score = strategy.get("current_score", 0)
        confidence = min(max(float(strategy.get("confidence", raw_score) or raw_score), 0.0), 1.0)

        params = strategy.get("parameters", {})
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                params = {}
        params = dict(params or {})

        coins = params.get("coins", params.get("coins_traded", []))
        if isinstance(coins, str):
            coins = [coins]
        target_coin = str(coins[0] if coins else "unknown").strip().upper() or "UNKNOWN"

        long_types = {"momentum_long"}
        short_types = {"momentum_short", "contrarian"}

        overall_regime = (regime_data or {}).get("overall_regime", "unknown")
        regime_conf = float((regime_data or {}).get("overall_confidence", 0.0) or 0.0)
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
            direction = params.get("direction") or regime_default

        metadata = strategy.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source = str(
            strategy.get("source", metadata.get("source", "strategy")) or "strategy"
        ).strip().lower()
        source_key = str(
            strategy.get("source_key", metadata.get("source_key", "")) or ""
        ).strip()

        return {
            "strategy_type": strategy_type,
            "confidence": confidence,
            "params": params,
            "target_coin": target_coin,
            "direction": direction,
            "source": source,
            "source_key": source_key,
            "metadata": metadata,
        }

    def _build_source_confluence_map(
        self,
        strategies: List[Dict],
        regime_data: Optional[Dict],
    ) -> Dict[int, Dict]:
        default = {
            "confluence_score": round(self.confluence_baseline, 4),
            "support_score": 0.0,
            "conflict_score": 0.0,
            "supporting_sources": [],
            "conflicting_sources": [],
            "same_coin_candidates": 0,
            "support_source_count": 0,
            "conflict_source_count": 0,
        }
        if not self.confluence_enabled:
            return {id(strategy): dict(default) for strategy in strategies}

        rows = []
        for strategy in strategies:
            context = self._resolve_candidate_context(strategy, regime_data)
            source_quality = min(
                max(
                    float(
                        strategy.get("agent_scorer_weight", strategy.get("source_accuracy", 0.5)) or 0.5
                    ),
                    0.0,
                ),
                1.0,
            )
            rows.append(
                {
                    "id": id(strategy),
                    "coin": context["target_coin"],
                    "side": context["direction"],
                    "source_bucket": context["source_key"] or context["source"] or "unknown",
                    "display_source": context["source_key"] or context["source"] or "unknown",
                    "weight": self._clamp((context["confidence"] * 0.60) + (source_quality * 0.40), 0.05, 1.0),
                }
            )

        full_weight = max(self.confluence_full_weight, 1e-6)
        target_support_sources = max(self.confluence_target_support_sources, 1)
        confluence_map: Dict[int, Dict] = {}
        for row in rows:
            support_by_source: Dict[str, float] = {}
            conflict_by_source: Dict[str, float] = {}
            same_coin_candidates = 0
            for other in rows:
                if other["id"] == row["id"] or other["coin"] != row["coin"]:
                    continue
                same_coin_candidates += 1
                bucket = support_by_source if other["side"] == row["side"] else conflict_by_source
                bucket[other["source_bucket"]] = max(
                    bucket.get(other["source_bucket"], 0.0),
                    other["weight"],
                )

            support_score = self._clamp(sum(support_by_source.values()) / full_weight, 0.0, 1.0)
            conflict_score = self._clamp(sum(conflict_by_source.values()) / full_weight, 0.0, 1.0)
            support_diversity = self._clamp(len(support_by_source) / target_support_sources, 0.0, 1.0)
            confluence_score = self._clamp(
                self.confluence_baseline
                + (support_score * 0.45)
                + (support_diversity * 0.20)
                - (conflict_score * 0.60),
                0.0,
                1.0,
            )

            confluence_map[row["id"]] = {
                "confluence_score": round(confluence_score, 4),
                "support_score": round(support_score, 4),
                "conflict_score": round(conflict_score, 4),
                "supporting_sources": sorted(support_by_source.keys()),
                "conflicting_sources": sorted(conflict_by_source.keys()),
                "same_coin_candidates": same_coin_candidates,
                "support_source_count": len(support_by_source),
                "conflict_source_count": len(conflict_by_source),
            }

        return confluence_map

    def _compute_composite_score(self, strategy: Dict,
                                  regime_data: Optional[Dict],
                                  open_coins: set,
                                  kelly_stats: Optional[Dict],
                                  confluence_map: Optional[Dict[int, Dict]] = None) -> Dict:
        """
        Compute a composite ranking score from multiple factors.
        Returns breakdown dict with individual component scores + total.
        """
        context = self._resolve_candidate_context(strategy, regime_data)
        strategy_type = context["strategy_type"]
        confidence = context["confidence"]
        target_coin = context["target_coin"]
        direction = context["direction"]
        regime_name = str((regime_data or {}).get("overall_regime", "unknown") or "unknown")

        # 1. Base score (normalized 0-1)
        raw_score = strategy.get("current_score", 0)
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
        # Store for logging
        strategy["_decision_coin"] = target_coin
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
        execution_quality = self._apply_execution_quality_feedback(strategy, source_quality)
        source_quality = execution_quality["source_quality"]
        strategy["_metadata_for_decision"] = execution_quality["metadata"]
        strategy["_execution_quality"] = execution_quality["profile"]
        divergence_feedback = self._apply_divergence_feedback(strategy, confidence, source_quality)
        confidence = divergence_feedback["confidence"]
        source_quality = divergence_feedback["source_quality"]
        strategy["_metadata_for_decision"] = divergence_feedback["metadata"]
        strategy["_divergence_control"] = divergence_feedback["profile"]
        divergence_score = float(divergence_feedback["divergence_score"] or 0.5)
        calibration_feedback = self._apply_calibration_feedback(strategy, confidence, source_quality)
        confidence = calibration_feedback["confidence"]
        source_quality = calibration_feedback["source_quality"]
        strategy["_metadata_for_decision"] = calibration_feedback["metadata"]
        strategy["_calibration"] = calibration_feedback["profile"]
        calibration_score = float(calibration_feedback["calibration_score"] or 0.5)
        memory_feedback = self._apply_trade_memory_feedback(
            strategy,
            confidence,
            source_quality,
            target_coin=target_coin,
            direction=direction,
            strategy_type=strategy_type,
        )
        confidence = memory_feedback["confidence"]
        source_quality = memory_feedback["source_quality"]
        strategy["_metadata_for_decision"] = memory_feedback["metadata"]
        strategy["_trade_memory"] = memory_feedback["profile"]
        memory_score = float(memory_feedback["memory_score"] or 0.5)
        adaptive_feedback = self._apply_adaptive_learning_feedback(strategy, confidence, source_quality)
        confidence = adaptive_feedback["confidence"]
        source_quality = adaptive_feedback["source_quality"]
        strategy["_metadata_for_decision"] = adaptive_feedback["metadata"]
        strategy["_adaptive_learning"] = adaptive_feedback["profile"]
        context_feedback = self._apply_context_performance_feedback(
            strategy,
            confidence,
            source_quality,
            target_coin=target_coin,
            direction=direction,
            regime_name=regime_name,
        )
        confidence = context_feedback["confidence"]
        source_quality = context_feedback["source_quality"]
        strategy["_metadata_for_decision"] = context_feedback["metadata"]
        strategy["_context_performance"] = context_feedback["profile"]
        context_score = float(context_feedback["context_score"] or 0.5)
        execution_policy = self._apply_execution_policy_feedback(strategy, confidence, source_quality)
        strategy["_metadata_for_decision"] = execution_policy["metadata"]
        strategy["_execution_policy"] = execution_policy["recommendation"]
        strategy["metadata"] = dict(strategy["_metadata_for_decision"])
        strategy["_decision_confidence"] = confidence
        strategy["_decision_source_quality"] = source_quality
        confluence_feedback = (confluence_map or {}).get(id(strategy), {})
        confluence_score = float(confluence_feedback.get("confluence_score", self.confluence_baseline) or 0.0)
        strategy["_source_confluence"] = confluence_feedback
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
            (confluence_score, self.w_confluence),
            (context_score, self.w_context),
            (calibration_score, self.w_calibration),
            (memory_score, self.w_memory),
            (divergence_score, self.w_divergence),
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
            "confluence": round(confluence_score, 3),
            "context": round(context_score, 3),
            "calibration": round(calibration_score, 3),
            "memory": round(memory_score, 3),
            "divergence": round(divergence_score, 3),
            "confluence_support": round(float(confluence_feedback.get("support_score", 0.0) or 0.0), 4),
            "confluence_conflict": round(float(confluence_feedback.get("conflict_score", 0.0) or 0.0), 4),
            "confluence_support_sources": list(confluence_feedback.get("supporting_sources", [])),
            "confluence_conflict_sources": list(confluence_feedback.get("conflicting_sources", [])),
            "context_closed_trades": int(context_feedback["profile"].get("closed_trades", 0) or 0),
            "context_win_rate": round(
                float(context_feedback["profile"].get("win_rate", 0.0) or 0.0),
                4,
            ),
            "context_avg_return_pct": round(
                float(context_feedback["profile"].get("avg_return_pct", 0.0) or 0.0),
                4,
            ),
            "context_profit_factor": round(
                float(context_feedback["profile"].get("profit_factor", 0.0) or 0.0),
                4,
            ),
            "context_regime": str(context_feedback["profile"].get("regime", regime_name) or regime_name),
            "calibration_ece": round(
                float(calibration_feedback["profile"].get("ece", 0.0) or 0.0),
                4,
            ) if calibration_feedback["profile"].get("ece") is not None else None,
            "calibration_total_records": int(
                calibration_feedback["profile"].get("total_records", 0) or 0
            ),
            "calibration_quality": str(
                calibration_feedback["profile"].get("calibration_quality", "unknown") or "unknown"
            ),
            "calibration_lookup_key": str(
                calibration_feedback["profile"].get("lookup_key", "") or ""
            ),
            "calibration_used_global": bool(
                calibration_feedback["profile"].get("used_global", False)
            ),
            "memory_total_found": int(memory_feedback["profile"].get("total_found", 0) or 0),
            "memory_win_rate": round(
                float(memory_feedback["profile"].get("win_rate", 0.0) or 0.0),
                4,
            ),
            "memory_avg_return": round(
                float(memory_feedback["profile"].get("avg_return", 0.0) or 0.0),
                4,
            ),
            "memory_avg_pnl": round(
                float(memory_feedback["profile"].get("avg_pnl", 0.0) or 0.0),
                4,
            ),
            "memory_avg_similarity": round(
                float(memory_feedback["profile"].get("avg_similarity", 0.0) or 0.0),
                4,
            ),
            "memory_recommendation": str(
                memory_feedback["profile"].get("recommendation", "unknown") or "unknown"
            ),
            "memory_reason": str(memory_feedback["profile"].get("reason", "") or ""),
            "memory_blocked": bool(memory_feedback["profile"].get("blocked", False)),
            "divergence_status": str(
                divergence_feedback["profile"].get("status", "unknown") or "unknown"
            ),
            "divergence_multiplier": round(
                float(divergence_feedback["profile"].get("multiplier", 1.0) or 1.0),
                4,
            ),
            "divergence_reason_count": len(divergence_feedback["profile"].get("reasons", []) or []),
            "divergence_global_status": str(
                (divergence_feedback["profile"].get("global", {}) or {}).get("status", "unknown") or "unknown"
            ),
            "divergence_source_status": str(
                (divergence_feedback["profile"].get("source_profile", {}) or {}).get("status", "unknown")
                or "unknown"
            ),
            "divergence_blocked": bool(divergence_feedback["profile"].get("blocked", False)),
            "calibrated_confidence": round(confidence, 4),
            "win_probability": round(expected_value["win_probability"], 3),
            "reward_risk_ratio": round(expected_value["reward_risk_ratio"], 3),
            "gross_expectancy_pct": round(expected_value["gross_expectancy_pct"], 4),
            "net_expectancy_pct": round(expected_value["net_expectancy_pct"], 4),
            "execution_cost_pct": round(expected_value["execution_cost_pct"], 4),
            "kelly_has_edge": expected_value["kelly_has_edge"],
            "execution_quality_events": int(
                execution_quality["profile"].get("total_events", 0) or 0
            ),
            "execution_rejection_rate": round(
                float(execution_quality["profile"].get("rejection_rate", 0.0) or 0.0),
                4,
            ),
            "execution_fill_ratio": round(
                float(execution_quality["profile"].get("avg_fill_ratio", 0.0) or 0.0),
                4,
            ),
            "execution_slippage_bps": round(
                float(execution_quality["profile"].get("avg_realized_slippage_bps", 0.0) or 0.0),
                4,
            ),
            "execution_penalty_bps": round(float(execution_quality["penalty_bps"] or 0.0), 4),
            "execution_route": execution_policy["recommendation"].get("execution_route", "market"),
            "execution_policy_reason": execution_policy["recommendation"].get("policy_reason", ""),
            "maker_price_offset_bps": round(
                float(execution_policy["recommendation"].get("maker_price_offset_bps", 0.0) or 0.0),
                4,
            ),
            "execution_urgency_score": round(
                float(execution_policy["recommendation"].get("urgency_score", 0.0) or 0.0),
                4,
            ),
            "adaptive_health_score": round(
                float(adaptive_feedback["profile"].get("health_score", 0.0) or 0.0),
                4,
            ),
            "adaptive_drift_score": round(
                float(adaptive_feedback["profile"].get("drift_score", 0.0) or 0.0),
                4,
            ),
            "adaptive_status": adaptive_feedback["profile"].get("status", "inactive"),
            "adaptive_weight_multiplier": round(float(adaptive_feedback["weight_multiplier"] or 1.0), 4),
            "adaptive_confidence_multiplier": round(
                float(adaptive_feedback["confidence_multiplier"] or 1.0),
                4,
            ),
        }

    def _decision_blockers(self, strategy: Dict) -> List[str]:
        """Hard floors that keep low-quality candidates out before execution."""
        blockers: List[str] = []
        confidence = float(
            strategy.get("_decision_confidence", strategy.get("confidence", 0.0)) or 0.0
        )
        source_quality = float(
            strategy.get(
                "_decision_source_quality",
                strategy.get("agent_scorer_weight", strategy.get("source_accuracy", 0.5)),
            )
            or 0.5
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

        adaptive_profile = strategy.get("_adaptive_learning", {}) or {}
        adaptive_status = str(adaptive_profile.get("status", "") or "").strip().lower()
        adaptive_health_score = float(adaptive_profile.get("health_score", 1.0) or 1.0)
        if (
            self.adaptive_learning_enabled
            and self.adaptive_learning_block_on_status
            and adaptive_status == "blocked"
        ):
            blockers.append("adaptive_blocked")
        elif (
            self.adaptive_learning_enabled
            and adaptive_health_score < self.adaptive_learning_min_health_score
        ):
            blockers.append(f"adaptive_health<{self.adaptive_learning_min_health_score:.2f}")

        context_profile = strategy.get("_context_performance", {}) or {}
        if self.context_performance_enabled and bool(context_profile.get("blocked", False)):
            blockers.append("context_underperforming")

        calibration_profile = strategy.get("_calibration", {}) or {}
        if self.calibration_enabled and bool(calibration_profile.get("blocked", False)):
            blockers.append("calibration_poor")

        memory_profile = strategy.get("_trade_memory", {}) or {}
        if self.memory_enabled and bool(memory_profile.get("blocked", False)):
            blockers.append("memory_avoid")

        divergence_profile = strategy.get("_divergence_control", {}) or {}
        if self.divergence_enabled and bool(divergence_profile.get("blocked", False)):
            blockers.append("divergence_guard")

        confluence = strategy.get("_source_confluence", {}) or {}
        conflict_score = float(confluence.get("conflict_score", 0.0) or 0.0)
        confluence_score = float(confluence.get("confluence_score", self.confluence_baseline) or 0.0)
        if (
            self.confluence_enabled
            and conflict_score >= self.confluence_conflict_block_threshold
            and confluence_score <= self.confluence_conflict_floor
        ):
            blockers.append("source_conflict")

        return blockers

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(value, upper))

    def _get_metadata(self, strategy: Dict) -> Dict:
        staged = strategy.get("_metadata_for_decision", {})
        if isinstance(staged, dict):
            return staged
        metadata = strategy.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _lookup_execution_quality(self, strategy: Dict) -> Dict:
        if not self.execution_quality_enabled:
            return {}

        metadata = strategy.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        candidate_keys = []
        for key in (strategy.get("source_key"), metadata.get("source_key")):
            normalized = str(key or "").strip()
            if normalized and normalized not in candidate_keys:
                candidate_keys.append(normalized)

        source = str(
            strategy.get("source", metadata.get("source", "strategy")) or "strategy"
        ).strip().lower()

        for source_key in candidate_keys:
            profile = db.get_execution_quality_summary(
                source_key=source_key,
                lookback_hours=self.execution_quality_lookback_hours,
            )
            if int(profile.get("total_events", 0) or 0) >= self.execution_quality_min_events:
                return profile

        profile = db.get_execution_quality_summary(
            source=source,
            lookback_hours=self.execution_quality_lookback_hours,
        )
        if int(profile.get("total_events", 0) or 0) >= self.execution_quality_min_events:
            return profile
        return {}

    def _apply_execution_quality_feedback(self, strategy: Dict, source_quality: float) -> Dict:
        metadata = dict(strategy.get("metadata", {}) if isinstance(strategy.get("metadata", {}), dict) else {})
        profile = self._lookup_execution_quality(strategy)
        if not profile:
            return {
                "metadata": metadata,
                "source_quality": source_quality,
                "profile": {},
                "penalty_bps": 0.0,
            }

        avg_slippage_bps = float(profile.get("avg_realized_slippage_bps", 0.0) or 0.0)
        avg_fill_ratio = float(profile.get("avg_fill_ratio", 1.0) or 1.0)
        rejection_rate = float(profile.get("rejection_rate", 0.0) or 0.0)
        protective_failure_rate = float(profile.get("protective_failure_rate", 0.0) or 0.0)
        maker_ratio = float(profile.get("maker_ratio", 0.0) or 0.0)

        penalty_bps = (
            rejection_rate * self.execution_rejection_penalty_bps
            + max(0.0, 1.0 - avg_fill_ratio) * self.execution_fill_gap_penalty_bps
            + protective_failure_rate * self.execution_protective_failure_penalty_bps
        )

        current_slippage_bps = float(
            metadata.get(
                "expected_slippage_bps",
                metadata.get("slippage_bps", self.expected_slippage_bps),
            )
            or self.expected_slippage_bps
        )
        metadata["expected_slippage_bps"] = max(current_slippage_bps, avg_slippage_bps + penalty_bps)
        metadata["execution_quality_penalty_bps"] = round(penalty_bps, 4)
        metadata["historical_fill_ratio"] = round(avg_fill_ratio, 4)
        metadata["historical_rejection_rate"] = round(rejection_rate, 4)
        metadata["historical_protective_failure_rate"] = round(protective_failure_rate, 4)
        if not metadata.get("execution_role"):
            metadata["execution_role"] = "maker" if maker_ratio >= 0.55 else self.default_execution_role

        quality_multiplier = self._clamp(
            1.0
            - (rejection_rate * 0.55)
            - (max(0.0, 1.0 - avg_fill_ratio) * 0.25)
            - (protective_failure_rate * 0.45),
            0.25,
            1.0,
        )

        return {
            "metadata": metadata,
            "source_quality": self._clamp(source_quality * quality_multiplier, 0.0, 1.0),
            "profile": profile,
            "penalty_bps": penalty_bps,
        }

    def _lookup_divergence_profile(self, strategy: Dict) -> Dict:
        if not self.divergence_enabled or not self.divergence_controller:
            return {}

        metadata = self._get_metadata(strategy)
        source_key = str(
            strategy.get("source_key", metadata.get("source_key", "")) or ""
        ).strip()
        source = str(
            strategy.get("source", metadata.get("source", "strategy")) or "strategy"
        ).strip().lower()
        cache_key = (source_key, source)
        if cache_key in self._divergence_profile_cache:
            return self._divergence_profile_cache[cache_key]

        try:
            profile = self.divergence_controller.evaluate(
                source_key=source_key,
                source=source,
            ) or {}
        except Exception as exc:
            logger.debug("divergence controller lookup error: %s", exc)
            profile = {}

        self._divergence_profile_cache[cache_key] = profile
        return profile

    def _apply_divergence_feedback(
        self,
        strategy: Dict,
        confidence: float,
        source_quality: float,
    ) -> Dict:
        metadata = dict(self._get_metadata(strategy))
        if not self.divergence_enabled or not self.divergence_controller:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "divergence_score": 0.5,
                "profile": {},
                "blocked": False,
            }

        profile = self._lookup_divergence_profile(strategy)
        if not profile:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "divergence_score": 0.5,
                "profile": {},
                "blocked": False,
            }

        status = str(profile.get("status", "warming_up") or "warming_up").strip().lower()
        multiplier = self._clamp(float(profile.get("multiplier", 1.0) or 1.0), 0.0, 1.0)
        divergence_score = float(profile.get("divergence_score", 0.5) or 0.5)
        blocked = bool(profile.get("blocked", False)) and self.divergence_block_on_status

        confidence_multiplier = 1.0
        source_quality_multiplier = 1.0
        if status == "caution":
            confidence_multiplier = self._clamp(0.75 + (multiplier * 0.25), 0.55, 0.95)
            source_quality_multiplier = self._clamp(0.65 + (multiplier * 0.35), 0.45, 0.95)
        elif status == "blocked":
            confidence_multiplier = self._clamp(0.40 + (multiplier * 0.20), 0.20, 0.60)
            source_quality_multiplier = self._clamp(0.35 + (multiplier * 0.20), 0.15, 0.55)

        metadata["divergence_status"] = status
        metadata["divergence_multiplier"] = round(multiplier, 4)
        metadata["divergence_reasons"] = list(profile.get("reasons", []) or [])
        metadata["divergence_global_status"] = str(
            (profile.get("global", {}) or {}).get("status", "unknown") or "unknown"
        )
        metadata["divergence_source_status"] = str(
            (profile.get("source_profile", {}) or {}).get("status", "unknown") or "unknown"
        )
        metadata["divergence_blocked"] = blocked
        metadata["divergence_score"] = round(divergence_score, 4)

        return {
            "metadata": metadata,
            "confidence": self._clamp(confidence * confidence_multiplier, 0.0, 1.0),
            "source_quality": self._clamp(source_quality * source_quality_multiplier, 0.0, 1.0),
            "divergence_score": divergence_score,
            "profile": profile,
            "blocked": blocked,
        }

    def _lookup_adaptive_profile(self, strategy: Dict) -> Dict:
        if not self.adaptive_learning_enabled or not self.adaptive_learning:
            return {}
        metadata = self._get_metadata(strategy)
        source_key = str(
            strategy.get("source_key", metadata.get("source_key", "")) or ""
        ).strip()
        source = str(
            strategy.get("source", metadata.get("source", "strategy")) or "strategy"
        ).strip().lower()
        try:
            return self.adaptive_learning.get_source_profile(source_key=source_key, source=source) or {}
        except Exception as exc:
            logger.debug("adaptive learning lookup error: %s", exc)
            return {}

    def _lookup_calibration_profile(self, strategy: Dict) -> Dict:
        if not self.calibration_enabled or not self.calibration:
            return {}

        metadata = self._get_metadata(strategy)
        source_key = str(
            strategy.get("source_key", metadata.get("source_key", ""))
            or ""
        ).strip()
        source = str(
            strategy.get("source", metadata.get("source", "strategy"))
            or "strategy"
        ).strip().lower()
        cache_key = (source_key, source)
        if cache_key in self._calibration_profile_cache:
            return self._calibration_profile_cache[cache_key]

        if self._calibration_stats_snapshot is None:
            try:
                self._calibration_stats_snapshot = self.calibration.get_all_stats() or {}
            except Exception as exc:
                logger.debug("calibration stats lookup error: %s", exc)
                self._calibration_stats_snapshot = {}

        stats_map = self._calibration_stats_snapshot or {}
        profile: Dict = {}
        for lookup_key in [source_key, source, "global"]:
            if lookup_key and lookup_key in stats_map:
                profile = dict(stats_map[lookup_key])
                profile["lookup_key"] = lookup_key
                profile["used_global"] = lookup_key == "global"
                break

        if not profile:
            for lookup_key in [source_key, source, "global"]:
                if not lookup_key:
                    continue
                try:
                    ece = self.calibration.get_ece(lookup_key)
                except Exception as exc:
                    logger.debug("calibration ece lookup error: %s", exc)
                    ece = None
                if ece is None:
                    continue
                profile = {
                    "lookup_key": lookup_key,
                    "used_global": lookup_key == "global",
                    "ece": ece,
                    "total_records": 0,
                    "calibration_quality": "unknown",
                }
                break

        self._calibration_profile_cache[cache_key] = profile
        return profile

    def _apply_calibration_feedback(
        self,
        strategy: Dict,
        confidence: float,
        source_quality: float,
    ) -> Dict:
        metadata = dict(self._get_metadata(strategy))
        if not self.calibration_enabled or not self.calibration:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "calibration_score": 0.5,
                "profile": {},
                "blocked": False,
            }

        profile = self._lookup_calibration_profile(strategy)
        if not profile:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "calibration_score": 0.5,
                "profile": {},
                "blocked": False,
            }

        lookup_key = str(profile.get("lookup_key", "global") or "global")
        try:
            adjusted_confidence = self._clamp(
                float(self.calibration.get_adjustment_factor(lookup_key, confidence) or confidence),
                0.0,
                1.0,
            )
        except Exception as exc:
            logger.debug("calibration adjustment error: %s", exc)
            adjusted_confidence = confidence

        total_records = int(profile.get("total_records", 0) or 0)
        ece_value = profile.get("ece")
        ece = float(ece_value) if ece_value is not None else None
        calibration_score = 0.5
        blocked = False
        quality_multiplier = 1.0

        if ece is not None and total_records >= self.calibration_min_records:
            if ece <= self.calibration_target_ece:
                calibration_score = 1.0
            elif ece >= self.calibration_max_ece:
                calibration_score = 0.0
            else:
                spread = max(self.calibration_max_ece - self.calibration_target_ece, 1e-8)
                calibration_score = self._clamp(
                    1.0 - ((ece - self.calibration_target_ece) / spread),
                    0.0,
                    1.0,
                )
            blocked = ece >= self.calibration_max_ece
            quality_multiplier = self._clamp(0.70 + (calibration_score * 0.40), 0.55, 1.10)

        enriched_profile = dict(profile)
        enriched_profile["blocked"] = blocked
        enriched_profile["adjusted_confidence"] = round(adjusted_confidence, 4)
        enriched_profile["calibration_score"] = round(calibration_score, 4)

        metadata["raw_confidence"] = round(confidence, 4)
        metadata["calibrated_confidence"] = round(adjusted_confidence, 4)
        metadata["calibration_lookup_key"] = lookup_key
        metadata["calibration_total_records"] = total_records
        metadata["calibration_ece"] = round(ece, 4) if ece is not None else None
        metadata["calibration_score"] = round(calibration_score, 4)
        metadata["calibration_blocked"] = blocked
        metadata["calibration_quality"] = enriched_profile.get("calibration_quality", "unknown")
        metadata["calibration_used_global"] = bool(enriched_profile.get("used_global", False))

        return {
            "metadata": metadata,
            "confidence": adjusted_confidence,
            "source_quality": self._clamp(source_quality * quality_multiplier, 0.0, 1.0),
            "calibration_score": calibration_score,
            "profile": enriched_profile,
            "blocked": blocked,
        }

    def _extract_trade_memory_features(self, strategy: Dict) -> Dict:
        metadata = self._get_metadata(strategy)
        features: Dict = {}
        for payload in (strategy.get("features"), metadata.get("features")):
            if isinstance(payload, dict):
                features.update(payload)

        for key, value in metadata.items():
            if key not in features and isinstance(value, (int, float)):
                features[key] = value

        raw_score = strategy.get("current_score")
        if "overall_score" not in features and isinstance(raw_score, (int, float)):
            features["overall_score"] = float(raw_score)
        return features

    def _lookup_trade_memory_profile(
        self,
        strategy: Dict,
        *,
        target_coin: str,
        direction: str,
        strategy_type: str,
    ) -> Dict:
        if not self.memory_enabled or not self.trade_memory:
            return {}

        features = self._extract_trade_memory_features(strategy)
        if not features:
            return {}

        try:
            feature_signature = json.dumps(features, sort_keys=True, default=str)
        except (TypeError, ValueError):
            feature_signature = str(features)

        cache_key = (
            str(target_coin or "").strip().upper(),
            str(direction or "").strip().lower(),
            str(strategy_type or "").strip().lower(),
            round(float(self.memory_min_similarity or 0.0), 4),
            int(self.memory_top_k),
            feature_signature,
        )
        if cache_key in self._memory_profile_cache:
            return self._memory_profile_cache[cache_key]

        try:
            result = self.trade_memory.find_similar(
                features=features,
                coin=str(target_coin or "").strip().upper() or None,
                strategy_type=str(strategy_type or "").strip() or None,
                side=str(direction or "").strip().lower() or None,
                top_k=self.memory_top_k,
                min_similarity=self.memory_min_similarity,
            )
        except Exception as exc:
            logger.debug("trade memory lookup error: %s", exc)
            self._memory_profile_cache[cache_key] = {}
            return {}

        def _extract_value(name: str, default):
            if isinstance(result, dict):
                return result.get(name, default)
            return getattr(result, name, default)

        similarity_scores = list(_extract_value("similarity_scores", []) or [])
        avg_similarity = (
            sum(float(score or 0.0) for score in similarity_scores) / len(similarity_scores)
            if similarity_scores
            else 0.0
        )
        profile = {
            "features": features,
            "similar_trades": list(_extract_value("similar_trades", []) or []),
            "total_found": int(_extract_value("total_found", 0) or 0),
            "win_rate": float(_extract_value("win_rate", 0.0) or 0.0),
            "avg_pnl": float(_extract_value("avg_pnl", 0.0) or 0.0),
            "avg_return": float(_extract_value("avg_return", 0.0) or 0.0),
            "recommendation": str(_extract_value("recommendation", "proceed") or "proceed").strip().lower(),
            "reason": str(_extract_value("reason", "") or ""),
            "similarity_scores": similarity_scores,
            "avg_similarity": round(avg_similarity, 4),
        }
        self._memory_profile_cache[cache_key] = profile
        return profile

    def _apply_trade_memory_feedback(
        self,
        strategy: Dict,
        confidence: float,
        source_quality: float,
        *,
        target_coin: str,
        direction: str,
        strategy_type: str,
    ) -> Dict:
        metadata = dict(self._get_metadata(strategy))
        if not self.memory_enabled or not self.trade_memory:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "memory_score": 0.5,
                "profile": {},
                "blocked": False,
            }

        profile = self._lookup_trade_memory_profile(
            strategy,
            target_coin=target_coin,
            direction=direction,
            strategy_type=strategy_type,
        )
        if not profile:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "memory_score": 0.5,
                "profile": {},
                "blocked": False,
            }

        total_found = int(profile.get("total_found", 0) or 0)
        win_rate = self._clamp(float(profile.get("win_rate", 0.0) or 0.0), 0.0, 1.0)
        avg_return = float(profile.get("avg_return", 0.0) or 0.0)
        avg_pnl = float(profile.get("avg_pnl", 0.0) or 0.0)
        avg_similarity = self._clamp(float(profile.get("avg_similarity", 0.0) or 0.0), 0.0, 1.0)
        recommendation = str(profile.get("recommendation", "proceed") or "proceed").strip().lower()
        blocked = False
        memory_score = 0.5
        quality_multiplier = 1.0
        confidence_multiplier = 1.0

        if total_found >= self.memory_min_trades:
            return_component = self._clamp(
                0.5 + max(-0.4, min(0.4, avg_return / max(self.context_performance_return_scale, 1e-8))),
                0.0,
                1.0,
            )
            recommendation_floor = {
                "proceed": 0.72,
                "caution": 0.42,
                "avoid": 0.10,
            }.get(recommendation, 0.5)
            memory_score = self._clamp(
                (recommendation_floor * 0.45)
                + (win_rate * 0.25)
                + (return_component * 0.20)
                + (avg_similarity * 0.10),
                0.0,
                1.0,
            )
            quality_multiplier = self._clamp(0.65 + (memory_score * 0.55), 0.40, 1.16)
            confidence_multiplier = self._clamp(0.72 + (memory_score * 0.38), 0.50, 1.12)
            if recommendation == "proceed" and avg_return > 0 and win_rate >= 0.55:
                quality_multiplier = self._clamp(quality_multiplier + 0.04, 0.40, 1.18)
                confidence_multiplier = self._clamp(confidence_multiplier + 0.02, 0.50, 1.14)
            elif recommendation == "avoid":
                quality_multiplier = self._clamp(quality_multiplier - 0.08, 0.35, 1.16)
                confidence_multiplier = self._clamp(confidence_multiplier - 0.05, 0.45, 1.12)
                blocked = self.memory_block_on_avoid

        enriched_profile = dict(profile)
        enriched_profile["memory_score"] = round(memory_score, 4)
        enriched_profile["blocked"] = blocked

        metadata["memory_total_found"] = total_found
        metadata["memory_recommendation"] = recommendation
        metadata["memory_reason"] = profile.get("reason", "")
        metadata["memory_avg_similarity"] = round(avg_similarity, 4)
        metadata["memory_win_rate"] = round(win_rate, 4)
        metadata["memory_avg_return"] = round(avg_return, 4)
        metadata["memory_avg_pnl"] = round(avg_pnl, 4)
        metadata["memory_score"] = round(memory_score, 4)
        metadata["memory_blocked"] = blocked

        return {
            "metadata": metadata,
            "confidence": self._clamp(confidence * confidence_multiplier, 0.0, 1.0),
            "source_quality": self._clamp(source_quality * quality_multiplier, 0.0, 1.0),
            "memory_score": memory_score,
            "profile": enriched_profile,
            "blocked": blocked,
        }

    def _lookup_context_performance(
        self,
        strategy: Dict,
        *,
        target_coin: str,
        direction: str,
        regime_name: str,
    ) -> Dict:
        if not self.context_performance_enabled:
            return {}

        metadata = self._get_metadata(strategy)
        source_key = str(strategy.get("source_key", metadata.get("source_key", "")) or "").strip()
        source = str(strategy.get("source", metadata.get("source", "strategy")) or "strategy").strip().lower()
        coin = str(target_coin or "").strip().upper()
        side = str(direction or "").strip().lower()
        regime = str(regime_name or "unknown").strip().lower() or "unknown"

        cache_key = (source_key, source, coin, side, regime)
        if cache_key in self._context_profile_cache:
            return self._context_profile_cache[cache_key]

        query_candidates = []
        if source_key:
            query_candidates.extend(
                [
                    {"source_key": source_key, "coin": coin, "side": side, "regime": regime},
                    {"source_key": source_key, "coin": coin, "side": side, "regime": ""},
                ]
            )
        if source:
            query_candidates.extend(
                [
                    {"source": source, "coin": coin, "side": side, "regime": regime},
                    {"source": source, "coin": coin, "side": side, "regime": ""},
                ]
            )

        profile: Dict = {}
        seen_queries = set()
        for query in query_candidates:
            query_signature = tuple(sorted(query.items()))
            if query_signature in seen_queries:
                continue
            seen_queries.add(query_signature)

            rows = db.get_context_trade_outcome_summary(
                lookback_hours=self.context_performance_lookback_hours,
                min_closed_trades=(
                    self.context_performance_min_trades if query.get("regime") else 0
                ),
                **query,
            )
            if rows:
                candidate = (
                    dict(rows[0])
                    if query.get("regime")
                    else self._merge_context_profiles(rows)
                )
                if int(candidate.get("closed_trades", 0) or 0) >= self.context_performance_min_trades:
                    profile = candidate
                    break

        self._context_profile_cache[cache_key] = profile
        return profile

    @staticmethod
    def _merge_context_profiles(rows: List[Dict]) -> Dict:
        if not rows:
            return {}
        if len(rows) == 1:
            return dict(rows[0])

        first = dict(rows[0])
        merged = {
            "context_key": "|".join(
                [
                    str(first.get("source_key", first.get("source", "unknown")) or "unknown"),
                    str(first.get("coin", "UNKNOWN") or "UNKNOWN"),
                    str(first.get("side", "unknown") or "unknown"),
                    "mixed",
                ]
            ),
            "source_key": first.get("source_key", first.get("source", "unknown")),
            "source": first.get("source", "unknown"),
            "coin": first.get("coin", "UNKNOWN"),
            "side": first.get("side", "unknown"),
            "regime": "mixed",
            "open_trades": 0,
            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "realized_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_return_pct": 0.0,
            "last_closed_at": None,
        }

        return_total = 0.0
        for row in rows:
            row_dict = dict(row)
            merged["open_trades"] += int(row_dict.get("open_trades", 0) or 0)
            merged["closed_trades"] += int(row_dict.get("closed_trades", 0) or 0)
            merged["winning_trades"] += int(row_dict.get("winning_trades", 0) or 0)
            merged["losing_trades"] += int(row_dict.get("losing_trades", 0) or 0)
            merged["realized_pnl"] += float(row_dict.get("realized_pnl", 0.0) or 0.0)
            merged["gross_profit"] += float(row_dict.get("gross_profit", 0.0) or 0.0)
            merged["gross_loss"] += float(row_dict.get("gross_loss", 0.0) or 0.0)
            row_closed = int(row_dict.get("closed_trades", 0) or 0)
            return_total += float(row_dict.get("avg_return_pct", 0.0) or 0.0) * row_closed
            last_closed_at = row_dict.get("last_closed_at")
            if last_closed_at and (
                not merged["last_closed_at"] or str(last_closed_at) > str(merged["last_closed_at"])
            ):
                merged["last_closed_at"] = str(last_closed_at)

        closed_trades = int(merged["closed_trades"] or 0)
        merged["win_rate"] = round(
            merged["winning_trades"] / closed_trades,
            4,
        ) if closed_trades else 0.0
        merged["profit_factor"] = round(
            merged["gross_profit"] / max(merged["gross_loss"], 1e-8),
            4,
        ) if merged["gross_profit"] > 0 and merged["gross_loss"] > 0 else (
            999.0 if merged["gross_profit"] > 0 and merged["gross_loss"] == 0 else 0.0
        )
        merged["avg_return_pct"] = round(
            return_total / closed_trades,
            4,
        ) if closed_trades else 0.0
        merged["realized_pnl"] = round(float(merged["realized_pnl"] or 0.0), 2)
        merged["gross_profit"] = round(float(merged["gross_profit"] or 0.0), 2)
        merged["gross_loss"] = round(float(merged["gross_loss"] or 0.0), 2)
        return merged

    def _apply_context_performance_feedback(
        self,
        strategy: Dict,
        confidence: float,
        source_quality: float,
        *,
        target_coin: str,
        direction: str,
        regime_name: str,
    ) -> Dict:
        metadata = dict(self._get_metadata(strategy))
        profile = self._lookup_context_performance(
            strategy,
            target_coin=target_coin,
            direction=direction,
            regime_name=regime_name,
        )
        if not profile:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "context_score": 0.5,
                "profile": {},
                "blocked": False,
            }

        closed_trades = int(profile.get("closed_trades", 0) or 0)
        win_rate = float(profile.get("win_rate", 0.0) or 0.0)
        avg_return_pct = float(profile.get("avg_return_pct", 0.0) or 0.0)
        realized_pnl = float(profile.get("realized_pnl", 0.0) or 0.0)
        profit_factor = float(profile.get("profit_factor", 0.0) or 0.0)

        return_component = self._clamp(
            0.5 + max(-0.4, min(0.4, avg_return_pct / max(self.context_performance_return_scale, 1e-8))),
            0.0,
            1.0,
        )
        context_score = self._clamp(
            (win_rate * 0.55)
            + (return_component * 0.35)
            + (0.10 if profit_factor >= 1.2 else 0.0),
            0.0,
            1.0,
        )

        blocked = (
            closed_trades >= self.context_performance_min_trades
            and win_rate <= self.context_performance_block_win_rate
            and avg_return_pct <= self.context_performance_block_avg_return_pct
        )

        quality_multiplier = self._clamp(0.55 + (context_score * 0.70), 0.35, 1.15)
        confidence_multiplier = self._clamp(0.70 + (context_score * 0.45), 0.45, 1.12)
        if win_rate >= self.context_performance_boost_win_rate and avg_return_pct > 0:
            quality_multiplier = self._clamp(quality_multiplier + 0.05, 0.35, 1.18)
            confidence_multiplier = self._clamp(confidence_multiplier + 0.03, 0.45, 1.15)

        enriched_profile = dict(profile)
        enriched_profile["context_score"] = round(context_score, 4)
        enriched_profile["blocked"] = blocked

        metadata["context_closed_trades"] = closed_trades
        metadata["context_win_rate"] = round(win_rate, 4)
        metadata["context_avg_return_pct"] = round(avg_return_pct, 4)
        metadata["context_realized_pnl"] = round(realized_pnl, 2)
        metadata["context_profit_factor"] = round(profit_factor, 4)
        metadata["context_score"] = round(context_score, 4)
        metadata["context_blocked"] = blocked
        metadata["context_regime"] = enriched_profile.get("regime", regime_name)

        return {
            "metadata": metadata,
            "confidence": self._clamp(confidence * confidence_multiplier, 0.0, 1.0),
            "source_quality": self._clamp(source_quality * quality_multiplier, 0.0, 1.0),
            "context_score": context_score,
            "profile": enriched_profile,
            "blocked": blocked,
        }

    def _apply_execution_policy_feedback(
        self,
        strategy: Dict,
        confidence: float,
        source_quality: float,
    ) -> Dict:
        metadata = dict(self._get_metadata(strategy))
        if not self.execution_policy_enabled or not self.execution_policy:
            return {
                "metadata": metadata,
                "recommendation": {},
            }

        try:
            recommendation = self.execution_policy.recommend(
                strategy=strategy,
                metadata=metadata,
                confidence=confidence,
                source_quality=source_quality,
            ) or {}
        except Exception as exc:
            logger.debug("execution policy lookup error: %s", exc)
            recommendation = {}

        if recommendation:
            execution_route = recommendation.get(
                "execution_route",
                recommendation.get("route", "market"),
            )
            recommendation["execution_route"] = execution_route
            recommendation.setdefault("route", execution_route)
            for key, value in recommendation.items():
                if value is not None:
                    metadata[key] = value

        return {
            "metadata": metadata,
            "recommendation": recommendation,
        }

    def _apply_adaptive_learning_feedback(
        self,
        strategy: Dict,
        confidence: float,
        source_quality: float,
    ) -> Dict:
        metadata = dict(self._get_metadata(strategy))
        profile = self._lookup_adaptive_profile(strategy)
        if not profile:
            return {
                "metadata": metadata,
                "confidence": confidence,
                "source_quality": source_quality,
                "profile": {},
                "weight_multiplier": 1.0,
                "confidence_multiplier": 1.0,
            }

        weight_multiplier = float(profile.get("weight_multiplier", 1.0) or 1.0)
        confidence_multiplier = float(profile.get("confidence_multiplier", 1.0) or 1.0)
        metadata["adaptive_status"] = profile.get("status", "active")
        metadata["adaptive_health_score"] = round(
            float(profile.get("health_score", 0.0) or 0.0),
            4,
        )
        metadata["adaptive_drift_score"] = round(
            float(profile.get("drift_score", 0.0) or 0.0),
            4,
        )
        metadata["adaptive_training_label"] = profile.get("training_label", "monitor")
        metadata["adaptive_recommended_action"] = profile.get("recommended_action", "monitor")

        return {
            "metadata": metadata,
            "confidence": self._clamp(confidence * confidence_multiplier, 0.0, 1.0),
            "source_quality": self._clamp(source_quality * weight_multiplier, 0.0, 1.0),
            "profile": profile,
            "weight_multiplier": weight_multiplier,
            "confidence_multiplier": confidence_multiplier,
        }

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
            confluence_score = float((s.get("_source_confluence", {}) or {}).get("confluence_score", 0.0) or 0.0)
            marker = " ← EXEC" if s in executions else ""
            logger.info(
                "  #%d %s %s composite=%.4f ev=%+.2f%% cost=%.2f%% cx=%.2f%s",
                i + 1,
                side.upper(),
                coin,
                composite,
                expected_value_pct,
                execution_cost_pct,
                confluence_score,
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
            "research_persistence_enabled": self.persist_research,
            "execution_quality_enabled": self.execution_quality_enabled,
            "execution_quality_lookback_hours": self.execution_quality_lookback_hours,
            "execution_quality_min_events": self.execution_quality_min_events,
            "confluence_enabled": self.confluence_enabled,
            "confluence_weight": self.w_confluence,
            "confluence_baseline": self.confluence_baseline,
            "confluence_conflict_block_threshold": self.confluence_conflict_block_threshold,
            "confluence_conflict_floor": self.confluence_conflict_floor,
            "context_performance_enabled": self.context_performance_enabled,
            "context_weight": self.w_context,
            "context_performance_lookback_hours": self.context_performance_lookback_hours,
            "context_performance_min_trades": self.context_performance_min_trades,
            "context_performance_return_scale": self.context_performance_return_scale,
            "context_performance_block_win_rate": self.context_performance_block_win_rate,
            "context_performance_block_avg_return_pct": self.context_performance_block_avg_return_pct,
            "context_performance_boost_win_rate": self.context_performance_boost_win_rate,
            "calibration_enabled": self.calibration_enabled,
            "calibration_weight": self.w_calibration,
            "calibration_min_records": self.calibration_min_records,
            "calibration_target_ece": self.calibration_target_ece,
            "calibration_max_ece": self.calibration_max_ece,
            "divergence_enabled": self.divergence_enabled,
            "divergence_weight": self.w_divergence,
            "divergence_block_on_status": self.divergence_block_on_status,
            "memory_enabled": self.memory_enabled,
            "memory_weight": self.w_memory,
            "memory_min_trades": self.memory_min_trades,
            "memory_min_similarity": self.memory_min_similarity,
            "memory_top_k": self.memory_top_k,
            "memory_block_on_avoid": self.memory_block_on_avoid,
            "execution_policy_enabled": self.execution_policy_enabled,
            "execution_policy": (
                self.execution_policy.get_stats()
                if self.execution_policy_enabled and self.execution_policy
                else {}
            ),
            "adaptive_learning_enabled": self.adaptive_learning_enabled,
            "adaptive_learning_block_on_status": self.adaptive_learning_block_on_status,
            "adaptive_learning_min_health_score": self.adaptive_learning_min_health_score,
            "last_research_cycle_id": self._last_research_cycle_id,
            "recent_decisions": list(self._decision_history)[-10:],
        }

    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        """Return recent decision history for dashboard."""
        return list(self._decision_history)[-limit:]
