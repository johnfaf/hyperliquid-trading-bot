"""
Signal Processor — Deduplication, Conflict Resolution, Aggregation, Culling
=============================================================================
Built from production log analysis showing:
  - 914 strategies generated per cycle (7447 accumulated)
  - momentum_long 0.90 AND momentum_short 0.90 simultaneously
  - concentrated_bet appearing with mediocre 0.46 confidence
  - 14 signals after regime filter still overwhelming firewall

This module sits BETWEEN strategy scoring and paper trading execution:
  Strategy Scorer → Signal Processor → Paper Trader Pipeline

Four layers:
  1. Strategy Culling — kill garbage strategies early
  2. Deduplication — collapse same coin+side+type into one signal
  3. Conflict Resolution — handle opposing signals on same coin
  4. Decision Compression — pick top N candidates from survivors
"""
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Pre-processes strategy signals before they reach the execution pipeline.
    Eliminates noise, resolves conflicts, and compresses decisions.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # ─── Culling thresholds ─────────────────────────
        # Lowered from 0.50 → 0.30: early-stage strategies with thin sample
        # sizes get heavily penalized by the scorer (5 trades = 0.35x), so
        # a 0.50 bar culls nearly everything before the firewall even sees it.
        self.min_score_threshold = cfg.get("min_score_threshold", 0.30)
        self.min_trades_for_cull = cfg.get("min_trades_for_cull", 10)
        self.concentrated_bet_min_confidence = cfg.get("concentrated_bet_min_confidence", 0.85)

        # ─── Dedup settings ─────────────────────────────
        # When multiple strategies say "long BTC momentum", merge them
        self.dedup_enabled = cfg.get("dedup_enabled", True)

        # ─── Conflict resolution ────────────────────────
        self.conflict_resolution = cfg.get("conflict_resolution", "regime_aligned")
        # Options: "regime_aligned" (pick side matching regime), "higher_confidence",
        #          "block_both" (if opposing, skip the coin entirely)

        # ─── Compression ────────────────────────────────
        self.max_signals_out = cfg.get("max_signals_out", 8)

        # Stats
        self.stats = {
            "total_in": 0,
            "culled": 0,
            "deduped": 0,
            "conflicts_resolved": 0,
            "compressed": 0,
            "total_out": 0,
        }

    def process(self, strategies: List[Dict],
                regime_data: Optional[Dict] = None) -> List[Dict]:
        """
        Full processing pipeline:
          strategies → cull → dedup → conflict resolve → compress → output

        Args:
            strategies: List of strategy dicts (from scorer.get_top_strategies)
            regime_data: Current market regime for conflict resolution

        Returns:
            Cleaned, deduplicated, conflict-resolved, compressed list of strategies
        """
        self.stats["total_in"] += len(strategies)

        if not strategies:
            return []

        # Step 1: Cull garbage (regime-aware)
        survivors = self._cull(strategies, regime_data=regime_data)
        culled_count = len(strategies) - len(survivors)
        self.stats["culled"] += culled_count
        if culled_count > 0:
            logger.info(f"SignalProcessor: culled {culled_count} low-quality strategies "
                       f"({len(survivors)} remaining)")

        # Step 2: Deduplicate
        dedup_count = 0
        if self.dedup_enabled:
            before_dedup = len(survivors)
            survivors = self._deduplicate(survivors)
            dedup_count = before_dedup - len(survivors)
            self.stats["deduped"] += dedup_count
            if dedup_count > 0:
                logger.info(f"SignalProcessor: merged {dedup_count} duplicate strategies "
                           f"({len(survivors)} remaining)")

        # Step 3: Resolve conflicts (opposing signals on same coin)
        before_conflict = len(survivors)
        survivors = self._resolve_conflicts(survivors, regime_data)
        # Conflict count should never be negative — it's the count of signals DROPPED
        # due to conflicts. If the count is negative or zero, log separately.
        conflict_count = max(0, before_conflict - len(survivors))
        self.stats["conflicts_resolved"] += conflict_count
        if conflict_count > 0:
            logger.info(f"SignalProcessor: resolved {conflict_count} conflicting signals "
                       f"({len(survivors)} remaining)")
        elif len(survivors) > before_conflict:
            # Safety check: if survivors increased, something went wrong
            logger.warning(f"SignalProcessor: conflict resolution increased strategies "
                          f"({before_conflict} → {len(survivors)}), possible dedup/copy issue")

        # Step 4: Compress — keep only top N
        before_compress = len(survivors)
        survivors = self._compress(survivors)
        compressed_count = before_compress - len(survivors)
        self.stats["compressed"] += compressed_count
        if compressed_count > 0:
            logger.info(f"SignalProcessor: compressed {compressed_count} excess signals "
                       f"({len(survivors)} remaining)")

        self.stats["total_out"] += len(survivors)

        logger.info(f"SignalProcessor: {len(strategies)} in → {len(survivors)} out "
                   f"(culled={culled_count}, deduped={dedup_count}, "
                   f"conflicts={conflict_count}, compressed={compressed_count})")

        return survivors

    # ─── Step 1: Culling ────────────────────────────────────────

    def _cull(self, strategies: List[Dict],
              regime_data: Optional[Dict] = None) -> List[Dict]:
        """
        Remove strategies that shouldn't exist:
          - Score below threshold (after enough trades to judge)
          - concentrated_bet with sub-0.85 confidence
          - Unknown/empty strategy types

        Regime-aware: in ranging markets, lower the cull bar for
        mean_reversion and delta_neutral strategies (they thrive in chop).
        """
        survivors = []
        regime = (regime_data or {}).get("overall_regime", "")
        is_ranging = "rang" in regime.lower() if regime else False

        # Ranging-friendly strategy types get a lower cull threshold
        _RANGING_FAVORED = {"mean_reversion", "delta_neutral", "grid", "market_making",
                            "mean_reversion_short", "mean_reversion_long"}

        for s in strategies:
            strategy_type = s.get("strategy_type", s.get("type", ""))
            score = s.get("current_score", 0)
            trade_count = s.get("trade_count", 0)

            # Dynamic threshold: softer for ranging-favored strategies in ranging regime
            threshold = self.min_score_threshold
            if is_ranging and strategy_type in _RANGING_FAVORED:
                threshold *= 0.6  # 0.50 → 0.30 effectively

            # Kill strategies with enough history but poor performance
            if trade_count >= self.min_trades_for_cull and score < threshold:
                logger.debug(f"Culled {strategy_type} (score={score:.2f}, "
                           f"trades={trade_count}) — below threshold {threshold:.2f}")
                continue

            # concentrated_bet requires high conviction by definition
            if strategy_type == "concentrated_bet" and score < self.concentrated_bet_min_confidence:
                logger.debug(f"Culled concentrated_bet (score={score:.2f} "
                           f"< {self.concentrated_bet_min_confidence})")
                continue

            # Skip empty/null strategy types
            if not strategy_type:
                continue

            survivors.append(s)

        return survivors

    # ─── Step 2: Deduplication ──────────────────────────────────

    def _deduplicate(self, strategies: List[Dict]) -> List[Dict]:
        """
        Merge strategies that are functionally identical:
          - Same coin + same implied direction = merge (regardless of strategy_type)
          - Keep the one with highest score, but boost confidence by
            number of agreeing sources (consensus signal)

        CRITICAL FIX: Use canonical key (coin, direction) not (type, direction, coin).
        This ensures all longs on BTC merge into ONE signal, not 3 separate ones
        from momentum_long, trend_following, and breakout all trying to trade BTC long.
        """
        import json

        groups: Dict[str, List[Dict]] = defaultdict(list)

        for s in strategies:
            strategy_type = s.get("strategy_type", s.get("type", ""))

            # Determine implied direction from strategy type
            direction = self._infer_direction(strategy_type, s)

            # Get coins — try to extract from parameters
            params = s.get("parameters", "{}")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}

            coins = params.get("coins", params.get("coins_traded", []))
            if isinstance(coins, str):
                coins = [coins]
            coin_key = coins[0] if coins else "any"

            # CANONICAL KEY: (coin, direction) — ignore strategy_type
            # This merges all longs on BTC into one, regardless of which strategy detected it
            key = f"{coin_key}:{direction}"
            groups[key].append(s)

        # Merge each group
        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Pick the best one, boost its score by consensus factor
                group.sort(key=lambda x: x.get("current_score", 0), reverse=True)
                best = group[0].copy()

                # Consensus boost: more agreeing strategies = higher confidence
                # But capped — 5 duplicates saying the same thing isn't 5x better
                consensus_factor = min(1.0 + 0.05 * (len(group) - 1), 1.20)
                original_score = best.get("current_score", 0)
                best["current_score"] = min(original_score * consensus_factor, 1.0)
                best["_dedup_count"] = len(group)
                best["_dedup_boost"] = consensus_factor

                logger.debug(f"Merged {len(group)} strategies for {key} "
                           f"(score {original_score:.2f} → {best['current_score']:.2f})")
                merged.append(best)

        return merged

    def _infer_direction(self, strategy_type: str, strategy: Dict) -> str:
        """Infer long/short direction from strategy type."""
        import json

        long_types = {"momentum_long", "trend_following", "breakout", "swing_trading"}
        short_types = {"momentum_short", "contrarian"}
        neutral_types = {"funding_arb", "delta_neutral", "diversified_portfolio"}

        if strategy_type in long_types:
            return "long"
        elif strategy_type in short_types:
            return "short"
        elif strategy_type in neutral_types:
            return "neutral"
        else:
            # Try to get from parameters
            params = strategy.get("parameters", "{}")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            return params.get("direction", params.get("bias", "long"))

    # ─── Step 3: Conflict Resolution ────────────────────────────

    def _resolve_conflicts(self, strategies: List[Dict],
                           regime_data: Optional[Dict] = None) -> List[Dict]:
        """
        Handle cases where we have both LONG and SHORT signals on the same coin.

        Resolution strategies:
          - "regime_aligned": keep the side matching market regime
          - "higher_confidence": keep whichever has higher score
          - "block_both": if they conflict, skip the coin entirely
        """
        import json

        # Group by coin (using a primary coin key only). A strategy may carry
        # multiple coins in params, but appending the same dict to every coin
        # bucket can duplicate survivors during conflict resolution.
        coin_signals: Dict[str, List[Dict]] = defaultdict(list)
        no_coin_signals = []  # Strategies without clear coin assignment

        for s in strategies:
            params = s.get("parameters", "{}")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}

            coins = params.get("coins", params.get("coins_traded", []))
            if isinstance(coins, str):
                coins = [coins]

            primary_coin = coins[0] if coins else None
            if primary_coin:
                coin_signals[primary_coin].append(s)
            else:
                no_coin_signals.append(s)

        resolved = []

        for coin, sigs in coin_signals.items():
            # Check for directional conflict
            directions = set()
            for s in sigs:
                strategy_type = s.get("strategy_type", s.get("type", ""))
                d = self._infer_direction(strategy_type, s)
                directions.add(d)

            has_conflict = "long" in directions and "short" in directions

            if not has_conflict:
                resolved.extend(sigs)
                continue

            # We have a conflict — resolve it
            if self.conflict_resolution == "block_both":
                logger.debug(f"Conflict: blocking ALL signals for {coin} "
                            f"(long+short both present)")
                continue

            elif self.conflict_resolution == "regime_aligned":
                regime = ""
                if regime_data:
                    regime = regime_data.get("overall_regime", "").upper()

                # Determine which side the regime favors
                if regime in ("TRENDING_UP",):
                    favored = "long"
                elif regime in ("TRENDING_DOWN",):
                    favored = "short"
                elif regime in ("RANGING", "VOLATILE", "LOW_LIQUIDITY"):
                    # In ranging/volatile, fall back to higher confidence
                    favored = None
                else:
                    favored = None

                if favored:
                    kept = [s for s in sigs if self._infer_direction(
                        s.get("strategy_type", s.get("type", "")), s) == favored]
                    dropped = len(sigs) - len(kept)
                    if dropped > 0:
                        logger.debug(f"Conflict resolved for {coin}: keeping {favored} "
                                    f"(regime={regime}), dropped {dropped} opposing signals")
                    resolved.extend(kept)
                else:
                    # No clear regime — use higher confidence fallback
                    best = max(sigs, key=lambda x: x.get("current_score", 0))
                    best_dir = self._infer_direction(
                        best.get("strategy_type", best.get("type", "")), best)
                    kept = [s for s in sigs if self._infer_direction(
                        s.get("strategy_type", s.get("type", "")), s) == best_dir]
                    logger.debug(f"Conflict resolved for {coin}: keeping {best_dir} "
                                f"(highest confidence in ambiguous regime)")
                    resolved.extend(kept)

            elif self.conflict_resolution == "higher_confidence":
                best = max(sigs, key=lambda x: x.get("current_score", 0))
                best_dir = self._infer_direction(
                    best.get("strategy_type", best.get("type", "")), best)
                kept = [s for s in sigs if self._infer_direction(
                    s.get("strategy_type", s.get("type", "")), s) == best_dir]
                logger.debug(f"Conflict resolved for {coin}: keeping {best_dir} "
                            f"(higher confidence)")
                resolved.extend(kept)

        # Add back strategies without specific coins
        resolved.extend(no_coin_signals)

        # Final guard: preserve order while removing duplicate object references.
        # This catches accidental fan-out where the same strategy object appears
        # in multiple buckets.
        unique = []
        seen_refs = set()
        for s in resolved:
            ref = id(s)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            unique.append(s)
        return unique

    # ─── Step 4: Decision Compression ───────────────────────────

    def _compress(self, strategies: List[Dict]) -> List[Dict]:
        """
        Keep only the top N strategies by score.
        This prevents firewall saturation (5/5 slots full blocking everything).
        """
        if len(strategies) <= self.max_signals_out:
            return strategies

        # Sort by score, keep top N
        strategies.sort(key=lambda x: x.get("current_score", 0), reverse=True)
        return strategies[:self.max_signals_out]

    # ─── Stats ──────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return processing statistics."""
        total_in = self.stats["total_in"]
        total_out = self.stats["total_out"]
        return {
            **self.stats,
            "reduction_rate": 1 - (total_out / total_in) if total_in > 0 else 0,
        }


class ArenaIncubator:
    """
    Requires new agents/strategies to prove themselves in backtesting
    before receiving live capital allocation.

    Problem from logs: Arena has 9 agents, 0 champions, PnL=-$44.15
    after 8 cycles. Agents get live capital too quickly.

    Solution: Agents must complete an incubation period with simulated
    trades before being promoted to live trading.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.min_incubation_trades = cfg.get("min_incubation_trades", 5)
        self.min_win_rate = cfg.get("min_win_rate", 0.45)
        self.min_profit_factor = cfg.get("min_profit_factor", 1.2)

        # Track incubating strategies
        # key: strategy_key, value: {trades: [], wins: int, losses: int, total_pnl: float}
        self._incubating: Dict[str, Dict] = {}
        self._promoted: set = set()
        self._rejected: set = set()

        self.stats = {
            "total_incubating": 0,
            "total_promoted": 0,
            "total_rejected": 0,
        }

        logger.info(f"ArenaIncubator initialized (min_trades={self.min_incubation_trades}, "
                   f"min_WR={self.min_win_rate:.0%})")

    def should_allow_live(self, strategy_key: str) -> Tuple[bool, str]:
        """
        Check if a strategy has passed incubation and can trade live.

        Returns:
            (allowed: bool, reason: str)
        """
        # Already promoted — allow
        if strategy_key in self._promoted:
            return True, "promoted"

        # Already rejected — block
        if strategy_key in self._rejected:
            return False, "rejected during incubation"

        # Check if currently incubating
        if strategy_key in self._incubating:
            record = self._incubating[strategy_key]
            total = record["wins"] + record["losses"]

            if total < self.min_incubation_trades:
                return False, (f"incubating: {total}/{self.min_incubation_trades} trades "
                              f"(WR={record['wins']/max(total,1):.0%})")

            # Enough trades — evaluate
            win_rate = record["wins"] / total
            avg_win = record["avg_win"]
            avg_loss = abs(record["avg_loss"]) if record["avg_loss"] != 0 else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

            if win_rate >= self.min_win_rate and profit_factor >= self.min_profit_factor:
                self._promoted.add(strategy_key)
                self.stats["total_promoted"] += 1
                del self._incubating[strategy_key]
                logger.info(f"ArenaIncubator: PROMOTED {strategy_key} "
                           f"(WR={win_rate:.0%}, PF={profit_factor:.2f})")
                return True, "just promoted"
            else:
                self._rejected.add(strategy_key)
                self.stats["total_rejected"] += 1
                del self._incubating[strategy_key]
                logger.info(f"ArenaIncubator: REJECTED {strategy_key} "
                           f"(WR={win_rate:.0%}, PF={profit_factor:.2f})")
                return False, (f"failed incubation: WR={win_rate:.0%}, "
                              f"PF={profit_factor:.2f}")

        # New strategy — start incubation
        self._incubating[strategy_key] = {
            "wins": 0, "losses": 0, "total_pnl": 0,
            "avg_win": 0, "avg_loss": 0,
            "trades": [],
        }
        self.stats["total_incubating"] += 1
        return False, f"starting incubation (0/{self.min_incubation_trades})"

    def record_sim_trade(self, strategy_key: str, pnl: float, win: bool):
        """
        Record a simulated trade result for an incubating strategy.
        Called from Arena backtester or paper trading sim.
        """
        if strategy_key not in self._incubating:
            # Already promoted or rejected, or not tracked
            return

        record = self._incubating[strategy_key]
        if win:
            record["wins"] += 1
        else:
            record["losses"] += 1
        record["total_pnl"] += pnl
        record["trades"].append({"pnl": pnl, "win": win})

        # Update averages
        wins_list = [t["pnl"] for t in record["trades"] if t["win"]]
        losses_list = [t["pnl"] for t in record["trades"] if not t["win"]]
        record["avg_win"] = sum(wins_list) / len(wins_list) if wins_list else 0
        record["avg_loss"] = sum(losses_list) / len(losses_list) if losses_list else 0

    def get_incubation_status(self) -> Dict:
        """Get status of all incubating strategies."""
        status = {}
        for key, record in self._incubating.items():
            total = record["wins"] + record["losses"]
            status[key] = {
                "trades_completed": total,
                "trades_required": self.min_incubation_trades,
                "win_rate": record["wins"] / total if total > 0 else 0,
                "total_pnl": record["total_pnl"],
                "progress_pct": total / self.min_incubation_trades * 100,
            }
        return status

    def get_stats(self) -> Dict:
        """Return incubation statistics."""
        return {
            **self.stats,
            "currently_incubating": len(self._incubating),
            "promoted_list": list(self._promoted),
            "rejected_list": list(self._rejected),
        }
