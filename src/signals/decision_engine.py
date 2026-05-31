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

from src.core import clock_provider

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
        self.w_diversity = cfg.get("w_diversity", 0.05)   # Tie-breaker, not a quality substitute
        self.w_freshness = cfg.get("w_freshness", 0.10)   # Prefer strategies with recent activity
        self.w_consensus = cfg.get("w_consensus", 0.10)   # Dedup consensus boost

        # Minimum composite score to even consider executing
        # Lowered from 0.30 → 0.20: with thin trade history the composite
        # scores are suppressed by sample-size penalties. 0.20 lets
        # promising-but-young strategies through for paper validation.
        self.min_decision_score = cfg.get("min_decision_score", 0.20)

        # EV-first ranking (algo #3). Reads global config with a cfg-dict
        # override so tests/callers can set it directly. Default OFF.
        try:
            import config as _gcfg
        except Exception:
            _gcfg = None
        self.ev_first_enabled = bool(cfg.get(
            "ev_first_enabled", getattr(_gcfg, "DECISION_EV_FIRST_ENABLED", False)))
        self.ev_cost_r = float(cfg.get(
            "ev_cost_r", getattr(_gcfg, "DECISION_EV_COST_R", 0.12)))
        self.min_ev_r = float(cfg.get(
            "min_ev_r", getattr(_gcfg, "DECISION_MIN_EV_R", 0.0)))
        # Portfolio correlation / net-exposure cap (algo #7). Default OFF.
        self.corr_cap_enabled = bool(cfg.get(
            "corr_cap_enabled", getattr(_gcfg, "PORTFOLIO_NET_EXPOSURE_CAP_ENABLED", False)))
        self.max_same_side_positions = int(cfg.get(
            "max_same_side_positions", getattr(_gcfg, "PORTFOLIO_MAX_SAME_SIDE_POSITIONS", 3)))

        # Max trades to execute per cycle (independent of position slots)
        self.max_trades_per_cycle = cfg.get("max_trades_per_cycle", 3)
        self.max_prescreen_candidates = cfg.get("max_prescreen_candidates", 8)
        self.max_positions = cfg.get("max_positions", 8)

        # Track decisions for audit
        self._decision_history: deque = deque(maxlen=100)
        self._cycle_count = 0

        # Stats
        self.stats = {
            "total_decisions": 0,
            "total_executions": 0,
            "total_no_trade": 0,
            "total_candidates": 0,
            "total_missing_asset": 0,
            "total_missing_direction": 0,
        }

    @staticmethod
    def _normalise_params(params):
        if isinstance(params, str):
            import json
            try:
                return json.loads(params)
            except (json.JSONDecodeError, TypeError):
                return {}
        return params if isinstance(params, dict) else {}

    @staticmethod
    def _extract_coins(params: Dict) -> List[str]:
        coins = params.get("coins") or params.get("coins_traded") or params.get("coin") or []
        if isinstance(coins, str):
            coins = [coins]
        return [str(coin).upper() for coin in coins if str(coin or "").strip()]

    @staticmethod
    def _normalise_direction(value) -> str:
        direction = str(value or "").strip().lower()
        if direction in {"buy", "long"}:
            return "long"
        if direction in {"sell", "short"}:
            return "short"
        return "neutral"

    @classmethod
    def _regime_default_direction(cls, params: Dict, regime_data: Optional[Dict]) -> str:
        regime = str((regime_data or {}).get("overall_regime", "") or "").strip().lower()
        try:
            confidence = float((regime_data or {}).get("overall_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= 0.60:
            if regime in {"trending_down", "bearish", "crash"}:
                return "short"
            if regime in {"trending_up", "bullish"}:
                return "long"
        return cls._normalise_direction(params.get("direction") or params.get("bias"))

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
        available_slots = max(0, self.max_positions - len(open_positions))

        self.stats["total_candidates"] += len(strategies)

        if not strategies:
            self._log_decision([], regime_data, available_slots)
            self.stats["total_no_trade"] += 1
            return []

        # Extract and validate coin field. Strategies without a concrete asset
        # are shadow-only; assigning a random liquid coin makes the bot trade a
        # market the strategy never actually identified.
        valid_strategies = []
        for s in strategies:
            params = self._normalise_params(s.get("parameters", {}))
            # Persist parsed version so downstream doesn't re-parse
            s["parameters"] = params

            coins = self._extract_coins(params)

            # Fallback 1: extract from trader's current positions in metrics
            if not coins or (coins and coins[0].lower() == "unknown"):
                metrics = s.get("metrics", {})
                traded_coins = (
                    metrics.get("coins")
                    or metrics.get("coins_traded")
                    or metrics.get("coin")
                    or []
                )
                if isinstance(traded_coins, str):
                    traded_coins = [traded_coins]
                if traded_coins and isinstance(traded_coins, list):
                    coins = [str(coin).upper() for coin in traded_coins if str(coin or "").strip()]
                    params["coins"] = coins  # Persist so _compute_composite_score sees it

            if not coins or (coins and coins[0].lower() == "unknown"):
                strategy_type = s.get("strategy_type", s.get("type", ""))
                s["_decision_skip_reason"] = "missing_asset"
                self.stats["total_missing_asset"] += 1
                logger.info(
                    "Skipping non-executable strategy %s (%s): no asset in parameters/metrics",
                    s.get("name", s.get("id", "?")),
                    strategy_type,
                )
                continue

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
                "_ev_proxy": composite.get("ev_proxy", 0.0),
                "_score_breakdown": composite,
            })

        # ─── Rank: EV-first (algo #3) or heuristic composite ─────────
        if self.ev_first_enabled:
            # Net-of-cost EV is the primary key; the composite only breaks
            # ties. A candidate must clear BOTH the composite floor and the
            # minimum EV, so EV-first is strictly MORE selective than the
            # composite gate alone -- it never loosens, only re-prioritises
            # toward expected profit.
            scored.sort(key=lambda x: (x["_ev_proxy"], x["_composite_score"]), reverse=True)

            def _ev_ok(x):
                return (x["_composite_score"] >= self.min_decision_score
                        and x["_ev_proxy"] >= self.min_ev_r)
            qualified = [s for s in scored if _ev_ok(s)]
            disqualified = [s for s in scored if not _ev_ok(s)]
        else:
            scored.sort(key=lambda x: x["_composite_score"], reverse=True)
            qualified = [s for s in scored if s["_composite_score"] >= self.min_decision_score]
            disqualified = [s for s in scored if s["_composite_score"] < self.min_decision_score]

        # ─── Portfolio correlation / net-exposure cap (algo #7) ─────
        qualified = self._apply_correlation_cap(qualified, open_positions)

        # ─── Limit by cycle trade cap AND available slots ─
        max_this_cycle = min(self.max_prescreen_candidates, available_slots)
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
            "timestamp": clock_provider.utc_now().isoformat(),
            "candidates": len(strategies),
            "qualified": len(qualified),
            "prescreened": len(executions),
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
        params = self._normalise_params(strategy.get("parameters", "{}"))
        coins = self._extract_coins(params)
        target_coin = coins[0] if coins else "unknown"

        # Store for logging
        strategy["_decision_coin"] = target_coin

        # Infer direction — regime-aware for all non-explicitly-directional types
        # Only momentum_long/short are unconditionally directional.
        # breakout, trend_following, swing_trading follow the dominant trend direction
        # (downside breakout = short in trending_down; upside breakout = long in trending_up).
        long_types  = {"momentum_long"}
        short_types = {"momentum_short", "contrarian"}

        # ★ H14 FIX: contrarian-class strategies must NOT follow the trend.
        # Mean reversion buys dips in trending-down regimes; if we flip its
        # direction to match the trend, we're trading it opposite to its
        # documented edge. These types must use their params["direction"]
        # (set by strategy_identifier) even if the regime is strongly trending.
        contrarian_types = {"mean_reversion", "contrarian", "fade", "counter_trend"}

        # Derive regime direction bias. If neither the regime nor strategy
        # explicitly says long/short, stay neutral instead of hiding it as long.
        regime_default = self._regime_default_direction(params, regime_data)

        if strategy_type in long_types:
            direction = "long"
        elif strategy_type in short_types:
            direction = "short"
        elif strategy_type in contrarian_types:
            # ★ H14: use the strategy's own direction, never the regime trend.
            # Fall back to "long" only if direction missing (shouldn't happen
            # after H17 fix ensures strategy_identifier sets direction).
            direction = self._normalise_direction(params.get("direction") or params.get("bias"))
            if direction == "neutral":
                logger.warning(
                    "Contrarian strategy %s missing direction param -- "
                    "marking neutral/no-trade. Check strategy_identifier.",
                    strategy_type,
                )
        else:
            # breakout, trend_following, swing_trading, concentrated_bet,
            # scalping, funding_arb, delta_neutral, etc.
            # — follow the current regime when confident.  Stored direction
            # is a fallback only, not a permanent long/short bias from an
            # older market.
            direction = regime_default
        strategy["_decision_side"] = direction
        if direction not in {"long", "short"}:
            strategy["_decision_skip_reason"] = "missing_direction"
            self.stats["total_missing_direction"] += 1

        diversity_bonus = 1.0 if target_coin not in open_coins else 0.0

        # 4. Freshness — prefer recently active strategies
        freshness = 0.5  # Default middle value
        discovered = strategy.get("discovered_at", "")
        if discovered:
            try:
                disc_dt = datetime.fromisoformat(discovered.replace("Z", "+00:00"))
                age_hours = (clock_provider.utc_now() - disc_dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
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
        if direction not in {"long", "short"}:
            total = -1.0

        # EV proxy (algo #3): net-of-cost expected value in R units, driven by
        # the signal's (calibrated) confidence. EV_R = p*R_win - (1-p)*R_loss
        # - cost_R. R_win comes from the risk policy's reward_multiple when
        # present, else a conservative default. Used as the primary ranking
        # key only when ev_first_enabled.
        conf_raw = strategy.get("confidence", strategy.get("current_score", base))
        try:
            p_win = max(0.0, min(float(conf_raw), 1.0))
        except (TypeError, ValueError):
            p_win = float(base)
        r_win = 1.75
        risk_policy = params.get("risk_policy") if isinstance(params, dict) else None
        if isinstance(risk_policy, dict):
            try:
                r_win = float(risk_policy.get("reward_multiple", r_win) or r_win)
            except (TypeError, ValueError):
                pass
        r_win = max(0.1, min(r_win, 10.0))
        ev_proxy = p_win * r_win - (1.0 - p_win) * 1.0 - self.ev_cost_r
        if direction not in {"long", "short"}:
            ev_proxy = -999.0

        return {
            "total": round(total, 4),
            "ev_proxy": round(ev_proxy, 4),
            "base_score": round(base, 3),
            "regime_alignment": round(regime_bonus, 3),
            "diversity": round(diversity_bonus, 3),
            "freshness": round(freshness, 3),
            "consensus": round(consensus, 3),
        }

    def _apply_correlation_cap(self, qualified: List[Dict],
                               open_positions: Optional[List]) -> List[Dict]:
        """Cap concurrent same-direction positions (algo #7).

        On a highly-correlated crypto book, N same-side positions are ~one big
        beta bet. When enabled, limit existing-open + new same-side positions to
        ``max_same_side_positions``, dropping the lowest-ranked new same-side
        candidates (``qualified`` is already in rank order). OFF => unchanged.
        """
        if not self.corr_cap_enabled:
            return qualified
        counts = {"long": 0, "short": 0}
        for p in (open_positions or []):
            side = p.get("side") if isinstance(p, dict) else getattr(p, "side", None)
            side = str(getattr(side, "value", side) or "").strip().lower()
            if side in counts:
                counts[side] += 1
        kept: List[Dict] = []
        dropped = 0
        for s in qualified:
            side = str(s.get("_decision_side", "")).strip().lower()
            if side in counts:
                if counts[side] >= self.max_same_side_positions:
                    dropped += 1
                    continue
                counts[side] += 1
            kept.append(s)
        if dropped:
            logger.info(
                "Correlation cap: dropped %d same-side candidate(s) over max %d "
                "concurrent (long=%d short=%d)",
                dropped, self.max_same_side_positions, counts["long"], counts["short"],
            )
        return kept

    def _aggregate_directional_scores(self, scored: List[Dict]) -> Tuple[float, float]:
        """
        Aggregate long vs short conviction across all candidates.
        This is the "forced convergence" insight from ChatGPT — we compute
        an overall directional bias even though we execute multiple trades.
        """
        long_score = 0.0
        short_score = 0.0

        for s in scored:
            direction = s.get("_decision_side", "neutral")
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
                    self._cycle_count, regime, available_slots, self.max_positions, len(scored), bias_str or "N/A")

        # Log only top 5 candidates on one line each
        for i, s in enumerate(scored[:5]):
            coin = s.get("_decision_coin", "?")
            side = s.get("_decision_side", "?")
            composite = s.get("_composite_score", 0)
            marker = " <- PRESCREEN" if s in executions else ""
            logger.info("  #%d %s %s composite=%.4f%s", i + 1, side.upper(), coin, composite, marker)

        if executions:
            logger.info("-> PRESCREENED %d candidate(s) for executable ranking", len(executions))
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
            "total_prescreened_candidates": self.stats["total_executions"],
            "execution_rate": (self.stats["total_executions"] /
                              max(self.stats["total_candidates"], 1)),
            "no_trade_rate": (self.stats["total_no_trade"] /
                             max(self.stats["total_decisions"], 1)),
            "recent_decisions": list(self._decision_history)[-10:],
        }

    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        """Return recent decision history for dashboard."""
        return list(self._decision_history)[-limit:]
