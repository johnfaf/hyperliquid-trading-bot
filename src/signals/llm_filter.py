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

Two modes:
  1. Rule-based (default) — fast, deterministic heuristics
  2. Claude API (opt-in via ANTHROPIC_API_KEY) — semantic reasoning on signals

When Claude API is enabled, the rule-based checks still run first as a
fast pre-filter.  Only signals that pass rules are sent to Claude for
deeper reasoning.  This keeps API costs low (~1 call per surviving signal).

Filter checks (rule-based):
  1. Regime contradiction — signal opposes current regime
  2. Memory warning — similar past trades lost money
  3. Multi-signal conflict — different sources disagree
  4. Exhaustion trap — signal is chasing an already-extended move
  5. Risk cluster — too many positions in correlated assets
"""
import logging
import os
import json
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMFilter:
    """
    Hybrid rule-based + LLM filter that catches bad setups
    the quantitative models might miss.

    When ANTHROPIC_API_KEY is set, rule-passing signals are sent to Claude
    for semantic reasoning.  Without the key, only rule-based filtering runs.
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

        # Claude API configuration
        self._api_key = cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = cfg.get("anthropic_model") or os.environ.get(
            "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
        )
        self._llm_enabled = bool(self._api_key)
        self._llm_max_tokens = cfg.get("llm_max_tokens", 300)
        self._llm_timeout = cfg.get("llm_timeout", 10)  # seconds
        # Rate limiting: max N Claude calls per minute
        self._llm_max_calls_per_min = cfg.get("llm_max_calls_per_min", 10)
        self._llm_call_timestamps: list = []
        self._client = None
        self._client_supports_request_options = False

        # Stats
        self.stats = {
            "total_filtered": 0,
            "passed": 0,
            "blocked_regime": 0,
            "blocked_memory": 0,
            "blocked_conflict": 0,
            "blocked_exhaustion": 0,
            "blocked_correlation": 0,
            "blocked_llm": 0,
            "llm_calls": 0,
            "llm_errors": 0,
        }

        if self._llm_enabled:
            try:
                import anthropic
                client_kwargs = {
                    "api_key": self._api_key,
                    "max_retries": 0,
                    "timeout": self._llm_timeout,
                }
                try:
                    self._client = anthropic.Anthropic(**client_kwargs)
                    self._client_supports_request_options = True
                except TypeError:
                    # Older SDKs may not accept timeout/max_retries on init.
                    # We still try per-request options below.
                    self._client = anthropic.Anthropic(api_key=self._api_key)
                    self._client_supports_request_options = hasattr(
                        self._client, "with_options"
                    )
                logger.info("LLMFilter initialized (Claude API mode: %s)", self._model)
            except ImportError:
                logger.warning("LLMFilter: anthropic package not installed, falling back to rule-based")
                self._llm_enabled = False
            except Exception as e:
                logger.warning("LLMFilter: Claude API init failed (%s), falling back to rule-based", e)
                self._llm_enabled = False
        else:
            logger.info("LLMFilter initialized (rule-based mode, set ANTHROPIC_API_KEY to enable Claude)")

    def filter(self, signal: Dict, context: Dict) -> Tuple[bool, float, str]:
        """
        Filter a trade signal through rule-based checks, then optionally Claude.

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

        # ─── Phase 1: Rule-based fast pre-filter ─────────────────
        approved, confidence, reason = self._rule_filter(signal, context)
        if not approved:
            return False, confidence, reason

        # ─── Phase 2: Claude API semantic reasoning ──────────────
        if self._llm_enabled and self._client:
            return self._claude_filter(signal, context, confidence, reason)

        # Rule-based only
        self.stats["passed"] += 1
        return True, confidence, reason

    def _rule_filter(self, signal: Dict, context: Dict) -> Tuple[bool, float, str]:
        """Fast deterministic rule-based checks."""
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
                reasons.append("contra-regime: shorting in TRENDING_UP")
            elif regime == "TRENDING_DOWN" and side == "long":
                confidence *= 0.6
                reasons.append("contra-regime: longing in TRENDING_DOWN")
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
        if confidence < 0.20:
            reason_str = " | ".join(reasons) if reasons else "combined filters"
            return False, confidence, f"Confidence too low after filters: {reason_str}"

        if reasons:
            logger.debug(f"LLMFilter [{coin} {side}]: passed rules with adjustments -- {', '.join(reasons)}")

        return True, confidence, "approved" + (f" ({', '.join(reasons)})" if reasons else "")

    # ─── Claude API Integration ──────────────────────────────────

    def _claude_filter(self, signal: Dict, context: Dict,
                       rule_confidence: float, rule_reason: str) -> Tuple[bool, float, str]:
        """Send the signal to Claude for semantic reasoning."""
        # Rate limit check
        now = time.time()
        self._llm_call_timestamps = [t for t in self._llm_call_timestamps if now - t < 60]
        if len(self._llm_call_timestamps) >= self._llm_max_calls_per_min:
            # Over rate limit — pass through with rule-based result
            self.stats["passed"] += 1
            return True, rule_confidence, rule_reason + " (LLM rate-limited)"

        try:
            prompt = self._build_prompt(signal, context, rule_confidence, rule_reason)
            self.stats["llm_calls"] += 1
            self._llm_call_timestamps.append(now)

            client = self._get_request_client()
            response = client.messages.create(
                model=self._model,
                max_tokens=self._llm_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_response(response, signal, rule_confidence)

        except Exception as e:
            self.stats["llm_errors"] += 1
            logger.warning("LLMFilter Claude API error: %s -- falling back to rules", e)
            self.stats["passed"] += 1
            return True, rule_confidence, rule_reason + " (LLM fallback)"

    def _get_request_client(self):
        """Return a client/view with the configured timeout budget applied."""
        client = self._client
        if client is None:
            raise RuntimeError("Claude client is not initialized")

        if not self._client_supports_request_options:
            return client

        with_options = getattr(client, "with_options", None)
        if not callable(with_options):
            return client

        try:
            return with_options(timeout=self._llm_timeout, max_retries=0)
        except TypeError:
            try:
                return with_options(timeout=self._llm_timeout)
            except TypeError:
                return client

    def _build_prompt(self, signal: Dict, context: Dict,
                      rule_confidence: float, rule_reason: str) -> str:
        """Build the Claude prompt for trade evaluation."""
        coin = signal.get("coin", "?")
        side = signal.get("side", "?")
        features = signal.get("features", {})
        regime_data = context.get("regime_data", {})
        open_positions = context.get("open_positions", [])

        # Summarize open positions
        open_summary = "None"
        if open_positions:
            pos_list = [f"{p.get('coin','?')} {p.get('side','?')}" for p in open_positions[:8]]
            open_summary = ", ".join(pos_list)

        # Memory context
        memory = context.get("memory_result")
        memory_summary = "No similar trades"
        if memory:
            if isinstance(memory, dict):
                wr = memory.get("win_rate", 0)
                rec = memory.get("recommendation", "none")
                memory_summary = f"Win rate on similar: {wr:.0%}, rec: {rec}"
            else:
                memory_summary = f"Rec: {getattr(memory, 'recommendation', 'none')}"

        return f"""You are a risk-management filter for a crypto perpetual futures trading bot.

TASK: Evaluate whether to APPROVE or REJECT this trade signal. You are NOT predicting price — you are checking for red flags the quant models might miss.

SIGNAL:
- Coin: {coin}
- Side: {side}
- Rule-adjusted confidence: {rule_confidence:.2f}
- Rule notes: {rule_reason}

FEATURES:
- RSI: {features.get('rsi', 'N/A')}
- Trend strength: {features.get('trend_strength', 'N/A')}
- Volatility: {features.get('volatility', 'N/A')}
- Volume ratio: {features.get('volume_ratio', 'N/A')}
- Funding rate: {features.get('funding_rate', 'N/A')}
- Bollinger position: {features.get('bollinger_position', 'N/A')}
- Momentum score: {features.get('momentum_score', 'N/A')}
- Setup type: {features.get('setup_type', 'N/A')}

MARKET CONTEXT:
- Regime: {regime_data.get('overall_regime', 'unknown')}
- Regime confidence: {regime_data.get('confidence', 'N/A')}
- Open positions: {open_summary}
- Trade memory: {memory_summary}

Respond with EXACTLY this JSON format (no other text):
{{"decision": "approve" or "reject", "confidence_adjustment": float between 0.5 and 1.2, "reason": "brief one-line reason"}}

Rules:
- Default to APPROVE unless you see a clear red flag
- confidence_adjustment multiplies the current confidence (1.0 = no change, 0.7 = reduce 30%, 1.1 = boost 10%)
- Be concise in reason (under 80 chars)
- Common red flags: regime mismatch, exhaustion, overcrowded trade, poor risk/reward timing"""

    def _parse_response(self, response, signal: Dict,
                        rule_confidence: float) -> Tuple[bool, float, str]:
        """Parse Claude's response into filter decision."""
        try:
            text = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            decision = result.get("decision", "approve").lower()
            adj = float(result.get("confidence_adjustment", 1.0))
            reason = result.get("reason", "Claude review")

            # Clamp adjustment
            adj = max(0.5, min(adj, 1.2))
            final_confidence = rule_confidence * adj

            if decision == "reject":
                self.stats["blocked_llm"] += 1
                logger.info("LLMFilter Claude REJECTED %s %s: %s",
                           signal.get("coin"), signal.get("side"), reason)
                return False, 0, f"Claude reject: {reason}"

            # Approved
            if final_confidence < 0.20:
                self.stats["blocked_llm"] += 1
                return False, final_confidence, f"Claude reduced confidence too low: {reason}"

            self.stats["passed"] += 1
            logger.debug("LLMFilter Claude APPROVED %s %s (conf %.2f -> %.2f): %s",
                        signal.get("coin"), signal.get("side"),
                        rule_confidence, final_confidence, reason)
            return True, final_confidence, f"Claude approved: {reason}"

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # Parse failure — fall through with rule-based result
            self.stats["llm_errors"] += 1
            logger.warning("LLMFilter: Claude response parse error: %s", e)
            self.stats["passed"] += 1
            return True, rule_confidence, "approved (Claude parse fallback)"

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
            "llm_enabled": self._llm_enabled,
            "llm_model": self._model if self._llm_enabled else "none",
            "pass_rate": self.stats["passed"] / total if total > 0 else 0,
            "block_rate": 1 - (self.stats["passed"] / total) if total > 0 else 0,
        }
