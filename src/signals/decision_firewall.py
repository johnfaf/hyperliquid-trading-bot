"""
Decision Firewall
=================
The CRITICAL validation layer between signal generation and execution.

Every signal must pass through this firewall before reaching the execution layer.
It enforces:
  1. Schema validation (is the signal well-formed?)
  2. Risk limits (per-trade, per-coin, portfolio-wide)
  3. Regime alignment (does the strategy fit current market conditions?)
  4. Conflict detection (no opposing positions on same coin)
  5. Source accuracy check (has this signal source been reliable?)
  6. Cooldown enforcement (no revenge trading)

Flow: Signal Source → TradeSignal → DecisionFirewall → Execution
"""
import logging
import threading
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.analysis.trade_analytics import evaluate_short_side_policy
from src.signals.signal_schema import TradeSignal
from src.data import database as db

# Optional: predictive regime forecaster for dynamic de-risking
try:
    from src.signals.predictive_regime_forecaster import PredictiveRegimeForecaster
    HAS_FORECASTER = True
except ImportError:
    HAS_FORECASTER = False

logger = logging.getLogger(__name__)


class DecisionFirewall:
    """
    Validates and filters trade signals before execution.
    Acts as the final gatekeeper — nothing trades without passing here.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # Risk limits
        self.max_risk_per_trade = cfg.get("max_risk_per_trade", 0.01)   # 1% of portfolio
        self.max_total_risk = cfg.get("max_total_risk", 0.05)           # 5% total
        self.max_positions = cfg.get("max_positions", 8)
        self.max_per_coin = cfg.get("max_per_coin", 3)                  # Max 3 positions per coin
        self.max_leverage = cfg.get("max_leverage", 5)
        # Lowered from 0.30 → 0.15: during paper-trading bootstrap, most
        # strategies have thin history so confidence is dampened by sample-size
        # penalties.  0.15 lets early signals through for validation while still
        # blocking truly garbage signals.  Raise back to 0.25-0.30 once the
        # strategy DB has 50+ scored strategies with >10 trades each.
        self.min_confidence = cfg.get("min_confidence", 0.15)
        self.min_source_accuracy = cfg.get("min_source_accuracy", 0.0)  # 0 = no filter
        self.max_signals_per_source_per_day = int(
            cfg.get("max_signals_per_source_per_day", 0)
        )
        self.agent_scorer = cfg.get("agent_scorer")
        self.event_scanner = cfg.get("event_scanner")
        self.event_risk_enabled = bool(cfg.get("event_risk_enabled", True))
        self.short_hardening_enabled = bool(cfg.get("short_hardening_enabled", True))
        self.short_hardening_lookback_trades = max(
            10, int(cfg.get("short_hardening_lookback_trades", 120))
        )
        self.short_hardening_min_closed_trades = max(
            1, int(cfg.get("short_hardening_min_closed_trades", 12))
        )
        self.short_hardening_degrade_win_rate = float(
            cfg.get("short_hardening_degrade_win_rate", 0.45)
        )
        self.short_hardening_block_win_rate = float(
            cfg.get("short_hardening_block_win_rate", 0.35)
        )
        self.short_hardening_block_net_pnl = float(
            cfg.get("short_hardening_block_net_pnl", -1.0)
        )
        self.short_hardening_confidence_multiplier = float(
            cfg.get("short_hardening_confidence_multiplier", 0.85)
        )
        self.short_hardening_size_multiplier = float(
            cfg.get("short_hardening_size_multiplier", 0.60)
        )
        self.canary_mode = bool(cfg.get("canary_mode", False))
        self.canary_max_positions = max(1, int(cfg.get("canary_max_positions", 2)))
        # Lowered from 300 → 60: 5-minute cooldown was blocking re-entries
        # during paper trading when signals fire frequently on the same coin.
        self.cooldown_seconds = cfg.get("cooldown_seconds", 60)
        self.daily_loss_limit_pct = cfg.get("daily_loss_limit_pct", 0.03)
        if self.canary_mode:
            self.max_positions = min(self.max_positions, self.canary_max_positions)

        # Portfolio-level aggregate exposure limit
        # With 2000 traders scanned and golden wallets auto-connected,
        # we need a hard cap on total notional exposure across ALL positions
        # Exposure cap: 8% position × 5x leverage = 40% notional per trade.
        # Two concurrent trades = 80%.  60% cap allows 1–2 leveraged positions.
        # Raised from 0.80 → 1.50: for paper trading, the 80% cap combined
        # with 5x leverage means only ~2 positions can co-exist. 1.50 allows
        # up to ~4-5 concurrent leveraged paper positions for better strategy
        # evaluation.  Reduce to 0.60-0.80 for live trading.
        self.max_aggregate_exposure_pct = cfg.get("max_aggregate_exposure", 1.50)

        # Predictive regime forecaster for dynamic de-risking
        self.enable_predictive_derisk = cfg.get("enable_predictive_derisk", True)
        self.crash_confidence_threshold = cfg.get("crash_confidence_threshold", 0.4)
        self.crash_size_multiplier = cfg.get("crash_size_multiplier", 0.20)    # 80% reduction
        self.crash_exposure_cap = cfg.get("crash_exposure_cap", 0.25)          # 25% vs normal 60%
        self._normal_exposure_cap = self.max_aggregate_exposure_pct            # Save default for reset

        # Funding rate risk: block new longs when funding is deeply negative
        # (means longs pay shorts → holding longs is expensive)
        self.funding_risk_enabled = cfg.get("funding_risk_enabled", True)
        self.funding_negative_threshold = cfg.get("funding_negative_threshold", -0.001)  # -0.1%/8h
        self.funding_positive_threshold = cfg.get("funding_positive_threshold", 0.003)   # +0.3%/8h
        self._funding_cache: Dict[str, float] = {}
        self._funding_cache_ts: float = 0.0
        self._funding_cache_ttl = 120  # 2 minutes

        self._forecaster = cfg.get("forecaster", None)
        if self._forecaster is None and self.enable_predictive_derisk and HAS_FORECASTER:
            try:
                self._forecaster = PredictiveRegimeForecaster()
                logger.info("DecisionFirewall: predictive de-risking ENABLED")
            except Exception as e:
                logger.debug(f"Could not init forecaster: {e}")
        elif self._forecaster is not None:
            logger.info("DecisionFirewall: using externally-provided forecaster")

        # State tracking (protected by _lock for thread safety)
        self._lock = threading.RLock()
        self._recent_trades: Dict[str, float] = {}  # coin -> last trade timestamp
        self._daily_losses: float = 0.0
        self._daily_reset_date: str = ""
        self._source_signal_counts: Dict[str, int] = defaultdict(int)
        self._side_policy_cache: Dict[str, object] = {"ts": 0.0, "short": {}}
        self._side_policy_cache_ttl_s = 300.0

        # Stats
        self.stats = {
            "total_signals": 0,
            "passed": 0,
            "rejected_schema": 0,
            "rejected_confidence": 0,
            "rejected_risk": 0,
            "rejected_regime": 0,
            "rejected_conflict": 0,
            "rejected_cooldown": 0,
            "rejected_accuracy": 0,
            "rejected_source_cap": 0,
            "rejected_source_policy": 0,
            "rejected_drawdown": 0,
            "rejected_exposure": 0,
            "rejected_funding": 0,
            "rejected_event_risk": 0,
            "rejected_side_policy": 0,
            # LOW-FIX LOW-1: count audit-log write failures so ops can detect
            # when the audit trail is silently broken (DB full, locked, etc.)
            "audit_log_failures": 0,
        }

        logger.info(
            "DecisionFirewall initialized: max_risk=%s/trade, max_positions=%d, "
            "min_confidence=%s, canary_mode=%s",
            f"{self.max_risk_per_trade:.0%}",
            self.max_positions,
            f"{self.min_confidence:.0%}",
            self.canary_mode,
        )

    @staticmethod
    def _source_key(signal: TradeSignal) -> str:
        source = getattr(signal, "source", None)
        if hasattr(source, "value"):
            source = source.value
        key = str(source or "unknown").strip().lower()
        key = key or "unknown"

        trader_address = str(getattr(signal, "trader_address", "") or "").strip().lower()
        if key == "copy_trade":
            if trader_address:
                return f"{key}:{trader_address}"
            return key

        strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
        if strategy_type:
            return f"{key}:{strategy_type}"
        return key

    def _effective_source_cap(self, policy: Dict) -> int:
        """Combine static per-source/day caps with allocator-driven caps."""
        configured_cap = int(self.max_signals_per_source_per_day or 0)
        policy_cap = int(policy.get("max_signals_per_day", 0) or 0)
        if configured_cap > 0 and policy_cap > 0:
            return min(configured_cap, policy_cap)
        return configured_cap or policy_cap

    def set_event_scanner(self, event_scanner) -> None:
        self.event_scanner = event_scanner

    def apply_runtime_overrides(self, overrides: Dict) -> None:
        """Apply hot-reloadable config values without recreating the firewall."""
        if not overrides:
            return
        with self._lock:
            self.min_confidence = float(
                overrides.get("FIREWALL_MIN_CONFIDENCE", self.min_confidence)
            )
            self.max_signals_per_source_per_day = int(
                overrides.get(
                    "FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY",
                    self.max_signals_per_source_per_day,
                )
                or 0
            )
            self.event_risk_enabled = bool(
                overrides.get("EVENT_RISK_ENABLED", self.event_risk_enabled)
            )
            self.short_hardening_enabled = bool(
                overrides.get("SHORT_HARDENING_ENABLED", self.short_hardening_enabled)
            )
            self.short_hardening_lookback_trades = max(
                10,
                int(
                    overrides.get(
                        "SHORT_HARDENING_LOOKBACK_TRADES",
                        self.short_hardening_lookback_trades,
                    )
                ),
            )
            self.short_hardening_min_closed_trades = max(
                1,
                int(
                    overrides.get(
                        "SHORT_HARDENING_MIN_CLOSED_TRADES",
                        self.short_hardening_min_closed_trades,
                    )
                ),
            )
            self.short_hardening_degrade_win_rate = float(
                overrides.get(
                    "SHORT_HARDENING_DEGRADE_WIN_RATE",
                    self.short_hardening_degrade_win_rate,
                )
            )
            self.short_hardening_block_win_rate = float(
                overrides.get(
                    "SHORT_HARDENING_BLOCK_WIN_RATE",
                    self.short_hardening_block_win_rate,
                )
            )
            self.short_hardening_block_net_pnl = float(
                overrides.get(
                    "SHORT_HARDENING_BLOCK_NET_PNL",
                    self.short_hardening_block_net_pnl,
                )
            )
            self.short_hardening_confidence_multiplier = float(
                overrides.get(
                    "SHORT_HARDENING_CONFIDENCE_MULTIPLIER",
                    self.short_hardening_confidence_multiplier,
                )
            )
            self.short_hardening_size_multiplier = float(
                overrides.get(
                    "SHORT_HARDENING_SIZE_MULTIPLIER",
                    self.short_hardening_size_multiplier,
                )
            )
            self._side_policy_cache = {"ts": 0.0, "short": {}}

        logger.info(
            "DecisionFirewall runtime overrides applied: min_confidence=%s, source_cap=%s, "
            "short_hardening=%s, event_risk=%s",
            f"{self.min_confidence:.0%}",
            self.max_signals_per_source_per_day,
            self.short_hardening_enabled,
            self.event_risk_enabled,
        )

    def _get_short_side_policy(self) -> Dict:
        if not self.short_hardening_enabled:
            return {
                "status": "disabled",
                "reason": "Short hardening disabled",
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
            }

        now = time.time()
        cached = self._side_policy_cache.get("short") or {}
        if cached and (now - float(self._side_policy_cache.get("ts", 0.0) or 0.0)) < self._side_policy_cache_ttl_s:
            return dict(cached)

        try:
            closed = db.get_paper_trade_history(limit=self.short_hardening_lookback_trades)
            policy = evaluate_short_side_policy(
                closed,
                min_trades=self.short_hardening_min_closed_trades,
                degrade_win_rate=self.short_hardening_degrade_win_rate,
                block_win_rate=self.short_hardening_block_win_rate,
                block_net_pnl=self.short_hardening_block_net_pnl,
            )
        except Exception as exc:
            logger.debug("Short-side policy lookup failed: %s", exc)
            policy = {
                "status": "policy_error",
                "reason": str(exc),
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
            }

        self._side_policy_cache = {"ts": now, "short": dict(policy)}
        return dict(policy)

    def _apply_side_policy(self, signal: TradeSignal) -> Tuple[bool, str]:
        side_val = signal.side.value if hasattr(signal.side, "value") else str(signal.side)
        if str(side_val).lower() != "short":
            return True, ""

        policy = self._get_short_side_policy()
        status = str(policy.get("status", "healthy") or "healthy")
        if status == "blocked":
            return False, policy.get("reason", "Short-side guardrail blocked the signal")
        if status == "degraded":
            original_confidence = float(signal.confidence)
            signal.confidence *= self.short_hardening_confidence_multiplier
            signal.position_pct *= self.short_hardening_size_multiplier
            if signal.size > 0:
                signal.size *= self.short_hardening_size_multiplier
            logger.warning(
                "Short hardening de-risked %s: confidence %.0f%% -> %.0f%%, size *= %.2f (%s)",
                signal.coin,
                original_confidence * 100,
                signal.confidence * 100,
                self.short_hardening_size_multiplier,
                policy.get("reason", "recent short underperformance"),
            )
        return True, ""

    def _apply_event_risk(self, signal: TradeSignal, dry_run: bool = False) -> Tuple[bool, str]:
        if not self.event_risk_enabled or not self.event_scanner:
            return True, ""

        try:
            risk = self.event_scanner.get_risk_state(signal.coin)
        except Exception as exc:
            logger.debug("Event risk lookup failed for %s: %s", signal.coin, exc)
            return True, ""

        reasons = "; ".join(risk.get("reasons", []) or [])
        if risk.get("block_new_entries"):
            return False, reasons or f"Event risk blocks new {signal.coin} entries"

        if risk.get("degrade"):
            conf_mult = float(risk.get("confidence_multiplier", 1.0) or 1.0)
            size_mult = float(risk.get("size_multiplier", 1.0) or 1.0)
            original_confidence = float(signal.confidence)
            signal.confidence *= conf_mult
            if size_mult < 1.0:
                signal.position_pct *= size_mult
                if signal.size > 0:
                    signal.size *= size_mult
            logger.warning(
                "Event risk de-risking %s: confidence %.0f%% -> %.0f%%, size *= %.2f (%s)",
                signal.coin,
                original_confidence * 100,
                signal.confidence * 100,
                size_mult,
                reasons or "scheduled event window",
            )
        return True, ""

    def _apply_source_policy(
        self,
        signal: TradeSignal,
        source_key: str,
        dry_run: bool = False,
    ) -> Tuple[bool, str, Dict]:
        if not self.agent_scorer:
            return True, "", {
                "source_key": source_key,
                "status": "unknown",
                "max_signals_per_day": int(self.max_signals_per_source_per_day or 0),
            }

        try:
            policy = self.agent_scorer.get_source_policy(source_key)
        except Exception as exc:
            logger.debug("Source policy lookup failed for %s: %s", source_key, exc)
            return True, "", {
                "source_key": source_key,
                "status": "policy_error",
                "max_signals_per_day": int(self.max_signals_per_source_per_day or 0),
            }

        status = str(policy.get("status", "unknown") or "unknown")
        if policy.get("blocked"):
            return False, f"Source allocator paused {source_key} ({status})", policy

        min_conf = float(policy.get("min_confidence", 0.0) or 0.0)
        if signal.confidence < min_conf:
            return (
                False,
                f"Source allocator requires {min_conf:.0%} confidence for {source_key} "
                f"(got {signal.confidence:.0%})",
                policy,
            )

        size_mult = float(policy.get("size_multiplier", 1.0) or 1.0)
        if size_mult < 1.0:
            signal.position_pct *= size_mult
            if signal.size > 0:
                signal.size *= size_mult
            logger.info(
                "Source allocator de-risked %s: size *= %.2f (status=%s, weight=%.2f)",
                source_key,
                size_mult,
                status,
                float(policy.get("dynamic_weight", 0.0) or 0.0),
            )

        return True, "", policy

    def validate(self, signal: TradeSignal, regime_data: Optional[Dict] = None,
                 open_positions: Optional[List[Dict]] = None,
                 ignore_position_limit: bool = False,
                 dry_run: bool = False,
                 account_balance: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate a single trade signal through all checks.
        Thread-safe: acquires _lock to prevent concurrent state corruption.

        Returns: (passed: bool, reason: str)
          - passed=True: signal is approved for execution
          - passed=False: signal is rejected with explanation
        """
        with self._lock:
            return self._validate_locked(
                signal,
                regime_data,
                open_positions,
                ignore_position_limit=ignore_position_limit,
                dry_run=dry_run,
                account_balance=account_balance,
            )

    def _validate_locked(self, signal: TradeSignal, regime_data: Optional[Dict] = None,
                         open_positions: Optional[List[Dict]] = None,
                         ignore_position_limit: bool = False,
                         dry_run: bool = False,
                         account_balance: Optional[float] = None) -> Tuple[bool, str]:
        """Inner validate — must be called with _lock held."""
        # Activate the signal's trace ID so every downstream log line
        # (execution, fill verification, SL/TP placement) includes it.
        if hasattr(signal, "activate_trace"):
            signal.activate_trace()

        if not dry_run:
            self.stats["total_signals"] += 1

        def _reject(reason_key, reason_msg):
            """Helper to reject + audit log in one step."""
            if not dry_run:
                self.stats[reason_key] += 1
                try:
                    db.audit_log(
                        action="signal_rejected",
                        coin=signal.coin,
                        side=signal.side.value if hasattr(signal.side, 'value') else str(signal.side),
                        source=getattr(signal, "source", None) or "unknown",
                        details={
                            "reason": reason_msg,
                            "confidence": getattr(signal, "confidence", 0),
                            "trace_id": signal.signal_id,
                        },
                    )
                except Exception:
                    self.stats["audit_log_failures"] += 1
            return False, reason_msg

        # 1. Schema validation
        if not signal.validate():
            return _reject("rejected_schema", f"Invalid signal schema: {signal.coin} {signal.side.value}")

        predictive_regime = None
        if self._forecaster and self.enable_predictive_derisk:
            try:
                predictive_regime = self._forecaster.predict_regime(signal.coin)
            except Exception as e:
                logger.debug(f"Forecaster check failed: {e}")

        if predictive_regime and predictive_regime.get("partial_signal"):
            original_confidence = float(signal.confidence)
            signal.confidence = original_confidence * 0.5
            logger.warning(
                "Partial predictive inputs for %s (%s) - halving confidence %.0f%% -> %.0f%%",
                signal.coin,
                ", ".join(predictive_regime.get("partial_inputs", [])) or "unknown",
                original_confidence * 100,
                signal.confidence * 100,
            )

        event_risk_ok, event_risk_reason = self._apply_event_risk(signal, dry_run=dry_run)
        if not event_risk_ok:
            return _reject("rejected_event_risk", event_risk_reason)

        side_policy_ok, side_policy_reason = self._apply_side_policy(signal)
        if not side_policy_ok:
            return _reject("rejected_side_policy", side_policy_reason)

        # 2. Minimum confidence
        if signal.confidence < self.min_confidence:
            return _reject("rejected_confidence",
                          f"Low confidence {signal.confidence:.0%} < {self.min_confidence:.0%}")

        # 2b. Per-source/day throughput cap (approved signals).
        self._check_daily_reset()
        source_key = self._source_key(signal)
        policy_ok, policy_reason, source_policy = self._apply_source_policy(
            signal,
            source_key,
            dry_run=dry_run,
        )
        if not policy_ok:
            return _reject("rejected_source_policy", policy_reason)

        effective_source_cap = self._effective_source_cap(source_policy)
        if effective_source_cap > 0:
            used = self._source_signal_counts.get(source_key, 0)
            if used >= effective_source_cap:
                return _reject(
                    "rejected_source_cap",
                    f"Source/day cap hit for {source_key} "
                    f"({used}/{effective_source_cap})",
                )

        # 3. Leverage check
        if signal.leverage > self.max_leverage:
            signal.leverage = self.max_leverage  # Clamp instead of reject
            logger.info(f"Clamped leverage to {self.max_leverage}x for {signal.coin}")

        # 4. Position limits
        positions = open_positions if open_positions is not None else db.get_open_paper_trades()
        if not ignore_position_limit and len(positions) >= self.max_positions:
            return _reject("rejected_risk",
                          f"Max positions reached ({len(positions)}/{self.max_positions})")

        # 5. Per-coin limit
        coin_positions = [p for p in positions if p.get("coin") == signal.coin]
        if len(coin_positions) >= self.max_per_coin:
            return _reject("rejected_risk",
                          f"Max positions for {signal.coin} ({len(coin_positions)}/{self.max_per_coin})")

        # 5b. Aggregate portfolio exposure — hard cap across ALL positions
        balance = account_balance
        if balance is None:
            account = db.get_paper_account()
            balance = account.get("balance", 10000) if account else None
        if balance:
            total_exposure = 0.0
            for pos in positions:
                pos_size = pos.get("size", 0)
                pos_price = pos.get("entry_price", pos.get("entryPx", 0))
                pos_leverage = pos.get("leverage", 1)
                total_exposure += abs(pos_size * pos_price * pos_leverage)

            new_notional = (
                signal.size * signal.entry_price * signal.leverage
                if signal.size and signal.entry_price
                else balance * signal.position_pct * signal.leverage
            )
            projected_exposure = total_exposure + new_notional
            exposure_pct = projected_exposure / balance if balance > 0 else 1.0

            if exposure_pct > self.max_aggregate_exposure_pct:
                return _reject("rejected_exposure",
                              f"Aggregate exposure {exposure_pct:.0%} would exceed "
                              f"{self.max_aggregate_exposure_pct:.0%} limit "
                              f"(${projected_exposure:,.0f}/${balance:,.0f})")

        # 6. Conflict detection — no opposing positions on same coin
        for pos in coin_positions:
            if pos.get("side") != signal.side.value:
                return _reject("rejected_conflict",
                              f"Conflict: have {pos.get('side')} {signal.coin}, "
                              f"signal wants {signal.side.value}")

        # 7. Cooldown — prevent revenge trading
        now = time.time()
        last_trade_ts = self._recent_trades.get(signal.coin, 0)
        if now - last_trade_ts < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (now - last_trade_ts))
            return _reject("rejected_cooldown",
                          f"Cooldown: {signal.coin} traded {remaining}s ago")

        # 8. Regime alignment check
        if regime_data:
            guidance = regime_data.get("strategy_guidance", {})
            paused = set(guidance.get("pause", []))
            if "all" in paused:
                return _reject("rejected_regime",
                              f"Regime {regime_data.get('overall_regime', '?')} pauses all trading")
            if signal.strategy_type and signal.strategy_type.lower() in paused:
                return _reject("rejected_regime",
                              f"Regime pauses {signal.strategy_type} "
                              f"(regime={regime_data.get('overall_regime', '?')})")

            # Apply size modifier from regime
            size_mod = guidance.get("size_modifier", 1.0)
            signal.regime_size_modifier = size_mod

        # 9. Source accuracy check (if we have history)
        if self.min_source_accuracy > 0 and signal.source_accuracy > 0:
            if signal.source_accuracy < self.min_source_accuracy:
                return _reject("rejected_accuracy",
                              f"Source accuracy {signal.source_accuracy:.0%} < "
                              f"{self.min_source_accuracy:.0%}")

        # 10. Daily drawdown circuit breaker
        self._check_daily_reset()
        drawdown_balance = account_balance
        if drawdown_balance is None:
            account = db.get_paper_account()
            drawdown_balance = account.get("balance", 10000) if account else None
        if drawdown_balance:
            if self._daily_losses / drawdown_balance > self.daily_loss_limit_pct:
                return _reject("rejected_drawdown",
                              f"Daily loss limit hit ({self._daily_losses / drawdown_balance:.1%} > "
                              f"{self.daily_loss_limit_pct:.0%})")

        # 11. Funding rate risk check
        #     Block new longs when funding is deeply negative (longs pay shorts heavily)
        #     Block new shorts when funding is extremely positive (shorts pay longs)
        if self.funding_risk_enabled:
            try:
                funding = self._get_funding_rate(signal.coin)
                side_val = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
                if side_val == "long" and funding < self.funding_negative_threshold:
                    return _reject("rejected_funding",
                                  f"Funding deeply negative ({funding:.4%}/8h) — "
                                  f"longs pay {abs(funding)*3*365:.0f}% annualized, blocking long")
                elif side_val == "short" and funding > self.funding_positive_threshold:
                    return _reject("rejected_funding",
                                  f"Funding extremely positive ({funding:.4%}/8h) — "
                                  f"shorts pay {funding*3*365:.0f}% annualized, blocking short")
            except Exception as e:
                logger.debug(f"Funding rate check failed: {e}")

        # 12. Predictive regime de-risking
        #     If forecaster detects crash with high confidence, dynamically reduce
        #     position size and tighten exposure cap instead of outright blocking.
        #     NOTE: We do NOT mutate self.max_aggregate_exposure_pct here —
        #     that was a thread-safety bug (one thread's crash detection affected
        #     all other threads). Instead, the size reduction is per-signal only.
        if predictive_regime:
            try:
                if (predictive_regime["regime"] == "crash" and
                        predictive_regime["confidence"] > self.crash_confidence_threshold):
                    # De-risk: cut position size dramatically
                    signal.size *= self.crash_size_multiplier  # 80% reduction
                    logger.warning(
                        f"CRASH REGIME detected for {signal.coin} "
                        f"(conf={predictive_regime['confidence']:.2f}) — "
                        f"de-risking: size *= {self.crash_size_multiplier}"
                    )
            except Exception as e:
                logger.debug(f"Forecaster check failed: {e}")

        # All checks passed
        if not dry_run:
            self.stats["passed"] += 1
            self._recent_trades[signal.coin] = now
            if effective_source_cap > 0:
                self._source_signal_counts[source_key] += 1

            # Audit trail: record approval
            try:
                db.audit_log(
                    action="signal_approved",
                    coin=signal.coin,
                    side=signal.side.value,
                    source=getattr(signal, "source", None) or "unknown",
                    details={
                        "confidence": signal.confidence,
                        "leverage": signal.leverage,
                        "strategy_type": signal.strategy_type,
                        "trace_id": signal.signal_id,
                    },
                )
            except Exception:
                pass  # audit logging must never break the trading path

        return True, "approved"

    def validate_batch(self, signals: List[TradeSignal],
                        regime_data: Optional[Dict] = None) -> List[Tuple[TradeSignal, bool, str]]:
        """
        Validate a batch of signals. Returns list of (signal, passed, reason).
        Processes in order of confidence (highest first).
        Thread-safe: holds _lock for the entire batch to prevent interleaving.
        """
        with self._lock:
            positions = db.get_open_paper_trades()

            # Sort by confidence descending
            sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

            results = []
            for signal in sorted_signals:
                passed, reason = self._validate_locked(signal, regime_data, positions)
                results.append((signal, passed, reason))

                # If signal passed, add to positions for subsequent checks.
                # CRIT-FIX CRIT-3: include size/entry_price/leverage so the aggregate
                # exposure check inside _validate_locked sees real notional for within-batch
                # approvals — without these fields pos.get("size", 0) returns 0 and a burst
                # of concurrent signals can all pass the exposure cap simultaneously.
                if passed:
                    positions.append({
                        "coin": signal.coin,
                        "side": signal.side.value,
                        "status": "open",
                        "entry_price": float(getattr(signal, "entry_price", 0) or 0),
                        "size": float(getattr(signal, "size", 0) or 0),
                        "leverage": float(getattr(signal, "leverage", 1) or 1),
                    })

            approved = sum(1 for _, p, _ in results if p)
            rejected = sum(1 for _, p, _ in results if not p)
            logger.info(f"Firewall batch: {approved} approved, {rejected} rejected out of {len(signals)}")

            return results

    def record_trade_outcome(self, coin: str, pnl: float):
        """Record a trade outcome for daily drawdown tracking."""
        with self._lock:
            if pnl < 0:
                self._daily_losses += abs(pnl)
            self._side_policy_cache["ts"] = 0.0

    def set_daily_losses(self, loss_amount: float):
        """Set the current day's realized loss snapshot directly."""
        with self._lock:
            self._daily_losses = max(float(loss_amount or 0.0), 0.0)

    def _get_funding_rate(self, coin: str) -> float:
        """
        Fetch current funding rate from Hyperliquid (cached).
        Returns per-8h rate (NOT annualized).
        """
        now = time.time()
        if now - self._funding_cache_ts < self._funding_cache_ttl and coin in self._funding_cache:
            return self._funding_cache[coin]

        try:
            # BUG-5 FIX: route through the centralized APIManager instead of
            # raw requests.post().  The old code bypassed rate limiting, TTL
            # cache, and the circuit breaker, risking untracked 429 responses.
            from src.core.api_manager import get_manager, Priority
            data = get_manager().post(
                payload={"type": "metaAndAssetCtxs"},
                priority=Priority.NORMAL,
                timeout=5,
            )
            if isinstance(data, list) and len(data) >= 2:
                meta = data[0]
                asset_ctxs = data[1]
                for i, asset in enumerate(meta.get("universe", [])):
                    if i < len(asset_ctxs):
                        name = asset.get("name", "").upper()
                        rate = float(asset_ctxs[i].get("funding", 0))
                        self._funding_cache[name] = rate
                self._funding_cache_ts = now
        except Exception as e:
            logger.debug(f"Funding rate fetch failed: {e}")

        return self._funding_cache.get(coin.upper(), 0.0)

    def _check_daily_reset(self):
        """Reset daily loss counter at midnight UTC. Must hold _lock."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_losses = 0.0
            self._source_signal_counts.clear()

    def get_stats(self) -> Dict:
        """Return firewall statistics."""
        total = self.stats["total_signals"]
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / total if total > 0 else 0,
            "daily_losses": round(self._daily_losses, 2),
            "top_rejection_reason": max(
                [(k, v) for k, v in self.stats.items() if k.startswith("rejected_")],
                key=lambda x: x[1], default=("none", 0)
            )[0],
            "canary_mode": self.canary_mode,
            "max_signals_per_source_per_day": int(self.max_signals_per_source_per_day),
            "source_signal_counts": dict(self._source_signal_counts),
            "source_policies": self.agent_scorer.get_scorecard() if self.agent_scorer else [],
            "short_side_policy": self._get_short_side_policy(),
        }
