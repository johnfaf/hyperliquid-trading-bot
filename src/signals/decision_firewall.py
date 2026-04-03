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
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.signals.signal_schema import TradeSignal, SignalStrength
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
        # Lowered from 300 → 60: 5-minute cooldown was blocking re-entries
        # during paper trading when signals fire frequently on the same coin.
        self.cooldown_seconds = cfg.get("cooldown_seconds", 60)

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
            "rejected_drawdown": 0,
            "rejected_exposure": 0,
            "rejected_funding": 0,
        }

        logger.info(f"DecisionFirewall initialized: max_risk={self.max_risk_per_trade:.0%}/trade, "
                    f"max_positions={self.max_positions}, min_confidence={self.min_confidence:.0%}")

    def validate(self, signal: TradeSignal, regime_data: Optional[Dict] = None,
                 open_positions: Optional[List[Dict]] = None,
                 ignore_position_limit: bool = False,
                 dry_run: bool = False) -> Tuple[bool, str]:
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
            )

    def _validate_locked(self, signal: TradeSignal, regime_data: Optional[Dict] = None,
                         open_positions: Optional[List[Dict]] = None,
                         ignore_position_limit: bool = False,
                         dry_run: bool = False) -> Tuple[bool, str]:
        """Inner validate — must be called with _lock held."""
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
                        details={"reason": reason_msg, "confidence": getattr(signal, "confidence", 0)},
                    )
                except Exception:
                    pass
            return False, reason_msg

        # 1. Schema validation
        if not signal.validate():
            return _reject("rejected_schema", f"Invalid signal schema: {signal.coin} {signal.side.value}")

        # 2. Minimum confidence
        if signal.confidence < self.min_confidence:
            return _reject("rejected_confidence",
                          f"Low confidence {signal.confidence:.0%} < {self.min_confidence:.0%}")

        # 3. Leverage check
        if signal.leverage > self.max_leverage:
            signal.leverage = self.max_leverage  # Clamp instead of reject
            logger.info(f"Clamped leverage to {self.max_leverage}x for {signal.coin}")

        # 4. Position limits
        positions = open_positions or db.get_open_paper_trades()
        if not ignore_position_limit and len(positions) >= self.max_positions:
            return _reject("rejected_risk",
                          f"Max positions reached ({len(positions)}/{self.max_positions})")

        # 5. Per-coin limit
        coin_positions = [p for p in positions if p.get("coin") == signal.coin]
        if len(coin_positions) >= self.max_per_coin:
            return _reject("rejected_risk",
                          f"Max positions for {signal.coin} ({len(coin_positions)}/{self.max_per_coin})")

        # 5b. Aggregate portfolio exposure — hard cap across ALL positions
        account = db.get_paper_account()
        if account:
            balance = account.get("balance", 10000)
            total_exposure = 0.0
            for pos in positions:
                pos_size = pos.get("size", 0)
                pos_price = pos.get("entry_price", 0)
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
        account = db.get_paper_account()
        if account:
            balance = account.get("balance", 10000)
            if self._daily_losses / balance > 0.03:  # 3% daily loss limit
                return _reject("rejected_drawdown",
                              f"Daily loss limit hit ({self._daily_losses / balance:.1%} > 3%)")

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
        if self._forecaster and self.enable_predictive_derisk:
            try:
                pred = self._forecaster.predict_regime(signal.coin)
                if (pred["regime"] == "crash" and
                        pred["confidence"] > self.crash_confidence_threshold):
                    # De-risk: cut position size dramatically
                    signal.size *= self.crash_size_multiplier  # 80% reduction
                    logger.warning(
                        f"CRASH REGIME detected for {signal.coin} "
                        f"(conf={pred['confidence']:.2f}) — "
                        f"de-risking: size *= {self.crash_size_multiplier}"
                    )
            except Exception as e:
                logger.debug(f"Forecaster check failed: {e}")

        # All checks passed
        if not dry_run:
            self.stats["passed"] += 1
            self._recent_trades[signal.coin] = now

            # Audit trail: record approval
            try:
                db.audit_log(
                    action="signal_approved",
                    coin=signal.coin,
                    side=signal.side.value,
                    source=getattr(signal, "source", None) or "unknown",
                    details={"confidence": signal.confidence, "leverage": signal.leverage,
                             "strategy_type": signal.strategy_type},
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

                # If signal passed, add to positions for subsequent checks
                if passed:
                    positions.append({
                        "coin": signal.coin,
                        "side": signal.side.value,
                        "status": "open",
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

    def _get_funding_rate(self, coin: str) -> float:
        """
        Fetch current funding rate from Hyperliquid (cached).
        Returns annualized rate approximation.
        """
        now = time.time()
        if now - self._funding_cache_ts < self._funding_cache_ttl and coin in self._funding_cache:
            return self._funding_cache[coin]

        try:
            import requests
            resp = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "metaAndAssetCtxs"},
                timeout=5
            )
            if resp.ok:
                data = resp.json()
                if len(data) >= 2:
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

    def get_stats(self) -> Dict:
        """Return firewall statistics."""
        total = self.stats["total_signals"]
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / total if total > 0 else 0,
            "top_rejection_reason": max(
                [(k, v) for k, v in self.stats.items() if k.startswith("rejected_")],
                key=lambda x: x[1], default=("none", 0)
            )[0],
        }
