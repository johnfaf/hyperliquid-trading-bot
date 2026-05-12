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
from src.core import clock_provider
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.analysis.trade_analytics import (
    evaluate_long_side_policy,
    evaluate_short_side_policy,
    evaluate_side_source_policy,
)
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
        # Calibration tracker — when wired in, sources whose calibration
        # has gone off the rails get gated as if they were "warmup" by
        # the source allocator. Optional; the firewall behaves the same
        # as before when this is None.
        self.calibration = cfg.get("calibration")
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
            cfg.get("short_hardening_degrade_win_rate", 0.48)
        )
        self.short_hardening_block_win_rate = float(
            cfg.get("short_hardening_block_win_rate", 0.40)
        )
        self.short_hardening_block_net_pnl = float(
            cfg.get("short_hardening_block_net_pnl", -0.5)
        )
        self.short_hardening_confidence_multiplier = float(
            cfg.get("short_hardening_confidence_multiplier", 0.80)
        )
        self.short_hardening_size_multiplier = float(
            cfg.get("short_hardening_size_multiplier", 0.50)
        )
        self.short_hardening_block_override_enabled = bool(
            cfg.get("short_hardening_block_override_enabled", True)
        )
        self.short_hardening_block_override_min_confidence = float(
            cfg.get("short_hardening_block_override_min_confidence", 0.70)
        )
        self.short_hardening_block_override_min_regime_confidence = float(
            cfg.get("short_hardening_block_override_min_regime_confidence", 0.60)
        )
        self.short_hardening_block_override_size_multiplier = float(
            cfg.get("short_hardening_block_override_size_multiplier", 0.35)
        )
        self.short_hardening_market_adaptive_override_enabled = bool(
            cfg.get("short_hardening_market_adaptive_override_enabled", True)
        )
        self.short_hardening_market_adaptive_min_momentum = float(
            cfg.get("short_hardening_market_adaptive_min_momentum", 0.003)
        )
        self.short_hardening_market_adaptive_scoped_size_multiplier = min(
            1.0,
            max(
                0.05,
                float(cfg.get("short_hardening_market_adaptive_scoped_size_multiplier", 0.25)),
            ),
        )
        self.short_hardening_source_guard_enabled = bool(
            cfg.get("short_hardening_source_guard_enabled", True)
        )
        self.short_hardening_source_min_closed_trades = max(
            1, int(cfg.get("short_hardening_source_min_closed_trades", 3))
        )
        self.short_hardening_source_block_net_pnl = float(
            cfg.get("short_hardening_source_block_net_pnl", -0.25)
        )
        self.short_hardening_coin_guard_enabled = bool(
            cfg.get("short_hardening_coin_guard_enabled", True)
        )
        self.short_hardening_coin_min_closed_trades = max(
            1, int(cfg.get("short_hardening_coin_min_closed_trades", 4))
        )
        self.short_hardening_coin_block_net_pnl = float(
            cfg.get("short_hardening_coin_block_net_pnl", -0.25)
        )
        # ── long_hardening: structural mirror of short_hardening ──
        # Previously only the short side was gated against recent losses, which
        # combined with naturally long-leaning signal sources (copy traders,
        # bullish-regime size boost, alpha_arena trend_following bug, etc.)
        # to push 88-90% of executed trades to the long side. Adding the same
        # gate against long history breaks that asymmetry — losing longs now
        # block/degrade new long entries the same way losing shorts do.
        self.long_hardening_enabled = bool(cfg.get("long_hardening_enabled", True))
        self.long_hardening_lookback_trades = max(
            10, int(cfg.get("long_hardening_lookback_trades", 120))
        )
        self.long_hardening_min_closed_trades = max(
            1, int(cfg.get("long_hardening_min_closed_trades", 12))
        )
        self.long_hardening_degrade_win_rate = float(
            cfg.get("long_hardening_degrade_win_rate", 0.48)
        )
        self.long_hardening_block_win_rate = float(
            cfg.get("long_hardening_block_win_rate", 0.40)
        )
        self.long_hardening_block_net_pnl = float(
            cfg.get("long_hardening_block_net_pnl", -0.5)
        )
        self.long_hardening_confidence_multiplier = float(
            cfg.get("long_hardening_confidence_multiplier", 0.80)
        )
        self.long_hardening_size_multiplier = float(
            cfg.get("long_hardening_size_multiplier", 0.50)
        )
        self.market_side_guard_enabled = bool(
            cfg.get("market_side_guard_enabled", True)
        )
        self.market_side_guard_min_confidence = float(
            cfg.get("market_side_guard_min_confidence", 0.60)
        )
        # Only enforce strategy-level regime pauses when the regime call itself
        # is confident enough. A 49%-confidence "ranging" call shouldn't pause
        # momentum strategies — that's how the bot ends up rejecting 5/5
        # candidates and just paying infra to do nothing.
        self.regime_pause_min_confidence = float(
            cfg.get("regime_pause_min_confidence", 0.55)
        )
        # Tolerance band on confidence-vs-threshold comparisons. A signal at
        # 0.4499 vs threshold 0.45 should not be rejected by an off-by-one-bp
        # rounding artefact.
        self.confidence_threshold_tolerance = max(
            0.0, float(cfg.get("confidence_threshold_tolerance", 0.005))
        )
        self.canary_mode = bool(cfg.get("canary_mode", False))
        self.canary_max_positions = max(1, int(cfg.get("canary_max_positions", 2)))
        self.cooldown_seconds = int(cfg.get("cooldown_seconds", 180))
        self.same_side_cooldown_seconds = int(
            cfg.get("same_side_cooldown_seconds", 900)
        )
        self.max_same_side_positions_per_coin = max(
            1, int(cfg.get("max_same_side_positions_per_coin", 2))
        )
        self.block_losing_averaging = bool(cfg.get("block_losing_averaging", True))
        self.averaging_max_loss_roe_pct = float(
            cfg.get("averaging_max_loss_roe_pct", 0.015)
        )
        self.entry_location_filter_enabled = bool(
            cfg.get("entry_location_filter_enabled", True)
        )
        self.entry_max_atr_extension = float(
            cfg.get("entry_max_atr_extension", 1.8)
        )
        self.entry_max_price_extension_pct = float(
            cfg.get("entry_max_price_extension_pct", 0.035)
        )
        self.side_imbalance_guard_enabled = bool(
            cfg.get("side_imbalance_guard_enabled", True)
        )
        self.side_imbalance_lookback_trades = max(
            10, int(cfg.get("side_imbalance_lookback_trades", 60))
        )
        self.side_imbalance_min_samples = max(
            5, int(cfg.get("side_imbalance_min_samples", 12))
        )
        self.side_imbalance_max_share = min(
            0.98, max(0.50, float(cfg.get("side_imbalance_max_share", 0.80)))
        )
        self.side_imbalance_confidence_bump = max(
            0.0, float(cfg.get("side_imbalance_confidence_bump", 0.15))
        )
        self.side_imbalance_size_multiplier = min(
            1.0, max(0.05, float(cfg.get("side_imbalance_size_multiplier", 0.50)))
        )
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
        #
        # AUDIT M1 — This cap is measured in *leveraged notional* because
        # _estimate_signal_notional() returns ``size × price × leverage``.
        # Conceptually this is "market exposure × leverage": a $200 margin
        # trade at 5x counts as $1000 against this cap.  The cap therefore
        # intentionally tightens with leverage — it's a circuit breaker on
        # both face value AND embedded leverage risk, not a pure
        # capital-at-risk limit.  For the capital-at-risk view we expose
        # ``max_aggregate_margin_pct`` separately below, so a trade must pass
        # BOTH caps to be approved.
        self.max_aggregate_exposure_pct = cfg.get("max_aggregate_exposure", 1.50)

        # AUDIT M1 — Aggregate margin cap (capital actually locked up).
        # This is the cleaner "how much of my balance is reserved as margin
        # across all open positions" view, independent of leverage math.
        # A single 5x-leveraged $200 margin trade counts as $200 here (not
        # $1000).  Default 0.60 = "never lock more than 60% of equity as
        # margin across all positions" — safer reading than the leveraged
        # exposure cap when leverage is high.  Set to 0 (or negative) to
        # disable this cap; both caps default to ON.
        self.max_aggregate_margin_pct = float(
            cfg.get("max_aggregate_margin_pct", 0.60)
        )

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
        self._recent_side_trades: Dict[Tuple[str, str], float] = {}
        self._daily_losses: float = 0.0
        self._daily_reset_date: str = ""
        self._source_signal_counts: Dict[str, int] = defaultdict(int)
        self._side_policy_cache: Dict[str, object] = {
            "ts": 0.0,
            "closed": [],
            "short": {},
            "scoped": {},
        }
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
            "rejected_pyramiding": 0,
            "rejected_accuracy": 0,
            "rejected_source_cap": 0,
            "rejected_source_policy": 0,
            "rejected_drawdown": 0,
            "rejected_exposure": 0,
            "rejected_funding": 0,
            "rejected_event_risk": 0,
            "rejected_side_policy": 0,
            "rejected_entry_location": 0,
            "rejected_side_imbalance": 0,
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
            self.short_hardening_block_override_enabled = bool(
                overrides.get(
                    "SHORT_HARDENING_BLOCK_OVERRIDE_ENABLED",
                    self.short_hardening_block_override_enabled,
                )
            )
            self.short_hardening_block_override_min_confidence = float(
                overrides.get(
                    "SHORT_HARDENING_BLOCK_OVERRIDE_MIN_CONFIDENCE",
                    self.short_hardening_block_override_min_confidence,
                )
            )
            self.short_hardening_block_override_min_regime_confidence = float(
                overrides.get(
                    "SHORT_HARDENING_BLOCK_OVERRIDE_MIN_REGIME_CONFIDENCE",
                    self.short_hardening_block_override_min_regime_confidence,
                )
            )
            self.short_hardening_block_override_size_multiplier = float(
                overrides.get(
                    "SHORT_HARDENING_BLOCK_OVERRIDE_SIZE_MULTIPLIER",
                    self.short_hardening_block_override_size_multiplier,
                )
            )
            self.short_hardening_market_adaptive_override_enabled = bool(
                overrides.get(
                    "SHORT_HARDENING_MARKET_ADAPTIVE_OVERRIDE_ENABLED",
                    self.short_hardening_market_adaptive_override_enabled,
                )
            )
            self.short_hardening_market_adaptive_min_momentum = float(
                overrides.get(
                    "SHORT_HARDENING_MARKET_ADAPTIVE_MIN_MOMENTUM",
                    self.short_hardening_market_adaptive_min_momentum,
                )
            )
            self.short_hardening_market_adaptive_scoped_size_multiplier = min(
                1.0,
                max(
                    0.05,
                    float(
                        overrides.get(
                            "SHORT_HARDENING_MARKET_ADAPTIVE_SCOPED_SIZE_MULTIPLIER",
                            self.short_hardening_market_adaptive_scoped_size_multiplier,
                        )
                    ),
                ),
            )
            self.short_hardening_source_guard_enabled = bool(
                overrides.get(
                    "SHORT_HARDENING_SOURCE_GUARD_ENABLED",
                    self.short_hardening_source_guard_enabled,
                )
            )
            self.short_hardening_source_min_closed_trades = max(
                1,
                int(
                    overrides.get(
                        "SHORT_HARDENING_SOURCE_MIN_CLOSED_TRADES",
                        self.short_hardening_source_min_closed_trades,
                    )
                ),
            )
            self.short_hardening_source_block_net_pnl = float(
                overrides.get(
                    "SHORT_HARDENING_SOURCE_BLOCK_NET_PNL",
                    self.short_hardening_source_block_net_pnl,
                )
            )
            self.short_hardening_coin_guard_enabled = bool(
                overrides.get(
                    "SHORT_HARDENING_COIN_GUARD_ENABLED",
                    self.short_hardening_coin_guard_enabled,
                )
            )
            self.short_hardening_coin_min_closed_trades = max(
                1,
                int(
                    overrides.get(
                        "SHORT_HARDENING_COIN_MIN_CLOSED_TRADES",
                        self.short_hardening_coin_min_closed_trades,
                    )
                ),
            )
            self.short_hardening_coin_block_net_pnl = float(
                overrides.get(
                    "SHORT_HARDENING_COIN_BLOCK_NET_PNL",
                    self.short_hardening_coin_block_net_pnl,
                )
            )
            self.market_side_guard_enabled = bool(
                overrides.get(
                    "FIREWALL_MARKET_SIDE_GUARD_ENABLED",
                    self.market_side_guard_enabled,
                )
            )
            self.market_side_guard_min_confidence = float(
                overrides.get(
                    "FIREWALL_MARKET_SIDE_GUARD_MIN_CONFIDENCE",
                    self.market_side_guard_min_confidence,
                )
            )
            self.cooldown_seconds = int(
                overrides.get("FIREWALL_COIN_COOLDOWN_SECONDS", self.cooldown_seconds)
            )
            self.same_side_cooldown_seconds = int(
                overrides.get(
                    "FIREWALL_SAME_SIDE_COOLDOWN_SECONDS",
                    self.same_side_cooldown_seconds,
                )
            )
            self.max_same_side_positions_per_coin = max(
                1,
                int(
                    overrides.get(
                        "FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN",
                        self.max_same_side_positions_per_coin,
                    )
                ),
            )
            self.block_losing_averaging = bool(
                overrides.get(
                    "FIREWALL_BLOCK_LOSING_AVERAGING",
                    self.block_losing_averaging,
                )
            )
            self.averaging_max_loss_roe_pct = float(
                overrides.get(
                    "FIREWALL_AVERAGING_MAX_LOSS_ROE_PCT",
                    self.averaging_max_loss_roe_pct,
                )
            )
            self.entry_location_filter_enabled = bool(
                overrides.get(
                    "FIREWALL_ENTRY_LOCATION_FILTER_ENABLED",
                    self.entry_location_filter_enabled,
                )
            )
            self.entry_max_atr_extension = float(
                overrides.get(
                    "FIREWALL_ENTRY_MAX_ATR_EXTENSION",
                    self.entry_max_atr_extension,
                )
            )
            self.entry_max_price_extension_pct = float(
                overrides.get(
                    "FIREWALL_ENTRY_MAX_PRICE_EXTENSION_PCT",
                    self.entry_max_price_extension_pct,
                )
            )
            self.side_imbalance_guard_enabled = bool(
                overrides.get(
                    "FIREWALL_SIDE_IMBALANCE_GUARD_ENABLED",
                    self.side_imbalance_guard_enabled,
                )
            )
            self.side_imbalance_lookback_trades = max(
                10,
                int(
                    overrides.get(
                        "FIREWALL_SIDE_IMBALANCE_LOOKBACK_TRADES",
                        self.side_imbalance_lookback_trades,
                    )
                ),
            )
            self.side_imbalance_min_samples = max(
                5,
                int(
                    overrides.get(
                        "FIREWALL_SIDE_IMBALANCE_MIN_SAMPLES",
                        self.side_imbalance_min_samples,
                    )
                ),
            )
            self.side_imbalance_max_share = min(
                0.98,
                max(
                    0.50,
                    float(
                        overrides.get(
                            "FIREWALL_SIDE_IMBALANCE_MAX_SHARE",
                            self.side_imbalance_max_share,
                        )
                    ),
                ),
            )
            self.side_imbalance_confidence_bump = max(
                0.0,
                float(
                    overrides.get(
                        "FIREWALL_SIDE_IMBALANCE_CONFIDENCE_BUMP",
                        self.side_imbalance_confidence_bump,
                    )
                ),
            )
            self.side_imbalance_size_multiplier = min(
                1.0,
                max(
                    0.05,
                    float(
                        overrides.get(
                            "FIREWALL_SIDE_IMBALANCE_SIZE_MULTIPLIER",
                            self.side_imbalance_size_multiplier,
                        )
                    ),
                ),
            )
            self._side_policy_cache = {"ts": 0.0, "closed": [], "short": {}, "scoped": {}}

        logger.info(
            "DecisionFirewall runtime overrides applied: min_confidence=%s, source_cap=%s, "
            "short_hardening=%s, event_risk=%s, coin_cooldown=%ss, "
            "same_side_cooldown=%ss, block_losing_averaging=%s, entry_location_filter=%s, "
            "market_side_guard=%s",
            f"{self.min_confidence:.0%}",
            self.max_signals_per_source_per_day,
            self.short_hardening_enabled,
            self.event_risk_enabled,
            self.cooldown_seconds,
            self.same_side_cooldown_seconds,
            self.block_losing_averaging,
            self.entry_location_filter_enabled,
            self.market_side_guard_enabled,
        )

    def _get_short_policy_cache(self) -> Dict[str, object]:
        now = clock_provider.unix_now()
        cached_ts = float(self._side_policy_cache.get("ts", 0.0) or 0.0)
        if (
            self._side_policy_cache.get("short")
            and (now - cached_ts) < self._side_policy_cache_ttl_s
        ):
            return dict(self._side_policy_cache)

        try:
            closed = db.get_paper_trade_history(
                limit=self.short_hardening_lookback_trades,
                mode=db._resolve_history_mode_for_runtime(),
            )
            policy = evaluate_short_side_policy(
                closed,
                min_trades=self.short_hardening_min_closed_trades,
                degrade_win_rate=self.short_hardening_degrade_win_rate,
                block_win_rate=self.short_hardening_block_win_rate,
                block_net_pnl=self.short_hardening_block_net_pnl,
            )
        except Exception as exc:
            logger.debug("Short-side policy lookup failed: %s", exc)
            closed = []
            policy = {
                "status": "policy_error",
                "reason": str(exc),
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
            }

        self._side_policy_cache = {
            "ts": now,
            "closed": list(closed or []),
            "short": dict(policy),
            "scoped": {},
        }
        return dict(self._side_policy_cache)

    def _get_short_side_policy(self) -> Dict:
        if not self.short_hardening_enabled:
            return {
                "status": "disabled",
                "reason": "Short hardening disabled",
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
            }
        return dict((self._get_short_policy_cache().get("short") or {}))

    def _get_long_policy_cache(self) -> Dict[str, object]:
        """Mirror of _get_short_policy_cache that evaluates the long side.

        Cached on the same _side_policy_cache structure under the "long" key
        so the dual evaluation only re-fetches the trade history once per TTL.
        """
        now = clock_provider.unix_now()
        cached_ts = float(self._side_policy_cache.get("ts", 0.0) or 0.0)
        if (
            self._side_policy_cache.get("long")
            and (now - cached_ts) < self._side_policy_cache_ttl_s
        ):
            return dict(self._side_policy_cache)

        try:
            closed = db.get_paper_trade_history(
                limit=self.long_hardening_lookback_trades,
                mode=db._resolve_history_mode_for_runtime(),
            )
            policy = evaluate_long_side_policy(
                closed,
                min_trades=self.long_hardening_min_closed_trades,
                degrade_win_rate=self.long_hardening_degrade_win_rate,
                block_win_rate=self.long_hardening_block_win_rate,
                block_net_pnl=self.long_hardening_block_net_pnl,
            )
        except Exception as exc:
            logger.debug("Long-side policy lookup failed: %s", exc)
            closed = []
            policy = {
                "status": "policy_error",
                "reason": str(exc),
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
            }

        # Update only the long slot (and ``closed`` if we actually re-fetched)
        # so a same-cycle short-side fetch and long-side fetch share a single
        # round-trip when both keys land in the same TTL window.
        prev = dict(self._side_policy_cache)
        prev["ts"] = now
        if not prev.get("closed"):
            prev["closed"] = list(closed or [])
        prev["long"] = dict(policy)
        prev.setdefault("short", prev.get("short") or {})
        prev.setdefault("scoped", prev.get("scoped") or {})
        self._side_policy_cache = prev
        return dict(self._side_policy_cache)

    def _get_long_side_policy(self) -> Dict:
        if not self.long_hardening_enabled:
            return {
                "status": "disabled",
                "reason": "Long hardening disabled",
                "metrics": {"count": 0, "win_rate": 0.0, "net_pnl": 0.0},
            }
        return dict((self._get_long_policy_cache().get("long") or {}))

    def _get_scoped_short_policies(self, signal: TradeSignal) -> List[Dict]:
        if not self.short_hardening_enabled:
            return []

        cache = self._get_short_policy_cache()
        closed = list(cache.get("closed") or [])
        scoped_cache = dict(cache.get("scoped") or {})
        policies: List[Dict] = []
        source_key = self._source_key(signal)
        coin = str(getattr(signal, "coin", "") or "").strip().upper()

        def _lookup(scope_key: str, **kwargs) -> Dict:
            if scope_key in scoped_cache:
                return dict(scoped_cache[scope_key])
            policy = evaluate_side_source_policy(
                closed,
                side="short",
                degrade_win_rate=self.short_hardening_degrade_win_rate,
                block_win_rate=self.short_hardening_block_win_rate,
                **kwargs,
            )
            scoped_cache[scope_key] = dict(policy)
            self._side_policy_cache["scoped"] = scoped_cache
            return dict(policy)

        if self.short_hardening_source_guard_enabled and source_key:
            try:
                policies.append(
                    _lookup(
                        f"source:{source_key}",
                        source_key=source_key,
                        min_trades=self.short_hardening_source_min_closed_trades,
                        block_net_pnl=self.short_hardening_source_block_net_pnl,
                    )
                )
            except Exception as exc:
                logger.debug("Scoped short source policy lookup failed for %s: %s", source_key, exc)

        if self.short_hardening_coin_guard_enabled and coin:
            try:
                policies.append(
                    _lookup(
                        f"coin:{coin}",
                        coin=coin,
                        min_trades=self.short_hardening_coin_min_closed_trades,
                        block_net_pnl=self.short_hardening_coin_block_net_pnl,
                    )
                )
            except Exception as exc:
                logger.debug("Scoped short coin policy lookup failed for %s: %s", coin, exc)

        return policies

    def _current_market_side_alignment(
        self,
        signal: TradeSignal,
        side: str,
        regime_data: Optional[Dict] = None,
    ) -> Dict:
        """Return whether the requested side agrees with the current market read."""
        side = str(side or "").strip().lower()
        regime_payload = regime_data if isinstance(regime_data, dict) else {}
        context = getattr(signal, "context", {}) if isinstance(getattr(signal, "context", {}), dict) else {}
        coin = str(getattr(signal, "coin", "") or "").strip().upper()

        bearish_regimes = {
            "bear",
            "bearish",
            "downtrend",
            "trend_down",
            "trending_down",
            "crash",
            "panic",
            "risk_off",
        }
        bullish_regimes = {
            "bull",
            "bullish",
            "uptrend",
            "trend_up",
            "trending_up",
            "risk_on",
        }

        def _float(value: object, default: float = 0.0) -> float:
            try:
                out = float(value)
                return out if out == out else default
            except (TypeError, ValueError):
                return default

        def _item_to_dict(value: object) -> Dict:
            if isinstance(value, dict):
                return dict(value)
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                try:
                    return dict(to_dict() or {})
                except Exception:
                    return {}
            return {}

        candidates: List[Dict] = []
        per_coin = regime_payload.get("per_coin", {})
        if isinstance(per_coin, dict) and coin:
            coin_payload = _item_to_dict(per_coin.get(coin))
            if coin_payload:
                candidates.append(
                    {
                        "source": f"coin:{coin}",
                        "regime": str(coin_payload.get("regime", "") or "").strip().lower(),
                        "confidence": _float(
                            coin_payload.get("confidence", coin_payload.get("regime_confidence")),
                            0.0,
                        ),
                        "momentum": _float(coin_payload.get("momentum"), 0.0),
                        "trend_direction": _float(coin_payload.get("trend_direction"), 0.0),
                    }
                )

        candidates.append(
            {
                "source": "overall",
                "regime": str(
                    regime_payload.get("overall_regime", "") or regime_payload.get("regime", "") or ""
                ).strip().lower(),
                "confidence": _float(
                    regime_payload.get("overall_confidence", regime_payload.get("regime_confidence")),
                    0.0,
                ),
                "momentum": _float(regime_payload.get("momentum"), 0.0),
                "trend_direction": _float(regime_payload.get("trend_direction"), 0.0),
            }
        )

        override = regime_payload.get("global_momentum_override")
        if isinstance(override, dict):
            direction = str(override.get("direction", "") or "").strip().lower()
            if direction in {"up", "down"}:
                candidates.append(
                    {
                        "source": "global_momentum",
                        "regime": "trending_down" if direction == "down" else "trending_up",
                        "confidence": max(
                            self.short_hardening_block_override_min_regime_confidence,
                            0.65,
                        ),
                        "momentum": -self.short_hardening_market_adaptive_min_momentum
                        if direction == "down"
                        else self.short_hardening_market_adaptive_min_momentum,
                        "trend_direction": -1.0 if direction == "down" else 1.0,
                    }
                )

        forecaster_regime = str(regime_payload.get("forecaster_regime", "") or "").strip().lower()
        forecaster_conf = _float(regime_payload.get("forecaster_confidence"), 0.0)
        forecaster_synthetic = bool(regime_payload.get("forecaster_synthetic_warm_start", False))
        if forecaster_regime and not forecaster_synthetic:
            candidates.append(
                {
                    "source": "forecaster",
                    "regime": forecaster_regime,
                    "confidence": forecaster_conf,
                    "momentum": 0.0,
                    "trend_direction": 0.0,
                }
            )

        context_regime = str(context.get("regime", "") or context.get("overall_regime", "") or "").strip().lower()
        if context_regime:
            candidates.append(
                {
                    "source": "signal_context",
                    "regime": context_regime,
                    "confidence": _float(
                        context.get("regime_confidence", context.get("overall_confidence")),
                        0.0,
                    ),
                    "momentum": _float(context.get("momentum"), 0.0),
                    "trend_direction": _float(context.get("trend_direction"), 0.0),
                }
            )

        min_conf = self.short_hardening_block_override_min_regime_confidence
        min_momentum = abs(self.short_hardening_market_adaptive_min_momentum)
        best = {
            "aligned": False,
            "direction": "unknown",
            "source": "",
            "regime": "",
            "confidence": 0.0,
            "momentum": 0.0,
            "reason": "no current market alignment found",
        }

        for candidate in candidates:
            regime = str(candidate.get("regime", "") or "").strip().lower()
            confidence = _float(candidate.get("confidence"), 0.0)
            momentum = _float(candidate.get("momentum"), 0.0)
            trend_direction = _float(candidate.get("trend_direction"), 0.0)
            direction = "unknown"
            if regime in bearish_regimes or momentum <= -min_momentum or trend_direction < -min_momentum:
                direction = "short"
            elif regime in bullish_regimes or momentum >= min_momentum or trend_direction > min_momentum:
                direction = "long"

            aligned = direction == side and confidence >= min_conf
            if aligned:
                return {
                    "aligned": True,
                    "direction": direction,
                    "source": candidate.get("source", ""),
                    "regime": regime or direction,
                    "confidence": confidence,
                    "momentum": momentum,
                    "reason": (
                        f"{candidate.get('source', 'market')} confirms {direction} "
                        f"(regime={regime or 'momentum'}, confidence={confidence:.0%})"
                    ),
                }
            if confidence > float(best.get("confidence", 0.0) or 0.0):
                best = {
                    "aligned": False,
                    "direction": direction,
                    "source": candidate.get("source", ""),
                    "regime": regime,
                    "confidence": confidence,
                    "momentum": momentum,
                    "reason": (
                        f"best current read is {direction or 'unknown'} from "
                        f"{candidate.get('source', 'market')} "
                        f"(regime={regime or 'unknown'}, confidence={confidence:.0%})"
                    ),
                }

        return best

    def _short_block_override(
        self,
        signal: TradeSignal,
        blocking_policies: List[Dict],
        regime_data: Optional[Dict] = None,
    ) -> Tuple[bool, str, Dict]:
        """Allow strong regime-aligned shorts through a global-only block."""
        if not self.short_hardening_block_override_enabled:
            return False, "short block override disabled", {}

        scoped_block = next(
            (
                policy for policy in blocking_policies
                if policy.get("scope")
                or policy.get("coin")
                or str(policy.get("source", "") or "").strip().lower() not in {"", "all"}
            ),
            None,
        )

        try:
            confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self.short_hardening_block_override_min_confidence:
            return (
                False,
                f"short confidence {confidence:.0%} below override threshold "
                f"{self.short_hardening_block_override_min_confidence:.0%}",
                {},
            )

        alignment = self._current_market_side_alignment(
            signal,
            "short",
            regime_data=regime_data,
        )
        if not alignment.get("aligned"):
            if scoped_block:
                return False, scoped_block.get("reason", "Scoped short-side policy is blocked"), {}
            return False, f"short block override requires current bearish alignment: {alignment.get('reason')}", {}

        if scoped_block and not self.short_hardening_market_adaptive_override_enabled:
            return False, scoped_block.get("reason", "Scoped short-side policy is blocked"), {}

        size_multiplier = self.short_hardening_block_override_size_multiplier
        if scoped_block:
            size_multiplier = min(
                size_multiplier,
                self.short_hardening_market_adaptive_scoped_size_multiplier,
            )

        meta = {
            "status": "override",
            "regime": alignment.get("regime", ""),
            "regime_confidence": alignment.get("confidence", 0.0),
            "confidence": confidence,
            "size_multiplier": size_multiplier,
            "scope": "scoped" if scoped_block else "global",
            "market_alignment": dict(alignment),
            "reason": (
                "Short-side history block overridden by current bearish market alignment"
                if scoped_block
                else "Global short-side block overridden by current bearish market alignment"
            ),
        }
        return True, meta["reason"], meta

    def _apply_market_side_guard(
        self,
        signal: TradeSignal,
        side: str,
        regime_data: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Block entries that fight a strong current market-side read."""
        if not self.market_side_guard_enabled:
            return True, ""

        side = str(side or "").strip().lower()
        if side not in {"long", "short"}:
            return True, ""

        alignment = self._current_market_side_alignment(signal, side, regime_data=regime_data)
        if isinstance(getattr(signal, "context", None), dict):
            signal.context["market_side_alignment"] = dict(alignment)

        if alignment.get("aligned"):
            return True, ""

        market_side = str(alignment.get("direction", "") or "").strip().lower()
        try:
            market_conf = float(alignment.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            market_conf = 0.0

        opposite = "short" if side == "long" else "long"
        if market_side == opposite and market_conf >= self.market_side_guard_min_confidence:
            return (
                False,
                f"Current market read blocks {side}: {alignment.get('reason')}",
            )
        return True, ""

    def _apply_long_hardening(self, signal: TradeSignal) -> Tuple[bool, str]:
        """Mirror of the short_hardening core gate against the long side.

        Same block / degrade thresholds as the short side. We don't (yet)
        replicate the short_hardening_block_override / market_adaptive /
        scoped source+coin variants here — the core gate alone closes the
        single biggest asymmetry that produces 88-90% long executions.
        Operators can re-enable longs entirely via LONG_HARDENING_ENABLED=
        false if they have an outside reason to allow losing longs.
        """
        if not self.long_hardening_enabled:
            return True, ""

        policy = self._get_long_side_policy()
        status = str(policy.get("status", "") or "").strip().lower()

        if status == "blocked":
            if isinstance(getattr(signal, "context", None), dict):
                signal.context["long_side_policy"] = {
                    "status": status,
                    "reason": policy.get("reason"),
                    "metrics": policy.get("metrics", {}),
                }
            return False, policy.get("reason", "Long-side guardrail blocked the signal")

        if status == "degraded":
            original_confidence = float(signal.confidence)
            signal.confidence *= self.long_hardening_confidence_multiplier
            signal.position_pct *= self.long_hardening_size_multiplier
            if signal.size > 0:
                signal.size *= self.long_hardening_size_multiplier
            if isinstance(getattr(signal, "context", None), dict):
                signal.context["long_side_policy"] = {
                    "status": status,
                    "reason": policy.get("reason"),
                    "metrics": policy.get("metrics", {}),
                }
            logger.warning(
                "Long hardening de-risked %s: confidence %.0f%% -> %.0f%%, "
                "size *= %.2f (%s)",
                signal.coin,
                original_confidence * 100,
                signal.confidence * 100,
                self.long_hardening_size_multiplier,
                policy.get("reason", "recent long underperformance"),
            )

        return True, ""

    def _apply_side_policy(
        self,
        signal: TradeSignal,
        regime_data: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        side_val = signal.side.value if hasattr(signal.side, "value") else str(signal.side)
        side_val = str(side_val or "").strip().lower()

        market_ok, market_reason = self._apply_market_side_guard(
            signal,
            side_val,
            regime_data=regime_data,
        )
        if not market_ok:
            return False, market_reason

        if side_val == "long":
            return self._apply_long_hardening(signal)

        if side_val != "short":
            return True, ""

        policies = [self._get_short_side_policy(), *self._get_scoped_short_policies(signal)]
        blocking_policies = [
            policy for policy in policies
            if str(policy.get("status", "")).lower() == "blocked"
        ]
        if blocking_policies:
            override_allowed, override_reason, override_meta = self._short_block_override(
                signal,
                blocking_policies,
                regime_data=regime_data,
            )
            if not override_allowed:
                blocking = blocking_policies[0]
                return False, blocking.get("reason", "Short-side guardrail blocked the signal")

            original_confidence = float(signal.confidence)
            size_multiplier = float(
                override_meta.get(
                    "size_multiplier",
                    self.short_hardening_block_override_size_multiplier,
                )
                or self.short_hardening_block_override_size_multiplier
            )
            signal.confidence *= self.short_hardening_confidence_multiplier
            signal.position_pct *= size_multiplier
            if signal.size > 0:
                signal.size *= size_multiplier
            if isinstance(getattr(signal, "context", None), dict):
                signal.context["short_side_policies"] = [
                    {
                        "status": p.get("status"),
                        "reason": p.get("reason"),
                        "metrics": p.get("metrics", {}),
                        "scope": p.get("scope", "global_short"),
                    }
                    for p in policies
                ]
                signal.context["short_side_policy_override"] = dict(override_meta)
            logger.warning(
                "Short hardening override allowed %s: confidence %.0f%% -> %.0f%%, "
                "size *= %.2f (%s)",
                signal.coin,
                original_confidence * 100,
                signal.confidence * 100,
                size_multiplier,
                override_reason,
            )
            return True, ""

        degraded_policies = [
            policy for policy in policies
            if str(policy.get("status", "")).lower() == "degraded"
        ]
        if degraded_policies:
            policy = degraded_policies[0]
            original_confidence = float(signal.confidence)
            signal.confidence *= self.short_hardening_confidence_multiplier
            signal.position_pct *= self.short_hardening_size_multiplier
            if signal.size > 0:
                signal.size *= self.short_hardening_size_multiplier
            if isinstance(getattr(signal, "context", None), dict):
                signal.context["short_side_policies"] = [
                    {
                        "status": p.get("status"),
                        "reason": p.get("reason"),
                        "metrics": p.get("metrics", {}),
                        "scope": p.get("scope", "global_short"),
                    }
                    for p in policies
                ]
            logger.warning(
                "Short hardening de-risked %s: confidence %.0f%% -> %.0f%%, size *= %.2f (%s)",
                signal.coin,
                original_confidence * 100,
                signal.confidence * 100,
                self.short_hardening_size_multiplier,
                policy.get("reason", "recent short underperformance"),
            )
        return True, ""

    @staticmethod
    def _float_or_none(value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            out = float(value)
            return out if out == out else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _signal_context(signal: TradeSignal) -> Dict:
        ctx = getattr(signal, "context", None)
        return dict(ctx) if isinstance(ctx, dict) else {}

    def _signal_reference_price(self, signal: TradeSignal) -> Optional[float]:
        ctx = self._signal_context(signal)
        features = ctx.get("features", {})
        if not isinstance(features, dict):
            features = {}
        for key in (
            "entry_price",
            "price",
            "current_price",
            "mid_price",
            "mark_price",
            "last_price",
        ):
            value = self._float_or_none(getattr(signal, key, None))
            if value and value > 0:
                return value
            value = self._float_or_none(ctx.get(key))
            if value and value > 0:
                return value
            value = self._float_or_none(features.get(key))
            if value and value > 0:
                return value
        return None

    def _same_side_position_loss_roe(
        self,
        pos: Dict,
        side_value: str,
        current_price: Optional[float],
        signal_leverage: float,
    ) -> Optional[float]:
        if not current_price or current_price <= 0:
            return None
        entry = (
            self._float_or_none(pos.get("entry_price"))
            or self._float_or_none(pos.get("entryPx"))
            or self._float_or_none(pos.get("avg_entry_price"))
            or self._float_or_none(pos.get("average_entry_price"))
        )
        if not entry or entry <= 0:
            return None
        leverage = (
            self._float_or_none(pos.get("leverage"))
            or self._float_or_none(pos.get("lev"))
            or self._float_or_none(signal_leverage)
            or 1.0
        )
        if side_value == "long":
            return max(0.0, (entry - current_price) / entry * leverage)
        if side_value == "short":
            return max(0.0, (current_price - entry) / entry * leverage)
        return None

    def _apply_losing_averaging_guard(
        self,
        signal: TradeSignal,
        same_side_positions: List[Dict],
        side_value: str,
    ) -> Tuple[bool, str]:
        if not self.block_losing_averaging or not same_side_positions:
            return True, ""
        ctx = self._signal_context(signal)
        if any(
            bool(ctx.get(flag))
            for flag in (
                "scale_in_allowed",
                "averaging_allowed",
                "new_information",
                "thesis_improved",
            )
        ):
            return True, ""

        current_price = self._signal_reference_price(signal)
        losses = [
            loss
            for loss in (
                self._same_side_position_loss_roe(
                    pos,
                    side_value,
                    current_price,
                    float(getattr(signal, "leverage", 1.0) or 1.0),
                )
                for pos in same_side_positions
            )
            if loss is not None
        ]
        if not losses:
            return True, ""
        max_loss = max(losses)
        if max_loss >= self.averaging_max_loss_roe_pct:
            return (
                False,
                f"Losing average-down blocked: existing {signal.coin} {side_value} "
                f"is down {max_loss:.2%} ROE without new-information override "
                f"(limit {self.averaging_max_loss_roe_pct:.2%})",
            )
        return True, ""

    @staticmethod
    def _normalize_side(side: object) -> str:
        value = str(side or "").strip().lower()
        if value in {"buy", "b", "long"}:
            return "long"
        if value in {"sell", "s", "a", "ask", "short"}:
            return "short"
        return value

    def _apply_side_imbalance_guard(
        self,
        signal: TradeSignal,
        side_value: str,
        positions: List[Dict],
    ) -> Tuple[bool, str]:
        """Avoid runaway same-side books without forcing junk countertrades.

        If the recent paper book is already overwhelmingly one-sided, new
        same-side entries must clear a higher confidence bar. Strong signals
        can still pass, but they get size-derisked so the bot does not keep
        compounding a long-only or short-only tape.
        """
        if not self.side_imbalance_guard_enabled:
            return True, ""

        side_value = self._normalize_side(side_value)
        if side_value not in {"long", "short"}:
            return True, ""

        sides: List[str] = []
        for pos in positions or []:
            side = self._normalize_side(pos.get("side"))
            if side in {"long", "short"}:
                sides.append(side)
        try:
            recent = db.get_paper_trade_history(
                limit=self.side_imbalance_lookback_trades,
                mode=db._resolve_history_mode_for_runtime(),
            )
        except Exception as exc:
            logger.debug("Side imbalance history lookup failed: %s", exc)
            recent = []
        for trade in recent or []:
            side = self._normalize_side(trade.get("side"))
            if side in {"long", "short"}:
                sides.append(side)

        total = len(sides)
        if total < self.side_imbalance_min_samples:
            return True, ""
        same = sides.count(side_value)
        share = same / total if total else 0.0
        if share < self.side_imbalance_max_share:
            return True, ""

        required_confidence = min(
            0.95,
            float(self.min_confidence or 0.0) + self.side_imbalance_confidence_bump,
        )
        if float(getattr(signal, "confidence", 0.0) or 0.0) < required_confidence:
            return (
                False,
                f"Side imbalance guard: {same}/{total} recent/open trades are {side_value}; "
                f"requires {required_confidence:.0%} confidence for another {side_value} "
                f"(got {signal.confidence:.0%})",
            )

        if self.side_imbalance_size_multiplier < 1.0:
            signal.position_pct *= self.side_imbalance_size_multiplier
            if signal.size > 0:
                signal.size *= self.side_imbalance_size_multiplier
            logger.info(
                "Side imbalance derisked %s %s: share=%.0f%% size*=%.2f",
                signal.coin,
                side_value,
                share * 100,
                self.side_imbalance_size_multiplier,
            )
        return True, ""

    def _apply_entry_location_filter(
        self,
        signal: TradeSignal,
        regime_data: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        if not self.entry_location_filter_enabled:
            return True, ""

        side_value = (
            signal.side.value if hasattr(signal.side, "value") else str(signal.side)
        ).strip().lower()
        ctx = self._signal_context(signal)
        features = ctx.get("features", {})
        if not isinstance(features, dict):
            features = {}
        coin_regime = {}
        if regime_data and isinstance(regime_data.get("per_coin"), dict):
            coin_regime = dict(regime_data.get("per_coin", {}).get(signal.coin, {}) or {})

        atr_pct = (
            self._float_or_none(ctx.get("atr_pct"))
            or self._float_or_none(features.get("atr_pct"))
            or self._float_or_none(coin_regime.get("atr_pct"))
            or self._float_or_none(ctx.get("volatility"))
            or 0.0
        )
        threshold = max(
            float(self.entry_max_price_extension_pct or 0.0),
            float(atr_pct or 0.0) * float(self.entry_max_atr_extension or 0.0),
        )
        if threshold <= 0:
            return True, ""

        distances = []
        for key in (
            "distance_from_vwap_pct",
            "vwap_distance_pct",
            "distance_from_ma_pct",
            "ma_distance_pct",
            "ema_distance_pct",
            "price_extension_pct",
        ):
            value = self._float_or_none(ctx.get(key))
            if value is None:
                value = self._float_or_none(features.get(key))
            if value is not None:
                distances.append((key, value))

        current_price = self._signal_reference_price(signal)
        if current_price and current_price > 0:
            for key in ("vwap", "moving_average", "ma", "ema", "sma"):
                anchor = self._float_or_none(ctx.get(key))
                if anchor is None:
                    anchor = self._float_or_none(features.get(key))
                if anchor and anchor > 0:
                    distances.append((f"price_vs_{key}", (current_price - anchor) / anchor))
                    break

        zscore = self._float_or_none(ctx.get("zscore"))
        if zscore is None:
            zscore = self._float_or_none(features.get("zscore"))
        if zscore is not None:
            z_limit = max(1.0, float(self.entry_max_atr_extension or 0.0))
            if side_value == "long" and zscore > z_limit:
                return False, f"Entry too extended for long: zscore={zscore:.2f} > {z_limit:.2f}"
            if side_value == "short" and zscore < -z_limit:
                return False, f"Entry too extended for short: zscore={zscore:.2f} < -{z_limit:.2f}"

        for key, distance in distances:
            if side_value == "long" and distance > threshold:
                return (
                    False,
                    f"Entry too extended for long: {key}={distance:.2%} "
                    f"> {threshold:.2%}",
                )
            if side_value == "short" and distance < -threshold:
                return (
                    False,
                    f"Entry too extended for short: {key}={distance:.2%} "
                    f"< -{threshold:.2%}",
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
        # Calibration-quality gate. A source above the auto-quarantine
        # ECE bar is treated as paused for live entries — its outcomes
        # still flow into the calibrator (paper continues), so it can
        # rejoin once the calibration recovers.
        if self.calibration is not None:
            try:
                side_bucket = (signal.side or "").lower() if hasattr(signal, "side") else None
                regime_bucket = None
                if isinstance(getattr(signal, "regime", None), str):
                    from src.signals.calibration import bucket_regime
                    regime_bucket = bucket_regime(signal.regime)
                if self.calibration.is_quarantined(
                    source_key, side=side_bucket, regime=regime_bucket
                ):
                    return False, (
                        f"Calibration quarantine for {source_key} "
                        f"(side={side_bucket or 'any'}, regime={regime_bucket or 'any'}): "
                        "ECE above auto-quarantine threshold"
                    ), {
                        "source_key": source_key,
                        "status": "calibration_quarantine",
                        "max_signals_per_day": 0,
                        "size_multiplier": 0.0,
                    }
            except Exception as exc:
                logger.debug("Calibration quarantine check failed for %s: %s", source_key, exc)

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
        if signal.confidence + self.confidence_threshold_tolerance < min_conf:
            # Surface confidence_inputs (if any) so operators can diagnose a
            # stuck-rejection pattern without having to grep the source code.
            # Copy-trader signals carry these via signal.context.confidence_inputs.
            ctx = getattr(signal, "context", None) or {}
            ci = ctx.get("confidence_inputs") if isinstance(ctx, dict) else None
            extra = ""
            if isinstance(ci, dict) and ci:
                stype = ci.get("signal_type", "?")
                wr = ci.get("win_rate")
                tc = ci.get("trade_count")
                if wr is not None:
                    extra = f" [signal_type={stype}, trader_win_rate={float(wr):.2%}, trades={tc}]"
            return (
                False,
                f"Source allocator requires {min_conf:.0%} confidence for {source_key} "
                f"(got {signal.confidence:.0%}){extra}",
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

    @staticmethod
    def _resolve_signal_size_units(signal: TradeSignal, balance: Optional[float]) -> float:
        """Resolve a signal's unit size from balance/position_pct when possible."""
        current_size = float(getattr(signal, "size", 0.0) or 0.0)
        if current_size > 0:
            return current_size

        entry_price = float(getattr(signal, "entry_price", 0.0) or 0.0)
        position_pct = float(getattr(signal, "position_pct", 0.0) or 0.0)
        if balance and balance > 0 and position_pct > 0 and entry_price > 0:
            resolved_size = (balance * position_pct) / entry_price
            if resolved_size > 0:
                signal.size = resolved_size
                return resolved_size
        return 0.0

    @classmethod
    def _estimate_signal_notional(cls, signal: TradeSignal, balance: Optional[float]) -> float:
        """
        Estimate a signal's *leveraged* notional exposure even before size is resolved.

        AUDIT M1 — Despite the name, this returns ``size × price × leverage``
        rather than pure notional (``size × price``).  The firewall's
        ``max_aggregate_exposure_pct`` cap has historically been measured
        against this leveraged-notional metric (it's effectively "market
        exposure × leverage"), and changing the metric here would silently
        loosen that cap.  For capital-at-risk (pure margin) math use
        :meth:`_estimate_signal_margin`.
        """
        size = cls._resolve_signal_size_units(signal, balance)
        entry_price = float(getattr(signal, "entry_price", 0.0) or 0.0)
        leverage = max(float(getattr(signal, "leverage", 1.0) or 1.0), 1.0)
        if size > 0 and entry_price > 0:
            return abs(size * entry_price * leverage)

        position_pct = float(getattr(signal, "position_pct", 0.0) or 0.0)
        if balance and balance > 0 and position_pct > 0:
            return abs(balance * position_pct * leverage)
        return 0.0

    @classmethod
    def _estimate_signal_margin(cls, signal: TradeSignal, balance: Optional[float]) -> float:
        """
        Estimate a signal's *margin requirement* — the capital actually locked
        up by the position (``notional / leverage``).

        AUDIT M1 — added as a leverage-agnostic companion to
        :meth:`_estimate_signal_notional`.  The margin cap check uses this so
        a 5x-leveraged trade with $200 margin counts as $200 against the
        margin budget (not $1000 the way leveraged-notional would count it).
        """
        size = cls._resolve_signal_size_units(signal, balance)
        entry_price = float(getattr(signal, "entry_price", 0.0) or 0.0)
        leverage = max(float(getattr(signal, "leverage", 1.0) or 1.0), 1.0)
        if size > 0 and entry_price > 0:
            return abs(size * entry_price) / leverage

        # Fallback: position_pct is the fraction of balance earmarked as
        # margin for the trade, so ``balance × position_pct`` is margin.
        position_pct = float(getattr(signal, "position_pct", 0.0) or 0.0)
        if balance and balance > 0 and position_pct > 0:
            return abs(balance * position_pct)
        return 0.0

    @staticmethod
    def _position_margin(pos: Dict) -> float:
        """Extract/estimate margin for an open position for margin-cap aggregation."""
        try:
            projected_margin = float(pos.get("projected_margin", 0) or 0)
        except (TypeError, ValueError):
            projected_margin = 0.0
        if projected_margin > 0:
            return projected_margin

        try:
            size = float(pos.get("size", 0) or 0)
            price = float(pos.get("entry_price", pos.get("entryPx", 0)) or 0)
            leverage = float(pos.get("leverage", 1) or 1) or 1.0
        except (TypeError, ValueError):
            return 0.0
        if size <= 0 or price <= 0:
            return 0.0
        return abs(size * price) / max(leverage, 1.0)

    @staticmethod
    def _journal_record(signal: TradeSignal, **kwargs) -> None:
        """Best-effort decision journaling; never blocks validation."""
        try:
            from src.data import decision_journal

            decision_journal.record_decision_snapshot(signal, **kwargs)
        except Exception:
            logger.debug("Decision journal record failed", exc_info=True)

    @staticmethod
    def _journal_update(signal: TradeSignal, **kwargs) -> None:
        """Best-effort decision status update; never blocks validation."""
        try:
            from src.data import decision_journal

            decision_id = decision_journal.resolve_decision_id(signal)
            if decision_id:
                decision_journal.update_decision_status(decision_id, **kwargs)
        except Exception:
            logger.debug("Decision journal update failed", exc_info=True)

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
            self._journal_record(
                signal,
                regime_data=regime_data,
                account_balance=account_balance,
                final_status="firewall_validation",
                firewall_decision="pending",
                metadata={
                    "dry_run": dry_run,
                    "ignore_position_limit": ignore_position_limit,
                },
            )

        if not dry_run:
            self.stats["total_signals"] += 1

        def _reject(reason_key, reason_msg):
            """Helper to reject + audit log in one step."""
            if not dry_run:
                self._journal_update(
                    signal,
                    final_status="rejected",
                    firewall_decision="rejected",
                    rejection_reason=reason_msg,
                    metadata={
                        "reason_key": reason_key,
                        "dry_run": dry_run,
                        "market_side_alignment": (
                            dict(signal.context.get("market_side_alignment", {}))
                            if isinstance(getattr(signal, "context", None), dict)
                            else {}
                        ),
                    },
                )
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

        side_policy_ok, side_policy_reason = self._apply_side_policy(signal, regime_data=regime_data)
        if not side_policy_ok:
            return _reject("rejected_side_policy", side_policy_reason)

        entry_location_ok, entry_location_reason = self._apply_entry_location_filter(
            signal,
            regime_data=regime_data,
        )
        if not entry_location_ok:
            return _reject("rejected_entry_location", entry_location_reason)

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

        side_value = (
            signal.side.value if hasattr(signal.side, "value") else str(signal.side)
        ).strip().lower()

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
        same_side_positions = [
            p for p in coin_positions
            if str(p.get("side", "") or "").strip().lower() == side_value
        ]
        if len(same_side_positions) >= self.max_same_side_positions_per_coin:
            return _reject(
                "rejected_pyramiding",
                f"Pyramiding blocked: {signal.coin} already has "
                f"{len(same_side_positions)} {side_value} positions",
            )
        averaging_ok, averaging_reason = self._apply_losing_averaging_guard(
            signal,
            same_side_positions,
            side_value,
        )
        if not averaging_ok:
            return _reject("rejected_pyramiding", averaging_reason)

        # 5b. Aggregate portfolio exposure — hard cap across ALL positions
        # AUDIT M1 — two independent caps apply:
        #   (1) max_aggregate_exposure_pct against *leveraged notional*
        #       (historical behavior — scales up with leverage so it doubles
        #       as a soft leverage-concentration guard).
        #   (2) max_aggregate_margin_pct against *margin actually locked*
        #       (leverage-agnostic capital-at-risk view).
        # A signal must pass BOTH to be approved.
        side_imbalance_ok, side_imbalance_reason = self._apply_side_imbalance_guard(
            signal,
            side_value,
            positions,
        )
        if not side_imbalance_ok:
            return _reject("rejected_side_imbalance", side_imbalance_reason)

        balance = account_balance
        if balance is None:
            account = db.get_paper_account()
            balance = account.get("balance", 10000) if account else None
        if balance:
            total_exposure = 0.0
            total_margin = 0.0
            for pos in positions:
                projected_notional = float(pos.get("projected_notional", 0) or 0)
                if projected_notional > 0:
                    total_exposure += abs(projected_notional)
                else:
                    pos_size = pos.get("size", 0)
                    pos_price = pos.get("entry_price", pos.get("entryPx", 0))
                    pos_leverage = pos.get("leverage", 1)
                    total_exposure += abs(pos_size * pos_price * pos_leverage)

                total_margin += self._position_margin(pos)

            new_notional = self._estimate_signal_notional(signal, balance)
            if new_notional <= 0:
                return _reject(
                    "rejected_exposure",
                    "Signal size unresolved; cannot validate aggregate exposure",
                )
            new_margin = self._estimate_signal_margin(signal, balance)

            projected_exposure = total_exposure + new_notional
            exposure_pct = projected_exposure / balance if balance > 0 else 1.0

            if exposure_pct > self.max_aggregate_exposure_pct:
                return _reject("rejected_exposure",
                              f"Aggregate exposure {exposure_pct:.0%} would exceed "
                              f"{self.max_aggregate_exposure_pct:.0%} limit "
                              f"(${projected_exposure:,.0f}/${balance:,.0f})")

            # Separate margin-based cap (AUDIT M1).  Disabled when
            # max_aggregate_margin_pct <= 0 so operators can opt out.
            if self.max_aggregate_margin_pct > 0:
                projected_margin = total_margin + new_margin
                margin_pct = projected_margin / balance if balance > 0 else 1.0
                if margin_pct > self.max_aggregate_margin_pct:
                    return _reject(
                        "rejected_exposure",
                        f"Aggregate margin {margin_pct:.0%} would exceed "
                        f"{self.max_aggregate_margin_pct:.0%} limit "
                        f"(${projected_margin:,.0f}/${balance:,.0f})",
                    )

        # 6. Conflict detection — no opposing positions on same coin
        for pos in coin_positions:
            if str(pos.get("side", "") or "").strip().lower() != side_value:
                return _reject("rejected_conflict",
                              f"Conflict: have {pos.get('side')} {signal.coin}, "
                              f"signal wants {side_value}")

        # 7. Cooldown — prevent revenge trading
        now = clock_provider.unix_now()
        last_trade_ts = self._recent_trades.get(signal.coin, 0)
        if now - last_trade_ts < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (now - last_trade_ts))
            return _reject("rejected_cooldown",
                          f"Cooldown: {signal.coin} traded {remaining}s ago")
        last_same_side_ts = self._recent_side_trades.get((signal.coin, side_value), 0)
        if now - last_same_side_ts < self.same_side_cooldown_seconds:
            remaining = int(self.same_side_cooldown_seconds - (now - last_same_side_ts))
            return _reject(
                "rejected_pyramiding",
                f"Pyramiding cooldown: {signal.coin} {side_value} traded {remaining}s ago",
            )

        # 8. Regime alignment check
        if regime_data:
            guidance = regime_data.get("strategy_guidance", {})
            paused = set(guidance.get("pause", []))
            # Only enforce regime-driven pauses when the regime call is
            # confident. A low-confidence "ranging" read shouldn't kill all
            # momentum strategies — that produces the 5/5-rejected gridlock.
            try:
                regime_conf = float(
                    regime_data.get("overall_confidence",
                        regime_data.get("regime_confidence", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                regime_conf = 0.0
            if regime_conf >= self.regime_pause_min_confidence:
                if "all" in paused:
                    return _reject("rejected_regime",
                                  f"Regime {regime_data.get('overall_regime', '?')} pauses all trading "
                                  f"(conf {regime_conf:.0%})")
                if signal.strategy_type and signal.strategy_type.lower() in paused:
                    return _reject("rejected_regime",
                                  f"Regime pauses {signal.strategy_type} "
                                  f"(regime={regime_data.get('overall_regime', '?')}, conf {regime_conf:.0%})")
            elif paused and (
                "all" in paused
                or (signal.strategy_type and signal.strategy_type.lower() in paused)
            ):
                # Low-confidence regime: don't block, just leave a breadcrumb
                # so we can see in logs that a pause was suppressed.
                logger.info(
                    "Regime pause suppressed for %s (regime=%s, conf %.0f%% < %.0f%%)",
                    signal.strategy_type or "?",
                    regime_data.get("overall_regime", "?"),
                    regime_conf * 100,
                    self.regime_pause_min_confidence * 100,
                )

            # Apply size modifier from regime
            size_mod = float(guidance.get("size_modifier", 1.0) or 1.0)
            signal.regime_size_modifier = size_mod
            if size_mod != 1.0:
                if getattr(signal, "size", None):
                    signal.size *= size_mod
                if getattr(signal, "position_pct", None):
                    signal.position_pct *= size_mod

            # Apply macro regime confidence drag (protective overlay)
            macro_conf_drag = float(regime_data.get("macro_confidence_drag", 0.0) or 0.0)
            if macro_conf_drag < 0 and signal.confidence > 0:
                signal.confidence = max(0.05, signal.confidence + macro_conf_drag)

            # Block new entries if macro regime says so
            if regime_data.get("macro_block_new_entries"):
                macro_reasons = regime_data.get("macro_reasons", [])
                reason_str = macro_reasons[0] if macro_reasons else "extreme macro risk"
                return _reject("rejected_macro_regime",
                              f"Macro regime blocks new entries: {reason_str}")

            countertrend_block_side = str(
                regime_data.get("countertrend_block_side", "") or ""
            ).strip().lower()
            if countertrend_block_side and side_value == countertrend_block_side:
                return _reject(
                    "rejected_regime",
                    f"Regime disagreement blocks countertrend {side_value} entries "
                    f"(detector={regime_data.get('detector_regime', regime_data.get('overall_regime', '?'))}, "
                    f"forecaster={regime_data.get('forecaster_regime', '?')})",
                )

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
                    if getattr(signal, "size", None):
                        signal.size *= self.crash_size_multiplier  # 80% reduction
                    if getattr(signal, "position_pct", None):
                        signal.position_pct *= self.crash_size_multiplier
                    logger.warning(
                        f"CRASH REGIME detected for {signal.coin} "
                        f"(conf={predictive_regime['confidence']:.2f}) -- "
                        f"de-risking: size *= {self.crash_size_multiplier}"
                    )
            except Exception as e:
                logger.debug(f"Forecaster check failed: {e}")

        # All checks passed
        if not dry_run:
            self.stats["passed"] += 1
            self._recent_trades[signal.coin] = now
            self._recent_side_trades[(signal.coin, side_value)] = now
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

        if not dry_run:
            self._journal_update(
                signal,
                final_status="approved",
                firewall_decision="approved",
                rejection_reason=None,
                metadata={
                    "dry_run": dry_run,
                    "market_side_alignment": (
                        dict(signal.context.get("market_side_alignment", {}))
                        if isinstance(getattr(signal, "context", None), dict)
                        else {}
                    ),
                },
            )

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
            account = db.get_paper_account()
            balance = account.get("balance", 10000) if account else None

            # Sort by confidence descending
            sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

            results = []
            for signal in sorted_signals:
                passed, reason = self._validate_locked(
                    signal,
                    regime_data,
                    positions,
                    account_balance=balance,
                )
                results.append((signal, passed, reason))

                # If signal passed, add to positions for subsequent checks.
                # CRIT-FIX CRIT-3: include size/entry_price/leverage so the aggregate
                # exposure check inside _validate_locked sees real notional for within-batch
                # approvals — without these fields pos.get("size", 0) returns 0 and a burst
                # of concurrent signals can all pass the exposure cap simultaneously.
                if passed:
                    resolved_size = self._resolve_signal_size_units(signal, balance)
                    positions.append({
                        "coin": signal.coin,
                        "side": signal.side.value,
                        "status": "open",
                        "entry_price": float(getattr(signal, "entry_price", 0) or 0),
                        "size": float(resolved_size or 0),
                        "leverage": float(getattr(signal, "leverage", 1) or 1),
                        "projected_notional": self._estimate_signal_notional(signal, balance),
                        # AUDIT M1 — carry projected_margin too so the
                        # margin-based cap sees within-batch accumulated
                        # margin for burst signal sequences.
                        "projected_margin": self._estimate_signal_margin(signal, balance),
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
        now = clock_provider.unix_now()
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
        today = clock_provider.utc_now().strftime("%Y-%m-%d")
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
            "coin_cooldown_seconds": int(self.cooldown_seconds),
            "same_side_cooldown_seconds": int(self.same_side_cooldown_seconds),
            "max_same_side_positions_per_coin": int(self.max_same_side_positions_per_coin),
            "block_losing_averaging": bool(self.block_losing_averaging),
            "averaging_max_loss_roe_pct": float(self.averaging_max_loss_roe_pct),
            "entry_location_filter_enabled": bool(self.entry_location_filter_enabled),
            "entry_max_atr_extension": float(self.entry_max_atr_extension),
            "entry_max_price_extension_pct": float(self.entry_max_price_extension_pct),
            "max_signals_per_source_per_day": int(self.max_signals_per_source_per_day),
            "source_signal_counts": dict(self._source_signal_counts),
            "source_policies": self.agent_scorer.get_scorecard() if self.agent_scorer else [],
            "short_side_policy": self._get_short_side_policy(),
            "market_side_guard_enabled": bool(self.market_side_guard_enabled),
            "market_side_guard_min_confidence": float(self.market_side_guard_min_confidence),
        }
