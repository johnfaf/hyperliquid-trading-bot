"""
Subsystem Registry
==================
Instantiates every subsystem and wires them together.  Each subsystem is
registered with the health registry on creation so the rest of the bot can
query ``is_trading_safe("firewall")`` etc.

Extracted from the ~300-line ``HyperliquidResearchBot.__init__`` so that:
* Each subsystem's init is isolated and can fail independently
* Health status is set atomically (HEALTHY / DEGRADED / FAILED)
* The "fundable core" profile can skip optional subsystems entirely

Usage::

    from src.core.subsystem_registry import SubsystemContainer, build_subsystems
    container = build_subsystems(health_registry, feature_profile)
    # container.firewall, container.paper_trader, etc.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Any

from src.core.health_registry import SubsystemHealthRegistry, SubsystemState

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Feature profiles
# ---------------------------------------------------------------------------
FUNDABLE_CORE = {
    "discovery", "strategy_identifier", "strategy_scorer", "decision_firewall",
    "paper_trader", "golden_wallet", "backtester", "database", "reporter",
    "regime_detector", "feature_engine", "agent_scorer", "portfolio_sizer",
}

FULL_PROFILE = FUNDABLE_CORE | {
    "copy_trader", "live_trader", "options_flow", "polymarket",
    "predictive_forecaster", "xgboost_forecaster", "multi_scanner",
    "liquidation_strategy", "kelly_sizer", "portfolio_sizer", "trade_memory", "calibration",
    "llm_filter", "signal_processor", "arena_incubator", "decision_engine",
    "alpha_arena", "adaptive_learning", "execution_policy", "position_monitor", "dashboard", "telegram",
    "source_allocator", "divergence_controller", "capital_governor",
    "cross_venue_hedger", "shadow_tracker", "adaptive_bot_detector",
    "regime_strategy_filter", "exchange_aggregator",
}


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class SubsystemContainer:
    """Bag of all subsystem instances — replaces ``self.xxx`` on the old bot."""
    # Core (always present)
    discovery: Any = None
    identifier: Any = None
    scorer: Any = None
    firewall: Any = None
    paper_trader: Any = None
    reporter: Any = None
    regime_detector: Any = None
    feature_engine: Any = None
    agent_scorer: Any = None
    exchange_agg: Any = None

    # Trading
    copy_trader: Any = None
    live_trader: Any = None
    position_monitor: Any = None

    # Signals & analysis
    options_scanner: Any = None
    polymarket: Any = None
    predictive_forecaster: Any = None
    multi_scanner: Any = None
    liquidation_strategy: Any = None
    kelly_sizer: Any = None
    portfolio_sizer: Any = None
    trade_memory: Any = None
    calibration: Any = None
    adaptive_learning: Any = None
    execution_policy: Any = None
    source_allocator: Any = None
    divergence_controller: Any = None
    capital_governor: Any = None
    llm_filter: Any = None
    signal_processor: Any = None
    arena_incubator: Any = None
    decision_engine: Any = None
    arena: Any = None
    adaptive_bot_detector: Any = None
    regime_strategy_filter: Any = None

    # Infra
    cross_venue_hedger: Any = None
    shadow_tracker: Any = None
    dashboard: Any = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _safe_init(name: str, factory, health: SubsystemHealthRegistry,
               affects_trading: bool = True):
    """
    Try to instantiate a subsystem; register it with the health registry.
    Returns the instance or None on failure.
    """
    health.register(name, affects_trading=affects_trading)
    try:
        instance = factory()
        health.set_status(
            name,
            SubsystemState.HEALTHY,
            dependency_ready=True,
            startup_status="READY",
        )
        health.heartbeat(name)
        logger.info("  ✓ %s", name)
        return instance
    except Exception as exc:
        health.set_status(
            name,
            SubsystemState.FAILED,
            reason=str(exc)[:120],
            dependency_ready=False,
            startup_status="FAILED",
        )
        logger.warning("  ✗ %s — %s", name, exc)
        return None


def build_subsystems(
    health: SubsystemHealthRegistry,
    profile: Optional[set] = None,
) -> SubsystemContainer:
    """
    Build and return a SubsystemContainer based on the given feature *profile*.
    Subsystems not in the profile are skipped (left as None).
    """
    import config

    if profile is None:
        profile = FULL_PROFILE

    c = SubsystemContainer()
    logger.info("Building subsystems (profile has %d features)…", len(profile))

    # ─── Core ─────────────────────────────────────────────────

    from src.discovery.trader_discovery import TraderDiscovery
    from src.analysis.strategy_identifier import StrategyIdentifier
    from src.analysis.strategy_scorer import StrategyScorer
    from src.data.exchange_aggregator import ExchangeAggregator
    from src.analysis.regime_detector import RegimeDetector
    from src.signals.decision_firewall import DecisionFirewall
    from src.signals.agent_scoring import AgentScorer
    from src.analysis.features import FeatureEngine
    from src.ui.reporter import Reporter

    c.exchange_agg = _safe_init("exchange_aggregator", ExchangeAggregator, health, affects_trading=False)
    c.discovery = _safe_init("discovery", TraderDiscovery, health, affects_trading=False)
    c.identifier = _safe_init("strategy_identifier", StrategyIdentifier, health, affects_trading=False)
    c.scorer = _safe_init("strategy_scorer", StrategyScorer, health)
    c.regime_detector = _safe_init(
        "regime_detector",
        lambda: RegimeDetector(exchange_agg=c.exchange_agg),
        health,
    )
    c.agent_scorer = _safe_init("agent_scorer", AgentScorer, health)
    c.feature_engine = _safe_init("feature_engine", FeatureEngine, health, affects_trading=False)
    c.reporter = _safe_init("reporter", Reporter, health, affects_trading=False)

    # ─── Predictive Forecaster ─────────────────────────────────
    # Use XGBoost-powered forecaster when enabled (wraps the base
    # PredictiveRegimeForecaster internally as fallback).
    if "predictive_forecaster" in profile:
        import config as _cfg
        if getattr(_cfg, "ENABLE_XGBOOST_FORECASTER", False) and "xgboost_forecaster" in profile:
            from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster
            c.predictive_forecaster = _safe_init(
                "predictive_forecaster",
                lambda: XGBoostRegimeForecaster({
                    "model_path": getattr(_cfg, "XGBOOST_MODEL_PATH", "models/regime_xgboost.json"),
                    "retrain_interval": getattr(_cfg, "XGBOOST_RETRAIN_INTERVAL", 86400),
                }),
                health,
            )
        else:
            from src.signals.predictive_regime_forecaster import PredictiveRegimeForecaster
            c.predictive_forecaster = _safe_init(
                "predictive_forecaster", PredictiveRegimeForecaster, health,
            )

    # Firewall — needs forecaster injected
    # min_confidence is env-tunable via FIREWALL_MIN_CONFIDENCE (default 0.45)
    import config as _fw_cfg
    c.firewall = _safe_init(
        "decision_firewall",
        lambda: DecisionFirewall({
            "forecaster": c.predictive_forecaster,
            "min_confidence": getattr(_fw_cfg, "FIREWALL_MIN_CONFIDENCE", 0.45),
        }),
        health,
    )

    # ─── V2.5 modules ────────────────────────────────────────
    if "liquidation_strategy" in profile:
        from src.analysis.liquidation_strategy import LiquidationStrategy
        c.liquidation_strategy = _safe_init("liquidation_strategy", LiquidationStrategy, health)

    if "kelly_sizer" in profile:
        from src.signals.kelly_sizing import KellySizer
        c.kelly_sizer = _safe_init("kelly_sizer", KellySizer, health)
        if c.kelly_sizer and c.agent_scorer:
            try:
                c.kelly_sizer.load_from_agent_scorer(c.agent_scorer)
            except Exception:
                pass

    if "portfolio_sizer" in profile:
        from src.signals.portfolio_sizer import PortfolioSizer
        c.portfolio_sizer = _safe_init(
            "portfolio_sizer",
            lambda: PortfolioSizer(
                {
                    "enabled": config.PORTFOLIO_SIZER_ENABLED,
                    "min_position_pct": config.PORTFOLIO_MIN_POSITION_PCT,
                    "max_position_pct": getattr(config, "PAPER_TRADING_MAX_POSITION_PCT", 0.08),
                    "max_coin_exposure_pct": config.PORTFOLIO_MAX_COIN_EXPOSURE_PCT,
                    "max_side_exposure_pct": config.PORTFOLIO_MAX_SIDE_EXPOSURE_PCT,
                    "max_cluster_exposure_pct": config.PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT,
                    "max_beta_abs": config.PORTFOLIO_MAX_BETA_ABS,
                    "target_volatility_pct": config.PORTFOLIO_TARGET_VOLATILITY_PCT,
                    "stop_loss_vol_multiplier": config.PORTFOLIO_STOP_LOSS_VOL_MULTIPLIER,
                    "trend_reward_risk": config.PORTFOLIO_TREND_REWARD_RISK,
                    "base_reward_risk": config.PORTFOLIO_BASE_REWARD_RISK,
                    "volatile_reward_risk": config.PORTFOLIO_VOLATILE_REWARD_RISK,
                    "trend_time_limit_hours": config.PORTFOLIO_TREND_TIME_LIMIT_HOURS,
                    "base_time_limit_hours": config.PORTFOLIO_BASE_TIME_LIMIT_HOURS,
                    "volatile_time_limit_hours": config.PORTFOLIO_VOLATILE_TIME_LIMIT_HOURS,
                }
            ),
            health,
        )

    if "trade_memory" in profile:
        from src.trading.trade_memory import TradeMemory
        c.trade_memory = _safe_init("trade_memory", TradeMemory, health, affects_trading=False)

    if "calibration" in profile:
        from src.signals.calibration import CalibrationTracker
        c.calibration = _safe_init("calibration", CalibrationTracker, health, affects_trading=False)

    if "adaptive_learning" in profile:
        from src.signals.adaptive_learning import AdaptiveLearningManager
        c.adaptive_learning = _safe_init(
            "adaptive_learning",
            lambda: AdaptiveLearningManager(
                {
                    "enabled": config.ADAPTIVE_LEARNING_ENABLED,
                    "lookback_hours": config.ADAPTIVE_LEARNING_LOOKBACK_HOURS,
                    "recent_lookback_hours": config.ADAPTIVE_LEARNING_RECENT_LOOKBACK_HOURS,
                    "report_limit_cycles": config.EXPERIMENT_REPORT_LIMIT_CYCLES,
                    "refresh_interval_cycles": config.ADAPTIVE_LEARNING_REFRESH_INTERVAL_CYCLES,
                    "min_closed_trades": config.ADAPTIVE_LEARNING_MIN_CLOSED_TRADES,
                    "min_recent_closed_trades": config.ADAPTIVE_LEARNING_MIN_RECENT_CLOSED_TRADES,
                    "min_selected_candidates": config.ADAPTIVE_LEARNING_MIN_SELECTED_CANDIDATES,
                    "caution_health_floor": config.ADAPTIVE_LEARNING_CAUTION_HEALTH_FLOOR,
                    "promotion_health_floor": config.ADAPTIVE_LEARNING_PROMOTION_HEALTH_FLOOR,
                    "caution_drift_threshold": config.ADAPTIVE_LEARNING_CAUTION_DRIFT_THRESHOLD,
                    "block_drift_threshold": config.ADAPTIVE_LEARNING_BLOCK_DRIFT_THRESHOLD,
                    "max_calibration_ece": config.ADAPTIVE_LEARNING_MAX_CALIBRATION_ECE,
                    "min_weight_multiplier": config.ADAPTIVE_LEARNING_MIN_WEIGHT_MULTIPLIER,
                    "return_scale": config.ADAPTIVE_LEARNING_RETURN_SCALE,
                    "scaled_promotion_closed_trades": (
                        config.ADAPTIVE_PROMOTION_SCALED_MIN_CLOSED_TRADES
                    ),
                    "scaled_promotion_recent_closed_trades": (
                        config.ADAPTIVE_PROMOTION_SCALED_MIN_RECENT_CLOSED_TRADES
                    ),
                    "scaled_promotion_health_floor": (
                        config.ADAPTIVE_PROMOTION_SCALED_MIN_HEALTH_SCORE
                    ),
                    "scaled_promotion_recent_win_rate": (
                        config.ADAPTIVE_PROMOTION_SCALED_MIN_RECENT_WIN_RATE
                    ),
                    "scaled_promotion_recent_return_pct": (
                        config.ADAPTIVE_PROMOTION_SCALED_MIN_RECENT_RETURN_PCT
                    ),
                    "full_promotion_closed_trades": (
                        config.ADAPTIVE_PROMOTION_FULL_MIN_CLOSED_TRADES
                    ),
                    "full_promotion_recent_closed_trades": (
                        config.ADAPTIVE_PROMOTION_FULL_MIN_RECENT_CLOSED_TRADES
                    ),
                    "full_promotion_health_floor": (
                        config.ADAPTIVE_PROMOTION_FULL_MIN_HEALTH_SCORE
                    ),
                    "full_promotion_recent_win_rate": (
                        config.ADAPTIVE_PROMOTION_FULL_MIN_RECENT_WIN_RATE
                    ),
                    "full_promotion_recent_return_pct": (
                        config.ADAPTIVE_PROMOTION_FULL_MIN_RECENT_RETURN_PCT
                    ),
                    "full_promotion_live_success_rate": (
                        config.ADAPTIVE_PROMOTION_FULL_MIN_LIVE_SUCCESS_RATE
                    ),
                    "incubating_promotion_multiplier": (
                        config.ADAPTIVE_PROMOTION_INCUBATING_MULTIPLIER
                    ),
                    "trial_promotion_multiplier": config.ADAPTIVE_PROMOTION_TRIAL_MULTIPLIER,
                    "scaled_promotion_multiplier": config.ADAPTIVE_PROMOTION_SCALED_MULTIPLIER,
                    "full_promotion_multiplier": config.ADAPTIVE_PROMOTION_FULL_MULTIPLIER,
                    "incubating_promotion_cap_pct": (
                        config.ADAPTIVE_PROMOTION_INCUBATING_CAP_PCT
                    ),
                    "trial_promotion_cap_pct": config.ADAPTIVE_PROMOTION_TRIAL_CAP_PCT,
                    "scaled_promotion_cap_pct": config.ADAPTIVE_PROMOTION_SCALED_CAP_PCT,
                    "full_promotion_cap_pct": config.ADAPTIVE_PROMOTION_FULL_CAP_PCT,
                    "arena_min_trades": config.ADAPTIVE_ARENA_MIN_TRADES,
                    "arena_min_win_rate": config.ADAPTIVE_ARENA_MIN_WIN_RATE,
                    "arena_min_sharpe": config.ADAPTIVE_ARENA_MIN_SHARPE,
                    "arena_max_drawdown": config.ADAPTIVE_ARENA_MAX_DRAWDOWN,
                },
                agent_scorer=c.agent_scorer,
                calibration=c.calibration,
            ),
            health,
            affects_trading=False,
        )

    if "execution_policy" in profile:
        from src.signals.execution_policy import ExecutionPolicyManager
        c.execution_policy = _safe_init(
            "execution_policy",
            lambda: ExecutionPolicyManager(
                {
                    "enabled": config.DECISION_EXECUTION_POLICY_ENABLED,
                    "lookback_hours": config.DECISION_EXECUTION_POLICY_LOOKBACK_HOURS,
                    "min_events": config.DECISION_EXECUTION_POLICY_MIN_EVENTS,
                    "maker_rejection_ceiling": config.DECISION_EXECUTION_POLICY_MAKER_REJECTION_CEILING,
                    "maker_fill_floor": config.DECISION_EXECUTION_POLICY_MAKER_FILL_FLOOR,
                    "maker_confidence_ceiling": config.DECISION_EXECUTION_POLICY_MAKER_CONFIDENCE_CEILING,
                    "maker_source_quality_floor": config.DECISION_EXECUTION_POLICY_MAKER_SOURCE_QUALITY_FLOOR,
                    "maker_offset_bps": config.DECISION_EXECUTION_POLICY_MAKER_OFFSET_BPS,
                    "min_maker_offset_bps": config.DECISION_EXECUTION_POLICY_MIN_MAKER_OFFSET_BPS,
                    "max_maker_offset_bps": config.DECISION_EXECUTION_POLICY_MAX_MAKER_OFFSET_BPS,
                    "maker_timeout_seconds": config.DECISION_EXECUTION_POLICY_MAKER_TIMEOUT_SECONDS,
                    "fallback_confidence_threshold": (
                        config.DECISION_EXECUTION_POLICY_FALLBACK_CONFIDENCE_THRESHOLD
                    ),
                    "fallback_source_quality_threshold": (
                        config.DECISION_EXECUTION_POLICY_FALLBACK_SOURCE_QUALITY_THRESHOLD
                    ),
                    "market_slippage_multiplier": (
                        config.DECISION_EXECUTION_POLICY_MARKET_SLIPPAGE_MULTIPLIER
                    ),
                    "maker_slippage_floor_bps": (
                        config.DECISION_EXECUTION_POLICY_MAKER_SLIPPAGE_FLOOR_BPS
                    ),
                    "default_market_slippage_bps": (
                        config.DECISION_EXECUTION_POLICY_DEFAULT_MARKET_SLIPPAGE_BPS
                    ),
                    "default_execution_role": config.DECISION_DEFAULT_EXECUTION_ROLE,
                    "urgent_sources": config.DECISION_EXECUTION_POLICY_URGENT_SOURCES,
                },
                adaptive_learning=c.adaptive_learning,
            ),
            health,
            affects_trading=False,
        )

    if "divergence_controller" in profile:
        from src.signals.divergence_controller import DivergenceController

        c.divergence_controller = _safe_init(
            "divergence_controller",
            lambda: DivergenceController(
                {
                    "enabled": config.RUNTIME_DIVERGENCE_CONTROL_ENABLED,
                    "live_trading_enabled": config.LIVE_TRADING_ENABLED,
                    "lookback_hours": config.EXPERIMENT_DIVERGENCE_LOOKBACK_HOURS,
                    "refresh_interval_seconds": config.RUNTIME_DIVERGENCE_REFRESH_INTERVAL_SECONDS,
                    "min_live_events": config.RUNTIME_DIVERGENCE_MIN_LIVE_EVENTS,
                    "source_min_selected": config.RUNTIME_DIVERGENCE_SOURCE_MIN_SELECTED,
                    "caution_multiplier": config.RUNTIME_DIVERGENCE_CAUTION_MULTIPLIER,
                    "blocked_multiplier": config.RUNTIME_DIVERGENCE_BLOCKED_MULTIPLIER,
                    "block_on_status": config.RUNTIME_DIVERGENCE_BLOCK_ON_STATUS,
                    "global_caution_open_gap_ratio": config.RUNTIME_DIVERGENCE_GLOBAL_CAUTION_OPEN_GAP_RATIO,
                    "global_block_open_gap_ratio": config.RUNTIME_DIVERGENCE_GLOBAL_BLOCK_OPEN_GAP_RATIO,
                    "global_caution_execution_gap_ratio": (
                        config.RUNTIME_DIVERGENCE_GLOBAL_CAUTION_EXECUTION_GAP_RATIO
                    ),
                    "global_block_execution_gap_ratio": (
                        config.RUNTIME_DIVERGENCE_GLOBAL_BLOCK_EXECUTION_GAP_RATIO
                    ),
                    "global_caution_pnl_gap_ratio": config.RUNTIME_DIVERGENCE_GLOBAL_CAUTION_PNL_GAP_RATIO,
                    "global_block_pnl_gap_ratio": config.RUNTIME_DIVERGENCE_GLOBAL_BLOCK_PNL_GAP_RATIO,
                    "global_caution_rejection_rate": (
                        config.RUNTIME_DIVERGENCE_GLOBAL_CAUTION_REJECTION_RATE
                    ),
                    "global_block_rejection_rate": (
                        config.RUNTIME_DIVERGENCE_GLOBAL_BLOCK_REJECTION_RATE
                    ),
                    "source_caution_execution_gap_ratio": (
                        config.RUNTIME_DIVERGENCE_SOURCE_CAUTION_EXECUTION_GAP_RATIO
                    ),
                    "source_block_execution_gap_ratio": (
                        config.RUNTIME_DIVERGENCE_SOURCE_BLOCK_EXECUTION_GAP_RATIO
                    ),
                    "source_caution_rejection_rate": (
                        config.RUNTIME_DIVERGENCE_SOURCE_CAUTION_REJECTION_RATE
                    ),
                    "source_block_rejection_rate": (
                        config.RUNTIME_DIVERGENCE_SOURCE_BLOCK_REJECTION_RATE
                    ),
                    "source_caution_fill_ratio": config.RUNTIME_DIVERGENCE_SOURCE_CAUTION_FILL_RATIO,
                    "source_block_fill_ratio": config.RUNTIME_DIVERGENCE_SOURCE_BLOCK_FILL_RATIO,
                }
            ),
            health,
            affects_trading=False,
        )

    if "capital_governor" in profile:
        from src.signals.capital_governor import CapitalGovernor

        c.capital_governor = _safe_init(
            "capital_governor",
            lambda: CapitalGovernor(
                {
                    "enabled": config.CAPITAL_GOVERNOR_ENABLED,
                    "lookback_hours": config.CAPITAL_GOVERNOR_LOOKBACK_HOURS,
                    "refresh_interval_seconds": config.CAPITAL_GOVERNOR_REFRESH_INTERVAL_SECONDS,
                    "min_paper_trades": config.CAPITAL_GOVERNOR_MIN_PAPER_TRADES,
                    "min_live_snapshots": config.CAPITAL_GOVERNOR_MIN_LIVE_SNAPSHOTS,
                    "min_source_profiles": config.CAPITAL_GOVERNOR_MIN_SOURCE_PROFILES,
                    "caution_multiplier": config.CAPITAL_GOVERNOR_CAUTION_MULTIPLIER,
                    "risk_off_multiplier": config.CAPITAL_GOVERNOR_RISK_OFF_MULTIPLIER,
                    "blocked_multiplier": config.CAPITAL_GOVERNOR_BLOCKED_MULTIPLIER,
                    "block_on_risk_off": config.CAPITAL_GOVERNOR_BLOCK_ON_RISK_OFF,
                    "caution_paper_drawdown_pct": config.CAPITAL_GOVERNOR_CAUTION_PAPER_DRAWDOWN_PCT,
                    "risk_off_paper_drawdown_pct": config.CAPITAL_GOVERNOR_RISK_OFF_PAPER_DRAWDOWN_PCT,
                    "block_paper_drawdown_pct": config.CAPITAL_GOVERNOR_BLOCK_PAPER_DRAWDOWN_PCT,
                    "caution_live_drawdown_pct": config.CAPITAL_GOVERNOR_CAUTION_LIVE_DRAWDOWN_PCT,
                    "risk_off_live_drawdown_pct": config.CAPITAL_GOVERNOR_RISK_OFF_LIVE_DRAWDOWN_PCT,
                    "block_live_drawdown_pct": config.CAPITAL_GOVERNOR_BLOCK_LIVE_DRAWDOWN_PCT,
                    "caution_paper_sharpe": config.CAPITAL_GOVERNOR_CAUTION_PAPER_SHARPE,
                    "risk_off_paper_sharpe": config.CAPITAL_GOVERNOR_RISK_OFF_PAPER_SHARPE,
                    "caution_live_sharpe": config.CAPITAL_GOVERNOR_CAUTION_LIVE_SHARPE,
                    "risk_off_live_sharpe": config.CAPITAL_GOVERNOR_RISK_OFF_LIVE_SHARPE,
                    "caution_degraded_source_ratio": config.CAPITAL_GOVERNOR_CAUTION_DEGRADED_SOURCE_RATIO,
                    "risk_off_blocked_source_ratio": config.CAPITAL_GOVERNOR_RISK_OFF_BLOCKED_SOURCE_RATIO,
                    "low_regime_confidence": config.CAPITAL_GOVERNOR_LOW_REGIME_CONFIDENCE,
                    "divergence_blocks": config.CAPITAL_GOVERNOR_DIVERGENCE_BLOCKS,
                },
                divergence_controller=c.divergence_controller,
            ),
            health,
            affects_trading=False,
        )

    if "source_allocator" in profile:
        from src.signals.source_allocator import SourceBudgetAllocator

        c.source_allocator = _safe_init(
            "source_allocator",
            lambda: SourceBudgetAllocator(
                {
                    "enabled": config.SOURCE_BUDGET_ALLOCATOR_ENABLED,
                    "lookback_hours": config.SOURCE_BUDGET_LOOKBACK_HOURS,
                    "refresh_interval_seconds": config.SOURCE_BUDGET_REFRESH_INTERVAL_SECONDS,
                    "min_position_pct": config.SOURCE_BUDGET_MIN_POSITION_PCT,
                    "min_multiplier": config.SOURCE_BUDGET_MIN_MULTIPLIER,
                    "max_multiplier": config.SOURCE_BUDGET_MAX_MULTIPLIER,
                    "active_multiplier": config.SOURCE_BUDGET_ACTIVE_MULTIPLIER,
                    "warming_multiplier": config.SOURCE_BUDGET_WARMING_MULTIPLIER,
                    "caution_multiplier": config.SOURCE_BUDGET_CAUTION_MULTIPLIER,
                    "blocked_multiplier": config.SOURCE_BUDGET_BLOCKED_MULTIPLIER,
                    "active_cap_pct": config.SOURCE_BUDGET_ACTIVE_CAP_PCT,
                    "warming_cap_pct": config.SOURCE_BUDGET_WARMING_CAP_PCT,
                    "caution_cap_pct": config.SOURCE_BUDGET_CAUTION_CAP_PCT,
                    "blocked_cap_pct": config.SOURCE_BUDGET_BLOCKED_CAP_PCT,
                    "min_health_score": config.SOURCE_BUDGET_MIN_HEALTH_SCORE,
                    "min_closed_trades": config.SOURCE_BUDGET_MIN_CLOSED_TRADES,
                    "return_scale": config.SOURCE_BUDGET_RETURN_SCALE,
                    "live_rejection_ceiling": config.SOURCE_BUDGET_LIVE_REJECTION_CEILING,
                    "live_fill_floor": config.SOURCE_BUDGET_LIVE_FILL_FLOOR,
                    "block_on_status": config.SOURCE_BUDGET_BLOCK_ON_STATUS,
                    "divergence_enabled": config.RUNTIME_DIVERGENCE_CONTROL_ENABLED,
                    "capital_governor_enabled": config.SOURCE_BUDGET_CAPITAL_GOVERNOR_ENABLED,
                    "promotion_ladder_enabled": config.SOURCE_BUDGET_PROMOTION_LADDER_ENABLED,
                },
                adaptive_learning=c.adaptive_learning,
                divergence_controller=c.divergence_controller,
                capital_governor=c.capital_governor,
            ),
            health,
            affects_trading=False,
        )

    if "llm_filter" in profile:
        from src.signals.llm_filter import LLMFilter
        c.llm_filter = _safe_init("llm_filter", LLMFilter, health)

    if "signal_processor" in profile:
        from src.signals.signal_processor import SignalProcessor, ArenaIncubator
        c.signal_processor = _safe_init("signal_processor", SignalProcessor, health)
        c.arena_incubator = _safe_init("arena_incubator", ArenaIncubator, health, affects_trading=False)

    if "decision_engine" in profile:
        from src.signals.decision_engine import DecisionEngine
        c.decision_engine = _safe_init(
            "decision_engine",
            lambda: DecisionEngine(
                {
                    "w_score": config.DECISION_W_SCORE,
                    "w_regime": config.DECISION_W_REGIME,
                    "w_diversity": config.DECISION_W_DIVERSITY,
                    "w_freshness": config.DECISION_W_FRESHNESS,
                    "w_consensus": config.DECISION_W_CONSENSUS,
                    "w_confidence": config.DECISION_W_CONFIDENCE,
                    "w_source_quality": config.DECISION_W_SOURCE_QUALITY,
                    "w_confirmation": config.DECISION_W_CONFIRMATION,
                    "w_expected_value": config.DECISION_W_EXPECTED_VALUE,
                    "w_confluence": config.DECISION_W_CONFLUENCE,
                    "w_context": config.DECISION_W_CONTEXT,
                    "w_calibration": config.DECISION_W_CALIBRATION,
                    "w_memory": config.DECISION_W_MEMORY,
                    "w_divergence": config.DECISION_W_DIVERGENCE,
                    "w_capital_governor": config.DECISION_W_CAPITAL_GOVERNOR,
                    "min_decision_score": config.DECISION_MIN_SCORE,
                    "min_signal_confidence": config.DECISION_MIN_CONFIDENCE,
                    "min_source_weight": config.DECISION_MIN_SOURCE_WEIGHT,
                    "min_expected_value_pct": config.DECISION_MIN_EXPECTED_VALUE_PCT,
                    "confluence_enabled": config.DECISION_CONFLUENCE_ENABLED,
                    "confluence_baseline": config.DECISION_CONFLUENCE_BASELINE,
                    "confluence_full_weight": config.DECISION_CONFLUENCE_FULL_WEIGHT,
                    "confluence_target_support_sources": (
                        config.DECISION_CONFLUENCE_TARGET_SUPPORT_SOURCES
                    ),
                    "confluence_conflict_block_threshold": (
                        config.DECISION_CONFLUENCE_CONFLICT_BLOCK_THRESHOLD
                    ),
                    "confluence_conflict_floor": config.DECISION_CONFLUENCE_CONFLICT_FLOOR,
                    "context_performance_enabled": config.DECISION_CONTEXT_PERFORMANCE_ENABLED,
                    "context_performance_lookback_hours": (
                        config.DECISION_CONTEXT_PERFORMANCE_LOOKBACK_HOURS
                    ),
                    "context_performance_min_trades": config.DECISION_CONTEXT_PERFORMANCE_MIN_TRADES,
                    "context_performance_return_scale": (
                        config.DECISION_CONTEXT_PERFORMANCE_RETURN_SCALE
                    ),
                    "context_performance_block_win_rate": (
                        config.DECISION_CONTEXT_PERFORMANCE_BLOCK_WIN_RATE
                    ),
                    "context_performance_block_avg_return_pct": (
                        config.DECISION_CONTEXT_PERFORMANCE_BLOCK_AVG_RETURN_PCT
                    ),
                    "context_performance_boost_win_rate": (
                        config.DECISION_CONTEXT_PERFORMANCE_BOOST_WIN_RATE
                    ),
                    "calibration": c.calibration,
                    "calibration_enabled": config.DECISION_CALIBRATION_ENABLED,
                    "calibration_min_records": config.DECISION_CALIBRATION_MIN_RECORDS,
                    "calibration_target_ece": config.DECISION_CALIBRATION_TARGET_ECE,
                    "calibration_max_ece": config.DECISION_CALIBRATION_MAX_ECE,
                    "trade_memory": c.trade_memory,
                    "memory_enabled": config.DECISION_MEMORY_ENABLED,
                    "memory_min_trades": config.DECISION_MEMORY_MIN_TRADES,
                    "memory_min_similarity": config.DECISION_MEMORY_MIN_SIMILARITY,
                    "memory_top_k": config.DECISION_MEMORY_TOP_K,
                    "memory_block_on_avoid": config.DECISION_MEMORY_BLOCK_ON_AVOID,
                    "divergence_controller": c.divergence_controller,
                    "divergence_enabled": config.DECISION_DIVERGENCE_ENABLED,
                    "divergence_block_on_status": config.DECISION_DIVERGENCE_BLOCK_ON_STATUS,
                    "capital_governor": c.capital_governor,
                    "capital_governor_enabled": config.DECISION_CAPITAL_GOVERNOR_ENABLED,
                    "capital_governor_block_on_status": config.DECISION_CAPITAL_GOVERNOR_BLOCK_ON_STATUS,
                    "max_trades_per_cycle": config.DECISION_MAX_TRADES_PER_CYCLE,
                    "maker_fee_bps": config.DECISION_MAKER_FEE_BPS,
                    "taker_fee_bps": config.DECISION_TAKER_FEE_BPS,
                    "expected_slippage_bps": config.DECISION_EXPECTED_SLIPPAGE_BPS,
                    "churn_penalty_bps": config.DECISION_CHURN_PENALTY_BPS,
                    "default_execution_role": config.DECISION_DEFAULT_EXECUTION_ROLE,
                    "persist_research": config.DECISION_PERSIST_RESEARCH,
                    "execution_quality_enabled": config.DECISION_EXECUTION_QUALITY_ENABLED,
                    "execution_quality_lookback_hours": config.DECISION_EXECUTION_QUALITY_LOOKBACK_HOURS,
                    "execution_quality_min_events": config.DECISION_EXECUTION_QUALITY_MIN_EVENTS,
                    "execution_rejection_penalty_bps": config.DECISION_EXECUTION_REJECTION_PENALTY_BPS,
                    "execution_fill_gap_penalty_bps": config.DECISION_EXECUTION_FILL_GAP_PENALTY_BPS,
                    "execution_protective_failure_penalty_bps": (
                        config.DECISION_EXECUTION_PROTECTIVE_FAILURE_PENALTY_BPS
                    ),
                    "execution_policy": c.execution_policy,
                    "execution_policy_enabled": config.DECISION_EXECUTION_POLICY_ENABLED,
                    "adaptive_learning": c.adaptive_learning,
                    "adaptive_learning_enabled": config.ADAPTIVE_LEARNING_ENABLED,
                    "adaptive_learning_block_on_status": config.ADAPTIVE_LEARNING_BLOCK_ON_STATUS,
                    "adaptive_learning_min_health_score": config.ADAPTIVE_LEARNING_MIN_HEALTH_SCORE,
                }
            ),
            health,
        )

    if "alpha_arena" in profile:
        from src.signals.alpha_arena import AlphaArena
        c.arena = _safe_init("alpha_arena", AlphaArena, health, affects_trading=False)
        if c.adaptive_learning:
            try:
                c.adaptive_learning.attach(arena=c.arena)
            except Exception:
                pass

    # ─── Options Flow ─────────────────────────────────────────
    if "options_flow" in profile:
        from src.data.options_flow import OptionsFlowScanner
        c.options_scanner = _safe_init("options_flow", OptionsFlowScanner, health)

    # ─── Polymarket ───────────────────────────────────────────
    if "polymarket" in profile:
        from src.data.polymarket_scanner import PolymarketScanner
        c.polymarket = _safe_init("polymarket", PolymarketScanner, health)

    # ─── Multi-exchange ───────────────────────────────────────
    if "multi_scanner" in profile:
        from src.exchanges.scanner import MultiExchangeScanner
        c.multi_scanner = _safe_init(
            "multi_scanner",
            lambda: MultiExchangeScanner(config={"lighter_enabled": config.LIGHTER_ENABLED}),
            health,
        )

    # ─── Copy trader ──────────────────────────────────────────
    if "copy_trader" in profile:
        from src.trading.copy_trader import CopyTrader
        c.copy_trader = _safe_init(
            "copy_trader",
            lambda: CopyTrader(
                firewall=c.firewall,
                agent_scorer=c.agent_scorer,
                kelly_sizer=c.kelly_sizer,
                portfolio_sizer=c.portfolio_sizer,
                source_allocator=c.source_allocator,
                trade_memory=c.trade_memory,
                calibration=c.calibration,
                regime_forecaster=c.predictive_forecaster,
            ),
            health,
        )

    # ─── Paper trader ─────────────────────────────────────────
    from src.trading.paper_trader import PaperTrader
    c.paper_trader = _safe_init(
        "paper_trader",
        lambda: PaperTrader(
            firewall=c.firewall,
            agent_scorer=c.agent_scorer,
            feature_engine=c.feature_engine,
            kelly_sizer=c.kelly_sizer,
            portfolio_sizer=c.portfolio_sizer,
            source_allocator=c.source_allocator,
            trade_memory=c.trade_memory,
            calibration=c.calibration,
            llm_filter=c.llm_filter,
        ),
        health,
    )

    # ─── Live trader ──────────────────────────────────────────
    if "live_trader" in profile:
        from src.trading.live_trader import LiveTrader
        c.live_trader = _safe_init(
            "live_trader",
            lambda: LiveTrader(
                firewall=c.firewall,
                dry_run=not getattr(config, "LIVE_TRADING_ENABLED", False),
                max_daily_loss=float(getattr(config, "LIVE_MAX_DAILY_LOSS_USD", 500)),
                max_order_usd=float(getattr(config, "LIVE_MAX_ORDER_USD", 3.0)),
                regime_forecaster=c.predictive_forecaster,
            ),
            health,
            affects_trading=bool(getattr(config, "LIVE_TRADING_ENABLED", False)),
        )
        if c.live_trader:
            if not getattr(config, "LIVE_TRADING_ENABLED", False):
                health.set_status(
                    "live_trader",
                    SubsystemState.DISABLED,
                    reason="LIVE_TRADING_ENABLED=false",
                    dependency_ready=False,
                    startup_status="DISABLED",
                )
            elif not c.live_trader.is_deployable():
                health.set_status(
                    "live_trader",
                    SubsystemState.DEGRADED,
                    reason=c.live_trader.get_stats().get("status_reason", "not deployable"),
                    dependency_ready=False,
                    startup_status="WAITING_FOR_DEPENDENCIES",
                )

    # ─── Cross-venue hedger ───────────────────────────────────
    if "cross_venue_hedger" in profile:
        from src.trading.cross_venue_hedger import CrossVenueHedger
        c.cross_venue_hedger = _safe_init(
            "cross_venue_hedger", CrossVenueHedger, health,
        )

    # ─── Shadow tracker ───────────────────────────────────────
    if "shadow_tracker" in profile:
        from src.analysis.shadow_tracker import ShadowTracker
        c.shadow_tracker = _safe_init(
            "shadow_tracker", ShadowTracker, health, affects_trading=False,
        )

    # ─── Bot detector + regime filter ─────────────────────────
    if "adaptive_bot_detector" in profile:
        from src.discovery.adaptive_bot_detector import AdaptiveBotDetector
        c.adaptive_bot_detector = _safe_init(
            "adaptive_bot_detector", AdaptiveBotDetector, health, affects_trading=False,
        )

    if "regime_strategy_filter" in profile:
        from src.analysis.regime_strategy_filter import RegimeStrategyFilter
        c.regime_strategy_filter = _safe_init(
            "regime_strategy_filter", RegimeStrategyFilter, health,
        )

    # ─── Position monitor (WebSocket) ─────────────────────────
    if "position_monitor" in profile:
        try:
            import src.data.database as db
            top_traders = db.get_active_traders()[:20]
            if top_traders:
                c.position_monitor = _safe_init(
                    "position_monitor",
                    lambda: _start_position_monitor(top_traders),
                    health,
                )
        except Exception as exc:
            logger.warning("  ✗ position_monitor — %s", exc)

    # ─── WebSocket feed ───────────────────────────────────────
    try:
        from src.data.hyperliquid_client import start_websocket
        start_websocket(coins=["BTC", "ETH", "SOL", "DOGE", "ARB", "OP"])
        logger.info("  ✓ websocket_feed")
    except Exception as exc:
        logger.warning("  ✗ websocket_feed — %s (REST fallback active)", exc)

    # ─── Dashboard ────────────────────────────────────────────
    if "dashboard" in profile:
        try:
            from src.ui.dashboard import start_dashboard, set_v2_components, set_live_trader
            set_v2_components(
                firewall=c.firewall,
                regime_detector=c.regime_detector,
                arena=c.arena,
                kelly_sizer=c.kelly_sizer,
                portfolio_sizer=c.portfolio_sizer,
                trade_memory=c.trade_memory,
                calibration=c.calibration,
                adaptive_learning=c.adaptive_learning,
                llm_filter=c.llm_filter,
                liquidation_strategy=c.liquidation_strategy,
                signal_processor=c.signal_processor,
                arena_incubator=c.arena_incubator,
                decision_engine=c.decision_engine,
                multi_scanner=c.multi_scanner,
                shadow_tracker=c.shadow_tracker,
                execution_policy=c.execution_policy,
                source_allocator=c.source_allocator,
                divergence_controller=c.divergence_controller,
                capital_governor=c.capital_governor,
            )
            if c.live_trader:
                set_live_trader(c.live_trader)
            c.dashboard = start_dashboard(options_scanner=c.options_scanner)
            logger.info("  ✓ dashboard")
        except Exception as exc:
            logger.warning("  ✗ dashboard — %s", exc)

    # ─── Telegram ─────────────────────────────────────────────
    try:
        from src.notifications import telegram_bot as tg
        if tg.is_configured():
            tg.send_startup_message()
            logger.info("  ✓ telegram")
        else:
            logger.info("  — telegram (not configured)")
    except Exception as exc:
        logger.debug("  ✗ telegram — %s", exc)

    logger.info("Subsystem build complete.")
    return c


# ---------------------------------------------------------------------------
# Heartbeat helper
# ---------------------------------------------------------------------------

# Maps container field names → health registry names.
# Only fields whose registry name differs from the field name need entries;
# fields that match (e.g. "discovery" → "discovery") are also listed for
# completeness and to keep the mapping explicit.
_FIELD_TO_HEALTH_NAME: dict = {
    "exchange_agg":           "exchange_aggregator",
    "discovery":              "discovery",
    "identifier":             "strategy_identifier",
    "scorer":                 "strategy_scorer",
    "regime_detector":        "regime_detector",
    "agent_scorer":           "agent_scorer",
    "feature_engine":         "feature_engine",
    "reporter":               "reporter",
    "predictive_forecaster":  "predictive_forecaster",
    "firewall":               "decision_firewall",
    "liquidation_strategy":   "liquidation_strategy",
    "kelly_sizer":            "kelly_sizer",
    "trade_memory":           "trade_memory",
    "calibration":            "calibration",
    "execution_policy":       "execution_policy",
    "divergence_controller":  "divergence_controller",
    "capital_governor":       "capital_governor",
    "llm_filter":             "llm_filter",
    "signal_processor":       "signal_processor",
    "arena_incubator":        "arena_incubator",
    "decision_engine":        "decision_engine",
    "arena":                  "alpha_arena",
    "options_scanner":        "options_flow",
    "polymarket":             "polymarket",
    "multi_scanner":          "multi_scanner",
    "copy_trader":            "copy_trader",
    "paper_trader":           "paper_trader",
    "live_trader":            "live_trader",
    "cross_venue_hedger":     "cross_venue_hedger",
    "shadow_tracker":         "shadow_tracker",
    "adaptive_bot_detector":  "adaptive_bot_detector",
    "regime_strategy_filter": "regime_strategy_filter",
    "position_monitor":       "position_monitor",
}


def heartbeat_active(container: SubsystemContainer,
                     health: SubsystemHealthRegistry) -> None:
    """
    Send a heartbeat for every non-None subsystem in *container*.

    Call this after each cycle (fast / trading / discovery / reporting)
    so the health registry knows subsystems are alive and doesn't
    auto-degrade them for stale heartbeats.
    """
    for field, health_name in _FIELD_TO_HEALTH_NAME.items():
        if getattr(container, field, None) is not None:
            try:
                health.heartbeat(health_name)
            except Exception:
                pass  # heartbeat is best-effort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_position_monitor(top_traders):
    from src.notifications.ws_position_monitor import PositionMonitor
    pm = PositionMonitor(max_signal_queue=500)
    pm.start([t["address"] for t in top_traders])
    return pm
