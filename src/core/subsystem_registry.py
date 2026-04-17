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
import json
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
    "regime_detector", "feature_engine", "agent_scorer",
}

FULL_PROFILE = FUNDABLE_CORE | {
    "copy_trader", "live_trader", "options_flow", "polymarket",
    "predictive_forecaster", "xgboost_forecaster", "alpha_pipeline", "multi_scanner", "event_scanner",
    "liquidation_strategy", "kelly_sizer", "trade_memory", "calibration",
    "llm_filter", "risk_policy_engine", "signal_processor", "arena_incubator", "decision_engine",
    "alpha_arena", "position_monitor", "dashboard", "telegram",
    "cross_venue_hedger", "shadow_tracker", "adaptive_bot_detector",
    "regime_strategy_filter", "exchange_aggregator",
    "lstm_agent", "rl_sizer", "macro_regime",
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
    event_scanner: Any = None
    alpha_pipeline: Any = None
    liquidation_strategy: Any = None
    kelly_sizer: Any = None
    trade_memory: Any = None
    calibration: Any = None
    llm_filter: Any = None
    risk_policy_engine: Any = None
    signal_processor: Any = None
    arena_incubator: Any = None
    decision_engine: Any = None
    arena: Any = None
    adaptive_bot_detector: Any = None
    regime_strategy_filter: Any = None
    lstm_agent: Any = None
    rl_sizer: Any = None
    macro_regime: Any = None

    # Infra
    cross_venue_hedger: Any = None
    shadow_tracker: Any = None
    dashboard: Any = None
    data_source_registry: Any = None


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
        logger.info("  OK %s", name)
        return instance
    except Exception as exc:
        health.set_status(
            name,
            SubsystemState.FAILED,
            reason=str(exc)[:120],
            dependency_ready=False,
            startup_status="FAILED",
        )
        logger.warning("  FAIL %s - %s", name, exc)
        return None


def _mark_degraded(
    name: str,
    reason: str,
    health: SubsystemHealthRegistry,
    *,
    dependency_ready: bool = True,
    startup_status: str = "DEGRADED",
) -> None:
    """Mark an already-initialized subsystem as degraded with a clear reason."""
    health.set_status(
        name,
        SubsystemState.DEGRADED,
        reason=str(reason)[:120],
        dependency_ready=dependency_ready,
        startup_status=startup_status,
    )
    logger.warning("  DEGRADED %s - %s", name, reason)


def _load_json_override(name: str, raw_value: str) -> Any:
    """Parse required JSON overrides and fail fast when malformed."""
    if not raw_value:
        return {}
    try:
        return json.loads(raw_value)
    except Exception as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


def _wire_kelly_sizer_from_agent_scorer(container: SubsystemContainer, health: SubsystemHealthRegistry) -> None:
    """Load Kelly sizing priors and expose wiring failures as degraded health."""
    if not container.kelly_sizer or not container.agent_scorer:
        return
    try:
        container.kelly_sizer.load_from_agent_scorer(container.agent_scorer)
    except Exception as exc:
        _mark_degraded(
            "kelly_sizer",
            f"agent scorer wiring failed: {exc}",
            health,
        )


def _wire_event_scanner(container: SubsystemContainer, health: SubsystemHealthRegistry) -> None:
    """Attach the event scanner to the firewall and surface wiring failures."""
    if not container.firewall or not container.event_scanner:
        return
    if not hasattr(container.firewall, "set_event_scanner"):
        _mark_degraded(
            "event_scanner",
            "firewall does not expose set_event_scanner",
            health,
        )
        _mark_degraded(
            "decision_firewall",
            "event scanner wiring unavailable",
            health,
        )
        return
    try:
        container.firewall.set_event_scanner(container.event_scanner)
    except Exception as exc:
        _mark_degraded(
            "event_scanner",
            f"firewall wiring failed: {exc}",
            health,
        )
        _mark_degraded(
            "decision_firewall",
            f"event scanner wiring failed: {exc}",
            health,
        )


def build_subsystems(
    health: SubsystemHealthRegistry,
    profile: Optional[set] = None,
) -> SubsystemContainer:
    """
    Build and return a SubsystemContainer based on the given feature *profile*.
    Subsystems not in the profile are skipped (left as None).
    """
    import config
    from src.core.data_source_registry import DataSourceRegistry

    if profile is None:
        profile = FULL_PROFILE

    c = SubsystemContainer()
    c.data_source_registry = DataSourceRegistry()
    for source_name in ("polymarket", "options_flow", "deribit"):
        c.data_source_registry.register_source(source_name)
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
    c.agent_scorer = _safe_init(
        "agent_scorer",
        lambda: AgentScorer(
            {
                "policy_enabled": getattr(config, "SOURCE_POLICY_ENABLED", True),
                "policy_min_closed_trades": getattr(config, "SOURCE_POLICY_MIN_CLOSED_TRADES", 3),
                "policy_keep_top_n": getattr(config, "SOURCE_POLICY_KEEP_TOP_N", 5),
                "policy_pause_weight": getattr(config, "SOURCE_POLICY_PAUSE_WEIGHT", 0.12),
                "policy_degrade_weight": getattr(config, "SOURCE_POLICY_DEGRADE_WEIGHT", 0.32),
                "policy_warmup_max_signals_per_day": getattr(
                    config, "SOURCE_POLICY_WARMUP_MAX_SIGNALS_PER_DAY", 1
                ),
                "policy_degraded_max_signals_per_day": getattr(
                    config, "SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY", 1
                ),
                "policy_warmup_size_multiplier": getattr(
                    config, "SOURCE_POLICY_WARMUP_SIZE_MULTIPLIER", 0.75
                ),
                "policy_degraded_size_multiplier": getattr(
                    config, "SOURCE_POLICY_DEGRADED_SIZE_MULTIPLIER", 0.60
                ),
                "policy_warmup_min_confidence": getattr(
                    config, "SOURCE_POLICY_WARMUP_MIN_CONFIDENCE", 0.45
                ),
                "policy_degraded_min_confidence": getattr(
                    config, "SOURCE_POLICY_DEGRADED_MIN_CONFIDENCE", 0.55
                ),
            }
        ),
        health,
    )
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
                    "source_registry": c.data_source_registry,
                }),
                health,
            )
        else:
            from src.signals.predictive_regime_forecaster import PredictiveRegimeForecaster
            c.predictive_forecaster = _safe_init(
                "predictive_forecaster",
                lambda: PredictiveRegimeForecaster({"source_registry": c.data_source_registry}),
                health,
            )

    if "alpha_pipeline" in profile and getattr(config, "ENABLE_ALPHA_PIPELINE", True):
        from src.signals.feature_store_alpha import FeatureStoreAlphaPipeline
        c.alpha_pipeline = _safe_init(
            "alpha_pipeline",
            lambda: FeatureStoreAlphaPipeline(
                {
                    "timeframe": getattr(config, "ALPHA_TIMEFRAME", "1h"),
                    "lookback_days": getattr(config, "ALPHA_LOOKBACK_DAYS", 120),
                    "min_training_samples": getattr(config, "ALPHA_MIN_TRAINING_SAMPLES", 250),
                    "retrain_interval": getattr(config, "ALPHA_RETRAIN_INTERVAL", 21600),
                    "walk_forward_splits": getattr(config, "ALPHA_WALK_FORWARD_SPLITS", 5),
                    "label_min_abs_return": getattr(config, "ALPHA_LABEL_MIN_ABS_RETURN", 0.0005),
                    "signal_min_confidence": getattr(config, "ALPHA_SIGNAL_MIN_CONFIDENCE", 0.58),
                    "min_significant_trades": getattr(config, "ALPHA_MIN_SIGNIFICANT_TRADES", 60),
                    "min_significance_pvalue": getattr(config, "ALPHA_MIN_SIGNIFICANCE_PVALUE", 0.10),
                    "max_prediction_coins": getattr(config, "ALPHA_MAX_PREDICTION_COINS", 12),
                    "cache_ttl": getattr(config, "ALPHA_CACHE_TTL", 180),
                    "model_dir": getattr(config, "ALPHA_MODEL_DIR", "models/alpha_direction"),
                }
            ),
            health,
            affects_trading=False,
        )

    # Firewall — needs forecaster injected
    # min_confidence is env-tunable via FIREWALL_MIN_CONFIDENCE (default 0.45)
    import config as _fw_cfg
    c.firewall = _safe_init(
        "decision_firewall",
        lambda: DecisionFirewall({
            "forecaster": c.predictive_forecaster,
            "agent_scorer": c.agent_scorer,
            "event_scanner": c.event_scanner,
            "event_risk_enabled": bool(getattr(_fw_cfg, "EVENT_RISK_ENABLED", True)),
            "min_confidence": getattr(_fw_cfg, "FIREWALL_MIN_CONFIDENCE", 0.45),
            "max_signals_per_source_per_day": getattr(
                _fw_cfg, "FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY", 0
            ),
            "short_hardening_enabled": bool(getattr(_fw_cfg, "SHORT_HARDENING_ENABLED", True)),
            "short_hardening_lookback_trades": int(
                getattr(_fw_cfg, "SHORT_HARDENING_LOOKBACK_TRADES", 120)
            ),
            "short_hardening_min_closed_trades": int(
                getattr(_fw_cfg, "SHORT_HARDENING_MIN_CLOSED_TRADES", 12)
            ),
            "short_hardening_degrade_win_rate": float(
                getattr(_fw_cfg, "SHORT_HARDENING_DEGRADE_WIN_RATE", 0.45)
            ),
            "short_hardening_block_win_rate": float(
                getattr(_fw_cfg, "SHORT_HARDENING_BLOCK_WIN_RATE", 0.35)
            ),
            "short_hardening_block_net_pnl": float(
                getattr(_fw_cfg, "SHORT_HARDENING_BLOCK_NET_PNL", -1.0)
            ),
            "short_hardening_confidence_multiplier": float(
                getattr(_fw_cfg, "SHORT_HARDENING_CONFIDENCE_MULTIPLIER", 0.80)
            ),
            "short_hardening_size_multiplier": float(
                getattr(_fw_cfg, "SHORT_HARDENING_SIZE_MULTIPLIER", 0.50)
            ),
            "cooldown_seconds": int(getattr(_fw_cfg, "FIREWALL_COIN_COOLDOWN_SECONDS", 180)),
            "same_side_cooldown_seconds": int(
                getattr(_fw_cfg, "FIREWALL_SAME_SIDE_COOLDOWN_SECONDS", 900)
            ),
            "max_same_side_positions_per_coin": int(
                getattr(_fw_cfg, "FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN", 2)
            ),
            "canary_mode": bool(getattr(_fw_cfg, "FIREWALL_CANARY_MODE", False)),
            "canary_max_positions": int(getattr(_fw_cfg, "FIREWALL_CANARY_MAX_POSITIONS", 2)),
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
        _wire_kelly_sizer_from_agent_scorer(c, health)

    if "lstm_agent" in profile and getattr(config, "ENABLE_LSTM_AGENT", False):
        from src.signals.lstm_agent import LSTMAgent
        c.lstm_agent = _safe_init(
            "lstm_agent",
            lambda: LSTMAgent({
                "sequence_length": getattr(config, "LSTM_SEQUENCE_LENGTH", 30),
                "hidden_size": getattr(config, "LSTM_HIDDEN_SIZE", 64),
                "retrain_interval": getattr(config, "LSTM_RETRAIN_INTERVAL", 21600),
                "model_dir": getattr(config, "LSTM_MODEL_DIR", "models/lstm_direction"),
            }),
            health,
            affects_trading=False,
        )

    if "rl_sizer" in profile and getattr(config, "ENABLE_RL_SIZER", False) and c.kelly_sizer:
        from src.signals.rl_position_sizer import RLPositionSizer
        c.rl_sizer = _safe_init(
            "rl_sizer",
            lambda: RLPositionSizer(c.kelly_sizer, {
                "model_dir": getattr(config, "RL_SIZER_MODEL_DIR", "models/rl_sizer"),
                "retrain_interval": getattr(config, "RL_SIZER_RETRAIN_INTERVAL", 43200),
                "training_episodes": getattr(config, "RL_SIZER_TRAINING_EPISODES", 500),
            }),
            health,
        )

    if "trade_memory" in profile:
        from src.trading.trade_memory import TradeMemory
        c.trade_memory = _safe_init("trade_memory", TradeMemory, health, affects_trading=False)

    if "calibration" in profile:
        from src.signals.calibration import CalibrationTracker
        c.calibration = _safe_init("calibration", CalibrationTracker, health, affects_trading=False)

    if "llm_filter" in profile:
        from src.signals.llm_filter import LLMFilter
        c.llm_filter = _safe_init("llm_filter", LLMFilter, health)

    if "risk_policy_engine" in profile:
        from src.signals.risk_policy import RiskPolicyEngine

        def _build_risk_policy_engine():
            source_profiles = _load_json_override(
                "RISK_POLICY_SOURCE_PROFILES_JSON",
                getattr(config, "RISK_POLICY_SOURCE_PROFILES_JSON", ""),
            )
            return RiskPolicyEngine(
                {
                    "default_reward_multiple": config.RISK_POLICY_DEFAULT_REWARD_MULTIPLE,
                    "min_reward_multiple": config.RISK_POLICY_MIN_REWARD_MULTIPLE,
                    "max_reward_multiple": config.RISK_POLICY_MAX_REWARD_MULTIPLE,
                    "atr_stop_multiplier": config.RISK_POLICY_ATR_STOP_MULTIPLIER,
                    "min_stop_roe_pct": config.RISK_POLICY_MIN_STOP_ROE_PCT,
                    "max_stop_roe_pct": config.RISK_POLICY_MAX_STOP_ROE_PCT,
                    "min_stop_price_pct": config.RISK_POLICY_MIN_STOP_PRICE_PCT,
                    "max_stop_price_pct": config.RISK_POLICY_MAX_STOP_PRICE_PCT,
                    "max_take_profit_price_pct": config.RISK_POLICY_MAX_TAKE_PROFIT_PRICE_PCT,
                    "stop_vol_cap_multiplier": config.RISK_POLICY_STOP_VOL_CAP_MULTIPLIER,
                    "target_vol_cap_multiplier": config.RISK_POLICY_TARGET_VOL_CAP_MULTIPLIER,
                    "default_time_limit_hours": config.RISK_POLICY_DEFAULT_TIME_LIMIT_HOURS,
                    "default_break_even_at_r": config.RISK_POLICY_DEFAULT_BREAKEVEN_AT_R,
                    "default_break_even_buffer_roe_pct": config.RISK_POLICY_DEFAULT_BREAKEVEN_BUFFER_ROE_PCT,
                    "default_trail_after_r": config.RISK_POLICY_DEFAULT_TRAIL_AFTER_R,
                    "default_trailing_distance_ratio": config.RISK_POLICY_DEFAULT_TRAILING_DISTANCE_RATIO,
                    "source_profiles": source_profiles,
                }
            )

        c.risk_policy_engine = _safe_init(
            "risk_policy_engine",
            _build_risk_policy_engine,
            health,
        )

    if "signal_processor" in profile:
        from src.signals.signal_processor import SignalProcessor, ArenaIncubator
        c.signal_processor = _safe_init("signal_processor", SignalProcessor, health)
        c.arena_incubator = _safe_init("arena_incubator", ArenaIncubator, health, affects_trading=False)

    if "decision_engine" in profile:
        from src.signals.decision_engine import DecisionEngine
        c.decision_engine = _safe_init("decision_engine", DecisionEngine, health)

    if "alpha_arena" in profile:
        from src.signals.alpha_arena import AlphaArena
        c.arena = _safe_init(
            "alpha_arena",
            lambda: AlphaArena(
                lstm_agent=c.lstm_agent,
                risk_policy_engine=c.risk_policy_engine,
            ),
            health,
            affects_trading=False,
        )

    # ─── Options Flow ─────────────────────────────────────────
    if "options_flow" in profile:
        from src.data.options_flow import OptionsFlowScanner
        c.options_scanner = _safe_init(
            "options_flow",
            lambda: OptionsFlowScanner(source_registry=c.data_source_registry),
            health,
        )

    # ─── Polymarket ───────────────────────────────────────────
    if "polymarket" in profile:
        from src.data.polymarket_scanner import PolymarketScanner
        c.polymarket = _safe_init(
            "polymarket",
            lambda: PolymarketScanner(
                config={
                    "min_volume_threshold": float(getattr(config, "POLYMARKET_MIN_VOLUME", 10_000.0)),
                    "min_liquidity_threshold": float(
                        getattr(config, "POLYMARKET_MIN_LIQUIDITY", 1_000.0)
                    ),
                    "max_markets_per_scan": int(
                        getattr(config, "POLYMARKET_MAX_MARKETS_PER_SCAN", 100)
                    ),
                    "source_registry": c.data_source_registry,
                }
            ),
            health,
        )

    # ─── Multi-exchange ───────────────────────────────────────
    if "multi_scanner" in profile:
        from src.exchanges.scanner import MultiExchangeScanner
        c.multi_scanner = _safe_init(
            "multi_scanner",
            lambda: MultiExchangeScanner(config={"lighter_enabled": config.LIGHTER_ENABLED}),
            health,
        )

    # Structured event scanner (official macro / policy releases)
    if "event_scanner" in profile and getattr(config, "EVENT_SCANNER_ENABLED", True):
        from src.data.event_scanner import EventScanner
        c.event_scanner = _safe_init(
            "event_scanner",
            EventScanner,
            health,
            affects_trading=False,
        )
        _wire_event_scanner(c, health)

    # Macro regime overlay (external data → protective risk posture)
    if "macro_regime" in profile and getattr(config, "MACRO_REGIME_ENABLED", True):
        from src.data.macro_regime_scraper import MacroRegimeScraper
        c.macro_regime = _safe_init(
            "macro_regime",
            MacroRegimeScraper,
            health,
            affects_trading=False,
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
                rl_sizer=c.rl_sizer,
                trade_memory=c.trade_memory,
                calibration=c.calibration,
                regime_forecaster=c.predictive_forecaster,
                risk_policy_engine=c.risk_policy_engine,
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
            rl_sizer=c.rl_sizer,
            trade_memory=c.trade_memory,
            calibration=c.calibration,
            llm_filter=c.llm_filter,
            risk_policy_engine=c.risk_policy_engine,
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
                max_daily_loss=float(getattr(config, "LIVE_MAX_DAILY_LOSS_USD", 100.0)),
                max_order_usd=float(getattr(config, "LIVE_MAX_ORDER_USD", 100.0)),
                regime_forecaster=c.predictive_forecaster,
                risk_policy_engine=c.risk_policy_engine,
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
        if c.paper_trader:
            c.paper_trader.shadow_tracker = c.shadow_tracker
        if c.copy_trader:
            c.copy_trader.shadow_tracker = c.shadow_tracker

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
            quarantined = db.quarantine_invalid_traders()
            if quarantined:
                logger.warning(
                    "Position monitor skipped %d malformed trader row(s) from runtime DB",
                    len(quarantined),
                )
            top_traders = db.get_active_traders(valid_only=True)[:20]
            if top_traders:
                c.position_monitor = _safe_init(
                    "position_monitor",
                    lambda: _start_position_monitor(top_traders),
                    health,
                )
        except Exception as exc:
            logger.warning("  FAIL position_monitor - %s", exc)

    # ─── WebSocket feed ───────────────────────────────────────
    try:
        from src.data.hyperliquid_client import start_websocket
        start_websocket(coins=["BTC", "ETH", "SOL", "DOGE", "ARB", "OP"])
        logger.info("  OK websocket_feed")
    except Exception as exc:
        logger.warning("  FAIL websocket_feed - %s (REST fallback active)", exc)

    # ─── Dashboard ────────────────────────────────────────────
    if "dashboard" in profile:
        try:
            from src.ui.dashboard import start_dashboard, set_v2_components, set_live_trader
            set_v2_components(
                firewall=c.firewall,
                regime_detector=c.regime_detector,
                arena=c.arena,
                agent_scorer=c.agent_scorer,
                kelly_sizer=c.kelly_sizer,
                trade_memory=c.trade_memory,
                calibration=c.calibration,
                llm_filter=c.llm_filter,
                liquidation_strategy=c.liquidation_strategy,
                signal_processor=c.signal_processor,
                arena_incubator=c.arena_incubator,
                decision_engine=c.decision_engine,
                multi_scanner=c.multi_scanner,
                event_scanner=c.event_scanner,
                shadow_tracker=c.shadow_tracker,
                health_registry=health,
                copy_trader=c.copy_trader,
            )
            if c.live_trader:
                set_live_trader(c.live_trader)
            c.dashboard = start_dashboard(options_scanner=c.options_scanner)
            logger.info("  OK dashboard")
        except Exception as exc:
            logger.warning("  FAIL dashboard - %s", exc)

    # ─── Telegram ─────────────────────────────────────────────
    try:
        from src.notifications import telegram_bot as tg
        if tg.is_configured():
            tg.send_startup_message()
            logger.info("  OK telegram")
        else:
            logger.info("  SKIP telegram (not configured)")
    except Exception as exc:
        logger.debug("  FAIL telegram - %s", exc)

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
    "llm_filter":             "llm_filter",
    "risk_policy_engine":     "risk_policy_engine",
    "signal_processor":       "signal_processor",
    "arena_incubator":        "arena_incubator",
    "decision_engine":        "decision_engine",
    "arena":                  "alpha_arena",
    "options_scanner":        "options_flow",
    "polymarket":             "polymarket",
    "multi_scanner":          "multi_scanner",
    "event_scanner":          "event_scanner",
    "alpha_pipeline":         "alpha_pipeline",
    "copy_trader":            "copy_trader",
    "paper_trader":           "paper_trader",
    "live_trader":            "live_trader",
    "cross_venue_hedger":     "cross_venue_hedger",
    "shadow_tracker":         "shadow_tracker",
    "adaptive_bot_detector":  "adaptive_bot_detector",
    "regime_strategy_filter": "regime_strategy_filter",
    "position_monitor":       "position_monitor",
    "lstm_agent":             "lstm_agent",
    "rl_sizer":               "rl_sizer",
    "macro_regime":           "macro_regime",
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

    try:
        from src.notifications import telegram_bot as tg

        if tg.is_configured():
            tg.heartbeat()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_position_monitor(top_traders):
    from src.notifications.ws_position_monitor import PositionMonitor
    pm = PositionMonitor(max_signal_queue=500)
    pm.start([t["address"] for t in top_traders])
    return pm
