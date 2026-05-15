"""
Trading Cycle (Tier 2)
======================
Lightweight trading cycle: score strategies, detect regime, trade.
Uses existing DB data from last discovery — no leaderboard scanning.
Runs every ~5 minutes to react to market changes quickly.

Extracted from ``HyperliquidResearchBot._run_trading_cycle``.
"""
import logging
import copy
from datetime import datetime, timezone

import config
from src.core import clock_provider
from src.data import database as db
from src.core.live_execution import (
    get_execution_open_positions,
    is_live_trading_active,
    mirror_executed_trades_to_live,
    sync_shadow_book_to_live,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_OPTIONS_FLOW_MIN_CONVICTION_PCT = float(
    getattr(config, "OPTIONS_FLOW_MIN_CONVICTION_PCT", 30.0)
)
_ARENA_CHAMPION_MIN_FITNESS = float(
    getattr(config, "ARENA_CHAMPION_MIN_FITNESS", 0.15)
)
_ARENA_CHAMPION_MIN_TRADES = int(
    getattr(config, "ARENA_CHAMPION_MIN_TRADES", 5)
)
_ARENA_CHAMPION_MIN_WIN_RATE = float(
    getattr(config, "ARENA_CHAMPION_MIN_WIN_RATE", 0.45)
)
_ARENA_COIN_UNIVERSE = [
    str(coin).upper()
    for coin in getattr(config, "ARENA_COIN_UNIVERSE", ["BTC", "ETH", "SOL"])
    if str(coin).strip()
]
_ARENA_MAX_COINS = int(getattr(config, "ARENA_MAX_COINS", 3))
_ARENA_INTERVAL = str(getattr(config, "ARENA_INTERVAL", "1h") or "1h").strip() or "1h"
_ARENA_LOOKBACK_HOURS = int(getattr(config, "ARENA_LOOKBACK_HOURS", 720))


def _apply_dynamic_risk_policy(container, trade_signal, regime_data=None, source_policy=None):
    engine = getattr(container, "risk_policy_engine", None)
    if not engine:
        return trade_signal
    try:
        return engine.apply(trade_signal, regime_data=regime_data, source_policy=source_policy)
    except Exception as exc:
        logger.debug("  Risk policy apply error for %s: %s", getattr(trade_signal, "coin", "?"), exc)
        return trade_signal


def _get_arena_coin_universe():
    coins = _ARENA_COIN_UNIVERSE or ["BTC"]
    unique = []
    for coin in coins:
        norm = str(coin).upper().strip()
        if not norm or norm in unique:
            continue
        unique.append(norm)
    return unique[: max(1, _ARENA_MAX_COINS)]


def _fetch_arena_candle_universe():
    from src.core.api_manager import get_manager, Priority

    manager = get_manager()
    end_ms = int(clock_provider.utc_now().timestamp() * 1000)
    start_ms = int((clock_provider.utc_now().timestamp() - (_ARENA_LOOKBACK_HOURS * 3600)) * 1000)
    candle_map = {}

    for coin in _get_arena_coin_universe():
        try:
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": _ARENA_INTERVAL,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            }
            raw = manager.post(payload=payload, priority=Priority.LOW, timeout=15)
            if not isinstance(raw, list) or len(raw) < 50:
                continue
            candle_map[coin] = [
                {
                    "open": float(c.get("o", 0)),
                    "high": float(c.get("h", 0)),
                    "low": float(c.get("l", 0)),
                    "close": float(c.get("c", 0)),
                    "volume": float(c.get("v", 0)),
                    "timestamp": int(c.get("t", c.get("time", 0)) or 0),
                }
                for c in raw
            ]
        except Exception as exc:
            logger.debug("  Arena candle fetch failed for %s: %s", coin, exc)
    return candle_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _live_safety_stop_reason(live_trader) -> str:
    """Return the actual reason live entries are stopped.

    ``LiveTrader.check_daily_loss()`` intentionally acts as a shared safety
    gate for daily loss, external kill switch, persisted kill switch and
    dualwrite health.  Keep the trading-cycle log honest so operators do not
    chase a fake daily-loss incident when the active stop came from another
    guard.
    """
    state = {}
    try:
        if hasattr(live_trader, "get_safety_stop_reason"):
            return str(live_trader.get_safety_stop_reason())
        if hasattr(live_trader, "get_kill_switch_state"):
            state = live_trader.get_kill_switch_state() or {}
        elif hasattr(live_trader, "get_stats"):
            stats = live_trader.get_stats() or {}
            state = {
                "active": bool(stats.get("kill_switch_active", False)),
                "reason": stats.get("kill_switch_reason"),
                "status_reason": stats.get("status_reason"),
            }
    except Exception:
        state = {}

    if state.get("active"):
        reason = str(state.get("reason") or state.get("status_reason") or "active")
        return f"kill_switch_active:{reason}"
    return "daily_loss_limit_exceeded"


def _inject_forecaster_signals(container, regime_data):
    """Feed options flow + polymarket into the predictive forecaster."""
    forecaster = container.predictive_forecaster
    if not forecaster:
        return

    # Options flow
    try:
        convictions = getattr(container.options_scanner, "top_convictions", None)
        if convictions:
            forecaster.update_options_flow(convictions)
            logger.debug("  Forecaster <- %d options convictions", len(convictions))
    except Exception as exc:
        logger.debug("  Forecaster options injection error: %s", exc)

    # Polymarket
    try:
        if container.polymarket:
            pm_sentiment = container.polymarket.get_market_sentiment()
            forecaster.update_polymarket_sentiment(pm_sentiment)
            logger.debug("  Forecaster <- Polymarket sentiment: %s", pm_sentiment.get("sentiment", "?"))
    except Exception as exc:
        logger.debug("  Forecaster polymarket injection error: %s", exc)


_REGIME_MAP_FORECASTER_TO_DETECTOR = {
    "crash": {"trending_down", "volatile"},
    "bullish": {"trending_up"},
    "neutral": {"ranging", "unknown", "low_liquidity"},
}


def _reconcile_regimes(regime_data: dict, container) -> dict:
    """
    Compare regime detector (technical, 6-class) with forecaster (predictive, 3-class).
    Annotates regime_data with agreement metadata and applies conservative crash
    overrides when predictive crash risk is high.
    """
    forecaster = container.predictive_forecaster if container else None
    if not forecaster or not regime_data:
        return regime_data

    try:
        pred = forecaster.predict_regime("BTC")
    except Exception:
        return regime_data

    if not pred or not isinstance(pred, dict):
        return regime_data

    pred_regime = pred.get("regime", "neutral")           # crash / neutral / bullish
    pred_conf = float(pred.get("confidence", 0))
    pred_synthetic = bool(pred.get("synthetic_warm_start", False))
    pred_training_source = pred.get("training_source", "unknown")
    det_regime = regime_data.get("overall_regime", "unknown")  # trending_up / trending_down / ranging / volatile / ...
    det_conf = float(regime_data.get("overall_confidence", 0))

    # Check compatibility
    compatible_detector_classes = _REGIME_MAP_FORECASTER_TO_DETECTOR.get(pred_regime, set())
    agree = det_regime in compatible_detector_classes

    regime_data["forecaster_regime"] = pred_regime
    regime_data["forecaster_confidence"] = pred_conf
    regime_data["detector_regime"] = det_regime
    regime_data["detector_confidence"] = det_conf
    regime_data["regime_agreement"] = agree
    regime_data["forecaster_training_source"] = pred_training_source
    regime_data["forecaster_synthetic_warm_start"] = pred_synthetic

    if not agree and min(pred_conf, det_conf) >= 0.5:
        # Conservative policy: if forecaster says crash, override to crash-equivalent.
        # Threshold raised from 0.60 -> 0.75: observed in production the
        # XGBoost forecaster would sit at 63–68% "crash" for hours while
        # the detector reported "trending_up", flipping regime to "volatile"
        # and dragging every signal's confidence -0.07 via macro overlay.
        # A 75% bar requires the forecaster to be meaningfully confident
        # before it can veto the consensus detector read.
        if pred_synthetic and pred_regime == "crash":
            logger.info(
                "  REGIME DISAGREEMENT: detector=%s (%.0f%%) vs forecaster=%s "
                "(%.0f%%, source=%s) -- synthetic warm-start cannot override detector",
                det_regime, det_conf * 100, pred_regime, pred_conf * 100, pred_training_source,
            )
        elif pred_regime == "crash" and pred_conf >= 0.75:
            logger.warning(
                "  REGIME DISAGREEMENT: detector=%s (%.0f%%) vs forecaster=%s (%.0f%%) -- "
                "applying crash-protective override",
                det_regime, det_conf * 100, pred_regime, pred_conf * 100,
            )
            regime_data["overall_regime"] = "volatile"
            regime_data["regime_override"] = "forecaster_crash"
            # Treat detector/forecaster disagreement as defensive, not as
            # permission to invert the trade book.  Dedicated hedge logic can
            # reduce risk; ordinary entry signals should not open fresh
            # counter-trend shorts while the technical detector still shows a
            # high-confidence uptrend.
            if det_regime == "trending_up":
                regime_data["countertrend_block_side"] = "short"
            elif det_regime == "trending_down":
                regime_data["countertrend_block_side"] = "long"
            # Suppress bullish strategy activation
            guidance = copy.deepcopy(regime_data.get("strategy_guidance", {}))
            guidance["pause"] = list(set(guidance.get("pause", []) + guidance.get("activate", [])))
            guidance["activate"] = []
            regime_data["strategy_guidance"] = guidance
        else:
            logger.info(
                "  REGIME DISAGREEMENT: detector=%s (%.0f%%) vs forecaster=%s (%.0f%%) -- "
                "no crash override applied",
                det_regime, det_conf * 100, pred_regime, pred_conf * 100,
            )
    elif agree:
        logger.debug(
            "  Regime consensus: detector=%s <-> forecaster=%s (both confident)",
            det_regime, pred_regime,
        )

    return regime_data


def _apply_macro_regime_overlay(container, regime_data: dict) -> dict:
    """
    Phase 3d3: Apply macro regime overlay as a protective posture adjustment.

    This is NOT event-driven trading — it adjusts the regime_data dict so that
    downstream risk policy, decision firewall, and position sizing all respond
    to the external macro environment.

    The macro scraper polls slowly (15-min cache) and the output is a risk
    posture that modifies:
      - strategy_guidance.size_modifier (multiplicative)
      - regime_data.macro_* fields for risk_policy_engine to read
      - strategy_guidance.pause (at extreme risk levels)
    """
    macro = getattr(container, "macro_regime", None)
    if not macro:
        return regime_data

    try:
        posture = macro.get_risk_posture()
    except Exception as exc:
        logger.debug("  Macro regime overlay error: %s", exc)
        return regime_data

    if not posture or posture.get("macro_risk_level") == "normal":
        return regime_data

    level = posture.get("macro_risk_level", "normal")
    size_mod = posture.get("size_modifier", 1.0)
    conf_drag = posture.get("confidence_drag", 0.0)
    block = posture.get("block_new_entries", False)
    reasons = posture.get("reasons", [])

    # Stamp macro data onto regime_data so risk_policy + firewall can see it
    regime_data["macro_risk_level"] = level
    regime_data["macro_score"] = posture.get("macro_score", 0.0)
    regime_data["macro_size_modifier"] = size_mod
    regime_data["macro_confidence_drag"] = conf_drag
    regime_data["macro_block_new_entries"] = block
    regime_data["macro_reasons"] = reasons

    # Apply size modifier to strategy guidance
    guidance = copy.deepcopy(regime_data.get("strategy_guidance", {}))
    current_size_mod = float(guidance.get("size_modifier", 1.0))
    guidance["size_modifier"] = round(current_size_mod * size_mod, 3)
    regime_data["strategy_guidance"] = guidance

    # At extreme levels, pause all strategies
    if block:
        guidance["pause"] = list(set(guidance.get("pause", []) + ["all"]))
        logger.warning(
            "  MACRO REGIME: %s -- blocking new entries (score=%.2f, reasons=%s)",
            level.upper(), posture.get("macro_score", 0), reasons[:2],
        )
    elif level in ("high", "elevated"):
        logger.info(
            "  MACRO REGIME: %s -- size_mod=%.2f, conf_drag=%.2f (reasons=%s)",
            level, size_mod, conf_drag, reasons[:2],
        )
    else:
        logger.debug("  Macro regime overlay: %s (no action)", level)

    return regime_data


def _float_or_default(value, default=0.0) -> float:
    try:
        out = float(value)
        return out if out == out else float(default)
    except (TypeError, ValueError):
        return float(default)


_BULLISH_REGIMES = {"trending_up", "bullish", "bull", "uptrend", "trend_up", "risk_on"}
_BEARISH_REGIMES = {
    "trending_down",
    "bearish",
    "bear",
    "downtrend",
    "trend_down",
    "crash",
    "panic",
    "risk_off",
}


def _normalise_coin_regime(item: dict) -> str:
    return str(
        item.get("regime")
        or item.get("predicted_regime")
        or item.get("overall_regime")
        or ""
    ).strip().lower()


def _direction_from_coin_item(
    item: dict,
    *,
    min_conf: float,
    min_momentum: float,
    min_volume_ratio: float,
) -> str:
    confidence = _float_or_default(item.get("confidence"), 0.0)
    if confidence <= 0:
        confidence = _float_or_default(item.get("regime_confidence"), 0.0)
    momentum = _float_or_default(item.get("momentum"), 0.0)
    trend_direction = _float_or_default(item.get("trend_direction"), 0.0)
    volume_ratio = _float_or_default(item.get("volume_ratio"), 1.0)
    regime = _normalise_coin_regime(item)

    if confidence < min_conf or volume_ratio < min_volume_ratio:
        return ""
    if momentum >= min_momentum or trend_direction >= min_momentum or regime in _BULLISH_REGIMES:
        return "up"
    if momentum <= -min_momentum or trend_direction <= -min_momentum or regime in _BEARISH_REGIMES:
        return "down"
    return ""


def _apply_directional_strategy_guidance(
    regime_data: dict,
    *,
    direction: str,
    block_side: str,
    agreeing: list[str],
    min_agree: int,
    reason: str,
) -> dict:
    guidance = copy.deepcopy(regime_data.get("strategy_guidance", {}) or {})
    pause = set(guidance.get("pause", []) or [])
    activate = set(guidance.get("activate", []) or [])
    if direction == "up":
        pause.update({"momentum_short", "contrarian", "mean_reversion", "scalping", "short_momentum"})
        activate.update({"momentum_long", "trend_following", "breakout"})
    else:
        pause.update({"momentum_long", "trend_following", "breakout", "contrarian", "mean_reversion", "scalping"})
        activate.update({"momentum_short", "short_momentum"})

    guidance["pause"] = sorted(pause)
    guidance["activate"] = sorted(activate)
    regime_data["strategy_guidance"] = guidance
    regime_data["countertrend_block_side"] = block_side
    regime_data["global_momentum_override"] = {
        "direction": direction,
        "block_side": block_side,
        "agreeing_coins": agreeing,
        "min_agreeing_coins": min_agree,
        "reason": reason,
    }

    logger.warning(
        "  GLOBAL MOMENTUM OVERRIDE: %s across %s -- blocking %s entries (%s)",
        direction.upper(),
        ",".join(agreeing),
        block_side,
        reason,
    )
    return regime_data


def _apply_global_momentum_override(container, regime_data: dict) -> dict:
    """Block countertrend entries when the major coins agree directionally."""
    if not getattr(config, "GLOBAL_MOMENTUM_OVERRIDE_ENABLED", True):
        return regime_data
    if not regime_data or not isinstance(regime_data, dict):
        return regime_data

    per_coin = regime_data.get("per_coin", {})
    if not isinstance(per_coin, dict) or not per_coin:
        return regime_data

    core_coins = [
        str(coin).upper()
        for coin in getattr(config, "GLOBAL_MOMENTUM_CORE_COINS", ["BTC", "ETH", "SOL"])
        if str(coin).strip()
    ]
    min_agree = int(getattr(config, "GLOBAL_MOMENTUM_MIN_AGREEING_COINS", 2))
    min_conf = float(getattr(config, "GLOBAL_MOMENTUM_MIN_CONFIDENCE", 0.55))
    min_momentum = float(getattr(config, "GLOBAL_MOMENTUM_MIN_MOMENTUM", 0.003))
    min_volume_ratio = float(getattr(config, "GLOBAL_MOMENTUM_MIN_VOLUME_RATIO", 0.80))

    bullish = []
    bearish = []
    for coin in core_coins:
        item = dict(per_coin.get(coin, {}) or {})
        if not item:
            continue
        item_direction = _direction_from_coin_item(
            item,
            min_conf=min_conf,
            min_momentum=min_momentum,
            min_volume_ratio=min_volume_ratio,
        )
        if item_direction == "up":
            bullish.append(coin)
        elif item_direction == "down":
            bearish.append(coin)

    direction = ""
    agreeing = []
    block_side = ""
    if len(bullish) >= min_agree and len(bullish) > len(bearish):
        direction = "up"
        agreeing = bullish
        block_side = "short"
    elif len(bearish) >= min_agree and len(bearish) > len(bullish):
        direction = "down"
        agreeing = bearish
        block_side = "long"

    reason = "core_coin_consensus"
    if not direction and bool(getattr(config, "BTC_MARKET_LEADER_GUARD_ENABLED", True)):
        leader_coin = str(getattr(config, "BTC_MARKET_LEADER_COIN", "BTC") or "BTC").strip().upper()
        leader_item = dict(per_coin.get(leader_coin, {}) or {})
        if leader_item:
            leader_direction = _direction_from_coin_item(
                leader_item,
                min_conf=float(getattr(config, "BTC_MARKET_LEADER_MIN_CONFIDENCE", min_conf)),
                min_momentum=float(getattr(config, "BTC_MARKET_LEADER_MIN_MOMENTUM", 0.003)),
                min_volume_ratio=float(
                    getattr(config, "BTC_MARKET_LEADER_MIN_VOLUME_RATIO", min_volume_ratio)
                ),
            )
            if leader_direction in {"up", "down"}:
                direction = leader_direction
                agreeing = [leader_coin]
                block_side = "short" if leader_direction == "up" else "long"
                reason = "btc_market_leader"

    if not direction:
        return regime_data

    regime_data = _apply_directional_strategy_guidance(
        regime_data,
        direction=direction,
        block_side=block_side,
        agreeing=agreeing,
        min_agree=min_agree,
        reason=reason,
    )

    if (
        bool(getattr(config, "GLOBAL_MOMENTUM_CLOSE_COUNTERTREND", False))
        and is_live_trading_active(container)
        and getattr(container, "live_trader", None)
    ):
        try:
            for pos in get_execution_open_positions(container):
                coin = str(pos.get("coin") or "").upper()
                side = str(pos.get("side") or "").strip().lower()
                if coin and side == block_side:
                    container.live_trader.close_position(coin)
                    logger.warning("  GLOBAL MOMENTUM: closed countertrend %s %s", side, coin)
        except Exception as exc:
            logger.warning("  Global momentum close-countertrend failed: %s", exc)

    return regime_data


def _run_hedger(container, regime_data):
    """Cross-venue hedging — auto-hedge on crash regime."""
    hedger = container.cross_venue_hedger
    if not hedger or not regime_data:
        return
    try:
        pred_regime = {}
        if container.predictive_forecaster:
            pred_regime = container.predictive_forecaster.predict_regime("BTC")
        else:
            pred_regime = {
                "regime": regime_data.get("overall_regime", "neutral"),
                "confidence": regime_data.get("overall_confidence", 0),
            }
        open_trades = get_execution_open_positions(container)
        hedge_result = hedger.check_and_hedge(pred_regime, open_trades)
        if hedge_result.get("action") != "idle":
            logger.info(
                "  Hedger: %s | placed=%d, closed=%d, coins=%s",
                hedge_result["action"], hedge_result["hedges_placed"],
                hedge_result["hedges_closed"], hedge_result["coins_affected"],
            )
    except Exception as exc:
        logger.debug("  Cross-venue hedger error: %s", exc)


def _record_shadow_trade(container, closed_trade, pnl, return_pct, entry):
    """Record a closed trade in the shadow tracker.

    The signal_source written here flows into the calibration tracker and
    becomes a key in calibration_records. Earlier versions defaulted to
    ``strategy:unknown`` whenever a row lacked ``strategy_type`` — that
    polluted the calibration table with one fat untagged bucket which
    logged as ``[strategy:unknown|_|any(ECE=0.47,n=83)]`` and was
    quarantined for poor calibration. The fix:
      1. Walk a fuller fallback chain (top-level field, then metadata).
      2. If we still can't identify the source, log a WARNING so the
         upstream pipeline gap is visible, and tag the trade with
         ``strategy:untagged`` (distinct from ``unknown``) so the
         existing calibration data isn't double-counted.
    """
    tracker = container.shadow_tracker
    if not tracker:
        return
    try:
        meta = closed_trade.get("metadata") or {}
        if isinstance(meta, str):
            try:
                import json as _json
                meta = _json.loads(meta or "{}")
            except (ValueError, TypeError):
                meta = {}

        # Prefer an explicit metadata.source if present (copy_trade:0xabc,
        # options_flow, polymarket, etc.).
        source = str(meta.get("source") or "").strip()
        stype = (
            str(closed_trade.get("strategy_type") or "").strip()
            or str(meta.get("strategy_type") or "").strip()
        )

        if source:
            signal_source = source
        elif stype and stype.lower() != "unknown":
            signal_source = f"strategy:{stype.lower()}"
        else:
            # Not fatal, but visible: a trade reached the shadow tracker
            # without a tagged source. Tag it distinctly so calibration
            # doesn't merge it with the legacy ``strategy:unknown`` bucket.
            logger.warning(
                "ShadowTracker: untagged trade (coin=%s, side=%s, pnl=%.3f, "
                "metadata_keys=%s) — upstream pipeline did not set source/"
                "strategy_type. Tagging as 'strategy:untagged'.",
                closed_trade.get("coin", "?"),
                closed_trade.get("side", "?"),
                float(pnl or 0),
                sorted(meta.keys()) if isinstance(meta, dict) else "non-dict-meta",
            )
            signal_source = "strategy:untagged"

        tracker.record_trade({
            "signal_source": signal_source,
            "coin": closed_trade.get("coin", "UNK"),
            "side": closed_trade.get("side", "long"),
            "entry_price": entry,
            "exit_price": closed_trade.get("exit_price", entry),
            "size": closed_trade.get("size", 0),
            "pnl": pnl,
            "pnl_pct": return_pct * 100,
            "entry_ts": closed_trade.get("entry_ts") or None,
            "exit_ts": closed_trade.get("exit_ts") or None,
            "regime_at_entry": closed_trade.get("regime") or None,
            "confidence": float(meta.get("confidence", 0.5)),
        })
    except Exception as exc:
        logger.debug("  ShadowTracker record error: %s", exc)


def _is_insufficient_margin_rejection(result) -> bool:
    """Return True when a rejection payload indicates insufficient margin."""
    if not isinstance(result, dict):
        return False

    reason = str(result.get("reason", "")).strip().lower()
    if "insufficient_margin" in reason:
        return True

    errors = result.get("errors")
    messages = []
    if isinstance(errors, list):
        messages.extend(str(e) for e in errors if e)
    elif errors:
        messages.append(str(errors))

    message = result.get("message")
    if message:
        messages.append(str(message))

    return any("insufficient margin" in msg.lower() for msg in messages)


def _drawdown_from_reference(account_balance: float, baseline: float = 10_000.0) -> float:
    baseline = max(float(baseline), 1.0)
    return max(0.0, (baseline - max(float(account_balance), 0.0)) / baseline)


def _get_dynamic_sizing(container, strategy_key: str, account_balance: float,
                        signal_confidence: float, regime_data=None, coin: str = "",
                        volatility: float = 0.02):
    """Use Kelly by default; RL sizing is shadow-only unless explicitly enabled."""
    rl_sizer = getattr(container, "rl_sizer", None)
    rl_apply_enabled = bool(getattr(config, "RL_SIZER_APPLY_TO_ORDERS", False))
    if rl_sizer and rl_apply_enabled:
        regime = "unknown"
        if regime_data:
            per_coin = dict(regime_data.get("per_coin", {}).get(coin, {}) or {})
            regime = str(per_coin.get("regime") or regime_data.get("overall_regime") or "unknown")
            volatility = float(per_coin.get("atr_pct", volatility) or volatility)
        return rl_sizer.get_sizing(
            strategy_key=strategy_key,
            account_balance=account_balance,
            signal_confidence=signal_confidence,
            regime=regime,
            recent_volatility=float(volatility or 0.02),
            drawdown_from_peak=_drawdown_from_reference(
                account_balance,
                float(getattr(config, "PAPER_TRADING_INITIAL_BALANCE", 10_000.0)),
            ),
        )
    elif rl_sizer and not getattr(container, "_rl_sizer_shadow_logged", False):
        logger.info("  RL sizer is shadow-only; Kelly/default sizing controls live orders")
        try:
            container._rl_sizer_shadow_logged = True
        except Exception:
            pass

    kelly_sizer = getattr(container, "kelly_sizer", None)
    if kelly_sizer:
        return kelly_sizer.get_sizing(
            strategy_key=strategy_key,
            account_balance=account_balance,
            signal_confidence=signal_confidence,
        )
    return None


def _train_rl_sizer_if_due(container):
    """Fire-and-forget: RLPositionSizer.train() is non-blocking; it
    spawns a daemon thread and logs completion from inside the thread."""
    rl_sizer = getattr(container, "rl_sizer", None)
    if not rl_sizer:
        return
    try:
        rl_sizer.train()
    except Exception as exc:
        logger.warning("  RL sizer training kickoff error: %s", exc)


def _execute_signal_live(container, trade_signal, source_label: str, bypass_firewall: bool = True):
    """Execute a TradeSignal directly on the live trader and log the outcome.

    Note: this is the direct-to-live path used by LCRS, Options Flow, and
    Alpha Arena -- it bypasses both the paper executor and (by default)
    the DecisionFirewall. Without the safety brakes below it would skip
    every gate added for the paper->live mirror path. The funding-
    divergence and unknown-source gates are applied here too because
    they're cheap, asymmetric (only block), and the same risks apply
    regardless of which path the signal came in on. Promotion gate is
    *not* applied here because LCRS/options/arena don't have the same
    strategy_id/agent_score tracking model as paper-mirror trades.
    """
    trader = getattr(container, "live_trader", None)
    if not trader or not is_live_trading_active(container):
        return None

    # Funding-divergence safety brake -- block longs into crowded
    # selloffs / shorts into crowded rallies regardless of source.
    try:
        from src.signals.funding_divergence import should_block_side
        side_val = (
            trade_signal.side.value
            if hasattr(trade_signal.side, "value")
            else str(trade_signal.side)
        )
        block, reason = should_block_side(side_val)
        if block:
            logger.warning(
                "  LIVE %s blocked by funding-divergence brake: %s %s (%s)",
                source_label,
                side_val.upper(),
                trade_signal.coin,
                reason,
            )
            return None
    except Exception as exc:
        logger.debug("Funding-divergence check failed for %s (fail-open): %s", source_label, exc)

    # Unknown-source gate -- mirror the firewall check at the direct
    # live path so an untagged signal can't slip past via this shortcut.
    try:
        from src.signals.decision_firewall import DecisionFirewall
        source = getattr(trade_signal, "source", None)
        if hasattr(source, "value"):
            source = source.value
        key = str(source or "unknown").strip().lower() or "unknown"
        strategy_type = str(getattr(trade_signal, "strategy_type", "") or "").strip().lower()
        if key == "copy_trade":
            trader_addr = str(getattr(trade_signal, "trader_address", "") or "").strip().lower()
            source_key = f"{key}:{trader_addr}" if trader_addr else key
        elif strategy_type:
            source_key = f"{key}:{strategy_type}"
        else:
            source_key = key
        if DecisionFirewall._is_unknown_source_key(source_key):
            logger.warning(
                "  LIVE %s blocked by unknown-source gate: %s %s (source_key=%s)",
                source_label,
                trade_signal.side.value.upper() if hasattr(trade_signal.side, "value") else trade_signal.side,
                trade_signal.coin,
                source_key,
            )
            return None
    except Exception as exc:
        logger.debug("Unknown-source check failed for %s (fail-open): %s", source_label, exc)

    try:
        result = trader.execute_signal(trade_signal, bypass_firewall=bypass_firewall)
    except Exception as exc:
        logger.error(
            "  LIVE %s execution error for %s %s: %s",
            source_label,
            trade_signal.side.value.upper(),
            trade_signal.coin,
            exc,
        )
        return None

    if not result:
        logger.info(
            "  LIVE %s skipped: %s %s (no execution result)",
            source_label,
            trade_signal.side.value.upper(),
            trade_signal.coin,
        )
        return None

    if result and result.get("status") not in ("error", "rejected"):
        logger.info(
            "  LIVE %s executed: %s %s (%s)",
            source_label,
            trade_signal.side.value.upper(),
            trade_signal.coin,
            result.get("status", "ok"),
        )
        return result

    if result.get("status") == "rejected":
        if _is_insufficient_margin_rejection(result):
            logger.warning(
                "  LIVE %s skipped due to insufficient margin: %s %s",
                source_label,
                trade_signal.side.value.upper(),
                trade_signal.coin,
            )
        else:
            logger.warning(
                "  LIVE %s rejected: %s %s -> %s",
                source_label,
                trade_signal.side.value.upper(),
                trade_signal.coin,
                result,
            )
        return None

    logger.error(
        "  LIVE %s failed: %s %s -> %s",
        source_label,
        trade_signal.side.value.upper(),
        trade_signal.coin,
        result,
    )
    return None


# ---------------------------------------------------------------------------
# Main trading cycle
# ---------------------------------------------------------------------------

def run_trading_cycle(container, cycle_count: int) -> None:
    """
    Execute one trading cycle using the subsystems in *container*.
    """
    from src.notifications import telegram_bot as tg

    # Kill switch check
    if container.live_trader and not container.live_trader.dry_run:
        container.live_trader.update_daily_pnl_from_fills()
        if container.live_trader.check_daily_loss():
            logger.warning(
                "LIVE SAFETY STOP -- %s; skipping new live entries",
                _live_safety_stop_reason(container.live_trader),
            )

    # Soft calibration pause -- not sticky, recovers automatically when
    # global ECE drops back below the threshold. Paper trading still
    # runs through this cycle so the calibrator keeps learning.
    cal = getattr(container, "calibration", None)
    if (
        cal is not None
        and getattr(cal, "is_live_paused", None)
        and container.live_trader
        and not container.live_trader.dry_run
    ):
        try:
            if cal.is_live_paused():
                ece = cal.get_ece("global")
                logger.warning(
                    "LIVE PAUSE (calibration) -- global ECE=%.3f >= %.3f; "
                    "skipping new live entries this cycle. Paper continues "
                    "to feed the calibrator.",
                    ece if ece is not None else float("nan"),
                    cal.live_pause_ece,
                )
                container.live_trader._calibration_live_paused_this_cycle = True
            else:
                container.live_trader._calibration_live_paused_this_cycle = False
        except Exception as exc:
            logger.debug("Calibration live-pause check failed: %s", exc)

        # Sweep for orphaned positions (opened successfully but SL/TP
        # placement was skipped due to an upstream error like the
        # get_positions float(dict) crash).  Safe to call every cycle —
        # protect_orphaned_positions checks for existing reduce-only
        # orders and no-ops for protected positions.
        try:
            container.live_trader.protect_orphaned_positions()
        except Exception as exc:
            logger.warning("Orphan protection sweep failed: %s", exc)

    logger.info("=" * 60)
    logger.info("Starting trading cycle #%d", cycle_count)
    logger.info("=" * 60)

    try:
        # ── Phase 3: Score all strategies ──
        logger.info("Phase 3: Strategy Scoring")
        score_results = container.scorer.score_all_strategies() if container.scorer else []
        logger.info("  Scored %d strategies", len(score_results))

        # ── Phase 3b: Multi-Exchange Volume Analysis ──
        logger.info("Phase 3b: Multi-Exchange Volume Analysis")
        market_overview = {}
        try:
            if container.exchange_agg:
                market_overview = container.exchange_agg.get_market_overview()
                logger.info(
                    "  Market bias: %s (score: %+.4f)",
                    market_overview.get("overall_bias", "?"),
                    market_overview.get("overall_bias_score", 0),
                )
                if tg.is_configured():
                    tg.notify_market_bias(market_overview)
        except Exception as exc:
            logger.warning("  Exchange aggregator error: %s", exc)

        # ── Phase 3c: Options Flow Scan ──
        logger.info("Phase 3c: Options Flow Scan")
        try:
            if container.options_scanner:
                flow_result = container.options_scanner.scan_flow()
                logger.info(
                    "  Unusual prints: %d  Top convictions: %d",
                    flow_result.get("unusual_prints", 0),
                    flow_result.get("top_convictions", 0),
                )
                if tg.is_configured() and container.options_scanner.top_convictions:
                    for conv in container.options_scanner.top_convictions[:3]:
                        if conv.get("conviction_pct", 0) >= _OPTIONS_FLOW_MIN_CONVICTION_PCT:
                            tg.notify_strong_signal(
                                coin=conv["ticker"],
                                side=str(conv.get("direction", "")).lower(),
                                reasons=[
                                    f"Options flow: {conv.get('total_prints', 0)} unusual prints",
                                    f"Net flow: ${conv.get('net_flow', 0):,.0f}",
                                    f"Conviction: {conv.get('conviction_pct', 0):.0f}%",
                                ],
                                confidence=conv.get("conviction_pct", 0) / 100.0,
                            )
        except Exception as exc:
            logger.warning("  Options flow scan error: %s", exc)

        # ── Phase 3d: Regime Detection ──
        logger.info("Phase 3d: Market Regime Detection")
        regime_data = {}
        try:
            if container.regime_detector:
                regime_data = container.regime_detector.get_market_regime()
                container._last_regime_data = regime_data  # Expose for health reporter
                logger.info(
                    "  Regime: %s (confidence=%s)",
                    regime_data.get("overall_regime", "?"),
                    f"{regime_data.get('overall_confidence', 0):.0%}",
                )
        except Exception as exc:
            logger.warning("  Regime detection error: %s", exc)

        # ── Phase 3d2: Polymarket Scan ──
        polymarket_signals = []
        if container.polymarket:
            logger.info("Phase 3d2: Polymarket Scan")
            try:
                polymarket_signals = container.polymarket.generate_signals(hl_regime=regime_data)
                sentiment = container.polymarket.get_market_sentiment()
                logger.info(
                    "  Polymarket: %d signals, sentiment=%s (conf=%s, markets=%d)",
                    len(polymarket_signals),
                    sentiment.get("sentiment", "?"),
                    f"{sentiment.get('confidence', 0):.0%}",
                    sentiment.get("markets_analyzed", 0),
                )
            except Exception as exc:
                logger.warning("  Polymarket scan error: %s", exc)

        # Inject into forecaster and reconcile detector vs forecaster
        _inject_forecaster_signals(container, regime_data)
        regime_data = _reconcile_regimes(regime_data, container)

        # ── Phase 3d3: Macro Regime Overlay ──
        regime_data = _apply_macro_regime_overlay(container, regime_data)
        regime_data = _apply_global_momentum_override(container, regime_data)

        # Cross-venue hedging
        _run_hedger(container, regime_data)

        # ── Phase 3e: Multi-Exchange Scan ──
        cross_venue_data, funding_arbs = _run_multi_exchange_scan(container)

        # ── Phase 3f: Liquidation Strategy ──
        lcrs_signals = _run_liquidation_scan(container, regime_data)

        # ── Phase 4: Paper Trading (regime-aware) ──
        logger.info("Phase 4: Paper Trading (regime-aware)")
        top_strategies = container.scorer.get_top_strategies() if container.scorer else []

        # Regime-aware filtering
        if regime_data and container.regime_strategy_filter:
            try:
                top_strategies = container.regime_strategy_filter.filter(top_strategies, regime_data)
            except Exception:
                if container.regime_detector:
                    top_strategies = container.regime_detector.filter_strategies_by_regime(
                        top_strategies, regime_data
                    )
            logger.info("  Post-regime filter: %d strategies active", len(top_strategies))

        # Signal processing
        if container.signal_processor:
            top_strategies = container.signal_processor.process(top_strategies, regime_data=regime_data)
            logger.info("  Post-signal-processor: %d strategies", len(top_strategies))

        # Inject Polymarket signals as synthetic strategies
        if polymarket_signals:
            for pm in polymarket_signals:
                # Default-to-long fallbacks removed. A polymarket signal
                # without a side is malformed and gets skipped rather than
                # silently coerced into a long.
                pm_raw_side = str(pm.get("side", "") or "").strip().lower()
                if pm_raw_side in {"buy", "long"}:
                    pm_side = "long"
                elif pm_raw_side in {"sell", "short"}:
                    pm_side = "short"
                else:
                    logger.debug(
                        "  Polymarket: skipping signal with no side (coin=%s)",
                        pm.get("coin", "?"),
                    )
                    continue
                synthetic = {
                    "id": None,
                    "name": f"polymarket_{pm.get('coin', 'UNK')}_{pm_side}",
                    "strategy_type": "event_driven",
                    "trader_address": "polymarket",
                    "current_score": pm.get("confidence", 0.5),
                    "confidence": pm.get("confidence", 0.5),
                    "direction": pm_side,
                    "side": pm_side,
                    "source": "polymarket",
                    "parameters": {
                        "coins": [pm.get("coin", "BTC")],
                        "market": pm.get("polymarket_market", ""),
                        "probability": pm.get("polymarket_probability", 0),
                    },
                    "metrics": {},
                    "metadata": {
                        "polymarket_volume_24h": pm.get("polymarket_volume_24h", 0),
                        "reason": pm.get("reason", ""),
                    },
                }
                top_strategies.append(synthetic)
            logger.info("  Injected %d Polymarket signals", len(polymarket_signals))

        # Inject high-conviction options flow as synthetic strategies
        # (Phase 4a2 still handles direct trades; this feeds the decision engine too)
        if container.options_scanner:
            convictions = getattr(container.options_scanner, "top_convictions", None) or []
            injected_options = 0
            for conv in convictions:
                if conv.get("conviction_pct", 0) < _OPTIONS_FLOW_MIN_CONVICTION_PCT:
                    continue
                direction = str(conv.get("direction", "")).upper()
                side = "long" if direction == "BULLISH" else "short"
                synthetic = {
                    "id": None,
                    "name": f"options_flow_{conv.get('ticker', 'UNK')}_{side}",
                    "strategy_type": "options_momentum",
                    "trader_address": "options_flow",
                    "current_score": conv.get(
                        "conviction_pct", _OPTIONS_FLOW_MIN_CONVICTION_PCT
                    ) / 100.0,
                    "confidence": conv.get(
                        "conviction_pct", _OPTIONS_FLOW_MIN_CONVICTION_PCT
                    ) / 100.0,
                    "direction": side,
                    "side": side,
                    "source": "options_flow",
                    "parameters": {
                        "coins": [conv.get("ticker", "BTC")],
                    },
                    "metrics": {},
                    "metadata": {
                        "net_flow": conv.get("net_flow", 0),
                        "total_prints": conv.get("total_prints", 0),
                        "conviction_pct": conv.get("conviction_pct", 0),
                    },
                }
                top_strategies.append(synthetic)
                injected_options += 1
            if injected_options:
                logger.info("  Injected %d options flow signals into decision engine", injected_options)

        if getattr(container, "alpha_pipeline", None):
            try:
                alpha_signals = container.alpha_pipeline.generate_signals()
                if alpha_signals:
                    top_strategies.extend(alpha_signals)
                    logger.info(
                        "  Injected %d ML alpha signals into decision engine",
                        len(alpha_signals),
                    )
            except Exception as exc:
                logger.warning("  Alpha pipeline signal generation error: %s", exc)

        # BUG-3 FIX (consumption): drain any whale signals queued by fast_cycle
        # into the trading pipeline so they actually influence decisions.
        whale_queue = getattr(container, "_whale_strategy_queue", None)
        if whale_queue:
            top_strategies.extend(whale_queue)
            logger.info("  Injected %d whale trade signals from fast cycle", len(whale_queue))
            container._whale_strategy_queue = []

        # Keep the shadow ledger synced to exchange truth when live mode is active.
        if is_live_trading_active(container):
            closed = sync_shadow_book_to_live(container)
        else:
            closed = container.paper_trader.check_open_positions() if container.paper_trader else []
        if closed:
            logger.info("  Closed %d positions", len(closed))

        # Cross-venue signal confirmation
        _run_cross_venue_confirmation(container, top_strategies)

        # Decision engine
        open_trades = get_execution_open_positions(container)
        kelly_stats = None
        if container.kelly_sizer:
            try:
                kelly_stats = container.kelly_sizer.get_all_sizing_stats()
            except Exception:
                pass
        if container.decision_engine:
            top_strategies = container.decision_engine.decide(
                top_strategies, regime_data=regime_data,
                open_positions=open_trades, kelly_stats=kelly_stats,
            )

        # AgentScorer dynamic weights
        _apply_agent_scorer_weights(container, top_strategies)

        # Execute new signals
        if top_strategies and container.paper_trader:
            executed = container.paper_trader.execute_strategy_signals(
                top_strategies, exchange_agg=container.exchange_agg,
                options_scanner=container.options_scanner,
                regime_data=regime_data, arena=container.arena,
            )
            logger.info("  Executed %d new paper trades", len(executed))

            mirror_executed_trades_to_live(
                container,
                executed,
                success_label="  LIVE",
                skip_label="  Live trader requested but not deployable; skipping strategy mirroring",
            )

            if tg.is_configured():
                for t in executed:
                    tg.notify_trade_opened(t, source="strategy")

        # Phase 4a: LCRS execution
        if lcrs_signals:
            _execute_lcrs_signals(container, lcrs_signals, regime_data)

        # Phase 4a2: Options flow standalone trades
        _execute_options_flow_trades(container, regime_data)

        # Phase 4b: Copy trading
        _run_copy_trading(container, regime_data)

        closed = _collect_closed_trade_events(container, closed)
        if closed and tg.is_configured():
            for c_trade in closed:
                tg.notify_trade_closed(
                    c_trade, c_trade.get("exit_price", 0),
                    c_trade.get("pnl", 0), c_trade.get("reason", ""),
                )

        # Phase 4c: Feed closed trade outcomes
        if closed:
            _process_closed_trades(container, closed)
        _train_rl_sizer_if_due(container)

        # Phase 5: Alpha Arena
        _run_alpha_arena(container, regime_data)

        logger.info("Trading cycle #%d complete.", cycle_count)

        # Notify the v2 dashboard's WS subscribers (no-op when v2 isn't
        # running). Wrapped so a dashboard glitch never breaks trading.
        try:
            from src.ui.v2.events import publish_event
            publish_event("cycle", cycle=cycle_count)
        except Exception:
            pass

    except Exception as exc:
        logger.error("Error in cycle #%d: %s", cycle_count, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Sub-phases (keep the main function readable)
# ---------------------------------------------------------------------------

def _run_multi_exchange_scan(container):
    """Phase 3e: multi-exchange scan."""
    cross_venue_data = {}
    funding_arbs = []
    logger.info("Phase 3e: Multi-Exchange Scanner")
    try:
        if container.multi_scanner:
            venue_health = container.multi_scanner.check_health()
            logger.info("  Venue health: %s", venue_health)
            common_markets = container.multi_scanner.get_common_markets()
            if common_markets:
                logger.info("  Common markets: %s...", common_markets[:15])
            funding_arbs = container.multi_scanner.scan_funding_arb()
            if funding_arbs:
                for arb in funding_arbs[:3]:
                    logger.info(
                        "  Funding arb: %s long@%s(%+.4f%%) / short@%s(%+.4f%%) = %.1f%% ann.",
                        arb.coin, arb.long_venue, arb.long_funding_rate,
                        arb.short_venue, arb.short_funding_rate,
                        arb.funding_spread_annualized,
                    )
            if getattr(config, "LIGHTER_STRATEGY_INJECTION_ENABLED", False):
                injected = container.multi_scanner.inject_lighter_strategies(
                    limit=int(getattr(config, "LIGHTER_STRATEGY_INJECTION_LIMIT", 25)),
                    min_volume_usd=float(getattr(config, "LIGHTER_STRATEGY_MIN_VOLUME_USD", 10_000.0)),
                )
                logger.info("  Lighter strategy injection: %s", injected)
            elif "lighter" in getattr(container.multi_scanner, "adapters", {}):
                logger.info(
                    "  Lighter strategy injection disabled "
                    "(LIGHTER_STRATEGY_INJECTION_ENABLED=false)"
                )
            cross_venue_data = {
                "health": venue_health,
                "common_markets": common_markets,
                "funding_arbs": funding_arbs,
            }
        else:
            logger.info("  Multi-exchange scanner not available")
    except Exception as exc:
        logger.warning("  Multi-exchange scanner error: %s", exc)
    return cross_venue_data, funding_arbs


def _run_liquidation_scan(container, regime_data):
    """Phase 3f: liquidation cascade reversal scan."""
    lcrs_signals = []
    if not container.liquidation_strategy:
        return lcrs_signals
    logger.info("Phase 3f: Liquidation Strategy Scan")
    try:
        from src.data import hyperliquid_client as hl_client
        from src.core.api_manager import get_manager, Priority
        mids = hl_client.get_all_mids() or {}
        coins = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "ARB",
                 "OP", "SUI", "APT", "INJ", "SEI"]
        for coin in coins:
            price = float(mids.get(coin, 0))
            if price <= 0:
                continue
            try:
                lcrs_features = {}
                if regime_data and "per_coin" in regime_data:
                    coin_regime = regime_data["per_coin"].get(coin, {})
                    lcrs_features["trend_strength"] = coin_regime.get("trend_strength", 0.5)
                    lcrs_features["volatility"] = coin_regime.get("atr_pct", 0.02)
                    lcrs_features["volume_ratio"] = coin_regime.get("volume_ratio", 1.0)

                # Funding rate
                try:
                    meta_data = get_manager().post(
                        payload={"type": "metaAndAssetCtxs"},
                        priority=Priority.NORMAL, timeout=10,
                    )
                    if isinstance(meta_data, list) and len(meta_data) > 1:
                        for asset_ctx in meta_data[1]:
                            if isinstance(asset_ctx, dict) and asset_ctx.get("coin") == coin:
                                lcrs_features["funding_rate"] = float(asset_ctx.get("funding", 0))
                                lcrs_features["oi_change"] = float(asset_ctx.get("openInterest", 0)) * 0.01
                                break
                except Exception as e:
                    # Incomplete LCRS features here -> the data-readiness
                    # gate later rejects this signal for missing
                    # feature_vector. Log so the rejection is traceable
                    # to this root cause, not just its symptom.
                    logger.debug(
                        "LCRS funding/oi fetch failed for %s "
                        "(features will be partial): %s", coin, e,
                    )

                # Feature engine enrichment
                if container.feature_engine:
                    try:
                        payload = {
                            "type": "candleSnapshot",
                            "req": {
                                "coin": coin, "interval": "1h",
                                "startTime": int((clock_provider.utc_now().timestamp() - 100 * 3600) * 1000),
                                "endTime": int(clock_provider.utc_now().timestamp() * 1000),
                            },
                        }
                        raw = get_manager().post(payload=payload, priority=Priority.NORMAL, timeout=10)
                        if isinstance(raw, list) and len(raw) >= 20:
                            candles = [
                                {"open": float(c.get("o", 0)), "high": float(c.get("h", 0)),
                                 "low": float(c.get("l", 0)), "close": float(c.get("c", 0)),
                                 "volume": float(c.get("v", 0))} for c in raw
                            ]
                            feat = container.feature_engine.compute(coin, candles)
                            lcrs_features.setdefault("rsi", feat.rsi)
                            lcrs_features.setdefault("momentum_score", feat.momentum_score)
                            lcrs_features.setdefault("trend_strength", feat.trend_strength)
                            lcrs_features.setdefault("volatility", feat.volatility)
                            lcrs_features.setdefault("volume_ratio", feat.volume_ratio)
                            lcrs_features.setdefault("overall_score", feat.overall_score)
                            lcrs_features.setdefault("bollinger_position", feat.bollinger_position)
                            if len(candles) >= 8:
                                lcrs_features["price_change"] = (
                                    (candles[-1]["close"] - candles[-8]["close"]) / candles[-8]["close"]
                                )
                    except Exception as e:
                        logger.debug(
                            "LCRS feature-engine enrichment failed for %s "
                            "(features will be partial -> may be rejected "
                            "by data-readiness gate): %s", coin, e,
                        )

                sig = container.liquidation_strategy.generate_signal(coin, lcrs_features, price)
                if sig:
                    lcrs_signals.append(sig)
                    logger.info(
                        "  LCRS: %s %s (conf=%s, type=%s)",
                        sig["side"].upper(), coin,
                        f"{sig['confidence']:.0%}",
                        sig["features"].get("setup_type", ""),
                    )
            except Exception as exc:
                logger.debug("  LCRS scan error %s: %s", coin, exc)

        if lcrs_signals:
            logger.info("  LCRS found %d setups", len(lcrs_signals))
        else:
            logger.info("  LCRS: no setups detected")
    except Exception as exc:
        logger.warning("  Liquidation strategy error: %s", exc)
    return lcrs_signals


def _run_cross_venue_confirmation(container, top_strategies):
    """Enrich strategies with cross-venue confirmation scores."""
    if not (container.multi_scanner and getattr(container.multi_scanner, "cross_venue", None) and top_strategies):
        return
    logger.info("Phase 4 cross-venue: Signal Confirmation")
    try:
        import json
        signals_to_confirm = []
        for s in top_strategies:
            params = s.get("parameters", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            coins = params.get("coins") or params.get("coins_traded") or params.get("coin") or []
            if isinstance(coins, str):
                coins = [coins]
            coin = coins[0] if coins else ""
            # Default-to-long removed. Drop strategies without a direction
            # rather than silently confirming them as long.
            raw_dir = str(s.get("direction", "") or s.get("side", "") or "").strip().lower()
            if raw_dir in {"buy", "long"}:
                direction = "long"
            elif raw_dir in {"sell", "short"}:
                direction = "short"
            else:
                continue
            score = s.get("score", 0.5)
            if coin and coin != "unknown":
                signals_to_confirm.append({"coin": coin, "direction": direction, "score": score})

        if signals_to_confirm:
            confirmed = container.multi_scanner.confirm_signals(signals_to_confirm)
            confirm_map = {f"{c.coin}:{c.direction}": c.confirmation_score for c in confirmed}
            for s in top_strategies:
                params = s.get("parameters", {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except (json.JSONDecodeError, TypeError):
                        params = {}
                coins = params.get("coins") or params.get("coins_traded") or params.get("coin") or []
                if isinstance(coins, str):
                    coins = [coins]
                coin = coins[0] if coins else ""
                # Same direction normalisation as the loop above. A strategy
                # without a usable side gets cv_score=0 instead of being
                # silently keyed as long:0:0.
                raw_dir = str(s.get("direction", "") or s.get("side", "") or "").strip().lower()
                if raw_dir in {"buy", "long"}:
                    direction = "long"
                elif raw_dir in {"sell", "short"}:
                    direction = "short"
                else:
                    direction = ""
                key = f"{coin}:{direction}" if direction else None
                cv_score = confirm_map.get(key, 0.0) if key else 0.0
                if "metadata" not in s:
                    s["metadata"] = {}
                s["metadata"]["cross_venue_score"] = cv_score
                if cv_score > 0.15:
                    original = s.get("current_score", s.get("score", 0.5))
                    s["current_score"] = min(1.0, original + cv_score * 0.15)

            boosted = sum(
                1 for s in top_strategies
                if s.get("metadata", {}).get("cross_venue_score", 0) > 0.15
            )
            logger.info("  Cross-venue: confirmed %d signals, %d boosted", len(signals_to_confirm), boosted)
    except Exception as exc:
        logger.warning("  Cross-venue confirmation error: %s", exc)


def _apply_agent_scorer_weights(container, top_strategies):
    """Apply AgentScorer dynamic weights to strategy confidences."""
    if not top_strategies or not container.agent_scorer:
        return
    try:
        if hasattr(container.agent_scorer, "apply_weights_to_signals"):
            for s in top_strategies:
                stype = s.get("strategy_type", "unknown")
                source_key = f"strategy:{stype}"
                weight = container.agent_scorer.get_weight(source_key)
                orig_conf = float(s.get("confidence", 0.5))
                s["confidence"] = round(orig_conf * 0.6 + weight * 0.4, 3)
                s["agent_scorer_weight"] = round(weight, 3)
    except Exception as exc:
        logger.debug("  AgentScorer weight apply error: %s", exc)


def _execute_lcrs_signals(container, lcrs_signals, regime_data):
    """Phase 4a: execute liquidation reversal signals."""
    from src.notifications import telegram_bot as tg
    logger.info("Phase 4a: Liquidation Strategy Execution")
    try:
        from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams
        lcrs_executed = []
        open_trades = get_execution_open_positions(container)
        # BUG-2 FIX: load account BEFORE the loop so dynamic sizing
        # (Kelly/RL) is actually applied to LCRS trades.  Previously
        # `account` was None at the sizing check, making it dead code.
        try:
            account = db.get_paper_account()
        except Exception:
            account = None

        for sig in lcrs_signals:
            try:
                trade_signal = TradeSignal(
                    coin=sig["coin"], side=SignalSide(sig["side"]),
                    confidence=sig["confidence"], source=SignalSource.STRATEGY,
                    reason=f"LCRS: {sig['features'].get('setup_type', 'unknown')}",
                    strategy_type="liquidation_reversal",
                    entry_price=sig["price"], leverage=sig["leverage"],
                    position_pct=sig.get("position_pct", 0.06),
                    risk=RiskParams(stop_loss_pct=0.025, take_profit_pct=0.125),
                    context={
                        "features": sig.get("features", {}) or {},
                        "volatility": (sig.get("features", {}) or {}).get("volatility"),
                        "expected_return": sig.get("expected_return"),
                    },
                    regime=regime_data.get("overall_regime", "") if regime_data else "",
                )
                trade_signal = _apply_dynamic_risk_policy(container, trade_signal, regime_data=regime_data)

                if container.firewall:
                    passed, reason = container.firewall.validate(
                        trade_signal, regime_data=regime_data, open_positions=open_trades
                    )
                    if not passed:
                        logger.info("  LCRS firewall rejected %s: %s", sig["coin"], reason)
                        continue

                if account and (container.kelly_sizer or getattr(container, "rl_sizer", None)):
                    sizing = _get_dynamic_sizing(
                        container,
                        "liquidation_reversal",
                        account["balance"],
                        trade_signal.confidence,
                        regime_data=regime_data,
                        coin=sig["coin"],
                        volatility=sig.get("features", {}).get("volatility", 0.02),
                    )
                    if sizing:
                        trade_signal.position_pct = sizing.position_pct

                if container.trade_memory:
                    mem = container.trade_memory.find_similar(
                        sig.get("features", {}), coin=sig["coin"], side=sig["side"]
                    )
                    if mem.recommendation == "avoid":
                        logger.info("  LCRS memory blocked %s: %s", sig["coin"], mem.reason)
                        continue

                if container.llm_filter:
                    ctx = {"regime_data": regime_data, "open_positions": open_trades}
                    approved, adj_conf, reason = container.llm_filter.filter(sig, ctx)
                    if not approved:
                        logger.info("  LCRS LLM filter blocked %s: %s", sig["coin"], reason)
                        continue
                    trade_signal.confidence = adj_conf

                if is_live_trading_active(container):
                    live_result = _execute_signal_live(container, trade_signal, "LCRS")
                    if live_result:
                        lcrs_executed.append(live_result)
                        if tg.is_configured():
                            tg.notify_trade_opened(
                                {"coin": sig["coin"], "side": sig["side"], "entry_price": sig["price"]},
                                source="liquidation_strategy",
                            )
                    continue

                if account is None:
                    account = db.get_paper_account()
                if not account:
                    continue
                size_usd = account["balance"] * trade_signal.effective_size
                size = size_usd / sig["price"]
                trade_id = db.open_paper_trade(
                    strategy_id=None, coin=sig["coin"], side=sig["side"],
                    entry_price=sig["price"], size=size, leverage=sig["leverage"],
                    stop_loss=sig["stop_loss"], take_profit=sig["take_profit"],
                    metadata={
                        "source": "liquidation_strategy",
                        "strategy_type": "liquidation_reversal",
                        "confidence": trade_signal.confidence,
                        "setup_type": sig["features"].get("setup_type", ""),
                        "features": sig["features"],
                    },
                )
                lcrs_executed.append({"id": trade_id, "coin": sig["coin"], "side": sig["side"]})
                logger.info(
                    "  LCRS executed: %s %s @ $%s (conf=%s)",
                    sig["side"].upper(), sig["coin"],
                    f"{sig['price']:,.2f}", f"{trade_signal.confidence:.0%}",
                )
                if tg.is_configured():
                    tg.notify_trade_opened(
                        {"coin": sig["coin"], "side": sig["side"], "entry_price": sig["price"]},
                        source="liquidation_strategy",
                    )
            except Exception as exc:
                logger.debug("  LCRS execution error %s: %s", sig.get("coin"), exc)

        if lcrs_executed:
            logger.info("  Executed %d LCRS trades", len(lcrs_executed))
    except Exception as exc:
        logger.warning("  LCRS execution phase error: %s", exc)


def _execute_options_flow_trades(container, regime_data):
    """Phase 4a2: high-conviction options flow → direct trade."""
    from src.notifications import telegram_bot as tg
    logger.info("Phase 4a2: Options Flow Trades")
    try:
        from src.signals.signal_schema import signal_from_options_flow
        from src.data import hyperliquid_client as hl_client
        mids = hl_client.get_all_mids() or {}
        options_executed = []
        live_active = is_live_trading_active(container)
        live_trader = getattr(container, "live_trader", None) if live_active else None
        live_account_value = None
        if live_active and live_trader:
            get_account_value = getattr(live_trader, "get_account_value", None)
            if callable(get_account_value):
                try:
                    live_account_value = get_account_value()
                except Exception as exc:
                    logger.debug("  Options flow live account check failed: %s", exc)

        convictions = getattr(container.options_scanner, "top_convictions", None) or []
        for conv in convictions:
            if conv.get("conviction_pct", 0) < _OPTIONS_FLOW_MIN_CONVICTION_PCT:
                continue
            flow_signal = signal_from_options_flow(
                ticker=conv["ticker"], direction=conv["direction"],
                net_flow=conv["net_flow"], prints=conv["total_prints"],
                conviction_pct=conv["conviction_pct"],
            )
            flow_signal.position_pct = 0.04
            flow_signal.leverage = 2.0
            price = float(mids.get(conv["ticker"], 0))
            if price <= 0:
                continue
            flow_signal.entry_price = price
            flow_signal.context = {
                "features": {},
                "expected_return": None,
                "volatility": None,
            }
            flow_signal = _apply_dynamic_risk_policy(container, flow_signal, regime_data=regime_data)

            if live_active and live_account_value is not None and live_account_value <= 0:
                logger.warning(
                    "  Skipping options flow live trade for %s: perps margin unavailable (account_value=%.2f)",
                    conv["ticker"],
                    live_account_value,
                )
                continue

            if live_active and live_trader and (not flow_signal.size or flow_signal.size <= 0):
                try:
                    max_position_size = float(getattr(live_trader, "max_position_size", 0.0) or 0.0)
                except (TypeError, ValueError):
                    max_position_size = 0.0
                if max_position_size > 0:
                    estimated_notional = max_position_size * flow_signal.position_pct
                    if estimated_notional > 0:
                        flow_signal.size = estimated_notional / price

            sizing_balance = live_account_value if live_active and live_account_value else None
            if sizing_balance is None:
                try:
                    paper_account = db.get_paper_account()
                    sizing_balance = float(paper_account["balance"]) if paper_account else None
                except Exception:
                    sizing_balance = None
            if sizing_balance and (getattr(container, "kelly_sizer", None) or getattr(container, "rl_sizer", None)):
                try:
                    sizing = _get_dynamic_sizing(
                        container,
                        "options_momentum",
                        sizing_balance,
                        flow_signal.confidence,
                        regime_data=regime_data,
                        coin=conv["ticker"],
                        volatility=0.02,
                    )
                    if sizing:
                        flow_signal.position_pct = sizing.position_pct
                except Exception as exc:
                    logger.debug("  Options flow dynamic sizing failed: %s", exc)

            if container.firewall:
                passed, reason = container.firewall.validate(
                    flow_signal, regime_data=regime_data,
                    open_positions=get_execution_open_positions(container),
                    account_balance=live_account_value if live_active else None,
                )
                if not passed:
                    logger.info("  Firewall rejected options flow %s: %s", conv["ticker"], reason)
                    continue

            if container.agent_scorer:
                container.agent_scorer.record_signal("options_flow", {
                    "coin": conv["ticker"], "side": flow_signal.side.value,
                    "confidence": flow_signal.confidence,
                })

            if live_active:
                # Run full live-trader firewall validation so account-based checks
                # use live balance context instead of paper defaults.
                live_result = _execute_signal_live(
                    container,
                    flow_signal,
                    "OPTIONS FLOW",
                    bypass_firewall=False,
                )
                if live_result:
                    options_executed.append(live_result)
                    if tg.is_configured():
                        tg.notify_trade_opened(
                            {"coin": conv["ticker"], "side": flow_signal.side.value, "entry_price": price},
                            source="options_flow",
                        )
                continue

            account = db.get_paper_account()
            if account:
                size_usd = account["balance"] * flow_signal.effective_size
                size = size_usd / price
                side = flow_signal.side.value
                sl, tp = flow_signal.risk.resolve_trigger_prices(price, side, flow_signal.leverage)
                trade_id = db.open_paper_trade(
                    strategy_id=None, coin=conv["ticker"], side=side,
                    entry_price=price, size=size, leverage=flow_signal.leverage,
                    stop_loss=sl, take_profit=tp,
                    metadata={
                        "source": "options_flow",
                        "conviction": conv["conviction_pct"],
                        "net_flow": conv["net_flow"],
                        "prints": conv["total_prints"],
                        "risk_policy": dict((flow_signal.context or {}).get("risk_policy", {}) or {}),
                    },
                )
                logger.info(
                    "  Options flow trade: %s %s @ $%s (conviction: %d%%)",
                    side.upper(), conv["ticker"], f"{price:,.2f}", conv["conviction_pct"],
                )
                options_executed.append({"id": trade_id, "coin": conv["ticker"], "side": side})
                if tg.is_configured():
                    tg.notify_trade_opened(
                        {"coin": conv["ticker"], "side": side, "entry_price": price},
                        source="options_flow",
                    )
        if options_executed:
            logger.info("  Executed %d options flow trades", len(options_executed))
    except Exception as exc:
        logger.warning("  Options flow trading error: %s", exc)


def _run_copy_trading(container, regime_data):
    """Phase 4b: copy trading — WebSocket + REST."""
    from src.notifications import telegram_bot as tg
    logger.info("Phase 4b: Copy Trading")
    ws_signals = []
    if container.position_monitor:
        ws_signals = container.position_monitor.drain_signals()
        if ws_signals:
            logger.info("  WebSocket: %d real-time signals", len(ws_signals))
    copy_signals = ws_signals + (
        container.copy_trader.scan_top_traders(top_n=10) if container.copy_trader else []
    )

    try:
        from src.discovery.golden_bridge import get_golden_copy_signals, auto_connect_golden_wallets
        auto_connect_golden_wallets()
        golden_signals = get_golden_copy_signals()
        if golden_signals:
            logger.info("  Golden bridge: %d signals", len(golden_signals))
            copy_signals = golden_signals + copy_signals
    except Exception as exc:
        logger.debug("  Golden bridge skipped: %s", exc)

    if copy_signals and container.copy_trader:
        copy_executed = container.copy_trader.execute_copy_signals(copy_signals, regime_data=regime_data)
        logger.info("  Executed %d copy trades", len(copy_executed))
        if tg.is_configured():
            for t in copy_executed:
                tg.notify_trade_opened(t, source="copy")
        mirror_executed_trades_to_live(
            container,
            copy_executed,
            success_label="  LIVE COPY",
            skip_label="  Live trader requested but not deployable; skipping copy mirroring",
        )


def _collect_closed_trade_events(container, initial_closed):
    """Merge in-memory close events from paper/copy traders, deduping by trade id."""
    merged = list(initial_closed or [])
    seen = set()
    for trade in merged:
        trade_id = trade.get("trade_id")
        if trade_id is not None:
            seen.add(trade_id)

    for trader_name in ("paper_trader", "copy_trader"):
        trader = getattr(container, trader_name, None)
        if not trader or not hasattr(trader, "drain_closed_events"):
            continue
        try:
            for event in trader.drain_closed_events() or []:
                trade_id = event.get("trade_id")
                if trade_id is not None and trade_id in seen:
                    continue
                merged.append(event)
                if trade_id is not None:
                    seen.add(trade_id)
        except Exception as exc:
            logger.debug("  Failed draining %s close events: %s", trader_name, exc)

    return merged


def _process_closed_trades(container, closed):
    """Phase 4c: feed outcomes to arena, agent scorer, shadow tracker, AND Kelly sizer."""
    for c_trade in closed:
        try:
            meta = c_trade.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    import json as _json
                    meta = _json.loads(meta or "{}")
                except (ValueError, TypeError):
                    meta = {}
            if meta.get("synthetic_reconciliation") or c_trade.get("reason") == "live_reconciled_closed":
                continue
            stype = str(
                c_trade.get("strategy_type")
                or meta.get("strategy_type")
                or ""
            ).strip().lower()
            if not stype:
                stype = "unknown"
            source = str(meta.get("source") or c_trade.get("source") or "").strip().lower()
            trader = str(
                meta.get("source_trader")
                or meta.get("trader_address")
                or c_trade.get("source_trader")
                or c_trade.get("trader_address")
                or ""
            ).strip().lower()
            source_key = str(meta.get("source_key") or "").strip().lower()
            if not source_key:
                if source == "copy_trade":
                    source_key = f"copy_trade:{trader}" if trader else "copy_trade:untagged"
                elif source:
                    source_key = source if stype == "unknown" else f"{source}:{stype}"
                elif stype != "unknown":
                    source_key = f"strategy:{stype}"
                else:
                    source_key = "strategy:untagged"
            pnl = c_trade.get("pnl", 0)
            entry = c_trade.get("entry_price", 1)
            size = c_trade.get("size", 0)
            leverage = c_trade.get("leverage", 1)
            # BUG-1 FIX: include leverage in notional so return_pct
            # matches how `pnl` was calculated (with leverage).  Without
            # this, return_pct was inflated by the leverage factor,
            # poisoning arena fitness scores and agent scorer weights.
            notional = entry * max(size, 1e-8) * max(leverage, 1)
            return_pct = pnl / max(notional, 1e-8)

            if container.arena:
                container.arena.record_trade_for_strategy(
                    stype,
                    pnl,
                    return_pct,
                    metadata=c_trade,
                )

            _record_shadow_trade(container, c_trade, pnl, return_pct, entry)

            # Kelly sizer: feed trade outcomes so it can compute win_rate + reward/risk
            if container.kelly_sizer:
                try:
                    # Use strategy_type as key; copy trades get source-specific key
                    if source == "copy_trade":
                        kelly_key = source_key
                    elif source == "options_flow":
                        kelly_key = f"options_flow:{c_trade.get('coin', 'UNK')}"
                    else:
                        kelly_key = stype or "unknown"
                    container.kelly_sizer.record_outcome(
                        strategy_key=kelly_key,
                        pnl=pnl,
                        entry_price=entry,
                        size=max(size, 1e-8),
                        leverage=max(leverage, 1),
                    )
                except Exception as e:
                    logger.debug(
                        "LEARN-WRITE kelly_sizer.record_outcome failed "
                        "(key=%s coin=%s): %s",
                        kelly_key, c_trade.get("coin"), e,
                    )

            # AgentScorer outcome
            if container.agent_scorer:
                try:
                    signal_id = meta.get("signal_id", "")
                    if signal_id:
                        container.agent_scorer.record_outcome(source_key, signal_id, pnl, return_pct)
                except Exception as e:
                    logger.debug(
                        "LEARN-WRITE agent_scorer.record_outcome failed "
                        "(source=%s coin=%s): %s",
                        source_key, c_trade.get("coin"), e,
                    )
        except Exception as e:
            # Umbrella over the whole closed-trade learning update
            # (arena / shadow / kelly / agent_scorer). A throw here
            # silently drops every learning signal for this trade --
            # the broadest version of the data-loss class.
            logger.warning(
                "LEARN-WRITE closed-trade outcome processing FAILED "
                "(coin=%s id=%s): %s",
                c_trade.get("coin") if isinstance(c_trade, dict) else "?",
                c_trade.get("id") if isinstance(c_trade, dict) else "?",
                e,
            )


def _run_alpha_arena(container, regime_data):
    """Phase 5: Alpha Arena cycle."""
    if not container.arena:
        return
    logger.info("Phase 5: Alpha Arena")
    try:
        arena_candle_map = _fetch_arena_candle_universe()
        container.arena.run_cycle(historical_candles=arena_candle_map)
        stats = container.arena.get_stats()
        logger.info(
            "  Arena: %d active, %d champions, %d signal-qualified, PnL=$%.2f",
            stats["active_agents"],
            stats["champions"],
            stats.get("qualified_signal_agents", 0),
            stats["total_arena_pnl"],
        )

        # Champion signals → paper trading
        eligible_candles = {
            coin: candles[-100:]
            for coin, candles in (arena_candle_map or {}).items()
            if len(candles) >= 30
        }
        if eligible_candles:
            try:
                from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams
                champion_signals = container.arena.get_champion_signals(
                    current_candles=eligible_candles,
                    min_fitness=_ARENA_CHAMPION_MIN_FITNESS,
                    min_trades=_ARENA_CHAMPION_MIN_TRADES,
                    min_win_rate=_ARENA_CHAMPION_MIN_WIN_RATE,
                )
                if champion_signals:
                    logger.info("  Arena champions: %d signals", len(champion_signals))
                    account = None
                    for sig in champion_signals:
                        try:
                            price = sig.get("price", 0)
                            if price <= 0:
                                continue
                            side = sig["side"]
                            conf = sig["confidence"]
                            risk = RiskParams(stop_loss_pct=0.05, take_profit_pct=0.25)
                            live_signal = TradeSignal(
                                coin=sig["coin"],
                                side=SignalSide(side),
                                confidence=conf,
                                source=SignalSource.STRATEGY,
                                reason=f"Arena champion: {sig['agent_name']}",
                                strategy_type=sig["strategy_type"],
                                entry_price=price,
                                leverage=2,
                                position_pct=0.05 * conf,
                                risk=risk,
                                context={
                                    "features": {},
                                    "atr_pct": sig.get("atr_pct", 0.0),
                                    "volatility": sig.get("atr_pct", 0.0),
                                    "failure_cases": list(sig.get("failure_cases") or []),
                                    "invalidation_price": sig.get("invalidation_price"),
                                    "validation_warnings": list(sig.get("validation_warnings") or []),
                                    "arena_validation_passed": bool(sig.get("validation_passed", False)),
                                },
                                regime=regime_data.get("overall_regime", "") if regime_data else "",
                            )
                            live_signal = _apply_dynamic_risk_policy(container, live_signal, regime_data=regime_data)
                            sl, tp = live_signal.risk.resolve_trigger_prices(price, side, live_signal.leverage)
                            position_pct = 0.05 * conf
                            # ★ M17 FIX: for live trading, size against real account value
                            # not PAPER_TRADING_INITIAL_BALANCE constant. For paper mode,
                            # fall back to the constant as before.
                            live_active = is_live_trading_active(container)
                            if getattr(container, "kelly_sizer", None) or getattr(container, "rl_sizer", None):
                                try:
                                    if live_active and container.live_trader is not None:
                                        try:
                                            fm = None
                                            if hasattr(container.live_trader, "get_free_margin"):
                                                fm = container.live_trader.get_free_margin()
                                            if fm is None and hasattr(container.live_trader, "get_account_value"):
                                                fm = container.live_trader.get_account_value()
                                            account_balance = float(fm) if fm is not None else float(
                                                getattr(config, "PAPER_TRADING_INITIAL_BALANCE", 10_000.0)
                                            )
                                        except Exception:
                                            account_balance = float(
                                                getattr(config, "PAPER_TRADING_INITIAL_BALANCE", 10_000.0)
                                            )
                                    else:
                                        account_balance = float(
                                            getattr(config, "PAPER_TRADING_INITIAL_BALANCE", 10_000.0)
                                        )
                                    sizing = _get_dynamic_sizing(
                                        container,
                                        sig["strategy_type"],
                                        account_balance,
                                        conf,
                                        regime_data=regime_data,
                                        coin=sig["coin"],
                                        volatility=sig.get("atr_pct", 0.02),
                                    )
                                    if sizing:
                                        position_pct = sizing.position_pct
                                except Exception as exc:
                                    logger.debug("  Arena champion sizing failed: %s", exc)
                            if live_active:
                                live_signal.position_pct = position_pct
                                # ★ H18 FIX: Arena signals previously bypassed the
                                # firewall entirely on the live path. Run the same
                                # validation LCRS does (line ~1063) so cooldowns,
                                # per-coin limits, source caps, side policies, and
                                # event risk gates all apply to Arena signals.
                                arena_firewall = getattr(container, "firewall", None)
                                if arena_firewall:
                                    open_trades = get_execution_open_positions(container)
                                    passed, reason = arena_firewall.validate(
                                        live_signal,
                                        regime_data=regime_data,
                                        open_positions=open_trades,
                                    )
                                    if not passed:
                                        logger.info(
                                            "  Arena firewall rejected %s %s: %s",
                                            sig["coin"], side, reason,
                                        )
                                        continue
                                _execute_signal_live(container, live_signal, "ARENA")
                                continue
                            if account is None:
                                account = db.get_paper_account()
                            if account:
                                if getattr(container, "kelly_sizer", None) or getattr(container, "rl_sizer", None):
                                    try:
                                        sizing = _get_dynamic_sizing(
                                            container,
                                            sig["strategy_type"],
                                            account["balance"],
                                            conf,
                                            regime_data=regime_data,
                                            coin=sig["coin"],
                                            volatility=sig.get("atr_pct", 0.02),
                                        )
                                        if sizing:
                                            position_pct = sizing.position_pct
                                    except Exception as exc:
                                        logger.debug("  Arena paper sizing failed: %s", exc)
                                size_usd = account["balance"] * position_pct
                                size = size_usd / price
                                # SILENT-4 FIX: use risk-policy-adjusted leverage
                                # instead of hardcoded 2x.
                                db.open_paper_trade(
                                    strategy_id=None, coin=sig["coin"], side=side,
                                    entry_price=price, size=size, leverage=live_signal.leverage,
                                    stop_loss=sl, take_profit=tp,
                                    metadata={
                                        "source": "arena_champion",
                                        "agent_id": sig["agent_id"],
                                        "agent_name": sig["agent_name"],
                                        "strategy_type": sig["strategy_type"],
                                        "agent_fitness": sig["agent_fitness"],
                                        "agent_elo": sig["agent_elo"],
                                        "confidence": conf,
                                        "failure_cases": list(sig.get("failure_cases") or []),
                                        "invalidation_price": sig.get("invalidation_price"),
                                        "validation_warnings": list(sig.get("validation_warnings") or []),
                                        "arena_validation_passed": bool(sig.get("validation_passed", False)),
                                        "risk_policy": dict((live_signal.context or {}).get("risk_policy", {}) or {}),
                                    },
                                )
                                logger.info(
                                    "  Champion trade: %s %s @ $%s | agent=%s",
                                    side.upper(), sig["coin"], f"{price:,.2f}", sig["agent_name"],
                                )
                        except Exception as exc:
                            logger.debug("  Champion trade exec error: %s", exc)
            except Exception as exc:
                logger.debug("  Champion signals error: %s", exc)
    except Exception as exc:
        logger.warning("  Arena error: %s", exc)
