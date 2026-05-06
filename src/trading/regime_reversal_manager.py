"""
Regime reversal manager for live position supervision.

The manager is deliberately decision-only: it reads the forecaster output,
keeps confirmation/cooldown state, and returns a staged action. LiveTrader
continues to own order placement, firewall checks, and shadow-book updates.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RegimeReversalConfig:
    enabled: bool = True
    tighten_enabled: bool = True
    close_enabled: bool = False
    reverse_enabled: bool = False
    min_confidence: float = 0.70
    reverse_confidence: float = 0.82
    confirm_cycles: int = 3
    min_position_age_seconds: int = 180
    cooldown_seconds: int = 900
    max_actions_per_coin_per_day: int = 2
    tighten_stop_r_multiple: float = 0.35
    reverse_position_pct: float = 0.03
    reverse_on_crash: bool = False


@dataclass
class RegimeReversalDecision:
    action: str = "none"  # none, tighten_stop, close_only, close_and_reverse
    reason: str = "no_action"
    coin: str = ""
    side: str = ""
    reverse_side: str = ""
    regime: str = "unknown"
    confidence: float = 0.0
    signal: float = 0.0
    confirmed_cycles: int = 0
    stop_price: Optional[float] = None
    position_age_seconds: Optional[float] = None
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata or {})
        return data


class RegimeReversalManager:
    """Stateful opposite-regime detector with live-safe action gating."""

    def __init__(self, cfg: Optional[RegimeReversalConfig] = None):
        self.cfg = cfg or RegimeReversalConfig()
        self._streaks: Dict[str, Dict[str, Any]] = {}
        self._last_action_ts: Dict[str, float] = {}
        self._daily_counts: Dict[str, int] = {}
        self._daily_key = datetime.now(timezone.utc).date().isoformat()

    @classmethod
    def from_config(cls, module: Any) -> "RegimeReversalManager":
        cfg = RegimeReversalConfig(
            enabled=bool(getattr(module, "REGIME_REVERSAL_ENABLED", True)),
            tighten_enabled=bool(getattr(module, "REGIME_REVERSAL_TIGHTEN_ENABLED", True)),
            close_enabled=bool(getattr(module, "REGIME_REVERSAL_CLOSE_ENABLED", False)),
            reverse_enabled=bool(getattr(module, "REGIME_REVERSAL_REVERSE_ENABLED", False)),
            min_confidence=float(getattr(module, "REGIME_REVERSAL_MIN_CONFIDENCE", 0.70)),
            reverse_confidence=float(getattr(module, "REGIME_REVERSAL_REVERSE_CONFIDENCE", 0.82)),
            confirm_cycles=int(getattr(module, "REGIME_REVERSAL_CONFIRM_CYCLES", 3)),
            min_position_age_seconds=int(getattr(module, "REGIME_REVERSAL_MIN_POSITION_AGE_SECONDS", 180)),
            cooldown_seconds=int(getattr(module, "REGIME_REVERSAL_COOLDOWN_SECONDS", 900)),
            max_actions_per_coin_per_day=int(getattr(module, "REGIME_REVERSAL_MAX_ACTIONS_PER_COIN_PER_DAY", 2)),
            tighten_stop_r_multiple=float(getattr(module, "REGIME_REVERSAL_TIGHTEN_STOP_R_MULTIPLE", 0.35)),
            reverse_position_pct=float(getattr(module, "REGIME_REVERSAL_REVERSE_POSITION_PCT", 0.03)),
            reverse_on_crash=bool(getattr(module, "REGIME_REVERSAL_REVERSE_ON_CRASH", False)),
        )
        return cls(cfg)

    def evaluate(
        self,
        *,
        position: Dict[str, Any],
        shadow_trades: Iterable[Dict[str, Any]],
        policy: Dict[str, Any],
        current_price: float,
        current_r: float,
        forecaster: Optional[Any] = None,
        regime_data: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> RegimeReversalDecision:
        now_dt = now or datetime.now(timezone.utc)
        coin = str(position.get("coin", "") or "").upper()
        side = str(position.get("side", "") or "").strip().lower()
        if not self.cfg.enabled:
            return self._decision(coin, side, reason="disabled")
        if not coin or side not in {"long", "short"}:
            return self._decision(coin, side, reason="invalid_position")

        age_seconds = self._position_age_seconds(shadow_trades, now_dt)
        if age_seconds is not None and age_seconds < self.cfg.min_position_age_seconds:
            return self._decision(
                coin,
                side,
                reason="position_too_new",
                position_age_seconds=age_seconds,
            )

        regime = self._fetch_regime(coin, forecaster=forecaster, regime_data=regime_data)
        if not regime:
            return self._decision(coin, side, reason="regime_unavailable", position_age_seconds=age_seconds)

        regime_name = str(regime.get("regime", "unknown") or "unknown").strip().lower()
        confidence = self._safe_float(regime.get("confidence"), 0.0)
        signal = self._safe_float(regime.get("signal"), 0.0)
        desired_side = self._desired_side(regime_name, signal)
        metadata = {
            "regime": regime_name,
            "confidence": round(confidence, 4),
            "signal": round(signal, 4),
            "current_r": round(float(current_r or 0.0), 4),
            "active_inputs": regime.get("active_inputs", []),
            "model": regime.get("model"),
            "partial_signal": bool(regime.get("partial_signal", False)),
        }

        if not desired_side:
            self._reset_streak(coin, side)
            return self._decision(
                coin,
                side,
                reason="no_directional_regime",
                regime=regime_name,
                confidence=confidence,
                signal=signal,
                position_age_seconds=age_seconds,
                metadata=metadata,
            )
        if desired_side == side:
            self._reset_streak(coin, side)
            return self._decision(
                coin,
                side,
                reason="regime_aligned",
                regime=regime_name,
                confidence=confidence,
                signal=signal,
                reverse_side=desired_side,
                position_age_seconds=age_seconds,
                metadata=metadata,
            )
        if confidence < self.cfg.min_confidence:
            self._reset_streak(coin, side)
            return self._decision(
                coin,
                side,
                reason="confidence_below_threshold",
                regime=regime_name,
                confidence=confidence,
                signal=signal,
                reverse_side=desired_side,
                position_age_seconds=age_seconds,
                metadata=metadata,
            )

        confirmed_cycles = self._record_streak(coin, side, regime_name, desired_side)
        metadata["confirmed_cycles"] = confirmed_cycles
        if confirmed_cycles < max(1, self.cfg.confirm_cycles):
            return self._decision(
                coin,
                side,
                reason="awaiting_confirmation",
                regime=regime_name,
                confidence=confidence,
                signal=signal,
                reverse_side=desired_side,
                confirmed_cycles=confirmed_cycles,
                position_age_seconds=age_seconds,
                metadata=metadata,
            )

        stop_price = self._tightened_stop_price(position, policy, current_price)
        action = "tighten_stop" if self.cfg.tighten_enabled and stop_price else "none"
        reason = "opposite_regime_confirmed_tighten_stop" if action == "tighten_stop" else "tighten_disabled"

        can_take_live_action, block_reason = self._can_take_live_action(coin, now_dt)
        if self.cfg.close_enabled and can_take_live_action:
            action = "close_only"
            reason = "opposite_regime_confirmed_close"
            crash_regime = regime_name in {"crash", "panic", "risk_off"}
            if (
                self.cfg.reverse_enabled
                and confidence >= self.cfg.reverse_confidence
                and (not crash_regime or self.cfg.reverse_on_crash)
            ):
                action = "close_and_reverse"
                reason = "opposite_regime_confirmed_reverse"
        elif self.cfg.close_enabled and not can_take_live_action:
            metadata["live_action_blocked"] = block_reason
            reason = f"live_action_blocked:{block_reason}"

        return self._decision(
            coin,
            side,
            action=action,
            reason=reason,
            regime=regime_name,
            confidence=confidence,
            signal=signal,
            reverse_side=desired_side,
            confirmed_cycles=confirmed_cycles,
            stop_price=stop_price,
            position_age_seconds=age_seconds,
            metadata=metadata,
        )

    def mark_action(self, coin: str, *, now: Optional[datetime] = None) -> None:
        now_dt = now or datetime.now(timezone.utc)
        self._roll_daily_counts(now_dt)
        key = str(coin or "").upper()
        if not key:
            return
        self._last_action_ts[key] = now_dt.timestamp()
        self._daily_counts[key] = self._daily_counts.get(key, 0) + 1

    def _can_take_live_action(self, coin: str, now: datetime) -> tuple[bool, str]:
        self._roll_daily_counts(now)
        key = str(coin or "").upper()
        last_ts = self._last_action_ts.get(key, 0.0)
        if last_ts and (now.timestamp() - last_ts) < self.cfg.cooldown_seconds:
            return False, "cooldown"
        if self.cfg.max_actions_per_coin_per_day > 0:
            if self._daily_counts.get(key, 0) >= self.cfg.max_actions_per_coin_per_day:
                return False, "daily_cap"
        return True, "ok"

    def _roll_daily_counts(self, now: datetime) -> None:
        key = now.date().isoformat()
        if key != self._daily_key:
            self._daily_key = key
            self._daily_counts.clear()

    def _record_streak(self, coin: str, side: str, regime: str, desired_side: str) -> int:
        key = f"{coin}:{side}"
        prev = self._streaks.get(key) or {}
        if prev.get("regime") == regime and prev.get("desired_side") == desired_side:
            count = int(prev.get("count", 0)) + 1
        else:
            count = 1
        self._streaks[key] = {
            "regime": regime,
            "desired_side": desired_side,
            "count": count,
            "last_ts": time.time(),
        }
        return count

    def _reset_streak(self, coin: str, side: str) -> None:
        self._streaks.pop(f"{coin}:{side}", None)

    def _fetch_regime(
        self,
        coin: str,
        *,
        forecaster: Optional[Any],
        regime_data: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if isinstance(regime_data, dict):
            return regime_data
        if forecaster is None:
            return None
        predict = getattr(forecaster, "predict_regime", None)
        if not callable(predict):
            return None
        try:
            result = predict(coin)
            return result if isinstance(result, dict) else None
        except Exception as exc:
            logger.warning("Regime reversal forecast failed for %s: %s", coin, exc)
            return None

    @staticmethod
    def _desired_side(regime: str, signal: float) -> Optional[str]:
        value = str(regime or "").strip().lower()
        if value in {"bullish", "trend_up", "trending_up", "uptrend", "risk_on"}:
            return "long"
        if value in {"bearish", "trend_down", "trending_down", "downtrend", "crash", "panic", "risk_off"}:
            return "short"
        if signal >= 0.25:
            return "long"
        if signal <= -0.25:
            return "short"
        return None

    def _tightened_stop_price(
        self,
        position: Dict[str, Any],
        policy: Dict[str, Any],
        current_price: float,
    ) -> Optional[float]:
        side = str(position.get("side", "") or "").strip().lower()
        leverage = max(self._safe_float(position.get("leverage"), 1.0), 1.0)
        stop_roe = max(self._safe_float(policy.get("stop_roe_pct"), 0.0), 0.0)
        if side not in {"long", "short"} or current_price <= 0 or stop_roe <= 0:
            return None
        distance_pct = (stop_roe / leverage) * max(self.cfg.tighten_stop_r_multiple, 0.01)
        distance_pct = min(max(distance_pct, 0.0005), 0.05)
        if side == "long":
            return float(current_price) * (1 - distance_pct)
        return float(current_price) * (1 + distance_pct)

    @staticmethod
    def _position_age_seconds(trades: Iterable[Dict[str, Any]], now: datetime) -> Optional[float]:
        opened_values = []
        for trade in trades:
            opened_at = trade.get("opened_at") or trade.get("timestamp")
            parsed = RegimeReversalManager._parse_datetime(opened_at)
            if parsed:
                opened_values.append(parsed)
        if not opened_values:
            return None
        earliest = min(opened_values)
        return max(0.0, (now - earliest).total_seconds())

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            if math.isfinite(parsed):
                return parsed
        except (TypeError, ValueError):
            pass
        return default

    @staticmethod
    def _decision(
        coin: str,
        side: str,
        *,
        action: str = "none",
        reason: str,
        regime: str = "unknown",
        confidence: float = 0.0,
        signal: float = 0.0,
        reverse_side: str = "",
        confirmed_cycles: int = 0,
        stop_price: Optional[float] = None,
        position_age_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegimeReversalDecision:
        return RegimeReversalDecision(
            action=action,
            reason=reason,
            coin=coin,
            side=side,
            reverse_side=reverse_side,
            regime=regime,
            confidence=round(float(confidence or 0.0), 4),
            signal=round(float(signal or 0.0), 4),
            confirmed_cycles=confirmed_cycles,
            stop_price=stop_price,
            position_age_seconds=position_age_seconds,
            metadata=dict(metadata or {}),
        )
