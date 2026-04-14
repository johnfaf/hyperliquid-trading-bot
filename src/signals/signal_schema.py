"""
Trade Signal Schema
===================
Strict dataclass schema that ALL signal sources must conform to.
This is the "contract" between signal generation and execution.

Every signal source (strategy scorer, copy trader, options flow)
must produce a TradeSignal before it reaches the decision firewall.
"""
import contextvars
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum

# Context variable carrying the current trade's trace ID.
# Set automatically when a TradeSignal is created; read by JSONFormatter
# so every log line in the call chain includes the same trace_id.
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_trace_id", default="",
)


class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalSource(str, Enum):
    STRATEGY = "strategy"
    COPY_TRADE = "copy_trade"
    OPTIONS_FLOW = "options_flow"
    POLYMARKET = "polymarket"
    LIQUIDATION_STRATEGY = "liquidation_strategy"
    ARENA_CHAMPION = "arena_champion"
    MANUAL = "manual"
    WHALE_TRADE = "whale_trade"


class SignalStrength(str, Enum):
    STRONG = "strong"       # High conviction — full size
    MODERATE = "moderate"   # Medium conviction — reduced size
    WEAK = "weak"           # Low conviction — minimum size or skip


@dataclass
class RiskParams:
    """Risk parameters attached to every signal."""
    stop_loss_pct: float = 0.05       # 5% stop on margin / ROE
    take_profit_pct: float = 0.25     # 25% take-profit on margin / ROE (5:1)
    max_leverage: float = 5.0
    trailing_stop: bool = True
    trailing_pct: float = 0.025       # Trailing distance (ROE or price, per risk_basis)
    time_limit_hours: float = 24.0    # Max time in position
    risk_basis: str = "roe"           # "roe" = margin-based, "price" = raw price move
    reward_to_risk_ratio: float = 5.0
    enforce_reward_to_risk: bool = True
    break_even_at_r: float = 1.0      # Promote stop to breakeven after this many R
    break_even_buffer_pct: float = 0.005  # Buffer above/below entry (same basis as risk_basis)
    trail_activate_at_r: float = 2.0  # Start trailing after this many R

    def __post_init__(self) -> None:
        self.stop_loss_pct = max(float(self.stop_loss_pct or 0.0), 0.0)
        self.take_profit_pct = max(float(self.take_profit_pct or 0.0), 0.0)
        self.reward_to_risk_ratio = max(float(self.reward_to_risk_ratio or 0.0), 0.0)
        self.break_even_at_r = max(float(self.break_even_at_r or 0.0), 0.0)
        self.break_even_buffer_pct = max(float(self.break_even_buffer_pct or 0.0), 0.0)
        self.trail_activate_at_r = max(float(self.trail_activate_at_r or 0.0), 0.0)
        basis = str(self.risk_basis or "roe").strip().lower()
        self.risk_basis = basis if basis in {"roe", "price"} else "roe"
        self.sync_reward_to_risk()

    def sync_reward_to_risk(self) -> None:
        """Keep TP aligned to the configured reward-to-risk ratio."""
        if self.enforce_reward_to_risk and self.stop_loss_pct > 0 and self.reward_to_risk_ratio > 0:
            self.take_profit_pct = self.stop_loss_pct * self.reward_to_risk_ratio

    @staticmethod
    def _normalized_leverage(leverage: float) -> float:
        try:
            return max(float(leverage or 1.0), 1.0)
        except (TypeError, ValueError):
            return 1.0

    def resolve_price_stop_loss_pct(self, leverage: float) -> float:
        """Resolve the raw price-move stop percentage for the given leverage."""
        if self.risk_basis == "roe":
            return self.stop_loss_pct / self._normalized_leverage(leverage)
        return self.stop_loss_pct

    def resolve_price_take_profit_pct(self, leverage: float) -> float:
        """Resolve the raw price-move TP percentage for the given leverage."""
        if self.risk_basis == "roe":
            return self.take_profit_pct / self._normalized_leverage(leverage)
        return self.take_profit_pct

    def resolve_price_trailing_pct(self, leverage: float) -> float:
        """Resolve trailing-stop distance in raw price terms."""
        if self.risk_basis == "roe":
            return self.trailing_pct / self._normalized_leverage(leverage)
        return self.trailing_pct

    def resolve_price_break_even_buffer_pct(self, leverage: float) -> float:
        """Resolve the breakeven buffer in raw price terms."""
        if self.risk_basis == "roe":
            return self.break_even_buffer_pct / self._normalized_leverage(leverage)
        return self.break_even_buffer_pct

    def resolve_roe_stop_loss_pct(self, leverage: float) -> float:
        """Resolve stop distance in ROE space."""
        if self.risk_basis == "roe":
            return self.stop_loss_pct
        return self.stop_loss_pct * self._normalized_leverage(leverage)

    def resolve_roe_take_profit_pct(self, leverage: float) -> float:
        """Resolve take-profit distance in ROE space."""
        if self.risk_basis == "roe":
            return self.take_profit_pct
        return self.take_profit_pct * self._normalized_leverage(leverage)

    def resolve_trigger_prices(self, entry_price: float, side: str, leverage: float) -> tuple[float, float]:
        """Convert risk targets into absolute trigger prices."""
        stop_loss_pct = self.resolve_price_stop_loss_pct(leverage)
        take_profit_pct = self.resolve_price_take_profit_pct(leverage)
        side_value = str(side or "").strip().lower()
        is_long = side_value in {"buy", "long"}
        if is_long:
            return (
                entry_price * (1 - stop_loss_pct),
                entry_price * (1 + take_profit_pct),
            )
        return (
            entry_price * (1 + stop_loss_pct),
            entry_price * (1 - take_profit_pct),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeSignal:
    """
    Universal trade signal that every source must produce.

    This is the ONLY format accepted by the decision firewall.
    Anything that doesn't conform gets rejected.
    """
    # Required fields
    coin: str                          # e.g. "BTC", "ETH", "SOL"
    side: SignalSide                   # long or short
    confidence: float                  # 0.0 to 1.0
    source: SignalSource               # where this signal came from
    reason: str                        # human-readable explanation

    # Risk management
    risk: RiskParams = field(default_factory=RiskParams)

    # Sizing
    position_pct: float = 0.08        # % of portfolio (default 8%)
    leverage: float = 2.0             # Leverage to use
    size: float = 0.0                 # Position size in asset units (0 = auto from position_pct)
    context: Dict[str, Any] = field(default_factory=dict)

    # Context
    entry_price: float = 0.0          # Expected entry price (0 = market)
    strategy_id: Optional[int] = None # Link to strategy if from strategy scorer
    strategy_type: str = ""           # e.g. "momentum_long", "mean_reversion"
    trader_address: str = ""          # Link to trader if from copy trade

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # Auto-generated trace ID
    regime: str = ""                  # Market regime when signal was generated
    regime_size_modifier: float = 1.0 # Size adjustment from regime
    options_flow_aligned: Optional[bool] = None  # Did options flow confirm?
    volume_confirmed: Optional[bool] = None       # Did multi-exchange volume confirm?

    # Scoring (filled by agent scoring system)
    source_accuracy: float = 0.0      # Historical accuracy of this signal source
    source_sharpe: float = 0.0        # Historical Sharpe of this source

    # Strength classification (derived)
    @property
    def strength(self) -> SignalStrength:
        if self.confidence >= 0.7:
            return SignalStrength.STRONG
        elif self.confidence >= 0.4:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @property
    def effective_size(self) -> float:
        """Position size adjusted by regime and confidence."""
        return self.position_pct * self.regime_size_modifier * min(self.confidence * 1.5, 1.0)

    def activate_trace(self) -> str:
        """Set this signal's ID as the active trace for downstream logging.

        Call once when the signal enters the execution pipeline (e.g. at
        firewall validation).  All log lines emitted in the same thread
        will include ``trace_id`` until another signal activates its trace
        or the context var is reset.

        Returns:
            The active trace_id.
        """
        current_trace_id.set(self.signal_id)
        return self.signal_id

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["source"] = self.source.value
        d["strength"] = self.strength.value
        d["effective_size"] = self.effective_size
        d["risk"] = self.risk.to_dict()
        return d

    def validate(self) -> bool:
        """Validate signal has required fields and reasonable values."""
        if not self.coin or len(self.coin) < 2:
            return False
        if not 0.0 <= self.confidence <= 1.0:
            return False
        if self.position_pct <= 0 or self.position_pct > 0.25:
            return False
        if self.leverage < 1 or self.leverage > 20:
            return False
        if self.entry_price < 0:
            return False
        return True


def signal_from_strategy(strategy: Dict, coin: str, side: str,
                          entry_price: float, confidence: float = 0.5) -> TradeSignal:
    """Helper: create a TradeSignal from a strategy dict."""
    return TradeSignal(
        coin=coin,
        side=SignalSide(side),
        confidence=confidence,
        source=SignalSource.STRATEGY,
        reason=f"Strategy: {strategy.get('name', 'unknown')} ({strategy.get('strategy_type', '')})",
        strategy_id=strategy.get("id"),
        strategy_type=strategy.get("strategy_type", ""),
        entry_price=entry_price,
    )


def signal_from_copy_trade(trader_address: str, coin: str, side: str,
                             entry_price: float, confidence: float = 0.6) -> TradeSignal:
    """Helper: create a TradeSignal from a copy trade detection."""
    return TradeSignal(
        coin=coin,
        side=SignalSide(side),
        confidence=confidence,
        source=SignalSource.COPY_TRADE,
        reason=f"Copy: {trader_address[:10]}... opened {side} {coin}",
        trader_address=trader_address,
        entry_price=entry_price,
    )


def signal_from_execution_dict(execution: Dict[str, Any]) -> TradeSignal:
    """
    Build a TradeSignal from an executed paper/copy trade dict.

    This keeps live mirroring resilient when upstream execution code returns
    plain dicts instead of TradeSignal instances.
    """
    source_raw = str(execution.get("source", "")).strip().lower()
    if source_raw == SignalSource.COPY_TRADE.value or execution.get("trader_address"):
        source = SignalSource.COPY_TRADE
    elif source_raw == SignalSource.OPTIONS_FLOW.value:
        source = SignalSource.OPTIONS_FLOW
    elif source_raw == SignalSource.POLYMARKET.value:
        source = SignalSource.POLYMARKET
    elif source_raw == SignalSource.LIQUIDATION_STRATEGY.value:
        source = SignalSource.LIQUIDATION_STRATEGY
    elif source_raw == SignalSource.ARENA_CHAMPION.value:
        source = SignalSource.ARENA_CHAMPION
    elif source_raw == SignalSource.WHALE_TRADE.value:
        source = SignalSource.WHALE_TRADE
    else:
        source = SignalSource.STRATEGY

    side_value = str(execution.get("side", "long")).strip().lower()
    entry_price = float(execution.get("entry_price", 0.0) or 0.0)

    # Reconstruct RiskParams from absolute SL/TP prices when available.
    # Paper trades carry stop_loss/take_profit as absolute prices; convert
    # them back to percentages so the live trader places correct triggers.
    risk = RiskParams(risk_basis="price", enforce_reward_to_risk=False)
    sl_price = float(execution.get("stop_loss", 0.0) or 0.0)
    tp_price = float(execution.get("take_profit", 0.0) or 0.0)
    if entry_price > 0 and sl_price > 0:
        risk.stop_loss_pct = abs(sl_price - entry_price) / entry_price
    if entry_price > 0 and tp_price > 0:
        risk.take_profit_pct = abs(tp_price - entry_price) / entry_price

    metadata = execution.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata or "{}")
        except Exception:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    risk_policy = metadata.get("risk_policy", {}) if isinstance(metadata, dict) else {}
    if isinstance(risk_policy, dict):
        if "stop_roe_pct" in risk_policy:
            risk.stop_loss_pct = float(risk_policy.get("stop_roe_pct") or risk.stop_loss_pct)
            risk.risk_basis = "roe"
        if "take_profit_roe_pct" in risk_policy:
            risk.take_profit_pct = float(risk_policy.get("take_profit_roe_pct") or risk.take_profit_pct)
        if "reward_multiple" in risk_policy:
            risk.reward_to_risk_ratio = float(risk_policy.get("reward_multiple") or risk.reward_to_risk_ratio)
        if "time_limit_hours" in risk_policy:
            risk.time_limit_hours = float(risk_policy.get("time_limit_hours") or risk.time_limit_hours)
        if "trailing_enabled" in risk_policy:
            risk.trailing_stop = bool(risk_policy.get("trailing_enabled"))
        if "trailing_distance_roe_pct" in risk_policy:
            risk.trailing_pct = float(risk_policy.get("trailing_distance_roe_pct") or risk.trailing_pct)
        if "breakeven_at_r" in risk_policy:
            risk.break_even_at_r = float(risk_policy.get("breakeven_at_r") or risk.break_even_at_r)
        if "breakeven_buffer_roe_pct" in risk_policy:
            risk.break_even_buffer_pct = float(
                risk_policy.get("breakeven_buffer_roe_pct") or risk.break_even_buffer_pct
            )
        if "trail_after_r" in risk_policy:
            risk.trail_activate_at_r = float(risk_policy.get("trail_after_r") or risk.trail_activate_at_r)

    signal = TradeSignal(
        coin=str(execution.get("coin", "")),
        side=SignalSide(side_value or "long"),
        confidence=float(execution.get("confidence", 0.5)),
        source=source,
        reason=str(execution.get("reason", "Mirrored execution")).strip() or "Mirrored execution",
        entry_price=entry_price,
        risk=risk,
        strategy_id=execution.get("strategy_id"),
        strategy_type=str(execution.get("strategy_type", "")),
        trader_address=str(execution.get("trader_address", execution.get("trader", ""))),
        leverage=float(execution.get("leverage", 2.0) or 2.0),
        position_pct=float(execution.get("position_pct", 0.08) or 0.08),
        size=float(execution.get("size", 0.0) or 0.0),
        context={
            "risk_policy": risk_policy if isinstance(risk_policy, dict) else {},
            "features": metadata.get("features", {}) if isinstance(metadata, dict) else {},
        },
        signal_id=str(execution.get("signal_id", "")),
        regime=str(execution.get("regime", "")),
        source_accuracy=float(execution.get("source_accuracy", 0.0) or 0.0),
    )
    return signal


def signal_from_options_flow(ticker: str, direction: str,
                              net_flow: float, prints: int,
                              conviction_pct: float) -> TradeSignal:
    """Helper: create a TradeSignal from options flow conviction."""
    side = SignalSide.LONG if direction.upper() == "BULLISH" else SignalSide.SHORT
    confidence = conviction_pct / 100.0
    return TradeSignal(
        coin=ticker,
        side=side,
        confidence=confidence,
        source=SignalSource.OPTIONS_FLOW,
        reason=f"Options flow: net ${net_flow:,.0f} {direction} across {prints} prints",
        options_flow_aligned=True,
    )


def signal_from_whale_trade(coin: str, whale_side: str, notional: float,
                            exchange: str = "crypto.com",
                            confidence: float = 0.55) -> TradeSignal:
    """Helper: create a TradeSignal from whale trade detection.

    Args:
        coin: Coin symbol (e.g., "BTC", "ETH")
        whale_side: "buy" or "sell" from whale trade
        notional: Trade notional value in USD
        exchange: Exchange name (default "crypto.com")
        confidence: Signal confidence (default 0.55, can be adjusted for notional size)

    Returns:
        TradeSignal with whale trade direction inferred from side
    """
    # Infer signal direction: whale buying = long signal, whale selling = short signal
    side = SignalSide.LONG if whale_side.lower() == "buy" else SignalSide.SHORT

    return TradeSignal(
        coin=coin,
        side=side,
        confidence=confidence,
        source=SignalSource.WHALE_TRADE,
        reason=f"Whale {whale_side.upper()} ${notional:,.0f} on {exchange}",
        entry_price=0.0,  # No specific entry price for whale signals
    )
