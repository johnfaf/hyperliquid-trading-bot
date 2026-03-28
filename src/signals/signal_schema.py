"""
Trade Signal Schema
===================
Strict dataclass schema that ALL signal sources must conform to.
This is the "contract" between signal generation and execution.

Every signal source (strategy scorer, copy trader, options flow)
must produce a TradeSignal before it reaches the decision firewall.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalSource(str, Enum):
    STRATEGY = "strategy"
    COPY_TRADE = "copy_trade"
    OPTIONS_FLOW = "options_flow"
    MANUAL = "manual"


class SignalStrength(str, Enum):
    STRONG = "strong"       # High conviction — full size
    MODERATE = "moderate"   # Medium conviction — reduced size
    WEAK = "weak"           # Low conviction — minimum size or skip


@dataclass
class RiskParams:
    """Risk parameters attached to every signal."""
    stop_loss_pct: float = 0.05       # 5% default stop loss
    take_profit_pct: float = 0.10     # 10% default take profit
    max_leverage: float = 5.0
    trailing_stop: bool = True
    trailing_pct: float = 0.025       # 2.5% trailing stop
    time_limit_hours: float = 24.0    # Max time in position

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

    # Context
    entry_price: float = 0.0          # Expected entry price (0 = market)
    strategy_id: Optional[int] = None # Link to strategy if from strategy scorer
    strategy_type: str = ""           # e.g. "momentum_long", "mean_reversion"
    trader_address: str = ""          # Link to trader if from copy trade

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    signal_id: str = ""               # Unique ID for tracking
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
