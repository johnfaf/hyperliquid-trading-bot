"""
Multi-exchange adapter layer.
Each exchange implements BaseExchangeAdapter for uniform trader discovery.
"""
from .base_adapter import (
    BaseExchangeAdapter,
    NormalizedTrader,
    NormalizedFill,
    NormalizedPosition,
    NormalizedMarketData,
)
from .hyperliquid_adapter import HyperliquidAdapter
from .lighter_adapter import LighterAdapter
from .cross_venue import CrossVenueConfirmation
from .scanner import MultiExchangeScanner

__all__ = [
    "BaseExchangeAdapter",
    "NormalizedTrader",
    "NormalizedFill",
    "NormalizedPosition",
    "NormalizedMarketData",
    "HyperliquidAdapter",
    "LighterAdapter",
    "CrossVenueConfirmation",
    "MultiExchangeScanner",
]
