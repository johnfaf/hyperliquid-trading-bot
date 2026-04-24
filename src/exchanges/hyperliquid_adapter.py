"""
Hyperliquid Exchange Adapter
=============================
Wraps the existing hyperliquid_client.py into the BaseExchangeAdapter interface.
This is the primary venue — has the richest trader discovery via leaderboard.
"""
import time
import logging
from typing import Dict, List, Optional

from .base_adapter import (
    BaseExchangeAdapter,
    NormalizedTrader,
    NormalizedFill,
    NormalizedPosition,
    NormalizedMarketData,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data import hyperliquid_client as hl

logger = logging.getLogger(__name__)


def _coerce_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present_float(payload: Dict, *keys: str) -> float:
    for key in keys:
        if key not in payload:
            continue
        parsed = _coerce_optional_float(payload.get(key))
        if parsed is not None:
            return parsed
    return 0.0


class HyperliquidAdapter(BaseExchangeAdapter):
    """
    Adapter for Hyperliquid's public Info API.

    This wraps the existing hyperliquid_client module so the multi-exchange
    pipeline can consume HL data through the same interface as other venues.

    Key endpoints:
      - POST https://api.hyperliquid.xyz/info
      - GET  https://stats-data.hyperliquid.xyz/Mainnet/leaderboard
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(
            name="hyperliquid",
            base_url="https://api.hyperliquid.xyz/info",
            config=config or {},
        )
        self._coin_cache: List[str] = []
        self._cache_time: float = 0
        self._CACHE_TTL = 300  # 5 min

    # ─── Required: Trader Discovery ────────────────────────

    def get_top_traders(self, limit: int = 100) -> List[NormalizedTrader]:
        """
        Fetch leaderboard and return normalized traders.
        Uses stats-data leaderboard → falls back to info endpoint.
        """
        self._request_count += 1
        try:
            raw = hl.get_leaderboard()
            if not raw:
                self._error_count += 1
                return []

            traders = []

            # Handle different leaderboard response formats
            entries = []
            if isinstance(raw, dict):
                entries = raw.get("leaderboardRows", raw.get("data", []))
            elif isinstance(raw, list):
                entries = raw

            for entry in entries[:limit]:
                addr = ""
                pnl = 0.0
                display = None

                if isinstance(entry, dict):
                    addr = entry.get("ethAddress", entry.get("address", ""))
                    display = hl.mask_display_name(
                        entry.get("displayName", entry.get("accountName"))
                    )
                    # Various PnL field names across HL API versions
                    pnl = _first_present_float(entry, "accountValue", "totalPnl", "allTime")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    addr = str(entry[0])
                    pnl = float(entry[1]) if len(entry) > 1 else 0.0

                if not addr:
                    continue

                traders.append(NormalizedTrader(
                    address=addr,
                    exchange="hyperliquid",
                    display_name=display,
                    pnl_total=pnl,
                    raw_data=(
                        hl.redact_leaderboard_identity_fields(entry)
                        if isinstance(entry, dict)
                        else {"raw": entry}
                    ),
                ))

            logger.info(f"Hyperliquid: fetched {len(traders)} traders from leaderboard")
            return traders

        except Exception as e:
            self._error_count += 1
            logger.error(f"Hyperliquid get_top_traders failed: {e}")
            return []

    # ─── Required: Fills ───────────────────────────────────

    def get_trader_fills(self, address: str, limit: int = 200) -> List[NormalizedFill]:
        """Fetch recent fills for a trader via userFills endpoint."""
        self._request_count += 1
        try:
            raw_fills = hl.get_user_fills(address)
            if not raw_fills:
                return []

            fills = []
            for f in raw_fills[:limit]:
                coin = f.get("coin", "")
                price = f.get("price", 0)
                size = f.get("size", 0)

                fills.append(NormalizedFill(
                    exchange="hyperliquid",
                    trader_address=address,
                    coin=self.normalize_coin_symbol(coin),
                    side=f.get("side", "").lower(),
                    size=size,
                    price=price,
                    notional_usd=price * size,
                    fee=f.get("fee", 0),
                    realized_pnl=f.get("closed_pnl", 0),
                    timestamp=str(f.get("time", "")),
                    trade_id=f.get("hash", ""),
                    is_liquidation=f.get("is_liquidation", False),
                ))

            return fills

        except Exception as e:
            self._error_count += 1
            logger.error(f"Hyperliquid get_trader_fills failed for {address[:10]}...: {e}")
            return []

    # ─── Required: Positions ───────────────────────────────

    def get_trader_positions(self, address: str) -> List[NormalizedPosition]:
        """Fetch current positions via clearinghouseState endpoint."""
        self._request_count += 1
        try:
            state = hl.get_user_state(address)
            if not state or "positions" not in state:
                return []

            positions = []
            for p in state["positions"]:
                if p.get("size", 0) == 0:
                    continue
                positions.append(NormalizedPosition(
                    exchange="hyperliquid",
                    trader_address=address,
                    coin=self.normalize_coin_symbol(p.get("coin", "")),
                    side=p.get("side", "long"),
                    size=abs(p.get("size", 0)),
                    entry_price=p.get("entry_price", 0),
                    mark_price=0,  # Not in clearinghouse response
                    unrealized_pnl=p.get("unrealized_pnl", 0),
                    leverage=p.get("leverage", 1),
                    margin_used=p.get("margin_used", 0),
                    liquidation_price=float(p.get("liquidation_price", 0) or 0),
                ))

            return positions

        except Exception as e:
            self._error_count += 1
            logger.error(f"Hyperliquid get_trader_positions failed for {address[:10]}...: {e}")
            return []

    # ─── Required: Market Data ─────────────────────────────

    def get_market_data(self, coins: Optional[List[str]] = None) -> List[NormalizedMarketData]:
        """Fetch market data from metaAndAssetCtxs + allMids."""
        self._request_count += 1
        try:
            contexts = hl.get_asset_contexts()
            mids = hl.get_all_mids() or {}

            if not contexts:
                return []

            results = []
            for coin, ctx in contexts.items():
                if coins and coin not in coins:
                    continue

                mid = hl._safe_float(mids.get(coin, 0))
                funding = _first_present_float(ctx, "funding")
                oi = _first_present_float(ctx, "open_interest", "openInterest")
                vol = _first_present_float(ctx, "day_volume", "dayNtlVlm")
                mark = _first_present_float(ctx, "mark_price", "markPx")
                index = _first_present_float(ctx, "oracle_price", "oraclePx")

                results.append(NormalizedMarketData(
                    exchange="hyperliquid",
                    coin=coin,
                    mid_price=mid,
                    funding_rate=funding,
                    open_interest=oi,
                    volume_24h=vol,
                    mark_price=mark,
                    index_price=index,
                ))

            return results

        except Exception as e:
            self._error_count += 1
            logger.error(f"Hyperliquid get_market_data failed: {e}")
            return []

    # ─── Required: Markets ─────────────────────────────────

    def get_available_markets(self) -> List[str]:
        """Get all available perpetual symbols."""
        now = time.time()
        if self._coin_cache and (now - self._cache_time) < self._CACHE_TTL:
            return self._coin_cache

        self._request_count += 1
        try:
            coins = hl.get_all_coins()
            if coins:
                self._coin_cache = coins
                self._cache_time = now
            return coins or []
        except Exception as e:
            self._error_count += 1
            logger.error(f"Hyperliquid get_available_markets failed: {e}")
            return self._coin_cache  # Return stale cache if available

    # ─── Required: Symbol Normalization ────────────────────

    def normalize_coin_symbol(self, raw_symbol: str) -> str:
        """
        Hyperliquid already uses clean symbols (BTC, ETH, SOL).
        Just strip any whitespace and uppercase.
        """
        return raw_symbol.strip().upper()

    # ─── Optional: Funding Rates ───────────────────────────

    def get_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates from asset contexts."""
        self._request_count += 1
        try:
            contexts = hl.get_asset_contexts()
            if not contexts:
                return {}
            return {
                coin: float(ctx.get("funding", 0))
                for coin, ctx in contexts.items()
            }
        except Exception:
            self._error_count += 1
            return {}

    # ─── Optional: PnL History ─────────────────────────────

    def get_trader_pnl_history(self, address: str, days: int = 30) -> List[Dict]:
        """Get portfolio PnL history."""
        self._request_count += 1
        try:
            portfolio = hl.get_user_portfolio(address)
            if not portfolio:
                return []
            # Portfolio response format varies; return raw for now
            return portfolio if isinstance(portfolio, list) else [portfolio]
        except Exception:
            self._error_count += 1
            return []
