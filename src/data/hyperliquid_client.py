"""
Hyperliquid API client for fetching leaderboard, trader positions, market data, and PnL.
Uses only the public Info endpoint — no API key required for read-only operations.

V3: All requests route through the centralized APIManager (token bucket +
    TTL cache + WebSocket feed).  The public function signatures are unchanged
    so every module that imports this file keeps working.
"""
import requests
import time
import logging
import random
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.core.api_manager import get_manager, Priority

logger = logging.getLogger(__name__)


# ─── Internal post — now delegates to APIManager ─────────────────

def _post(payload: dict, retries: int = 3,
          priority: Priority = None) -> Optional[dict]:
    """
    POST to the Hyperliquid info endpoint.
    Routes through the global APIManager for rate limiting + caching.
    """
    mgr = get_manager()
    return mgr.post(payload, priority=priority, retries=retries)


# Legacy compatibility: modules that imported _rate_limit directly
def _rate_limit():
    """No-op — rate limiting now handled by APIManager's token bucket."""
    pass


# ─── Market Metadata ───────────────────────────────────────────

def get_meta():
    """Get exchange metadata (list of all perpetual assets)."""
    return _post({"type": "meta"}, priority=Priority.LOW)


def get_all_mids():
    """
    Get mid-prices for all assets. Returns dict like {'BTC': '67432.5', ...}.
    Served from WebSocket when available (zero REST cost).
    """
    return _post({"type": "allMids"}, priority=Priority.HIGH)


def get_asset_contexts():
    """Get detailed context for all assets (funding, open interest, etc.)."""
    data = _post({"type": "metaAndAssetCtxs"}, priority=Priority.NORMAL)
    if data and len(data) == 2:
        meta, contexts = data
        universe = meta.get("universe", [])
        result = {}
        for i, ctx in enumerate(contexts):
            if i < len(universe):
                coin = universe[i]["name"]
                def _safe_float(val, default=0):
                    try:
                        return float(val) if val is not None else default
                    except (ValueError, TypeError):
                        return default

                result[coin] = {
                    "funding": _safe_float(ctx.get("funding")),
                    "open_interest": _safe_float(ctx.get("openInterest")),
                    "day_volume": _safe_float(ctx.get("dayNtlVlm")),
                    "mark_price": _safe_float(ctx.get("markPx")),
                    "oracle_price": _safe_float(ctx.get("oraclePx")),
                    "premium": _safe_float(ctx.get("premium")),
                }
                result[coin].update(ctx)
        return result
    return {}


# ─── User / Trader Data ───────────────────────────────────────

def get_user_state(address: str):
    """
    Get a trader's clearinghouse state: positions, margin, account value.
    This is the main endpoint for analyzing what a trader is doing.
    """
    data = _post({"type": "clearinghouseState", "user": address},
                 priority=Priority.HIGH)
    if not data:
        return None

    positions = []
    for pos_data in data.get("assetPositions", []):
        pos = pos_data.get("position", pos_data)
        szi = float(pos.get("szi", 0))
        if szi == 0:
            continue  # Skip flat/zero positions entirely — no side to assign
        positions.append({
            "coin": pos.get("coin", ""),
            "side": "long" if szi > 0 else "short",
            "size": abs(szi),
            "entry_price": float(pos.get("entryPx", 0)),
            "leverage": float(pos.get("leverage", {}).get("value", 1)) if isinstance(pos.get("leverage"), dict) else float(pos.get("leverage", 1)),
            "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
            "return_on_equity": float(pos.get("returnOnEquity", 0)),
            "margin_used": float(pos.get("marginUsed", 0)),
            "liquidation_price": pos.get("liquidationPx"),
        })

    margin_summary = data.get("marginSummary", data.get("crossMarginSummary", {}))
    return {
        "positions": positions,
        "account_value": float(margin_summary.get("accountValue", 0)),
        "total_margin_used": float(margin_summary.get("totalMarginUsed", 0)),
        "total_ntl_pos": float(margin_summary.get("totalNtlPos", 0)),
        "withdrawable": float(data.get("withdrawable", 0)),
    }


def get_user_fills(address: str, start_time: Optional[int] = None):
    """
    Get a trader's recent fills (executed trades).
    start_time is in milliseconds epoch.
    """
    payload = {"type": "userFills", "user": address}
    if start_time:
        payload["startTime"] = start_time
    # LOW priority: historical fill downloads should yield to live trading/monitoring
    data = _post(payload, priority=Priority.LOW)
    if not data:
        return []

    fills = []
    for fill in data:
        fills.append({
            "coin": fill.get("coin", ""),
            "side": fill.get("side", "").lower(),
            "price": float(fill.get("px", 0)),
            "size": float(fill.get("sz", 0)),
            "time": fill.get("time", 0),
            "fee": float(fill.get("fee", 0)),
            "is_liquidation": fill.get("liquidation", False),
            "start_position": fill.get("startPosition", ""),
            "direction": fill.get("dir", ""),
            "closed_pnl": float(fill.get("closedPnl", 0)),
            "hash": fill.get("hash", ""),
            "oid": fill.get("oid", 0),
            "crossed": fill.get("crossed", False),
        })
    return fills


def get_user_funding(address: str, start_time: Optional[int] = None):
    """Get a trader's funding payment history."""
    payload = {"type": "userFunding", "user": address}
    if start_time:
        payload["startTime"] = start_time
    return _post(payload) or []


def get_user_non_funding_ledger(address: str, start_time: Optional[int] = None):
    """Get non-funding ledger updates (deposits, withdrawals, liquidations, etc.)."""
    payload = {"type": "userNonFundingLedgerUpdates", "user": address}
    if start_time:
        payload["startTime"] = start_time
    return _post(payload) or []


# ─── Leaderboard ───────────────────────────────────────────────

def get_leaderboard():
    """
    Fetch the Hyperliquid leaderboard from multiple sources.
    Returns raw data from whichever source responds.

    Note: The info endpoint's {"type": "leaderboard"} returns 422 BAD_PAYLOAD
    on newer API versions, so we try the stats endpoint first.
    """
    # Method 1: Stats data endpoint (used by the frontend — most reliable)
    try:
        mgr = get_manager()
        if not mgr.bucket.acquire(priority=Priority.LOW, timeout=10):
            logger.debug("Leaderboard: couldn't acquire token for stats endpoint")
        else:
            resp = requests.get(
                "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard",
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"Leaderboard from stats endpoint: type={type(data).__name__}")
                return data
            else:
                logger.debug(f"Stats leaderboard returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Stats leaderboard failed: {e}")

    # Method 2: Fallback to info endpoint (may 422 on newer API versions)
    try:
        data = _post({"type": "leaderboard"}, priority=Priority.LOW)
        if data:
            logger.info(f"Leaderboard from info endpoint: type={type(data).__name__}, "
                       f"keys={list(data.keys()) if isinstance(data, dict) else f'list[{len(data)}]'}")
            return data
    except Exception as e:
        logger.debug(f"Info leaderboard fallback failed: {e}")

    return None


def get_user_portfolio(address: str):
    """Get historical portfolio/PnL data for a user."""
    return _post({"type": "portfolio", "user": address}, priority=Priority.LOW)


def get_vault_details(vault_address: str):
    """Get details about a Hyperliquid vault (community-managed fund)."""
    return _post({"type": "vaultDetails", "user": vault_address}, priority=Priority.LOW)


# ─── Order Book & Market Data ─────────────────────────────────

def get_l2_book(coin: str):
    """Get L2 order book for a coin."""
    return _post({"type": "l2Book", "coin": coin}, priority=Priority.NORMAL)


def get_candles(coin: str, interval: str = "1h", start_time: Optional[int] = None,
                end_time: Optional[int] = None):
    """
    Get candlestick data.
    interval: '1m', '5m', '15m', '1h', '4h', '1d'
    """
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval}}
    if start_time:
        payload["req"]["startTime"] = start_time
    if end_time:
        payload["req"]["endTime"] = end_time
    return _post(payload, priority=Priority.LOW) or []


def get_recent_trades(coin: str):
    """Get recent trades for a coin."""
    return _post({"type": "recentTrades", "coin": coin}, priority=Priority.NORMAL)


# ─── Utility ───────────────────────────────────────────────────

def get_all_coins():
    """Get list of all available perpetual coins."""
    meta = get_meta()
    if meta and "universe" in meta:
        return [asset["name"] for asset in meta["universe"]]
    return []


def get_funding_history(coin: str, start_time: Optional[int] = None):
    """Get funding rate history for a coin."""
    payload = {"type": "fundingHistory", "coin": coin}
    if start_time:
        payload["req"] = {"coin": coin, "startTime": start_time}
    return _post(payload, priority=Priority.LOW) or []


# ─── Manager access for other modules ────────────────────────────

def start_websocket(coins: list = None):
    """Start the WebSocket feed. Call from main.py during init."""
    mgr = get_manager()
    mgr.start_websocket(coins=coins)


def get_api_stats() -> dict:
    """Get API manager stats for dashboard/logging."""
    mgr = get_manager()
    return mgr.get_stats()


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    print("Testing Hyperliquid API client (V3 — via APIManager)...")

    coins = get_all_coins()
    print(f"Available coins ({len(coins)}): {coins[:10]}...")

    mids = get_all_mids()
    if mids:
        print(f"BTC mid: {mids.get('BTC', 'N/A')}")
        print(f"ETH mid: {mids.get('ETH', 'N/A')}")

    # Test with a known active address (Hyperliquid's own vault)
    state = get_user_state("0xdfc24b077bc1425ad1dea75bcb6f8158e3df2f0f")
    if state:
        print(f"Test account value: ${state['account_value']:,.2f}")
        print(f"Open positions: {len(state['positions'])}")

    stats = get_api_stats()
    print(f"\nAPI Manager stats: {json.dumps(stats, indent=2)}")
    print("API client test complete.")
