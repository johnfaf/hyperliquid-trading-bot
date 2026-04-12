"""
Hyperliquid API client for fetching leaderboard, trader positions, market data, and PnL.
Uses only the public Info endpoint — no API key required for read-only operations.

V3: All requests route through the centralized APIManager (token bucket +
    TTL cache + WebSocket feed).  The public function signatures are unchanged
    so every module that imports this file keeps working.
"""
import json
import logging
import math
import os
import re
import sys
from typing import Optional, Any
import requests


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.api_manager import get_manager, Priority  # noqa: E402

logger = logging.getLogger(__name__)

# Hyperliquid's /info endpoint requires `user` to be a well-formed 0x-prefixed
# 40-hex-char Ethereum address.  Anything else returns HTTP 422
# "Failed to deserialize the JSON body into the target type".  The discovery
# pipeline occasionally inserts placeholder / truncated addresses into the
# DB — filter them out here instead of spamming the log with 422 warnings.
_ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def _is_valid_eth_address(address: Optional[str]) -> bool:
    """Return True iff the string is a canonical 0x + 40-hex-char address."""
    if not address or not isinstance(address, str):
        return False
    return bool(_ETH_ADDRESS_RE.match(address.strip()))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort numeric coercion for externally sourced payload fields."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    data = _post({"type": "allMids"}, priority=Priority.HIGH)
    if not isinstance(data, dict):
        logger.warning("get_all_mids: unexpected payload type %s", type(data).__name__)
        return {}
    sanitized = {}
    for coin, price in data.items():
        try:
            parsed = float(price)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        sanitized[str(coin)] = price
    return sanitized


def get_asset_contexts():
    """Get detailed context for all assets (funding, open interest, etc.)."""
    data = _post({"type": "metaAndAssetCtxs"}, priority=Priority.NORMAL)
    if isinstance(data, list) and len(data) == 2:
        meta, contexts = data
        if not isinstance(meta, dict) or not isinstance(contexts, list):
            logger.warning(
                "get_asset_contexts: invalid payload shape meta=%s contexts=%s",
                type(meta).__name__,
                type(contexts).__name__,
            )
            return {}
        universe = meta.get("universe", [])
        result = {}
        for i, ctx in enumerate(contexts):
            if i < len(universe):
                coin = universe[i]["name"]
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


def get_user_fee_rate(address: str) -> Optional[dict]:
    """Fetch the account's current maker/taker fee rates from Hyperliquid.

    Returns a dict like ``{"makerRate": "0.00002", "takerRate": "0.000245"}``
    or *None* on failure / invalid address.  Rates are decimal strings
    (e.g. "0.000245" = 2.45 bps).
    """
    if not _is_valid_eth_address(address):
        return None
    try:
        data = _post(
            {"type": "userFees", "user": address.strip()},
            priority=Priority.LOW,
        )
        if isinstance(data, dict):
            return data
        # Some API versions nest inside a list; unwrap if needed.
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else None
    except Exception as exc:
        logger.debug("get_user_fee_rate failed for %s: %s", address[:10], exc)
    return None


# ─── User / Trader Data ───────────────────────────────────────

def get_user_state(address: str):
    """
    Get a trader's clearinghouse state: positions, margin, account value.
    This is the main endpoint for analyzing what a trader is doing.
    """
    if not _is_valid_eth_address(address):
        logger.debug(
            "get_user_state: skipping malformed address %r",
            (address[:16] + "...") if isinstance(address, str) and len(address) > 16 else address,
        )
        return None
    data = _post({"type": "clearinghouseState", "user": address.strip()},
                 priority=Priority.HIGH)
    if not data:
        return None
    if not isinstance(data, dict):
        logger.warning("get_user_state: unexpected payload type %s", type(data).__name__)
        return None

    positions = []
    asset_positions = data.get("assetPositions", [])
    if not isinstance(asset_positions, list):
        logger.warning(
            "get_user_state: invalid assetPositions type %s for address %s",
            type(asset_positions).__name__,
            address[:10],
        )
        return None

    for pos_data in asset_positions:
        if not isinstance(pos_data, dict):
            continue
        pos = pos_data.get("position", pos_data)
        if not isinstance(pos, dict):
            continue
        szi = _safe_float(pos.get("szi", 0))
        if szi == 0:
            continue  # Skip flat/zero positions entirely — no side to assign
        leverage = pos.get("leverage", 1)
        if isinstance(leverage, dict):
            leverage = _safe_float(leverage.get("value", 1), 1.0)
        else:
            leverage = _safe_float(leverage, 1.0)
        positions.append({
            "coin": pos.get("coin", ""),
            "side": "long" if szi > 0 else "short",
            "size": abs(szi),
            "entry_price": _safe_float(pos.get("entryPx", 0)),
            "leverage": leverage,
            "unrealized_pnl": _safe_float(pos.get("unrealizedPnl", 0)),
            "return_on_equity": _safe_float(pos.get("returnOnEquity", 0)),
            "margin_used": _safe_float(pos.get("marginUsed", 0)),
            "liquidation_price": pos.get("liquidationPx"),
        })

    margin_summary = data.get("marginSummary", data.get("crossMarginSummary", {}))
    if not isinstance(margin_summary, dict):
        logger.warning(
            "get_user_state: invalid marginSummary type %s for address %s",
            type(margin_summary).__name__,
            address[:10],
        )
        return None
    account_value = _safe_float(margin_summary.get("accountValue", 0))
    if account_value < 0:
        logger.warning(
            "get_user_state: negative account_value %.4f for address %s; treating payload as invalid",
            account_value,
            address[:10],
        )
        return None
    return {
        "positions": positions,
        "account_value": account_value,
        "total_margin_used": max(0.0, _safe_float(margin_summary.get("totalMarginUsed", 0))),
        "total_ntl_pos": abs(_safe_float(margin_summary.get("totalNtlPos", 0))),
        "withdrawable": max(0.0, _safe_float(data.get("withdrawable", 0))),
    }


def get_user_fills(address: str, start_time: Optional[int] = None):
    """
    Get a trader's recent fills (executed trades).
    start_time is in milliseconds epoch.
    """
    if not _is_valid_eth_address(address):
        return []
    payload = {"type": "userFills", "user": address.strip()}
    if start_time:
        payload["startTime"] = start_time
    # LOW priority: historical fill downloads should yield to live trading/monitoring
    data = _post(payload, priority=Priority.LOW)
    if not data:
        return []
    if not isinstance(data, list):
        logger.warning("get_user_fills: unexpected payload type %s", type(data).__name__)
        return []

    fills = []
    for fill in data:
        if not isinstance(fill, dict):
            continue
        fills.append({
            "coin": fill.get("coin", ""),
            "side": fill.get("side", "").lower(),
            "price": _safe_float(fill.get("px", 0)),
            "size": _safe_float(fill.get("sz", 0)),
            "time": _safe_int(fill.get("time", 0)),
            "fee": _safe_float(fill.get("fee", 0)),
            "is_liquidation": fill.get("liquidation", False),
            "start_position": fill.get("startPosition", ""),
            "direction": fill.get("dir", ""),
            "closed_pnl": _safe_float(fill.get("closedPnl", 0)),
            "hash": fill.get("hash", ""),
            "oid": _safe_int(fill.get("oid", 0)),
            "crossed": fill.get("crossed", False),
        })
    return fills


def get_user_funding(address: str, start_time: Optional[int] = None):
    """Get a trader's funding payment history."""
    if not _is_valid_eth_address(address):
        return []
    payload = {"type": "userFunding", "user": address.strip()}
    if start_time:
        payload["startTime"] = start_time
    return _post(payload) or []


def get_user_non_funding_ledger(address: str, start_time: Optional[int] = None):
    """Get non-funding ledger updates (deposits, withdrawals, liquidations, etc.)."""
    if not _is_valid_eth_address(address):
        return []
    payload = {"type": "userNonFundingLedgerUpdates", "user": address.strip()}
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
    # Note: this is a different domain from the HL info API so the bucket
    # token was previously wasted. We still acquire a token to self-throttle
    # the bot's overall outbound request rate.
    try:
        mgr = get_manager()
        if not mgr.bucket.acquire(priority=Priority.LOW, timeout=10):
            logger.debug("Leaderboard: couldn't acquire token for stats endpoint")
        else:
            resp = requests.get(
                "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard",
                timeout=30,
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


def stop_websocket():
    """Stop the shared WebSocket feed."""
    mgr = get_manager()
    mgr.stop_websocket()


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
