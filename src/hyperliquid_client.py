"""
Hyperliquid API client for fetching leaderboard, trader positions, market data, and PnL.
Uses only the public Info endpoint — no API key required for read-only operations.
"""
import requests
import time
import logging
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# Rate limiting — adaptive to avoid 429s
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 0.6   # 600ms between requests (safe baseline for HL)
_backoff_until = 0            # timestamp until which we should back off
_consecutive_429s = 0         # Track consecutive rate limits to escalate backoff
_BACKOFF_COOLDOWN = 60        # After a 429 storm, stay slow for 60s


def _rate_limit():
    global _last_request_time, _backoff_until
    now = time.time()
    # If we're in a backoff period, wait it out
    if now < _backoff_until:
        time.sleep(_backoff_until - now)
    # After recent 429s, use a longer interval to avoid re-triggering
    interval = _MIN_REQUEST_INTERVAL
    if _consecutive_429s > 0:
        interval = min(_MIN_REQUEST_INTERVAL * (1 + _consecutive_429s * 0.5), 5.0)
    elapsed = time.time() - _last_request_time
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_request_time = time.time()


def _post(payload: dict, retries: int = 3) -> Optional[dict]:
    """POST to the Hyperliquid info endpoint with retries and 429 backoff."""
    global _backoff_until, _consecutive_429s
    _rate_limit()
    req_type = payload.get("type", "unknown")
    for attempt in range(retries):
        try:
            resp = requests.post(
                config.HYPERLIQUID_INFO_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if resp.status_code == 429:
                _consecutive_429s += 1
                base_wait = min(5 * (2 ** min(attempt, 3)), 60)
                storm_penalty = min(_consecutive_429s * 3, 30)
                wait = base_wait + storm_penalty
                logger.warning(f"Rate limited (429), backing off {wait}s "
                             f"(consecutive={_consecutive_429s}, attempt={attempt+1})")
                _backoff_until = time.time() + wait
                time.sleep(wait)
                continue

            # 4xx client errors (400, 404, 422) — don't retry, these are our fault
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                body_preview = resp.text[:200] if resp.text else "(empty)"
                logger.warning(f"Client error {resp.status_code} for type='{req_type}': "
                             f"{body_preview} (payload keys: {list(payload.keys())})")
                return None

            resp.raise_for_status()
            # Successful request — decay the consecutive counter
            if _consecutive_429s > 0:
                _consecutive_429s = max(0, _consecutive_429s - 1)
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt+1}/{retries}, "
                         f"type='{req_type}'): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ─── Market Metadata ───────────────────────────────────────────

def get_meta():
    """Get exchange metadata (list of all perpetual assets)."""
    return _post({"type": "meta"})


def get_all_mids():
    """Get mid-prices for all assets. Returns dict like {'BTC': '67432.5', ...}."""
    return _post({"type": "allMids"})


def get_asset_contexts():
    """Get detailed context for all assets (funding, open interest, etc.)."""
    data = _post({"type": "metaAndAssetCtxs"})
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
    data = _post({"type": "clearinghouseState", "user": address})
    if not data:
        return None

    positions = []
    for pos_data in data.get("assetPositions", []):
        pos = pos_data.get("position", pos_data)
        positions.append({
            "coin": pos.get("coin", ""),
            "side": "long" if float(pos.get("szi", 0)) > 0 else "short",
            "size": abs(float(pos.get("szi", 0))),
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
    data = _post(payload)
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
    """
    # Method 1: Official info endpoint with "leaderboard" type
    try:
        data = _post({"type": "leaderboard"})
        if data:
            logger.info(f"Leaderboard from info endpoint: type={type(data).__name__}, "
                       f"keys={list(data.keys()) if isinstance(data, dict) else f'list[{len(data)}]'}")
            return data
    except Exception as e:
        logger.warning(f"Info leaderboard failed: {e}")

    # Method 2: Stats data endpoint (used by the frontend)
    try:
        _rate_limit()
        resp = requests.get(
            "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard",
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Leaderboard from stats endpoint: type={type(data).__name__}")
            return data
    except Exception as e:
        logger.warning(f"Stats leaderboard failed: {e}")

    return None


def get_user_portfolio(address: str):
    """Get historical portfolio/PnL data for a user."""
    return _post({"type": "portfolio", "user": address})


def get_vault_details(vault_address: str):
    """Get details about a Hyperliquid vault (community-managed fund)."""
    return _post({"type": "vaultDetails", "user": vault_address})


# ─── Order Book & Market Data ─────────────────────────────────

def get_l2_book(coin: str):
    """Get L2 order book for a coin."""
    return _post({"type": "l2Book", "coin": coin})


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
    return _post(payload) or []


def get_recent_trades(coin: str):
    """Get recent trades for a coin."""
    return _post({"type": "recentTrades", "coin": coin})


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
    return _post(payload) or []


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    print("Testing Hyperliquid API client...")

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

    print("API client test complete.")
