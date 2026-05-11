"""Drop-in replacement for `src.core.api_manager.APIManager` in replay mode.

The production APIManager is the single chokepoint for Hyperliquid traffic --
every module that needs price/funding/orderbook data routes through its
`post(payload, ...)` method. By replacing the singleton with this shim during
replay, we redirect ALL of that traffic to the candle oracle without
modifying every caller.

What's intercepted:
  - candleSnapshot   -> oracle.get_range, translated to HL response format
  - allMids          -> {coin: str_price} from oracle.get_latest_price
  - metaAndAssetCtxs -> universe + neutral contexts (funding=0, oi=0, vol=0)
  - l2Book           -> single-level book at mid +/- 1bp (paper trader needs *something*)
  - fundingHistory   -> empty list; phase 2 will replay from a funding archive
  - userFills,       -> safe defaults; paper mode shouldn't hit these
    clearinghouseState,
    userState
  - anything else    -> raises ReplayInterceptError; "we missed a request type"
                        is a bug in the harness, not something to silently swallow.

What's NOT here yet (phase 2):
  - Real funding rate replay (need historical funding archive)
  - L2 orderbook replay (need orderbook snapshots; currently faked from mid)
  - Multi-coin universe (cache currently has BTC only; non-BTC requests return empty)
"""
from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from src.backtest.replay.candle_oracle import CandleOracle, TIMEFRAME_MS
from src.backtest.replay.clock import Clock

logger = logging.getLogger(__name__)


HL_INTERVAL_TO_TF = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


class ReplayInterceptError(RuntimeError):
    """Raised when the shim sees a request type it doesn't know how to handle.

    Better to fail loud than silently return None and have the live bot's
    fallback path inadvertently make decisions on degenerate data.
    """


class ReplayAPIManager:
    """Drop-in for APIManager during replay.

    Constructed with a `CandleOracle` and a `Clock`. The harness builds one
    of these at startup and replaces the singleton via `set_replay_manager`.

    Telemetry: every intercepted call is counted by `req_type` and unknown
    request types are tallied separately so the harness operator can see
    which paths got hit.
    """

    def __init__(
        self,
        oracle: CandleOracle,
        clock: Clock,
        *,
        known_coins: Optional[List[str]] = None,
        funding_rate_8h: float = 0.0,
        # Set False to allow unhandled request types (returns None instead of raising).
        # In the default strict mode we fail loud on anything we didn't anticipate.
        strict: bool = True,
    ):
        self._oracle = oracle
        self._clock = clock
        self._funding_rate_8h = float(funding_rate_8h)
        self._strict = bool(strict)
        self._lock = threading.Lock()
        self._call_counts: Counter[str] = Counter()
        self._unknown_calls: Counter[str] = Counter()
        self._coin_cache_misses: Counter[str] = Counter()

        if known_coins is None:
            known_coins = oracle.available_coins("1m") or oracle.available_coins("1h")
        self._known_coins = sorted({c.upper() for c in known_coins})
        if not self._known_coins:
            raise RuntimeError("ReplayAPIManager: no coins available in oracle cache")

        # Match the live manager's surface enough that callers don't break when
        # they probe `manager.ws.is_connected()` or similar.
        self.ws = _NullWebSocket()
        self.bucket = _NullBucket()
        self.cache = _NullCache()

    # ---- main intercept ------------------------------------------------

    def post(
        self,
        payload: Dict[str, Any],
        priority: Any = None,
        cache_ttl: Optional[float] = None,
        retries: int = 0,
        endpoint_url: Optional[str] = None,
        cache_response: bool = True,
        req_type: Optional[str] = None,
        timeout: int = 30,
        raise_on_timeout: bool = False,
        force_fresh: bool = False,
    ) -> Optional[Any]:
        rt = req_type or payload.get("type") or "unknown"
        with self._lock:
            self._call_counts[rt] += 1

        handler = _HANDLERS.get(rt)
        if handler is None:
            with self._lock:
                self._unknown_calls[rt] += 1
            if self._strict:
                raise ReplayInterceptError(
                    f"ReplayAPIManager: no handler for req_type={rt!r}. "
                    f"Either add a handler or run with strict=False to ignore."
                )
            logger.warning("ReplayAPIManager: unhandled req_type=%r returning None", rt)
            return None
        return handler(self, payload)

    # ---- handlers ------------------------------------------------------

    def _handle_all_mids(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Return latest mid for every known coin. HL returns string-typed prices."""
        out: Dict[str, str] = {}
        for coin in self._known_coins:
            px = self._oracle.get_latest_price(coin, "1m") or self._oracle.get_latest_price(coin, "1h")
            if px is None:
                with self._lock:
                    self._coin_cache_misses[coin] += 1
                continue
            out[coin] = f"{px}"
        return out

    def _handle_meta_and_asset_ctxs(self, payload: Dict[str, Any]) -> List[Any]:
        """Mirror HL's [{universe: [...]}, [{coin_ctx}, ...]] shape with neutral ctxs."""
        universe = [
            {"name": coin, "szDecimals": 4, "maxLeverage": 50}
            for coin in self._known_coins
        ]
        ctxs = []
        for coin in self._known_coins:
            px = self._oracle.get_latest_price(coin, "1m") or self._oracle.get_latest_price(coin, "1h")
            ctxs.append({
                "funding": str(self._funding_rate_8h),
                "openInterest": "0",
                "prevDayPx": str(px) if px is not None else "0",
                "dayNtlVlm": "0",
                "premium": "0",
                "oraclePx": str(px) if px is not None else "0",
                "markPx": str(px) if px is not None else "0",
                "midPx": str(px) if px is not None else "0",
                "impactPxs": [str(px) if px is not None else "0"] * 2,
            })
        return [{"universe": universe}, ctxs]

    def _handle_candle_snapshot(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Translate `{type, req: {coin, interval, startTime, endTime}}` to oracle reads."""
        req = payload.get("req") or {}
        coin = (req.get("coin") or "").upper()
        interval = req.get("interval") or "1h"
        start_ts = req.get("startTime")
        end_ts = req.get("endTime")

        tf = HL_INTERVAL_TO_TF.get(interval)
        if tf is None:
            logger.warning("ReplayAPIManager: unknown interval %r", interval)
            return []

        if not coin:
            return []
        if coin not in self._known_coins:
            with self._lock:
                self._coin_cache_misses[coin] += 1
            return []

        # If caller didn't bound the request, give them the most recent ~5000 bars
        # below the clock horizon -- mirrors HL's default cap.
        if start_ts is None or end_ts is None:
            recent = self._oracle.get_recent(coin, tf, count=5000)
            return [_to_hl_candle(c) for c in recent]

        # Clamp end_ts to clock horizon so we never serve future bars.
        horizon = self._oracle.get_horizon_ms(tf)
        end_ts = min(int(end_ts), horizon)
        if end_ts <= int(start_ts):
            return []
        candles = self._oracle.get_range(coin, tf, int(start_ts), end_ts)
        return [_to_hl_candle(c) for c in candles]

    def _handle_l2_book(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fake a single-level orderbook at mid +/- 1 bp.

        The paper trader checks `levels` shape but doesn't simulate slippage off
        depth in any meaningful way, so a thin book is fine for v1. Phase 2
        could replay real snapshots if we backfill them.
        """
        coin = (payload.get("coin") or "").upper()
        if coin not in self._known_coins:
            return {}
        mid = self._oracle.get_latest_price(coin, "1m") or self._oracle.get_latest_price(coin, "1h")
        if mid is None:
            return {}
        bid = mid * (1 - 1e-4)
        ask = mid * (1 + 1e-4)
        return {
            "coin": coin,
            "time": self._clock.now_ms(),
            "levels": [
                [{"px": f"{bid}", "sz": "10", "n": 1}],
                [{"px": f"{ask}", "sz": "10", "n": 1}],
            ],
        }

    def _handle_funding_history(self, payload: Dict[str, Any]) -> List[Any]:
        """Stubbed to empty until phase 2 backfills funding."""
        return []

    def _handle_user_fills(self, payload: Dict[str, Any]) -> List[Any]:
        return []

    def _handle_clearinghouse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Paper mode reads this for parity checks; empty + zero is the safe default.
        return {
            "marginSummary": {"accountValue": "0", "totalRawUsd": "0", "totalNtlPos": "0"},
            "crossMarginSummary": {"accountValue": "0"},
            "assetPositions": [],
            "withdrawable": "0",
            "time": self._clock.now_ms(),
        }

    def _handle_user_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._handle_clearinghouse_state(payload)

    def _handle_recent_trades(self, payload: Dict[str, Any]) -> List[Any]:
        return []

    def _handle_meta(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"universe": [{"name": c, "szDecimals": 4, "maxLeverage": 50}
                              for c in self._known_coins]}

    # ---- mirror APIManager methods that callers probe ------------------

    def start_websocket(self, coins: Optional[List[str]] = None) -> None:
        # No-op: replay doesn't subscribe to live feeds.
        return None

    def stop_websocket(self) -> None:
        return None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": "replay",
                "clock_ms": self._clock.now_ms(),
                "known_coins": list(self._known_coins),
                "calls_by_type": dict(self._call_counts),
                "unhandled_calls": dict(self._unknown_calls),
                "coin_cache_misses": dict(self._coin_cache_misses),
            }


def _to_hl_candle(c) -> Dict[str, Any]:
    """OracleCandle -> HL `candleSnapshot` dict shape."""
    return {
        "t": c.timestamp_ms,
        "T": c.close_time_ms,
        "s": c.coin,
        "i": c.timeframe,
        "o": str(c.open),
        "h": str(c.high),
        "l": str(c.low),
        "c": str(c.close),
        "v": str(c.volume),
        "n": 0,
    }


# Module-level dispatch so subclasses can extend cleanly.
_HANDLERS = {
    "allMids": ReplayAPIManager._handle_all_mids,
    "metaAndAssetCtxs": ReplayAPIManager._handle_meta_and_asset_ctxs,
    "meta": ReplayAPIManager._handle_meta,
    "candleSnapshot": ReplayAPIManager._handle_candle_snapshot,
    "l2Book": ReplayAPIManager._handle_l2_book,
    "fundingHistory": ReplayAPIManager._handle_funding_history,
    "userFills": ReplayAPIManager._handle_user_fills,
    "clearinghouseState": ReplayAPIManager._handle_clearinghouse_state,
    "userState": ReplayAPIManager._handle_user_state,
    "recentTrades": ReplayAPIManager._handle_recent_trades,
}


# ---- Singleton swap helpers ---------------------------------------

def install_replay_manager(shim: ReplayAPIManager) -> None:
    """Install `shim` as the global APIManager singleton.

    Production code does `from src.core.api_manager import get_manager` and
    captures the result. This swaps the cached `_manager` so subsequent
    `get_manager()` calls return our shim. Existing references on long-lived
    objects survive (they're already pointing at whatever they grabbed at
    init time), so the harness must run this BEFORE building subsystems.
    """
    import src.core.api_manager as am
    am._manager = shim    # bypass the lazy init, force the swap


def uninstall_replay_manager() -> None:
    """Restore lazy-init so the next get_manager() call rebuilds a real APIManager."""
    import src.core.api_manager as am
    am._manager = None


# ---- Inert stand-ins for APIManager subobjects --------------------

class _NullWebSocket:
    """Mirrors HyperliquidWebSocket's surface enough to not crash callers."""
    def is_connected(self) -> bool: return False
    def mids_are_fresh(self, max_age_s: float = 2.0) -> bool: return False
    def get_mid(self, coin: str): return None
    def get_all_mids(self) -> Dict[str, float]: return {}
    def subscribe_coin(self, coin: str) -> None: return None
    def start(self) -> None: return None
    def stop(self) -> None: return None
    def get_stats(self) -> Dict[str, Any]: return {"mode": "replay-disabled"}
    @property
    def mids(self) -> Dict[str, float]: return {}


class _NullBucket:
    def acquire(self, priority: Any = None, timeout: float = 60.0) -> bool: return True
    def report_429(self) -> None: return None
    def report_success(self) -> None: return None
    def get_stats(self) -> Dict[str, Any]: return {"mode": "replay-disabled"}


class _NullCache:
    def get(self, key: str): return None
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None: return None
    def invalidate(self, key: str) -> None: return None
    def clear(self) -> None: return None
    def get_stats(self) -> Dict[str, Any]: return {"mode": "replay-disabled"}
