"""Historical trader-position oracle for the replay harness.

Copy-trade signals are driven by what the *tracked source traders* hold, which
the bot reads via ``hl.get_user_state(addr)`` (clearinghouseState).  In replay
that call is sandboxed and returned empty, so copy trades never fired in
backtests -- yet copy-trade longs are the one bucket with a positive live edge,
so excluding them makes a backtest unrepresentative.

This oracle reconstructs each tracked trader's open positions *as of a replay
timestamp* from their historical ``wallet_fills``, so the clearinghouse shim
can serve real source positions.  As the replay clock advances, fills
accumulate and positions appear/grow/close -- which is exactly the position
*delta* copy_trader keys off when deciding to mirror.

Reconstruction: per (address, coin), net the signed fill sizes up to ``ts_ms``
(open-long/close-short add; open-short/close-long subtract).  A non-zero net is
an open position whose side follows the sign and whose entry price is the
size-weighted average of the fills on the dominant side.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _signed_size(direction: str, side: str, size: float) -> float:
    """Map a fill to a signed position delta (+ adds long exposure)."""
    d = (direction or "").strip().lower()
    if "open long" in d or "close short" in d:
        return abs(size)
    if "open short" in d or "close long" in d:
        return -abs(size)
    # Fallback to raw side: HL uses B=bid/buy (+), A=ask/sell (-).
    s = (side or "").strip().lower()
    if s in ("b", "buy", "bid", "long"):
        return abs(size)
    if s in ("a", "sell", "ask", "short"):
        return -abs(size)
    return 0.0


class TraderPositionOracle:
    """In-memory reconstruction of tracked-trader positions at a timestamp."""

    def __init__(self, fills_by_addr: Dict[str, List[Dict[str, Any]]]):
        # Each address -> fills sorted ascending by time_ms.
        self._fills: Dict[str, List[Dict[str, Any]]] = {
            addr.lower(): sorted(rows, key=lambda r: int(r.get("time_ms") or 0))
            for addr, rows in fills_by_addr.items()
        }

    # ── loading ─────────────────────────────────────────────────

    @classmethod
    def from_db(cls, db_path: str, addresses: Optional[List[str]] = None) -> "TraderPositionOracle":
        """Load wallet_fills (optionally restricted to ``addresses``) grouped
        by wallet.  Returns an empty oracle on any failure (replay degrades to
        the old empty-clearinghouse behavior)."""
        import sqlite3

        by_addr: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT wallet_address, coin, side, original_price, size, "
                "time_ms, direction FROM wallet_fills"
            )
            wanted = {a.lower() for a in addresses} if addresses else None
            for r in cur:
                addr = str(r["wallet_address"] or "").lower()
                if not addr or (wanted is not None and addr not in wanted):
                    continue
                by_addr[addr].append({
                    "coin": str(r["coin"] or "").upper(),
                    "side": r["side"],
                    "original_price": float(r["original_price"] or 0.0),
                    "size": float(r["size"] or 0.0),
                    "time_ms": int(r["time_ms"] or 0),
                    "direction": r["direction"],
                })
            conn.close()
        except Exception:
            return cls({})
        return cls(dict(by_addr))

    # ── queries ─────────────────────────────────────────────────

    def addresses(self) -> List[str]:
        return list(self._fills.keys())

    def _net_by_coin(self, address: str, ts_ms: int) -> Dict[str, Tuple[float, float]]:
        """coin -> (net_size, size_weighted_entry_px) using fills <= ts_ms."""
        net: Dict[str, float] = defaultdict(float)
        px_num: Dict[str, float] = defaultdict(float)  # sum(|delta|*px) on dominant side
        px_den: Dict[str, float] = defaultdict(float)
        for f in self._fills.get(address.lower(), []):
            if f["time_ms"] > ts_ms:
                break  # fills are time-sorted
            delta = _signed_size(f["direction"], f["side"], f["size"])
            if delta == 0.0:
                continue
            coin = f["coin"]
            prev = net[coin]
            net[coin] = prev + delta
            # Track entry px only while accumulating in the current direction.
            px = f["original_price"]
            if px > 0:
                if prev == 0 or (prev > 0) == (delta > 0):
                    px_num[coin] += abs(delta) * px
                    px_den[coin] += abs(delta)
                elif (net[coin] > 0) != (prev > 0):
                    # flipped sides -> reset entry basis to this fill
                    px_num[coin] = abs(net[coin]) * px
                    px_den[coin] = abs(net[coin])
        out: Dict[str, Tuple[float, float]] = {}
        for coin, sz in net.items():
            if abs(sz) < 1e-9:
                continue
            entry = (px_num[coin] / px_den[coin]) if px_den[coin] > 0 else 0.0
            out[coin] = (sz, entry)
        return out

    def clearinghouse_state(self, address: str, ts_ms: int, *, now_ms: Optional[int] = None) -> Dict[str, Any]:
        """HL clearinghouseState-shaped dict of the trader's positions at ts_ms."""
        positions = self._net_by_coin(address, ts_ms)
        asset_positions: List[Dict[str, Any]] = []
        for coin, (szi, entry) in positions.items():
            asset_positions.append({
                "position": {
                    "coin": coin,
                    "szi": f"{szi}",
                    "entryPx": f"{entry}" if entry else None,
                    "leverage": {"type": "cross", "value": 1},
                    "unrealizedPnl": "0",
                    "positionValue": f"{abs(szi) * entry}" if entry else "0",
                },
                "type": "oneWay",
            })
        return {
            "marginSummary": {"accountValue": "0", "totalRawUsd": "0", "totalNtlPos": "0"},
            "crossMarginSummary": {"accountValue": "0"},
            "assetPositions": asset_positions,
            "withdrawable": "0",
            "time": now_ms if now_ms is not None else ts_ms,
        }
