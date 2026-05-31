#!/usr/bin/env python
"""Faithful P&L analysis for a replay run.

The ad-hoc breakdown we used during development counted **closed trades only**,
which made every run look like a ~100% win rate: winners hit take-profit and
close, while positions that moved against entry sit OPEN until the window ends
and were silently excluded. That is a survivorship artifact, not performance.

This analyzer marks every still-open position to market -- using the candle
cache close at the replay's as-of time, NOT the (often stale) ``last_path_price``
metadata -- and reports realized + unrealized + combined P&L and a TRUE win rate
that includes the marked-open book, split by source x side.

P&L convention matches the paper engine: ``size * dPrice * leverage`` (signed by
side). Closed ``pnl`` is read straight from the row (already net of fees); the
marked unrealized leg is gross (no exit fee paid yet), which is the honest
mark-to-market.

Usage:
    python scripts/analyze_replay_pnl.py --replay-db data/replay_local_6w.db
    python scripts/analyze_replay_pnl.py --replay-db <db> --cache-db <cache> \
        --as-of-ms <window_end_ms> --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def _parse_iso_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _source_of(metadata: Optional[str], strategy_id) -> str:
    """Canonical source bucket: 'copy_trade' or 'strategy' (fallbacks included)."""
    if metadata:
        try:
            m = json.loads(metadata)
            if m.get("is_copy_trade") or str(m.get("type", "")).startswith("copy"):
                return "copy_trade"
            src = m.get("source")
            if src:
                return "copy_trade" if str(src).startswith("copy") else str(src)
        except (ValueError, TypeError):
            pass
    return "strategy" if strategy_id is not None else "unknown"


def _cache_columns(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Return (ts_col, close_col) names, tolerating schema variants."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(candles)")}
    ts = next((c for c in ("timestamp_ms", "timestamp", "T", "t") if c in cols), None)
    close = next((c for c in ("close", "c") if c in cols), None)
    if not ts or not close:
        raise ValueError(f"candle cache 'candles' table missing ts/close columns; has {cols}")
    return ts, close


def _mark_price(cache: sqlite3.Connection, ts_col: str, close_col: str,
                coin: str, as_of_ms: int) -> Optional[float]:
    """Latest candle close for `coin` at or before as_of_ms (no lookahead)."""
    row = cache.execute(
        f"SELECT {close_col} FROM candles WHERE coin=? AND {ts_col}<=? "
        f"ORDER BY {ts_col} DESC LIMIT 1",
        (coin, as_of_ms),
    ).fetchone()
    if row is None:
        # fall back to the earliest available (position opened before cache start)
        row = cache.execute(
            f"SELECT {close_col} FROM candles WHERE coin=? ORDER BY {ts_col} ASC LIMIT 1",
            (coin,),
        ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def analyze(replay_db: str, cache_db: str, as_of_ms: Optional[int] = None) -> Dict:
    rconn = sqlite3.connect(f"file:{replay_db}?mode=ro", uri=True)
    rconn.row_factory = sqlite3.Row
    cache = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)
    try:
        ts_col, close_col = _cache_columns(cache)
        rows = list(rconn.execute(
            "SELECT id, strategy_id, coin, side, entry_price, exit_price, size, "
            "leverage, pnl, status, opened_at, closed_at, metadata FROM paper_trades"
        ))

        # Infer as-of (window end) from the latest trade event if not supplied.
        if as_of_ms is None:
            stamps = []
            for r in rows:
                for col in ("opened_at", "closed_at"):
                    ms = _parse_iso_ms(r[col])
                    if ms is not None:
                        stamps.append(ms)
            as_of_ms = max(stamps) if stamps else None

        buckets: Dict[Tuple[str, str], Dict] = {}
        unmarked: List[int] = []
        realized_total = 0.0
        unrealized_total = 0.0
        closed_wins = closed_n = 0
        open_wins = open_n = 0

        for r in rows:
            source = _source_of(r["metadata"], r["strategy_id"])
            side = (r["side"] or "").lower()
            key = (source, side)
            b = buckets.setdefault(key, dict(
                closed_n=0, open_n=0, realized=0.0, unrealized=0.0,
                wins=0, losses=0))
            lev = float(r["leverage"] or 1.0)
            size = float(r["size"] or 0.0)
            entry = float(r["entry_price"] or 0.0)
            sign = 1.0 if side == "long" else -1.0

            if (r["status"] or "").lower() == "closed":
                pnl = float(r["pnl"] or 0.0)
                realized_total += pnl
                b["realized"] += pnl
                b["closed_n"] += 1
                closed_n += 1
                if pnl > 0:
                    closed_wins += 1
                    b["wins"] += 1
                elif pnl < 0:
                    b["losses"] += 1
            else:  # open -> mark to market
                mark = _mark_price(cache, ts_col, close_col, r["coin"], as_of_ms) if as_of_ms else None
                if mark is None or entry <= 0:
                    unmarked.append(int(r["id"]))
                    b["open_n"] += 1
                    open_n += 1
                    continue
                upnl = size * (mark - entry) * sign * lev
                unrealized_total += upnl
                b["unrealized"] += upnl
                b["open_n"] += 1
                open_n += 1
                if upnl > 0:
                    open_wins += 1
                    b["wins"] += 1
                elif upnl < 0:
                    b["losses"] += 1

        graded = closed_n + (open_n - len(unmarked))
        total_wins = closed_wins + open_wins
        return {
            "replay_db": replay_db,
            "as_of_ms": as_of_ms,
            "as_of_iso": datetime.fromtimestamp(as_of_ms / 1000, timezone.utc).isoformat() if as_of_ms else None,
            "closed_n": closed_n,
            "open_n": open_n,
            "unmarked_open_ids": unmarked,
            "realized_pnl": round(realized_total, 2),
            "unrealized_pnl": round(unrealized_total, 2),
            "combined_pnl": round(realized_total + unrealized_total, 2),
            "closed_win_rate": round(closed_wins / closed_n, 4) if closed_n else None,
            "true_win_rate": round(total_wins / graded, 4) if graded else None,
            "true_wins": total_wins,
            "true_graded": graded,
            "buckets": {f"{k[0]}|{k[1]}": {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                                            for kk, vv in v.items()}
                        for k, v in sorted(buckets.items())},
        }
    finally:
        rconn.close()
        cache.close()


def format_report(a: Dict) -> str:
    lines = [
        f"REPLAY P&L ANALYSIS  db={a['replay_db']}",
        f"  as-of (mark time): {a['as_of_iso']}",
        f"  closed:   n={a['closed_n']:<3} realized={a['realized_pnl']:+.2f}",
        f"  open:     n={a['open_n']:<3} unrealized(marked)={a['unrealized_pnl']:+.2f}"
        + (f"  [unmarked ids: {a['unmarked_open_ids']}]" if a["unmarked_open_ids"] else ""),
        f"  COMBINED: {a['combined_pnl']:+.2f}",
        f"  win rate: closed-only={a['closed_win_rate']}  "
        f"TRUE(incl marked-open)={a['true_win_rate']} ({a['true_wins']}/{a['true_graded']})",
        "  by source x side (realized + unrealized):",
    ]
    for k, v in a["buckets"].items():
        tot = round(v["realized"] + v["unrealized"], 2)
        lines.append(
            f"    {k:<22} closed={v['closed_n']} open={v['open_n']}  "
            f"realized={v['realized']:+.2f} unreal={v['unrealized']:+.2f} "
            f"total={tot:+.2f}  W/L={v['wins']}/{v['losses']}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Faithful mark-to-market P&L for a replay run")
    p.add_argument("--replay-db", required=True, help="path to replay_*.db")
    p.add_argument("--cache-db", default="data/candle_cache.db", help="candle cache for marking open trades")
    p.add_argument("--as-of-ms", type=int, default=None,
                   help="mark open trades to this epoch-ms (default: latest trade timestamp in the DB)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of the text report")
    args = p.parse_args(argv)

    a = analyze(args.replay_db, args.cache_db, args.as_of_ms)
    if args.json:
        print(json.dumps(a, indent=2))
    else:
        print(format_report(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
