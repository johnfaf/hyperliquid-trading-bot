"""Read-only inputs for the trading overview and settings pages."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sqlite3

import config
from src.trading.alpaca_paper import (
    DAILY_LOSS_USD, DEFAULT_SYMBOL, MAX_ENTRY_USD, MAX_QUOTE_AGE_SECONDS,
    MAX_QUOTE_SPREAD_FRACTION, REFERENCE_BALANCE_USD,
)
from src.ui.v2.state import get_components

log = logging.getLogger(__name__)

STOCK_LABELS = {
    "waiting_for_paper_credentials": ("Paper keys needed", "warn"),
    "outside_entry_or_exit_window": ("US session closed", "neutral"),
    "no_order": ("Monitoring", "good"),
    "shadow_intent": ("Shadow signal recorded", "good"),
    "intent_sent": ("Paper order sent", "good"),
    "order_pending": ("Awaiting paper fill", "warn"),
    "submission_unknown": ("Order status unresolved", "bad"),
    "broker_position_mismatch": ("Position mismatch", "bad"),
    "unmanaged_or_unconfirmed_fill": ("Unmatched broker fill", "bad"),
    "unmanaged_open_order": ("Unexpected open order", "bad"),
    "cycle_failed": ("Last cycle failed", "bad"),
    "quote_not_usable": ("Quote unavailable", "warn"),
    "not_enough_completed_bars": ("Waiting for market history", "warn"),
    "insufficient_cash": ("Not enough paper cash", "warn"),
    "broker_trading_blocked": ("Broker paused", "bad"),
    "entries_paused": ("Entries paused", "warn"),
    "daily_limit_reached": ("Daily loss limit reached", "bad"),
    "order_rejected": ("Paper order rejected", "bad"),
    "drill_entry_not_ready": ("Waiting for fresh shadow evidence", "warn"),
    "drill_position_unverified": ("Test position not verified", "warn"),
    "drill_position_open": ("Supervised paper test open", "warn"),
}


def read_sqlite(path: Path):
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    return conn


def _datestamp(raw):
    try:
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def host_status(path: Path, now: datetime) -> dict:
    try:
        with path.open(encoding="utf-8") as source:
            data = json.loads(source.read(65536))
        observed = _datestamp(data.get("observed_at"))
        if observed is None or not 0 <= (now - observed).total_seconds() <= 180:
            return {}
        return data
    except (OSError, ValueError, AttributeError):
        return {}


def paper_book(path: Path, start: datetime) -> dict:
    result = {"cash": None, "lifetime_pnl": None, "open": None, "closes": [], "decisions": None}
    try:
        with closing(read_sqlite(path)) as conn:
            account = conn.execute("SELECT balance, total_pnl FROM paper_account WHERE id = 1").fetchone()
            if account:
                result.update(cash=account["balance"], lifetime_pnl=account["total_pnl"])
            result["open"] = conn.execute("SELECT count(*) FROM paper_trades WHERE status = 'open'").fetchone()[0]
            rows = conn.execute("""
                SELECT coin, side, pnl, closed_at, metadata FROM paper_trades
                WHERE status = 'closed' AND closed_at >= ? ORDER BY closed_at DESC, id DESC LIMIT 1001
                """, (start.isoformat(),)).fetchall()
            result["complete_window"] = len(rows) <= 1000
            for row in rows[:1000]:
                try:
                    meta = json.loads(row["metadata"] or "{}")
                    if not isinstance(meta, dict):
                        meta = {}
                except (ValueError, TypeError):
                    meta = {}
                if meta.get("tainted") or meta.get("synthetic_reconciliation") or (
                    "live_reconciled_closed" in (meta.get("close_reason"), meta.get("reconciliation_reason"))
                ):
                    continue
                if row["pnl"] is not None:
                    result["closes"].append({
                        "coin": row["coin"], "side": row["side"], "pnl": row["pnl"], "timestamp": row["closed_at"],
                        "source": meta.get("strategy_type") or meta.get("source") or "Unattributed",
                        "reason": meta.get("close_reason") or "Unrecorded",
                    })
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='decision_snapshots'").fetchone():
                result["decisions"] = dict(conn.execute("""
                    SELECT count(*) AS total, count(paper_trade_id) AS linked,
                    count(CASE WHEN final_status LIKE '%rejected' OR firewall_decision='rejected' THEN 1 END) AS rejected
                    FROM decision_snapshots WHERE created_at >= ?
                    """, (start.isoformat(),)).fetchone())
    except sqlite3.Error as exc:
        log.debug("Trading desk ledger unavailable: %s", exc)
    return result


def chart_for(closes: list[dict]) -> dict | None:
    if not closes:
        return None
    values = [0.0]
    for trade in reversed(closes):
        values.append(values[-1] + trade["pnl"])
    lo, hi = min(values), max(values)
    padding = max((hi - lo) * 0.12, 0.01)
    lo, hi = lo - padding, hi + padding
    xy = [
        (round(i * 600 / (len(values) - 1), 1), round(162 - (v - lo) / (hi - lo) * 154, 1))
        for i, v in enumerate(values)
    ]
    points = " ".join(f"{x},{y}" for x, y in xy)
    return {"points": points, "fill": f"0,170 {points} 600,170",
            "zero": round(162 - (0 - lo) / (hi - lo) * 154, 1), "last_y": xy[-1][1]}


def snapshot(hours: int = 24) -> dict:
    now = datetime.now(timezone.utc)
    path = Path(config.DB_PATH)
    host = host_status(path.with_name("dashboard_ops.json"), now)
    book = paper_book(path, now - timedelta(hours=hours))
    closes = book.pop("closes")
    measured = bool(book.get("complete_window"))
    book.update(
        closed_count=len(closes) if measured else None,
        win_rate=100 * sum(t["pnl"] > 0 for t in closes) / len(closes) if measured and closes else None,
        net=round(sum(t["pnl"] for t in closes), 2) if measured else None,
        chart=chart_for(closes) if measured else None,
        recent=closes[:10],
    )
    stock = host.get("stock") or {}
    seen = _datestamp(stock.get("observed_at"))
    stock_fresh = seen is not None and 0 <= (now - seen).total_seconds() <= 900
    stock_job = (host.get("jobs") or {}).get("stock") or {}
    if not host:
        stock_label = ("Worker status unavailable", "neutral")
    elif stock_job.get("result") not in {None, "success"}:
        stock_label = ("Last cycle failed", "bad")
    elif not stock_job.get("timer_active"):
        stock_label = ("Schedule stopped", "warn")
    elif not stock_fresh:
        stock_label = ("Waiting for a fresh cycle", "warn")
    else:
        stock_label = STOCK_LABELS.get(stock.get("state"), ("Waiting for a usable signal", "warn"))
    trader = get_components().live_trader
    requested = bool(getattr(config, "LIVE_TRADING_ENABLED", False) or getattr(trader, "live_requested", False))
    signer = bool(getattr(trader, "signer", None))
    unresolved = None
    address, info_url = getattr(trader, "public_address", None), getattr(trader, "info_url", None)
    if address and info_url and signer:
        try:
            with closing(read_sqlite(path.with_suffix(".orders.sqlite3"))) as conn:
                unresolved = conn.execute(
                    "SELECT count(*) FROM orders WHERE account=? AND status IN ('submitting','unknown','cancel_pending')",
                    (f"hyperliquid:{info_url}:{address.lower()}",),
                ).fetchone()[0]
        except sqlite3.Error:
            pass
    return {
        "updated": now, "hours": hours, "paper": book, "host": host,
        "stock_status": stock_label, "stock_mode": stock.get("mode") if stock_fresh else None,
        "stock_last_run": seen,
        "crypto": {"requested": requested, "signer": signer, "paper_only": not requested and not signer,
                   "unresolved": unresolved, "paused": bool(getattr(trader, "kill_switch_active", False))},
        "settings": {
            "paper_risk_pct": config.PAPER_MAX_PLANNED_STOP_LOSS_PCT * 100,
            "paper_start_usd": config.PAPER_TRADING_INITIAL_BALANCE,
            "live_max_order_usd": getattr(trader, "max_order_usd", config.LIVE_MAX_ORDER_USD),
            "live_daily_limit_usd": getattr(trader, "max_daily_loss", config.LIVE_MAX_DAILY_LOSS_USD),
            "stock_symbol": DEFAULT_SYMBOL, "stock_max_order_usd": MAX_ENTRY_USD,
            "stock_daily_limit_usd": DAILY_LOSS_USD, "stock_reference_usd": REFERENCE_BALANCE_USD,
            "stock_max_quote_age": MAX_QUOTE_AGE_SECONDS, "stock_max_quote_spread_pct": MAX_QUOTE_SPREAD_FRACTION * 100,
        },
    }
