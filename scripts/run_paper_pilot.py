#!/usr/bin/env python3
"""Single-strategy BTC paper pilot. Reads public market data; no live order code."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOUR = 3_600_000
LOOKBACK = 24
COIN = "BTC"
STARTING_CASH = 1000.0
MAX_NOTIONAL = 75.0
STOP = 0.02
TAKE_PROFIT = 0.04
STRATEGY = "btc_hourly_sma24_cross_long_v1"
RISK = {"stop_roe_pct": STOP, "time_limit_hours": 24, "breakeven_at_r": 3.0, "trailing_enabled": False}
SPEC = {
    "strategy": STRATEGY, "starting_cash": STARTING_CASH, "max_notional": MAX_NOTIONAL,
    "coin": COIN, "candle_minutes": 60, "lookback": LOOKBACK, "entry_window_minutes": 5, "stop": STOP,
    "take_profit": TAKE_PROFIT, "max_hold_hours": 24, "poll_seconds": 60, "maker_fee_bps": 1.5,
    "taker_fee_bps": 4.5, "slippage_bps_each_way": 5.0, "max_quote_age_seconds": 30, "leverage": 1,
}
STATE_KEY = "paper_pilot_spec"
logger = logging.getLogger(__name__)


def upward_cross(candles, closed_by: int) -> bool | None:
    """True/False for contiguous, fully closed candles, otherwise unknown."""
    if not isinstance(candles, list):
        return None
    first = closed_by - (LOOKBACK + 1) * HOUR
    prices = {}
    for candle in candles:
        try:
            start = int(candle["t"])
            if not first <= start < closed_by:
                continue
            if candle["s"] != COIN or candle["i"] != "1h" or int(candle["T"]) != start + HOUR - 1:
                return None
            price = float(candle["c"])
            if start in prices or not math.isfinite(price) or price <= 0:
                return None
            prices[start] = price
        except (KeyError, TypeError, OverflowError, ValueError):
            return None
    try:
        series = [prices[start] for start in range(first, closed_by, HOUR)]
    except KeyError:
        return None
    return series[-2] <= math.fsum(series[:-1]) / LOOKBACK and series[-1] > math.fsum(series[1:]) / LOOKBACK


def verified_mid(book, now_ms: int) -> float | None:
    try:
        age = now_ms - int(book["time"])
        bid, ask = (float(level[0]["px"]) for level in book["levels"])
        if (book["coin"] == COIN and -2_000 <= age <= SPEC["max_quote_age_seconds"] * 1000
                and 0 < bid <= ask and math.isfinite(ask)):
            return bid + (ask - bid) / 2
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        pass
    return None


def validate_environment():
    raw_path = os.environ.get("HL_BOT_DB")
    if not raw_path or not Path(raw_path).is_absolute() or Path(raw_path).name != "audit-paper-1000.db":
        raise ValueError("pilot requires an explicit absolute path to its own audit-paper-1000.db")
    if os.environ.get("DB_BACKEND") != "sqlite":
        raise ValueError("pilot requires explicit DB_BACKEND=sqlite")
    for name in ("HL_AGENT_PRIVATE_KEY", "HL_PRIVATE_KEY", "HL_PUBLIC_ADDRESS", "LIGHTER_PRIVATE_KEY", "DATABASE_URL"):
        if os.environ.get(name):
            raise ValueError(f"pilot must not inherit {name}")
    if os.environ.get("SECRET_MANAGER_PROVIDER", "none") != "none":
        raise ValueError("pilot must not load a secret manager")
    path = Path(raw_path).resolve()
    if path != Path(raw_path):
        raise ValueError("pilot database path must be canonical and not a symlink")
    return path


def initialize(path, db):
    if path.exists():
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT value FROM bot_state WHERE key=?", (STATE_KEY,)).fetchone()
        if not row or db.get_bot_state(STATE_KEY) != SPEC or not db.get_paper_account():
            raise ValueError("will not open an unrelated database or silently change the frozen study")
    else:
        db.init_db()
        db.init_paper_account(STARTING_CASH)
        if not db.set_bot_state(STATE_KEY, SPEC):
            raise RuntimeError("could not persist the study spec")
    with db.get_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_pilot_marks (observed_at TEXT PRIMARY KEY, "
            "quote REAL NOT NULL, equity REAL NOT NULL, passive_reference REAL NOT NULL)"
        )


def _record_event(db, action, **details):
    db.audit_log(f"paper_pilot_{action}", coin=COIN, source=STRATEGY, details=details)


def _record_mark(db, observed_at: str, mid: float, cash: float, trades: list[dict]):
    equity = cash
    for trade in trades:
        meta = metadata(trade)
        equity += (mid - trade["entry_price"]) * trade["size"] - meta.get("funding_accrued", 0)
        equity -= trade["entry_price"] * trade["size"] * SPEC["taker_fee_bps"] / 10_000
    with db.get_connection() as conn:
        first = conn.execute("SELECT quote FROM paper_pilot_marks ORDER BY observed_at LIMIT 1").fetchone()
        ref = STARTING_CASH + MAX_NOTIONAL * (mid / first["quote"] - 1) if first else STARTING_CASH
        conn.execute(
            "INSERT INTO paper_pilot_marks (observed_at, quote, equity, passive_reference) VALUES (?, ?, ?, ?)",
            (observed_at, mid, equity, ref),
        )
    logger.info("Paper mark: equity=%.2f open_positions=%d", equity, len(trades))


def metadata(trade):
    data = trade["metadata"]
    return json.loads(data) if isinstance(data, str) else data


def tick(trader, db, market):
    book = market.get_l2_book(COIN)
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    mid = verified_mid(book, now_ms)
    if mid is None:
        _record_event(db, "quote_unavailable")
        logger.warning("No fresh BTC quote; no valuation, entries or exits")
        return False
    open_trades = db.get_open_paper_trades()
    if len(open_trades) > 1 or any(
        t["coin"] != COIN or t["side"] != "long" or t["leverage"] != 1 or metadata(t).get("source_key") != STRATEGY
        for t in open_trades
    ):
        raise RuntimeError("unexpected paper position; stop and investigate")
    trader.check_open_positions(mids={COIN: mid})
    trader.drain_closed_events()
    account = db.get_paper_account()
    open_trades = db.get_open_paper_trades()
    boundary = now_ms - now_ms % HOUR
    if now_ms - boundary < SPEC["entry_window_minutes"] * 60_000 and db.get_bot_state("paper_pilot_last_bar") != boundary:
        if not open_trades:
            candles = market.get_candles(COIN, "1h", boundary - (LOOKBACK + 1) * HOUR, boundary - 1)
            enter = upward_cross(candles, boundary)
            if enter is None:
                _record_event(db, "candles_unavailable", boundary=boundary)
                _record_mark(db, now.isoformat(), mid, account["balance"], open_trades)
                return True
            if enter:
                book = market.get_l2_book(COIN)
                observed = datetime.now(timezone.utc)
                new_ms = int(observed.timestamp() * 1000)
                entry_mid = verified_mid(book, new_ms)
                if entry_mid is None or new_ms - boundary >= SPEC["entry_window_minutes"] * 60_000:
                    _record_event(db, "entry_skipped", reason="quote_stale_or_signal_window_elapsed", boundary=boundary)
                elif account["balance"] < MAX_NOTIONAL:
                    _record_event(db, "entry_skipped", reason="paper_cash_below_fixed_allocation", boundary=boundary)
                else:
                    fill = entry_mid * (1 + SPEC["slippage_bps_each_way"] / 10_000)
                    stop, target = fill * (1 - STOP), fill * (1 + TAKE_PROFIT)
                    trade_id = db.open_paper_trade(
                        None, COIN, "long", fill, MAX_NOTIONAL / fill, leverage=1, stop_loss=stop, take_profit=target,
                        metadata={
                            "strategy_type": STRATEGY, "source_key": STRATEGY, "execution_role": "taker",
                            "intended_entry_price": entry_mid, "initial_stop_loss": stop, "risk_policy": RISK,
                            "taker_fee_bps": SPEC["taker_fee_bps"], "candle_boundary_ms": boundary,
                        },
                        idempotency_key=f"{STRATEGY}:{boundary}",
                    )
                    _record_event(db, "entry_simulated", trade_id=trade_id)
                    mid = entry_mid
                    now = observed
        if not db.set_bot_state("paper_pilot_last_bar", boundary):
            raise RuntimeError("could not persist the processed hour")
        open_trades = db.get_open_paper_trades()
    _record_mark(db, now.isoformat(), mid, account["balance"], open_trades)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit nonzero if no fresh quote")
    args = parser.parse_args()
    path = validate_environment()
    import config
    from src.data import database as db
    from src.data import hyperliquid_client as market
    from src.trading.paper_trader import PaperTrader

    if (config.LIVE_TRADING_ENABLED or config.LIGHTER_LIVE_TRADING_ENABLED
            or config.PAPER_TRADING_INITIAL_BALANCE != STARTING_CASH or config.PAPER_TRADING_MAX_LEVERAGE != 1):
        raise ValueError("pilot requires both live venues off, $1,000 simulated capital and a 1x cap")
    config.PAPER_TRADING_TAKER_FEE_BPS = SPEC["taker_fee_bps"]
    config.PAPER_TRADING_MAKER_FEE_BPS = SPEC["maker_fee_bps"]
    config.PAPER_TRADING_SLIPPAGE_MIN_BPS = config.PAPER_TRADING_SLIPPAGE_MAX_BPS = SPEC["slippage_bps_each_way"]
    config.PAPER_TRADING_FUNDING_ENABLED = True
    config.SL_MIN_HOLD_SECONDS = 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    initialize(path, db)
    trader = PaperTrader()
    while True:
        ok = tick(trader, db, market)
        if args.once:
            raise SystemExit(0 if ok else 1)
        time.sleep(SPEC["poll_seconds"])


if __name__ == "__main__":
    main()
