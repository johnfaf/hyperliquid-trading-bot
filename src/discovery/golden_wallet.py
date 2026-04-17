"""
Golden Wallet Pipeline
======================
1. Download 3 months of fills for every "human-like" wallet (bot_score < 3)
2. Apply realistic execution penalties (+100ms delay, -0.045% slippage+fees)
3. Build equity curves and tag wallets whose penalised curve is still rising
4. Persist results so the backtest dashboard can read them

Hyperliquid API returns max 2000 fills per call.  We page backwards with
startTime to cover the full 90-day window.

Incremental mode: on subsequent evaluations only new fills are fetched.
The wallet's last_fill_sync_time is stored in the DB; when non-zero the
download is limited to fills after that timestamp and merged with what's
already stored.  This reduces API calls by ~75% for already-evaluated wallets.
"""
import logging
import random
import sqlite3
import time
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import database as db
from src.data import hyperliquid_client as hl

logger = logging.getLogger("golden_wallet")

# ─── Constants ───────────────────────────────────────────────────
LOOKBACK_DAYS = 90
EXECUTION_DELAY_MS = 100        # +100 ms assumed latency
FEE_SLIPPAGE_BPS = 4.5          # 0.045% total (taker fee + slippage)
PENALTY_FACTOR = 1 - FEE_SLIPPAGE_BPS / 10_000  # ~0.99955
MIN_FILLS_FOR_EVAL = 30         # need at least 30 round-trips
MAX_FILLS_FOR_EVAL = 3000       # cap: anything above this is not human-like
GOLDEN_THRESHOLD = 0.0          # penalised equity must be net-positive
PAGE_SIZE = 2000                # HL max fills per request
REQUEST_SLEEP = 1.5             # rate-limit padding between pages (was 0.8 — caused 429s)
BETWEEN_WALLET_SLEEP = 2.5      # pause between evaluating different wallets

# Jittered backoff sequence (seconds) used when we hit rate limits.
# Fibonacci-inspired: avoids the thundering-herd effect of pure exponential.
# Jitter of ±10% is applied on top to spread simultaneous retries.
_RATE_LIMIT_BACKOFF_SEQ = [5, 8, 15, 30, 60, 90, 120]

# Superhuman detection: no real human trader sustains these numbers
# over 90 days.  Wallets hitting these thresholds are bots/vaults/arb.
MAX_HUMAN_WIN_RATE = float(os.environ.get("GOLDEN_WALLET_MAX_HUMAN_WIN_RATE", "92.0"))
MIN_HUMAN_DRAWDOWN = float(os.environ.get("GOLDEN_WALLET_MIN_HUMAN_DRAWDOWN", "0.5"))
SHARPE_RETURN_MIN = float(os.environ.get("GOLDEN_WALLET_SHARPE_RETURN_MIN", "-1.0"))
SHARPE_RETURN_MAX = float(os.environ.get("GOLDEN_WALLET_SHARPE_RETURN_MAX", "10.0"))


def _rate_limit_backoff(attempt: int, label: str = "") -> float:
    """
    Return backoff seconds for the given attempt number (1-based).
    Applies ±10% jitter to avoid thundering-herd retries.
    Logs the wait and returns the chosen duration.
    """
    idx = min(attempt - 1, len(_RATE_LIMIT_BACKOFF_SEQ) - 1)
    base = _RATE_LIMIT_BACKOFF_SEQ[idx]
    jitter = base * random.uniform(0.0, 0.10)
    duration = base + jitter
    logger.warning("Rate limited%s (attempt %d) — backing off %.1fs", label, attempt, duration)
    return duration


# ─── Data classes ────────────────────────────────────────────────
@dataclass
class PenalisedFill:
    """A single fill with execution-reality adjustments applied."""
    coin: str
    side: str           # "buy" / "sell"
    original_price: float
    penalised_price: float
    size: float
    time_ms: int        # original timestamp
    delayed_time_ms: int  # +100ms
    closed_pnl: float   # raw from exchange
    penalised_pnl: float # after slippage/fee deduction
    fee: float
    is_liquidation: bool
    direction: str       # "Open Long", "Close Long", etc.


@dataclass
class WalletReport:
    """Full backtest report for one wallet."""
    address: str
    bot_score: int
    total_fills: int
    fills_in_window: int
    raw_pnl: float
    penalised_pnl: float
    raw_equity_curve: List[float] = field(default_factory=list)
    penalised_equity_curve: List[float] = field(default_factory=list)
    equity_timestamps: List[int] = field(default_factory=list)
    max_drawdown_pct: float = 0.0
    penalised_max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    trades_per_day: float = 0.0
    is_golden: bool = False
    coins_traded: List[str] = field(default_factory=list)
    best_coin: str = ""
    worst_coin: str = ""
    avg_hold_time_hours: float = 0.0
    evaluated_at: str = ""


# ─── Core functions ──────────────────────────────────────────────

def download_fills_90d(address: str) -> List[Dict]:
    """
    Download up to 90 days of fills for a single address.
    Pages backwards using startTime to get past the 2000-fill cap.

    Uses LOW priority so live trading / position monitoring always
    takes precedence over historical backfill.

    Rate-limit backoff uses _rate_limit_backoff() which applies Fibonacci
    spacing + ±10% jitter to avoid thundering-herd retries.
    """
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
    all_fills = []
    oldest_seen = None
    consecutive_errors = 0

    for page in range(50):  # safety cap: 50 pages = 100k fills max
        try:
            if page == 0:
                fills = hl.get_user_fills(address)
            else:
                fills = hl.get_user_fills(address, start_time=cutoff_ms)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                backoff = _rate_limit_backoff(
                    consecutive_errors,
                    label=f" downloading fills for {address[:10]} page {page}",
                )
                time.sleep(backoff)
                if consecutive_errors >= len(_RATE_LIMIT_BACKOFF_SEQ):
                    logger.warning(
                        "Giving up on %s after %d rate limits", address[:10], consecutive_errors,
                    )
                    break
                continue
            else:
                logger.error("Error downloading fills for %s: %s", address[:10], e)
                break

        if not fills:
            break

        new_fills = [f for f in fills if f["time"] >= cutoff_ms]
        if not new_fills:
            break

        seen_hashes = {f["hash"] for f in all_fills}
        added = 0
        for f in new_fills:
            if f["hash"] not in seen_hashes:
                all_fills.append(f)
                seen_hashes.add(f["hash"])
                added += 1

        if added == 0:
            break

        times = [f["time"] for f in new_fills]
        min_time = min(times)
        if oldest_seen is not None and min_time >= oldest_seen:
            break
        oldest_seen = min_time

        if len(fills) < PAGE_SIZE:
            break

        logger.debug("  Page %d: +%d fills (total: %d, oldest: %d)", page + 1, added, len(all_fills), min_time)
        time.sleep(REQUEST_SLEEP)

    all_fills.sort(key=lambda f: f["time"])
    logger.debug(
        "Downloaded %d fills for %s... (span: %dd window)", len(all_fills), address[:10], LOOKBACK_DAYS,
    )
    return all_fills


def download_fills_incremental(address: str, last_sync_time_ms: int) -> List[Dict]:
    """
    Fetch only fills NEWER than last_sync_time_ms.

    On the first evaluation last_sync_time_ms is 0 and this falls back to
    download_fills_90d() automatically.  On subsequent evaluations only new
    fills need to be fetched (typically 1 API call instead of 2-3), reducing
    API load by ~75% for already-tracked wallets.

    Returns fills sorted oldest-first.
    """
    if last_sync_time_ms <= 0:
        return download_fills_90d(address)

    # CRIT-FIX C3: Hyperliquid's `start_time` filter semantics are
    # inconsistent across endpoints (some inclusive, some exclusive).  A fill
    # with `time_ms == last_sync_time_ms` was being silently skipped when the
    # API treated the bound as exclusive, creating gaps in wallet history.
    #
    # Fix: query with `start_time = last_sync_time_ms - 1` (ensures we never
    # skip a boundary fill on inclusive endpoints) and then defensively
    # filter out any fills we've already saved (`time <= last_sync_time_ms`)
    # so we cannot double-count on inclusive endpoints either.  The ±1 ms
    # window is cheap and makes the function correct for both API behaviours.
    query_start = max(0, int(last_sync_time_ms) - 1)

    consecutive_errors = 0
    try:
        fills = hl.get_user_fills(address, start_time=query_start)
        consecutive_errors = 0
    except Exception as e:
        consecutive_errors += 1
        err_str = str(e).lower()
        if "429" in err_str or "rate" in err_str:
            backoff = _rate_limit_backoff(consecutive_errors, label=f" incremental fetch {address[:10]}")
            time.sleep(backoff)
            try:
                fills = hl.get_user_fills(address, start_time=query_start)
            except Exception:
                logger.error("Incremental fetch failed for %s; falling back to full 90d", address[:10])
                return download_fills_90d(address)
        else:
            logger.error("Error in incremental fetch for %s: %s; falling back to full 90d", address[:10], e)
            return download_fills_90d(address)

    if not fills:
        return []

    # Drop any fill at or before the last synced timestamp — these are already
    # persisted; keeping them would double-count PnL on re-evaluation.
    try:
        fills = [f for f in fills if int(f.get("time", 0) or 0) > int(last_sync_time_ms)]
    except Exception:
        pass

    if not fills:
        return []

    fills.sort(key=lambda f: f["time"])
    logger.debug(
        "Incremental fetch: %d new fills for %s since %d",
        len(fills), address[:10], last_sync_time_ms,
    )
    return fills


def _history_cutoff_ms() -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)


def _fill_identity(fill: Dict) -> Tuple:
    return (
        str(fill.get("coin", "")),
        str(fill.get("side", "")),
        int(fill.get("time", fill.get("time_ms", 0)) or 0),
        float(fill.get("price", fill.get("original_price", 0.0)) or 0.0),
        float(fill.get("size", 0.0) or 0.0),
        float(fill.get("closed_pnl", 0.0) or 0.0),
        str(fill.get("direction", "")),
        bool(fill.get("is_liquidation", False)),
    )


def _stored_fill_to_raw(row: Dict) -> Dict:
    return {
        "coin": row.get("coin", ""),
        "side": row.get("side", ""),
        "price": float(row.get("original_price", 0.0) or 0.0),
        "size": float(row.get("size", 0.0) or 0.0),
        "time": int(row.get("time_ms", 0) or 0),
        "closed_pnl": float(row.get("closed_pnl", 0.0) or 0.0),
        "fee": float(row.get("fee", 0.0) or 0.0),
        "is_liquidation": bool(row.get("is_liquidation", False)),
        "direction": row.get("direction", ""),
    }


def _stored_fill_to_penalised(row: Dict) -> PenalisedFill:
    return PenalisedFill(
        coin=row.get("coin", ""),
        side=row.get("side", ""),
        original_price=float(row.get("original_price", 0.0) or 0.0),
        penalised_price=float(row.get("penalised_price", 0.0) or 0.0),
        size=float(row.get("size", 0.0) or 0.0),
        time_ms=int(row.get("time_ms", 0) or 0),
        delayed_time_ms=int(row.get("delayed_time_ms", 0) or 0),
        closed_pnl=float(row.get("closed_pnl", 0.0) or 0.0),
        penalised_pnl=float(row.get("penalised_pnl", 0.0) or 0.0),
        fee=float(row.get("fee", 0.0) or 0.0),
        is_liquidation=bool(row.get("is_liquidation", False)),
        direction=row.get("direction", ""),
    )


def _merge_fill_history(existing_fills: List[Dict], new_fills: List[Dict]) -> List[Dict]:
    cutoff_ms = _history_cutoff_ms()
    merged: Dict[Tuple, Dict] = {}
    for fill in existing_fills or []:
        if int(fill.get("time", fill.get("time_ms", 0)) or 0) >= cutoff_ms:
            merged[_fill_identity(fill)] = dict(fill)
    for fill in new_fills or []:
        if int(fill.get("time", fill.get("time_ms", 0)) or 0) >= cutoff_ms:
            merged[_fill_identity(fill)] = dict(fill)
    history = list(merged.values())
    history.sort(key=lambda f: int(f.get("time", f.get("time_ms", 0)) or 0))
    return history


def _merge_penalised_history(
    existing_fills: List[PenalisedFill],
    new_fills: List[PenalisedFill],
) -> List[PenalisedFill]:
    cutoff_ms = _history_cutoff_ms()
    merged: Dict[Tuple, PenalisedFill] = {}
    for fill in existing_fills or []:
        if int(fill.time_ms) >= cutoff_ms:
            merged[_fill_identity(fill.__dict__)] = fill
    for fill in new_fills or []:
        if int(fill.time_ms) >= cutoff_ms:
            merged[_fill_identity(fill.__dict__)] = fill
    history = list(merged.values())
    history.sort(key=lambda f: int(f.time_ms))
    return history


def apply_execution_penalties(fills: List[Dict]) -> List[PenalisedFill]:
    """
    Apply realistic execution penalties to each fill:
    - +100ms timestamp delay (you'd enter slightly later)
    - -0.045% price penalty (taker fees + slippage)

    For buys:  penalised_price = original * (1 + 0.00045)  — you pay more
    For sells: penalised_price = original * (1 - 0.00045)  — you receive less
    """
    penalised = []

    for f in fills:
        side = f["side"]  # "buy" or "sell"
        price = f["price"]
        size = f["size"]
        closed_pnl = f["closed_pnl"]

        # Delay
        delayed_time = f["time"] + EXECUTION_DELAY_MS

        # Price penalty
        if side == "buy":
            pen_price = price * (1 + FEE_SLIPPAGE_BPS / 10_000)
        else:
            pen_price = price * (1 - FEE_SLIPPAGE_BPS / 10_000)

        # Penalised PnL: if this fill closed a position, adjust the PnL
        # The penalty applies to BOTH entry and exit, so double the single-leg fee
        if closed_pnl != 0:
            # PnL hit = notional * 2 * fee_rate (entry + exit penalty)
            notional = price * size
            pnl_penalty = notional * 2 * (FEE_SLIPPAGE_BPS / 10_000)
            if closed_pnl > 0:
                pen_pnl = max(0, closed_pnl - pnl_penalty)
            else:
                pen_pnl = closed_pnl - pnl_penalty  # loss gets worse
        else:
            pen_pnl = 0.0

        penalised.append(PenalisedFill(
            coin=f["coin"],
            side=side,
            original_price=price,
            penalised_price=round(pen_price, 6),
            size=size,
            time_ms=f["time"],
            delayed_time_ms=delayed_time,
            closed_pnl=closed_pnl,
            penalised_pnl=round(pen_pnl, 6),
            fee=f.get("fee", 0),
            is_liquidation=f.get("is_liquidation", False),
            direction=f.get("direction", ""),
        ))

    return penalised


def build_equity_curve(penalised_fills: List[PenalisedFill],
                       initial_equity: float = 10_000.0
                       ) -> Tuple[List[float], List[float], List[int]]:
    """
    Build two equity curves from penalised fills:
    - raw_curve: using original closed_pnl
    - penalised_curve: using penalised_pnl

    Returns (raw_curve, penalised_curve, timestamps)
    """
    raw_eq = initial_equity
    pen_eq = initial_equity
    raw_curve = [raw_eq]
    pen_curve = [pen_eq]
    timestamps = [penalised_fills[0].time_ms if penalised_fills else 0]

    for fill in penalised_fills:
        if fill.closed_pnl != 0 or fill.penalised_pnl != 0:
            raw_eq += fill.closed_pnl
            pen_eq += fill.penalised_pnl
            raw_curve.append(raw_eq)
            pen_curve.append(pen_eq)
            timestamps.append(fill.time_ms)

    return raw_curve, pen_curve, timestamps


def compute_max_drawdown(curve: List[float]) -> float:
    """Compute maximum drawdown as a percentage."""
    if not curve or len(curve) < 2:
        return 0.0
    peak = curve[0]
    max_dd = 0.0
    for val in curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100, 2)


def compute_sharpe(daily_returns: List[float], periods_per_year: float = 365.0) -> float:
    """
    Annualised Sharpe ratio from a series of daily RETURNS (not raw PnL).

    Expects percentage returns (e.g. 0.02 for +2%), not dollar amounts.
    Values outside the configured [min, max] range are treated as invalid and dropped.

    Returns 0.0 if fewer than 5 valid data points are available.
    """
    import math

    valid = []
    dropped = 0
    for value in daily_returns:
        if not isinstance(value, (int, float)):
            dropped += 1
            continue
        if math.isnan(value) or math.isinf(value):
            dropped += 1
            continue
        if not (SHARPE_RETURN_MIN <= value <= SHARPE_RETURN_MAX):
            dropped += 1
            continue
        valid.append(value)

    if dropped:
        logger.info(
            "Sharpe filter dropped %d/%d daily returns outside configured bounds [%.2f, %.2f]",
            dropped,
            len(daily_returns),
            SHARPE_RETURN_MIN,
            SHARPE_RETURN_MAX,
        )

    if len(valid) < 5:
        return 0.0

    import statistics
    mean_r = statistics.mean(valid)
    std_r = statistics.stdev(valid)
    if std_r < 1e-10:
        return 0.0
    return round((mean_r / std_r) * (periods_per_year ** 0.5), 3)


def _equity_curve_to_daily_returns(equity_curve: List[float],
                                     timestamps: List[int]) -> List[float]:
    """
    Resample an intra-day equity curve into daily returns.
    Takes the last equity value for each calendar day and computes
    day-over-day percentage returns.
    """
    if len(equity_curve) < 2 or len(timestamps) < 2:
        return []

    # Get end-of-day equity for each calendar day
    daily_equity = {}
    for eq, ts in zip(equity_curve, timestamps):
        day = ts // (86400 * 1000)
        daily_equity[day] = eq  # last value for each day wins

    sorted_days = sorted(daily_equity.keys())
    if len(sorted_days) < 2:
        return []

    daily_returns = []
    for i in range(1, len(sorted_days)):
        prev_eq = daily_equity[sorted_days[i - 1]]
        curr_eq = daily_equity[sorted_days[i]]
        if prev_eq > 0:
            daily_returns.append((curr_eq - prev_eq) / prev_eq)

    return daily_returns


def compute_avg_hold_time_hours(fills: List[Dict]) -> float:
    """
    Estimate average hold time by tracking each coin's signed net position.

    A hold is only recorded when the running position returns to zero or flips
    through zero. This avoids over-pairing partial closes against multiple
    synthetic "open" fills.
    """
    if len(fills) < 2:
        return 0.0

    position_by_coin: Dict[str, float] = {}
    entry_ts_by_coin: Dict[str, int] = {}
    hold_times_ms: List[float] = []

    for fill in sorted(fills, key=lambda x: x.get("time", 0)):
        coin = str(fill.get("coin", "") or "")
        ts = int(fill.get("time", 0) or 0)
        size = abs(float(fill.get("size", 0) or 0))
        if not coin or ts < 0 or size <= 0:
            continue

        direction = str(fill.get("direction", "") or "").strip().lower()
        side = str(fill.get("side", "") or "").strip().lower()
        current_pos = float(position_by_coin.get(coin, 0.0) or 0.0)

        if ">" in direction:
            open_ts = entry_ts_by_coin.get(coin)
            if current_pos != 0 and open_ts is not None and ts > open_ts:
                hold_times_ms.append(ts - open_ts)
            target_leg = direction.split(">")[-1].strip()
            if "short" in target_leg:
                position_by_coin[coin] = -size
            elif "long" in target_leg:
                position_by_coin[coin] = size
            else:
                position_by_coin[coin] = -size if side == "sell" else size
            entry_ts_by_coin[coin] = ts
            continue

        if "open long" in direction or "close short" in direction:
            delta = size
        elif "open short" in direction or "close long" in direction:
            delta = -size
        elif fill.get("closed_pnl", 0) != 0 and abs(current_pos) >= 1e-12:
            delta = -min(size, abs(current_pos)) if current_pos > 0 else min(size, abs(current_pos))
        elif side == "buy":
            delta = size
        elif side == "sell":
            delta = -size
        else:
            continue

        new_pos = current_pos + delta
        if abs(current_pos) < 1e-12 and abs(new_pos) >= 1e-12:
            entry_ts_by_coin[coin] = ts
        elif abs(current_pos) >= 1e-12 and abs(new_pos) < 1e-12:
            open_ts = entry_ts_by_coin.pop(coin, None)
            if open_ts is not None and ts > open_ts:
                hold_times_ms.append(ts - open_ts)
        elif current_pos * new_pos < 0:
            open_ts = entry_ts_by_coin.get(coin)
            if open_ts is not None and ts > open_ts:
                hold_times_ms.append(ts - open_ts)
            entry_ts_by_coin[coin] = ts

        position_by_coin[coin] = new_pos

    if not hold_times_ms:
        return 0.0

    avg_ms = sum(hold_times_ms) / len(hold_times_ms)
    return round(avg_ms / (1000 * 3600), 2)  # convert ms → hours


def evaluate_wallet(address: str, bot_score: int = 0,
                    last_sync_time_ms: int = 0) -> Optional[Tuple[WalletReport, List[PenalisedFill]]]:
    """
    Full evaluation pipeline for one wallet:
    download → penalise → equity curve → score → tag golden/not

    Parameters
    ----------
    address : str
        Wallet address to evaluate.
    bot_score : int
        Pre-computed bot score from the discovery phase (0 = clean human).
    last_sync_time_ms : int
        If > 0, only fills newer than this timestamp are fetched (incremental
        mode).  Pass 0 to force a full 90-day download.

    Returns (WalletReport, penalised_fills) tuple so the caller can
    persist fills without re-downloading them (fixes double-download bug).
    Returns None if wallet is skipped.
    """
    new_fills = download_fills_incremental(address, last_sync_time_ms)
    new_penalised = apply_execution_penalties(new_fills)

    if last_sync_time_ms > 0:
        stored_rows = get_wallet_fills(address)
        stored_fills = [_stored_fill_to_raw(row) for row in stored_rows]
        stored_penalised = [_stored_fill_to_penalised(row) for row in stored_rows]
        fills = _merge_fill_history(stored_fills, new_fills)
        penalised = _merge_penalised_history(stored_penalised, new_penalised)
    else:
        fills = new_fills
        penalised = new_penalised

    if len(fills) < MIN_FILLS_FOR_EVAL:
        logger.debug("Skipping %s: only %d fills (need %d)", address[:10], len(fills), MIN_FILLS_FOR_EVAL)
        return None

    # Hard cap: >3000 fills in 90 days = not a human trader
    if len(fills) > MAX_FILLS_FOR_EVAL:
        logger.debug(
            "Skipping %s: %d fills exceeds %d cap (not human-like)",
            address[:10], len(fills), MAX_FILLS_FOR_EVAL,
        )
        return None

    raw_curve, pen_curve, timestamps = build_equity_curve(penalised)

    # Daily returns for Sharpe: computed from the equity curve, not raw PnL.
    # Raw PnL was buggy because days with only opening fills had $0 PnL,
    # diluting the mean to ~0 and making Sharpe always round to 0.00.
    pen_daily_returns = _equity_curve_to_daily_returns(pen_curve, timestamps)

    # Coin breakdown
    coin_pnl: Dict[str, float] = {}
    for f in penalised:
        coin_pnl.setdefault(f.coin, 0.0)
        coin_pnl[f.coin] += f.penalised_pnl

    coins_traded = list(coin_pnl.keys())
    best_coin = max(coin_pnl, key=coin_pnl.get) if coin_pnl else ""
    worst_coin = min(coin_pnl, key=coin_pnl.get) if coin_pnl else ""

    # Win rate on closing fills
    closing_fills = [f for f in penalised if f.penalised_pnl != 0]
    wins = len([f for f in closing_fills if f.penalised_pnl > 0])
    win_rate = (wins / len(closing_fills) * 100) if closing_fills else 0.0

    # Trades per day (based on actual time span, not raw count)
    if len(fills) >= 2:
        span_days = max((fills[-1]["time"] - fills[0]["time"]) / (86400 * 1000), 1)
        tpd = len(fills) / span_days
    else:
        span_days = 1.0
        tpd = 0.0

    raw_total = sum(f.closed_pnl for f in penalised)
    pen_total = sum(f.penalised_pnl for f in penalised)

    # Pre-compute DD once (used twice below)
    pen_dd = compute_max_drawdown(pen_curve)

    # Superhuman filter: these metrics indicate bots/vaults, not real traders.
    # No human sustains 92%+ WR or <0.5% DD over 90 days with meaningful PnL.
    superhuman = False
    if win_rate > MAX_HUMAN_WIN_RATE and abs(pen_total) > 1000:
        superhuman = True
        logger.info(
            "Superhuman filter: %s WR=%.0f%% (>%.0f%%) — likely bot/vault",
            address[:10], win_rate, MAX_HUMAN_WIN_RATE,
        )
    if pen_dd < MIN_HUMAN_DRAWDOWN and pen_total > 5000:
        superhuman = True
        logger.info(
            "Superhuman filter: %s DD=%.1f%% (<%.1f%%) with $%s — likely bot/vault",
            address[:10], pen_dd, MIN_HUMAN_DRAWDOWN, f"{pen_total:+,.0f}",
        )

    # Golden = penalised equity still positive AND curve trending up
    # AND passes the superhuman reality check
    is_golden = False
    if not superhuman and pen_total > GOLDEN_THRESHOLD and len(pen_curve) > 10:
        split = len(pen_curve) // 3
        first_third_avg = sum(pen_curve[:split]) / split if split > 0 else 0.0
        last_third_avg = sum(pen_curve[-split:]) / split if split > 0 else 0.0
        is_golden = last_third_avg > first_third_avg and pen_curve[-1] > pen_curve[0]

    # Compute avg hold time from raw fills (before penalisation, same timestamps)
    avg_hold_hours = compute_avg_hold_time_hours(fills)

    report = WalletReport(
        address=address,
        bot_score=bot_score,
        total_fills=len(fills),
        fills_in_window=len(fills),
        raw_pnl=round(raw_total, 2),
        penalised_pnl=round(pen_total, 2),
        raw_equity_curve=raw_curve,
        penalised_equity_curve=pen_curve,
        equity_timestamps=timestamps,
        max_drawdown_pct=compute_max_drawdown(raw_curve),
        penalised_max_drawdown_pct=pen_dd,
        sharpe_ratio=compute_sharpe(pen_daily_returns),
        win_rate=round(win_rate, 1),
        trades_per_day=round(tpd, 1),
        is_golden=is_golden,
        coins_traded=coins_traded,
        best_coin=best_coin,
        worst_coin=worst_coin,
        avg_hold_time_hours=avg_hold_hours,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )

    tag = "GOLDEN" if is_golden else "not golden"
    logger.info(
        "%s %s: raw=$%s → penalised=$%s "
        "| DD=%.1f%% | Sharpe=%.2f | WR=%.0f%% | hold=%.1fh | %s",
        "★" if is_golden else "·", address[:10],
        f"{raw_total:+,.0f}", f"{pen_total:+,.0f}",
        report.penalised_max_drawdown_pct, report.sharpe_ratio,
        report.win_rate, avg_hold_hours, tag,
    )

    return report, penalised


# ─── Database persistence ────────────────────────────────────────

def _get_db():
    return db.get_connection()


def init_golden_tables():
    """Create tables for golden wallet pipeline."""
    with _get_db() as conn:
        if db.get_backend_name() == "postgres":
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS golden_wallets (
                address TEXT PRIMARY KEY,
                bot_score INTEGER DEFAULT 0,
                total_fills INTEGER DEFAULT 0,
                raw_pnl DOUBLE PRECISION DEFAULT 0,
                penalised_pnl DOUBLE PRECISION DEFAULT 0,
                max_drawdown_pct DOUBLE PRECISION DEFAULT 0,
                penalised_max_drawdown_pct DOUBLE PRECISION DEFAULT 0,
                sharpe_ratio DOUBLE PRECISION DEFAULT 0,
                win_rate DOUBLE PRECISION DEFAULT 0,
                trades_per_day DOUBLE PRECISION DEFAULT 0,
                is_golden BOOLEAN DEFAULT FALSE,
                coins_traded TEXT DEFAULT '[]',
                best_coin TEXT DEFAULT '',
                worst_coin TEXT DEFAULT '',
                raw_equity_curve TEXT DEFAULT '[]',
                penalised_equity_curve TEXT DEFAULT '[]',
                equity_timestamps TEXT DEFAULT '[]',
                evaluated_at TIMESTAMPTZ NOT NULL,
                connected_to_live BOOLEAN DEFAULT FALSE,
                avg_hold_time_hours DOUBLE PRECISION DEFAULT 0,
                last_fill_sync_time BIGINT DEFAULT 0
            );
            -- Migrate existing Postgres deployments that pre-date these columns.
            ALTER TABLE golden_wallets
                ADD COLUMN IF NOT EXISTS avg_hold_time_hours DOUBLE PRECISION DEFAULT 0;
            ALTER TABLE golden_wallets
                ADD COLUMN IF NOT EXISTS last_fill_sync_time BIGINT DEFAULT 0;
            CREATE TABLE IF NOT EXISTS wallet_fills (
                id BIGSERIAL PRIMARY KEY,
                wallet_address TEXT NOT NULL REFERENCES golden_wallets(address),
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                original_price DOUBLE PRECISION NOT NULL,
                penalised_price DOUBLE PRECISION NOT NULL,
                size DOUBLE PRECISION NOT NULL,
                time_ms BIGINT NOT NULL,
                delayed_time_ms BIGINT NOT NULL,
                closed_pnl DOUBLE PRECISION DEFAULT 0,
                penalised_pnl DOUBLE PRECISION DEFAULT 0,
                fee DOUBLE PRECISION DEFAULT 0,
                is_liquidation BOOLEAN DEFAULT FALSE,
                direction TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_wf_addr ON wallet_fills(wallet_address);
            CREATE INDEX IF NOT EXISTS idx_wf_time ON wallet_fills(time_ms);
            CREATE INDEX IF NOT EXISTS idx_wf_coin ON wallet_fills(coin);
            """)
        else:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS golden_wallets (
                address TEXT PRIMARY KEY,
                bot_score INTEGER DEFAULT 0,
                total_fills INTEGER DEFAULT 0,
                raw_pnl REAL DEFAULT 0,
                penalised_pnl REAL DEFAULT 0,
                max_drawdown_pct REAL DEFAULT 0,
                penalised_max_drawdown_pct REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                trades_per_day REAL DEFAULT 0,
                is_golden INTEGER DEFAULT 0,
                coins_traded TEXT DEFAULT '[]',
                best_coin TEXT DEFAULT '',
                worst_coin TEXT DEFAULT '',
                raw_equity_curve TEXT DEFAULT '[]',
                penalised_equity_curve TEXT DEFAULT '[]',
                equity_timestamps TEXT DEFAULT '[]',
                evaluated_at TEXT NOT NULL,
                connected_to_live INTEGER DEFAULT 0,
                avg_hold_time_hours REAL DEFAULT 0,
                last_fill_sync_time INTEGER DEFAULT 0
            );
            -- Migrate: add new columns to existing tables (safe to run repeatedly)
            CREATE TABLE IF NOT EXISTS _migration_guard (id INTEGER PRIMARY KEY);


            CREATE TABLE IF NOT EXISTS wallet_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                original_price REAL NOT NULL,
                penalised_price REAL NOT NULL,
                size REAL NOT NULL,
                time_ms INTEGER NOT NULL,
                delayed_time_ms INTEGER NOT NULL,
                closed_pnl REAL DEFAULT 0,
                penalised_pnl REAL DEFAULT 0,
                fee REAL DEFAULT 0,
                is_liquidation INTEGER DEFAULT 0,
                direction TEXT DEFAULT '',
                FOREIGN KEY (wallet_address) REFERENCES golden_wallets(address)
            );
            CREATE INDEX IF NOT EXISTS idx_wf_addr ON wallet_fills(wallet_address);
            CREATE INDEX IF NOT EXISTS idx_wf_time ON wallet_fills(time_ms);
            CREATE INDEX IF NOT EXISTS idx_wf_coin ON wallet_fills(coin);
            """)
        # Migrate existing databases: add new columns if they don't exist yet.
        for col, definition in [
            ("avg_hold_time_hours", "REAL DEFAULT 0"),
            ("last_fill_sync_time", "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(
                    f"ALTER TABLE golden_wallets ADD COLUMN {col} {definition}"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise


def save_wallet_report(report: WalletReport, last_fill_sync_time_ms: int = 0):
    """
    Persist a wallet evaluation report.

    last_fill_sync_time_ms is recorded so that the next evaluation can use
    incremental fill fetching instead of re-downloading the full 90-day window.
    Pass the timestamp of the newest fill in the current batch as this value.
    """
    with _get_db() as conn:
        conn.execute("""
            INSERT INTO golden_wallets
            (address, bot_score, total_fills, raw_pnl, penalised_pnl,
             max_drawdown_pct, penalised_max_drawdown_pct, sharpe_ratio,
             win_rate, trades_per_day, is_golden, coins_traded, best_coin,
             worst_coin, raw_equity_curve, penalised_equity_curve,
             equity_timestamps, evaluated_at, avg_hold_time_hours,
             last_fill_sync_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (address) DO UPDATE SET
                bot_score = EXCLUDED.bot_score,
                total_fills = EXCLUDED.total_fills,
                raw_pnl = EXCLUDED.raw_pnl,
                penalised_pnl = EXCLUDED.penalised_pnl,
                max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                penalised_max_drawdown_pct = EXCLUDED.penalised_max_drawdown_pct,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                win_rate = EXCLUDED.win_rate,
                trades_per_day = EXCLUDED.trades_per_day,
                is_golden = EXCLUDED.is_golden,
                coins_traded = EXCLUDED.coins_traded,
                best_coin = EXCLUDED.best_coin,
                worst_coin = EXCLUDED.worst_coin,
                raw_equity_curve = EXCLUDED.raw_equity_curve,
                penalised_equity_curve = EXCLUDED.penalised_equity_curve,
                equity_timestamps = EXCLUDED.equity_timestamps,
                evaluated_at = EXCLUDED.evaluated_at,
                avg_hold_time_hours = EXCLUDED.avg_hold_time_hours,
                last_fill_sync_time = EXCLUDED.last_fill_sync_time
        """, (
            report.address, report.bot_score, report.total_fills,
            report.raw_pnl, report.penalised_pnl,
            report.max_drawdown_pct, report.penalised_max_drawdown_pct,
            report.sharpe_ratio, report.win_rate, report.trades_per_day,
            bool(report.is_golden),
            json.dumps(report.coins_traded), report.best_coin, report.worst_coin,
            json.dumps(report.raw_equity_curve[-500:]),  # cap storage
            json.dumps(report.penalised_equity_curve[-500:]),
            json.dumps(report.equity_timestamps[-500:]),
            report.evaluated_at,
            report.avg_hold_time_hours,
            last_fill_sync_time_ms,
        ))


def save_wallet_fills(address: str, penalised_fills: List[PenalisedFill]):
    """Persist penalised fills for backtest replay."""
    with _get_db() as conn:
        # Clear old fills for this wallet
        conn.execute("DELETE FROM wallet_fills WHERE wallet_address = ?", (address,))
        for f in penalised_fills:
            conn.execute("""
                INSERT INTO wallet_fills
                (wallet_address, coin, side, original_price, penalised_price,
                 size, time_ms, delayed_time_ms, closed_pnl, penalised_pnl,
                 fee, is_liquidation, direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                address, f.coin, f.side, f.original_price, f.penalised_price,
                f.size, f.time_ms, f.delayed_time_ms, f.closed_pnl,
                f.penalised_pnl, f.fee, bool(f.is_liquidation),
                f.direction,
            ))
        logger.debug(f"Saved {len(penalised_fills)} fills for {address[:10]}")


def get_golden_wallets() -> List[Dict]:
    """Get all wallets flagged as golden."""
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM golden_wallets WHERE is_golden = 1 "
            "ORDER BY penalised_pnl DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def purge_non_golden_wallets() -> int:
    """
    Remove all non-golden wallets from the golden_wallets table + their fills.
    Frees DB space and ensures only proven wallets remain.
    Returns number of wallets purged.
    """
    with _get_db() as conn:
        # Get non-golden addresses first (for fill cleanup)
        rows = conn.execute(
            "SELECT address FROM golden_wallets WHERE is_golden = 0"
        ).fetchall()
        non_golden = [r["address"] for r in rows]

        if not non_golden:
            logger.info("No non-golden wallets to purge")
            return 0

        # Delete fills for non-golden wallets
        for addr in non_golden:
            conn.execute("DELETE FROM wallet_fills WHERE wallet_address = ?", (addr,))

        # Delete non-golden wallet records
        conn.execute("DELETE FROM golden_wallets WHERE is_golden = 0")

        logger.info(f"Purged {len(non_golden)} non-golden wallets and their fills")
        return len(non_golden)


def get_all_wallet_reports() -> List[Dict]:
    """Get all evaluated wallets (golden and non-golden)."""
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM golden_wallets ORDER BY penalised_pnl DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_wallet_fills(address: str) -> List[Dict]:
    """Get all stored fills for a wallet."""
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM wallet_fills WHERE wallet_address = ? ORDER BY time_ms",
            (address,)
        ).fetchall()
        return [dict(r) for r in rows]


# ─── Batch evaluation ────────────────────────────────────────────

def get_human_wallets_from_db() -> List[Dict]:
    """Get all active human-like traders from the bot's DB."""
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT address, metadata FROM traders WHERE active = ?",
            (True,),
        ).fetchall()
        wallets = []
        for r in rows:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            bot_score = meta.get("bot_score", 0)
            if bot_score < 3:  # Human-like threshold
                wallets.append({
                    "address": r["address"],
                    "bot_score": bot_score,
                })
        return wallets


def _load_last_sync_times() -> Dict[str, int]:
    """
    Load last_fill_sync_time for all wallets from the DB in one query.
    Returns {address: last_fill_sync_time_ms}.
    """
    try:
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT address, last_fill_sync_time FROM golden_wallets"
            ).fetchall()
            return {r["address"]: int(r["last_fill_sync_time"] or 0) for r in rows}
    except Exception:
        return {}


def run_golden_scan(max_wallets: int = 200) -> Dict:
    """
    Main entry point: scan human-like wallets, evaluate, tag golden ones.
    Returns summary stats.

    Uses incremental fill fetching: wallets already in the DB only download
    fills newer than their last evaluation, cutting API calls by ~75%.
    """
    init_golden_tables()

    wallets = get_human_wallets_from_db()
    if not wallets:
        logger.warning("No human-like wallets found in DB. Run discovery first.")
        return {"scanned": 0, "golden": 0, "error": "no wallets"}

    logger.info(
        "Starting golden wallet scan: %d human-like wallets (evaluating up to %d)",
        len(wallets), max_wallets,
    )

    wallets = wallets[:max_wallets]
    results = []
    golden_count = 0
    incremental_count = 0

    # Load existing sync times for all wallets in one DB query
    sync_times = _load_last_sync_times()

    rate_limit_hits = 0
    for i, w in enumerate(wallets):
        addr = w["address"]
        last_sync = sync_times.get(addr, 0)
        mode = "incremental" if last_sync > 0 else "full"
        logger.debug("[%d/%d] Evaluating %s... (%s)", i + 1, len(wallets), addr[:10], mode)
        if last_sync > 0:
            incremental_count += 1
        try:
            result = evaluate_wallet(addr, w.get("bot_score", 0), last_sync_time_ms=last_sync)
            if result:
                report, penalised_fills = result
                # Determine newest fill timestamp for next incremental fetch
                newest_ts = max((f.time_ms for f in penalised_fills), default=0)
                save_wallet_report(report, last_fill_sync_time_ms=newest_ts)
                # Reuse the fills from evaluate_wallet — no re-download!
                save_wallet_fills(addr, penalised_fills)
                results.append(report)
                if report.is_golden:
                    golden_count += 1
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                rate_limit_hits += 1
                backoff = _rate_limit_backoff(rate_limit_hits, label=" golden scan")
                time.sleep(backoff)
            else:
                logger.error("Error evaluating %s: %s", w["address"][:10], e)

        # Generous pause between wallets — each wallet triggers multiple API calls
        time.sleep(BETWEEN_WALLET_SLEEP)

    summary = {
        "scanned": len(results),
        "golden": golden_count,
        "total_human_wallets": len(wallets),
        "incremental_fetches": incremental_count,
        "full_fetches": len(wallets) - incremental_count,
        "golden_addresses": [r.address for r in results if r.is_golden],
        "top_by_penalised_pnl": sorted(
            [
                {
                    "addr": r.address[:10],
                    "pnl": r.penalised_pnl,
                    "sharpe": r.sharpe_ratio,
                    "dd": r.penalised_max_drawdown_pct,
                    "hold_h": r.avg_hold_time_hours,
                    "golden": r.is_golden,
                }
                for r in results
            ],
            key=lambda x: x["pnl"],
            reverse=True,
        )[:10],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Golden scan complete: %d/%d wallets are golden (%d incremental, %d full fetches)",
        golden_count, len(results), incremental_count, len(wallets) - incremental_count,
    )
    for r in results:
        if r.is_golden:
            logger.debug(
                "  ★ %s: PnL=$%s Sharpe=%.2f DD=%.1f%% hold=%.1fh",
                r.address[:10], f"{r.penalised_pnl:+,.0f}", r.sharpe_ratio,
                r.penalised_max_drawdown_pct, r.avg_hold_time_hours,
            )

    return summary


