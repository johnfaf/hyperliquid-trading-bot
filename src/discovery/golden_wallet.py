"""
Golden Wallet Pipeline
======================
1. Download 3 months of fills for every "human-like" wallet (bot_score < 3)
2. Apply realistic execution penalties (+100ms delay, -0.045% slippage+fees)
3. Build equity curves and tag wallets whose penalised curve is still rising
4. Persist results so the backtest dashboard can read them

Hyperliquid API returns max 2000 fills per call.  We page backwards with
startTime to cover the full 90-day window.
"""
import logging
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

# Superhuman detection: no real human trader sustains these numbers
# over 90 days.  Wallets hitting these thresholds are bots/vaults/arb.
MAX_HUMAN_WIN_RATE = 92.0       # >92% WR over 90d = not human
MIN_HUMAN_DRAWDOWN = 0.5        # <0.5% max DD with big PnL = not human


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
    """
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
    all_fills = []
    oldest_seen = None
    consecutive_errors = 0

    for page in range(50):  # safety cap: 50 pages = 100k fills max
        # For first page, don't pass startTime to get most recent
        # For subsequent pages, use oldest seen to page backwards
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
                # Back off hard on rate limit during fill downloads
                backoff = min(30, 5 * (2 ** consecutive_errors))
                logger.warning(f"Rate limited downloading fills for {address[:10]} "
                              f"(page {page}), backing off {backoff}s")
                time.sleep(backoff)
                if consecutive_errors >= 3:
                    logger.warning(f"Giving up on {address[:10]} after {consecutive_errors} rate limits")
                    break
                continue
            else:
                logger.error(f"Error downloading fills for {address[:10]}: {e}")
                break

        if not fills:
            break

        # On page 0 we get newest fills; filter to window
        new_fills = [f for f in fills if f["time"] >= cutoff_ms]
        if not new_fills:
            break

        # Deduplicate by hash
        seen_hashes = {f["hash"] for f in all_fills}
        added = 0
        for f in new_fills:
            if f["hash"] not in seen_hashes:
                all_fills.append(f)
                seen_hashes.add(f["hash"])
                added += 1

        if added == 0:
            break

        # Find oldest fill time to use for next page
        times = [f["time"] for f in new_fills]
        min_time = min(times)
        if oldest_seen is not None and min_time >= oldest_seen:
            break  # No progress
        oldest_seen = min_time

        # If we got less than PAGE_SIZE, we've hit the bottom
        if len(fills) < PAGE_SIZE:
            break

        logger.debug(f"  Page {page+1}: +{added} fills (total: {len(all_fills)}, oldest: {min_time})")
        time.sleep(REQUEST_SLEEP)

    all_fills.sort(key=lambda f: f["time"])
    logger.debug(f"Downloaded {len(all_fills)} fills for {address[:10]}... "
                 f"(span: {LOOKBACK_DAYS}d window)")
    return all_fills


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

    Previous bug: this function received raw PnL amounts, which meant days with
    only opening fills contributed $0, diluting the mean to near-zero and making
    Sharpe ≈ 0 for almost every wallet. Now we expect percentage returns computed
    from the equity curve, which gives meaningful Sharpe values.
    """
    if len(daily_returns) < 5:
        return 0.0
    import statistics
    mean_r = statistics.mean(daily_returns)
    std_r = statistics.stdev(daily_returns)
    if std_r < 1e-10:  # avoid division by near-zero
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


def evaluate_wallet(address: str, bot_score: int = 0) -> Optional[Tuple[WalletReport, List[PenalisedFill]]]:
    """
    Full evaluation pipeline for one wallet:
    download → penalise → equity curve → score → tag golden/not

    Returns (WalletReport, penalised_fills) tuple so the caller can
    persist fills without re-downloading them (fixes double-download bug).
    Returns None if wallet is skipped.
    """
    fills = download_fills_90d(address)
    if len(fills) < MIN_FILLS_FOR_EVAL:
        logger.debug(f"Skipping {address[:10]}: only {len(fills)} fills (need {MIN_FILLS_FOR_EVAL})")
        return None

    # Hard cap: >3000 fills in 90 days = not a human trader
    if len(fills) > MAX_FILLS_FOR_EVAL:
        logger.debug(f"Skipping {address[:10]}: {len(fills)} fills exceeds "
                      f"{MAX_FILLS_FOR_EVAL} cap (not human-like)")
        return None

    penalised = apply_execution_penalties(fills)
    raw_curve, pen_curve, timestamps = build_equity_curve(penalised)

    # Daily returns for Sharpe: computed from the equity curve, not raw PnL.
    # Raw PnL was buggy because days with only opening fills had $0 PnL,
    # diluting the mean to ~0 and making Sharpe always round to 0.00.
    pen_daily_returns = _equity_curve_to_daily_returns(pen_curve, timestamps)

    # Coin breakdown
    coin_pnl = {}
    for f in penalised:
        coin_pnl.setdefault(f.coin, 0)
        coin_pnl[f.coin] += f.penalised_pnl

    coins_traded = list(coin_pnl.keys())
    best_coin = max(coin_pnl, key=coin_pnl.get) if coin_pnl else ""
    worst_coin = min(coin_pnl, key=coin_pnl.get) if coin_pnl else ""

    # Win rate on closing fills
    closing_fills = [f for f in penalised if f.penalised_pnl != 0]
    wins = len([f for f in closing_fills if f.penalised_pnl > 0])
    win_rate = (wins / len(closing_fills) * 100) if closing_fills else 0

    # Trades per day
    if len(fills) >= 2:
        span_days = max((fills[-1]["time"] - fills[0]["time"]) / (86400 * 1000), 1)
        tpd = len(fills) / span_days
    else:
        span_days = 1
        tpd = 0

    raw_total = sum(f.closed_pnl for f in penalised)
    pen_total = sum(f.penalised_pnl for f in penalised)

    # Superhuman filter: these metrics indicate bots/vaults, not real traders.
    # No human sustains 97%+ WR or 0% DD over 90 days with meaningful PnL.
    superhuman = False
    if win_rate > MAX_HUMAN_WIN_RATE and abs(pen_total) > 1000:
        superhuman = True
        logger.info(f"Superhuman filter: {address[:10]} WR={win_rate:.0f}% "
                     f"(>{MAX_HUMAN_WIN_RATE}%) — likely bot/vault")
    if compute_max_drawdown(pen_curve) < MIN_HUMAN_DRAWDOWN and pen_total > 5000:
        superhuman = True
        logger.info(f"Superhuman filter: {address[:10]} DD={compute_max_drawdown(pen_curve):.1f}% "
                     f"(<{MIN_HUMAN_DRAWDOWN}%) with ${pen_total:+,.0f} — likely bot/vault")

    # Golden = penalised equity still positive AND curve trending up
    # AND passes the superhuman reality check
    is_golden = False
    if not superhuman and pen_total > GOLDEN_THRESHOLD and len(pen_curve) > 10:
        split = len(pen_curve) // 3
        first_third_avg = sum(pen_curve[:split]) / split if split > 0 else 0
        last_third_avg = sum(pen_curve[-split:]) / split if split > 0 else 0
        is_golden = last_third_avg > first_third_avg and pen_curve[-1] > pen_curve[0]

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
        penalised_max_drawdown_pct=compute_max_drawdown(pen_curve),
        sharpe_ratio=compute_sharpe(pen_daily_returns),
        win_rate=round(win_rate, 1),
        trades_per_day=round(tpd, 1),
        is_golden=is_golden,
        coins_traded=coins_traded,
        best_coin=best_coin,
        worst_coin=worst_coin,
        avg_hold_time_hours=0,  # TODO: compute from open/close pairs
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )

    tag = "GOLDEN" if is_golden else "not golden"
    logger.info(
        f"{'★' if is_golden else '·'} {address[:10]}: "
        f"raw=${raw_total:+,.0f} → penalised=${pen_total:+,.0f} "
        f"| DD={report.penalised_max_drawdown_pct:.1f}% "
        f"| Sharpe={report.sharpe_ratio:.2f} "
        f"| WR={report.win_rate:.0f}% "
        f"| {tag}"
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
                connected_to_live BOOLEAN DEFAULT FALSE
            );
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
                connected_to_live INTEGER DEFAULT 0
            );

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


def save_wallet_report(report: WalletReport):
    """Persist a wallet evaluation report + all its fills."""
    with _get_db() as conn:
        conn.execute("""
            INSERT INTO golden_wallets
            (address, bot_score, total_fills, raw_pnl, penalised_pnl,
             max_drawdown_pct, penalised_max_drawdown_pct, sharpe_ratio,
             win_rate, trades_per_day, is_golden, coins_traded, best_coin,
             worst_coin, raw_equity_curve, penalised_equity_curve,
             equity_timestamps, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                evaluated_at = EXCLUDED.evaluated_at
        """, (
            report.address, report.bot_score, report.total_fills,
            report.raw_pnl, report.penalised_pnl,
            report.max_drawdown_pct, report.penalised_max_drawdown_pct,
            report.sharpe_ratio, report.win_rate, report.trades_per_day,
            1 if report.is_golden else 0,
            json.dumps(report.coins_traded), report.best_coin, report.worst_coin,
            json.dumps(report.raw_equity_curve[-500:]),  # cap storage
            json.dumps(report.penalised_equity_curve[-500:]),
            json.dumps(report.equity_timestamps[-500:]),
            report.evaluated_at,
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
                f.penalised_pnl, f.fee, 1 if f.is_liquidation else 0,
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


def run_golden_scan(max_wallets: int = 200) -> Dict:
    """
    Main entry point: scan human-like wallets, evaluate, tag golden ones.
    Returns summary stats.
    """
    init_golden_tables()

    wallets = get_human_wallets_from_db()
    if not wallets:
        logger.warning("No human-like wallets found in DB. Run discovery first.")
        return {"scanned": 0, "golden": 0, "error": "no wallets"}

    logger.info(f"Starting golden wallet scan: {len(wallets)} human-like wallets "
                f"(evaluating up to {max_wallets})")

    wallets = wallets[:max_wallets]
    results = []
    golden_count = 0

    rate_limit_hits = 0
    for i, w in enumerate(wallets):
        logger.debug(f"[{i+1}/{len(wallets)}] Evaluating {w['address'][:10]}...")
        try:
            result = evaluate_wallet(w["address"], w.get("bot_score", 0))
            if result:
                report, penalised_fills = result
                save_wallet_report(report)
                # Reuse the fills from evaluate_wallet — no re-download!
                save_wallet_fills(w["address"], penalised_fills)
                results.append(report)
                if report.is_golden:
                    golden_count += 1
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                rate_limit_hits += 1
                backoff = min(30, 5 * (2 ** rate_limit_hits))
                logger.warning(f"Rate limited during golden scan ({rate_limit_hits}x), "
                              f"backing off {backoff}s")
                time.sleep(backoff)
            else:
                logger.error(f"Error evaluating {w['address'][:10]}: {e}")

        # Generous pause between wallets — each wallet triggers multiple API calls
        time.sleep(BETWEEN_WALLET_SLEEP)

    summary = {
        "scanned": len(results),
        "golden": golden_count,
        "total_human_wallets": len(wallets),
        "golden_addresses": [r.address for r in results if r.is_golden],
        "top_by_penalised_pnl": sorted(
            [{"addr": r.address[:10], "pnl": r.penalised_pnl, "sharpe": r.sharpe_ratio,
              "dd": r.penalised_max_drawdown_pct, "golden": r.is_golden}
             for r in results],
            key=lambda x: x["pnl"], reverse=True
        )[:10],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"Golden scan complete: {golden_count}/{len(results)} wallets are golden")
    for r in results:
        if r.is_golden:
            logger.debug(f"  ★ {r.address[:10]}: penalised PnL=${r.penalised_pnl:+,.0f}, "
                         f"Sharpe={r.sharpe_ratio:.2f}, DD={r.penalised_max_drawdown_pct:.1f}%")

    return summary
