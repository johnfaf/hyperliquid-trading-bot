"""
Backtest Engine
===============
Runs multi-timeframe analysis on stored wallet fills.
Splits the 90-day fill history into configurable windows (1d, 1h, 15m)
and computes performance metrics for each period.

Used by the backtest dashboard to show granular performance heatmaps.
"""
import logging
import json
import sqlite3
import os
import sys
import statistics
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("backtest_engine")


# ─── Timeframe definitions ──────────────────────────────────────
class Timeframe(Enum):
    DAY = "1d"
    HOUR_4 = "4h"
    HOUR_1 = "1h"
    MIN_15 = "15m"


TIMEFRAME_MS = {
    Timeframe.DAY: 86400 * 1000,
    Timeframe.HOUR_4: 4 * 3600 * 1000,
    Timeframe.HOUR_1: 3600 * 1000,
    Timeframe.MIN_15: 900 * 1000,
}

TIMEFRAME_LABELS = {
    Timeframe.DAY: "Daily",
    Timeframe.HOUR_4: "4-Hour",
    Timeframe.HOUR_1: "Hourly",
    Timeframe.MIN_15: "15-Minute",
}


# ─── Data structures ─────────────────────────────────────────────
@dataclass
class PeriodResult:
    """Performance for a single time period."""
    period_start_ms: int
    period_end_ms: int
    period_label: str
    fill_count: int
    closing_fills: int
    pnl: float
    penalised_pnl: float
    volume: float
    wins: int
    losses: int
    win_rate: float
    best_trade: float
    worst_trade: float
    coins_active: List[str] = field(default_factory=list)


@dataclass
class TimeframeReport:
    """Full report for one timeframe across all periods."""
    timeframe: str
    total_periods: int
    active_periods: int
    periods: List[PeriodResult] = field(default_factory=list)
    # Aggregated
    total_pnl: float = 0
    total_penalised_pnl: float = 0
    avg_period_pnl: float = 0
    std_period_pnl: float = 0
    best_period_pnl: float = 0
    worst_period_pnl: float = 0
    profitable_periods: int = 0
    profitable_pct: float = 0
    avg_fills_per_period: float = 0
    consistency_score: float = 0  # % of active periods that are profitable


@dataclass
class BacktestResult:
    """Complete backtest result for one wallet across all timeframes."""
    address: str
    evaluated_at: str
    timeframes: Dict[str, TimeframeReport] = field(default_factory=dict)
    # Per-coin breakdown
    coin_performance: Dict[str, Dict] = field(default_factory=dict)
    # Hour-of-day analysis
    hourly_pnl: Dict[int, float] = field(default_factory=dict)
    # Day-of-week analysis
    daily_pnl: Dict[int, float] = field(default_factory=dict)
    # Overall
    total_raw_pnl: float = 0
    total_penalised_pnl: float = 0


# ─── Database helpers ────────────────────────────────────────────

def _get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_fills(address: str) -> List[Dict]:
    """Load penalised fills from DB."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM wallet_fills WHERE wallet_address = ? ORDER BY time_ms",
            (address,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Core analysis ───────────────────────────────────────────────

def analyse_timeframe(fills: List[Dict], timeframe: Timeframe) -> TimeframeReport:
    """
    Bucket fills into fixed-width time windows and compute per-period metrics.
    """
    if not fills:
        return TimeframeReport(timeframe=timeframe.value, total_periods=0, active_periods=0)

    window_ms = TIMEFRAME_MS[timeframe]
    min_time = fills[0]["time_ms"]
    max_time = fills[-1]["time_ms"]

    # Generate all period boundaries
    period_start = (min_time // window_ms) * window_ms
    period_end = ((max_time // window_ms) + 1) * window_ms

    periods = []
    current = period_start

    while current < period_end:
        next_boundary = current + window_ms
        # Get fills in this window
        bucket = [f for f in fills if current <= f["time_ms"] < next_boundary]

        if bucket:
            closing = [f for f in bucket if f["penalised_pnl"] != 0]
            pnl_values = [f["penalised_pnl"] for f in closing]
            raw_pnl_values = [f["closed_pnl"] for f in closing]
            wins = len([p for p in pnl_values if p > 0])
            losses = len([p for p in pnl_values if p < 0])

            period = PeriodResult(
                period_start_ms=current,
                period_end_ms=next_boundary,
                period_label=_format_period_label(current, timeframe),
                fill_count=len(bucket),
                closing_fills=len(closing),
                pnl=round(sum(raw_pnl_values), 2),
                penalised_pnl=round(sum(pnl_values), 2),
                volume=round(sum(f["original_price"] * f["size"] for f in bucket), 2),
                wins=wins,
                losses=losses,
                win_rate=round(wins / len(closing) * 100, 1) if closing else 0,
                best_trade=round(max(pnl_values), 2) if pnl_values else 0,
                worst_trade=round(min(pnl_values), 2) if pnl_values else 0,
                coins_active=list(set(f["coin"] for f in bucket)),
            )
            periods.append(period)

        current = next_boundary

    # Aggregate
    active_periods = [p for p in periods if p.closing_fills > 0]
    period_pnls = [p.penalised_pnl for p in active_periods]

    profitable = [p for p in active_periods if p.penalised_pnl > 0]

    report = TimeframeReport(
        timeframe=timeframe.value,
        total_periods=len(periods) if periods else 0,
        active_periods=len(active_periods),
        periods=periods,
        total_pnl=round(sum(p.pnl for p in periods), 2),
        total_penalised_pnl=round(sum(p.penalised_pnl for p in periods), 2),
        avg_period_pnl=round(statistics.mean(period_pnls), 2) if period_pnls else 0,
        std_period_pnl=round(statistics.stdev(period_pnls), 2) if len(period_pnls) > 1 else 0,
        best_period_pnl=round(max(period_pnls), 2) if period_pnls else 0,
        worst_period_pnl=round(min(period_pnls), 2) if period_pnls else 0,
        profitable_periods=len(profitable),
        profitable_pct=round(len(profitable) / len(active_periods) * 100, 1) if active_periods else 0,
        avg_fills_per_period=round(statistics.mean([p.fill_count for p in active_periods]), 1) if active_periods else 0,
        consistency_score=round(len(profitable) / len(active_periods) * 100, 1) if active_periods else 0,
    )

    return report


def analyse_by_coin(fills: List[Dict]) -> Dict[str, Dict]:
    """Per-coin performance breakdown."""
    coins = {}
    for f in fills:
        c = f["coin"]
        if c not in coins:
            coins[c] = {"fills": 0, "closing": 0, "raw_pnl": 0, "pen_pnl": 0,
                        "volume": 0, "wins": 0, "losses": 0}
        coins[c]["fills"] += 1
        coins[c]["volume"] += f["original_price"] * f["size"]
        if f["penalised_pnl"] != 0:
            coins[c]["closing"] += 1
            coins[c]["raw_pnl"] += f["closed_pnl"]
            coins[c]["pen_pnl"] += f["penalised_pnl"]
            if f["penalised_pnl"] > 0:
                coins[c]["wins"] += 1
            else:
                coins[c]["losses"] += 1

    # Round and add win_rate
    for c in coins:
        coins[c]["raw_pnl"] = round(coins[c]["raw_pnl"], 2)
        coins[c]["pen_pnl"] = round(coins[c]["pen_pnl"], 2)
        coins[c]["volume"] = round(coins[c]["volume"], 2)
        total = coins[c]["wins"] + coins[c]["losses"]
        coins[c]["win_rate"] = round(coins[c]["wins"] / total * 100, 1) if total > 0 else 0

    return coins


def analyse_by_hour(fills: List[Dict]) -> Dict[int, float]:
    """PnL bucketed by hour of day (UTC)."""
    hourly = {h: 0 for h in range(24)}
    for f in fills:
        if f["penalised_pnl"] != 0:
            hour = (f["time_ms"] // 1000 % 86400) // 3600
            hourly[hour] += f["penalised_pnl"]
    return {h: round(v, 2) for h, v in hourly.items()}


def analyse_by_weekday(fills: List[Dict]) -> Dict[int, float]:
    """PnL bucketed by day of week (0=Monday, 6=Sunday)."""
    daily = {d: 0 for d in range(7)}
    for f in fills:
        if f["penalised_pnl"] != 0:
            dt = datetime.fromtimestamp(f["time_ms"] / 1000, tz=timezone.utc)
            daily[dt.weekday()] += f["penalised_pnl"]
    return {d: round(v, 2) for d, v in daily.items()}


def _format_period_label(start_ms: int, tf: Timeframe) -> str:
    """Human-readable label for a time period."""
    dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    if tf == Timeframe.DAY:
        return dt.strftime("%Y-%m-%d")
    elif tf == Timeframe.HOUR_4:
        return dt.strftime("%m-%d %H:%M")
    elif tf == Timeframe.HOUR_1:
        return dt.strftime("%m-%d %H:%M")
    else:
        return dt.strftime("%m-%d %H:%M")


# ─── Main entry point ────────────────────────────────────────────

def run_backtest(address: str) -> Optional[BacktestResult]:
    """
    Run full multi-timeframe backtest for a single wallet.
    Loads fills from DB, runs analysis at each timeframe.
    """
    fills = _load_fills(address)
    if not fills:
        logger.warning(f"No fills found for {address[:10]}. Run golden scan first.")
        return None

    logger.info(f"Running backtest for {address[:10]}: {len(fills)} fills")

    result = BacktestResult(
        address=address,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Multi-timeframe analysis
    for tf in [Timeframe.DAY, Timeframe.HOUR_1, Timeframe.MIN_15]:
        tf_report = analyse_timeframe(fills, tf)
        result.timeframes[tf.value] = tf_report
        logger.info(
            f"  {TIMEFRAME_LABELS[tf]}: {tf_report.active_periods} active periods, "
            f"penalised PnL=${tf_report.total_penalised_pnl:+,.0f}, "
            f"consistency={tf_report.consistency_score:.0f}%"
        )

    # Coin breakdown
    result.coin_performance = analyse_by_coin(fills)

    # Time-of-day and day-of-week
    result.hourly_pnl = analyse_by_hour(fills)
    result.daily_pnl = analyse_by_weekday(fills)

    # Totals
    result.total_raw_pnl = round(sum(f["closed_pnl"] for f in fills), 2)
    result.total_penalised_pnl = round(sum(f["penalised_pnl"] for f in fills), 2)

    return result


def run_all_backtests() -> List[BacktestResult]:
    """Run backtests for all evaluated wallets (golden and non-golden)."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT address FROM golden_wallets ORDER BY penalised_pnl DESC"
        ).fetchall()
        addresses = [r["address"] for r in rows]
    finally:
        conn.close()

    if not addresses:
        logger.warning("No evaluated wallets found. Run golden scan first.")
        return []

    results = []
    for addr in addresses:
        try:
            result = run_backtest(addr)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Backtest error for {addr[:10]}: {e}")

    logger.info(f"Completed backtests for {len(results)}/{len(addresses)} wallets")
    return results


# ─── Persistence ─────────────────────────────────────────────────

def init_backtest_tables():
    """Create backtest results table."""
    conn = _get_db()
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            address TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            total_periods INTEGER DEFAULT 0,
            active_periods INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            total_penalised_pnl REAL DEFAULT 0,
            avg_period_pnl REAL DEFAULT 0,
            std_period_pnl REAL DEFAULT 0,
            best_period_pnl REAL DEFAULT 0,
            worst_period_pnl REAL DEFAULT 0,
            profitable_periods INTEGER DEFAULT 0,
            profitable_pct REAL DEFAULT 0,
            consistency_score REAL DEFAULT 0,
            periods_json TEXT DEFAULT '[]',
            evaluated_at TEXT NOT NULL,
            PRIMARY KEY (address, timeframe)
        );

        CREATE TABLE IF NOT EXISTS backtest_coin_perf (
            address TEXT NOT NULL,
            coin TEXT NOT NULL,
            fills INTEGER DEFAULT 0,
            closing INTEGER DEFAULT 0,
            raw_pnl REAL DEFAULT 0,
            pen_pnl REAL DEFAULT 0,
            volume REAL DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            PRIMARY KEY (address, coin)
        );

        CREATE TABLE IF NOT EXISTS backtest_time_analysis (
            address TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            bucket INTEGER NOT NULL,
            pnl REAL DEFAULT 0,
            PRIMARY KEY (address, analysis_type, bucket)
        );
        """)
        conn.commit()
    finally:
        conn.close()


def save_backtest_result(result: BacktestResult):
    """Persist a complete backtest result."""
    conn = _get_db()
    try:
        # Save timeframe results
        for tf_key, tf_report in result.timeframes.items():
            # Serialise periods with capped detail
            periods_data = []
            for p in tf_report.periods:
                periods_data.append({
                    "start": p.period_start_ms,
                    "end": p.period_end_ms,
                    "label": p.period_label,
                    "fills": p.fill_count,
                    "pnl": p.penalised_pnl,
                    "raw_pnl": p.pnl,
                    "wr": p.win_rate,
                    "vol": p.volume,
                    "coins": p.coins_active,
                })

            conn.execute("""
                INSERT OR REPLACE INTO backtest_results
                (address, timeframe, total_periods, active_periods, total_pnl,
                 total_penalised_pnl, avg_period_pnl, std_period_pnl,
                 best_period_pnl, worst_period_pnl, profitable_periods,
                 profitable_pct, consistency_score, periods_json, evaluated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.address, tf_key,
                tf_report.total_periods, tf_report.active_periods,
                tf_report.total_pnl, tf_report.total_penalised_pnl,
                tf_report.avg_period_pnl, tf_report.std_period_pnl,
                tf_report.best_period_pnl, tf_report.worst_period_pnl,
                tf_report.profitable_periods, tf_report.profitable_pct,
                tf_report.consistency_score,
                json.dumps(periods_data),
                result.evaluated_at,
            ))

        # Save coin performance
        for coin, stats in result.coin_performance.items():
            conn.execute("""
                INSERT OR REPLACE INTO backtest_coin_perf
                (address, coin, fills, closing, raw_pnl, pen_pnl,
                 volume, wins, losses, win_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.address, coin, stats["fills"], stats["closing"],
                stats["raw_pnl"], stats["pen_pnl"], stats["volume"],
                stats["wins"], stats["losses"], stats["win_rate"],
            ))

        # Save hourly analysis
        for hour, pnl in result.hourly_pnl.items():
            conn.execute("""
                INSERT OR REPLACE INTO backtest_time_analysis
                (address, analysis_type, bucket, pnl) VALUES (?, 'hour', ?, ?)
            """, (result.address, hour, pnl))

        # Save weekday analysis
        for day, pnl in result.daily_pnl.items():
            conn.execute("""
                INSERT OR REPLACE INTO backtest_time_analysis
                (address, analysis_type, bucket, pnl) VALUES (?, 'weekday', ?, ?)
            """, (result.address, day, pnl))

        conn.commit()
        logger.info(f"Saved backtest results for {result.address[:10]}")
    finally:
        conn.close()
