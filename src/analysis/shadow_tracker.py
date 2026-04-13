"""
Shadow Mode PnL Attribution Tracker
====================================
Tracks paper-trading performance over a rolling 30-day window with
PnL attribution per signal source. Enables comparison of:
  - Strategy-derived signals vs copy-trading
  - Polymarket event signals vs technical analysis
  - Options flow signals vs regime-based sizing

Stores data in the shared runtime database for persistence across restarts.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from statistics import mean, stdev

from src.data import database as db

logger = logging.getLogger(__name__)


class ShadowTracker:
    """
    Paper-trading shadow mode tracker with per-signal-source PnL attribution.

    Supports multiple signal sources (strategy:momentum, copy:golden_wallet, polymarket, etc.)
    and computes risk-adjusted performance metrics per source over a rolling window.
    """

    def __init__(self, db_path=None):
        """
        Initialize the shadow tracker.

        Args:
            db_path (str, optional): Legacy parameter retained for compatibility.
                                    ShadowTracker now uses the shared runtime DB.
        """
        if db_path:
            logger.warning(
                "ShadowTracker db_path override (%s) is ignored; "
                "using shared runtime database instead.",
                db_path,
            )
        self.db_path = db.get_db_path()

        self._ensure_tables()
        logger.info("ShadowTracker initialized with shared DB at %s", self.db_path)

    @contextmanager
    def _get_connection(self):
        """Context manager for shared database connections."""
        with db.get_connection() as conn:
            yield conn

    def _ensure_tables(self):
        """Create shadow_trades and shadow_attribution tables if they don't exist."""
        with self._get_connection() as conn:
            if db.get_backend_name() == "postgres":
                conn.executescript("""
                CREATE TABLE IF NOT EXISTS shadow_trades (
                    id BIGSERIAL PRIMARY KEY,
                    signal_source TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    exit_price DOUBLE PRECISION,
                    size DOUBLE PRECISION NOT NULL,
                    pnl DOUBLE PRECISION DEFAULT 0,
                    pnl_pct DOUBLE PRECISION DEFAULT 0,
                    entry_ts TIMESTAMPTZ NOT NULL,
                    exit_ts TIMESTAMPTZ,
                    regime_at_entry TEXT,
                    confidence DOUBLE PRECISION DEFAULT 1.0,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_source ON shadow_trades(signal_source);
                CREATE INDEX IF NOT EXISTS idx_shadow_entry_ts ON shadow_trades(entry_ts);
                CREATE INDEX IF NOT EXISTS idx_shadow_exit_ts ON shadow_trades(exit_ts);
                CREATE TABLE IF NOT EXISTS shadow_attribution (
                    id BIGSERIAL PRIMARY KEY,
                    date TEXT NOT NULL,
                    signal_source TEXT NOT NULL,
                    trade_count INTEGER DEFAULT 0,
                    total_pnl DOUBLE PRECISION DEFAULT 0,
                    avg_pnl_pct DOUBLE PRECISION DEFAULT 0,
                    win_rate DOUBLE PRECISION DEFAULT 0,
                    best_trade_pnl DOUBLE PRECISION,
                    worst_trade_pnl DOUBLE PRECISION,
                    sharpe_proxy DOUBLE PRECISION DEFAULT 0,
                    UNIQUE(date, signal_source)
                );
                CREATE INDEX IF NOT EXISTS idx_attribution_date ON shadow_attribution(date);
                CREATE INDEX IF NOT EXISTS idx_attribution_source ON shadow_attribution(signal_source);
                """)
            else:
                conn.executescript("""
                CREATE TABLE IF NOT EXISTS shadow_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_source TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    entry_ts TEXT NOT NULL,
                    exit_ts TEXT,
                    regime_at_entry TEXT,
                    confidence REAL DEFAULT 1.0,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_source ON shadow_trades(signal_source);
                CREATE INDEX IF NOT EXISTS idx_shadow_entry_ts ON shadow_trades(entry_ts);
                CREATE INDEX IF NOT EXISTS idx_shadow_exit_ts ON shadow_trades(exit_ts);
                CREATE TABLE IF NOT EXISTS shadow_attribution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    signal_source TEXT NOT NULL,
                    trade_count INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    avg_pnl_pct REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    best_trade_pnl REAL,
                    worst_trade_pnl REAL,
                    sharpe_proxy REAL DEFAULT 0,
                    UNIQUE(date, signal_source)
                );
                CREATE INDEX IF NOT EXISTS idx_attribution_date ON shadow_attribution(date);
                CREATE INDEX IF NOT EXISTS idx_attribution_source ON shadow_attribution(signal_source);
                """)
            logger.info("Shadow tracker tables ensured")

    def record_trade(self, trade_dict):
        """
        Record a completed paper trade with its signal source attribution.

        Args:
            trade_dict (dict): Trade record with keys:
                - signal_source (str): e.g., "strategy:momentum", "copy:golden_wallet"
                - coin (str): Trading pair (BTC, ETH, etc.)
                - side (str): "long" or "short"
                - entry_price (float): Entry execution price
                - exit_price (float): Exit execution price
                - size (float): Position size
                - entry_ts (str): ISO 8601 entry timestamp
                - exit_ts (str, optional): ISO 8601 exit timestamp
                - pnl (float, optional): P&L in absolute terms
                - pnl_pct (float, optional): P&L as percentage
                - regime_at_entry (str, optional): Market regime label
                - confidence (float, optional): Signal confidence 0-1
                - metadata (dict, optional): Extra context
        """
        try:
            # Compute PnL if not provided
            pnl = trade_dict.get("pnl")
            pnl_pct = trade_dict.get("pnl_pct")

            if pnl is None and "entry_price" in trade_dict and "exit_price" in trade_dict:
                entry = trade_dict["entry_price"]
                exit_p = trade_dict["exit_price"]
                size = trade_dict.get("size", 1)
                side = trade_dict.get("side", "long")

                if side.lower() == "long":
                    pnl = (exit_p - entry) * size
                else:  # short
                    pnl = (entry - exit_p) * size

            if pnl_pct is None and pnl is not None:
                entry = trade_dict.get("entry_price", 0)
                size = trade_dict.get("size", 1)
                pnl_pct = (pnl / (entry * size)) * 100 if entry != 0 else 0

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO shadow_trades
                    (signal_source, coin, side, entry_price, exit_price, size,
                     pnl, pnl_pct, entry_ts, exit_ts, regime_at_entry, confidence, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_dict["signal_source"],
                    trade_dict["coin"],
                    trade_dict["side"],
                    trade_dict["entry_price"],
                    trade_dict.get("exit_price"),
                    trade_dict["size"],
                    pnl or 0,
                    pnl_pct or 0,
                    trade_dict.get("entry_ts", datetime.now(timezone.utc).isoformat()),
                    trade_dict.get("exit_ts", datetime.now(timezone.utc).isoformat()),
                    trade_dict.get("regime_at_entry"),
                    trade_dict.get("confidence", 1.0),
                    json.dumps(trade_dict.get("metadata", {}))
                ))

            logger.info(f"Recorded {trade_dict['signal_source']} {trade_dict['side']} "
                       f"{trade_dict['coin']} | PnL: {pnl:.2f}")
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            raise

    def get_attribution(self, days=30):
        """
        Get per-source PnL attribution for the last N days.

        Returns dict with signal sources as keys and metrics as values:
        {
            "strategy:momentum": {
                "trades": 12,
                "pnl": 450.0,
                "avg_pnl_pct": 1.2,
                "win_rate": 0.67,
                "best_trade": 120.0,
                "worst_trade": -50.0,
                "sharpe": 0.8
            },
            ...
        }
        """
        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT signal_source, pnl, pnl_pct
                    FROM shadow_trades
                    WHERE exit_ts IS NOT NULL AND exit_ts > ?
                    ORDER BY exit_ts DESC
                """, (cutoff_ts,)).fetchall()

            # Group by source
            by_source = {}
            for row in rows:
                source = row["signal_source"]
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append({
                    "pnl": row["pnl"],
                    "pnl_pct": row["pnl_pct"]
                })

            # Compute metrics per source
            attribution = {}
            for source, trades in by_source.items():
                pnls = [t["pnl"] for t in trades]
                pnl_pcts = [t["pnl_pct"] for t in trades]

                total_pnl = sum(pnls)
                avg_pnl_pct = mean(pnl_pcts) if pnl_pcts else 0
                win_count = sum(1 for p in pnls if p > 0)
                win_rate = win_count / len(pnls) if pnls else 0
                best_trade = max(pnls) if pnls else 0
                worst_trade = min(pnls) if pnls else 0
                sharpe = self.compute_sharpe_proxy(pnls)

                attribution[source] = {
                    "trades": len(pnls),
                    "pnl": round(total_pnl, 2),
                    "avg_pnl_pct": round(avg_pnl_pct, 3),
                    "win_rate": round(win_rate, 3),
                    "best_trade": round(best_trade, 2),
                    "worst_trade": round(worst_trade, 2),
                    "sharpe": round(sharpe, 3)
                }

            return attribution
        except Exception as e:
            logger.error(f"Error computing attribution: {e}")
            return {}

    def get_daily_pnl(self, days=30):
        """
        Get daily PnL per source for charting.

        Returns dict: {
            "2025-03-20": {
                "strategy:momentum": 150.5,
                "copy:golden": -45.0,
                ...
            },
            ...
        }
        """
        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT
                        date(exit_ts) as day,
                        signal_source,
                        SUM(pnl) as daily_pnl
                    FROM shadow_trades
                    WHERE exit_ts IS NOT NULL AND exit_ts > ?
                    GROUP BY day, signal_source
                    ORDER BY day DESC
                """, (cutoff_ts,)).fetchall()

            daily = {}
            for row in rows:
                day = row["day"]
                source = row["signal_source"]
                pnl = row["daily_pnl"]

                if day not in daily:
                    daily[day] = {}
                daily[day][source] = round(pnl, 2)

            return daily
        except Exception as e:
            logger.error(f"Error fetching daily PnL: {e}")
            return {}

    def get_source_rankings(self):
        """
        Rank signal sources by risk-adjusted return (Sharpe proxy).

        Returns list of (source, metrics) tuples sorted by Sharpe descending.
        """
        attribution = self.get_attribution(days=30)

        ranked = sorted(
            attribution.items(),
            key=lambda x: x[1]["sharpe"],
            reverse=True
        )

        return ranked

    def prune_old_data(self, keep_days=90):
        """
        Delete shadow trades older than N days.

        Args:
            keep_days (int): Keep records from last N days, delete older.

        Returns:
            dict: Counts of deleted records.
        """
        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()

        try:
            with self._get_connection() as conn:
                # Count before delete
                count_before = conn.execute(
                    "SELECT COUNT(*) as c FROM shadow_trades WHERE exit_ts < ?",
                    (cutoff_ts,)
                ).fetchone()["c"]

                # Delete old trades
                conn.execute(
                    "DELETE FROM shadow_trades WHERE exit_ts < ?",
                    (cutoff_ts,)
                )

                # Delete orphaned attribution records
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=keep_days)).date().isoformat()
                conn.execute(
                    "DELETE FROM shadow_attribution WHERE date < ?",
                    (cutoff_date,)
                )

            logger.info(f"Pruned {count_before} shadow trades older than {keep_days} days")
            return {"deleted_trades": count_before}
        except Exception as e:
            logger.error(f"Error pruning old data: {e}")
            return {"error": str(e)}

    def get_summary(self, days=30):
        """
        Get 30-day summary: total PnL, total trades, best/worst source, regime breakdown.

        Returns dict with aggregate metrics.
        """
        attribution = self.get_attribution(days=days)

        if not attribution:
            return {
                "total_pnl": 0,
                "total_trades": 0,
                "best_source": None,
                "worst_source": None,
                "avg_win_rate": 0,
                "sources": {}
            }

        total_pnl = sum(s["pnl"] for s in attribution.values())
        total_trades = sum(s["trades"] for s in attribution.values())
        avg_win_rate = mean([s["win_rate"] for s in attribution.values()]) if attribution else 0

        best_source = max(attribution.items(), key=lambda x: x[1]["sharpe"], default=(None, {}))[0]
        worst_source = min(attribution.items(), key=lambda x: x[1]["sharpe"], default=(None, {}))[0]

        return {
            "period_days": days,
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "best_source": best_source,
            "worst_source": worst_source,
            "avg_win_rate": round(avg_win_rate, 3),
            "sources": attribution
        }

    @staticmethod
    def compute_sharpe_proxy(pnl_list):
        """
        Compute a simple Sharpe ratio approximation.

        Approximates annualized Sharpe: mean(pnl) / std(pnl) * sqrt(252)
        Assumes ~252 trading days per year.

        Args:
            pnl_list (list): List of P&L values

        Returns:
            float: Sharpe approximation, or 0 if insufficient data
        """
        if not pnl_list or len(pnl_list) < 2:
            return 0.0

        try:
            avg_pnl = mean(pnl_list)
            std_pnl = stdev(pnl_list)

            if std_pnl == 0:
                return 0.0

            # MED-FIX MED-2: removed sqrt(252) annualisation — pnl_list contains
            # per-trade P&L values, not daily returns.  sqrt(252) is only valid for
            # a daily return series.  Return the raw per-trade information ratio
            # instead; comparisons across sources remain consistent and meaningful.
            sharpe = avg_pnl / std_pnl
            return sharpe
        except Exception:
            return 0.0

    def export_csv(self, filepath, days=30):
        """
        Export shadow trades to CSV for analysis.

        Args:
            filepath (str): Output CSV path
            days (int): Days to include
        """
        import csv

        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM shadow_trades
                    WHERE entry_ts > ?
                    ORDER BY entry_ts DESC
                """, (cutoff_ts,)).fetchall()

            if not rows:
                logger.warning("No trades to export")
                return

            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "signal_source", "coin", "side", "entry_price", "exit_price",
                    "size", "pnl", "pnl_pct", "entry_ts", "exit_ts", "regime_at_entry",
                    "confidence"
                ])
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))

            logger.info(f"Exported {len(rows)} trades to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise


if __name__ == "__main__":
    # Quick test
    tracker = ShadowTracker()

    # Log a test trade
    tracker.record_trade({
        "signal_source": "strategy:momentum",
        "coin": "BTC",
        "side": "long",
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "size": 0.1,
        "entry_ts": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        "exit_ts": datetime.now(timezone.utc).isoformat(),
        "confidence": 0.85
    })

    # Get summary
    summary = tracker.get_summary()
    print(json.dumps(summary, indent=2))
