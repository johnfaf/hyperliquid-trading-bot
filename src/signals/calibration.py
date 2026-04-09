"""
Confidence Calibration Tracker
===============================
Measures whether our confidence scores are actually calibrated.

A well-calibrated system means:
  - When we say 0.7 confidence, we win ~70% of the time
  - When we say 0.5 confidence, we win ~50% of the time

If our 0.7 confidence signals only win 40% of the time,
our confidence scores are garbage and need correction.

This module:
  1. Tracks predicted confidence vs actual outcomes
  2. Builds calibration curves (predicted vs realized)
  3. Computes calibration error (ECE — Expected Calibration Error)
  4. Provides adjustment factors to correct miscalibrated confidence
"""
import logging
import json
import sqlite3
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)

# Number of bins for calibration curve
N_BINS = 10  # 0-0.1, 0.1-0.2, ..., 0.9-1.0


class CalibrationTracker:
    """
    Tracks prediction calibration across all signal sources.

    Answers the question: "When we predict X% confidence, do we actually win X% of the time?"
    """

    def __init__(self, db_path: Optional[str] = None):
        import config
        self.db_path = db_path or config.DB_PATH
        self._init_table()

        # In-memory bins for fast computation
        # Key: source_key or "global", Value: {bin_idx: {"total": N, "wins": N}}
        self._bins: Dict[str, Dict[int, Dict]] = defaultdict(
            lambda: {i: {"total": 0, "wins": 0} for i in range(N_BINS)}
        )
        self._load_from_db()

        total_records = sum(
            b["total"] for bins in self._bins.values() for b in bins.values()
        )
        logger.info(f"CalibrationTracker initialized with {total_records} records")

    def _init_table(self):
        """Create calibration tracking table."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL,
                    predicted_confidence REAL NOT NULL,
                    actual_win INTEGER NOT NULL,
                    pnl REAL,
                    coin TEXT,
                    side TEXT,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calibration_source
                ON calibration_records(source_key)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not init calibration table: {e}")

    def _load_from_db(self):
        """Load existing calibration data from DB into bins."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT source_key, predicted_confidence, actual_win FROM calibration_records"
            ).fetchall()
            conn.close()

            for row in rows:
                source = row["source_key"]
                conf = row["predicted_confidence"]
                win = row["actual_win"]
                bin_idx = min(int(conf * N_BINS), N_BINS - 1)

                if source not in self._bins:
                    self._bins[source] = {i: {"total": 0, "wins": 0} for i in range(N_BINS)}

                self._bins[source][bin_idx]["total"] += 1
                if win:
                    self._bins[source][bin_idx]["wins"] += 1

                # Also update global
                if "global" not in self._bins:
                    self._bins["global"] = {i: {"total": 0, "wins": 0} for i in range(N_BINS)}
                self._bins["global"][bin_idx]["total"] += 1
                if win:
                    self._bins["global"][bin_idx]["wins"] += 1

        except Exception as e:
            logger.debug(f"Could not load calibration data: {e}")

    def record(self, source_key: str, predicted_confidence: float,
                actual_win: bool, pnl: float = 0, coin: str = "", side: str = ""):
        """
        Record a prediction outcome for calibration tracking.

        Args:
            source_key: Signal source (e.g. "strategy:momentum_long")
            predicted_confidence: What we predicted (0-1)
            actual_win: Did the trade actually win?
            pnl: Actual PnL
            coin: Trading pair
            side: Trade direction
        """
        # Clamp confidence to valid range
        conf = max(0.0, min(predicted_confidence, 1.0))
        win = 1 if actual_win else 0
        bin_idx = min(int(conf * N_BINS), N_BINS - 1)

        # Update in-memory bins
        for key in [source_key, "global"]:
            if key not in self._bins:
                self._bins[key] = {i: {"total": 0, "wins": 0} for i in range(N_BINS)}
            self._bins[key][bin_idx]["total"] += 1
            if actual_win:
                self._bins[key][bin_idx]["wins"] += 1

        # Persist to DB
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO calibration_records
                (source_key, predicted_confidence, actual_win, pnl, coin, side, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source_key, conf, win, pnl, coin, side, datetime.now(timezone.utc).isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Could not save calibration record: {e}")

    def get_calibration_curve(self, source_key: str = "global") -> List[Dict]:
        """
        Get the calibration curve for a source.

        Returns list of {bin_center, predicted, actual_win_rate, count}
        """
        bins = self._bins.get(source_key, {})
        curve = []

        for i in range(N_BINS):
            bin_data = bins.get(i, {"total": 0, "wins": 0})
            total = bin_data["total"]
            wins = bin_data["wins"]

            bin_center = (i + 0.5) / N_BINS  # e.g. 0.05, 0.15, ..., 0.95
            actual_wr = wins / total if total > 0 else 0

            curve.append({
                "bin_center": round(bin_center, 2),
                "predicted": round(bin_center, 2),
                "actual_win_rate": round(actual_wr, 3),
                "count": total,
            })

        return curve

    def get_ece(self, source_key: str = "global") -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = sum over bins of: (n_bin / n_total) * |predicted - actual|

        Lower is better. 0 = perfectly calibrated. > 0.1 = poorly calibrated.
        """
        bins = self._bins.get(source_key, {})
        total_samples = sum(b.get("total", 0) for b in bins.values())

        if total_samples == 0:
            return None  # No data — don't pretend we're calibrated

        ece = 0.0
        for i in range(N_BINS):
            bin_data = bins.get(i, {"total": 0, "wins": 0})
            n = bin_data["total"]
            if n == 0:
                continue

            predicted = (i + 0.5) / N_BINS
            actual = bin_data["wins"] / n
            ece += (n / total_samples) * abs(predicted - actual)

        return round(ece, 4)

    def get_adjustment_factor(self, source_key: str,
                                predicted_confidence: float) -> float:
        """
        Get an adjustment factor to correct miscalibrated confidence.

        If our 0.7 predictions actually win 50%, the adjusted confidence
        should be closer to 0.5.

        Returns adjusted confidence (0-1).
        """
        bins = self._bins.get(source_key, self._bins.get("global", {}))
        bin_idx = min(int(predicted_confidence * N_BINS), N_BINS - 1)
        bin_data = bins.get(bin_idx, {"total": 0, "wins": 0})

        if bin_data["total"] < 10:
            # Not enough data in this bin — return original
            return predicted_confidence

        actual_wr = bin_data["wins"] / bin_data["total"]

        # Blend: 60% actual calibrated rate + 40% original prediction
        # This prevents over-correction when sample size is small
        adjusted = actual_wr * 0.6 + predicted_confidence * 0.4

        return max(0.05, min(adjusted, 0.95))

    def get_all_stats(self) -> Dict:
        """Get calibration stats for all tracked sources."""
        stats = {}
        for source_key in self._bins:
            bins = self._bins[source_key]
            total = sum(b["total"] for b in bins.values())
            if total == 0:
                continue
            stats[source_key] = {
                "total_records": total,
                "ece": self.get_ece(source_key),
                "calibration_quality": self._quality_label(self.get_ece(source_key)),
            }
        return stats

    def _quality_label(self, ece) -> str:
        """Human-readable calibration quality."""
        if ece is None:
            return "cold start (no data)"
        if ece < 0.05:
            return "excellent"
        elif ece < 0.10:
            return "good"
        elif ece < 0.20:
            return "fair"
        else:
            return "poor"
