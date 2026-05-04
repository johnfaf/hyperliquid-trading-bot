"""
Confidence Calibration Tracker
===============================
Measures whether our confidence scores are actually calibrated and
adjusts them when they aren't.

A well-calibrated source means:
  - When we say 0.7 confidence, we win ~70% of the time

Beyond the basic "predicted vs realized" check, this tracker also:
  1. Keys outcomes on (source, side, regime) so long/short and
     trend/range/crash regimes calibrate independently.
  2. Time-decays old outcomes (exp half-life) so calibration tracks
     the current strategy stack, not all-time history.
  3. Applies Bayesian shrinkage (Beta prior) for sparse bins.
  4. Enforces monotonicity across bins via Pool-Adjacent-Violators —
     a regularized isotonic calibration curve — once a source has
     enough outcomes.
  5. Reports both ECE and Brier score (Brier penalises lack of
     discrimination; a constant 0.5 predictor has perfect ECE but
     zero edge).
  6. Hard-gates: cold-start cap below MIN_OUTCOMES, source
     quarantine above QUARANTINE_ECE, global live-pause above
     LIVE_PAUSE_ECE.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict

from src.data import database as db

logger = logging.getLogger(__name__)

N_BINS = 10
_BETA_PRIOR_ALPHA = 2.0
_BETA_PRIOR_BETA = 2.0
_DEFAULT_HALF_LIFE_DAYS = 30.0
_DEFAULT_MIN_OUTCOMES = 30
_DEFAULT_COLDSTART_PRIOR = 0.50
_DEFAULT_ISOTONIC_MIN = 100
_DEFAULT_QUARANTINE_ECE = 0.25
_DEFAULT_QUARANTINE_MIN_SAMPLES = 50
_DEFAULT_LIVE_PAUSE_ECE = 0.50

_KEY_SEP = "|"
_REGIME_ANY = "any"
_SIDE_ANY = "_"


def bucket_regime(raw_regime: str) -> str:
    """Bucket a regime string into one of trend/range/crash/any.

    Per-regime calibration helps because the same source typically has
    very different reliability across regimes. We bucket coarsely
    (3 + any) so per-source samples don't fragment too aggressively.
    """
    r = (raw_regime or "").strip().lower()
    if not r or r in ("unknown", "none"):
        return "any"
    if "crash" in r or "panic" in r:
        return "crash"
    if "trend" in r or "bullish" in r or "bearish" in r or "momentum" in r:
        return "trend"
    if (
        "range" in r or "rang" in r or "neutral" in r
        or "chop" in r or "consolidation" in r or "sideways" in r
    ):
        return "range"
    return "any"


def compose_calibration_key(source: str, side: Optional[str] = None,
                            regime: Optional[str] = None) -> str:
    """Compose a (source, side, regime) calibration key.

    Side and regime default to wildcards so legacy callers that only have
    a raw source key get a consistent compound representation.
    """
    src = (source or "").strip() or "unknown"
    s = (side or "").strip().lower() or _SIDE_ANY
    if s in {"buy", "long"}:
        s = "long"
    elif s in {"sell", "short"}:
        s = "short"
    elif s != _SIDE_ANY:
        s = _SIDE_ANY
    r = (regime or _REGIME_ANY).strip().lower() or _REGIME_ANY
    return f"{src}{_KEY_SEP}{s}{_KEY_SEP}{r}"


def decompose_calibration_key(key: str) -> Tuple[str, str, str]:
    """Reverse compose_calibration_key. Legacy keys without separators
    are returned as ``(key, _SIDE_ANY, _REGIME_ANY)``.
    """
    if _KEY_SEP not in (key or ""):
        return (key or "unknown", _SIDE_ANY, _REGIME_ANY)
    parts = key.split(_KEY_SEP)
    if len(parts) == 2:
        return (parts[0], parts[1] or _SIDE_ANY, _REGIME_ANY)
    return (parts[0], parts[1] or _SIDE_ANY, parts[2] or _REGIME_ANY)


class CalibrationTracker:
    """Tracks prediction calibration across all signal sources.

    Sources are keyed on (source, side, regime). Records are bucketed
    into 10 confidence bins with exponential time decay applied to
    older outcomes.
    """

    def __init__(self, db_path: Optional[str] = None,
                 half_life_days: Optional[float] = None,
                 min_outcomes: Optional[int] = None,
                 coldstart_prior: Optional[float] = None,
                 isotonic_min_outcomes: Optional[int] = None,
                 quarantine_ece: Optional[float] = None,
                 quarantine_min_samples: Optional[int] = None,
                 live_pause_ece: Optional[float] = None):
        import config
        self.db_path = db_path or config.DB_PATH
        self._use_shared_db = self.db_path == config.DB_PATH
        self.half_life_days = float(
            half_life_days
            if half_life_days is not None
            else getattr(config, "CALIBRATION_HALF_LIFE_DAYS", _DEFAULT_HALF_LIFE_DAYS)
        )
        self.min_outcomes = int(
            min_outcomes
            if min_outcomes is not None
            else getattr(config, "CALIBRATION_MIN_OUTCOMES", _DEFAULT_MIN_OUTCOMES)
        )
        self.coldstart_prior = float(
            coldstart_prior
            if coldstart_prior is not None
            else getattr(config, "CALIBRATION_COLDSTART_PRIOR", _DEFAULT_COLDSTART_PRIOR)
        )
        self.isotonic_min_outcomes = int(
            isotonic_min_outcomes
            if isotonic_min_outcomes is not None
            else getattr(config, "CALIBRATION_ISOTONIC_MIN_OUTCOMES", _DEFAULT_ISOTONIC_MIN)
        )
        self.quarantine_ece = float(
            quarantine_ece
            if quarantine_ece is not None
            else getattr(config, "CALIBRATION_QUARANTINE_ECE", _DEFAULT_QUARANTINE_ECE)
        )
        self.quarantine_min_samples = int(
            quarantine_min_samples
            if quarantine_min_samples is not None
            else getattr(config, "CALIBRATION_QUARANTINE_MIN_SAMPLES",
                         _DEFAULT_QUARANTINE_MIN_SAMPLES)
        )
        self.live_pause_ece = float(
            live_pause_ece
            if live_pause_ece is not None
            else getattr(config, "CALIBRATION_LIVE_PAUSE_ECE", _DEFAULT_LIVE_PAUSE_ECE)
        )

        self._init_table()

        # In-memory bins: per source, per bin, weighted counts (floats).
        self._bins: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: {i: {"total": 0.0, "wins": 0.0} for i in range(N_BINS)}
        )
        # Brier accumulators (sum of weighted squared error and weight).
        self._brier: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"sse": 0.0, "weight": 0.0}
        )
        # Memoised PAV-fitted curves and the source size at fit time.
        self._curve_cache: Dict[str, List[float]] = {}
        self._curve_size_at_fit: Dict[str, float] = {}

        self._load_from_db()

        total_records = sum(
            b["total"] for bins in self._bins.values() for b in bins.values()
        )
        logger.info(
            "CalibrationTracker initialized with effective records=%.1f "
            "across %d source/side/regime keys (half_life=%.1fd)",
            total_records, len(self._bins), self.half_life_days,
        )

    # ── DB plumbing ────────────────────────────────────────────────
    def _init_table(self):
        try:
            if self._use_shared_db:
                with db.get_connection() as conn:
                    if db.get_backend_name() == "postgres":
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS calibration_records (
                                id BIGSERIAL PRIMARY KEY,
                                source_key TEXT NOT NULL,
                                predicted_confidence DOUBLE PRECISION NOT NULL,
                                actual_win INTEGER NOT NULL,
                                pnl DOUBLE PRECISION,
                                coin TEXT,
                                side TEXT,
                                timestamp TIMESTAMPTZ
                            )
                        """)
                    else:
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
            else:
                import sqlite3
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
        try:
            if self._use_shared_db:
                with db.get_connection(for_read=True) as conn:
                    rows = conn.execute(
                        "SELECT source_key, predicted_confidence, actual_win, "
                        "timestamp FROM calibration_records"
                    ).fetchall()
            else:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT source_key, predicted_confidence, actual_win, "
                    "timestamp FROM calibration_records"
                ).fetchall()
                conn.close()

            now = datetime.now(timezone.utc)
            for row in rows:
                key = row["source_key"]
                conf = float(row["predicted_confidence"])
                win = int(row["actual_win"])
                weight = self._weight_for_timestamp(row["timestamp"], now)
                self._apply_record(key, conf, win, weight, propagate_global=True)
        except Exception as e:
            logger.debug(f"Could not load calibration data: {e}")

    def _weight_for_timestamp(self, ts, now: datetime) -> float:
        if self.half_life_days <= 0 or ts is None:
            return 1.0
        try:
            if isinstance(ts, datetime):
                t = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            else:
                s = str(ts)
                # SQLite stores ISO-8601; tolerate "Z" suffix and naive datetimes.
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                t = datetime.fromisoformat(s)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
        except Exception:
            return 1.0
        age_days = max(0.0, (now - t).total_seconds() / 86400.0)
        return math.pow(0.5, age_days / self.half_life_days)

    # ── Recording ──────────────────────────────────────────────────
    def record(self, source_key: str, predicted_confidence: float,
               actual_win: bool, pnl: float = 0, coin: str = "",
               side: str = "", regime: Optional[str] = None):
        """Record a prediction outcome for calibration tracking.

        ``source_key`` may be a raw source name or an already-composed
        ``source|side|regime`` key. If ``side`` or ``regime`` are passed
        and ``source_key`` is not already composed, they are folded in
        so legacy callers automatically get per-side/per-regime keying.
        """
        conf = max(0.0, min(float(predicted_confidence), 1.0))
        win = 1 if actual_win else 0
        composed = self._normalize_key(source_key, side=side, regime=regime)

        # Live records carry no time discount. Prior records are weighted
        # at load time.
        self._apply_record(composed, conf, win, 1.0, propagate_global=True)
        self._invalidate_curve(composed)

        try:
            ts_iso = datetime.now(timezone.utc).isoformat()
            if self._use_shared_db:
                with db.get_connection() as conn:
                    conn.execute(
                        "INSERT INTO calibration_records "
                        "(source_key, predicted_confidence, actual_win, pnl, "
                        "coin, side, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (composed, conf, win, pnl, coin, side, ts_iso),
                    )
            else:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    "INSERT INTO calibration_records "
                    "(source_key, predicted_confidence, actual_win, pnl, "
                    "coin, side, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (composed, conf, win, pnl, coin, side, ts_iso),
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.debug(f"Could not save calibration record: {e}")

    def _normalize_key(self, source_key: str, *, side: Optional[str] = None,
                       regime: Optional[str] = None) -> str:
        if _KEY_SEP in (source_key or ""):
            # Already composed -- trust the caller.
            return source_key
        return compose_calibration_key(source_key, side, regime)

    def _apply_record(self, composed_key: str, conf: float, win: int,
                      weight: float, *, propagate_global: bool) -> None:
        bin_idx = min(int(conf * N_BINS), N_BINS - 1)
        keys = [composed_key]
        if propagate_global and composed_key != "global":
            keys.append("global")
        for key in keys:
            self._bins[key][bin_idx]["total"] += weight
            if win:
                self._bins[key][bin_idx]["wins"] += weight
            err = (conf - float(win))
            self._brier[key]["sse"] += weight * err * err
            self._brier[key]["weight"] += weight

    def _invalidate_curve(self, composed_key: str) -> None:
        # Refit lazily; pop both the source and the global aggregate.
        self._curve_cache.pop(composed_key, None)
        self._curve_size_at_fit.pop(composed_key, None)
        self._curve_cache.pop("global", None)
        self._curve_size_at_fit.pop("global", None)

    # ── Lookups ────────────────────────────────────────────────────
    def _resolve_key(self, source_key: str, *, side: Optional[str] = None,
                     regime: Optional[str] = None) -> str:
        """Return the most specific stored key matching ``(source, side, regime)``.

        We try (source|side|regime), then (source|side|any), then
        (source|*|any), then "global". This lets callers ask the
        tracker by raw source and still get the most specific calibration
        available without knowing the storage layout.
        """
        if _KEY_SEP in (source_key or "") or source_key in self._bins:
            return source_key
        candidates: List[str] = []
        if side and regime:
            candidates.append(compose_calibration_key(source_key, side, regime))
        if side:
            candidates.append(compose_calibration_key(source_key, side, _REGIME_ANY))
        candidates.append(compose_calibration_key(source_key, None, _REGIME_ANY))
        # Legacy raw key fallback.
        candidates.append(source_key)
        for cand in candidates:
            if cand in self._bins and self._source_total(cand) > 0:
                return cand
        return source_key

    def _source_total(self, key: str) -> float:
        bins = self._bins.get(key, {})
        return float(sum(b.get("total", 0.0) for b in bins.values()))

    def get_calibration_curve(self, source_key: str = "global", *,
                              shrinkage: bool = True,
                              monotone: Optional[bool] = None) -> List[Dict]:
        """Return the calibration curve for a source.

        With ``shrinkage`` enabled (default), each bin's actual win rate
        is smoothed by a Beta(α, β) prior — important for sparse bins.
        With ``monotone`` enabled (defaults to True once the source has
        ``isotonic_min_outcomes`` records), Pool-Adjacent-Violators is
        applied so the curve is non-decreasing.
        """
        key = source_key
        bins = self._bins.get(key, {})
        total = self._source_total(key)
        if monotone is None:
            monotone = total >= self.isotonic_min_outcomes

        rows: List[Tuple[float, float, float, float]] = []  # bin_center, raw_wr, smoothed_wr, count
        for i in range(N_BINS):
            bin_data = bins.get(i, {"total": 0.0, "wins": 0.0})
            n = float(bin_data.get("total", 0.0))
            wins = float(bin_data.get("wins", 0.0))
            bin_center = (i + 0.5) / N_BINS
            raw_wr = (wins / n) if n > 0 else 0.0
            if shrinkage:
                smoothed = (wins + _BETA_PRIOR_ALPHA) / (n + _BETA_PRIOR_ALPHA + _BETA_PRIOR_BETA)
            else:
                smoothed = raw_wr
            rows.append((bin_center, raw_wr, smoothed, n))

        if monotone:
            smoothed_seq = [r[2] for r in rows]
            weights = [max(r[3], 1e-9) for r in rows]
            smoothed_seq = _pool_adjacent_violators(smoothed_seq, weights)
            rows = [(c, raw, sm, n) for (c, raw, _, n), sm in zip(rows, smoothed_seq)]

        return [
            {
                "bin_center": round(c, 2),
                "predicted": round(c, 2),
                "actual_win_rate": round(raw, 3),
                "smoothed_win_rate": round(sm, 3),
                "count": round(n, 2),
            }
            for c, raw, sm, n in rows
        ]

    def get_ece(self, source_key: str = "global") -> Optional[float]:
        bins = self._bins.get(source_key, {})
        total = float(sum(b.get("total", 0.0) for b in bins.values()))
        if total <= 0:
            return None
        ece = 0.0
        for i in range(N_BINS):
            bin_data = bins.get(i, {"total": 0.0, "wins": 0.0})
            n = float(bin_data.get("total", 0.0))
            if n <= 0:
                continue
            predicted = (i + 0.5) / N_BINS
            actual = bin_data.get("wins", 0.0) / n
            ece += (n / total) * abs(predicted - actual)
        return round(ece, 4)

    def get_brier(self, source_key: str = "global") -> Optional[float]:
        """Return the time-weighted Brier score for a source.

        Brier = E[(confidence - outcome)^2]. Lower is better; 0 is
        perfect. Unlike ECE, Brier penalises lack of discrimination
        — a constant 0.5 predictor scores 0.25 here even with perfect
        ECE.
        """
        bucket = self._brier.get(source_key)
        if not bucket or bucket["weight"] <= 0:
            return None
        return round(bucket["sse"] / bucket["weight"], 4)

    def get_sample_size(self, source_key: str = "global") -> float:
        return self._source_total(source_key)

    # ── Adjustment logic ──────────────────────────────────────────
    def get_adjustment_factor(self, source_key: str,
                              predicted_confidence: float, *,
                              side: Optional[str] = None,
                              regime: Optional[str] = None) -> float:
        """Return a calibrated confidence in [0.05, 0.95].

        Decision rules:
          * Below ``min_outcomes``: return ``min(predicted, coldstart_prior)``
            so an uncalibrated source cannot emit aggressive confidences.
          * Below ``isotonic_min_outcomes``: blend predicted with the
            shrinkage-smoothed bin rate, weighted by sample size in
            that bin.
          * At/above ``isotonic_min_outcomes``: snap to the PAV-monotone
            calibrated curve, with linear interpolation between bin
            centers.
        """
        conf = max(0.0, min(float(predicted_confidence), 1.0))
        key = self._resolve_key(source_key, side=side, regime=regime)
        total = self._source_total(key)
        if total <= 0 and key != "global":
            key = "global"
            total = self._source_total(key)

        if total < self.min_outcomes:
            return float(min(conf, self.coldstart_prior))

        curve = self._fit_curve(key)
        if total >= self.isotonic_min_outcomes and curve is not None:
            adjusted = _interpolate_curve(curve, conf)
        else:
            # Shrinkage blend toward the smoothed bin rate, weighted by
            # how informative the bin is.
            bin_idx = min(int(conf * N_BINS), N_BINS - 1)
            bins = self._bins.get(key, {})
            bd = bins.get(bin_idx, {"total": 0.0, "wins": 0.0})
            n = float(bd.get("total", 0.0))
            wins = float(bd.get("wins", 0.0))
            smoothed = (wins + _BETA_PRIOR_ALPHA) / (n + _BETA_PRIOR_ALPHA + _BETA_PRIOR_BETA)
            # As bin sample size grows, lean harder on the empirical rate.
            shrink_n = 10.0
            w_emp = n / (n + shrink_n)
            adjusted = w_emp * smoothed + (1.0 - w_emp) * conf

        return float(max(0.05, min(adjusted, 0.95)))

    def _fit_curve(self, source_key: str) -> Optional[List[float]]:
        bins = self._bins.get(source_key)
        if not bins:
            return None
        total = self._source_total(source_key)
        last_total = self._curve_size_at_fit.get(source_key)
        # Refit when total moves by 10% or 20 records, whichever is smaller.
        if (
            source_key in self._curve_cache
            and last_total is not None
            and abs(total - last_total) < min(20.0, 0.10 * max(last_total, 1.0))
        ):
            return self._curve_cache[source_key]

        smoothed = []
        weights = []
        for i in range(N_BINS):
            bd = bins.get(i, {"total": 0.0, "wins": 0.0})
            n = float(bd.get("total", 0.0))
            wins = float(bd.get("wins", 0.0))
            smoothed.append(
                (wins + _BETA_PRIOR_ALPHA) / (n + _BETA_PRIOR_ALPHA + _BETA_PRIOR_BETA)
            )
            weights.append(max(n, 1e-9))
        curve = _pool_adjacent_violators(smoothed, weights)
        self._curve_cache[source_key] = curve
        self._curve_size_at_fit[source_key] = total
        return curve

    def get_reliability_multiplier(self, source_key: str = "global") -> float:
        """Return a confidence-derisk multiplier based on calibration error.

        We require a minimum sample size before trusting a per-source
        ECE; below that we fall back to the global ECE rather than
        reading noise.
        """
        total = self._source_total(source_key)
        ece = self.get_ece(source_key) if total >= max(self.min_outcomes * 3, 100) else None
        if ece is None and source_key != "global":
            ece = self.get_ece("global")
        if ece is None:
            return 1.0
        if ece >= 0.35:
            return 0.65
        if ece >= 0.25:
            return 0.75
        if ece >= 0.20:
            return 0.85
        return 1.0

    # ── Operator / governance helpers ─────────────────────────────
    def is_quarantined(self, source_key: str, *,
                       side: Optional[str] = None,
                       regime: Optional[str] = None) -> bool:
        """Return True if a source has reached the auto-quarantine bar.

        Sources at or above ``quarantine_ece`` with at least
        ``quarantine_min_samples`` outcomes should be routed to shadow
        only until calibration recovers.
        """
        key = self._resolve_key(source_key, side=side, regime=regime)
        total = self._source_total(key)
        if total < self.quarantine_min_samples:
            return False
        ece = self.get_ece(key)
        if ece is None:
            return False
        return ece >= self.quarantine_ece

    def is_live_paused(self) -> bool:
        """Return True if the global calibrator says live trading is unsafe."""
        ece = self.get_ece("global")
        if ece is None:
            return False
        return ece >= self.live_pause_ece

    def get_quarantined_sources(self) -> List[Dict]:
        """List sources currently above the auto-quarantine bar."""
        out: List[Dict] = []
        for key in self._bins:
            if key == "global":
                continue
            total = self._source_total(key)
            if total < self.quarantine_min_samples:
                continue
            ece = self.get_ece(key)
            if ece is None or ece < self.quarantine_ece:
                continue
            source, side, regime = decompose_calibration_key(key)
            out.append({
                "source_key": key,
                "source": source,
                "side": side,
                "regime": regime,
                "ece": ece,
                "brier": self.get_brier(key),
                "samples": round(total, 2),
            })
        out.sort(key=lambda r: (-r["ece"], -r["samples"]))
        return out

    def get_all_stats(self) -> Dict[str, Dict]:
        stats: Dict[str, Dict] = {}
        for key in self._bins:
            total = self._source_total(key)
            if total <= 0:
                continue
            ece = self.get_ece(key)
            stats[key] = {
                "total_records": round(total, 2),
                "ece": ece,
                "brier": self.get_brier(key),
                "calibration_quality": self._quality_label(ece),
                "quarantined": self.is_quarantined(key),
            }
        return stats

    def _quality_label(self, ece) -> str:
        if ece is None:
            return "cold start (no data)"
        if ece < 0.05:
            return "excellent"
        if ece < 0.10:
            return "good"
        if ece < 0.20:
            return "fair"
        return "poor"


# ── Helpers (module level so they're easy to test) ────────────────
def _pool_adjacent_violators(values: List[float], weights: List[float]) -> List[float]:
    """Weighted Pool-Adjacent-Violators — produce a non-decreasing
    sequence that minimises weighted squared error vs ``values``.

    Pure Python so we don't take a sklearn dependency for a 30-line
    algorithm. Stable for ties; respects per-bin weights.
    """
    n = len(values)
    if n == 0:
        return []
    means = [float(v) for v in values]
    ws = [max(float(w), 1e-12) for w in weights]
    # Block boundaries: each block stores (mean, weight, length).
    blocks: List[List[float]] = [[means[i], ws[i], 1.0] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0]:
            i += 1
            continue
        # Merge blocks i and i+1.
        m1, w1, l1 = blocks[i]
        m2, w2, l2 = blocks[i + 1]
        merged_w = w1 + w2
        merged_m = (m1 * w1 + m2 * w2) / merged_w if merged_w > 0 else (m1 + m2) / 2.0
        blocks[i] = [merged_m, merged_w, l1 + l2]
        del blocks[i + 1]
        # Walk back to fix earlier violations.
        while i > 0 and blocks[i - 1][0] > blocks[i][0]:
            m1, w1, l1 = blocks[i - 1]
            m2, w2, l2 = blocks[i]
            merged_w = w1 + w2
            merged_m = (m1 * w1 + m2 * w2) / merged_w if merged_w > 0 else (m1 + m2) / 2.0
            blocks[i - 1] = [merged_m, merged_w, l1 + l2]
            del blocks[i]
            i -= 1
    out: List[float] = []
    for m, _w, l in blocks:
        out.extend([m] * int(l))
    return out


def _interpolate_curve(curve: List[float], conf: float) -> float:
    """Linearly interpolate between bin centers on a calibration curve.

    ``curve[i]`` is the calibrated probability at bin center
    ``(i + 0.5) / N_BINS``.
    """
    if not curve:
        return float(conf)
    n = len(curve)
    x = max(0.0, min(float(conf), 1.0)) * n - 0.5
    i_lo = int(math.floor(x))
    i_hi = i_lo + 1
    if i_lo < 0:
        return float(curve[0])
    if i_hi >= n:
        return float(curve[-1])
    frac = x - i_lo
    return float(curve[i_lo] * (1.0 - frac) + curve[i_hi] * frac)
