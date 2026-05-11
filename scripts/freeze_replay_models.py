#!/usr/bin/env python3
"""Train + freeze ML model artifacts for use inside the replay harness.

The bot's ML layer in production is two components:

  1. XGBoostRegimeForecaster (src/signals/xgboost_regime_forecaster.py)
     - Multi-class classifier over {crash, neutral, bullish}
     - Features: funding_rate, funding_slope, orderbook_imbalance,
       volatility_5m, basis_spread, options_flow_conviction
     - Trained walk-forward on the `regime_history` table that the bot
       accumulates each cycle.
     - Artifact: models/regime_xgboost.json + .meta.json

  2. AlphaPipeline (src/signals/feature_store_alpha.py)
     - Reads from Postgres `feature_store`
     - Trains its own walk-forward classifier
     - Artifact: depends on its config

The replay harness has these excluded from REPLAY_PROFILE by default --
running them as-is would either (a) use a model trained on data that
overlaps the replay window (look-ahead!), or (b) retrain mid-replay
using state that didn't exist at replay-time.

This script implements the audit + the training-data scaffolding for
freezing a model with cutoff < replay-start. It does NOT call into
the production XGBoost training -- the bot's training reads from the
DB's `regime_history` table, which is what the bot would have observed
*live*. Reproducing it offline means generating that table from
historical candles via the rule-based regime detector.

What this script does:
  --audit          Print which inputs each model needs + what's
                   available in the cache, and report whether a frozen
                   train would be possible right now.
  --generate-history
                   Walk through `data/candle_cache.db` for a date range,
                   run RegimeDetector at each step, write rows to a
                   replay_regime_history.db. This produces the training
                   data the XGBoost model needs (without ever pulling
                   live data).
  --train-xgboost  Take a replay_regime_history.db produced above and
                   train an XGBoost model with --cutoff. Artifact is
                   saved to models/regime_xgboost_replay_<cutoff>.json.

After training, add `"xgboost_forecaster"` to REPLAY_PROFILE and
configure the model_path to point at your frozen artifact, then re-run
the replay harness.

Honest caveats:
  - The regime_history table the bot accumulates in production reflects
    live multi-exchange features (funding, orderbook, options) that we
    cannot reproduce historically from candles alone. The synthesized
    history is therefore a degraded approximation of what the real
    history was. Replay with this model is "what would the bot have
    done IF its ML signal was derived from rule-based regime labels."
  - For a truly faithful replay you'd need a snapshot of the live bot's
    regime_history from a date < replay-start. If you have one, dump it
    and pass --import-regime-history INSTEAD of --generate-history.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("freeze_models")

DEFAULT_CACHE_DB = "data/candle_cache.db"


def _cache_coverage(cache_db: str) -> Dict[str, Dict[str, int]]:
    """Inspect the candle cache. Returns {coin: {timeframe: (count, first_ms, last_ms)}}."""
    if not os.path.exists(cache_db):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    with sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True) as conn:
        rows = conn.execute("""
            SELECT coin, timeframe, COUNT(*), MIN(timestamp_ms), MAX(timestamp_ms)
            FROM candles
            GROUP BY coin, timeframe
        """).fetchall()
    for coin, tf, n, mn, mx in rows:
        out.setdefault(coin, {})[tf] = {"count": n, "first_ms": mn, "last_ms": mx}
    return out


def cmd_audit(args: argparse.Namespace) -> int:
    """Report what model training would require + what's available."""
    print("=" * 78)
    print("  ML MODEL TRAINING AUDIT")
    print("=" * 78)
    print()
    print("Models the production bot uses:")
    print("  1. XGBoostRegimeForecaster")
    print("     features: funding_rate, funding_slope, orderbook_imbalance,")
    print("               volatility_5m, basis_spread, options_flow_conviction")
    print("     training data: rows from the `regime_history` table")
    print("     artifact:     models/regime_xgboost.json")
    print()
    print("  2. AlphaPipeline (feature_store_alpha.py)")
    print("     features: Postgres feature_store (alpha_features schema)")
    print("     training data: live feature_store rows")
    print("     artifact:     model bytes in alpha_pipeline state")
    print()
    print("Available locally:")
    coverage = _cache_coverage(args.cache_db)
    if not coverage:
        print(f"  candle cache: NOT FOUND at {args.cache_db}")
    else:
        for coin in sorted(coverage):
            tfs = coverage[coin]
            for tf in sorted(tfs):
                m = tfs[tf]
                first = datetime.fromtimestamp(m["first_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                last = datetime.fromtimestamp(m["last_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"  {coin} {tf}: {m['count']:>10,} rows  ({first} -> {last})")

    print()
    rh = "data/replay_regime_history.db"
    if os.path.exists(rh):
        with sqlite3.connect(f"file:{rh}?mode=ro", uri=True) as conn:
            n = conn.execute("SELECT COUNT(*) FROM regime_history").fetchone()[0]
        print(f"  replay regime_history: {n:,} rows ({rh})")
    else:
        print(f"  replay regime_history: NOT GENERATED. Run --generate-history.")

    prod_rh = "data/bot.db"
    if os.path.exists(prod_rh):
        try:
            with sqlite3.connect(f"file:{prod_rh}?mode=ro", uri=True) as conn:
                n = conn.execute("SELECT COUNT(*) FROM regime_history").fetchone()[0]
            print(f"  prod  regime_history: {n:,} rows ({prod_rh})")
        except sqlite3.OperationalError:
            print(f"  prod  regime_history: table missing")

    artifact = "models/regime_xgboost.json"
    if os.path.exists(artifact):
        mtime = datetime.fromtimestamp(os.path.getmtime(artifact), tz=timezone.utc)
        print(f"  models/regime_xgboost.json: exists, mtime={mtime.isoformat()}")
    else:
        print(f"  models/regime_xgboost.json: missing")

    print()
    print("Verdict:")
    print("  - XGBoost training is possible iff replay_regime_history.db has")
    print("    >= ~50 rows. Generate with --generate-history.")
    print("  - Alpha pipeline training needs a Postgres feature_store -- skipped")
    print("    by default in REPLAY_PROFILE.")
    print()
    print("Reproducibility note:")
    print("  Synthesized regime_history is derived from RULE-BASED labels on")
    print("  cached candles. Replay with such a model represents 'what the bot")
    print("  would have done IF its ML used rule-based regime labels' -- not")
    print("  what the bot actually did historically. For the second, you need")
    print("  a snapshot of the production regime_history from before replay-start.")
    print("=" * 78)
    return 0


def cmd_generate_history(args: argparse.Namespace) -> int:
    """Walk through cached candles, run the rule-based regime detector,
    and persist rows to replay_regime_history.db. Output table is shaped
    like the bot's `regime_history` table so XGBoost training can read it.
    """
    if not os.path.exists(args.cache_db):
        logger.error("Cache not found: %s", args.cache_db)
        return 1

    from src.analysis.regime_detector import RegimeDetector
    from src.backtest.replay.clock import ReplayClock
    from src.backtest.replay.candle_oracle import CandleOracle
    from src.backtest.replay.api_manager_shim import (
        ReplayAPIManager, install_replay_manager, uninstall_replay_manager,
    )
    from src.core import clock_provider

    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    step_ms = 3_600_000  # 1h sample rate -- matches production cycle cadence

    logger.info("Generating regime_history rows from %s to %s (1h ticks)", args.start, args.end)

    # Build a minimal replay environment (no full container -- we only need
    # the regime detector + candle access).
    clk = ReplayClock(start_ts_ms=start_ms, label="history-gen")
    prev = clock_provider.install(clk)
    oracle = CandleOracle(args.cache_db, clk)
    api = ReplayAPIManager(oracle, clk,
                           known_coins=oracle.available_coins("1h") or ["BTC"])
    install_replay_manager(api)

    rows_to_insert = []
    try:
        det = RegimeDetector()
        coins = args.coins.split(",") if args.coins else ["BTC"]
        t = start_ms
        i = 0
        while t < end_ms:
            clk.set(t)
            for coin in coins:
                try:
                    state = det.detect_regime(coin)
                except Exception as e:
                    logger.debug("regime detect failed for %s @ %d: %s", coin, t, e)
                    continue
                rows_to_insert.append({
                    "coin": coin,
                    "timestamp_ms": t,
                    "regime_label": _regime_to_label(state.regime.value
                                                    if hasattr(state.regime, "value")
                                                    else state.regime),
                    "confidence": float(state.confidence or 0.0),
                })
            t += step_ms
            i += 1
            if i % 100 == 0:
                logger.info("  generated %d ticks (%d rows so far)", i, len(rows_to_insert))
    finally:
        uninstall_replay_manager()
        clock_provider.restore(prev)

    out_db = args.history_db
    Path(out_db).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(out_db) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                coin TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                regime_label INTEGER NOT NULL,
                confidence REAL,
                PRIMARY KEY (coin, timestamp_ms)
            )
        """)
        conn.executemany(
            "INSERT OR IGNORE INTO regime_history (coin, timestamp_ms, regime_label, confidence) VALUES (?, ?, ?, ?)",
            [(r["coin"], r["timestamp_ms"], r["regime_label"], r["confidence"]) for r in rows_to_insert],
        )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM regime_history").fetchone()[0]

    logger.info("Wrote %d rows (total: %d) to %s", len(rows_to_insert), n, out_db)
    return 0


def _regime_to_label(regime_str: str) -> int:
    """Map RegimeState's value (string) to XGBoost label.

    Production XGBoost uses {crash=0, neutral=1, bullish=2}. The regime
    detector emits more nuanced regimes (trending_up, trending_down,
    ranging, choppy, etc.). Crude collapse for synthetic labels:
    """
    s = str(regime_str).lower()
    if "down" in s or "crash" in s:
        return 0
    if "up" in s or "bull" in s:
        return 2
    return 1


def cmd_train_xgboost(args: argparse.Namespace) -> int:
    """Train XGBoost on a replay_regime_history.db with --cutoff."""
    try:
        import xgboost as xgb  # noqa: F401
    except ImportError:
        logger.error("xgboost not installed (`pip install xgboost`)")
        return 1

    if not os.path.exists(args.history_db):
        logger.error("History DB not found: %s -- run --generate-history first", args.history_db)
        return 1

    cutoff_ms = int(datetime.strptime(args.cutoff, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)

    import numpy as np
    import xgboost as xgb

    with sqlite3.connect(f"file:{args.history_db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT timestamp_ms, regime_label, confidence FROM regime_history "
            "WHERE timestamp_ms < ? ORDER BY timestamp_ms",
            (cutoff_ms,),
        ).fetchall()

    if len(rows) < args.min_samples:
        logger.error("Only %d rows < cutoff (need >= %d). Generate more history or move cutoff later.",
                     len(rows), args.min_samples)
        return 1

    logger.info("Training XGBoost on %d samples with cutoff < %s", len(rows), args.cutoff)

    # Simple feature: lagged confidence + label distribution over last N rows.
    # NOTE: this is a STAND-IN for the full feature set the production model
    # uses (funding_rate, orderbook_imbalance, ...) which we don't have
    # historically. This is intentionally simple; real training needs the
    # production feature_store.
    X = np.array([[float(r[2] or 0.0)] for r in rows], dtype=np.float64)
    y = np.array([int(r[1]) for r in rows], dtype=np.int64)

    model = xgb.XGBClassifier(
        n_estimators=60, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=3, verbosity=0,
    )
    model.fit(X, y)

    out = args.out or f"models/regime_xgboost_replay_{args.cutoff}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": args.cutoff,
        "n_samples": len(rows),
        "feature_set": "confidence_only (stand-in -- real production model uses 6 features)",
        "warning": "This model is trained on synthesized rule-based regime labels, "
                   "not on the bot's live regime_history. Use only for replay "
                   "smoke tests; cannot reproduce live ML behaviour.",
    }
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved frozen XGBoost artifact to %s (+ .meta.json)", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", "-v", action="store_true")

    p.add_argument("--audit", action="store_true",
                   help="Report training-data availability and feasibility")
    p.add_argument("--generate-history", action="store_true",
                   help="Walk cached candles, run rule-based regime detector, write rows "
                        "to a replay_regime_history.db")
    p.add_argument("--train-xgboost", action="store_true",
                   help="Train an XGBoost model with --cutoff on replay_regime_history.db")

    p.add_argument("--cache-db", default=DEFAULT_CACHE_DB)
    p.add_argument("--history-db", default="data/replay_regime_history.db")
    p.add_argument("--start", default="2025-04-05", help="History generation start date")
    p.add_argument("--end", default="2026-05-09", help="History generation end date")
    p.add_argument("--coins", default="BTC", help="Comma-separated coins to generate history for")
    p.add_argument("--cutoff", help="Training-cutoff date (YYYY-MM-DD). Required for --train-xgboost.")
    p.add_argument("--out", help="Output path for the frozen model artifact")
    p.add_argument("--min-samples", type=int, default=50,
                   help="Minimum training rows before --cutoff (default 50)")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.audit:
        return cmd_audit(args)
    if args.generate_history:
        return cmd_generate_history(args)
    if args.train_xgboost:
        if not args.cutoff:
            p.error("--cutoff is required for --train-xgboost")
        return cmd_train_xgboost(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
