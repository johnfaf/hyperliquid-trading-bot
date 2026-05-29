"""Seed CalibrationTracker bins from clean closed paper_trades.

Usage
-----
    # Default lookback (30 days), default thresholds
    python scripts/bootstrap_calibration.py

    # Custom lookback + cap
    python scripts/bootstrap_calibration.py --lookback-days 60 --cap-per-bucket 100

    # Dry run -- report what would happen but do not write to calibration
    python scripts/bootstrap_calibration.py --dry-run

When to run
-----------
* After a calibration table reset / quarantine clear
* After a fresh deploy where calibration history is sparse
* After running the tainted-trades migration (the bootstrap honours
  the tainted flag, so clean data only)

Idempotent: re-running is safe.  Any (source_key, side, regime)
bucket that already has > CALIBRATION_BOOTSTRAP_SKIP_THRESHOLD
records is skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Iterable

logger = logging.getLogger("bootstrap_calibration")


def _build_tracker(dry_run: bool):
    """Return a calibration tracker -- real or a counting stub."""
    from src.signals.calibration import CalibrationTracker

    if not dry_run:
        return CalibrationTracker()

    class _DryRunTracker:
        """Pretends to be a CalibrationTracker but writes nothing."""
        def __init__(self):
            self.seen = 0

        def _resolve_key(self, source_key, *, side=None, regime=None):
            from src.signals.calibration import compose_calibration_key
            return compose_calibration_key(source_key, side, regime)

        def get_sample_size(self, _key):
            return 0

        def record(self, *args, **kwargs):
            self.seen += 1

    return _DryRunTracker()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookback-days", type=int, default=None,
        help="Days of paper_trades to read (default: env "
             "CALIBRATION_BOOTSTRAP_LOOKBACK_DAYS or 30).",
    )
    parser.add_argument(
        "--cap-per-bucket", type=int, default=None,
        help="Max records seeded per (source_key, side, regime) bucket "
             "(default: env CALIBRATION_BOOTSTRAP_MAX_PER_BUCKET or 200).",
    )
    parser.add_argument(
        "--skip-threshold", type=int, default=None,
        help="Skip buckets already exceeding this many records "
             "(default: env CALIBRATION_BOOTSTRAP_SKIP_THRESHOLD or 100).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count trades and buckets but do not write to calibration_records.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.learning.calibration_bootstrap import (
        bootstrap_calibration_from_history,
    )

    tracker = _build_tracker(args.dry_run)
    stats = bootstrap_calibration_from_history(
        tracker,
        lookback_days_v=args.lookback_days,
        skip_threshold=args.skip_threshold,
        cap_per_bucket=args.cap_per_bucket,
    )
    if args.dry_run:
        stats["mode"] = "dry_run"
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
