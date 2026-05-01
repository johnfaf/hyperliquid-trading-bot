#!/usr/bin/env python3
"""Quarantine fixture/invalid strategy-source data in the configured DB."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import database as db  # noqa: E402


def main() -> int:
    summary = db.quarantine_contaminated_runtime_data()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
