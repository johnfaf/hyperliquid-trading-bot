#!/usr/bin/env python3
"""One-command investor evidence report orchestration."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.investor_evidence import (  # noqa: E402
    build_baselines,
    build_investor_report,
    build_live_evidence_pack,
    build_walk_forward_report,
    parse_window,
    snapshot_dataset,
    utc_now_slug,
    write_json,
)


def _sqlite_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _default_out() -> Path:
    return Path("reports") / f"investor_{utc_now_slug()}.md"


def generate_investor_report(
    *,
    window: str = "90d",
    out: str | Path | None = None,
    capital: float = 10_000.0,
    num_trials: int = 1,
    db_path: str | None = None,
    candle_cache_db: str = "data/candle_cache.db",
    dataset_dir: str | Path | None = None,
    require_parquet: bool = False,
) -> dict:
    out_path = Path(out or _default_out())
    run_dir = out_path.parent / f"{out_path.stem}_artifacts"
    dataset_path = Path(dataset_dir or run_dir / "dataset")
    evidence_dir = run_dir / "live_evidence"
    json_out = out_path.with_suffix(".json")

    if db_path:
        db_cm = _sqlite_conn(db_path)
    else:
        from src.data import database as db

        db_cm = db.get_connection(for_read=True)

    candle_conn = None
    candle_path = Path(candle_cache_db)
    if candle_path.exists():
        candle_conn = _sqlite_conn(str(candle_path))

    try:
        with db_cm as conn:
            days = parse_window(window)
            baseline = build_baselines(
                conn,
                candle_conn=candle_conn,
                starting_capital=capital,
                window_days=days,
            )
            manifest = snapshot_dataset(
                conn,
                dataset_path,
                window_days=days,
                require_parquet=require_parquet,
            )
            walk_forward = build_walk_forward_report(
                conn,
                window_days=days,
                starting_capital=capital,
                num_trials=num_trials,
            )
            evidence = build_live_evidence_pack(
                conn,
                evidence_dir,
                window_days=days,
                starting_capital=capital,
                num_trials=num_trials,
            )
    finally:
        if candle_conn is not None:
            candle_conn.close()

    markdown = build_investor_report(
        baseline_report=baseline,
        walk_forward_report=walk_forward,
        dataset_manifest=manifest,
        evidence_pack=evidence,
        config_path=ROOT / "config.py",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    result = {
        "out": str(out_path),
        "json_out": str(json_out),
        "dataset_manifest": str(dataset_path / "manifest.json"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "evidence_report": evidence.get("report_path"),
        "trade_csv": evidence.get("trade_csv_path"),
        "signature_path": evidence.get("signature_path"),
        "signed": evidence.get("signature", {}).get("signed"),
        "baseline": baseline,
        "walk_forward": walk_forward,
        "dataset": manifest,
        "evidence": evidence,
    }
    write_json(json_out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="90d")
    parser.add_argument("--out", default=str(_default_out()))
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--db-path")
    parser.add_argument("--candle-cache-db", default="data/candle_cache.db")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--require-parquet", action="store_true")
    args = parser.parse_args(argv)
    result = generate_investor_report(
        window=args.window,
        out=args.out,
        capital=args.capital,
        num_trials=args.num_trials,
        db_path=args.db_path,
        candle_cache_db=args.candle_cache_db,
        dataset_dir=args.dataset_dir,
        require_parquet=args.require_parquet,
    )
    print(json.dumps({
        "out": result["out"],
        "dataset_sha256": result["dataset_sha256"],
        "trade_csv": result["trade_csv"],
        "signature_path": result["signature_path"],
        "signed": result["signed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
