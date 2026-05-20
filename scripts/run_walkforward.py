#!/usr/bin/env python3
"""Create an immutable evidence snapshot and chronological walk-forward report."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.investor_evidence import (  # noqa: E402
    build_walk_forward_report,
    parse_window,
    render_walk_forward_markdown,
    snapshot_dataset,
    utc_now_slug,
    write_json,
)


def _sqlite_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _default_dataset_dir() -> Path:
    return Path("reports") / "evidence" / "datasets" / f"snapshot_{utc_now_slug()}"


def _default_report_path() -> Path:
    return Path("reports") / "evidence" / f"walkforward_{utc_now_slug()}.md"


def _run_replay_windows(report: dict, args: argparse.Namespace) -> list[dict]:
    replay_results = []
    replay_dir = Path(args.replay_report_dir or "reports/evidence/replay")
    replay_dir.mkdir(parents=True, exist_ok=True)
    for name, metrics in (report.get("windows") or {}).items():
        if name == "train" or int(metrics.get("trades") or 0) <= 0:
            continue
        start = str(metrics.get("period_start", ""))[:10]
        end = str(metrics.get("period_end", ""))[:10]
        if not start or not end or start == "None" or end == "None":
            continue
        out = replay_dir / f"replay_{name}_{utc_now_slug()}.json"
        cmd = [
            sys.executable,
            "scripts/run_replay.py",
            "--start",
            start,
            "--end",
            end,
            "--step",
            args.step,
            "--coins",
            args.coins,
            "--cache-db",
            args.cache_db,
            "--run-id",
            f"wf_{name}_{utc_now_slug()}",
            "--report-out",
            str(out),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=args.replay_timeout_s)
        replay_results.append({
            "window": name,
            "start": start,
            "end": end,
            "returncode": proc.returncode,
            "report_out": str(out),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        })
        if proc.returncode != 0 and args.halt_on_replay_error:
            break
    return replay_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="90d")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Number of tried strategies/configs for DSR deflation")
    parser.add_argument("--db-path", help="Optional SQLite DB path. Defaults to configured bot DB.")
    parser.add_argument("--dataset-dir", default=str(_default_dataset_dir()))
    parser.add_argument("--out", default=str(_default_report_path()))
    parser.add_argument("--json-out")
    parser.add_argument("--require-parquet", action="store_true",
                        help="Fail if Parquet cannot be written")
    parser.add_argument("--run-replay", action="store_true",
                        help="Invoke scripts/run_replay.py for validation/test windows")
    parser.add_argument("--coins", default="BTC,ETH")
    parser.add_argument("--step", default="1h")
    parser.add_argument("--cache-db", default="data/candle_cache.db")
    parser.add_argument("--replay-report-dir")
    parser.add_argument("--replay-timeout-s", type=int, default=900)
    parser.add_argument("--halt-on-replay-error", action="store_true")
    args = parser.parse_args(argv)

    if args.db_path:
        db_cm = _sqlite_conn(args.db_path)
    else:
        from src.data import database as db

        db_cm = db.get_connection(for_read=True)

    with db_cm as conn:
        manifest = snapshot_dataset(
            conn,
            args.dataset_dir,
            window_days=parse_window(args.window),
            require_parquet=args.require_parquet,
        )
        report = build_walk_forward_report(
            conn,
            window_days=parse_window(args.window),
            starting_capital=args.capital,
            num_trials=args.num_trials,
        )

    if args.run_replay:
        report["replay_runs"] = _run_replay_windows(report, args)

    report["dataset_manifest"] = manifest
    markdown = render_walk_forward_markdown(report)
    if report.get("replay_runs"):
        markdown += "\n## Replay Runs\n\n"
        for replay in report["replay_runs"]:
            markdown += (
                f"- {replay['window']}: rc={replay['returncode']} "
                f"`{replay['report_out']}` ({replay['start']} -> {replay['end']})\n"
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    write_json(args.json_out or out.with_suffix(".json"), report)
    print(json.dumps({
        "dataset_manifest": str(Path(args.dataset_dir) / "manifest.json"),
        "walkforward_report": str(out),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "row_counts": manifest.get("row_counts"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
