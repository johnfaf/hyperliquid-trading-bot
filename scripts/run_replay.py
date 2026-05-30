#!/usr/bin/env python3
"""Replay harness CLI.

Boots the historical replay harness, runs the real production
trading_cycle over a window of cached candle data, and dumps a report
of api / stub activity plus the resulting paper_trades / audit_trail.

Examples
--------
Minimal run on the cached BTC window:

    python scripts/run_replay.py --start 2025-08-01 --end 2025-08-08 \\
        --step 1h --coins BTC

Custom strategy snapshot, persistent replay DB, JSON report:

    python scripts/run_replay.py --start 2025-08-01 --end 2025-09-01 \\
        --step 1h --strategy-snapshot fixtures/replay_pool_aug2025.json \\
        --run-id aug2025_baseline --report-out reports/replay_aug2025.json

Export a frozen strategy snapshot from a live-DB backup so future runs
have a known-good pool:

    python scripts/run_replay.py --export-snapshot data/bot_backup.db \\
        --snapshot-date 2025-08-01 --report-out fixtures/replay_pool.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_date(s: str) -> int:
    """Accept YYYY-MM-DD or full ISO. Return epoch ms."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date {s!r}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")


def _parse_step(s: str) -> int:
    """Parse step like '1m', '5m', '1h', '6h', '1d' to ms."""
    units = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if not s:
        raise ValueError("--step required")
    unit = s[-1]
    if unit not in units:
        raise ValueError(f"step unit must be one of {list(units)}, got {unit!r}")
    try:
        n = int(s[:-1])
    except ValueError:
        raise ValueError(f"step value must be int, got {s[:-1]!r}")
    if n <= 0:
        raise ValueError(f"step must be positive, got {n}")
    return n * units[unit]


def _cmd_run(args: argparse.Namespace) -> int:
    from src.backtest.replay.harness import ReplayHarness
    from src.backtest.replay.strategy_seed import (
        load_snapshot, build_default_smoke_snapshot, seed_into,
    )
    from src.core.cycles.trading_cycle import run_trading_cycle

    start_ms = _parse_date(args.start)
    end_ms = _parse_date(args.end)
    step_ms = _parse_step(args.step)
    coins = [c.strip().upper() for c in (args.coins or "BTC").split(",") if c.strip()]

    logger = logging.getLogger("replay")
    logger.info("Replay window: %s -> %s (step=%s) coins=%s",
                args.start, args.end, args.step, coins)

    snapshot = (
        load_snapshot(args.strategy_snapshot)
        if args.strategy_snapshot
        else build_default_smoke_snapshot()
    )
    logger.info("Strategy pool: %d strategies, %d traders (from %s)",
                len(snapshot.strategies), len(snapshot.traders),
                args.strategy_snapshot or "default smoke snapshot")

    with ReplayHarness(
        start_ts_ms=start_ms,
        end_ts_ms=end_ms,
        cache_db=args.cache_db,
        coins=coins,
        engage_network_sandbox=not args.allow_network,
        build_container=True,
        run_id=args.run_id,
        keep_replay_db=not args.discard_replay_db,
        strict_api=not args.lax_api,
        frozen_xgb_model=args.frozen_xgb_model,
        fills_db=args.fills_db,
    ) as h:
        seed_into(str(h.replay_db.db_path), snapshot)

        completed_ticks = 0
        failed_ticks = 0
        last_err: Exception | None = None
        for tick in h.iter_ticks(step_ms=step_ms):
            try:
                run_trading_cycle(h.container, cycle_count=tick.index)
                completed_ticks += 1
            except Exception as e:
                failed_ticks += 1
                last_err = e
                if args.halt_on_error:
                    logger.error("Tick %d (%d) raised %s: %s",
                                 tick.index, tick.ts_ms, type(e).__name__, e)
                    break
                else:
                    logger.warning("Tick %d (%d) raised %s: %s -- continuing",
                                   tick.index, tick.ts_ms, type(e).__name__, e)

        report = h.build_report(tick_count=completed_ticks, step_ms=step_ms)
        report_dict = _build_report_dict(h, args, snapshot, report,
                                         completed_ticks, failed_ticks, last_err)
        if args.diff_live_db:
            report_dict["decision_diff"] = _build_decision_diff_report(
                live_db=args.diff_live_db,
                replay_db=str(h.replay_db.db_path),
                start_ms=start_ms,
                end_ms=end_ms,
                match_window_s=args.diff_match_window,
                min_live_match_rate=args.diff_min_live_match_rate,
                min_replay_match_rate=args.diff_min_replay_match_rate,
                report_out=args.diff_report_out,
            )

        if args.report_out:
            out_path = Path(args.report_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2, default=str, sort_keys=True)
            logger.info("Report written to %s", out_path)

        _print_summary(report_dict, h)
        if failed_ticks:
            return 1
        decision_diff = report_dict.get("decision_diff") or {}
        diagnostics = decision_diff.get("diagnostics") or {}
        if args.diff_live_db and diagnostics.get("trustworthy") is False:
            logger.error("Decision diff did not meet trust threshold: %s", diagnostics)
            return 1
        return 0


def _build_report_dict(h, args, snapshot, report, completed_ticks, failed_ticks, last_err):
    import sqlite3

    db_path = str(h.replay_db.db_path) if h.replay_db else None
    paper_trade_count = 0
    audit_count = 0
    if db_path and Path(db_path).exists():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            paper_trade_count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
            audit_count = conn.execute("SELECT COUNT(*) FROM audit_trail").fetchone()[0]

    return {
        "config": {
            "start": args.start,
            "end": args.end,
            "step": args.step,
            "coins": [c.strip().upper() for c in (args.coins or "BTC").split(",")],
            "cache_db": args.cache_db,
            "run_id": h.replay_db.run_id if h.replay_db else None,
            "strict_api": not args.lax_api,
            "network_sandbox": not args.allow_network,
        },
        "snapshot": {
            "snapshot_date": snapshot.snapshot_date,
            "description": snapshot.description,
            "n_strategies": len(snapshot.strategies),
            "n_traders": len(snapshot.traders),
        },
        "execution": {
            "completed_ticks": completed_ticks,
            "failed_ticks": failed_ticks,
            "last_error": f"{type(last_err).__name__}: {last_err}" if last_err else None,
            "step_ms": report.step_ms,
        },
        "api_activity": {
            "calls_by_type": report.api_calls_by_type,
            "coin_cache_misses": report.api_coin_cache_misses,
        },
        "stub_activity": report.stub_calls,
        "replay_db_path": db_path,
        "outputs": {
            "paper_trades": paper_trade_count,
            "audit_trail_rows": audit_count,
        },
    }


def _iso_from_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def _build_decision_diff_report(
    *,
    live_db: str,
    replay_db: str,
    start_ms: int,
    end_ms: int,
    match_window_s: float,
    min_live_match_rate: float,
    min_replay_match_rate: float,
    report_out: str | None = None,
) -> dict:
    """Attach replay-vs-live audit diff to a replay run report."""
    from scripts import replay_audit_diff

    start_iso = _iso_from_ms(start_ms)
    end_iso = _iso_from_ms(end_ms)
    live_rows = replay_audit_diff._load_audit(live_db, start_iso, end_iso)
    replay_rows = replay_audit_diff._load_audit(replay_db, start_iso, end_iso)
    diff = replay_audit_diff.diff_audit_trails(
        live_rows,
        replay_rows,
        match_window_s=match_window_s,
    )
    out = diff.to_dict()
    out["diagnostics"] = diff.diagnostics(
        min_live_match_rate=min_live_match_rate,
        min_replay_match_rate=min_replay_match_rate,
    )
    out["config"] = {
        "live": live_db,
        "replay": replay_db,
        "start": start_iso,
        "end": end_iso,
        "match_window_s": match_window_s,
    }
    if report_out:
        out_path = Path(report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str, sort_keys=True)
    return out


def _print_summary(report: dict, h) -> None:
    print()
    print("=" * 78)
    print(f"  REPLAY SUMMARY  (run_id={report['config']['run_id']})")
    print("=" * 78)
    print(f"  Window:           {report['config']['start']} -> {report['config']['end']} "
          f"(step={report['config']['step']})")
    print(f"  Coins:            {report['config']['coins']}")
    print(f"  Strategy pool:    {report['snapshot']['n_strategies']} strategies, "
          f"{report['snapshot']['n_traders']} traders")
    print(f"  Replay DB:        {report['replay_db_path']}")
    print()
    print(f"  Ticks completed:  {report['execution']['completed_ticks']}")
    print(f"  Ticks failed:     {report['execution']['failed_ticks']}")
    if report['execution']['last_error']:
        print(f"  Last error:       {report['execution']['last_error']}")
    print()
    print(f"  paper_trades:     {report['outputs']['paper_trades']}")
    print(f"  audit_trail:      {report['outputs']['audit_trail_rows']}")
    print()
    if report['api_activity']['calls_by_type']:
        print("  API calls by type:")
        for k, v in sorted(report['api_activity']['calls_by_type'].items(),
                           key=lambda x: -x[1]):
            print(f"    {k:<25} {v:>6}")
    if report['api_activity']['coin_cache_misses']:
        print(f"  Coin cache misses:  {report['api_activity']['coin_cache_misses']}")
    print()
    stub_totals = {
        name: sum(calls.values())
        for name, calls in report['stub_activity'].items()
        if calls
    }
    if stub_totals:
        print("  Stub subsystems consulted:")
        for name, total in sorted(stub_totals.items(), key=lambda x: -x[1]):
            print(f"    {name:<25} {total:>6} calls")
    print("=" * 78)


def _cmd_export_snapshot(args: argparse.Namespace) -> int:
    from src.backtest.replay.strategy_seed import export_from_live_db, save_snapshot

    snap = export_from_live_db(args.export_snapshot, snapshot_date=args.snapshot_date)
    out = args.report_out or "fixtures/replay_strategy_pool.json"
    save_snapshot(snap, out)
    print(f"Exported {len(snap.traders)} traders, {len(snap.strategies)} strategies "
          f"from {args.export_snapshot} -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", "-v", action="store_true")

    # Subcommand-less: if --export-snapshot is set, run the export instead.
    parser.add_argument("--export-snapshot", metavar="LIVE_DB",
                        help="Dump live-DB's strategy + trader pool as a JSON snapshot")
    parser.add_argument("--snapshot-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        help="Date stamp for the exported snapshot")

    # Replay configuration
    parser.add_argument("--start", help="Replay window start (YYYY-MM-DD or full ISO)")
    parser.add_argument("--end", help="Replay window end (YYYY-MM-DD or full ISO)")
    parser.add_argument("--step", default="1h",
                        help="Tick step (e.g. 1m / 5m / 1h / 6h / 1d). Default 1h")
    parser.add_argument("--coins", default="BTC",
                        help="Comma-separated coin universe (default BTC)")

    # Data sources
    parser.add_argument("--cache-db", default="data/candle_cache.db",
                        help="Path to the candle cache (default data/candle_cache.db)")
    parser.add_argument("--strategy-snapshot",
                        help="Path to a strategy-pool JSON. If omitted, uses the built-in smoke snapshot.")

    # Run lifecycle
    parser.add_argument("--run-id",
                        help="Stable identifier for this replay (used in replay_<run_id>.db). "
                             "Defaults to a random short id.")
    parser.add_argument("--discard-replay-db", action="store_true",
                        help="Delete the replay DB on exit (default: keep for forensics)")

    # Safety / behavior
    parser.add_argument("--allow-network", action="store_true",
                        help="Disable the outbound HTTP sandbox (NOT recommended)")
    parser.add_argument("--lax-api", action="store_true",
                        help="Return None for unknown HL req_types instead of raising")
    parser.add_argument("--halt-on-error", action="store_true",
                        help="Stop replay on the first tick that raises")

    # ML opt-in
    parser.add_argument("--frozen-xgb-model",
                        help="Path to a frozen XGBoost regime model "
                             "(produced by scripts/freeze_replay_models.py --train-xgboost). "
                             "When set, the predictive_forecaster stub is bypassed and the "
                             "frozen model serves regime forecasts at decision time.")
    parser.add_argument("--fills-db",
                        help="Path to a DB with a wallet_fills table (e.g. the live "
                             "bot.db). When set, the replay reconstructs each tracked "
                             "trader's historical positions from their fills so "
                             "COPY-TRADE signals fire during the backtest. Omit to keep "
                             "the legacy empty-clearinghouse behavior (strategy-only).")

    # Output
    parser.add_argument("--report-out",
                        help="Where to write the JSON run report")
    parser.add_argument("--diff-live-db",
                        help="Optional live bot DB to diff against the replay audit_trail")
    parser.add_argument("--diff-match-window", type=float, default=600.0,
                        help="Decision diff matching tolerance in seconds. Default 600.")
    parser.add_argument("--diff-min-live-match-rate", type=float, default=0.70,
                        help="Fail replay when matched/live is below this decimal threshold. Default 0.70.")
    parser.add_argument("--diff-min-replay-match-rate", type=float, default=0.70,
                        help="Fail replay when matched/replay is below this decimal threshold. Default 0.70.")
    parser.add_argument("--diff-report-out",
                        help="Optional standalone JSON output for the decision diff")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.export_snapshot:
        return _cmd_export_snapshot(args)

    if not args.start or not args.end:
        parser.error("--start and --end are required for a replay run "
                     "(or pass --export-snapshot LIVE_DB to dump a snapshot)")
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
