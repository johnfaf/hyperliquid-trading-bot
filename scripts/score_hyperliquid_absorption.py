#!/usr/bin/env python3
"""Read the research ledger and publish a matched, costed absorption scorecard."""
from argparse import ArgumentParser
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
import time

from src.backtest.absorption_study import DAY, FEE_BPS, HAIRCUTS, HORIZONS, SIZE_USD, analyze
from src.backtest.candle_entry_replay import summarize
from src.signals.flow_absorption import RULE_ID, SLOT_MS


def instant(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat() if value is not None else "none"


def bps(value):
    return "Unmeasured" if value is None else f"{value:+.1f} bps"


def build(path, days=21, at=None, *, structured=False):
    at = at if at is not None else time.time_ns() // 1_000_000
    start = at - days * DAY
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        frames = db.execute("""
            SELECT count(*), count(CASE WHEN state='ready' THEN 1 END), min(at), max(at), count(DISTINCT session),
                   coalesce(sum(trades),0), coalesce(sum(sign_agrees),0), coalesce(sum(sign_disagrees),0)
            FROM frames WHERE at>=? AND at<?
        """, (start, at)).fetchone()
        coverage = Counter({name: count for name, count in db.execute("""
            SELECT state,count(*) FROM decisions WHERE rule=? AND at>=? AND at<? GROUP BY state
        """, (RULE_ID, start, at))})
        studies = {horizon: analyze(db, start, at, horizon) for horizon in HORIZONS}
    primary, _, bands = studies[HORIZONS[0]]
    result = summarize(primary)
    side_supported = frames[6] + frames[7] >= 100 and frames[6] > frames[7]
    last_expected = (at - 1) // SLOT_MS * SLOT_MS
    possible = 2 * ((last_expected - frames[2]) // SLOT_MS + 1) if frames[2] is not None else 0
    healthy = possible > 0 and frames[1] / possible >= .95 and at - frames[3] <= 30_000
    passes = (healthy and side_supported and result["pairs"] >= 50 and result["days"] >= 14 and bands is not None
              and result["adverse"]["all"] > 0 and all(bounds[0] > 0 for bounds in bands.values()))
    state = ("Eligible for research review only" if passes else "Insufficient forward evidence"
             if not healthy or result["pairs"] < 50 or result["days"] < 14 or not side_supported else "Failed the forward screen")
    report = {"version": 1, "observed_at": instant(at), "start": instant(start), "rule": RULE_ID,
                "state": state, "healthy_at_scoring": healthy, "live_approved": False,
                "frames": frames[0], "ready": frames[1], "expected": possible,
                "first_frame": instant(frames[2]), "last_frame": instant(frames[3]),
                "side_supported": side_supported, "coverage": dict(coverage),
                "horizons": [{"minutes": horizon, **summarize(rows)} for horizon, (rows, _, _) in studies.items()],
                "primary_intervals": bands}
    lines = ["# BTC/ETH order-flow absorption", "", f"**{state}. No orders or trading changes.**", "",
             f"Rule: {RULE_ID}. Lookback: {days} days ending {instant(at)}. Five minutes is the primary horizon; 15 minutes is diagnostic.",
             f"- Observed {frames[0]:,} five-second frames ({frames[1]:,} ready) and {frames[5]:,} public trades from "
             f"{instant(frames[2])} through {instant(frames[3])}; {frames[4]} connection sessions.",
             f"- Ready slots: {frames[1]:,} / {possible:,} expected since observation began in this window.",
             f"- Minutes per market: {coverage['warming_up']} warming up, {coverage['coverage_gap']} incomplete, "
             f"{coverage['no_pressure']} no burst, {coverage['pressure_without_absorption']} burst controls, {coverage['absorption']} absorption.",
             f"- Trade-side sanity check against a recent prior quote: {frames[6]} agree, {frames[7]} disagree. "
             "This does not prove the exact aggressor sequence.", "",
             f"Quotes are simulated for ${SIZE_USD:g} at the first five-second slot after the decision, using ask/bid and "
             f"available top-of-book depth, then a {FEE_BPS:g} bps fee plus {HAIRCUTS[0]:g} or {HAIRCUTS[1]:g} bps extra cost per side. "
             "Funding-hour crossings are excluded. No simulated fill is sent to the exchange.", "",
             "| Horizon | Matched / scored | Base signal / control | Stress signal / control | Stress lift | All scored, stress |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for horizon, (rows, _, _) in studies.items():
        stat = summarize(rows)
        base, stress = stat["base"], stat["adverse"]
        lines.append(f"| {horizon}m | {stat['pairs']} / {stat['signals']} | {bps(base['signal'])} / {bps(base['control'])} | "
                     f"{bps(stress['signal'])} / {bps(stress['control'])} | {bps(stress['lift'])} | {bps(stress['all'])} |")
    lines.append("")
    for horizon, (_, counts, _) in studies.items():
        lines.append(f"- {horizon}m: " + (", ".join(f"{key.replace('_', ' ')}={count}" for key, count in sorted(counts.items()))
                                            if counts else "no signals yet"))
    if bands:
        for key, bounds in bands.items():
            lines.append(f"- Primary stress {key} 95% exploratory block interval: {bps(bounds[0])} to {bps(bounds[1])}.")
    lines += ["", "Matched controls are earlier bursts in the same asset, direction and 8-hour UTC session, with similar signed flow "
              "and relative volume. The control must have matured before the signal. No control exposure is reused, and a missing label "
              "never causes rematching. Confidence bands use two-day blocks across both markets.",
              "Review needs 95% ready slots and a fresh feed, 50 matched primary pairs on 14 distinct UTC days, positive stressed net on all scored signals, "
              "and positive lower bands for matched net and lift. Data or quote failures are not zero returns. A pass only requests research review.", ""]
    body = "\n".join(lines)
    return {**report, "markdown": body} if structured else body


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="Atomically refresh the private research scorecard")
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--json-out", type=Path, help="Separate structured dashboard scorecard")
    args = parser.parse_args()
    if not 1 <= args.days <= 30:
        parser.error("use 1-30 days")
    if args.json_out and (args.json_out.suffix != ".json" or args.json_out.resolve() == args.db.resolve()):
        parser.error("use a separate .json report path")
    if args.out and (args.out.resolve().suffix != ".md" or args.out.resolve() == args.db.resolve()):
        parser.error("use a separate .md report path")
    report = build(args.db, args.days, structured=True)
    body = report.pop("markdown")
    if args.json_out:
        atomic_write(args.json_out, json.dumps(report, allow_nan=False))
    if args.out is None:
        print(body, end="")
        return
    atomic_write(args.out, body)


def atomic_write(path, body):
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as out:
        stage = Path(out.name)
        out.write(body)
    try:
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
