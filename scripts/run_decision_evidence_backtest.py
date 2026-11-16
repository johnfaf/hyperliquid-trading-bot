#!/usr/bin/env python3
"""Read-only replay of recorded decisions and a matched copy-exclusion diagnostic."""
import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time
from urllib.request import Request, urlopen


BAR = 15 * 60 * 1000
HORIZON = 4
COSTS = (14, 30)  # Assumed all-in round trip bps, not measured venue fills.


def dt(text):
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("dates require an explicit UTC offset")
    return value.astimezone(timezone.utc)


def display(value):
    return f"${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}"


def obj(raw):
    try:
        value = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_readonly(path, start, end):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        decisions = [dict(r) for r in conn.execute("""
            SELECT decision_id,created_at,coin,side,source,final_status,paper_trade_id,entry_price
            FROM decision_snapshots WHERE created_at>=? AND created_at<?
            ORDER BY created_at,decision_id
            """, (start.isoformat(), end.isoformat()))]
        ledger = [dict(r) for r in conn.execute("""
            SELECT id,opened_at,closed_at,coin,side,status,pnl,metadata,entry_price,size FROM paper_trades
            WHERE opened_at<? ORDER BY opened_at,id
            """, (end.isoformat(),))]
    for row in ledger:
        meta = obj(row.pop("metadata"))
        row["copy"] = bool(meta.get("is_copy_trade") or meta.get("source") == "copy_trade"
                           or str(meta.get("source_key") or "").startswith("copy_trade:") or meta.get("source_trader"))
        row["clean"] = not (meta.get("tainted") or meta.get("synthetic_reconciliation") or
                            "live_reconciled_closed" in (meta.get("close_reason"), meta.get("reconciliation_reason")))
        row["source"] = "copy" if row["copy"] else str(meta.get("source") or "unattributed")
        row["coverage_revision"] = meta.get("coverage_revision")
        row["shadow"] = (meta.get("entry_decision") or {}).get("copy_concentration_study") if isinstance(meta.get("entry_decision"), dict) else None
    return decisions, ledger


def market_candles(assets, start, end):
    """Fetch public snapshots only; never sends order or account data."""
    cache, failures = {}, []
    for coin in sorted(assets):
        body = {"type": "candleSnapshot", "req": {
            "coin": coin, "interval": "15m", "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
        }}
        req = Request("https://api.hyperliquid.xyz/info", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=15) as response:
                rows = json.load(response)
            slot = {}
            for bar in rows:
                start_ms = int(bar["t"])
                end_ms = int(bar["T"])
                price = float(bar["o"]), float(bar["c"])
                if (end_ms - start_ms != BAR - 1 or start_ms % BAR or start_ms in slot
                        or not all(math.isfinite(p) and p > 0 for p in price)):
                    raise ValueError("invalid candle sequence")
                slot[start_ms] = price
            if not slot:
                raise ValueError("no candle data")
            cache[coin] = slot
        except (OSError, ValueError, TypeError, KeyError):
            failures.append(coin)
        time.sleep(0.15)
    return cache, failures


def forward_price(row, candles, end):
    # Even a signal on an exact boundary must wait for the next bar.
    when = int(dt(row["created_at"]).timestamp() * 1000)
    first = (when // BAR + 1) * BAR
    last = first + (HORIZON - 1) * BAR
    if last + BAR > int(end.timestamp() * 1000):
        return None, "window_ends_before_horizon"
    series = candles.get(row["coin"], {})
    checkpoints = [series.get(t) for t in range(first, last + BAR, BAR)]
    if not all(checkpoints):
        return None, "missing_contiguous_bars"
    first_price, last_close = checkpoints[0][0], checkpoints[-1][1]
    sign = 1 if row["side"] == "long" else -1
    return sign * (last_close / first_price - 1) * 10_000, None


def simulated_paths(decisions, ledger, candles, end):
    buckets = defaultdict(list)
    missing = defaultdict(int)
    known = {row["id"] for row in ledger if row["clean"]}
    for item in decisions:
        if item["side"] not in {"long", "short"} or not item["coin"]:
            missing["missing_direction_or_asset"] += 1
            continue
        if item["paper_trade_id"] in known:
            bucket = "Linked paper trade"
        elif "rejected" in str(item["final_status"] or ""):
            bucket = "Explicitly filtered"
        else:
            bucket = "Other / execution unverified"
        path, problem = forward_price(item, candles, end)
        buckets[bucket].append(path)
        if problem:
            missing[problem] += 1
    return buckets, missing


def realized_stats(rows):
    pnl, peak, worst, wins = 0., 0., 0., 0
    for row in sorted(rows, key=lambda row: (dt(row["closed_at"]), row["id"])):
        pnl += row["pnl"]
        peak = max(peak, pnl)
        worst = max(worst, peak - pnl)
        wins += row["pnl"] > 0
    return {"n": len(rows), "pnl": pnl, "drawdown": worst, "wins": wins}


def prospective_review(ledger, evidence, start, end, protocol):
    """A screening result for human paper-policy review, never an approval."""
    rows = [r for r in ledger if r["copy"] and start <= dt(r["opened_at"]) < end]
    revisions = evidence.get("revisions", [])
    revision = revisions[0] if len(revisions) == 1 and revisions[0] != "unknown" else None
    tagged = [r for r in rows if r["coverage_revision"] == revision and revision is not None
              and isinstance(r["shadow"], dict) and r["shadow"].get("policy") == protocol
              and type(r["shadow"].get("peers")) is int and r["shadow"]["peers"] >= 0]
    paired = []
    for r in tagged:
        if not (r["clean"] and r["closed_at"] and dt(r["closed_at"]) < end and r["pnl"] is not None):
            continue
        try:
            notional = float(r["entry_price"]) * float(r["size"])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(notional) and notional > 0:
            paired.append({**r, "pnl": r["pnl"] - notional * (COSTS[1] - COSTS[0]) / 10_000})
    base = realized_stats(paired)
    kept = realized_stats([r for r in paired if not r["shadow"]["peers"]])
    expected = evidence.get("expected", 0)
    reasons = []
    checks = [
        (end - start >= timedelta(days=14), "Fewer than 14 days of prospective observation"),
        (evidence.get("available") and evidence.get("schedule_contiguous"), "Coverage schedule is incomplete"),
        (revision and evidence.get("protocols") == [protocol], "No single known build and pre-recorded study protocol"),
        (expected and evidence.get("terminal", 0) / expected >= .95, "Final coverage below 95% or unmeasured"),
        (evidence.get("unresolved_cycles") == 0 and evidence.get("unresolved_intents") == 0
         and evidence.get("unresolved_orders") in (0, None) and evidence.get("errors") == 0,
         "Execution evidence has unresolved outcomes or errors"),
        (len(rows) == len(tagged), "Not every copy entry was measured by the pre-registered rule"),
        (len(paired) >= 30 and kept["n"] > 0, "Fewer than 30 matched, clean closes with known notional, or no retained closes"),
        (base["n"] > kept["n"], "No observed overlap exclusions"),
        (kept["pnl"] > max(base["pnl"], 0) and kept["drawdown"] <= base["drawdown"],
         "Retained trades are not profitable and better than baseline without increased realized drawdown under cost stress"),
    ]
    reasons.extend(reason for passed, reason in checks if not passed)
    return {"state": "Not eligible" if reasons else "Operator paper review required", "reasons": reasons,
            "baseline": base, "retained": kept, "unknown_venue_orders": evidence.get("unresolved_orders") is None}


def copy_replay(ledger, start, end):
    candidates = []
    skipped = set()
    active = defaultdict(list)
    for row in sorted((r for r in ledger if r["clean"] and r["copy"] and r["opened_at"]),
                      key=lambda r: (dt(r["opened_at"]), r["id"])):
        at = dt(row["opened_at"])
        key = (row["coin"], row["side"])
        active[key] = [close for close in active[key] if close is None or close > at]
        if active[key]:
            skipped.add(row["id"])
        close = dt(row["closed_at"]) if row["closed_at"] else None
        active[key].append(close)
        if start <= at < end and close is not None and close < end and row["pnl"] is not None:
            candidates.append(row)
    retained = [row for row in candidates if row["id"] not in skipped]
    omitted = [row for row in candidates if row["id"] in skipped]
    return realized_stats(candidates), realized_stats(retained), realized_stats(omitted)


def report(decisions, ledger, candles, failures, start, split, end):
    actual = [r for r in ledger if r["clean"] and r["closed_at"] and
              start <= dt(r["closed_at"]) < end and r["pnl"] is not None]
    result = realized_stats(actual)
    counts = defaultdict(int)
    for d in decisions:
        counts[d["final_status"] or "unmeasured"] += 1
    by_id = defaultdict(int)
    for d in decisions:
        if d["paper_trade_id"] is not None:
            by_id[d["paper_trade_id"]] += 1
    lines = [
        "# Decision evidence and backtest", "",
        f"Window UTC: {start.isoformat()} to {end.isoformat()} (exclusive).",
        f"Chronological split UTC: {split.isoformat()}. Both slices are retrospective.",
        "", "## Ledger facts",
        f"- Decision snapshots: {len(decisions)}. Linked to a paper trade ID: {sum(by_id.values())}. "
        f"Duplicate links: {sum(n - 1 for n in by_id.values())}.",
        "- Status counts: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + ".",
        f"- Recorded paper closes: {result['n']}; wins: {result['wins']}; ledger net: {display(result['pnl'])}; "
        f"realized drawdown: {display(result['drawdown'])}.",
        "- Closed-trade attribution: " + "; ".join(f"{k}: {display(realized_stats(closed)['pnl'])} / {len(closed)}"
                                                    for k, closed in sorted(_by_source(actual).items())) + ".",
        "- Historic snapshots do not establish how many evaluations were expected. A missing link does not establish a missed trade.",
        "", "## Copy concentration: matched trade exclusion",
        "Uses the recorded pre-entry paper book. Omitted positions are not replaced, and changes to later opportunities "
        "or funding are not simulated. Drawdown is based on realized closes only.",
        "", "| Slice | Paper cohort net / N | Omit repeats net / N | Skipped net / winners | Drawdown before / after |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, lo, hi in (("Whole", start, end), ("Earlier", start, split), ("Later", split, end)):
        base, excluded, skipped = copy_replay(ledger, lo, hi)
        lines.append(f"| {label} | {display(base['pnl'])} / {base['n']} | {display(excluded['pnl'])} / {excluded['n']} | "
                     f"{display(skipped['pnl'])} / {skipped['wins']} | {display(base['drawdown'])} / {display(excluded['drawdown'])} |")
    lines += [
        "", "## Hypothetical one-hour price paths",
        "Directional move from the next 15-minute bar's open through the fourth bar's close. "
        "Requires consecutive bars; no stops, funding, order fill, or portfolio is simulated.",
        f"Round-trip cost assumptions: {COSTS[0]} bps base and {COSTS[1]} bps adverse. "
        "Not exchange P&L and not added to the paper ledger.",
        f"Public Hyperliquid data: {len(candles)} assets available. Missing assets: {', '.join(failures) or 'none'}.",
        "", "| Slice / category | Modeled / recorded | Mean net base bps | Mean net adverse bps | Base positive paths |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, lo, hi in (("Earlier", start, split), ("Later", split, end)):
        cohort = [d for d in decisions if lo <= dt(d["created_at"]) < hi]
        paths, missing = simulated_paths(cohort, ledger, candles, hi)
        for bucket, values in sorted(paths.items()):
            known = [x for x in values if x is not None]
            base = f"{statistics.fmean(known) - COSTS[0]:.1f}" if known else "Unmeasured"
            adverse = f"{statistics.fmean(known) - COSTS[1]:.1f}" if known else "Unmeasured"
            lines.append(f"| {label}: {bucket} | {len(known)} / {len(values)} | {base} | {adverse} | "
                         f"{sum(x > COSTS[0] for x in known)} |")
        if missing:
            lines.append(f"| {label}: missing ({', '.join(f'{k}={v}' for k, v in sorted(missing.items()))}) | — | — | — | — |")
    lines += ["", "## Promotion decision", "**Not eligible for promotion.** Retrospective evidence is not a prospective holdout. "
              "The new decision roster must first collect complete observations, then a fixed candidate needs "
              "a separately observed test period and operator approval. No trading settings were changed.", ""]
    return "\n".join(lines)


def _by_source(trades):
    groups = defaultdict(list)
    for trade in trades:
        groups[trade["source"]].append(trade)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--start", type=dt, required=True)
    parser.add_argument("--split", type=dt, required=True)
    parser.add_argument("--end", type=dt, required=True)
    parser.add_argument("--candles", type=Path, help="Read archived public candle evidence rather than downloading")
    parser.add_argument("--save-candles", type=Path, help="Write the public input for reproducible reruns")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prospective", action="store_true", help="Also screen the pre-recorded shadow study; run as a module")
    args = parser.parse_args()
    if not args.start < args.split < args.end <= datetime.now(timezone.utc) or args.end - args.start > timedelta(days=30):
        parser.error("choose a past window of at most 30 days and a split strictly within it")
    paths = [p.resolve() for p in (args.db, args.candles, args.save_candles, args.output) if p]
    if len(set(paths)) != len(paths):
        parser.error("input and output paths must be distinct")
    decisions, ledger = load_readonly(args.db, args.start, args.end)
    if args.candles:
        data = json.loads(args.candles.read_text(encoding="utf-8"))
        candles = {coin: {int(t): value for t, value in rows.items()} for coin, rows in data["candles"].items()}
        failures = data["failures"]
    else:
        assets = {d["coin"] for d in decisions if d["coin"]}
        candles, failures = market_candles(assets, args.start, args.end)
    if args.save_candles:
        args.save_candles.write_text(json.dumps({"candles": candles, "failures": failures}), encoding="utf-8")
    body = report(decisions, ledger, candles, failures, args.start, args.split, args.end)
    if args.prospective:
        from src.data.decision_coverage import read_report
        from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY

        evidence = read_report(paper_path=args.db, now=args.end, window_start=args.start)
        prospect = prospective_review(ledger, evidence, args.start, args.end, COPY_CONCENTRATION_STUDY)
        body += f"\n## Prospective paper review screen\n**{prospect['state']}.** Never authorizes live trading.\n"
        body += "\n".join(f"- {reason}" for reason in prospect["reasons"]) + "\n"
        body += (f"- Additional stress: {COSTS[1] - COSTS[0]} bps of entry notional beyond ledger P&L. "
                 f"Baseline {display(prospect['baseline']['pnl'])}; retained {display(prospect['retained']['pnl'])}.\n")
        if prospect["unknown_venue_orders"]:
            body += "- Venue order coverage unmeasured; this screen is paper-only.\n"
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
