#!/usr/bin/env python3
"""Screen candle entry signals against earlier matched no-signal events, offline."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from scripts.run_hyperliquid_exit_replay import CORE, load_market, offline_runtime
from scripts.fetch_hyperliquid_replay_market import DAY, HOUR, iso
from src.backtest.candle_entry_replay import (
    FILTERS, HORIZONS, LOOKBACK_MS, block_intervals, fit_filter, match_entries, materialize, summarize,
)
from src.backtest.candle_exit_replay import BASE_SLIPPAGE_BPS, STRESS_SLIPPAGE_BPS, arena_decisions

WEEK = 7 * DAY


def date_ms(raw):
    value = datetime.fromisoformat(raw)
    if value.utcoffset() is None:
        raise ValueError("provide a timezone-aware cutoff")
    stamp = int(value.timestamp() * 1000)
    if stamp % HOUR:
        raise ValueError("use an hour-aligned cutoff")
    return stamp


def windows(meta, forward_from=None):
    begin, end = meta["start_ms"], meta["end_ms"]
    if forward_from is not None:
        if not meta["first_ms"] <= forward_from - LOOKBACK_MS:
            raise ValueError("capture needs at least 14 days of control context before the declared cutoff")
        result = [(t, t + WEEK) for t in range(forward_from, end - WEEK + 1, WEEK)]
        if not result:
            raise ValueError("no complete seven-day window after the declared cutoff")
        return result
    days, remainder = divmod(end - begin, DAY)
    if remainder or days < 28:
        raise ValueError("retrospective screen needs an aligned capture of at least 28 days")
    return [(end - n * WEEK, end - (n - 1) * WEEK) for n in (3, 2, 1)]


def evaluate(markets, ranges, trace=None):
    prepared = {coin: (bars, arena_decisions(coin, bars), funding) for coin, (bars, funding) in markets.items()}
    other = tuple(coin for coin in prepared if coin not in CORE)
    results = {}
    for name, holding in HORIZONS.items():
        used = {coin: set() for coin in prepared}
        weeks = []
        total = defaultdict(list)
        for index, (start, end) in enumerate(ranges, 1):
            scopes = {}
            for scope, coins in (("BTC+ETH", CORE), ("Other ranked markets", other)):
                if not coins:
                    continue
                rows, coverage = [], Counter()
                for coin in coins:
                    bars, decisions, funding = prepared[coin]
                    matches, counts = match_entries(bars, decisions, start, end, holding, used[coin])
                    coverage.update(counts)
                    scored = materialize(coin, bars, decisions, funding, matches, holding, BASE_SLIPPAGE_BPS, STRESS_SLIPPAGE_BPS)
                    if trace:
                        for row in scored:
                            trace({"study": "unfiltered", "week": index, "scope": scope, "horizon": name, **row})
                    rows.extend(scored)
                scopes[scope] = {"rows": rows, "summary": summarize(rows), "coverage": coverage}
                total[scope].extend(rows)
            weeks.append(scopes)
        results[name] = {"weeks": weeks, "total": total}
    used = {coin: set() for coin in CORE}
    core = [prepared[coin] for coin in CORE]
    training = []
    for index, (start, end) in enumerate(ranges, 1):
        choice, comparisons, intervals = fit_filter(core, start, STRESS_SLIPPAGE_BPS)
        rows = []
        for coin in CORE:
            bars, decisions, funding = prepared[coin]
            matches, _ = match_entries(bars, decisions, start, end, HORIZONS["4h"], used[coin], FILTERS[choice])
            scored = materialize(coin, bars, decisions, funding, matches, HORIZONS["4h"], BASE_SLIPPAGE_BPS, STRESS_SLIPPAGE_BPS)
            if trace:
                for row in scored:
                    trace({"study": "trained_filter", "filter": choice, "week": index, "scope": "BTC+ETH", "horizon": "4h", **row})
            rows.extend(scored)
        training.append({"selected": choice, "comparisons": comparisons, "training": intervals, "test": summarize(rows)})
    results["training"] = training
    return results


def bps(value):
    return "Unmeasured" if value is None else f"{value:+.1f}"


def row_metrics(result):
    base, stress = result["base"], result["adverse"]
    return (f"{result['pairs']} / {result['signals']} | {bps(base['signal'])} / {bps(base['control'])} | "
            f"{bps(stress['signal'])} / {bps(stress['control'])} | {bps(stress['lift'])} | {bps(stress['all'])}")


def render(meta, markets, ranges, results, captured_sha, forward_from):
    primary = results["4h"]["total"]["BTC+ETH"]
    summary = summarize(primary)
    intervals = block_intervals(primary)
    confirmed = (forward_from is not None and len(ranges) >= 2 and summary["pairs"] >= 50
                 and summary["days"] >= 14 and intervals is not None and summary["adverse"]["all"] > 0
                 and all(bounds[0] > 0 for bounds in intervals.values()))
    if forward_from is None:
        result = "Retrospective only. This archive was available for prior exit research; no fresh edge established."
    elif confirmed:
        result = "Positive under the declared post-cutoff screen; review dataset provenance and shadow-test. Not approved for trading."
    else:
        result = "Did not meet the post-cutoff screen. Do not change trading rules on this result."
    lines = ["# Hyperliquid entry-edge screen", "", f"**{result}**", "",
             f"Source capture: {iso(meta['captured_ms'])}, SHA-256 `{captured_sha}`. Markets: {', '.join(markets)}.",
             "Primary test: BTC/ETH at a four-hour hold. One hour and the other ranked coins are diagnostics, not extra chances to pass.",
             "The Arena trend-following entry rule is frozen; stops, take profit, and position sizing are deliberately removed. "
             "Returns are per initial traded dollar. Enter at the next open and exit at the scheduled open, with actual historical "
             "hourly funding (open price notional proxy), 4.5 bps taker fee per side, and 5/12 bps adverse fill slippage per side.",
             "", "Each control is an earlier neutral (no-signal) bar on the same asset, assigned the same direction as the signal. "
             "Match the same 8-hour UTC session and trailing 24h regime, with ATR within a factor of 1.5. Choose the closest ATR "
             "relative to the signal within 14 days; tie-break by recency. The control must finish a full bar before the signal. "
             "Skip overlapping signal horizons and never reuse a control holding window within a coin/horizon across scored weeks. "
             "Choose both entries before evaluating any return.", "",
             "The net return and matched lift both matter: beating a losing control is not an edge. A later positive screen needs "
             "at least two complete weeks with 50 matched core pairs and 14 matched UTC days, positive net on all eligible core "
             "signals, and a positive lower band on both matched net and lift. The bootstrap uses 95% exploratory bands "
             "from two-day blocks resampled across the whole basket; it does not make this an independent sample.", "",
             "## Results (basis points per traded dollar)", "",
             "| Horizon / window / basket | Pairs / eligible signals | Base signal / control | Stress signal / control | Stress lift | All eligible signals, stress |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in HORIZONS:
        data = results[name]
        for index, scopes in enumerate(data["weeks"], 1):
            for scope, item in scopes.items():
                lines.append(f"| {name} / Week {index} / {scope} | {row_metrics(item['summary'])} |")
        for scope, total in data["total"].items():
            lines.append(f"| {name} / All weeks / {scope} | {row_metrics(summarize(total))} |")
    lines += ["", "### Primary coverage and uncertainty", ""]
    for index, (start, end) in enumerate(ranges, 1):
        scope = results["4h"]["weeks"][index - 1]["BTC+ETH"]
        c = scope["coverage"]
        lines.append(f"- Week {index}: {iso(start)} to {iso(end)} (exclusive). {c['raw_signals']} raw signals; "
                     f"{c['overlapping_signals']} overlapping, {c['right_censored']} cut off, "
                     f"{c['matched']} matched and {c['unmatched']} eligible but unmatched.")
    if intervals:
        for key, label in (("signal", "matched signal net"), ("lift", "matched lift")):
            lo, hi = intervals[key]
            lines.append(f"- Stressed {label} 95% exploratory band: {bps(lo)} to {bps(hi)} bps.")
    else:
        lines.append("- Uncertainty bands unmeasured: fewer than seven UTC days with matched observations.")
    lines += ["", "### Primary breakdown, descriptive only", "", "| Group | Matched / eligible signals | Stress signal / control | Stress lift |",
              "| --- | ---: | ---: | ---: |"]
    for field in ("coin", "side", "regime"):
        for value in sorted({row[field] for row in primary}):
            part = summarize([r for r in primary if r[field] == value])
            label = {1: "Long", -1: "Short"}[value] if field == "side" else value
            lines.append(f"| {label} | {part['pairs']} / {part['signals']} | {bps(part['adverse']['signal'])} / "
                         f"{bps(part['adverse']['control'])} | {bps(part['adverse']['lift'])} |")
    lines += ["", "### Repeated entry-filter fitting (BTC/ETH, 4h only)", "",
              "Eight predeclared trade filters: all, three confidence thresholds, trend alignment, trend alignment plus confidence, "
              "choppy plus confidence, and low ATR plus confidence. Fit them on two completed seven-day training blocks ending "
              "one bar before each test. Require at least 30 training trades overall, ten in each block, and positive stressed net "
              "in both blocks. Rank on worst training week, then mean. If no candidate qualifies, choose cash. "
              "Only the frozen winner is scored again on the following window; no retry after seeing it.", "",
              "| Week | Chosen before test | Matched / eligible | Base signal / control | Stress signal / control | Stress lift | All eligible signals, stress |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for index, item in enumerate(results["training"], 1):
        choice = item["selected"]
        lines.append(f"| {index} | {choice} | {row_metrics(item['test'])} |")
    lines += ["", "| Test week | Candidate | Earlier week stress bps / n | Later week stress bps / n | Qualified on training |",
              "| --- | --- | ---: | ---: | --- |"]
    for index, item in enumerate(results["training"], 1):
        for candidate in item["comparisons"]:
            cells = " | ".join(f"{bps(x['net'])} / {x['n']}" for x in candidate["segments"])
            lines.append(f"| {index} | {candidate['name']} | {cells} | {'yes' if candidate['qualifies'] else 'no'} |")
    lines.extend(["", "This is a diagnostic of one candle-only signal, with simulated fills. It is not the full bot, a copy trade "
                        "replay, a portfolio return, or authorization to trade. Matching on observed features does not remove "
                        "unmeasured differences between older control bars and signal bars. No trading settings were changed.", ""])
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--forward-from", type=date_ms, help="Predeclared cutoff; score only complete weeks after it")
    args = parser.parse_args()
    market, report, trace_path = (p.resolve() for p in (args.market, args.report, args.trace))
    if len({market, report, trace_path}) != 3 or report.exists() or trace_path.exists():
        parser.error("use distinct input and new output paths")
    meta, _, markets = load_market(market)
    try:
        ranges = windows(meta, args.forward_from)
    except ValueError as exc:
        parser.error(str(exc))
    with market.open("rb") as raw:
        sha = hashlib.file_digest(raw, "sha256").hexdigest()
    created = []
    try:
        with offline_runtime(), trace_path.open("x", encoding="utf-8") as output:
            created.append(trace_path)
            results = evaluate(markets, ranges, lambda event: print(json.dumps(event, allow_nan=False), file=output))
        with report.open("x", encoding="utf-8") as target:
            created.append(report)
            target.write(render(meta, markets, ranges, results, sha, args.forward_from))
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print(f"Entry screen: {len(ranges)} weeks, {len(markets)} markets. No trading changes.")


if __name__ == "__main__":
    main()
