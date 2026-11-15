#!/usr/bin/env python3
"""Build a private daily paper-exit comparison from two read-only ledgers."""
from __future__ import annotations

import argparse
import math
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_exit_experiment import _ms, _utc, load_entries, load_samples
from src.backtest.paper_exit_study import (
    ADVERSE_COSTS, BASE_COSTS, HOUR_MS, MAX_GAP_MS, MINUTE_MS, POLICIES, RISK_WINDOW, ExitResult, build_frames, compare,
)
from src.data.paper_exit_feed import SAMPLE_MS, MarketSample


HOLDOUT_START = "2026-11-16T00:00:00Z"
HOLDOUT_END = "2027-01-11T00:00:00Z"
DAY_MS = 24 * HOUR_MS
FOLLOWUP_MS = RISK_WINDOW + HOUR_MS + 5 * MINUTE_MS
MIN_PAIRS = 30
MIN_DAYS = 14
MIN_WEEKS = 6
MIN_READY_FRACTION = 0.99
MIN_MATCH_FRACTION = 0.95
HIGH_ATR_BPS = 5.0
MIN_PER_REGIME = 5


@dataclass(frozen=True)
class Pair:
    at_ms: int
    regime: str
    base: dict[str, ExitResult]
    adverse: dict[str, ExitResult]


@dataclass
class Score:
    expected: int = 0
    samples: Counter = field(default_factory=Counter)
    max_ready_gap_ms: int = 0
    entries: int = 0
    excluded: int = 0
    reasons: Counter = field(default_factory=Counter)
    pairs: list[Pair] = field(default_factory=list)

    def merge(self, other: Score):
        self.expected += other.expected
        self.samples.update(other.samples)
        self.max_ready_gap_ms = max(self.max_ready_gap_ms, other.max_ready_gap_ms)
        self.entries += other.entries
        self.excluded += other.excluded
        self.reasons.update(other.reasons)
        self.pairs.extend(other.pairs)


def _market_coverage(samples: list[MarketSample], start: int, end: int) -> Score:
    if start >= end:
        raise ValueError("scorecard window must have positive duration")
    expected = (end + SAMPLE_MS - 1) // SAMPLE_MS - (start + SAMPLE_MS - 1) // SAMPLE_MS
    result = Score(expected=expected)
    prev = start
    for sample in samples:
        # Use the same bucket alignment as the collector's uniqueness index.
        if sample.at_ms // SAMPLE_MS * SAMPLE_MS < start or sample.at_ms >= end:
            continue
        status = "ready" if sample.ready else sample.status if sample.status != "ready" else "invalid"
        result.samples[status] += 1
        if sample.ready:
            result.max_ready_gap_ms = max(result.max_ready_gap_ms, sample.at_ms - prev)
            prev = sample.at_ms
    if result.samples.total() > expected:
        raise ValueError("recorded observations exceed the scheduled buckets")
    result.max_ready_gap_ms = max(result.max_ready_gap_ms, end - prev)
    return result


def score_window(market: Path, pilot: Path, start: int, entry_end: int, sample_end: int) -> Score:
    if not start < entry_end <= sample_end:
        raise ValueError("entry window must fall within the requested sample window")
    samples = load_samples(market, start, sample_end)
    entries = load_entries(pilot, start, entry_end)
    score = _market_coverage(samples, start, entry_end)
    score.entries = len(entries)
    if not entries:
        return score
    frames = build_frames(samples)
    timestamps = [frame.sample.at_ms for frame in frames]
    base_cohorts = compare(entries, frames, costs=BASE_COSTS)
    adverse_cohorts = compare(entries, frames, costs=ADVERSE_COSTS)
    for (entry, base), (_, adverse) in zip(base_cohorts, adverse_cohorts, strict=True):
        results = (*base.values(), *adverse.values())
        flags = []
        if any(result.exit_ms is None for result in results):
            flags.append("at least one arm unclosed")
        if any(result.material_gap for result in results):
            flags.append("material market-data gap")
        if any(result.unknown_funding_hours for result in results):
            flags.append("missing funding estimate")
        if flags:
            score.excluded += 1
            score.reasons.update(flags)
            continue
        before = bisect_right(timestamps, entry.at_ms) - 1
        regime = "unknown"
        if before >= 0:
            prior = frames[before]
            if prior.atr is not None and prior.sample.ready and entry.at_ms - prior.sample.at_ms <= MAX_GAP_MS:
                regime = "elevated" if prior.atr / prior.sample.mark * 10_000 >= HIGH_ATR_BPS else "lower"
        score.pairs.append(Pair(entry.at_ms, regime, base, adverse))
    return score


def _delta(pair: Pair, scenario: str) -> float:
    arms = getattr(pair, scenario)
    return arms["trailing_5s"].modeled_pnl - arms["fixed_5s"].modeled_pnl


def decision(score: Score) -> tuple[str, list[str]]:
    """Only apply this after the full prospective window and follow-up close."""
    start, end = _ms(HOLDOUT_START), _ms(HOLDOUT_END)
    if any(not start <= p.at_ms < end for p in score.pairs):
        raise ValueError("the frozen gate cannot use entries outside the holdout")
    minimums = []
    days = {p.at_ms // DAY_MS for p in score.pairs}
    by_week = defaultdict(list)
    for pair in score.pairs:
        by_week[(pair.at_ms - start) // (7 * DAY_MS)].append(_delta(pair, "base"))
    regimes = Counter(p.regime for p in score.pairs)
    if len(score.pairs) < MIN_PAIRS or len(days) < MIN_DAYS or len(by_week) < MIN_WEEKS:
        minimums.append("Too few matched exits, entry days or distinct UTC weeks.")
    if score.samples["ready"] < MIN_READY_FRACTION * score.expected or not score.expected:
        minimums.append("Usable capture below 99% of scheduled slots.")
    if not score.entries or len(score.pairs) < MIN_MATCH_FRACTION * score.entries:
        minimums.append("Fewer than 95% of eligible paper entries have paired results in both scenarios.")
    if min(regimes["lower"], regimes["elevated"]) < MIN_PER_REGIME:
        minimums.append("Fewer than five matched exits in either predeclared volatility bucket.")
    if minimums:
        return "Not enough evidence", minimums

    failures = []
    base_delta = math.fsum(_delta(pair, "base") for pair in score.pairs)
    adverse_delta = math.fsum(_delta(pair, "adverse") for pair in score.pairs)
    if round(base_delta, 6) <= 0:
        failures.append("Trailing did not beat five-second fixed exits after base costs.")
    if round(adverse_delta, 6) < 0:
        failures.append("Trailing earned less than five-second fixed exits under adverse execution.")
    if any(round(base_delta - math.fsum(deltas), 6) <= 0 for deltas in by_week.values()):
        failures.append("The base advantage did not survive omitting each entry week in turn.")
    fixed = [pair.base["fixed_5s"] for pair in score.pairs]
    trailing = [pair.base["trailing_5s"] for pair in score.pairs]
    if fmean(x.profit_giveback for x in trailing) >= fmean(x.profit_giveback for x in fixed):
        failures.append("Trailing did not reduce average modeled profit giveback.")
    if max(x.worst_trade_drawdown for x in trailing) > max(x.worst_trade_drawdown for x in fixed):
        failures.append("Trailing increased the worst observed single-trade drawdown.")
    if failures:
        return "Keep fixed exits", failures
    return "Candidate for further paper testing", ["This gate is not live trading approval."]


def _dollars(value: float) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def _arm_lines(pairs: list[Pair], scenario: str) -> list[str]:
    if not pairs:
        return ["No fully matched exits in both scenarios. No P&L comparison yet."]
    lines = [
        "| Exit policy | Net modeled P&L | Avg. giveback | Worst trade drawdown | Delayed fills | Avg. 5m rebound |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in POLICIES:
        outcomes = [getattr(pair, scenario)[name] for pair in pairs]
        rebound = [x.subsequent_5m_rebound for x in outcomes if x.subsequent_5m_rebound is not None]
        delayed = sum(x.exit_ms > x.trigger_ms for x in outcomes)
        formatted_rebound = f"{_dollars(fmean(rebound))} ({len(rebound)}/{len(outcomes)})" if rebound else "n/a"
        lines.append(
            f"| `{name}` | {_dollars(math.fsum(x.modeled_pnl for x in outcomes))} | "
            f"{_dollars(fmean(x.profit_giveback for x in outcomes))} | "
            f"{_dollars(max(x.worst_trade_drawdown for x in outcomes))} | {delayed}/{len(outcomes)} | {formatted_rebound} |"
        )
    faster_delta = math.fsum(
        getattr(pair, scenario)["fixed_5s"].modeled_pnl - getattr(pair, scenario)["fixed_60s"].modeled_pnl for pair in pairs
    )
    trailing_delta = math.fsum(_delta(pair, scenario) for pair in pairs)
    lines.extend(["", f"Faster fixed vs. minute: {_dollars(faster_delta)}. "
                      f"Trailing vs. faster fixed: {_dollars(trailing_delta)}."])
    return lines


def render(score: Score, start: int, end: int, through: int, *, holdout: bool = False) -> str:
    verdict, details = decision(score) if holdout else (
        "Not enough evidence for a strategy decision",
        ["This is a descriptive check. The prospective gate runs only after its fixed window and follow-up."],
    )
    missing = score.expected - score.samples.total()
    regimes = Counter(pair.regime for pair in score.pairs)
    gap_scope = "within any one UTC day" if holdout else "within the requested window"
    lines = [
        "# BTC paper exit scorecard", "", f"**{verdict}.**", "",
        f"Entry window: `{_utc(start)}` to `{_utc(end)}` (end exclusive).",
        f"Market observations through `{_utc(through)}` (exclusive).",
        "",
        *[f"- {detail}" for detail in details],
        "",
        "## Coverage",
        f"- Usable quote slots: {score.samples['ready']:,}/{score.expected:,}; no recorded sample: {missing:,}.",
        f"- Other recorded statuses: {', '.join(f'{k}: {v:,}' for k, v in sorted(score.samples.items()) if k != 'ready') or 'none'}.",
        f"- Largest span between usable quotes or boundaries {gap_scope}: {score.max_ready_gap_ms / 1000:,.1f}s.",
        "- The denominator includes periods before capture. No recorded sample does not by itself prove an outage.",
        f"- Paper entries: {score.entries}; paired in both scenarios: {len(score.pairs)}; excluded: {score.excluded}.",
        f"- Paired entries by starting volatility: {regimes['lower']} lower, {regimes['elevated']} elevated, {regimes['unknown']} unknown.",
    ]
    if score.reasons:
        lines.append("- Exclusion flags (can overlap): " + ", ".join(f"{k}: {v}" for k, v in sorted(score.reasons.items())) + ".")
    lines += [
        "", "## Base costs", *_arm_lines(score.pairs, "base"),
        "", "## Adverse execution", *_arm_lines(score.pairs, "adverse"),
        "", "Adverse case: 2x observed spread, 6 bps fees per leg, 10 bps exit impact, and at least 5 seconds to fill.",
        "Paper entries and exit rules are shared. This is a modeled comparison, not exchange P&L.",
        "Drawdown is per trade, not portfolio drawdown; a post-exit rebound is a diagnostic, not an earned return.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--pilot-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--from", dest="start", type=_ms)
    parser.add_argument("--until", dest="end", type=_ms)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    as_of = int(now.timestamp() * 1000)
    if (args.start is None) != (args.end is None) or (args.holdout and args.start is not None):
        parser.error("use both --from and --until for a historical check, or --holdout alone")
    if args.holdout:
        start, end = _ms(HOLDOUT_START), _ms(HOLDOUT_END)
        through = end + FOLLOWUP_MS
        if as_of < through:
            parser.error(f"prospective gate opens after {_utc(through)}; no interim verdict")
        score = Score()
        for day in range(start, end, DAY_MS):
            score.merge(score_window(args.market_db, args.pilot_db, day, day + DAY_MS, day + DAY_MS + FOLLOWUP_MS))
        filename = "holdout-scorecard.md"
    else:
        if args.start is None:
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = int(midnight.timestamp() * 1000)
            start = end - 7 * DAY_MS
            filename = f"scorecard-{(midnight - timedelta(days=1)).date().isoformat()}.md"
        else:
            start, end = args.start, args.end
            if not start < end <= as_of or end - start > 7 * DAY_MS:
                parser.error("choose a past time range of up to seven days")
            filename = "historical-check.md"
        through = end
        score = score_window(args.market_db, args.pilot_db, start, end, through)

    body = render(score, start, end, through, holdout=args.holdout)
    if args.output_dir:
        target = args.output_dir / filename
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output_dir, delete=False) as tmp:
            temporary = Path(tmp.name)
            tmp.write(body)
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
