"""Matched, point-in-time evaluation of directional entries without stop policies."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from random import Random
from statistics import fmean

from src.backtest.candle_exit_replay import BAR_MS, HOUR_MS, TAKER_FEE

DAY_MS = 24 * HOUR_MS
LOOKBACK_MS = 14 * DAY_MS
VOL_RATIO = 1.5
HORIZONS = {"4h": 16, "1h": 4}
WEEK_MS = 7 * DAY_MS


@dataclass(frozen=True)
class Match:
    signal: int
    control: int | None


def _stratum(bar, decision):
    session = ((bar.t + BAR_MS) // HOUR_MS % 24) // 8
    return decision.regime, session


def eligible_entries(bars, decisions, start, end, horizon, accept=None):
    if len(bars) != len(decisions) or horizon < 1:
        raise ValueError("invalid input or holding horizon")
    entries = []
    counts = Counter()
    last_exit = -1
    for i, (bar, d) in enumerate(zip(bars, decisions)):
        if not d.side or not start <= bar.t + BAR_MS < end:
            continue
        counts["raw_signals"] += 1
        if accept is not None and not accept(d):
            counts["filtered"] += 1
            continue
        exit_idx = i + 1 + horizon
        if exit_idx >= len(bars) or bars[exit_idx].t >= end:
            counts["right_censored"] += 1
            continue
        if i + 1 <= last_exit:
            counts["overlapping_signals"] += 1
            continue
        last_exit = exit_idx
        entries.append(i)
    return entries, counts


def match_entries(bars, decisions, start, end, horizon, used_controls, accept=None):
    """Choose pairs using past entry covariates only. Outcomes are not an input."""
    entries, counts = eligible_entries(bars, decisions, start, end, horizon, accept)
    neutral = defaultdict(list)
    for i, (bar, d) in enumerate(zip(bars, decisions)):
        if d.side == 0 and d.atr_pct and math.isfinite(d.atr_pct) and d.atr_pct > 0:
            neutral[_stratum(bar, d)].append(i)
    result = []
    earliest = bars[0].t
    max_age_bars = LOOKBACK_MS // BAR_MS
    for i in entries:
        bar, d = bars[i], decisions[i]
        control = None
        if d.atr_pct and math.isfinite(d.atr_pct) and d.atr_pct > 0 and bar.t - LOOKBACK_MS >= earliest:
            available = neutral[_stratum(bar, d)]
            lo = bisect_left(available, i - max_age_bars)
            # The historical control exit must be at least one bar before this decision.
            hi = bisect_right(available, i - 1 - horizon)
            ranked = []
            for j in available[lo:hi]:
                relative = decisions[j].atr_pct / d.atr_pct
                if 1 / VOL_RATIO <= relative <= VOL_RATIO:
                    ranked.append((abs(math.log(relative)), -j, j))
            for _, _, j in sorted(ranked):
                holding_bars = range(j + 1, j + horizon + 2)
                if used_controls.isdisjoint(holding_bars):
                    used_controls.update(holding_bars)
                    control = j
                    break
        counts["matched" if control is not None else "unmatched"] += 1
        result.append(Match(i, control))
    return result, counts


def forward_bps(bars, funding, index, side, horizon, slippage_bps):
    """Price after the signal plus pre-scheduled exit; normalized by initial fill notional."""
    if side not in (-1, 1):
        raise ValueError("missing directional signal")
    opened, closed = index + 1, index + 1 + horizon
    entry, exit_ = bars[opened].open, bars[closed].open
    slip = slippage_bps / 10_000
    entry_fill, exit_fill = entry * (1 + side * slip), exit_ * (1 - side * slip)
    funding_debit = math.fsum(bars[j].open * funding[bars[j].t] * side
                              for j in range(opened + 1, closed + 1) if bars[j].t % HOUR_MS == 0)
    fees = (entry_fill + exit_fill) * TAKER_FEE
    return ((exit_fill - entry_fill) * side - fees - funding_debit) / entry_fill * 10_000


def materialize(coin, bars, decisions, funding, matches, horizon, base_cost, adverse_cost):
    rows = []
    for match in matches:
        i, j = match.signal, match.control
        d = decisions[i]
        values = {"coin": coin, "t": bars[i].t + BAR_MS, "side": d.side, "regime": d.regime,
                  "signal_atr_pct": d.atr_pct, "control_t": bars[j].t + BAR_MS if j is not None else None,
                  "control_atr_pct": decisions[j].atr_pct if j is not None else None}
        for name, slip in (("base", base_cost), ("adverse", adverse_cost)):
            values[name] = forward_bps(bars, funding, i, d.side, horizon, slip)
            values[f"control_{name}"] = forward_bps(bars, funding, j, d.side, horizon, slip) if j is not None else None
        rows.append(values)
    return rows


def summarize(rows):
    summary = {"signals": len(rows)}
    pairs = [r for r in rows if r["control_t"] is not None]
    summary["pairs"] = len(pairs)
    summary["days"] = len({r["t"] // DAY_MS for r in pairs})
    for cost in ("base", "adverse"):
        summary[cost] = {
            "all": fmean(r[cost] for r in rows) if rows else None,
            "signal": fmean(r[cost] for r in pairs) if pairs else None,
            "control": fmean(r[f"control_{cost}"] for r in pairs) if pairs else None,
            "lift": fmean(r[cost] - r[f"control_{cost}"] for r in pairs) if pairs else None,
        }
    return summary


def block_intervals(rows, *, draws=3000):
    """Exploratory 95% bands, resampling both assets together in two-day blocks."""
    days = defaultdict(lambda: [0., 0., 0])
    for row in rows:
        control = row["control_adverse"]
        if control is not None:
            summary = days[row["t"] // DAY_MS]
            summary[0] += row["adverse"]
            summary[1] += row["adverse"] - control
            summary[2] += 1
    if len(days) < 7:
        return None
    ordered = [days.get(day, [0., 0., 0]) for day in range(min(days), max(days) + 1)]
    n_days = len(ordered)
    random = Random(42)
    results = [[], []]
    for _ in range(draws):
        sampled = []
        for _ in range((n_days + 1) // 2):
            i = random.randrange(n_days)
            sampled.extend((ordered[i], ordered[(i + 1) % n_days]))
        selected = sampled[:n_days]
        n = sum(row[2] for row in selected)
        if n:
            for j, values in enumerate(results):
                values.append(math.fsum(row[j] for row in selected) / n)
    if not all(results):
        return None
    bands = {}
    for key, values in zip(("signal", "lift"), results):
        ranked = sorted(values)
        bands[key] = (ranked[int(.025 * (len(ranked) - 1))], ranked[math.ceil(.975 * (len(ranked) - 1))])
    return bands


def _aligned(decision):
    return (decision.side == 1 and decision.regime == "rising") or (decision.side == -1 and decision.regime == "falling")


FILTERS = {
    "all": lambda _: True,
    "confidence_053": lambda d: d.confidence >= .53,
    "confidence_056": lambda d: d.confidence >= .56,
    "confidence_060": lambda d: d.confidence >= .60,
    "trend_aligned": _aligned,
    "aligned_confidence_053": lambda d: _aligned(d) and d.confidence >= .53,
    "choppy_confidence_053": lambda d: d.regime == "choppy" and d.confidence >= .53,
    "low_atr_confidence_053": lambda d: d.atr_pct is not None and d.atr_pct <= .012 and d.confidence >= .53,
    "cash": lambda _: False,
}


def fit_filter(core, test_start, adverse_cost):
    """Compare predeclared filters on two completed weeks; never receive test returns."""
    cutoff = test_start - BAR_MS
    ranges = ((cutoff - 2 * WEEK_MS, cutoff - WEEK_MS), (cutoff - WEEK_MS, cutoff))
    if any(bars[0].t >= ranges[0][0] for bars, _, _ in core):
        return "cash", [], ranges
    summaries, qualified = [], []
    for name, accept in FILTERS.items():
        if name == "cash":
            continue
        segments = []
        for start, end in ranges:
            net = []
            for bars, decisions, funding in core:
                indices, _ = eligible_entries(bars, decisions, start, end, HORIZONS["4h"], accept)
                net.extend(forward_bps(bars, funding, i, decisions[i].side, HORIZONS["4h"], adverse_cost) for i in indices)
            segments.append({"n": len(net), "net": fmean(net) if net else None})
        qualifies = (sum(s["n"] for s in segments) >= 30
                     and all(s["n"] >= 10 and s["net"] > 0 for s in segments))
        summaries.append({"name": name, "segments": segments, "qualifies": qualifies})
        if qualifies:
            qualified.append((min(s["net"] for s in segments), fmean(s["net"] for s in segments), name))
    return (max(qualified)[-1] if qualified else "cash"), summaries, ranges
