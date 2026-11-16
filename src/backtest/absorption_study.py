"""Score a frozen order-flow signal from received quotes, outside the collector."""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math

from src.backtest.candle_entry_replay import block_intervals
from src.signals.flow_absorption import MINUTE_MS, RULE_ID, SLOT_MS

HOUR = 3_600_000
DAY = 24 * HOUR
LOOKBACK = 7 * DAY
HORIZONS = (5, 15)
SIZE_USD = 100.
FEE_BPS = 4.5
HAIRCUTS = (1., 4.)


def same_funding_hour(at, horizon):
    entry = at + SLOT_MS
    return entry // HOUR == (entry + horizon * MINUTE_MS) // HOUR


def round_trip(db, decision, side, horizon):
    """Known-at-decision metadata is separated from the future executable-quote label."""
    start = decision["at"] + SLOT_MS
    stop = start + horizon * MINUTE_MS
    key = (decision["coin"],)
    quotes = db.execute("""
        SELECT at,state,session,bid,ask,bid_usd,ask_usd,book_received_ms FROM frames
        WHERE coin=? AND at IN (?,?) ORDER BY at
    """, (*key, start, stop)).fetchall()
    if len(quotes) != 2 or any(q["state"] != "ready" or q["session"] != decision["session"]
                              or not q["at"] - SLOT_MS < q["book_received_ms"] < q["at"] for q in quotes):
        return None, "quote_missing_or_stale"
    continuous = db.execute("""
        SELECT count(*) FROM frames WHERE coin=? AND at BETWEEN ? AND ? AND session=? AND state='ready'
    """, (*key, decision["at"], stop, decision["session"])).fetchone()[0]
    if continuous != (stop - decision["at"]) // SLOT_MS + 1:
        return None, "feed_gap"
    opening, closing = quotes
    enter = opening["ask" if side > 0 else "bid"]
    exit_ = closing["bid" if side > 0 else "ask"]
    entry_depth = opening["ask_usd" if side > 0 else "bid_usd"]
    exit_depth = closing["bid_usd" if side > 0 else "ask_usd"]
    if entry_depth < SIZE_USD or exit_depth < SIZE_USD * exit_ / enter:
        return None, "insufficient_quoted_depth"
    values = []
    for bps in HAIRCUTS:
        slip = bps / 10_000
        entry_fill = enter * (1 + side * slip)
        exit_fill = exit_ * (1 - side * slip)
        fee = (entry_fill + exit_fill) * FEE_BPS / 10_000
        values.append(((exit_fill - entry_fill) * side - fee) / entry_fill * 10_000)
    return tuple(values), None


def _stratum(row):
    return (row["coin"], row["pressure"], (row["at"] // HOUR % 24) // 8)


def pairs_at_cutoff(rows, start, cutoff, horizon):
    """Pair events based only on their archived decision data; do not read outcomes here."""
    tracks = defaultdict(list)
    known_times = defaultdict(list)
    used, decisions, rejected = defaultdict(set), [], Counter()
    last_exit = {}
    visibility = SLOT_MS + horizon * MINUTE_MS
    for row in rows:
        at = row["at"]
        coin = row["coin"]
        if row["state"] == "pressure_without_absorption" and same_funding_hour(at, horizon):
            key = _stratum(row)
            tracks[key].append(row)
            known_times[key].append(at)
        if row["state"] != "absorption" or not start <= at < cutoff:
            continue
        rejected["signals"] += 1
        if at + visibility >= cutoff:
            rejected["not_mature"] += 1
            continue
        if not same_funding_hour(at, horizon):
            rejected["funding_boundary"] += 1
            continue
        if at <= last_exit.get(coin, -1):
            rejected["overlapping_signal"] += 1
            continue
        last_exit[coin] = at + visibility
        key = _stratum(row)
        ratio = row["features"]["ratio"]
        flow = abs(row["features"]["flow"])
        ranked = []
        hi = bisect_right(known_times[key], at - visibility - SLOT_MS)
        lo = bisect_right(known_times[key], at - LOOKBACK - 1)
        for control in tracks[key][lo:hi]:
            volume_ratio = control["features"]["ratio"] / ratio
            flow_distance = abs(abs(control["features"]["flow"]) - flow)
            if 2 / 3 <= volume_ratio <= 1.5 and flow_distance <= .15:
                ranked.append((abs(math.log(volume_ratio)) + flow_distance, -control["at"], control))
        match = None
        for _, neg_time, control in sorted(ranked):
            exposure = range(-neg_time + SLOT_MS, -neg_time + visibility + SLOT_MS, SLOT_MS)
            if used[coin].isdisjoint(exposure):
                used[coin].update(exposure)
                match = control
                break
        decisions.append((row, match))
    return decisions, rejected


def analyze(db, start, cutoff, horizon):
    rows = [dict(row) for row in db.execute("""
        SELECT * FROM decisions WHERE at>=? AND at<? AND rule=? AND state IN ('absorption','pressure_without_absorption')
        ORDER BY at,coin
    """, (start - LOOKBACK, cutoff, RULE_ID))]
    for row in rows:
        row["features"] = json.loads(row["features"])
    decisions, counts = pairs_at_cutoff(rows, start, cutoff, horizon)
    scores = []
    for signal, control in decisions:
        side = signal["side"]
        observed, reason = round_trip(db, signal, side, horizon)
        if observed is None:
            counts[reason] += 1
            continue
        c = None
        if control is not None:
            c, reason = round_trip(db, control, side, horizon)
            if c is None:
                counts["control_unpriced"] += 1  # Never rematch based on future label availability.
        else:
            counts["unmatched"] += 1
        scores.append({"coin": signal["coin"], "t": signal["at"], "side": side, "base": observed[0],
                       "adverse": observed[1], "control_t": control["at"] if c else None,
                       "control_base": c[0] if c else None, "control_adverse": c[1] if c else None})
    return scores, counts, block_intervals(scores)
