#!/usr/bin/env python3
"""Train a two-rule copy selector on prior closes, then score later entries offline."""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data.decision_coverage import open_readonly
from src.trading.copy_shadow_report import (
    EXTRA_COST_BPS, as_dict, clean_close, money, realized, stress_cost, study_peers,
)
from src.trading.trade_decision_memory import is_copy_metadata, number

EMBARGO = timedelta(hours=1)
MIN_TRAIN = 12
MIN_OVERLAPS = 4
MAX_ROWS = 50_000


@dataclass(frozen=True)
class Entry:
    id: int
    at: datetime
    overlaps: bool
    tagged: bool


@dataclass(frozen=True)
class Label:
    at: datetime
    net: float
    stress: float | None


def stamp(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("timezone offset required")
    return dt.astimezone(timezone.utc)


def load(path, start, end):
    with closing(open_readonly(path)) as db:
        rows = db.execute("""
            SELECT id,opened_at,closed_at,coin,side,status,pnl,entry_price,size,metadata FROM paper_trades
            WHERE opened_at<? AND (closed_at IS NULL OR closed_at>=?)
            ORDER BY opened_at,id LIMIT ?
        """, (end.isoformat(), start.isoformat(), MAX_ROWS + 1)).fetchall()
    if len(rows) > MAX_ROWS:
        raise ValueError("ledger exceeds bounded replay size; choose a shorter date range")
    return rows


def point_in_time(rows):
    """Deliver completed closes as events, not as advance knowledge of open positions."""
    actions, labels = [], {}
    for row in rows:
        meta = as_dict(row["metadata"])
        if not is_copy_metadata(meta) or row["side"] not in {"long", "short"} or not row["coin"]:
            continue
        opened = stamp(row["opened_at"])
        closed = stamp(row["closed_at"]) if row["closed_at"] else None
        ident = row["id"]
        key = (row["coin"], row["side"])
        actions.append((opened, 0, ident, key, study_peers(meta)))
        if closed is not None and closed >= opened and row["status"] == "closed":
            actions.append((closed, 1, ident, key, None))
            pnl = number(row["pnl"])
            if pnl is not None and clean_close(meta):
                cost = stress_cost(row)
                labels[ident] = Label(closed, pnl, pnl - cost if cost is not None else None)
    entries = []
    active = defaultdict(set)
    # Opens at the same timestamp are processed in ledger order, before equally timed closes.
    for when, event, ident, key, tagged_peers in sorted(actions):
        if event == 1:
            active[key].discard(ident)
        else:
            overlaps = bool(active[key]) if tagged_peers is None else tagged_peers > 0
            entries.append(Entry(ident, when, overlaps, tagged_peers is not None))
            active[key].add(ident)
    return entries, labels


def pair(entries, labels, cutoff):
    return [(entry, labels[entry.id]) for entry in entries if entry.id in labels and labels[entry.id].at < cutoff]


def metrics(pairs, column):
    ordered = sorted(pairs, key=lambda item: (item[1].at, item[0].id))
    values = [getattr(label, column) for _, label in ordered]
    return realized(values) if None not in values else None


def select(train):
    """The only fitted choice is whether to skip the tagged/reconstructed overlap group."""
    repeats = sum(entry.overlaps for entry, _ in train)
    if len(train) < MIN_TRAIN or repeats < MIN_OVERLAPS:
        return False, "insufficient prior closes or overlaps"
    base = metrics(train, "stress")
    kept = [(entry, label) for entry, label in train if not entry.overlaps]
    candidate = metrics(kept, "stress")
    return (True, "qualified on matured training outcomes") if (
        kept and base and candidate and candidate["net"] > max(0, base["net"])
        and candidate["drawdown"] <= base["drawdown"]
    ) else (False, "no positive improvement under cost stress without worse drawdown")


def evaluate(entries, labels, start, cuts, end):
    folds = []
    for lo, hi in zip(cuts, (*cuts[1:], end)):
        train = pair([e for e in entries if start <= e.at < lo], labels, lo - EMBARGO)
        omit, reason = select(train)
        test = [e for e in entries if lo <= e.at < hi]
        # Freeze every test decision using only Entry fields before reading any test outcomes.
        predicted = {e.id for e in test if not (omit and e.overlaps)}
        scored = pair(test, labels, hi)
        kept = [p for p in scored if not p[0].overlaps]
        selected = [p for p in scored if p[0].id in predicted]
        folds.append({
            "lo": lo, "hi": hi, "train": len(train), "train_overlaps": sum(e.overlaps for e, _ in train),
            "omit": omit, "reason": reason, "test": len(test), "scored": len(scored),
            "tagged": sum(e.tagged for e in test), "baseline": metrics(scored, "net") if scored else None,
            "fixed": metrics(kept, "net") if scored else None, "selected": metrics(selected, "net") if scored else None,
            "baseline_stress": metrics(scored, "stress") if scored else None,
            "fixed_stress": metrics(kept, "stress") if scored else None,
            "selected_stress": metrics(selected, "stress") if scored else None,
            "predictions": predicted,
        })
    return folds


def render(folds, start):
    lines = ["# Copy rule: walk-forward historical screen", "", f"Earliest training entry: {start.isoformat()}.",
             "No prices or outcomes after an entry are supplied to its selector. Training uses clean closes strictly "
             "before the test start minus a one-hour embargo. Each test scores its own entries that closed before the test end. "
             "There are no independent holdout claims for already inspected history.", "",
             f"Minimum training: {MIN_TRAIN} priced closes and {MIN_OVERLAPS} overlaps. Extra cost: +{EXTRA_COST_BPS} bps of entry notional. "
             "The fixed-rule column is a benchmark, not a policy selected using the test labels.", "",
             "| Test interval UTC | Train / repeats | Selected before test | Test scored / entries (tagged) | Recorded | Fixed skip | Selected |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    def cell(value, stress):
        if value is None:
            return "Unmeasured"
        adverse = money(stress['net']) if stress else "unmeasured"
        return f"{money(value['net'])} ({value['n']}); stress {adverse}; DD {money(value['drawdown'])}"

    for fold in folds:
        lo, hi = fold["lo"].isoformat(), fold["hi"].isoformat()
        name = "Skip overlaps" if fold["omit"] else "Keep baseline"
        lines.append(f"| {lo} to {hi} | {fold['train']} / {fold['train_overlaps']} | {name}: {fold['reason']} | "
                     f"{fold['scored']} / {fold['test']} ({fold['tagged']}) | " + " | ".join(
                         cell(fold[k], fold[k + "_stress"]) for k in ("baseline", "fixed", "selected")) + " |")
    lines += ["", "Untagged legacy overlap uses recorded historical entry/exit times as an approximation. Synthetic or "
              "tainted closes can affect observed exposure before exit, but never train or score the selector. "
              "This is not a separate portfolio, arbitrary model training, or an authorization to change trading rules.", ""]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--start", type=stamp, required=True)
    parser.add_argument("--test-start", type=stamp, required=True, action="append", help="A chronological fold start; repeat at least twice")
    parser.add_argument("--end", type=stamp, required=True)
    args = parser.parse_args()
    bounds = [args.start, *args.test_start, args.end]
    if (len(args.test_start) < 2 or any(a >= b for a, b in zip(bounds, bounds[1:]))
            or args.end > datetime.now(timezone.utc) or args.end - args.start > timedelta(days=90)):
        parser.error("use at least two strictly increasing fold starts in a past window of at most 90 days")
    entries, labels = point_in_time(load(args.db, args.start, args.end))
    print(render(evaluate(entries, labels, args.start, args.test_start, args.end), args.start), end="")


if __name__ == "__main__":
    main()
