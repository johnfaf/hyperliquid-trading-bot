"""Analyzer for A4 (CARRY_SHADOW) and A6 (MAKER_SHADOW) log lines.

When the operator flips ``FUNDING_CARRY_SHADOW_ENABLED=true`` and/or
``MAKER_FIRST_SHADOW_ENABLED=true`` in the prod env, the application
logs start populating with shadow lines like::

    INFO ... MAKER_SHADOW [BTC buy] src=copy_trade:0xabc age=2.3s
        mid=68421.12 action=post_alo reason='initial_post'
        target=68421.04
    INFO ... CARRY_SHADOW [BTC] hl↔binance: edge=12.34bps hold=4.0h
        actionable=True veto='' long=hyperliquid short=binance

This script reads such logs (file path or stdin) and produces a
markdown report:

* A6 — action histogram per source class, count of TAKER_FALLBACK
  recommended vs ABANDON-stale, distribution of signal_age_s.
* A4 — count of actionable carry opportunities vs vetoed, average
  edge_bps when actionable, top veto reasons.

The output is the data the operator needs to decide:

* "Is copy_trade signal latency under the maker_only timeout? If
  too many ABANDON-stale, raise the timeout or admit copy_trade
  shouldn't use this lane."
* "Are HL↔CEX edges above the 8 bps min consistently? If yes,
  promote A4 to live wiring."

Usage::

    python scripts/analyze_shadow_logs.py path/to/app.log
    tail -F app.log | python scripts/analyze_shadow_logs.py -

Output: markdown to stdout. Optional ``--json out.json`` for the
machine-readable dump.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional


# Compiled once. The log format is exactly what live_trader.py and
# cross_venue.py emit -- if either format changes, update here.

_RE_MAKER = re.compile(
    r"MAKER_SHADOW \[(?P<coin>\S+) (?P<side>\S+)\] "
    r"src=(?P<src>\S+) age=(?P<age>[-0-9.]+)s "
    r"mid=(?P<mid>[-0-9.eE+]+) action=(?P<action>\S+) "
    r"reason='(?P<reason>[^']*)' target=(?P<target>\S+)"
)

_RE_CARRY = re.compile(
    r"CARRY_SHADOW \[(?P<coin>\S+)\] "
    r"hl[↔⇔↔→]?(?P<cex>\S+): edge=(?P<edge>[-0-9.eE+]+)bps "
    r"hold=(?P<hold>[-0-9.]+)h actionable=(?P<actionable>True|False) "
    r"veto='?(?P<veto>[^']*?)'?\s+long=(?P<long>\S+) short=(?P<short>\S+)"
)


@dataclass
class MakerEvent:
    coin: str
    side: str
    src: str
    age_s: float
    mid: float
    action: str
    reason: str
    target: str


@dataclass
class CarryEvent:
    coin: str
    cex: str
    edge_bps: float
    hold_hours: float
    actionable: bool
    veto: str
    long_venue: str
    short_venue: str


@dataclass
class MakerSummary:
    n_events: int = 0
    by_src_class: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    age_seconds: List[float] = field(default_factory=list)
    reason_counts: Counter = field(default_factory=Counter)
    abandons: int = 0
    taker_fallbacks: int = 0


@dataclass
class CarrySummary:
    n_events: int = 0
    actionable: int = 0
    vetoed: int = 0
    edge_when_actionable: List[float] = field(default_factory=list)
    veto_reasons: Counter = field(default_factory=Counter)
    cex_actionable: Counter = field(default_factory=Counter)
    cex_total: Counter = field(default_factory=Counter)


# ── Parsing ──────────────────────────────────────────────────────────


def parse_maker_line(line: str) -> Optional[MakerEvent]:
    m = _RE_MAKER.search(line)
    if not m:
        return None
    try:
        return MakerEvent(
            coin=m["coin"], side=m["side"], src=m["src"],
            age_s=float(m["age"]), mid=float(m["mid"]),
            action=m["action"], reason=m["reason"], target=m["target"],
        )
    except (TypeError, ValueError):
        return None


def parse_carry_line(line: str) -> Optional[CarryEvent]:
    m = _RE_CARRY.search(line)
    if not m:
        return None
    try:
        return CarryEvent(
            coin=m["coin"], cex=m["cex"], edge_bps=float(m["edge"]),
            hold_hours=float(m["hold"]),
            actionable=(m["actionable"] == "True"),
            veto=m["veto"].strip("'\""),
            long_venue=m["long"], short_venue=m["short"],
        )
    except (TypeError, ValueError):
        return None


def _src_class(src: str) -> str:
    """Group source_key into its class (copy_trade / funding_carry / other)."""
    s = (src or "").lower()
    if s.startswith("copy_trade"):
        return "copy_trade"
    if s.startswith("funding_carry"):
        return "funding_carry"
    if s.startswith("alpha_arena") or s.startswith("strategy"):
        return "alpha_arena"
    if s.startswith("xgboost"):
        return "xgboost"
    return "other"


# ── Aggregation ──────────────────────────────────────────────────────


def summarize(lines: Iterable[str]) -> tuple[MakerSummary, CarrySummary]:
    m_sum = MakerSummary()
    c_sum = CarrySummary()
    for line in lines:
        me = parse_maker_line(line)
        if me is not None:
            m_sum.n_events += 1
            klass = _src_class(me.src)
            m_sum.by_src_class[klass][me.action] += 1
            m_sum.age_seconds.append(me.age_s)
            m_sum.reason_counts[me.reason] += 1
            if me.action == "abandon":
                m_sum.abandons += 1
            elif me.action == "taker_fallback":
                m_sum.taker_fallbacks += 1
            continue
        ce = parse_carry_line(line)
        if ce is not None:
            c_sum.n_events += 1
            c_sum.cex_total[ce.cex] += 1
            if ce.actionable:
                c_sum.actionable += 1
                c_sum.edge_when_actionable.append(ce.edge_bps)
                c_sum.cex_actionable[ce.cex] += 1
            else:
                c_sum.vetoed += 1
                if ce.veto:
                    c_sum.veto_reasons[ce.veto] += 1
    return m_sum, c_sum


# ── Rendering ────────────────────────────────────────────────────────


def render(m: MakerSummary, c: CarrySummary) -> str:
    out: List[str] = []
    out.append("# Shadow log analysis\n")

    out.append(f"## A6 — MAKER_SHADOW ({m.n_events} events)\n")
    if m.n_events == 0:
        out.append("No MAKER_SHADOW events. Confirm MAKER_FIRST_SHADOW_ENABLED=true.\n")
    else:
        out.append("### Action histogram by source class\n")
        out.append("| source class | post_alo | hold | repost_at_bbo | taker_fallback | abandon | filled |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for klass, counts in sorted(m.by_src_class.items()):
            row = [klass]
            for a in ("post_alo", "hold", "repost_at_bbo", "taker_fallback", "abandon", "filled"):
                row.append(str(counts.get(a, 0)))
            out.append("| " + " | ".join(row) + " |")
        out.append("")
        if m.age_seconds:
            out.append(f"signal_age_s — mean={mean(m.age_seconds):.2f}, "
                       f"median={median(m.age_seconds):.2f}, "
                       f"max={max(m.age_seconds):.2f}\n")
        out.append(f"Total ABANDON: **{m.abandons}** "
                   f"({m.abandons / max(m.n_events,1) * 100:.1f}%) — these would have been "
                   "rejected as stale; current production submits anyway.")
        out.append(f"Total TAKER_FALLBACK: **{m.taker_fallbacks}** "
                   f"({m.taker_fallbacks / max(m.n_events,1) * 100:.1f}%) — these would be "
                   "the only times A6 takes liquidity.\n")
        out.append("### Top decision reasons")
        for reason, n in m.reason_counts.most_common(8):
            out.append(f"- `{reason}` — {n}")
        out.append("")

    out.append(f"## A4 — CARRY_SHADOW ({c.n_events} events)\n")
    if c.n_events == 0:
        out.append("No CARRY_SHADOW events. Confirm FUNDING_CARRY_SHADOW_ENABLED=true "
                   "and that cross_venue.confirm_signal is running with HL + ≥1 CEX in "
                   "funding_rates.\n")
    else:
        rate = c.actionable / max(c.n_events, 1)
        out.append(f"Actionable: **{c.actionable}** / {c.n_events} ({rate * 100:.1f}%)")
        if c.edge_when_actionable:
            edges = c.edge_when_actionable
            out.append(f"Edge bps when actionable — mean={mean(edges):.2f}, "
                       f"median={median(edges):.2f}, max={max(edges):.2f}, "
                       f"min={min(edges):.2f}\n")
        if c.cex_total:
            out.append("### Per-CEX hit rate\n")
            out.append("| CEX | actionable | total | rate |")
            out.append("|---|---:|---:|---:|")
            for cex, total in c.cex_total.most_common():
                a = c.cex_actionable.get(cex, 0)
                pct = a / max(total, 1) * 100
                out.append(f"| {cex} | {a} | {total} | {pct:.1f}% |")
            out.append("")
        if c.veto_reasons:
            out.append("### Top veto reasons")
            for reason, n in c.veto_reasons.most_common(8):
                out.append(f"- `{reason}` — {n}")
            out.append("")
    return "\n".join(out)


# ── Entrypoint ──────────────────────────────────────────────────────


def _iter_lines(path: str) -> Iterable[str]:
    if path == "-":
        yield from sys.stdin
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        yield from f


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to the log file, or '-' for stdin.")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Optional path to dump structured summary as JSON.")
    args = parser.parse_args()

    m_sum, c_sum = summarize(_iter_lines(args.log_path))
    print(render(m_sum, c_sum))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "maker": {
                "n_events": m_sum.n_events,
                "abandons": m_sum.abandons,
                "taker_fallbacks": m_sum.taker_fallbacks,
                "by_src_class": {k: dict(v) for k, v in m_sum.by_src_class.items()},
                "age_seconds": m_sum.age_seconds,
                "reason_counts": dict(m_sum.reason_counts),
            },
            "carry": {
                "n_events": c_sum.n_events,
                "actionable": c_sum.actionable,
                "vetoed": c_sum.vetoed,
                "edge_when_actionable": c_sum.edge_when_actionable,
                "veto_reasons": dict(c_sum.veto_reasons),
                "cex_actionable": dict(c_sum.cex_actionable),
                "cex_total": dict(c_sum.cex_total),
            },
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
