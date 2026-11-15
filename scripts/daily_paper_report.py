#!/usr/bin/env python3
"""Report paper results in UTC. Reads SQLite only; never imports an order client."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def metadata(value) -> dict:
    try:
        data = json.loads(value or "{}")
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def trade_costs(row, meta: dict) -> dict:
    entry, exit_price = number(row["entry_price"]), number(row["exit_price"])
    planned_in, planned_out = number(meta.get("intended_entry_price")), number(meta.get("intended_exit_price"))
    size, lev = number(row["size"]), number(row["leverage"])
    slip = None
    if row["side"] in {"long", "short"} and all(
        n is not None and n > 0 for n in (entry, exit_price, planned_in, planned_out, size, lev)
    ):
        direction = 1 if row["side"] == "long" else -1
        slip = max(0, direction * (entry - planned_in) * size * lev) + max(
            0, direction * (planned_out - exit_price) * size * lev
        )
    gross = number(meta.get("gross_pnl_before_fees", meta.get("gross_pnl")))
    return {
        "Net realized P&L": number(row["pnl"]),
        "Gross before costs (un-slipped quotes)": gross + slip if gross is not None and slip is not None else None,
        "Gross at simulated fills (slippage already included)": gross,
        "Simulated fees": number(meta.get("total_fees_paid", meta.get("fees_paid"))),
        "Modeled slippage (included in simulated fills)": slip,
        "Estimated funding (positive = paid)": number(meta.get("funding_accrued")),
    }


def _money(value: float) -> str:
    return f"${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}"


def _drawdown(equity: list[float]) -> str:
    if len(equity) < 2:
        return "not available (fewer than two marks)"
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(value, peak)
        worst = max(worst, peak - value)
    return _money(worst)


def build_report(path: Path, day: date | None, label: str) -> str:
    lo = day.isoformat() if day else "0001-01-01"
    hi = (day + timedelta(days=1)).isoformat() if day else "9999-12-31"
    with closing(sqlite3.connect(f"{path.resolve(strict=True).as_uri()}?mode=ro", uri=True, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        rows = conn.execute(
            "SELECT coin, side, pnl, entry_price, exit_price, size, leverage, metadata FROM paper_trades "
            "WHERE status='closed' AND closed_at >= ? AND closed_at < ? ORDER BY closed_at, id", (lo, hi),
        ).fetchall()
        open_count = conn.execute(
            "SELECT count(*) FROM paper_trades WHERE opened_at < ? AND (closed_at IS NULL OR closed_at >= ?)",
            (hi, hi),
        ).fetchone()[0]
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        events = Counter()
        if "audit_trail" in tables:
            events = Counter(dict(conn.execute(
                "SELECT action, count(*) FROM audit_trail WHERE timestamp >= ? AND timestamp < ? "
                "AND (action LIKE 'paper_pilot_%' OR action IN ('live_mirror_unconfirmed', 'live_reconciliation_snapshot')) "
                "GROUP BY action", (lo, hi),
            )))
        decision_rejects = None
        if "decision_snapshots" in tables:
            decision_rejects = conn.execute(
                "SELECT count(*) FROM decision_snapshots WHERE created_at >= ? AND created_at < ? "
                "AND (LOWER(COALESCE(firewall_decision, '')) = 'rejected' "
                "OR LOWER(COALESCE(final_status, '')) IN ('rejected', 'blocked', 'paper_rejected'))",
                (lo, hi),
            ).fetchone()[0]
        marks = []
        if "paper_pilot_marks" in tables:
            marks = conn.execute(
                "SELECT equity, passive_reference FROM paper_pilot_marks WHERE observed_at >= ? AND observed_at < ? "
                "ORDER BY observed_at", (lo, hi),
            ).fetchall()
        spec = None
        if "bot_state" in tables:
            saved = conn.execute("SELECT value FROM bot_state WHERE key='paper_pilot_spec'").fetchone()
            spec = metadata(saved[0]) if saved else None

    amounts: dict[str, list[float]] = {}
    reasons, counts = Counter(), Counter()
    for row in rows:
        meta = metadata(row["metadata"])
        if (meta.get("tainted") or meta.get("synthetic_reconciliation")
                or "live_reconciled_closed" in (meta.get("close_reason"), meta.get("reconciliation_reason"))):
            counts["excluded"] += 1
            continue
        counts["included"] += 1
        reasons[str(meta.get("close_reason") or "unknown")] += 1
        for key, value in trade_costs(row, meta).items():
            amounts.setdefault(key, [])
            if value is not None:
                amounts[key].append(value)

    lines = [
        f"# {label}",
        f"Period: {day.isoformat() + ' UTC' if day else 'all recorded history (UTC)'}",
        "",
        "**Simulated results only.** No venue statement or account return is reported here.",
        f"- Closed paper trades: {counts['included']}; synthetic or tainted closes excluded: {counts['excluded']}.",
        f"- Open paper trades at the end of the period or latest recorded state: {open_count}.",
    ]
    if spec and number(spec.get("starting_cash")) is not None:
        lines.append(f"- Frozen profile: `{spec.get('strategy', 'unknown')}`; starting paper cash: {_money(spec['starting_cash'])}.")
    lines += ["", "## Cost breakdown", "| Measure | USD | Coverage |", "| --- | ---: | ---: |"]
    for key, values in amounts.items():
        lines.append(f"| {key} | {_money(math.fsum(values)) if values else 'n/a'} | {len(values)}/{counts['included']} |")
    if not amounts:
        lines.append("| No eligible closes | n/a | 0/0 |")
    lines += [
        "", "Slippage is not subtracted twice. Spread and queue impact are not independently estimated.",
        "Missing fields are not counted as zero; recorded funding is simulated, not a statement amount.",
        "", "## Risk and operations",
        f"- Sampled mark-to-market drawdown: {_drawdown([row[0] for row in marks])} ({len(marks)} marks).",
    ]
    if marks:
        lines.append(f"- Latest estimated equity: {_money(marks[-1][0])}; fixed-allocation BTC price reference: "
                     f"{_money(marks[-1][1])}.")
        if spec and number(spec.get("starting_cash")) is not None:
            lines.append(f"- Flat cash reference: {_money(spec['starting_cash'])}.")
    if reasons:
        lines.append("- Close reasons: " + ", ".join(f"{key}: {value}" for key, value in sorted(reasons.items())) + ".")
    if decision_rejects is not None:
        lines.append(f"- Rejected or blocked signal decisions: {decision_rejects} (not exchange rejections).")
    if events:
        lines.extend(f"- `{key}`: {value}" for key, value in sorted(events.items()))
    else:
        lines.append("- No tracked pilot or live execution exceptions recorded for this period.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--day", type=date.fromisoformat, help="YYYY-MM-DD UTC; default is yesterday")
    period.add_argument("--all-history", action="store_true")
    parser.add_argument("--label", default="Daily paper trading report")
    parser.add_argument("--output", type=Path, help="Use {day} in the filename for one report per UTC day")
    args = parser.parse_args()
    day = None if args.all_history else args.day or (datetime.now(timezone.utc).date() - timedelta(days=1))
    report = build_report(args.db, day, args.label)
    if args.output is None:
        print(report, end="")
        return
    out = Path(str(args.output).replace("{day}", day.isoformat() if day else "all"))
    if out.resolve() == args.db.resolve():
        parser.error("--output must not be the database")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(report)
    try:
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
