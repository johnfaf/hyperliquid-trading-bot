"""Read-only matched copy results, organized by the UTC day of the exit."""
from contextlib import closing
from datetime import datetime, time, timedelta, timezone
import json

from src.data.decision_coverage import open_readonly, read_report
from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY, is_copy_metadata, number

EXTRA_COST_BPS = 16
MAX_CLOSES = 12_000


def as_dict(value):
    try:
        result = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


def study_peers(meta):
    entry = meta.get("entry_decision")
    study = entry.get("copy_concentration_study") if isinstance(entry, dict) else None
    if isinstance(study, dict) and study.get("policy") == COPY_CONCENTRATION_STUDY:
        peers = study.get("peers")
        if type(peers) is int and peers >= 0:
            return peers
    return None


def clean_close(meta):
    closed = meta.get("exit_decision")
    captured_reason = closed.get("reason") if isinstance(closed, dict) else None
    return not (meta.get("tainted") or meta.get("synthetic_reconciliation") or
                "live_reconciled_closed" in (captured_reason, meta.get("close_reason"), meta.get("reconciliation_reason")))


def stress_cost(row):
    price, size = number(row["entry_price"]), number(row["size"])
    if price is None or size is None or min(price, size) <= 0:
        return None
    return number(price * size * EXTRA_COST_BPS / 10_000)


def realized(values):
    net = peak = drawdown = 0.0
    wins = 0
    for value in values:
        net += value
        peak = max(peak, net)
        drawdown = max(drawdown, peak - net)
        wins += value > 0
    return {"n": len(values), "net": net, "drawdown": drawdown, "wins": wins}


def utc_day(day):
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def build_daily(path, day):
    start, end = utc_day(day)
    with closing(open_readonly(path)) as conn:
        rows = conn.execute("""
            SELECT id,closed_at,entry_price,size,pnl,metadata FROM paper_trades
            WHERE status='closed' AND closed_at>=? AND closed_at<? ORDER BY closed_at,id LIMIT ?
        """, (start.isoformat(), end.isoformat(), MAX_CLOSES + 1)).fetchall()
    if len(rows) > MAX_CLOSES:
        raise ValueError("daily paper close limit exceeded")
    all_copy, tracked, kept, skipped, stressed, stressed_kept = [], [], [], [], [], []
    excluded = untagged = unpriced = missing_cost = 0
    for row in rows:
        meta = as_dict(row["metadata"])
        if not is_copy_metadata(meta):
            continue
        if not clean_close(meta):
            excluded += 1
            continue
        pnl = number(row["pnl"])
        if pnl is None:
            unpriced += 1
            continue
        all_copy.append(pnl)
        peers = study_peers(meta)
        if peers is None:
            untagged += 1
            continue
        tracked.append(pnl)
        (skipped if peers else kept).append(pnl)
        cost = stress_cost(row)
        if cost is None:
            missing_cost += 1
        else:
            stressed.append(pnl - cost)
            if not peers:
                stressed_kept.append(pnl - cost)
    coverage = read_report(paper_path=path, now=end, window_start=start)
    return {
        "date": day.isoformat(), "all_copy": realized(all_copy) if all_copy else None,
        "baseline": realized(tracked) if tracked else None,
        "shadow": realized(kept) if tracked else None, "skipped": len(skipped),
        "missed_winners": sum(p > 0 for p in skipped), "excluded": excluded, "untagged": untagged,
        "unpriced": unpriced, "missing_cost": missing_cost, "extra_cost_bps": EXTRA_COST_BPS,
        "stress_baseline": realized(stressed) if tracked and not missing_cost else None,
        "stress_shadow": realized(stressed_kept) if tracked and not missing_cost else None,
        "coverage": ({k: coverage[k] for k in ("cycles", "finished", "expected", "terminal", "errors", "unresolved_intents")}
                     if coverage["available"] and coverage["cycles"] else None),
    }


def money(value):
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


def digest(report):
    lines = [f"Copy comparison (paper) | {report['date']} UTC"]
    actual = report.get("all_copy")
    if actual:
        lines.append(f"All clean copy closes: {money(actual['net'])} / {actual['n']} closes.")
    baseline, shadow = report["baseline"], report["shadow"]
    if baseline is None:
        lines.append("No matched, priced closes; P&L comparison unmeasured.")
    else:
        lines += [f"Recorded: {money(baseline['net'])} / {baseline['n']} closes; drawdown {money(baseline['drawdown'])}.",
                  f"Shadow: {money(shadow['net'])} / {shadow['n']} closes; drawdown {money(shadow['drawdown'])}.",
                  f"Skipped {report['skipped']}; missed {report['missed_winners']} winners."]
    base_stress, shadow_stress = report["stress_baseline"], report["stress_shadow"]
    if base_stress and shadow_stress:
        lines.append(f"+{EXTRA_COST_BPS} bps stress: {money(base_stress['net'])} recorded; {money(shadow_stress['net'])} shadow.")
    coverage = report["coverage"]
    if coverage:
        lines.append(f"Decision coverage: {coverage['terminal']}/{coverage['expected']} final; "
                     f"{coverage['finished']}/{coverage['cycles']} cycles finished at UTC cutoff.")
    else:
        lines.append("Decision coverage unmeasured for this UTC day.")
    if any(report[k] for k in ("untagged", "excluded", "unpriced", "missing_cost")):
        lines.append(f"Unmatched: {report['untagged']} untagged, {report['excluded']} excluded, "
                     f"{report['unpriced']} unpriced, {report['missing_cost']} missing stress costs.")
    return "\n".join(lines)


def markdown(report):
    return "# " + digest(report).replace("\n", "\n\n") + (
        "\n\nCloses are assigned by their UTC exit date. The shadow column removes tagged overlapping entries "
        "from the observed paper book; it does not replace them or simulate a second portfolio. Drawdown is realized "
        "within the day, not mark-to-market. No trading settings are changed.\n"
    )
