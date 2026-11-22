"""Read-only release checklist; no score or attestation can unlock trading."""
from src.learning.strategy_evidence import MIN_CLOSES
from src.trading.trade_decision_memory import number


def review(evidence, host):
    funding = host.get("paper_funding") or {}
    forward = host.get("forward_shadow") or {}
    backup = host.get("offhost_restore") or {}
    stock = (host.get("stock_pilot") or {}).get("drill") or {}
    cohorts = sum(row.get("state") == "Ready for human review" for row in evidence.get("rows", ()))
    if not evidence.get("available"):
        cohort = "Evidence unavailable"
    else:
        cohort = f"{cohorts} exact cohorts qualify for review" if cohorts else "No qualifying cohort"
    shadow = "Running or awaiting data"
    if forward.get("stale") is True:
        shadow = "Scorecard stale"
    elif forward.get("state") == "Window complete" and forward.get("stale") is False:
        scores = forward.get("results") or {}
        checks = [(scores.get(key) or {}) for key in ("base", "stress")]
        if any(type(row.get("closed")) is not int or row["closed"] < MIN_CLOSES for row in checks):
            shadow = "Too few completed trades"
        elif not all(
            None not in (pnl, cash, without) and min(pnl, without) > cash
            for pnl, cash, without in ((number(row.get("net")), number(row.get("cash")),
                                         number(row.get("net_without_largest"))) for row in checks)
        ):
            shadow = "Fails cash or largest-winner check"
        else:
            shadow = "Passed evidence screen only"
    if funding.get("state") == "Funded closes verified":
        settlement = f"{funding['verified_closes']} cross-hour closes verified"
    else:
        settlement = funding.get("state", "Evidence unavailable")
    return [
        {"check": "Crypto paper strategy evidence", "result": cohort},
        {"check": "Frozen BTC study", "result": shadow},
        {"check": "Paper funding accounting", "result": settlement},
        {"check": "Off-host recovery", "result": (
            "Snapshot older than 7 days" if backup.get("stale") else
            f"Restore verified {backup['verified_at'][:10]}" if backup else "No verified snapshot"
        )},
        {"check": "Stock paper broker round trip", "result": stock.get("state", "No broker evidence")},
        {"check": "Broker outage and restart drill", "result": "Separate venue verification required"},
        {"check": "Live release approval", "result": "Locked"},
    ]
