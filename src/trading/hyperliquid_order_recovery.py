"""Recover ambiguous Hyperliquid submissions by exchange client-order ID."""
from decimal import Decimal

from src.trading.order_journal import Fill, OrderJournal


_CANCELED = {
    "canceled", "marginCanceled", "vaultWithdrawalCanceled", "openInterestCapCanceled",
    "selfTradeCanceled", "reduceOnlyCanceled", "siblingFilledCanceled", "delistedCanceled",
    "liquidatedCanceled", "scheduledCancel",
}


def recover_submissions(journal: OrderJournal, lookup, fetch_fills, *, limit: int = 8) -> None:
    recent_fills = None
    for cid in journal.unresolved()[:limit]:
        data = lookup(cid)
        if not isinstance(data, dict) or data.get("status") != "order":
            continue  # A negative lookup is not proof an ambiguous order never reached the venue.
        record = data.get("order") or {}
        order = record.get("order") or {}
        if order.get("cloid") != cid or order.get("oid") is None:
            continue
        raw = record.get("status")
        if journal.status(cid) == "cancel_pending" and raw in {"open", "triggered"}:
            continue
        if raw in {"open", "triggered"}:
            status = "accepted"
        elif raw == "filled":
            status = "filled"
        elif raw in _CANCELED:
            status = "canceled"
        elif raw == "rejected" or (isinstance(raw, str) and raw.endswith("Rejected")):
            status = "rejected"
        else:
            continue
        if status != "rejected":
            if not all(key in order for key in ("coin", "side", "origSz", "sz")):
                continue
            original, remaining = Decimal(order["origSz"]), Decimal(order["sz"])
            if not (original.is_finite() and remaining.is_finite() and 0 <= remaining <= original and original > 0):
                continue
            executed = original if status == "filled" else original - remaining
            if executed:
                if recent_fills is None:
                    recent_fills = fetch_fills()
                if not isinstance(recent_fills, list):
                    continue
                matched = [
                    Fill(str(fill["tid"]), str(fill["oid"]), fill["coin"], {"B": "buy", "A": "sell"}[fill["side"]],
                         Decimal(fill["sz"]), Decimal(fill["px"]))
                    for fill in recent_fills if fill.get("oid") == order["oid"]
                    and fill.get("coin") == order["coin"] and fill.get("side") == order["side"]
                ]
                journal.record_fills(matched)
                if sum(fill.quantity for fill in {fill.fill_id: fill for fill in matched}.values()) != executed:
                    continue
        journal.observe(cid, status, str(order["oid"]))
