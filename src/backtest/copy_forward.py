"""Paired intrahour paper books; only the copy toggle differs."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

RULE = "paired_copy_intrahour_v1"
SYMBOLS = ("BTC", "ETH", "SOL")
LOT = {"BTC": Decimal(".00001"), "ETH": Decimal(".0001"), "SOL": Decimal(".01")}
FEEDS = frozenset(("copy_trade", "strategy", "arena_champion", "options_flow", "polymarket"))
CAPITAL = Decimal("1000")
SLEEVE = Decimal("75")
DAILY_LOSS = Decimal("10")
COSTS = {"base": (5, Decimal("4.5")), "stress": (12, 8)}


@dataclass(frozen=True)
class Candidate:
    coin: str
    side: str
    source: str
    confidence: float
    at: str
    ref: str


@dataclass(frozen=True)
class Quote:
    bid: Decimal
    ask: Decimal
    bid_qty: Decimal
    ask_qty: Decimal
    observed_ms: int
    venue_ms: int


def match(candidates, quotes, *, with_copy, slip, fee, cash):
    for c in sorted(candidates, key=lambda row: (-row.confidence, row.at, row.ref)):
        if c.source == "copy_trade" and not with_copy:
            continue
        quote = quotes[c.coin]
        long = c.side == "long"
        px = quote.ask*(1+slip) if long else quote.bid*(1-slip)
        qty = (min(SLEEVE, cash)/(px*(1+fee))).quantize(LOT[c.coin], rounding=ROUND_DOWN)
        if qty*px >= 10 and qty <= (quote.ask_qty if long else quote.bid_qty):
            return c, px, qty
    return None


def paired(hours, costs):
    """Close both books on the same scheduled window, never on recorded trade fills."""
    slip, fee = (Decimal(x)/10000 for x in costs)
    books = {name: {"cash": CAPITAL, "peak": CAPITAL, "baseline": CAPITAL, "closes": [], "dd": Decimal(0)}
             for name in ("copy_on", "copy_off")}
    last_day = None
    changed = unscored = scored = 0
    trace = []
    for hour, candidates, entry, leave in hours:
        if entry is None or leave is None or candidates is None:
            unscored += 1
            continue
        day = hour//86400
        if day != last_day:
            last_day = day
            for book in books.values():
                book["baseline"] = book["cash"]
        positions = {}
        for name, book in books.items():
            if book["baseline"]-book["cash"] < DAILY_LOSS:
                positions[name] = match(candidates, entry, with_copy=name == "copy_on",
                                        slip=slip, fee=fee, cash=book["cash"])
            else:
                positions[name] = None
        exits = {}
        for name, p in positions.items():
            if p is None:
                continue
            candidate, price, qty = p
            quote = leave[candidate.coin]
            long = candidate.side == "long"
            if qty > (quote.bid_qty if long else quote.ask_qty):
                break
            px = quote.bid*(1-slip) if long else quote.ask*(1+slip)
            fees = (price+px)*qty*fee
            pnl = (px-price)*qty*(1 if long else -1)-fees
            exits[name] = (candidate, pnl, fees)
        else:
            scored += 1
            refs = [p[0].ref if p is not None else None for p in positions.values()]
            changed += refs[0] != refs[1]
            for name, (candidate, pnl, fees) in exits.items():
                book = books[name]
                book["cash"] += pnl
                book["peak"] = max(book["cash"], book["peak"])
                book["dd"] = max(book["dd"], book["peak"]-book["cash"])
                book["closes"].append(pnl)
                trace.append({"hour": hour, "book": name, "ref": candidate.ref, "coin": candidate.coin,
                              "source": candidate.source, "pnl": str(pnl), "fees": str(fees)})
            continue
        # Never give one leg credit when the other leg could not be reconciled.
        unscored += 1
    result = {}
    for name, book in books.items():
        net = book["cash"]-CAPITAL
        closes = book["closes"]
        result[name] = {"net": float(net), "closes": len(closes), "marked_after_exit_drawdown": float(book["dd"]),
                        "net_without_largest": float(net-max([Decimal(0), *closes])),
                        "copy_closes": sum(row["book"] == name and row["source"] == "copy_trade" for row in trace)}
    result.update(delta_off_minus_on=result["copy_off"]["net"]-result["copy_on"]["net"],
                  scored_hours=scored, unscored_hours=unscored, different_hours=changed)
    return result, trace
