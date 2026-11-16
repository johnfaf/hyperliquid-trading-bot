"""Frozen, read-only signal definition for the public order-flow study."""
from dataclasses import dataclass
from statistics import median

SLOT_MS = 5_000
MINUTE_MS = 60_000
SLOTS_PER_MINUTE = MINUTE_MS // SLOT_MS
REFERENCE_MINUTES = 10
HISTORY = (REFERENCE_MINUTES + 1) * SLOTS_PER_MINUTE + 1
RULE_ID = "flow_absorption_v1"


@dataclass(frozen=True)
class Frame:
    at: int
    session: str
    state: str
    bid: float | None = None
    ask: float | None = None
    bid_usd: float | None = None
    ask_usd: float | None = None
    bid_depth: float | None = None
    ask_depth: float | None = None
    buy: float = 0.
    sell: float = 0.
    trades: int = 0

    @property
    def mid(self):
        return (self.bid + self.ask) / 2


def absorption(frames):
    """Evaluate a completed minute using only continuous, observed frames up to its boundary."""
    if len(frames) < HISTORY:
        return {"state": "warming_up", "side": 0, "pressure": 0}
    visible = tuple(frames)[-HISTORY:]
    if any(f.state != "ready" or f.session != visible[-1].session for f in visible) or any(
            b.at - a.at != SLOT_MS for a, b in zip(visible, visible[1:])):
        return {"state": "coverage_gap", "side": 0, "pressure": 0}
    previous = visible[1:-SLOTS_PER_MINUTE]
    recent = visible[-SLOTS_PER_MINUTE:]
    anchor = visible[-SLOTS_PER_MINUTE - 1]
    reference = median(sum(f.buy + f.sell for f in previous[i:i + SLOTS_PER_MINUTE])
                       for i in range(0, len(previous), SLOTS_PER_MINUTE))
    buy, sell = sum(f.buy for f in recent), sum(f.sell for f in recent)
    flow = (buy - sell) / (buy + sell) if buy + sell else 0.
    ratio = (buy + sell) / reference if reference else None
    last = recent[-1]
    spread = (last.ask - last.bid) / last.mid * 10_000
    if ratio is None or ratio < 2 or abs(flow) < .60 or sum(f.trades for f in recent) < 3 or spread > 3:
        return {"state": "no_pressure", "side": 0, "pressure": 0, "flow": flow, "ratio": ratio, "spread_bps": spread}
    pressure = 1 if flow > 0 else -1
    progress = pressure * (last.mid / anchor.mid - 1) * 10_000
    extreme = (max if pressure > 0 else min)(f.mid for f in (*recent, anchor))
    rebound = -pressure * (last.mid / extreme - 1) * 10_000
    support = (last.ask_depth if pressure > 0 else last.bid_depth) / (last.ask_depth + last.bid_depth)
    signal = progress <= 2 and rebound >= 1 and support >= .60
    return {"state": "absorption" if signal else "pressure_without_absorption", "side": -pressure if signal else 0,
            "pressure": pressure, "flow": flow, "ratio": ratio, "progress_bps": progress,
            "rebound_bps": rebound, "support": support, "spread_bps": spread}
