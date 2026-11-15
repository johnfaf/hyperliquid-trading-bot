"""Replay three paper exit policies over identical entries and sampled BTC data."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from statistics import fmean

from src.data.paper_exit_feed import MARK_MAX_AGE_MS, MarketSample, SAMPLE_MS, finite, positive


MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
MAX_GAP_MS = 15_000
ATR_PERIOD = 14
RISK_WINDOW = 24 * HOUR_MS
TRAIL_ATR_MULTIPLE = 2.0
TRAIL_FLOOR_PCT = 0.0035
ARM_AFTER_R = 1.0
HEALTHY_SPREAD_BPS = 20.0
HEALTHY_CHECKS = 3
FEE_RATE = 4.5 / 10_000
SLIPPAGE = 5.0 / 10_000
POLICIES = {"fixed_60s": 60_000, "fixed_5s": SAMPLE_MS, "trailing_5s": SAMPLE_MS}


@dataclass(frozen=True)
class ExecutionCosts:
    fee_rate: float = FEE_RATE
    impact: float = SLIPPAGE
    spread_multiple: float = 1.0
    delay_ms: int = 0

    def bid(self, sample: MarketSample) -> float:
        return sample.bid - (sample.ask - sample.bid) * (self.spread_multiple - 1) / 2


BASE_COSTS = ExecutionCosts()
ADVERSE_COSTS = ExecutionCosts(fee_rate=6.0 / 10_000, impact=10.0 / 10_000, spread_multiple=2, delay_ms=5_000)


@dataclass(frozen=True)
class Entry:
    id: int
    at_ms: int
    fill: float
    quantity: float
    hard_stop: float
    target: float

    def __post_init__(self):
        if not all(positive(v) is not None for v in (self.fill, self.quantity, self.hard_stop, self.target)):
            raise ValueError("entry prices and quantity must be finite and positive")
        if not self.hard_stop < self.fill < self.target:
            raise ValueError("expected a long entry with a stop below and target above the entry")


@dataclass(frozen=True)
class Frame:
    sample: MarketSample
    atr: float | None
    one_minute_bps: float | None
    five_minute_bps: float | None


def _complete_minute(samples: list[MarketSample], start: int) -> bool:
    if len(samples) < 10 or not all(sample.ready for sample in samples):
        return False
    stamps = [start, *(s.at_ms for s in samples), start + MINUTE_MS]
    return all(b - a <= 2 * SAMPLE_MS for a, b in zip(stamps, stamps[1:]))


def build_frames(samples: list[MarketSample]) -> list[Frame]:
    """ATR and returns use completed, contiguous one-minute buckets only."""
    frames = []
    ranges: deque[float] = deque(maxlen=ATR_PERIOD)
    closes: deque[float] = deque(maxlen=6)
    bucket: list[MarketSample] = []
    minute = None
    last_ms = None
    for sample in samples:
        if last_ms is not None and sample.at_ms <= last_ms:
            raise ValueError("market samples must be strictly chronological")
        current = sample.at_ms // MINUTE_MS * MINUTE_MS
        if minute is not None and current != minute:
            if current == minute + MINUTE_MS and _complete_minute(bucket, minute):
                marks = [s.mark for s in bucket]
                if closes:
                    ranges.append(max(max(marks), closes[-1]) - min(min(marks), closes[-1]))
                closes.append(marks[-1])
            else:
                closes.clear()
                ranges.clear()
            bucket = []
        if not sample.ready or (last_ms is not None and sample.at_ms - last_ms > 2 * SAMPLE_MS):
            closes.clear()
            ranges.clear()
        last_ms = sample.at_ms
        minute = current
        bucket.append(sample)
        frames.append(Frame(
            sample, fmean(ranges) if len(ranges) == ATR_PERIOD else None,
            (closes[-1] / closes[-2] - 1) * 10_000 if len(closes) >= 2 else None,
            (closes[-1] / closes[-6] - 1) * 10_000 if len(closes) == 6 else None,
        ))
    return frames


@dataclass
class _FundingEstimate:
    next_hour_ms: int
    cost: float = 0.0
    missing: int = 0

    def observe(self, sample: MarketSample, quantity: float):
        while sample.at_ms >= self.next_hour_ms:
            age = sample.context_age_ms
            receipt_ms = sample.at_ms - age if age is not None and 0 <= age <= MARK_MAX_AGE_MS else None
            if (sample.at_ms - self.next_hour_ms <= MAX_GAP_MS
                    and receipt_ms is not None and abs(receipt_ms - self.next_hour_ms) <= MAX_GAP_MS
                    and finite(sample.funding_rate) is not None and positive(sample.oracle_price) is not None):
                self.cost += quantity * sample.oracle_price * sample.funding_rate
            else:
                self.missing += 1
            self.next_hour_ms += HOUR_MS


@dataclass
class ExitResult:
    reason: str | None = None
    trigger_ms: int | None = None
    exit_ms: int | None = None
    exit_price: float | None = None
    modeled_pnl: float | None = None
    estimated_funding: float = 0.0
    unknown_funding_hours: int = 0
    profit_giveback: float | None = None
    subsequent_5m_rebound: float | None = None
    worst_trade_drawdown: float = 0.0
    material_gap: bool = False
    missing_samples: int = 0
    trail_changes: list[dict] = field(default_factory=list)


def _pnl(entry: Entry, exit_price: float, funding: float, costs: ExecutionCosts) -> float:
    return ((exit_price - entry.fill) - costs.fee_rate * (entry.fill + exit_price)) * entry.quantity - funding


def _profit_floor(entry: Entry, funding: float, mark_to_bid: float, costs: ExecutionCosts) -> float:
    breakeven_bid = (entry.fill * (1 + costs.fee_rate) + max(0.0, funding) / entry.quantity) / (
        (1 - costs.fee_rate) * (1 - costs.impact)
    )
    # Use the observed mark-to-bid gap, not the mark itself, as the sale reference.
    return breakeven_bid + max(0.0, mark_to_bid)


def evaluate_exit(
    entry: Entry, frames: list[Frame], interval_ms: int, *, trailing: bool, costs: ExecutionCosts = BASE_COSTS,
) -> ExitResult:
    result = ExitResult()
    stop = entry.hard_stop
    peak_mark = entry.fill
    peak_pnl = None
    liquid_streak = 0
    last_liquid_book = None
    last_good = entry.at_ms
    next_check = entry.at_ms + interval_ms
    funding = _FundingEstimate((entry.at_ms // HOUR_MS + 1) * HOUR_MS)
    for frame in frames:
        sample = frame.sample
        now = sample.at_ms
        if now <= entry.at_ms:
            continue
        funding.observe(sample, entry.quantity)
        result.estimated_funding = funding.cost
        result.unknown_funding_hours = funding.missing
        ready = (sample.ready and now - sample.book_age_ms >= entry.at_ms
                 and now - sample.context_age_ms >= entry.at_ms)
        bid = costs.bid(sample) if ready else None
        spread_bps = sample.spread_bps * costs.spread_multiple if ready else None
        if not ready:
            result.missing_samples += 1
            result.material_gap |= now - last_good > MAX_GAP_MS
            liquid_streak = 0
            last_liquid_book = None
        else:
            result.material_gap |= now - last_good > MAX_GAP_MS
            if now - last_good > 2 * SAMPLE_MS:
                liquid_streak = 0
                last_liquid_book = None
            last_good = now
            if spread_bps <= HEALTHY_SPREAD_BPS and sample.bid_size >= entry.quantity and bid > 0:
                book_ms = now - sample.book_age_ms
                if last_liquid_book is None or book_ms > last_liquid_book:
                    liquid_streak += 1
                    last_liquid_book = book_ms
            else:
                liquid_streak = 0
                last_liquid_book = None
        check_due = now >= next_check
        if check_due:
            next_check = entry.at_ms + ((now - entry.at_ms) // interval_ms + 1) * interval_ms
        if not ready:
            continue
        if liquid_streak >= HEALTHY_CHECKS:
            peak_mark = max(peak_mark, sample.mark)
        can_fill = bid > 0 and sample.bid_size >= entry.quantity
        exit_price = bid * (1 - costs.impact)
        pnl = _pnl(entry, exit_price, funding.cost, costs)
        if can_fill:
            peak_pnl = max(peak_pnl, pnl) if peak_pnl is not None else pnl
            result.worst_trade_drawdown = max(result.worst_trade_drawdown, max(0.0, peak_pnl) - pnl)
        if check_due and result.trigger_ms is None:
            if sample.mark <= stop:
                result.reason = "trailing_stop" if stop > entry.hard_stop else "hard_stop"
            elif sample.mark >= entry.target:
                result.reason = "take_profit"
            elif now >= entry.at_ms + RISK_WINDOW:
                result.reason = "time_limit"
            if result.reason is not None:
                result.trigger_ms = now
            elif (trailing and frame.atr is not None and can_fill and liquid_streak >= HEALTHY_CHECKS
                    and not funding.missing and pnl >= ARM_AFTER_R * (entry.fill - entry.hard_stop) * entry.quantity):
                distance = max(TRAIL_ATR_MULTIPLE * frame.atr, TRAIL_FLOOR_PCT * entry.fill)
                candidate = max(_profit_floor(entry, funding.cost, sample.mark - bid, costs), peak_mark - distance)
                if stop < candidate < sample.mark:
                    result.trail_changes.append({
                        "at_ms": now, "previous_stop": stop, "new_stop": candidate, "atr": frame.atr,
                        "spread_bps": spread_bps, "one_minute_bps": frame.one_minute_bps,
                        "five_minute_bps": frame.five_minute_bps,
                    })
                    stop = candidate

        # A triggered exit with inadequate top-level depth stays pending until executable; it is not un-triggered.
        if result.trigger_ms is not None and now - result.trigger_ms >= costs.delay_ms and can_fill:
            result.exit_ms, result.exit_price, result.modeled_pnl = now, exit_price, pnl
            result.profit_giveback = max(0.0, peak_pnl - pnl)
            break
    return result


def _post_exit_rebound(entry: Entry, result: ExitResult, frames: list[Frame], times: list[int], costs: ExecutionCosts):
    if result.exit_ms is None:
        return
    expiry = result.exit_ms + 5 * MINUTE_MS
    if not times or times[-1] < expiry:
        return
    following = frames[bisect_right(times, result.exit_ms):bisect_right(times, expiry)]
    if not following:
        return
    stamps = [result.exit_ms, *(f.sample.at_ms for f in following), expiry]
    if any(b - a > MAX_GAP_MS for a, b in zip(stamps, stamps[1:])):
        return
    if all(f.sample.ready and f.sample.bid_size >= entry.quantity and costs.bid(f.sample) > 0 for f in following):
        best = max(costs.bid(f.sample) for f in following) * (1 - costs.impact)
        result.subsequent_5m_rebound = max(0.0, best - result.exit_price) * (1 - costs.fee_rate) * entry.quantity


def compare(
    entries: list[Entry], frames: list[Frame], *, costs: ExecutionCosts = BASE_COSTS,
) -> list[tuple[Entry, dict[str, ExitResult]]]:
    times = [frame.sample.at_ms for frame in frames]
    comparisons = []
    for entry in entries:
        start = bisect_right(times, entry.at_ms)
        # Allow late execution after the nominal time limit, without unbounded scan on broken data.
        end = bisect_left(times, entry.at_ms + RISK_WINDOW + HOUR_MS)
        outcomes = {}
        for name, interval in POLICIES.items():
            outcome = evaluate_exit(entry, frames[start:end], interval, trailing=name == "trailing_5s", costs=costs)
            _post_exit_rebound(entry, outcome, frames, times, costs)
            outcomes[name] = outcome
        comparisons.append((entry, outcomes))
    return comparisons
