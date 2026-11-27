"""Fixed SPY opening-range rule shared by replay and the optional paper pilot."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import math
from pathlib import Path

RULE = "spy_opening_range_30m_v1"
STEP = 300
OPEN_BARS = 6
ENTRY_BARS = 18
HOLD = 1800
SLEEVE = Decimal("75")
MAX_CHASE = Decimal(".0015")
MAX_SPREAD = Decimal(".001")
MIN_CLOSES = 20


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("timestamp without offset")
    return result.timestamp()


def positive(value):
    result = Decimal(str(value))
    if not result.is_finite() or not Decimal(".01") <= result <= 100_000:
        raise ValueError("unusable SPY price")
    return result


@dataclass(frozen=True)
class Bar:
    at: int
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True)
class Signal:
    at: int
    price: Decimal
    stop: Decimal
    range_high: Decimal


def bars_before(rows, opened, closed, through):
    """Ignore partial/future bars and refuse duplicate or malformed completed bars."""
    by_time = {}
    for row in rows:
        at = stamp(row["t"])
        if not opened <= at < closed or at+STEP > through:
            continue
        if at != int(at) or (at-opened) % STEP or at in by_time:
            raise ValueError("duplicate or misaligned bar")
        high,low,last = (positive(row[k]) for k in ("h","l","c"))
        if not low <= last <= high:
            raise ValueError("malformed OHLC")
        by_time[int(at)] = Bar(int(at),high,low,last)
    return by_time


def first_signal(bars, opened, through):
    n = min(ENTRY_BARS, (int(through)-opened)//STEP)
    if n <= OPEN_BARS:
        return None
    window = [bars.get(opened+i*STEP) for i in range(n)]
    if any(bar is None for bar in window[:OPEN_BARS]):
        raise ValueError("opening range has a missing bar")
    high = max(b.high for b in window[:OPEN_BARS])
    low = min(b.low for b in window[:OPEN_BARS])
    prior = window[OPEN_BARS-1]
    for bar in window[OPEN_BARS:]:
        if bar is None:
            raise ValueError("breakout window has a missing bar")
        if prior.close <= high < bar.close:
            return Signal(bar.at+STEP,bar.close,(high+low)/2,high)
        prior = bar
    return None


def exit_decision(bars, signal_at, stop, through):
    """Exit on the first completed stop check or 30-minute time limit."""
    for opening in range(signal_at, min(signal_at+HOLD,int(through)), STEP):
        decision = opening+STEP
        if decision > through:
            break
        bar = bars.get(opening)
        if bar is None:
            return decision, "missing_protection_data"
        if bar.close <= stop:
            return decision, "range_stop"
        if decision == signal_at+HOLD:
            return decision, "time_limit"
    return None


def qualifies(report, *, digest):
    """Structural gate. Neither a forged label nor a small winning sample enables orders."""
    if (not isinstance(report, dict) or report.get("rule") != RULE or report.get("strategy_sha256") != digest
            or report.get("live_approval") is not False or report.get("feed") != "iex"
            or report.get("window_closed") is not True or report.get("capital") != 1000 or report.get("sleeve") != 75):
        return False
    sessions, covered = report.get("sessions"),report.get("covered")
    if (type(sessions) is not int or type(covered) is not int or not 40 <= sessions <= 100
            or not sessions*.9 <= covered <= sessions):
        return False
    try:
        for name in ("base","stress"):
            row = report["results"][name]
            if (any(type(row[k]) is not int for k in ("closed","unresolved"))
                    or not MIN_CLOSES <= row["closed"] <= covered or row["unresolved"] != 0):
                return False
            if not all(type(row[k]) in (int,float) and math.isfinite(row[k])
                       for k in ("net","without_largest","drawdown")):
                return False
        stress = report["results"]["stress"]
        return stress["net"] > 0 and stress["without_largest"] > 0 and 0 <= stress["drawdown"] <= 10
    except (KeyError, TypeError, OverflowError):
        return False


def rule_digest():
    return sha256(Path(__file__).read_bytes()).hexdigest()
