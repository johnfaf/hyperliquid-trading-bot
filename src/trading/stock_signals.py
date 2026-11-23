"""Completed regular-session inputs for the existing stock trend rule."""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def completed_closes(bars, sessions, today):
    previous = [day for day in sessions if day < today][-21:]
    if len(previous) != 21 or len(set(previous)) != 21:
        return None
    closes = {}
    for bar in bars:
        stamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            return None
        day = stamp.astimezone(NY).date().isoformat()
        if day in previous:
            value = Decimal(str(bar["c"]))
            if day in closes or not value.is_finite() or value <= 0:
                return None
            closes[day] = value
    if any(day not in closes for day in previous):
        return None
    return [(day, closes[day]) for day in previous]


def sma20_from_closes(closes):
    prices = [price for _, price in closes]
    return prices[-2]*20 <= sum(prices[:-1]) and prices[-1]*20 > sum(prices[1:]), prices[-1]*20 < sum(prices[1:])


def completed_sma20(bars, sessions, today):
    closes = completed_closes(bars, sessions, today)
    return None if closes is None else sma20_from_closes(closes)
