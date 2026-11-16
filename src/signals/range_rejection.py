"""Frozen candle-only failed-breakout entry rule; no execution or state changes."""
import math

RANGE_BARS = 32             # 8h at 15m
VOLUME_BARS = 96            # 24h at 15m
VOLUME_SPIKE_RATIO = 1.5
EXCURSION_ATR = .10
REENTRY_ATR = .25


def failed_break_side(high, low, close, *, range_high, range_low, prior_atr, volume_ratio):
    """Return +1 for rejected downside, -1 for rejected upside, or 0 to abstain."""
    values = (high, low, close, range_high, range_low, prior_atr, volume_ratio)
    if not all(math.isfinite(x) for x in values) or not 0 < low <= close <= high:
        return 0
    if not 0 < range_low < range_high or prior_atr <= 0 or volume_ratio < VOLUME_SPIKE_RATIO:
        return 0
    if not range_low < close < range_high:
        return 0
    swept_high = high >= range_high + EXCURSION_ATR * prior_atr
    swept_low = low <= range_low - EXCURSION_ATR * prior_atr
    if swept_high == swept_low:  # Do not guess direction on a two-sided candle.
        return 0
    if swept_high and close <= range_high - REENTRY_ATR * prior_atr:
        return -1
    if swept_low and close >= range_low + REENTRY_ATR * prior_atr:
        return 1
    return 0
