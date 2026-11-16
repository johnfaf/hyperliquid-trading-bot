"""Build point-in-time failed-breakout features for the matched entry study."""
import math
from statistics import median

from src.backtest.candle_exit_replay import Decision, candle_atr, trailing_regime
from src.signals.range_rejection import RANGE_BARS, VOLUME_BARS, VOLUME_SPIKE_RATIO, failed_break_side


def range_rejection_decisions(bars):
    atr = candle_atr(bars)
    decisions, ratios = [], []
    for i, bar in enumerate(bars):
        if i < VOLUME_BARS:
            decisions.append(Decision(0, 0., None, "warmup"))
            ratios.append(None)
            continue
        volume_baseline = median(b.volume for b in bars[i - VOLUME_BARS:i])
        prior_atr = float(atr[i - 1])
        reference = bars[i - 1].close
        prior_atr_pct = prior_atr / reference if math.isfinite(prior_atr) and prior_atr > 0 else None
        ratio = bar.volume / volume_baseline if volume_baseline > 0 else None
        side = 0
        if ratio is not None and ratio >= VOLUME_SPIKE_RATIO and prior_atr_pct is not None:
            prior = bars[i - RANGE_BARS:i]
            side = failed_break_side(bar.high, bar.low, bar.close, range_high=max(b.high for b in prior),
                                     range_low=min(b.low for b in prior), volume_ratio=ratio, prior_atr=prior_atr)
        decisions.append(Decision(side, 0., prior_atr_pct, trailing_regime(bars, i)))
        ratios.append(ratio)
    return decisions, ratios
