"""Offline Arena entry replay with causal order timing and alternative exit distances."""
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
import math


BAR_MS = 900_000
HOUR_MS = 4 * BAR_MS
SIGNAL_WARMUP = 30
ATR_PERIOD = 14
MAX_HOLD_BARS = 18 * 4
TAKER_FEE = 4.5 / 10_000
POSITION_FRACTION = .05
BASE_SLIPPAGE_BPS = 5
STRESS_SLIPPAGE_BPS = 12
POLICIES = {"fixed": None, "atr_2x": 2.0, "atr_3x": 3.0, "atr_2x_tighten": 2.0}


@dataclass(frozen=True)
class Bar:
    t: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Decision:
    side: int
    confidence: float
    atr_pct: float | None
    regime: str


def candle_atr(bars):
    import numpy as np

    from src.backtest.candle_backtester import _atr

    return _atr(*(np.array([getattr(b, name) for b in bars], dtype=float) for name in ("high", "low", "close")), ATR_PERIOD)


def trailing_regime(bars, index):
    reference = bars[max(0, index - 96)].close
    change = bars[index].close / reference - 1
    return "rising" if change >= .02 else "falling" if change <= -.02 else "choppy"


def arena_decisions(coin, bars):
    """Run the bot's champion entry generator on bounded completed-bar prefixes."""
    import numpy as np

    from src.signals.alpha_arena import ArenaAgent, Backtester

    backtester = Backtester()
    agent = ArenaAgent("offline-trend", "Frozen trend following", "trend_following", params={"confidence_threshold": 0.5})
    historical = [{"timestamp": b.t, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
                  for b in bars]
    atr = candle_atr(bars)
    decisions = []
    with np.errstate(divide="ignore"):
        for i, bar in enumerate(bars):
            if i + 1 < SIGNAL_WARMUP:
                decisions.append(Decision(0, 0., None, "warmup"))
                continue
            visible = historical[max(0, i - 99):i + 1]
            signal = backtester._agent_generate_signal(agent, visible, visible[-1], coin=coin)
            decisions.append(Decision({"long": 1, "short": -1}[signal["side"]] if signal else 0,
                                      float(signal["confidence"]) if signal else 0.0,
                                      float(atr[i] / bar.close), trailing_regime(bars, i)))
    return decisions


def stop_fraction(policy, atr_pct):
    multiple = POLICIES[policy]
    if multiple is None:
        return .02
    if atr_pct is None or not math.isfinite(atr_pct) or atr_pct < 0:
        return None
    return min(.05, max(.0075, atr_pct * multiple))


def trigger(bar, side, stop, take):
    """Use the open when it proves a gap; otherwise resolve a two-touch bar against the trade."""
    if (side > 0 and bar.open <= stop) or (side < 0 and bar.open >= stop):
        return "stop_gap", bar.open, False
    if (side > 0 and bar.open >= take) or (side < 0 and bar.open <= take):
        return "take_profit", take, False
    stop_hit = bar.low <= stop if side > 0 else bar.high >= stop
    take_hit = bar.high >= take if side > 0 else bar.low <= take
    if stop_hit:
        return "stop_loss", stop, take_hit
    if take_hit:
        return "take_profit", take, False
    return None


def simulate(bars, decisions, funding, start, end, *, policy, slip_bps, capital, trace=None):
    stamps = [bar.t for bar in bars]
    lo, hi = bisect_left(stamps, start), bisect_left(stamps, end)
    if (hi <= lo or bars[lo].t != start or bars[hi - 1].t + BAR_MS != end
            or any(bars[i].t + BAR_MS != bars[i + 1].t for i in range(lo, hi - 1))):
        raise ValueError("simulation requires an aligned, complete window")
    balance = capital
    position = pending = None
    trades = []
    equity = [capital]
    counts = Counter()
    slip = slip_bps / 10_000

    def log(kind, **data):
        if trace is not None:
            trace({"kind": kind, **data})

    def close_position(price, reason, exit_bar, ambiguous=False):
        nonlocal balance, position
        s, units = position["side"], position["units"]
        fill = price * (1 - s * slip)
        fee = fill * units * TAKER_FEE
        gross = (fill - position["entry_price"]) * units * s
        pnl = gross - position["entry_fee"] - fee - position["funding"]
        balance += gross - fee
        trade = {**position, "exit_bar": exit_bar, "exit_price": fill, "exit_reason": reason,
                 "ambiguous": ambiguous, "exit_fee": fee, "pnl": pnl}
        trades.append(trade)
        log("exit", **trade)
        position = None

    for index in range(lo, hi):
        bar, signal = bars[index], decisions[index]
        if position:
            if bar.t % HOUR_MS == 0:
                debit = position["units"] * bar.open * funding[bar.t] * position["side"]
                position["funding"] += debit
                balance -= debit
            if bar.t >= position["entry_bar"] + MAX_HOLD_BARS * BAR_MS:
                close_position(bar.open, "time_limit", bar.t)
        if pending:
            previous, symbol_signal, distance = pending
            pending = None
            side = symbol_signal.side
            if balance <= 0:
                counts["no_balance"] += 1
                log("rejected", t=bar.t, reason="no_balance")
            else:
                fill = bar.open * (1 + side * slip)
                units = balance * POSITION_FRACTION / fill
                fee = fill * units * TAKER_FEE
                balance -= fee
                position = {"side": side, "entry_bar": bar.t, "signal_bar": previous, "entry_price": fill,
                            "units": units, "stop": fill * (1 - side * distance),
                            "take": fill * (1 + side * 2 * distance), "entry_fee": fee, "funding": 0.,
                            "regime": symbol_signal.regime, "distance": distance}
                log("entry", **position)

        if position:
            hit = trigger(bar, position["side"], position["stop"], position["take"])
            if hit:
                reason, price, ambiguous = hit
                close_position(price, reason, bar.t, ambiguous)
        if index == hi - 1 and position:
            close_position(bar.close, "window_end", bar.t)

        if position and policy == "atr_2x_tighten":
            distance = stop_fraction(policy, signal.atr_pct)
            side, original = position["side"], position["entry_price"]
            if distance is not None and (bar.close - original) * side >= original * position["distance"]:
                next_stop = bar.close * (1 - side * distance)
                if (next_stop - position["stop"]) * side > 0:
                    # The bar has already been matched against the old stop. Apply changes forward only.
                    log("stop_update", t=bar.t + BAR_MS, previous=position["stop"], stop=next_stop)
                    position["stop"] = next_stop

        if not signal.side:
            reason = "warmup" if signal.regime == "warmup" else "no_signal"
        elif position:
            reason = "position_open"
        elif index == hi - 1:
            reason = "no_next_bar_in_window"
        elif (distance := stop_fraction(policy, signal.atr_pct)) is None:
            reason = "atr_unavailable"
        else:
            reason = "queued_next_open"
            pending = (bar.t, signal, distance)
        counts[reason] += 1
        log("decision", t=bar.t + BAR_MS, signal_bar=bar.t, reason=reason,
            side=signal.side, confidence=signal.confidence, atr_pct=signal.atr_pct, regime=signal.regime)

        if position:
            mark = bar.close * (1 - position["side"] * slip)
            unrealized = (mark - position["entry_price"]) * position["units"] * position["side"]
            equity.append(balance + unrealized - mark * position["units"] * TAKER_FEE)
        else:
            equity.append(balance)
    return {"equity": equity, "trades": trades, "counts": counts}


def aggregate(sleeves, capital):
    path = [math.fsum(values) for values in zip(*(item["equity"] for item in sleeves), strict=True)]
    peak, drawdown = capital, 0.
    for mark in path:
        peak = max(peak, mark)
        drawdown = max(drawdown, (peak - mark) / peak)
    trades = [trade for item in sleeves for trade in item["trades"]]
    cohorts = defaultdict(list)
    for trade in trades:
        cohorts[trade["regime"]].append(trade["pnl"])
    counts = Counter()
    for item in sleeves:
        counts.update(item["counts"])
    return {"n": len(trades), "net": path[-1] - capital, "drawdown_pct": drawdown * 100,
            "wins": sum(t["pnl"] > 0 for t in trades), "fees": math.fsum(t["entry_fee"] + t["exit_fee"] for t in trades),
            "funding": math.fsum(t["funding"] for t in trades), "ambiguous": sum(t["ambiguous"] for t in trades),
            "counts": dict(counts), "regimes": {k: {"n": len(v), "net": math.fsum(v)} for k, v in cohorts.items()}}


def select(training, minimum=30):
    baseline = training["fixed"]
    qualified = [name for name, stats in training.items() if name != "fixed" and stats["n"] >= minimum
                 and baseline["n"] >= minimum and stats["net"] > max(0., baseline["net"])
                 and stats["drawdown_pct"] <= baseline["drawdown_pct"]]
    return max(qualified, key=lambda name: (training[name]["net"], -training[name]["drawdown_pct"])) if qualified else "fixed"
