"""Replay one frozen BTC strategy on an isolated 1,000 USD paper book."""
from argparse import ArgumentParser
from bisect import bisect_left
from datetime import datetime
from hashlib import file_digest
import json
from pathlib import Path

from scripts.run_hyperliquid_exit_replay import BAR, iso, load_market, offline_runtime
from src.backtest.candle_exit_replay import POSITION_FRACTION, aggregate, simulate
from src.backtest.range_rejection_replay import range_rejection_decisions

RULE = "btc_failed_breakout_fixed_v1"
CAPITAL = 1000.
SCENARIOS = {"base": (5, 4.5), "stress": (12, 6)}
WINDOWS = {
    "screen": ("2026-10-26T00:00:00+00:00", "2026-11-16T13:00:00+00:00"),
    "date_holdout": ("2026-11-16T13:15:00+00:00", "2026-11-22T00:00:00+00:00"),
    "prospective": ("2026-11-23T00:00:00+00:00", "2026-11-30T00:00:00+00:00"),
}


def window_ms(name):
    return tuple(int(datetime.fromisoformat(s).timestamp()*1000) for s in WINDOWS[name])


def passive(bars, funding, *, slip_bps, fee_rate):
    slip = slip_bps / 10_000
    buy, sell = bars[0].open * (1+slip), bars[-1].close * (1-slip)
    units = CAPITAL * POSITION_FRACTION / buy
    # Same settlement and modeled costs as a continuously held long sleeve.
    paid = sum(units * bar.open * funding[bar.t] for bar in bars[1:] if bar.t in funding)
    return (sell-buy)*units - (buy+sell)*units*fee_rate - paid


def evaluate(bars, funding, start, end, trace):
    stamps = [b.t for b in bars]
    lo, hi = bisect_left(stamps, start), bisect_left(stamps, end)
    if lo < 100 or hi <= lo or bars[hi-1].t + BAR != end:
        raise ValueError("frozen window not yet complete or missing warmup")
    # Decisions for each bar use its own close and earlier bars only.
    decisions, _ = range_rejection_decisions(bars[:hi])
    out = {}
    for scenario, (slip, fee_bps) in SCENARIOS.items():
        fee_rate = fee_bps / 10_000
        sim = simulate(bars[:hi], decisions, funding, start, end, policy="fixed",
                       slip_bps=slip, fee_rate=fee_rate, capital=CAPITAL,
                       trace=lambda record: trace({"scenario": scenario, **record}))
        summary = aggregate([sim], CAPITAL)
        completed = [t for t in sim["trades"] if t["exit_reason"] != "window_end"]
        summary.update(cash_net=0., passive_net=passive(bars[lo:hi], funding, slip_bps=slip, fee_rate=fee_rate),
                       completed=len(completed), boundary_closes=len(sim["trades"])-len(completed),
                       entry_days=len({iso(t["entry_bar"])[:10] for t in completed}),
                       net_without_largest=summary["net"]-max([0., *(t["pnl"] for t in sim["trades"])]))
        out[scenario] = summary
    return out


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new, separate report directory")
    parser.add_argument("--prospective", action="store_true", help="Score the fixed forward dates after they complete")
    args = parser.parse_args()
    source = args.market.resolve(strict=True)
    with source.open("rb") as f:
        digest = file_digest(f, "sha256").hexdigest()
    names = ("prospective",) if args.prospective else ("screen", "date_holdout")
    with offline_runtime():
        meta, _, markets = load_market(source)
        if any(meta["end_ms"] < window_ms(name)[1] for name in names):
            parser.error("capture ends before the last frozen window closes")
        args.output.mkdir(mode=0o700)
        report = {"rule": RULE, "capital": CAPITAL, "input_sha256": digest, "source_capture_end": iso(meta["end_ms"]),
                  "costs_per_leg_slippage_and_fee_bps": SCENARIOS, "windows": {}, "approval": False}
        with (args.output / "trace.jsonl").open("x", encoding="utf-8") as trace:
            for name in names:
                start, end = window_ms(name)
                result = evaluate(*markets["BTC"], start, end,
                                  lambda row: trace.write(json.dumps({"window": name, **row}, allow_nan=False)+"\n"))
                report["windows"][name] = {"start": iso(start), "end_exclusive": iso(end), **result}
        with (args.output / "summary.json").open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
