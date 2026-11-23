"""Offline-only screen of one predeclared BTC volatility entry brake."""
from argparse import ArgumentParser
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path
from statistics import median

from scripts.run_forward_shadow import BAR, CAPITAL, COSTS, LOOKBACK, decisions
from scripts.run_hyperliquid_exit_replay import load_market, offline_runtime
from scripts.run_strategy_shadow import window_ms
from src.backtest.candle_exit_replay import aggregate, simulate


def signals_with_brake(bars, signals):
    """Skip risk additions after a range shock. Never change a stop or exit."""
    result, blocked = [], 0
    for i, signal in enumerate(signals):
        if i >= LOOKBACK and signal.side:
            typical = median(b.high-b.low for b in bars[i-LOOKBACK:i])
            if typical <= 0 or bars[i].high-bars[i].low >= 3*typical:
                signal = replace(signal, side=0)
                blocked += 1
        result.append(signal)
    return result, blocked


def screen(markets):
    from scripts.run_forward_shadow import compare

    bars, rates = markets["BTC"]
    result = {"type":"retrospective_diagnostic_only","rule":"btc_range_shock_skip_3x_v1",
              "live_approved":False,"windows":{}}
    for name in ("screen","date_holdout"):
        start,end = window_ms(name)
        visible = [b for b in bars if start-LOOKBACK*BAR <= b.t < end]
        base = compare(visible, rates, start, end, complete=True)
        filtered, blocked = signals_with_brake(visible, decisions(visible))
        comparison = {}
        for cost,(slip,fee) in COSTS.items():
            sim = simulate(visible, filtered, rates, start, end, policy="fixed", slip_bps=slip,
                           fee_rate=fee/10_000, capital=CAPITAL)
            stats = aggregate([sim], CAPITAL)
            comparison[cost] = {"original_net":base[cost]["net"],"braked_net":stats["net"],
                                "original_closes":base[cost]["closed"],
                                "braked_closes":sum(t["exit_reason"]!="window_end" for t in sim["trades"]),
                                "braked_drawdown_pct":stats["drawdown_pct"],"cash":0.,
                                "passive":base[cost]["passive"]}
        result["windows"][name] = {"screened_out_signals":blocked, "costs":comparison}
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = args.market.resolve(strict=True)
    with source.open("rb") as stream:
        digest = file_digest(stream,"sha256").hexdigest()
    with offline_runtime():
        _, _, markets = load_market(source)
        result = screen(markets)
    result["input_sha256"] = digest
    with args.out.open("x",encoding="utf-8") as out:
        json.dump(result,out,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    main()
