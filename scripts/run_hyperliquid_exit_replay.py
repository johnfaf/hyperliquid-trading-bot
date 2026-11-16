#!/usr/bin/env python3
"""Evaluate the bot's candle-only Arena signal with fixed versus ATR stops offline."""
from argparse import ArgumentParser
from contextlib import closing, ExitStack
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from scripts.fetch_hyperliquid_replay_market import INFO, BAR, HOUR, DAY, INTERVAL, iso
from src.backtest.candle_exit_replay import (
    BASE_SLIPPAGE_BPS, POLICIES, STRESS_SLIPPAGE_BPS, Bar, aggregate, arena_decisions, select, simulate,
)
from src.backtest.replay import network_sandbox

CORE = ("BTC", "ETH")
CAPITAL = 1000.


def load_market(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        meta = {row[0]: json.loads(row[1]) for row in db.execute("SELECT key,value FROM capture")}
        if meta.get("format") != 1 or meta.get("interval") != INTERVAL or meta.get("source") != INFO:
            raise ValueError("unsupported capture format/source")
        candles_expected = list(range(meta["first_ms"], meta["end_ms"], BAR))
        first_hour = ((meta["first_ms"] + HOUR - 1) // HOUR) * HOUR
        funding_expected = list(range(first_hour, meta["end_ms"], HOUR))
        ranked = [dict(row) for row in db.execute("SELECT * FROM markets ORDER BY rank")]
        if len(ranked) != 10:
            raise ValueError("capture does not contain the ten ranked markets")
        usable = {}
        for market in ranked:
            coin = market["coin"]
            bars = [Bar(*row) for row in db.execute("""
                SELECT timestamp_ms,open,high,low,close,volume FROM candles WHERE coin=? AND timeframe=? ORDER BY timestamp_ms
            """, (coin, INTERVAL))]
            funding = dict(db.execute("SELECT hour,rate FROM funding WHERE coin=? ORDER BY hour", (coin,)))
            market["candle_gaps"] = len(set(candles_expected).difference(b.t for b in bars))
            market["funding_gaps"] = len(set(funding_expected).difference(funding))
            market["usable"] = not market["error"] and [b.t for b in bars] == candles_expected and list(funding) == funding_expected
            if market["usable"]:
                usable[coin] = (bars, funding)
    if not all(c in usable for c in CORE):
        raise ValueError("BTC and ETH require complete candles and settlement data before baseline evaluation")
    return meta, ranked, usable


def walk_forward(markets, meta, trace=None):
    begin, end = meta["start_ms"], meta["end_ms"]
    days, remainder = divmod(end - begin, DAY)
    if days < 21 or remainder:
        raise ValueError("capture window is not aligned to the predeclared split")
    boundaries = [begin + d * DAY for d in (days // 2, days * 2 // 3, days * 5 // 6, days)]
    prepared = {coin: (bars, arena_decisions(coin, bars), funding) for coin, (bars, funding) in markets.items()}

    def score(coins, start, finish, policy, slip, *, fold=None, scope=None):
        sleeves = []
        for coin in coins:
            listener = (partial(trace, fold, scope, coin, policy) if trace is not None and fold is not None
                        and slip == BASE_SLIPPAGE_BPS else None)
            sleeves.append(simulate(*prepared[coin], start, finish, policy=policy, slip_bps=slip,
                                    capital=CAPITAL / len(coins), trace=listener))
        combined = aggregate(sleeves, CAPITAL)
        combined["per_coin"] = {coin: aggregate([item], CAPITAL / len(coins)) for coin, item in zip(coins, sleeves)}
        return combined

    folds = []
    broad = tuple(c for c in prepared if c not in CORE)
    for index, (start, finish) in enumerate(zip(boundaries, boundaries[1:]), 1):
        cutoff = start - BAR
        training = {policy: score(CORE, begin, cutoff, policy, STRESS_SLIPPAGE_BPS) for policy in POLICIES}
        chosen = select(training)
        fold = {"number": index, "start": start, "end": finish, "training_cutoff": cutoff,
                "chosen": chosen, "training_stress": training, "result": {}}
        for scope, coins in (("BTC+ETH", CORE), ("Other ranked markets", broad)):
            if not coins:
                continue
            fold["result"][scope] = {policy: {
                "base": score(coins, start, finish, policy, BASE_SLIPPAGE_BPS, fold=index, scope=scope),
                "stress": score(coins, start, finish, policy, STRESS_SLIPPAGE_BPS),
            } for policy in POLICIES}
        folds.append(fold)
    return folds


def usd(value):
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


def render(meta, ranked, folds, sha):
    usable = sum(m["usable"] for m in ranked)
    lines = ["# Hyperliquid candle replay: momentum exits", "",
             f"Captured {iso(meta['captured_ms'])}. Replay {iso(meta['start_ms'])} through {iso(meta['end_ms'])} (exclusive).",
             "Universe: current top ten default Hyperliquid crypto perpetuals by 24h notional, not a historical rank. "
             "BTC/ETH were fixed for model selection; other markets are a retrospective robustness check only.",
             f"Usable datasets: {usable}/10. Capture SHA-256: `{sha}`.", "",
             "| Rank | Perpetual | 24h notional | 15m candles | Hourly funding | Excluded / gaps |",
             "| ---: | --- | ---: | ---: | ---: | --- |"]
    for m in ranked:
        quality = "none" if m["usable"] else f"{m['candle_gaps']} candles / {m['funding_gaps']} funding" + ("; fetch error" if m["error"] else "")
        lines.append(f"| {m['rank']} | {m['coin']} | {usd(m['day_notional'])} | {m['bars']:,} | {m['funding_events']:,} | {quality} |")
    lines += ["", "## Method", "",
              "Same Arena `trend_following` entry signal in all runs, with fixed confidence threshold 0.5. No current sentiment, "
              "wallets, ranking features, production firewall, or trained regime model used in the entry selection. "
              "$1,000 split into independent equal sleeves per basket; each position uses 5% of its sleeve at 1x, up to one position per asset.",
              "Fixed stop: 2% from fill. ATR stops: 2x or 3x the Wilder 14-bar ATR fraction from the signal close, bounded to "
              "0.75%-5%; targets are twice the initial risk and do not move. `atr_2x_tighten` starts like ATR 2x, then can "
              "tighten at the next bar after a completed close is at least 1R favorable. It never widens a stop. "
              "Same 18h time limit for all. Market entry at the "
              "following bar's open with adverse slippage, including a stop-gap fill at the open and stop-first on ambiguous bars.",
              "Assumed taker fees 4.5 bps per fill and slippage 5 bps base / 12 bps stress per fill. Historical hourly funding "
              "is applied to positions held at hour boundaries, using the bar open as a notional proxy. Both prices and fillability "
              "are still modeled; candle data cannot tell the intrabar sequence or the order book. Drawdown uses candle-close marks.",
              "Training uses only BTC/ETH and ends a full bar before each test. It requires at least 30 training trades for both "
              "the fixed and candidate rules, positive stressed net improvement, and no worse stressed drawdown. The selected "
              "policy is frozen for each subsequent window. Each window starts flat and marks/closes remaining positions at its end.",
              "", "## Forward windows", "",
              "| Fold / basket | Stop policy | Trades | Base net | Stress net | Base drawdown | Stress drawdown | Stop-first bars |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        for scope, policies in fold["result"].items():
            for name, metrics in policies.items():
                base, stress = metrics["base"], metrics["stress"]
                label = name + (" (frozen selection)" if name == fold["chosen"] else "")
                lines.append(f"| {fold['number']} {scope} | {label} | {base['n']} | {usd(base['net'])} | {usd(stress['net'])} | "
                             f"{base['drawdown_pct']:.2f}% | {stress['drawdown_pct']:.2f}% | {base['ambiguous']} |")
    lines += ["", "### Selection and market regimes", ""]
    for fold in folds:
        totals = "; ".join(f"{name}: {usd(m['net'])}, {m['n']} trades, {m['drawdown_pct']:.2f}% DD"
                           for name, m in fold["training_stress"].items())
        lines.append(f"- Fold {fold['number']} {iso(fold['start'])} to {iso(fold['end'])}: **{fold['chosen']}**. Prior BTC/ETH stressed: {totals}.")
    lines.append("- Regimes are assigned at entry signal time using the trailing 24h return: >=2% rising, <=-2% falling, otherwise choppy.")
    lines += ["", "| Basket / policy (final holdout, base costs) | Rising net / n | Falling net / n | Choppy net / n |",
              "| --- | ---: | ---: | ---: |"]
    for scope, policies in folds[-1]["result"].items():
        for name, result in policies.items():
            groups = result["base"]["regimes"]
            cells = [f"{usd(groups[s]['net'])} / {groups[s]['n']}" if s in groups else "No trades" for s in ("rising", "falling", "choppy")]
            lines.append(f"| {scope} / {name} | " + " | ".join(cells) + " |")
    lines += ["", "| Asset (final holdout, stress net / trades) | Fixed | ATR 2x | ATR 3x | ATR 2x tighten |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for policies in folds[-1]["result"].values():
        for coin in policies["fixed"]["stress"]["per_coin"]:
            cells = []
            for name in POLICIES:
                m = policies[name]["stress"]["per_coin"][coin]
                cells.append(f"{usd(m['net'])} / {m['n']}")
            lines.append(f"| {coin} | " + " | ".join(cells) + " |")
    lines += ["", "The traces record every base-cost forward-window decision and simulated entry/exit, including no-signal and "
              "skipped bars. No policy was changed. This evaluates one price-only entry route, not copy trading or the full bot.", ""]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.market, args.report, args.trace)]
    if len(set(paths)) != 3 or any(p.exists() for p in paths[1:]):
        parser.error("use distinct input and new output paths")
    market_path, report_path, trace_path = paths
    meta, ranked, markets = load_market(market_path)
    with market_path.open("rb") as raw:
        sha = hashlib.file_digest(raw, "sha256").hexdigest()
    created = []
    try:
        with TemporaryDirectory(prefix="hl-exit-replay-") as sandbox, ExitStack() as stack:
            os.environ["HL_BOT_DB"] = str(Path(sandbox) / "offline.db")
            os.environ["DB_BACKEND"] = "sqlite"
            network_sandbox.engage(allow_loopback=False)
            stack.callback(network_sandbox.disengage)
            output = stack.enter_context(trace_path.open("x", encoding="utf-8"))
            created.append(trace_path)

            def trace(fold, scope, coin, policy, event):
                print(json.dumps({"fold": fold, "scope": scope, "coin": coin, "policy": policy, **event}, allow_nan=False), file=output)

            folds = walk_forward(markets, meta, trace)
            with report_path.open("x", encoding="utf-8") as report:
                created.append(report_path)
                report.write(render(meta, ranked, folds, sha))
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print(f"Ran {len(folds)} forward tests on {len(markets)}/10 markets; final selection: {folds[-1]['chosen']}. No trading changes.")


if __name__ == "__main__":
    main()
