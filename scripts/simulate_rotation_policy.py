#!/usr/bin/env python3
"""
Executive benchmark pack for portfolio/rotation policy quality.

Scenarios:
1) current_main_baseline (hard-cap / no replacement; proxy baseline)
2) branch_rotation_enabled
3) branch_rotation_disabled_same_code
4) lower_leverage_variant
5) smaller_size_variant

Outputs per scenario:
- PnL, Sharpe, max drawdown, turnover
- replacement count, churn cost, rejection mix
- concentration (coin/side/cluster max exposure)
- worst 10 trade clusters
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.trading.portfolio_rotation import PortfolioRotationManager  # noqa: E402


COINS = ["BTC", "ETH", "SOL", "DOGE", "LINK", "ARB", "AVAX", "APT", "INJ", "SUI", "OP", "SEI"]
SIDES = ["long", "short"]
CLUSTERS = {
    "majors": {"BTC", "ETH"},
    "l1": {"SOL", "AVAX", "APT", "SUI", "INJ", "SEI"},
    "infra": {"ARB", "OP", "LINK"},
    "memes": {"DOGE"},
}


@dataclass
class CandidateSignal:
    coin: str
    side: str
    confidence: float
    source_accuracy: float
    quality: float
    entry_price: float
    leverage: float
    position_pct: float


def _cluster_for_coin(coin: str) -> str:
    coin = str(coin or "").upper()
    for cluster, members in CLUSTERS.items():
        if coin in members:
            return cluster
    return coin


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (peak - x) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _sharpe_proxy(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sd = statistics.pstdev(returns)
    if sd <= 1e-12:
        return 0.0
    return (statistics.mean(returns) / sd) * math.sqrt(252)


def _trade_template(
    trade_id: int,
    sig: CandidateSignal,
    now: datetime,
    rng: random.Random,
) -> Dict:
    opened_at = (now - timedelta(minutes=rng.randint(5, 480))).isoformat()
    return {
        "id": trade_id,
        "coin": sig.coin,
        "side": sig.side,
        "entry_price": sig.entry_price,
        "current_price": sig.entry_price,
        "size": max(0.001, (sig.position_pct * 10_000) / max(sig.entry_price, 1.0)),
        "leverage": sig.leverage,
        "opened_at": opened_at,
        "metadata": {
            "confidence": sig.confidence,
            "source_accuracy": sig.source_accuracy,
            "quality": sig.quality,
        },
    }


def _signal_stream(n: int, seed: int, leverage_scale: float, size_scale: float) -> List[CandidateSignal]:
    rng = random.Random(seed)
    stream: List[CandidateSignal] = []
    for _ in range(n):
        coin = rng.choice(COINS)
        side = rng.choice(SIDES)
        confidence = _clamp(rng.uniform(0.22, 0.96), 0.0, 1.0)
        source_accuracy = _clamp(rng.uniform(0.10, 0.95), 0.0, 1.0)
        quality = _clamp(0.50 * confidence + 0.50 * source_accuracy + rng.uniform(-0.15, 0.15), 0.0, 1.0)
        px = rng.uniform(0.1, 100_000.0)
        leverage = _clamp(rng.uniform(1.0, 5.0) * leverage_scale, 1.0, 8.0)
        position_pct = _clamp(rng.uniform(0.02, 0.10) * size_scale, 0.005, 0.20)
        stream.append(
            CandidateSignal(
                coin=coin,
                side=side,
                confidence=confidence,
                source_accuracy=source_accuracy,
                quality=quality,
                entry_price=px,
                leverage=leverage,
                position_pct=position_pct,
            )
        )
    return stream


def _exposure_snapshot(book: List[Dict]) -> Dict[str, float]:
    if not book:
        return {"coin_max": 0.0, "side_max": 0.0, "cluster_max": 0.0}
    notionals = []
    for t in book:
        px = float(t.get("current_price", t.get("entry_price", 0)) or 0)
        notionals.append(max(px * float(t.get("size", 0) or 0) * max(float(t.get("leverage", 1) or 1), 1.0), 0.0))
    total = sum(notionals) or 1.0
    coin_map: Dict[str, float] = {}
    side_map: Dict[str, float] = {}
    cluster_map: Dict[str, float] = {}
    for t, n in zip(book, notionals):
        coin = str(t.get("coin", "")).upper()
        side = str(t.get("side", "")).lower()
        cluster = _cluster_for_coin(coin)
        coin_map[coin] = coin_map.get(coin, 0.0) + n
        side_map[side] = side_map.get(side, 0.0) + n
        cluster_map[cluster] = cluster_map.get(cluster, 0.0) + n
    return {
        "coin_max": max(coin_map.values()) / total if coin_map else 0.0,
        "side_max": max(side_map.values()) / total if side_map else 0.0,
        "cluster_max": max(cluster_map.values()) / total if cluster_map else 0.0,
    }


def _simulate_exit_pnl(sig: CandidateSignal, stop_loss_pct: float, take_profit_pct: float, rng: random.Random) -> float:
    """
    Simulate realized PnL with entry-quality sensitivity and stop/TP shaping.
    """
    base_edge = (sig.quality - 0.5) * 0.20  # +/-10%
    noise = rng.gauss(0, 0.03)
    raw_return = base_edge + noise
    # Apply risk controls
    capped = _clamp(raw_return, -abs(stop_loss_pct), abs(take_profit_pct))
    notional = sig.entry_price * max(sig.position_pct, 0.001) * 100.0 * sig.leverage
    return capped * notional


def run_scenario(
    name: str,
    signals: int,
    seed: int,
    policy: str,
    leverage_scale: float = 1.0,
    size_scale: float = 1.0,
    entry_quality_threshold: float = 0.0,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.10,
) -> Dict:
    """
    policy: hard_cap | rotation_enabled | rotation_disabled_same_code
    """
    mgr = PortfolioRotationManager(
        {
            "target_positions": 8,
            "hard_max_positions": 8,
            "reserved_high_conviction_slots": 2,
            "replacement_threshold": 0.10,
            "max_replacements_per_cycle": 3,
            "max_replacements_per_hour": 6,
            "max_replacements_per_day": 18,
            "forced_exit_cooldown_minutes": 30,
            "round_trip_block_minutes": 15,
        }
    )
    rng = random.Random(seed + abs(hash(name)) % 1_000_000)
    now = datetime.now(timezone.utc)
    stream = _signal_stream(signals, seed, leverage_scale=leverage_scale, size_scale=size_scale)

    trade_id = 1
    book: List[Dict] = []
    trades_closed: List[Dict] = []
    equity = 10_000.0
    equity_curve = [equity]
    turnover = 0.0
    replacement_count = 0
    churn_cost = 0.0
    rejection_mix: Dict[str, int] = {}
    concentration_track = {"coin_max": 0.0, "side_max": 0.0, "cluster_max": 0.0}

    def _reject(reason: str) -> None:
        rejection_mix[reason] = rejection_mix.get(reason, 0) + 1

    for idx, sig in enumerate(stream):
        # economic quality first
        if sig.quality < entry_quality_threshold:
            _reject("entry_quality_gate")
            continue

        if len(book) < mgr.target_positions:
            # fill capacity directly
            t = _trade_template(trade_id, sig, now, rng)
            trade_id += 1
            book.append(t)
            turnover += t["entry_price"] * t["size"] * t["leverage"]
            continue

        if policy == "hard_cap":
            _reject("hard_cap_full")
            continue

        decision = mgr.decide(sig, book, regime_data={}, replacements_used=0)
        if decision.action == "reject":
            _reject(decision.reason)
            continue
        if decision.action == "replace":
            if policy == "rotation_disabled_same_code":
                mgr.record_dry_run_replacement_skip(decision)
                _reject("rotation_disabled")
                continue
            victim_idx = next(
                (i for i, t in enumerate(book) if t.get("id") == decision.replacement_trade_id),
                None,
            )
            if victim_idx is None:
                _reject("replacement_victim_missing")
                continue
            victim = book.pop(victim_idx)
            mgr.register_replacement(victim, sig.coin, sig.side)
            replacement_count += 1
            # churn proxy cost: two-way execution friction on replaced notional
            replaced_notional = victim["entry_price"] * victim["size"] * victim["leverage"]
            friction = replaced_notional * (0.0009 + 0.0006) * 2.0
            churn_cost += friction
            # close victim pnl on replacement event
            victim_sig = CandidateSignal(
                coin=victim["coin"],
                side=victim["side"],
                confidence=float(victim.get("metadata", {}).get("confidence", 0.5)),
                source_accuracy=float(victim.get("metadata", {}).get("source_accuracy", 0.5)),
                quality=float(victim.get("metadata", {}).get("quality", 0.5)),
                entry_price=float(victim.get("entry_price", 0) or 0),
                leverage=float(victim.get("leverage", 1) or 1),
                position_pct=_clamp((victim["size"] * victim["entry_price"]) / 10_000.0, 0.005, 0.20),
            )
            realized = _simulate_exit_pnl(victim_sig, stop_loss_pct, take_profit_pct, rng)
            trades_closed.append(
                {
                    "coin": victim_sig.coin,
                    "side": victim_sig.side,
                    "cluster": _cluster_for_coin(victim_sig.coin),
                    "pnl": realized,
                }
            )
            equity += realized
            equity_curve.append(equity)

        # open candidate
        t = _trade_template(trade_id, sig, now, rng)
        trade_id += 1
        book.append(t)
        turnover += t["entry_price"] * t["size"] * t["leverage"]

        # periodically realize one random trade for metric stability
        if idx % 3 == 0 and book:
            pick = rng.randrange(len(book))
            pos = book.pop(pick)
            pos_sig = CandidateSignal(
                coin=pos["coin"],
                side=pos["side"],
                confidence=float(pos.get("metadata", {}).get("confidence", 0.5)),
                source_accuracy=float(pos.get("metadata", {}).get("source_accuracy", 0.5)),
                quality=float(pos.get("metadata", {}).get("quality", 0.5)),
                entry_price=float(pos.get("entry_price", 0) or 0),
                leverage=float(pos.get("leverage", 1) or 1),
                position_pct=_clamp((pos["size"] * pos["entry_price"]) / 10_000.0, 0.005, 0.20),
            )
            realized = _simulate_exit_pnl(pos_sig, stop_loss_pct, take_profit_pct, rng)
            trades_closed.append(
                {
                    "coin": pos_sig.coin,
                    "side": pos_sig.side,
                    "cluster": _cluster_for_coin(pos_sig.coin),
                    "pnl": realized,
                }
            )
            equity += realized
            equity_curve.append(equity)

        snap = _exposure_snapshot(book)
        concentration_track["coin_max"] = max(concentration_track["coin_max"], snap["coin_max"])
        concentration_track["side_max"] = max(concentration_track["side_max"], snap["side_max"])
        concentration_track["cluster_max"] = max(concentration_track["cluster_max"], snap["cluster_max"])

    # close remaining positions
    for pos in list(book):
        pos_sig = CandidateSignal(
            coin=pos["coin"],
            side=pos["side"],
            confidence=float(pos.get("metadata", {}).get("confidence", 0.5)),
            source_accuracy=float(pos.get("metadata", {}).get("source_accuracy", 0.5)),
            quality=float(pos.get("metadata", {}).get("quality", 0.5)),
            entry_price=float(pos.get("entry_price", 0) or 0),
            leverage=float(pos.get("leverage", 1) or 1),
            position_pct=_clamp((pos["size"] * pos["entry_price"]) / 10_000.0, 0.005, 0.20),
        )
        realized = _simulate_exit_pnl(pos_sig, stop_loss_pct, take_profit_pct, rng)
        trades_closed.append(
            {
                "coin": pos_sig.coin,
                "side": pos_sig.side,
                "cluster": _cluster_for_coin(pos_sig.coin),
                "pnl": realized,
            }
        )
        equity += realized
        equity_curve.append(equity)

    pnls = [t["pnl"] for t in trades_closed]
    ret_series = [p / 10_000.0 for p in pnls]
    max_dd = _max_drawdown_pct(equity_curve)
    sharpe = _sharpe_proxy(ret_series)
    total_pnl = sum(pnls)
    turnover_ratio = turnover / 10_000.0

    # worst trade clusters
    cluster_pnl: Dict[str, float] = {}
    for t in trades_closed:
        key = f"{t['cluster']}:{t['side']}"
        cluster_pnl[key] = cluster_pnl.get(key, 0.0) + t["pnl"]
    worst_clusters = sorted(
        [{"cluster": k, "pnl": round(v, 2)} for k, v in cluster_pnl.items()],
        key=lambda x: x["pnl"],
    )[:10]

    stats = mgr.get_stats()
    # prefer realized local count for actual applied replacements
    stats["replacement_count"] = replacement_count
    stats["estimated_churn_cost"] = round(churn_cost if churn_cost > 0 else stats.get("estimated_churn_cost", 0.0), 2)

    return {
        "scenario": name,
        "policy": policy,
        "config": {
            "signals": signals,
            "seed": seed,
            "leverage_scale": leverage_scale,
            "size_scale": size_scale,
            "entry_quality_threshold": entry_quality_threshold,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        },
        "metrics": {
            "pnl": round(total_pnl, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "turnover": round(turnover_ratio, 3),
            "replacement_count": int(stats["replacement_count"]),
            "churn_cost": round(float(stats["estimated_churn_cost"]), 2),
            "rejection_mix": rejection_mix,
            "concentration": {
                "coin_max_exposure_pct": round(concentration_track["coin_max"] * 100, 2),
                "side_max_exposure_pct": round(concentration_track["side_max"] * 100, 2),
                "cluster_max_exposure_pct": round(concentration_track["cluster_max"] * 100, 2),
            },
            "worst_10_trade_clusters": worst_clusters,
            "trades_closed": len(trades_closed),
        },
    }


def run_benchmark_pack(signals: int, seed: int) -> Dict:
    scenarios = [
        {
            "name": "current_main_baseline",
            "policy": "hard_cap",
            "leverage_scale": 1.0,
            "size_scale": 1.0,
            "entry_quality_threshold": 0.0,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
        },
        {
            "name": "branch_rotation_enabled",
            "policy": "rotation_enabled",
            "leverage_scale": 1.0,
            "size_scale": 1.0,
            "entry_quality_threshold": 0.0,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
        },
        {
            "name": "rotation_disabled_same_codebase",
            "policy": "rotation_disabled_same_code",
            "leverage_scale": 1.0,
            "size_scale": 1.0,
            "entry_quality_threshold": 0.0,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
        },
        {
            "name": "lower_leverage_variant",
            "policy": "rotation_enabled",
            "leverage_scale": 0.65,
            "size_scale": 1.0,
            "entry_quality_threshold": 0.40,
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.08,
        },
        {
            "name": "smaller_size_variant",
            "policy": "rotation_enabled",
            "leverage_scale": 1.0,
            "size_scale": 0.60,
            "entry_quality_threshold": 0.45,
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.08,
        },
    ]
    runs = [
        run_scenario(
            name=s["name"],
            signals=signals,
            seed=seed,
            policy=s["policy"],
            leverage_scale=s["leverage_scale"],
            size_scale=s["size_scale"],
            entry_quality_threshold=s["entry_quality_threshold"],
            stop_loss_pct=s["stop_loss_pct"],
            take_profit_pct=s["take_profit_pct"],
        )
        for s in scenarios
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
        "seed": seed,
        "runs": runs,
    }


def _print_summary(report: Dict) -> None:
    print("\n=== Executive Benchmark Pack ===")
    print(f"generated_at={report['generated_at']} signals={report['signals']} seed={report['seed']}")
    print(
        f"\n{'Scenario':<34} {'PnL':>10} {'Sharpe':>8} {'MaxDD%':>8} "
        f"{'Turnover':>9} {'Repl':>6} {'Churn$':>9}"
    )
    print("-" * 96)
    for run in report["runs"]:
        m = run["metrics"]
        print(
            f"{run['scenario']:<34} {m['pnl']:>10.2f} {m['sharpe']:>8.3f} "
            f"{m['max_drawdown_pct']:>8.2f} {m['turnover']:>9.3f} "
            f"{m['replacement_count']:>6} {m['churn_cost']:>9.2f}"
        )

    print("\nWorst 10 trade clusters (branch_rotation_enabled):")
    target = next((r for r in report["runs"] if r["scenario"] == "branch_rotation_enabled"), None)
    if not target:
        print("  n/a")
        return
    for c in target["metrics"]["worst_10_trade_clusters"]:
        print(f"  {c['cluster']:<20} pnl={c['pnl']:>10.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run executive benchmark pack for strategy/rotation quality.")
    parser.add_argument("--signals", type=int, default=500, help="Number of candidate signals to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/executive_benchmark_pack.json",
        help="Path to write JSON report.",
    )
    args = parser.parse_args()

    report = run_benchmark_pack(signals=args.signals, seed=args.seed)
    _print_summary(report)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
