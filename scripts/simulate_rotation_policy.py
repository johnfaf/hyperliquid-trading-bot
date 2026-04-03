#!/usr/bin/env python3
"""
Compare hard-cap rejection vs soft-cap rotation on identical signal streams.

This is a deterministic simulation for policy comparison only. It does not
touch live/paper databases.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.trading.portfolio_rotation import PortfolioRotationManager  # noqa: E402


COINS = ["BTC", "ETH", "SOL", "DOGE", "LINK", "ARB", "AVAX", "APT", "INJ", "SUI"]
SIDES = ["long", "short"]


@dataclass
class CandidateSignal:
    coin: str
    side: str
    confidence: float
    source_accuracy: float


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_of_trade(trade: Dict, manager: PortfolioRotationManager) -> float:
    meta = trade.get("metadata", {}) or {}
    base = float(meta.get("confidence", 0.35))
    src = float(meta.get("source_accuracy", 0.0))
    return base + (src * manager._source_accuracy_weight)


def _make_trade(trade_id: int, signal: CandidateSignal, now: datetime, rng: random.Random) -> Dict:
    entry_price = rng.uniform(10.0, 100_000.0)
    size = rng.uniform(0.01, 2.5)
    leverage = rng.choice([1, 2, 3, 4, 5])
    pnl_move = rng.uniform(-0.08, 0.08)
    if signal.side == "long":
        current_price = entry_price * (1 + pnl_move)
    else:
        current_price = entry_price * (1 - pnl_move)
    return {
        "id": trade_id,
        "coin": signal.coin,
        "side": signal.side,
        "entry_price": entry_price,
        "size": size,
        "leverage": leverage,
        "opened_at": (now - timedelta(minutes=rng.randint(5, 720))).isoformat(),
        "current_price": current_price,
        "metadata": {
            "confidence": signal.confidence,
            "source_accuracy": signal.source_accuracy,
        },
    }


def _initial_book(manager: PortfolioRotationManager, rng: random.Random, now: datetime) -> List[Dict]:
    book = []
    for idx in range(manager.target_positions):
        sig = CandidateSignal(
            coin=COINS[idx % len(COINS)],
            side=SIDES[idx % len(SIDES)],
            confidence=rng.uniform(0.35, 0.72),
            source_accuracy=rng.uniform(0.2, 0.7),
        )
        book.append(_make_trade(idx + 1, sig, now, rng))
    return book


def _signal_stream(n: int, seed: int) -> List[CandidateSignal]:
    rng = random.Random(seed)
    stream: List[CandidateSignal] = []
    for _ in range(n):
        stream.append(
            CandidateSignal(
                coin=rng.choice(COINS),
                side=rng.choice(SIDES),
                confidence=rng.uniform(0.25, 0.97),
                source_accuracy=rng.uniform(0.1, 0.95),
            )
        )
    return stream


def run_simulation(signals: int, seed: int, cycle_size: int) -> Dict[str, Dict]:
    base_cfg = {
        "target_positions": 5,
        "hard_max_positions": 5,
        "reserved_high_conviction_slots": 1,
        "high_conviction_threshold": 0.72,
        "min_hold_minutes": 15,
        "replacement_threshold": 0.08,
        "max_replacements_per_cycle": 4,
    }
    soft_mgr = PortfolioRotationManager(base_cfg)

    now = datetime.now(timezone.utc)
    rng = random.Random(seed + 99)
    signal_stream = _signal_stream(signals, seed)

    hard_book = _initial_book(soft_mgr, rng, now)
    soft_book = [dict(t) for t in hard_book]

    hard = {"accepted": 0, "rejected": 0, "replaced": 0, "scores": []}
    soft = {"accepted": 0, "rejected": 0, "replaced": 0, "scores": []}

    next_id_hard = max(t["id"] for t in hard_book) + 1
    next_id_soft = max(t["id"] for t in soft_book) + 1
    soft_replacements_used = 0

    for idx, sig in enumerate(signal_stream):
        if idx > 0 and cycle_size > 0 and idx % cycle_size == 0:
            soft_replacements_used = 0

        # Hard-cap policy: reject when full.
        if len(hard_book) < soft_mgr.target_positions:
            trade = _make_trade(next_id_hard, sig, now, rng)
            next_id_hard += 1
            hard_book.append(trade)
            hard["accepted"] += 1
            hard["scores"].append(_score_of_trade(trade, soft_mgr))
        else:
            hard["rejected"] += 1

        # Soft policy: open/replace/reject based on rotation manager.
        decision = soft_mgr.decide(
            signal=sig,
            open_positions=soft_book,
            regime_data={},
            replacements_used=soft_replacements_used,
        )
        if decision.action == "reject":
            soft["rejected"] += 1
            continue

        if decision.action == "replace":
            victim_idx = next(
                (i for i, t in enumerate(soft_book) if t.get("id") == decision.replacement_trade_id),
                None,
            )
            if victim_idx is None:
                soft["rejected"] += 1
                continue
            soft_book.pop(victim_idx)
            soft["replaced"] += 1
            soft_replacements_used += 1

        trade = _make_trade(next_id_soft, sig, now, rng)
        next_id_soft += 1
        soft_book.append(trade)
        soft["accepted"] += 1
        soft["scores"].append(_score_of_trade(trade, soft_mgr))

    hard_final_scores = [_score_of_trade(t, soft_mgr) for t in hard_book]
    soft_final_scores = [_score_of_trade(t, soft_mgr) for t in soft_book]

    return {
        "hard_cap": {
            **hard,
            "avg_accepted_score": round(_avg(hard["scores"]), 4),
            "final_book_avg_score": round(_avg(hard_final_scores), 4),
            "final_book_size": len(hard_book),
        },
        "soft_rotation": {
            **soft,
            "avg_accepted_score": round(_avg(soft["scores"]), 4),
            "final_book_avg_score": round(_avg(soft_final_scores), 4),
            "final_book_size": len(soft_book),
        },
        "config": {
            "signals": signals,
            "seed": seed,
            "cycle_size": cycle_size,
            **base_cfg,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hard-cap vs soft-rotation policies.")
    parser.add_argument("--signals", type=int, default=300, help="Number of candidate signals to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument(
        "--cycle-size",
        type=int,
        default=50,
        help="How many signals per logical cycle (resets replacement budget).",
    )
    args = parser.parse_args()

    results = run_simulation(signals=args.signals, seed=args.seed, cycle_size=args.cycle_size)
    hard = results["hard_cap"]
    soft = results["soft_rotation"]

    print("\n=== Rotation Policy Simulation ===")
    print(
        f"signals={results['config']['signals']} seed={results['config']['seed']} "
        f"cycle_size={results['config']['cycle_size']}"
    )
    print("\nPolicy: Hard Cap Reject")
    print(
        f"accepted={hard['accepted']} replaced={hard['replaced']} rejected={hard['rejected']} "
        f"avg_accepted_score={hard['avg_accepted_score']:.4f} "
        f"final_book_avg_score={hard['final_book_avg_score']:.4f} "
        f"final_book_size={hard['final_book_size']}"
    )
    print("\nPolicy: Soft Cap + Rotation")
    print(
        f"accepted={soft['accepted']} replaced={soft['replaced']} rejected={soft['rejected']} "
        f"avg_accepted_score={soft['avg_accepted_score']:.4f} "
        f"final_book_avg_score={soft['final_book_avg_score']:.4f} "
        f"final_book_size={soft['final_book_size']}"
    )

    delta_final = soft["final_book_avg_score"] - hard["final_book_avg_score"]
    print(f"\nDelta final_book_avg_score (soft - hard): {delta_final:+.4f}")


if __name__ == "__main__":
    main()
