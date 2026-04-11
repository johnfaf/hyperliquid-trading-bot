#!/usr/bin/env python3
# ruff: noqa: E402
"""
Firewall Rejection Diagnostics
===============================
Run this to see EXACTLY why signals are being rejected.

Usage:
    python scripts/diagnose_rejections.py

Simulates 20 synthetic signals through the full pipeline and prints
which firewall check blocks each one. Also tests with your actual
database state (open positions, account balance, regime data).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── Import pipeline components ────────────────────────────────
from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource, RiskParams
from src.signals.decision_firewall import DecisionFirewall
from src.data import database as db


def make_test_signals(n=20):
    """Generate synthetic signals spanning different confidence levels and coins."""
    coins = ["BTC", "ETH", "SOL", "DOGE", "ARB", "OP", "AVAX", "LINK",
             "WLD", "SUI", "TIA", "SEI", "INJ", "NEAR", "APT", "MATIC",
             "PEPE", "WIF", "JUP", "AAVE"]
    sides = [SignalSide.LONG, SignalSide.SHORT]
    stypes = ["momentum_long", "momentum_short", "trend_following", "mean_reversion",
              "breakout", "swing_trading", "scalping", "delta_neutral"]

    signals = []
    for i in range(min(n, len(coins))):
        conf = 0.15 + (i * 0.04)  # Range: 0.15 to 0.91
        side = sides[i % 2]
        stype = stypes[i % len(stypes)]

        sig = TradeSignal(
            coin=coins[i],
            side=side,
            confidence=round(min(conf, 0.95), 2),
            source=SignalSource.STRATEGY,
            reason=f"test_signal_{i}",
            strategy_type=stype,
            entry_price=100.0,
            leverage=3,
            position_pct=0.06,
            risk=RiskParams(stop_loss_pct=0.03, take_profit_pct=0.06),
            regime="",
        )
        signals.append(sig)
    return signals


def diagnose():
    print("=" * 70)
    print("  FIREWALL REJECTION DIAGNOSTIC")
    print("=" * 70)

    # ─── 1. Check database state ───────────────────────────────
    print("\n--- Database State ---")
    account = db.get_paper_account()
    if account:
        print(f"  Balance: ${account.get('balance', 0):,.2f}")
        print(f"  Unrealized PnL: ${account.get('unrealized_pnl', 0):,.2f}")
    else:
        print("  ⚠ No paper account found!")

    positions = db.get_open_paper_trades()
    print(f"  Open positions: {len(positions)}")
    if positions:
        total_exposure = 0
        for p in positions:
            notional = abs(p.get("size", 0) * p.get("entry_price", 0) * p.get("leverage", 1))
            total_exposure += notional
            print(f"    {p.get('side', '?')} {p.get('coin', '?')} "
                  f"size={p.get('size', 0):.4f} @ {p.get('entry_price', 0):.2f} "
                  f"lev={p.get('leverage', 1)}x notional=${notional:,.2f}")
        balance = account.get("balance", 10000) if account else 10000
        print(f"  Total exposure: ${total_exposure:,.2f} ({total_exposure/balance:.0%} of balance)")

    # ─── 2. Test signals through firewall ──────────────────────
    print("\n--- Testing 20 Synthetic Signals Through Firewall ---")
    fw = DecisionFirewall()
    signals = make_test_signals(20)

    results = []
    for sig in signals:
        passed, reason = fw.validate(sig, regime_data=None, open_positions=positions)
        status = "✓ PASS" if passed else "✗ FAIL"
        results.append((sig, passed, reason))
        print(f"  {status} | {sig.side.value:5s} {sig.coin:6s} conf={sig.confidence:.0%} "
              f"type={sig.strategy_type:20s} | {reason}")

    # ─── 3. Show stats breakdown ───────────────────────────────
    print("\n--- Firewall Stats Breakdown ---")
    stats = fw.get_stats()
    print(f"  Total signals tested: {stats['total_signals']}")
    print(f"  Passed:              {stats['passed']}")
    print(f"  Pass rate:           {stats['pass_rate']:.0%}")
    print()
    for key, val in sorted(stats.items()):
        if key.startswith("rejected_") and val > 0:
            print(f"  {key:25s}: {val:3d} ({val/stats['total_signals']:.0%})")

    top = stats.get("top_rejection_reason", "none")
    print(f"\n  >>> Top rejection reason: {top}")

    # ─── 4. Identify the bottleneck ────────────────────────────
    print("\n--- Diagnosis ---")
    passed_count = stats["passed"]
    if passed_count == 0:
        print("  ALL signals rejected! Checking which check is the primary blocker:")
        rejection_counts = {k: v for k, v in stats.items() if k.startswith("rejected_") and v > 0}
        if rejection_counts:
            sorted_reasons = sorted(rejection_counts.items(), key=lambda x: -x[1])
            for reason, count in sorted_reasons:
                pct = count / stats["total_signals"] * 100
                print(f"    {reason}: {count} ({pct:.0f}%)")

            primary = sorted_reasons[0]
            print(f"\n  PRIMARY BLOCKER: {primary[0]} ({primary[1]} rejections)")

            # Specific remediation advice
            if primary[0] == "rejected_confidence":
                print("  FIX: Your signals have confidence < 30%. Either:")
                print("    a) Lower min_confidence in firewall config (try 0.15)")
                print("    b) Improve scoring to produce higher-confidence signals")
                print("    c) Ensure strategies have enough trade history (>10 trades)")
            elif primary[0] == "rejected_regime":
                print("  FIX: Regime is blocking your strategy types. Either:")
                print("    a) Wait for regime to change")
                print("    b) Add strategies that match current regime")
                print("    c) Relax regime filtering in the firewall")
            elif primary[0] == "rejected_exposure":
                print("  FIX: Aggregate exposure cap is too low. Either:")
                print("    a) Raise max_aggregate_exposure (try 1.5 for paper)")
                print("    b) Reduce leverage on signals")
                print("    c) Close some existing positions")
            elif primary[0] == "rejected_risk":
                print("  FIX: Position limits reached. Either:")
                print("    a) Raise max_positions (try 12)")
                print("    b) Close some positions")
            elif primary[0] == "rejected_cooldown":
                print("  FIX: Cooldown too aggressive. Lower cooldown_seconds (try 60)")
            elif primary[0] == "rejected_funding":
                print("  FIX: Funding rate check blocking trades. Either:")
                print("    a) Disable funding_risk_enabled for paper trading")
                print("    b) Adjust thresholds (funding_negative_threshold, funding_positive_threshold)")
        else:
            print("  No specific rejection reasons logged — signals may be filtered")
            print("  BEFORE reaching the firewall (in SignalProcessor or DecisionEngine)")
    else:
        print(f"  {passed_count} signals passed — firewall is working.")

    # ─── 5. Test with relaxed settings ─────────────────────────
    print("\n--- Test With RELAXED Settings (paper-trading friendly) ---")
    relaxed_fw = DecisionFirewall({
        "min_confidence": 0.10,        # was 0.30
        "max_positions": 12,           # was 8
        "max_per_coin": 4,             # was 3
        "max_aggregate_exposure": 2.0, # was 0.80
        "cooldown_seconds": 30,        # was 300
        "funding_risk_enabled": False,  # disable for paper
        "enable_predictive_derisk": False,
    })

    signals2 = make_test_signals(20)
    for sig in signals2:
        passed, reason = relaxed_fw.validate(sig, regime_data=None, open_positions=[])
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            print(f"  {status} | {sig.side.value:5s} {sig.coin:6s} conf={sig.confidence:.0%} | {reason}")

    relaxed_stats = relaxed_fw.get_stats()
    print(f"\n  Relaxed settings: {relaxed_stats['passed']}/{relaxed_stats['total_signals']} passed "
          f"({relaxed_stats['pass_rate']:.0%})")

    print("\n" + "=" * 70)
    print("  Done. Review the PRIMARY BLOCKER above to fix your pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    diagnose()
