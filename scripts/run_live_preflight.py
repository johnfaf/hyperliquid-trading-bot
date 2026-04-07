"""
Run a live-readiness preflight for the Hyperliquid live trader.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.core.boot import init_database, setup_logging, validate_dependencies  # noqa: E402
from src.signals.decision_firewall import DecisionFirewall  # noqa: E402
from src.trading.live_trader import LiveTrader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live trading preflight.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the preflight report.",
    )
    args = parser.parse_args()

    logger = setup_logging()
    validate_dependencies(logger)
    init_database(logger)

    firewall = DecisionFirewall(
        {
            "funding_risk_enabled": False,
            "enable_predictive_derisk": False,
            "min_confidence": getattr(config, "FIREWALL_MIN_CONFIDENCE", 0.45),
        }
    )
    trader = LiveTrader(
        firewall=firewall,
        dry_run=not getattr(config, "LIVE_TRADING_ENABLED", False),
        max_daily_loss=float(getattr(config, "LIVE_MAX_DAILY_LOSS_USD", 500)),
        max_order_usd=float(getattr(config, "LIVE_MAX_ORDER_USD", 12.0)),
    )
    report = trader.run_preflight(force=True)

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    print("Live preflight complete")
    print(f"Status: {report.get('status', 'unknown')}")
    print(f"Deployable: {report.get('deployable', False)}")
    print(f"Blocking checks: {', '.join(report.get('blocking_checks', [])) or 'none'}")
    print(f"Warning checks: {', '.join(report.get('warning_checks', [])) or 'none'}")
    if args.output:
        print(f"Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
