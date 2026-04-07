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


def _extract_runtime_profile(argv):
    for index, arg in enumerate(argv):
        if arg.startswith("--runtime-profile="):
            return arg.split("=", 1)[1].strip()
        if arg == "--runtime-profile" and index + 1 < len(argv):
            return str(argv[index + 1]).strip()
    return ""


_runtime_profile_override = _extract_runtime_profile(sys.argv[1:])
if _runtime_profile_override:
    os.environ["BOT_RUNTIME_PROFILE"] = _runtime_profile_override

import config  # noqa: E402
from src.core.boot import init_database, setup_logging, validate_dependencies  # noqa: E402
from src.signals.decision_firewall import DecisionFirewall  # noqa: E402
from src.trading.live_trader import LiveTrader  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the live trading preflight.")
    parser.add_argument(
        "--runtime-profile",
        choices=("paper", "shadow", "live"),
        default=None,
        help="Apply the named runtime profile before config loads.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the preflight report.",
    )
    args = parser.parse_args(argv)

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
    preflight = trader.run_preflight(force=True)
    activation = trader.evaluate_activation_guard()
    readiness = trader.get_live_readiness(force_preflight=False)
    report = {
        "runtime_profile": getattr(config, "RUNTIME_PROFILE", "paper"),
        "preflight": preflight,
        "activation_guard": activation,
        "live_readiness": readiness,
        "certified_for_live_entries": bool(readiness.get("deployable", False)),
    }
    exit_code = 0 if report["certified_for_live_entries"] else 1

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    print("Live preflight complete")
    print(f"Runtime profile: {getattr(config, 'RUNTIME_PROFILE', 'paper')}")
    print(f"Preflight: {preflight.get('status', 'unknown')} (deployable={preflight.get('deployable', False)})")
    print(
        f"Activation: {activation.get('status', 'unknown')} "
        f"(deployable={activation.get('deployable', False)}, approved_by={activation.get('approved_by') or 'unset'})"
    )
    print(f"Readiness: {readiness.get('status', 'unknown')} ({readiness.get('status_reason', 'unknown')})")
    print(f"Blocking checks: {', '.join(readiness.get('blocking_checks', [])) or 'none'}")
    print(f"Preflight warnings: {', '.join(preflight.get('warning_checks', [])) or 'none'}")
    print(f"Activation warnings: {', '.join(activation.get('warning_checks', [])) or 'none'}")
    print(
        "Certification: "
        + ("PASS" if report["certified_for_live_entries"] else "FAIL")
    )
    if args.output:
        print(f"Output: {os.path.abspath(args.output)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
