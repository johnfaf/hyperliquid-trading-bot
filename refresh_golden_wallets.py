#!/usr/bin/env python3
"""
Refresh golden-wallet discovery.

Default mode is discover-only. Use --connect-live to explicitly bridge newly
qualified wallets into live copy-trading.

Usage:
    python refresh_golden_wallets.py
    python refresh_golden_wallets.py --connect-live
"""
from __future__ import annotations

import argparse
import os
import sys


ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh golden-wallet discovery.")
    parser.add_argument(
        "--connect-live",
        action="store_true",
        help="Explicitly auto-connect qualified golden wallets to live copy trading",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_dotenv()

    import config
    from src.core.boot import init_database, setup_logging, validate_dependencies
    from src.core.health_registry import registry as health_registry
    from src.core.subsystem_registry import FULL_PROFILE, build_subsystems

    logger = setup_logging()

    print("\n" + "=" * 70)
    print("GOLDEN WALLET REFRESH UTILITY")
    print("=" * 70)
    print(f"DB: {config.DB_PATH}")
    print("Profile: FULL_PROFILE")
    print("Mode: discover + live connect" if args.connect_live else "Mode: discover only")
    print("=" * 70 + "\n")

    try:
        validate_dependencies(logger)
        init_database(logger)

        container = build_subsystems(health_registry, FULL_PROFILE)

        print("\n[1/2] Running trader discovery...")
        from src.core.cycles.research_cycle import run_discovery

        run_discovery(container)

        from src.discovery.golden_wallet import get_all_wallet_reports, get_golden_wallets

        all_wallets = get_all_wallet_reports()
        golden = get_golden_wallets()

        print("\n" + "=" * 70)
        print("DISCOVERY COMPLETE")
        print("=" * 70)
        print(f"Total wallets evaluated: {len(all_wallets)}")
        print(f"Golden wallets found: {len(golden)}")

        if golden:
            print("\n--- GOLDEN WALLETS ---")
            for g in golden:
                print(
                    f"  {g['address'][:10]}...: "
                    f"PnL=${g['penalised_pnl']:+,.0f}, "
                    f"Sharpe={g['sharpe_ratio']:.2f}, "
                    f"DD={g['penalised_max_drawdown_pct']:.1f}%"
                )

        connected = 0
        if args.connect_live:
            from src.discovery.golden_bridge import auto_connect_golden_wallets

            connected = auto_connect_golden_wallets()
            print(f"\nConnected to live: {connected} wallets")
        else:
            print("\nLive auto-connect skipped. Rerun with --connect-live to opt in.")
        print("=" * 70 + "\n")

        print("[OK] Golden wallet refresh complete.")
        return 0

    except Exception as exc:
        logger.error("Golden wallet refresh failed: %s", exc, exc_info=True)
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
