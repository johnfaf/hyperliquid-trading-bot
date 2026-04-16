#!/usr/bin/env python3
"""
Force refresh of golden wallets (arena agents).
This manually triggers the discovery that was stalled for 27 days due to missing dependencies.

Usage:
    python refresh_golden_wallets.py
"""
import os
import sys

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import config
from src.core.boot import setup_logging, validate_dependencies, init_database
from src.core.subsystem_registry import build_subsystems, FULL_PROFILE
from src.core.health_registry import registry as health_registry

logger = setup_logging()

print("\n" + "="*70)
print("GOLDEN WALLET REFRESH UTILITY")
print("="*70)
print(f"DB: {config.DB_PATH}")
print("Profile: FULL_PROFILE")
print("="*70 + "\n")

# Boot sequence
try:
    validate_dependencies(logger)
    init_database(logger)

    # Build subsystems
    container = build_subsystems(health_registry, FULL_PROFILE)

    # Run discovery (Phase 1 + Phase 2)
    print("\n[1/2] Running trader discovery...")
    from src.core.cycles.research_cycle import run_discovery
    run_discovery(container)

    # Get the results
    from src.discovery.golden_wallet import get_all_wallet_reports, get_golden_wallets
    all_wallets = get_all_wallet_reports()
    golden = get_golden_wallets()

    print("\n" + "="*70)
    print("DISCOVERY COMPLETE")
    print("="*70)
    print(f"Total wallets evaluated: {len(all_wallets)}")
    print(f"Golden wallets found: {len(golden)}")

    if golden:
        print("\n--- GOLDEN WALLETS ---")
        for g in golden:
            print(f"  {g['address'][:10]}...: "
                  f"PnL=${g['penalised_pnl']:+,.0f}, "
                  f"Sharpe={g['sharpe_ratio']:.2f}, "
                  f"DD={g['penalised_max_drawdown_pct']:.1f}%")

    # Auto-connect qualified golden wallets to live
    from src.discovery.golden_bridge import auto_connect_golden_wallets
    connected = auto_connect_golden_wallets()

    print(f"\nAuto-connected to live: {connected} wallets")
    print("="*70 + "\n")

    print("[OK] Golden wallet refresh complete!")
    sys.exit(0)

except Exception as e:
    logger.error("Golden wallet refresh failed: %s", e, exc_info=True)
    print(f"\n[ERROR] {e}")
    sys.exit(1)
