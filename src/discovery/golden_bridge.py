"""
Golden Bridge
=============
Connects golden wallets (verified profitable after penalties) to the live
copy-trading pipeline.  Golden wallets get:
  - Higher priority in the CopyTrader signal queue
  - Boosted confidence scores (they've proven they survive fees)
  - Automatic inclusion in the scan_top_traders pool
  - Larger position sizing (kelly sizer trusts them more)

This module is imported by main.py and injects golden addresses into the
copy trader's scan cycle.
"""
import logging
import json
import sqlite3
import os
import sys
from typing import List, Dict, Optional, Set
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import database as db
from src.data import hyperliquid_client as hl

logger = logging.getLogger("golden_bridge")

# How much to boost golden wallet confidence scores (additive)
GOLDEN_CONFIDENCE_BOOST = 0.15
# Minimum Sharpe from backtest to qualify for live
MIN_SHARPE_FOR_LIVE = 0.3
# Maximum drawdown to qualify for live
MAX_DD_FOR_LIVE = 40.0


def _get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_live_golden_wallets() -> List[Dict]:
    """
    Get golden wallets that are connected to live execution.
    Returns list of dicts with address, sharpe, penalised_pnl, etc.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT address, sharpe_ratio, penalised_pnl, win_rate, "
            "penalised_max_drawdown_pct, trades_per_day, best_coin, coins_traded "
            "FROM golden_wallets "
            "WHERE is_golden = 1 AND connected_to_live = 1 "
            "ORDER BY penalised_pnl DESC"
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["coins_traded"] = json.loads(d.get("coins_traded", "[]"))
            except:
                d["coins_traded"] = []
            results.append(d)
        return results
    finally:
        conn.close()


def auto_connect_golden_wallets() -> int:
    """
    Automatically connect golden wallets that meet quality thresholds.
    Returns number of newly connected wallets.
    """
    conn = _get_db()
    try:
        # Find golden wallets not yet connected that meet quality bars
        rows = conn.execute(
            "SELECT address, sharpe_ratio, penalised_max_drawdown_pct, penalised_pnl "
            "FROM golden_wallets "
            "WHERE is_golden = 1 AND connected_to_live = 0 "
            "AND sharpe_ratio >= ? AND penalised_max_drawdown_pct <= ?",
            (MIN_SHARPE_FOR_LIVE, MAX_DD_FOR_LIVE)
        ).fetchall()

        connected = 0
        for r in rows:
            conn.execute(
                "UPDATE golden_wallets SET connected_to_live = 1 WHERE address = ?",
                (r["address"],)
            )
            connected += 1
            logger.info(
                f"Auto-connected golden wallet {r['address'][:10]}: "
                f"Sharpe={r['sharpe_ratio']:.2f}, DD={r['penalised_max_drawdown_pct']:.1f}%, "
                f"PnL=${r['penalised_pnl']:+,.0f}"
            )

        conn.commit()
        if connected:
            logger.info(f"Connected {connected} new golden wallets to live execution")
        return connected
    finally:
        conn.close()


def disconnect_wallet(address: str):
    """Manually disconnect a golden wallet from live execution."""
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE golden_wallets SET connected_to_live = 0 WHERE address = ?",
            (address,)
        )
        conn.commit()
        logger.info(f"Disconnected golden wallet {address[:10]} from live execution")
    finally:
        conn.close()


def get_golden_copy_signals(mids: Optional[Dict] = None) -> List[Dict]:
    """
    Generate copy-trade signals from golden wallets' current positions.
    These signals get a confidence boost because the wallet has proven
    profitability after fee/slippage penalties.
    """
    golden = get_live_golden_wallets()
    if not golden:
        return []

    if mids is None:
        mids = hl.get_all_mids() or {}

    signals = []
    for wallet in golden:
        try:
            state = hl.get_user_state(wallet["address"])
            if not state:
                continue

            for pos in state.get("positions", []):
                if pos["size"] <= 0:
                    continue

                coin = pos["coin"]
                price = float(mids.get(coin, pos["entry_price"]))
                if price <= 0:
                    continue

                # Base confidence from backtest win rate + golden boost
                base_conf = min(0.9, 0.5 + float(wallet.get("win_rate", 50) or 50) / 100 * 0.4)
                boosted_conf = min(0.95, base_conf + GOLDEN_CONFIDENCE_BOOST)

                signals.append({
                    "type": "golden_copy",
                    "coin": coin,
                    "side": pos["side"],
                    "price": price,
                    "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                    "source_trader": wallet["address"][:10],
                    "source_pnl": wallet.get("penalised_pnl", 0),
                    "confidence": boosted_conf,
                    "sharpe": wallet.get("sharpe_ratio", 0),
                    "is_golden": True,
                    "metadata": {
                        "golden_wallet": True,
                        "backtest_sharpe": wallet.get("sharpe_ratio", 0),
                        "backtest_pnl": wallet.get("penalised_pnl", 0),
                        "backtest_wr": wallet.get("win_rate", 0),
                    }
                })

        except Exception as e:
            logger.debug(f"Error scanning golden wallet {wallet['address'][:10]}: {e}")

    if signals:
        logger.info(f"Golden bridge: {len(signals)} signals from {len(golden)} golden wallets")

    return signals


def get_stats() -> Dict:
    """Get golden bridge stats for dashboard/logging."""
    conn = _get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM golden_wallets").fetchone()[0]
        golden = conn.execute("SELECT COUNT(*) FROM golden_wallets WHERE is_golden = 1").fetchone()[0]
        live = conn.execute("SELECT COUNT(*) FROM golden_wallets WHERE connected_to_live = 1").fetchone()[0]
        return {
            "total_evaluated": total,
            "golden_wallets": golden,
            "live_connected": live,
        }
    except:
        return {"total_evaluated": 0, "golden_wallets": 0, "live_connected": 0}
    finally:
        conn.close()
