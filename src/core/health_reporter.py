"""
Health Reporter — Structured JSON status for external monitoring
================================================================
Writes a machine-readable health report to /data/health_report.json
every trading cycle. Designed to be consumed by a Claude scheduled
task that reads and analyzes the bot's state, catching issues that
would otherwise require manual log review.

Report includes:
  - Timestamp and cycle number
  - Paper account balance/PnL/ROI
  - Open positions (coins, sides, unrealized PnL)
  - Signal pipeline stats (in/out/dropped per source)
  - Subsystem health (stale detection)
  - Error log (last N errors)
  - Performance metrics (cache hit rate, API 429s, etc.)
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Persistent across cycles — accumulates errors and warnings
_error_buffer: List[Dict] = []
_MAX_ERRORS = 50


def record_error(message: str, source: str = "unknown"):
    """Call from anywhere to record an error for the health report."""
    _error_buffer.append({
        "ts": datetime.utcnow().isoformat(),
        "source": source,
        "message": str(message)[:500],
    })
    if len(_error_buffer) > _MAX_ERRORS:
        _error_buffer.pop(0)


def write_health_report(
    container,
    cycle_count: int,
    health_registry=None,
    regime_data: Optional[Dict] = None,
    output_path: str = "/data/health_report.json",
) -> str:
    """
    Collect all health metrics and write to JSON.

    Returns:
        Path to the written file.
    """
    report: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cycle": cycle_count,
        "version": "1.0",
    }

    # ── Paper Account ──
    try:
        from src.data import database as db
        account = db.get_paper_account()
        if account:
            balance = account.get("balance", 10000)
            initial = 10000
            pnl = account.get("total_pnl", 0)
            report["account"] = {
                "balance": round(balance, 2),
                "pnl": round(pnl, 2),
                "roi_pct": round((balance - initial) / initial * 100, 2),
                "total_trades": account.get("total_trades", 0),
                "winning_trades": account.get("winning_trades", 0),
                "win_rate": round(
                    account.get("winning_trades", 0) / max(account.get("total_trades", 1), 1) * 100, 1
                ),
            }
    except Exception as e:
        report["account"] = {"error": str(e)}

    # ── Open Positions ──
    try:
        from src.data import database as db
        from src.data.hyperliquid_client import get_all_mids
        open_trades = db.get_open_paper_trades()
        mids = get_all_mids() or {}
        positions = []
        total_unrealized = 0.0
        for t in open_trades:
            current_price = float(mids.get(t["coin"], 0))
            entry = t.get("entry_price", 0)
            size = t.get("size", 0)
            leverage = t.get("leverage", 1)
            if entry and current_price:
                if t.get("side") == "long":
                    unrealized = (current_price - entry) / entry * size * entry * leverage
                else:
                    unrealized = (entry - current_price) / entry * size * entry * leverage
            else:
                unrealized = 0
            total_unrealized += unrealized

            opened_at = t.get("opened_at", "")
            age_hours = 0
            try:
                if opened_at:
                    age_hours = round(
                        (datetime.utcnow() - datetime.fromisoformat(opened_at)).total_seconds() / 3600, 1
                    )
            except Exception:
                pass

            positions.append({
                "coin": t["coin"],
                "side": t.get("side", "?"),
                "entry_price": round(entry, 2),
                "current_price": round(current_price, 2),
                "size": round(size, 6),
                "leverage": leverage,
                "unrealized_pnl": round(unrealized, 2),
                "stop_loss": t.get("stop_loss"),
                "take_profit": t.get("take_profit"),
                "age_hours": age_hours,
                "strategy_type": t.get("strategy_type", "unknown"),
            })
        report["positions"] = {
            "count": len(positions),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "details": positions,
        }
    except Exception as e:
        report["positions"] = {"error": str(e)}

    # ── Regime ──
    if regime_data:
        report["regime"] = {
            "overall": regime_data.get("overall_regime", "unknown"),
            "confidence": regime_data.get("confidence", 0),
            "active_strategies": regime_data.get("strategy_guidance", {}).get("activate", []),
            "paused_strategies": regime_data.get("strategy_guidance", {}).get("pause", []),
        }

    # ── Signal Pipeline Stats ──
    pipeline = {}
    try:
        if container.signal_processor:
            pipeline["signal_processor"] = dict(container.signal_processor.stats)
    except Exception:
        pass
    try:
        if container.decision_engine:
            pipeline["decision_engine"] = dict(container.decision_engine.stats)
    except Exception:
        pass
    try:
        if container.options_scanner:
            pipeline["options_flow"] = {
                "top_convictions": len(getattr(container.options_scanner, "top_convictions", []) or []),
            }
    except Exception:
        pass
    try:
        if container.kelly_sizer:
            all_stats = container.kelly_sizer.get_all_sizing_stats()
            edge_count = sum(1 for v in all_stats.values() if v.get("has_edge"))
            pipeline["kelly"] = {
                "strategies_tracked": len(all_stats),
                "with_proven_edge": edge_count,
            }
    except Exception:
        pass
    report["pipeline"] = pipeline

    # ── Arena ──
    try:
        if container.arena:
            arena_stats = container.arena.get_stats()
            report["arena"] = {
                "total_agents": arena_stats.get("total_agents", 0),
                "champions": arena_stats.get("champions", 0),
                "active": arena_stats.get("active", 0),
                "incubating": arena_stats.get("incubating", 0),
                "total_pnl": round(arena_stats.get("total_pnl", 0), 2),
            }
    except Exception:
        pass

    # ── API Health ──
    try:
        from src.data.hyperliquid_client import get_api_stats
        api_s = get_api_stats()
        if api_s:
            report["api"] = {
                "rest_requests": api_s["rest_requests"],
                "cache_served": api_s["cache_served"],
                "cache_hit_pct": api_s["cache_hit_pct"],
                "ws_served": api_s["ws_served"],
                "consecutive_429s": api_s["bucket"]["consecutive_429s"],
                "tokens_available": api_s["bucket"]["tokens_available"],
            }
    except Exception:
        pass

    # ── Subsystem Health ──
    if health_registry:
        try:
            stale = health_registry.check_stale(timeout_seconds=600)
            stale_list = [k for k, v in stale.items() if v]
            report["subsystems"] = {
                "total": len(stale),
                "stale": stale_list,
                "all_healthy": len(stale_list) == 0,
            }
        except Exception:
            pass

    # ── Copy Trading ──
    try:
        if container.copy_trader:
            report["copy_trading"] = {
                "total_executed": getattr(container.copy_trader, "_copy_count", 0),
            }
    except Exception:
        pass

    # ── Recent Errors ──
    report["errors"] = list(_error_buffer[-20:])  # Last 20

    # ── Write ──
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Health report written to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to write health report: {e}")
        # Fallback to /tmp
        fallback = "/tmp/health_report.json"
        try:
            with open(fallback, "w") as f:
                json.dump(report, f, indent=2, default=str)
            output_path = fallback
        except Exception:
            pass

    return output_path
