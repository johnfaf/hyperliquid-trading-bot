"""
Health Reporter - Structured JSON status for external monitoring.

Writes a machine-readable health report each trading cycle so operators can
inspect both shadow and live state from a single artifact.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.live_execution import is_live_trading_active

logger = logging.getLogger(__name__)

# Persistent across cycles - accumulates errors and warnings
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


def _account_report(container, live_active: bool) -> Dict[str, Any]:
    from src.data import database as db

    source = "paper"
    account = None
    if live_active and getattr(container, "live_trader", None):
        balance = container.live_trader.get_account_value()
        if balance is not None:
            source = "live"
            live_stats = container.live_trader.get_stats()
            account = {
                "balance": balance,
                "total_pnl": live_stats.get("daily_pnl", 0),
                "total_trades": None,
                "winning_trades": None,
            }

    if not account:
        account = db.get_paper_account()

    if not account:
        return {}

    balance = float(account.get("balance", 10000) or 10000)
    pnl = float(account.get("total_pnl", 0) or 0)
    total_trades = account.get("total_trades", 0)
    winning_trades = account.get("winning_trades", 0)
    win_rate = None
    if total_trades is not None and winning_trades is not None:
        win_rate = round(winning_trades / max(total_trades or 1, 1) * 100, 1)

    return {
        "source": source,
        "balance": round(balance, 2),
        "pnl": round(pnl, 2),
        "roi_pct": round((balance - 10000) / 10000 * 100, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
    }


def _positions_report(container, live_active: bool) -> Dict[str, Any]:
    from src.data import database as db
    from src.data.hyperliquid_client import get_all_mids

    mids = get_all_mids() or {}
    source = "paper"
    if live_active and getattr(container, "live_trader", None):
        open_trades = container.live_trader.get_positions()
        source = "live"
    else:
        open_trades = db.get_open_paper_trades()

    positions = []
    total_unrealized = 0.0
    for trade in open_trades:
        coin = trade.get("coin", "")
        side = trade.get("side", "?")
        entry = float(trade.get("entry_price", trade.get("entryPx", 0)) or 0)
        current_price = float(mids.get(coin, 0) or 0)
        size = float(trade.get("size", 0) or 0)
        leverage = float(trade.get("leverage", 1) or 1)

        if entry and current_price:
            if side == "long":
                unrealized = (current_price - entry) / entry * size * entry * leverage
            else:
                unrealized = (entry - current_price) / entry * size * entry * leverage
        else:
            unrealized = float(trade.get("unrealized_pnl", trade.get("unrealizedPnl", 0)) or 0)
        total_unrealized += unrealized

        opened_at = trade.get("opened_at", "")
        age_hours = 0.0
        try:
            if opened_at:
                age_hours = round(
                    (datetime.utcnow() - datetime.fromisoformat(opened_at)).total_seconds() / 3600,
                    1,
                )
        except Exception:
            pass

        positions.append({
            "coin": coin,
            "side": side,
            "entry_price": round(entry, 2),
            "current_price": round(current_price, 2),
            "size": round(size, 6),
            "leverage": leverage,
            "unrealized_pnl": round(unrealized, 2),
            "stop_loss": trade.get("stop_loss"),
            "take_profit": trade.get("take_profit"),
            "age_hours": age_hours,
            "strategy_type": trade.get("strategy_type", "unknown"),
        })

    return {
        "source": source,
        "count": len(positions),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "details": positions,
    }


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

    try:
        live_active = is_live_trading_active(container)
    except Exception:
        live_active = False

    try:
        report["account"] = _account_report(container, live_active)
    except Exception as exc:
        report["account"] = {"error": str(exc)}

    try:
        report["positions"] = _positions_report(container, live_active)
    except Exception as exc:
        report["positions"] = {"error": str(exc)}

    if regime_data:
        report["regime"] = {
            "overall": regime_data.get("overall_regime", "unknown"),
            "confidence": regime_data.get("confidence", 0),
            "active_strategies": regime_data.get("strategy_guidance", {}).get("activate", []),
            "paused_strategies": regime_data.get("strategy_guidance", {}).get("pause", []),
        }

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
            edge_count = sum(1 for value in all_stats.values() if value.get("has_edge"))
            pipeline["kelly"] = {
                "strategies_tracked": len(all_stats),
                "with_proven_edge": edge_count,
            }
    except Exception:
        pass
    report["pipeline"] = pipeline

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

    try:
        from src.data.hyperliquid_client import get_api_stats

        api_stats = get_api_stats()
        if api_stats:
            report["api"] = {
                "rest_requests": api_stats["rest_requests"],
                "cache_served": api_stats["cache_served"],
                "cache_hit_pct": api_stats["cache_hit_pct"],
                "ws_served": api_stats["ws_served"],
                "consecutive_429s": api_stats["bucket"]["consecutive_429s"],
                "tokens_available": api_stats["bucket"]["tokens_available"],
            }
    except Exception:
        pass

    if health_registry:
        try:
            stale = health_registry.check_stale(timeout_seconds=600)
            stale_list = [name for name, is_stale in stale.items() if is_stale]
            detailed = {}
            for name, status in health_registry.get_all().items():
                detailed[name] = {
                    "state": status.state.value,
                    "reason": status.reason,
                    "affects_trading": status.affects_trading,
                }
            report["subsystems"] = {
                "total": len(stale),
                "stale": stale_list,
                "all_healthy": len(stale_list) == 0,
                "details": detailed,
            }
        except Exception:
            pass

    try:
        if container.copy_trader:
            report["copy_trading"] = {
                "total_executed": getattr(container.copy_trader, "_copy_count", 0),
            }
    except Exception:
        pass

    try:
        if getattr(container, "live_trader", None):
            report["live_trading"] = container.live_trader.get_stats()
    except Exception:
        pass

    report["errors"] = list(_error_buffer[-20:])

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        logger.info("Health report written to %s", output_path)
    except Exception as exc:
        logger.warning("Failed to write health report: %s", exc)
        fallback = "/tmp/health_report.json"
        try:
            with open(fallback, "w") as handle:
                json.dump(report, handle, indent=2, default=str)
            output_path = fallback
        except Exception:
            pass

    return output_path
