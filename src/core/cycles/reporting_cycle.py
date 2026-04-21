"""
Reporting & Maintenance Cycle
=============================
Module status logging, Telegram alerts, HTML report export, DB backup,
and health-registry staleness checks.

Extracted from the tail end of ``_run_trading_cycle`` so the trading
module focuses purely on signal generation and execution.
"""
import logging

import config
from src.data.database import backup_to_json

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_reporting(container, cycle_count: int, health_registry=None) -> None:
    """
    Phase 6+: status update, module stats, alerts, backup.
    Called at the end of every trading cycle.
    """
    try:
        from src.notifications import telegram_bot as tg
    except ImportError:
        tg = None
    try:
        from src.notifications import telegram_alerts as tg_alerts
    except ImportError:
        tg_alerts = None
    try:
        from src.ui import report_exporter
    except ImportError:
        report_exporter = None
    try:
        from src.discovery.golden_bridge import get_stats as golden_stats
    except ImportError:
        golden_stats = None
    try:
        from src.data.hyperliquid_client import get_api_stats
    except ImportError:
        get_api_stats = None

    # ── Phase 6: Status ──
    logger.info("Phase 6: Status Update")
    if container.reporter:
        status = container.reporter.print_live_status()
        print(status)

    # Telegram cycle summary
    if tg and tg.is_configured() and container.paper_trader:
        try:
            summary = container.paper_trader.get_account_summary()
            tg.notify_cycle_summary(summary)
        except Exception:
            pass

    # ── Module stats ──
    logger.info("Module Status:")
    _log_module_stats(container)

    # ── Shadow tracker + hedger stats ──
    try:
        if container.shadow_tracker:
            shadow_summary = container.shadow_tracker.get_summary(days=30)
            logger.info(
                "  ShadowTracker: %d trades, PnL=$%.2f, win_rate=%s, best_source=%s",
                shadow_summary["total_trades"], shadow_summary["total_pnl"],
                f"{shadow_summary['avg_win_rate']:.0%}",
                shadow_summary.get("best_source", "N/A"),
            )
    except Exception:
        pass
    try:
        if container.cross_venue_hedger:
            hedge_stats = container.cross_venue_hedger.get_stats()
            logger.info(
                "  Hedger: %d placed, %d closed, %d active (%s)",
                hedge_stats["total_hedges_placed"], hedge_stats["total_hedges_closed"],
                hedge_stats["active_hedges_count"],
                "DRY" if hedge_stats["dry_run"] else "LIVE",
            )
    except Exception:
        pass

    # ── Golden wallet stats ──
    try:
        gs = golden_stats() if golden_stats else None
        if gs and gs["total_evaluated"] > 0:
            logger.info(
                "  Golden Wallets: %d golden / %d evaluated, %d connected",
                gs["golden_wallets"], gs["total_evaluated"], gs["live_connected"],
            )
    except Exception:
        pass

    # ── API manager stats ──
    try:
        api_s = get_api_stats() if get_api_stats else None
        if not api_s:
            raise ValueError("skip")
        logger.info(
            "  API Manager: %d REST, %d cached (%s%% hit), %d from WS | "
            "bucket: %.0f tokens, 429s=%d",
            api_s["rest_requests"], api_s["cache_served"], api_s["cache_hit_pct"],
            api_s["ws_served"], api_s["bucket"]["tokens_available"],
            api_s["bucket"]["consecutive_429s"],
        )
    except Exception:
        pass

    # ── Improvement report ──
    if container.scorer:
        try:
            improvement = container.scorer.generate_improvement_report()
            logger.info("  Bot health: %s", improvement.get("health", "unknown"))
        except Exception:
            pass

    # ── DB backup ──
    backup_to_json()

    # ── Telegram daily/weekly alerts ──
    cycles_per_day = max(int(86400 / config.TRADING_CYCLE_INTERVAL), 1)
    try:
        if tg and tg.is_configured() and tg_alerts and cycle_count % cycles_per_day == 0:
            tg_alerts.send_daily_pnl_summary()
            logger.info("  Sent daily P&L Telegram summary")
        if tg and tg.is_configured() and tg_alerts and cycle_count % (cycles_per_day * 7) == 0:
            tg_alerts.send_weekly_digest()
            logger.info("  Sent weekly Telegram digest")
    except Exception as exc:
        logger.debug("  Enhanced alerts error: %s", exc)

    # ── HTML report (daily) ──
    try:
        if report_exporter and cycle_count % cycles_per_day == 0:
            report_path = report_exporter.export_html_report()
            logger.info("  HTML report exported: %s", report_path)
    except Exception as exc:
        logger.debug("  Report export error: %s", exc)

    # ── Health registry staleness check ──
    if health_registry:
        try:
            stale = health_registry.check_stale(timeout_seconds=600)
            stale_names = {name: state for name, state in stale.items() if state}
            if stale_names:
                logger.warning("  Stale subsystems: %s", stale_names)
        except Exception:
            pass

    # ── Machine-readable health report for Claude monitoring ──
    try:
        from src.core.health_reporter import write_health_report
        # Grab regime_data from container if available
        regime_data = getattr(container, "_last_regime_data", None)
        write_health_report(
            container, cycle_count,
            health_registry=health_registry,
            regime_data=regime_data,
        )
    except Exception as exc:
        logger.debug("  Health report error: %s", exc)


def _log_module_stats(container):
    """Log V2.5+ module statistics."""
    _safe_stat("LCRS", lambda: (
        container.liquidation_strategy and
        _fmt("setups_detected={setups_detected}, signals={signals_generated}",
             container.liquidation_strategy.get_stats())
    ))
    _safe_stat("Kelly", lambda: (
        container.kelly_sizer and
        _fmt_kelly(container.kelly_sizer.get_all_sizing_stats())
    ))
    _safe_stat("Memory", lambda: (
        container.trade_memory and
        _fmt("trades={total_trades}, coins={unique_coins}",
             container.trade_memory.get_stats())
    ))
    _safe_stat("Calibration", lambda: (
        container.calibration and
        _fmt_calibration(container.calibration)
    ))
    _safe_stat("LLM Filter", lambda: (
        container.llm_filter and
        _fmt("filtered={total_filtered}, pass_rate={pass_rate:.0%}",
             container.llm_filter.get_stats())
    ))
    _safe_stat("SignalProcessor", lambda: (
        container.signal_processor and
        _fmt("in={total_in} → out={total_out} (reduction={reduction_rate:.0%})",
             container.signal_processor.get_stats())
    ))
    _safe_stat("Incubator", lambda: (
        container.arena_incubator and
        _fmt("incubating={currently_incubating}, promoted={total_promoted}",
             container.arena_incubator.get_stats())
    ))
    _safe_stat("DecisionEngine", lambda: (
        container.decision_engine and
        _fmt("decisions={total_decisions}, executions={total_executions}",
             container.decision_engine.get_stats())
    ))
    _safe_stat("MultiExchange", lambda: (
        container.multi_scanner and
        _fmt_multi(container.multi_scanner.get_stats())
    ))


def _safe_stat(name, fn):
    try:
        result = fn()
        if result:
            logger.info("  %s: %s", name, result)
    except Exception:
        pass


def _fmt(template, stats):
    return template.format(**stats)


def _fmt_kelly(stats):
    edge_count = sum(1 for v in stats.values() if v.get("has_edge"))
    return f"{len(stats)} strategies tracked, {edge_count} with proven edge"


def _fmt_calibration(cal):
    global_ece = cal.get_ece("global")
    ece_str = f"{global_ece:.3f}" if global_ece is not None else "N/A"
    return f"ECE={ece_str}, {len(cal.get_all_stats())} sources tracked"


def _fmt_multi(stats):
    return (
        f"{stats['venue_count']} venues ({', '.join(stats['venues'])}), "
        f"{stats.get('health_check_count', 0)} health checks, "
        f"{stats.get('funding_scan_count', 0)} funding scans, "
        f"{stats['scan_count']} discovery scans, {stats['cached_traders']} cached"
    )
