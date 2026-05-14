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
from src.data import database as db

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
            if (
                int(shadow_summary.get("total_trades", 0) or 0) >= 30
                and float(shadow_summary.get("total_pnl", 0.0) or 0.0) < 0
                and float(shadow_summary.get("avg_win_rate", 0.0) or 0.0) < 0.45
            ):
                logger.warning(
                    "  ShadowTracker degraded: negative 30d shadow edge "
                    "(PnL=$%.2f, WR=%.0f%%). Review/pause weak copy sources.",
                    float(shadow_summary.get("total_pnl", 0.0) or 0.0),
                    float(shadow_summary.get("avg_win_rate", 0.0) or 0.0) * 100,
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
            health = improvement.get("health") or improvement.get("status") or "unknown"
            logger.info("  Bot health: %s", health)
            if str(health).startswith("degraded"):
                runtime_status = improvement.get("strategy_runtime_status") or {}
                logger.warning(
                    "  Strategy engine degraded: total=%s active_valid=%s "
                    "inactive_valid=%s invalid_reasons=%s",
                    runtime_status.get("total", "?"),
                    runtime_status.get("active_valid", "?"),
                    runtime_status.get("inactive_valid", "?"),
                    runtime_status.get("invalid_reasons", {}),
                )
        except Exception:
            pass

    # ── DB backup ──
    try:
        db.backup_to_json()
    except Exception as exc:
        logger.warning("  DB backup failed: %s", exc)

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

    # ── Calibration trend monitor (auto-derisk on deteriorating Brier) ──
    try:
        from src.signals.calibration_trend import apply_trend_derisk
        adjusted = apply_trend_derisk(getattr(container, "agent_scorer", None))
        if adjusted:
            logger.warning(
                "  Calibration trend derisked %d source(s): %s",
                len(adjusted),
                ", ".join(
                    f"{a['source_key']} ({a['old_weight']:.2f}->{a['new_weight']:.2f})"
                    for a in adjusted
                ),
            )
    except Exception as exc:
        logger.debug("  Calibration trend monitor error: %s", exc)

    # ── Orphan position reaper (opt-in via ORPHAN_REAPER_ENABLED) ──
    try:
        from src.trading.orphan_reaper import reap_orphan_positions
        reaped = reap_orphan_positions(container)
        if reaped:
            logger.warning(
                "  Orphan reaper closed %d position(s): %s",
                len(reaped),
                ", ".join(f"{r['coin']} {r['side']}" for r in reaped),
            )
    except Exception as exc:
        logger.debug("  Orphan reaper error: %s", exc)

    # ── Pipeline-health observability ──
    # The calibration loop is broken if these tables don't accumulate
    # rows. Surface stagnation here so it's obvious in ops logs.
    try:
        _check_pipeline_health(container, cycle_count)
    except Exception as exc:
        logger.debug("  Pipeline health check error: %s", exc)

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


# Module-level cache so the health check can detect stagnation across
# cycles without each cycle re-querying counts twice.
_PIPELINE_HEALTH_PREV: dict = {}
_PIPELINE_HEALTH_STREAK: dict = {}


def _check_pipeline_health(container, cycle_count: int) -> None:
    """Warn when calibration/learning tables don't accumulate rows.

    Three tables drive the bot's ability to learn: ``decision_outcomes``
    (what happened to each decision), ``agent_scores`` (per-source
    track records), and ``calibration_records`` (predicted vs realised
    accuracy). If any of these stays at 0 rows or stops growing while
    the bot is taking trades, the gates that depend on them silently
    fall back to cold-start values -- which is what happened on
    2026-05-14 when the bucketed-threshold gate locked the bot into
    its existing positions because global ECE never had real data to
    refine against.
    """
    global _PIPELINE_HEALTH_PREV, _PIPELINE_HEALTH_STREAK
    tables = ("decision_outcomes", "agent_scores", "calibration_records")
    try:
        with db.get_connection(for_read=True) as conn:
            counts = {}
            for t in tables:
                try:
                    counts[t] = int(
                        conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                    )
                except Exception:
                    counts[t] = None
    except Exception as exc:
        logger.debug("  Pipeline health: db query failed: %s", exc)
        return

    for t in tables:
        cur = counts.get(t)
        prev = _PIPELINE_HEALTH_PREV.get(t)
        if cur is None:
            continue
        if prev is None:
            _PIPELINE_HEALTH_PREV[t] = cur
            _PIPELINE_HEALTH_STREAK[t] = 0
            continue
        if cur == prev:
            _PIPELINE_HEALTH_STREAK[t] = _PIPELINE_HEALTH_STREAK.get(t, 0) + 1
        else:
            _PIPELINE_HEALTH_STREAK[t] = 0
        _PIPELINE_HEALTH_PREV[t] = cur

    logger.info(
        "  Pipeline health: decision_outcomes=%s, agent_scores=%s, "
        "calibration_records=%s",
        counts.get("decision_outcomes", "?"),
        counts.get("agent_scores", "?"),
        counts.get("calibration_records", "?"),
    )

    # Warn loudly when a table has been at zero rows for several cycles
    # -- this is the smoking gun that the learning loop isn't writing.
    stagnant_cycles_warn = 5
    for t in tables:
        cur = counts.get(t)
        streak = _PIPELINE_HEALTH_STREAK.get(t, 0)
        if cur == 0 and streak >= stagnant_cycles_warn:
            logger.warning(
                "  Pipeline health: %s has been at 0 rows for %d cycles. "
                "Calibration/promotion gates will fall back to cold-start "
                "values until this table accumulates data.",
                t, streak,
            )


def _log_module_stats(container):
    """Log V2.5+ module statistics."""
    _safe_stat("LCRS", lambda: (
        container.liquidation_strategy and
        _fmt("setups_detected={setups_detected}, signals={signals_generated}, "
             "oi_spike={oi_spike_threshold:.3f}, funding_long={funding_extreme_long:.4f}",
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
        _fmt("decisions={total_decisions}, prescreened_candidates={total_prescreened_candidates}",
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
    global_brier = None
    try:
        global_brier = cal.get_brier("global")
    except Exception:
        global_brier = None
    ece_str = f"{global_ece:.3f}" if global_ece is not None else "N/A"
    brier_str = f"{global_brier:.3f}" if global_brier is not None else "N/A"
    quality = cal._quality_label(global_ece)
    parts = [
        f"ECE={ece_str}",
        f"Brier={brier_str}",
        f"({quality})",
        f"{len(cal.get_all_stats())} sources tracked",
    ]
    quarantined = []
    try:
        quarantined = cal.get_quarantined_sources()
    except Exception:
        quarantined = []
    if quarantined:
        worst = quarantined[:3]
        worst_str = ", ".join(
            f"{q['source']}|{q['side']}|{q['regime']}(ECE={q['ece']:.2f},n={int(q['samples'])})"
            for q in worst
        )
        parts.append(f"quarantined={len(quarantined)} [{worst_str}]")
    if getattr(cal, "is_live_paused", None):
        try:
            if cal.is_live_paused():
                parts.append("LIVE-PAUSED")
        except Exception:
            pass
    return ", ".join(parts)


def _fmt_multi(stats):
    injection_state = "on" if stats.get("lighter_strategy_injection_enabled") else "off"
    return (
        f"{stats['venue_count']} venues ({', '.join(stats['venues'])}), "
        f"{stats.get('health_check_count', 0)} health checks, "
        f"{stats.get('funding_scan_count', 0)} funding scans, "
        f"{stats['scan_count']} discovery scans, {stats['cached_traders']} cached, "
        f"lighter_injected={stats.get('lighter_injected_strategy_count', 0)} "
        f"(injection={injection_state})"
    )
