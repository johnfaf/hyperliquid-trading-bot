"""
Live Simulation Dashboard (V2)
Serves a unified web dashboard on port 8080 with:
  - Main bot dashboard (strategies, paper trades, trader data)
  - Options flow dashboard (convictions, heatmap, unusual prints)

Both dashboards served from the same port for Railway compatibility.
"""
import errno
import hmac
import json
import os
import sys
import threading
from http.cookies import SimpleCookie
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Dict
from urllib.parse import parse_qs, quote, urlparse

import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass
import config
from src.analysis.trade_analytics import compute_live_paper_drift, compute_trade_analytics
from src.core.readiness import evaluate_readiness
from src.data import database as db

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_CLIENT_DISCONNECT_ERRNOS = {
    errno.EPIPE,
    errno.ECONNRESET,
    errno.ECONNABORTED,
}

# Module-level options scanner reference (set by set_options_scanner)
_options_scanner = None

# ─── Dashboard Authentication ─────────────────────────────────
# Paths that are always open (health probes, Railway readiness checks, login)
_AUTH_EXEMPT_PATHS = {"/api/health", "/api/ready", "/api/live_ready", "/login", "/api/auth/login"}
_AUTH_COOKIE_NAME = "dashboard_auth"
# H9 (audit): every mutating / compute-intensive POST endpoint must require
# auth when DASHBOARD_AUTH_TOKEN is set.  When the token is not set and the
# dashboard is bound publicly (e.g., Railway), any path in this set returns
# 403 with a configuration hint instead of executing — this prevents
# unauthenticated CPU/IO exhaustion, destructive cache clears, and external
# API rate-limit abuse on the exchange's behalf.
_AUTH_REQUIRED_POST_PATHS = {
    # Finance / paper-book mutations
    "/api/order",
    "/api/paper/reset",
    "/api/trade/close",
    "/api/trade/close-all",
    # Compute-intensive background jobs (DoS surface)
    "/api/backtest/run",
    "/api/candle-backtest/run",
    "/api/stress/run",
    # External API fetch (exchange rate-limit abuse surface)
    "/api/candle-backtest/fetch",
    # Destructive cache operations
    "/api/candle-backtest/cache/clear",
}


def _dashboard_auth_token() -> str:
    """Read the dashboard auth token lazily so env changes apply immediately."""
    return os.environ.get("DASHBOARD_AUTH_TOKEN", "").strip()


def _secure_token_eq(supplied: str, expected: str) -> bool:
    """
    Constant-time comparison of supplied token vs expected token.

    Using `==` on the dashboard auth token leaks the token one byte at a
    time to a remote attacker via CPU-timing side-channels.  hmac.compare_digest
    runs in time proportional to the longer string regardless of match point.
    """
    if not supplied or not expected:
        return False
    try:
        return hmac.compare_digest(str(supplied), str(expected))
    except (TypeError, ValueError):
        return False


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


def _is_client_disconnect(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (BrokenPipeError, ConnectionResetError, ConnectionAbortedError),
    ) or (isinstance(exc, OSError) and getattr(exc, "errno", None) in _CLIENT_DISCONNECT_ERRNOS)


def _is_hosted_dashboard_environment() -> bool:
    """Detect hosted runtimes that need a public bind host."""
    return any(
        os.environ.get(name, "").strip()
        for name in (
            "RAILWAY_PUBLIC_DOMAIN",
            "RAILWAY_STATIC_URL",
            "RENDER_EXTERNAL_URL",
            "FLY_APP_NAME",
            "K_SERVICE",
        )
    )


def _resolve_dashboard_host() -> str:
    """Resolve dashboard bind host with secure localhost default."""
    explicit = os.environ.get("DASHBOARD_HOST", "").strip()
    if explicit:
        return explicit
    public = _truthy_env("DASHBOARD_BIND_PUBLIC")
    if not public and _is_hosted_dashboard_environment():
        public = True
    return "0.0.0.0" if public else "127.0.0.1"


def _resolve_dashboard_base_url(host: str, port: int) -> str:
    """Return the best URL to present to the operator for dashboard access."""
    explicit = os.environ.get("DASHBOARD_PUBLIC_URL", "").strip().rstrip("/")
    if explicit:
        return explicit

    railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if railway_domain:
        return f"https://{railway_domain}"

    railway_static = os.environ.get("RAILWAY_STATIC_URL", "").strip().rstrip("/")
    if railway_static:
        return railway_static

    render_external = os.environ.get("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if render_external:
        return render_external

    fly_app = os.environ.get("FLY_APP_NAME", "").strip()
    if fly_app:
        return f"https://{fly_app}.fly.dev"

    if host == "0.0.0.0" and not _is_hosted_dashboard_environment():
        return f"http://127.0.0.1:{port}"

    return f"http://{host}:{port}"


def _validate_dashboard_auth_configuration(host: str) -> None:
    """Require auth when the dashboard is publicly exposed from a hosted runtime."""
    if host != "0.0.0.0":
        return
    if _dashboard_auth_token():
        return
    if _is_hosted_dashboard_environment():
        raise RuntimeError(
            "DASHBOARD_AUTH_TOKEN is required for hosted public dashboard access. "
            "Set it in Railway Variables, then log in at /login with that token."
        )
    logger.warning(
        "Dashboard is binding publicly but DASHBOARD_AUTH_TOKEN is not set. "
        "The dashboard will be publicly accessible WITHOUT authentication. "
        "Set DASHBOARD_AUTH_TOKEN to secure it."
    )


def _get_db():
    return db.get_connection(for_read=True)


def _safe_json(obj):
    """JSON serializer that handles numpy/sqlite types."""
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode()
    return str(obj)


def _login_html(error_message: str = "", next_path: str = "/") -> str:
    """Render a small token login page that establishes the dashboard cookie."""
    error_block = (
        f'<p style="margin:0 0 16px;color:#b42318;background:#fef3f2;'
        f'border:1px solid #fecdca;padding:12px 14px;border-radius:10px;">{error_message}</p>'
        if error_message
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard Login</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    :root {{
      --bg: #f7f3ea;
      --panel: #fffaf1;
      --ink: #1f1a17;
      --muted: #6e625a;
      --accent: #1f6f5f;
      --accent-2: #174f44;
      --border: #d9cfc1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top, rgba(31,111,95,0.12), transparent 32%),
        linear-gradient(160deg, #f3ecdf 0%, var(--bg) 42%, #efe7d6 100%);
      color: var(--ink);
      font-family: 'IBM Plex Sans', sans-serif;
      padding: 24px;
    }}
    .card {{
      width: min(100%, 420px);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(43, 35, 24, 0.12);
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: 'Fraunces', serif;
      font-size: 2rem;
      line-height: 1.05;
    }}
    .eyebrow {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }}
    p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.5;
    }}
    label {{
      display: block;
      margin-bottom: 8px;
      font-size: 0.95rem;
      font-weight: 700;
    }}
    input {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      font-size: 1rem;
      margin-bottom: 14px;
      background: #fffdf8;
    }}
    button {{
      width: 100%;
      border: 0;
      border-radius: 999px;
      padding: 14px 18px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #fff;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
    }}
    .hint {{
      margin-top: 16px;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    code {{
      font-family: 'IBM Plex Mono', monospace;
      background: rgba(31, 111, 95, 0.08);
      padding: 0.15rem 0.35rem;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main class="card">
    <p class="eyebrow">Secure Access</p>
    <h1>Dashboard Login</h1>
    <p>Enter the dashboard auth token once. We will set a secure cookie for this browser and send you back to the dashboard.</p>
    {error_block}
    <form method="post" action="/api/auth/login">
      <input type="hidden" name="next" value="{next_path}">
      <label for="token">Auth Token</label>
      <input id="token" name="token" type="password" autocomplete="current-password" required>
      <button type="submit">Open Dashboard</button>
    </form>
    <p class="hint">The token is the value of <code>DASHBOARD_AUTH_TOKEN</code> in Railway Variables.</p>
  </main>
</body>
</html>"""


def _parse_trade_costs(trade: Dict) -> Dict:
    """Extract fee/slippage cost fields from trade metadata with sensible fallbacks."""
    meta = trade.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except Exception:
            meta = {}
    elif not isinstance(meta, dict):
        meta = {}

    fee_rate_bps = (
        float(meta.get("maker_fee_bps", config.PAPER_TRADING_MAKER_FEE_BPS))
        if str(meta.get("execution_role", config.PAPER_TRADING_DEFAULT_EXECUTION_ROLE)).lower() == "maker"
        else float(meta.get("taker_fee_bps", config.PAPER_TRADING_TAKER_FEE_BPS))
    )
    fee_rate = max(fee_rate_bps, 0.0) / 10_000
    size = float(trade.get("size", 0) or 0)
    leverage = float(trade.get("leverage", 1) or 1)
    entry_price = float(trade.get("entry_price", 0) or 0)
    exit_price = float(trade.get("exit_price", 0) or 0)
    entry_notional = entry_price * size * leverage
    exit_notional = exit_price * size * leverage
    fallback_fees = (entry_notional + exit_notional) * fee_rate

    fees_paid = float(meta.get("total_fees_paid", fallback_fees) or 0.0)
    slippage_cost = float(meta.get("total_slippage_cost", 0.0) or 0.0)
    gross_pnl = float(meta.get("gross_pnl_before_fees", (trade.get("pnl", 0) or 0) + fees_paid) or 0.0)
    return {
        "fees_paid": round(fees_paid, 4),
        "slippage_cost": round(slippage_cost, 4),
        "gross_pnl": round(gross_pnl, 4),
        "execution_role": str(meta.get("execution_role", config.PAPER_TRADING_DEFAULT_EXECUTION_ROLE)),
    }


def get_dashboard_data(runtime_snapshot: Dict | None = None):
    """Collect all data needed for the dashboard."""
    with _get_db() as conn:
        # Account
        account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        account = dict(account) if account else {"balance": 0, "total_pnl": 0, "total_trades": 0, "winning_trades": 0}

        # Traders
        traders = [dict(r) for r in conn.execute(
            "SELECT address, account_value, total_pnl, roi_pct, win_rate, trade_count, last_updated "
            "FROM traders WHERE active = ? ORDER BY total_pnl DESC LIMIT 100",
            (True,),
        ).fetchall()]

        # Strategies
        strategies = [dict(r) for r in conn.execute(
            "SELECT id, name, strategy_type, current_score, total_pnl, win_rate, trade_count, discovered_at "
            "FROM strategies WHERE active = ? ORDER BY current_score DESC",
            (True,),
        ).fetchall()]

        # Open paper trades
        open_trades = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades WHERE status = 'open'"
        ).fetchall()]

        # Closed paper trades (last 50)
        closed_trades = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades WHERE status = 'closed' ORDER BY closed_at DESC LIMIT 50"
        ).fetchall()]
        for trade in closed_trades:
            trade.update(_parse_trade_costs(trade))

        # Lifetime execution-cost rollup across all closed trades
        all_closed = [dict(r) for r in conn.execute(
            "SELECT id, side, pnl, entry_price, exit_price, size, leverage, metadata FROM paper_trades WHERE status = 'closed'"
        ).fetchall()]
        total_fees = 0.0
        total_slippage = 0.0
        for trade in all_closed:
            costs = _parse_trade_costs(trade)
            total_fees += costs["fees_paid"]
            total_slippage += costs["slippage_cost"]
        recent_fees = sum(float(t.get("fees_paid", 0) or 0) for t in closed_trades)
        recent_slippage = sum(float(t.get("slippage_cost", 0) or 0) for t in closed_trades)
        trade_analytics = compute_trade_analytics(all_closed, source_limit=10)
        audit_rows = [dict(r) for r in conn.execute(
            "SELECT timestamp, action, coin, side, source, details "
            "FROM audit_trail ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()]

        # Copy trades: filter in Python so this works on both SQLite TEXT and Postgres JSONB.
        recent_trade_rows = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades ORDER BY opened_at DESC LIMIT 120"
        ).fetchall()]
        copy_trades = []
        for trade in recent_trade_rows:
            metadata = trade.get("metadata")
            if isinstance(metadata, (dict, list)):
                metadata_blob = json.dumps(metadata)
            else:
                metadata_blob = str(metadata or "")
            if "copy_trade" in metadata_blob or "golden_wallet" in metadata_blob:
                copy_trades.append(trade)
                if len(copy_trades) >= 30:
                    break

        # Strategy type distribution
        type_dist = [dict(r) for r in conn.execute(
            "SELECT strategy_type, COUNT(*) as count, AVG(current_score) as avg_score, "
            "SUM(total_pnl) as total_pnl FROM strategies WHERE active = ? "
            "GROUP BY strategy_type ORDER BY count DESC",
            (True,),
        ).fetchall()]

        # Recent research logs
        logs = [dict(r) for r in conn.execute(
            "SELECT timestamp, cycle_type, summary, traders_analyzed, strategies_found "
            "FROM research_logs ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()]

        # Score history for top strategies
        score_history = {}
        for s in strategies[:5]:
            rows = conn.execute(
                "SELECT timestamp, score FROM strategy_scores WHERE strategy_id = ? "
                "ORDER BY timestamp DESC LIMIT 30", (s["id"],)
            ).fetchall()
            score_history[s["id"]] = [{"t": r["timestamp"], "s": round(r["score"], 4)} for r in reversed(rows)]

        initial_balance = config.PAPER_TRADING_INITIAL_BALANCE
        balance = account.get("balance", initial_balance)
        roi = ((balance / initial_balance) - 1) * 100 if initial_balance else 0
        try:
            events = _event_scanner.get_dashboard_data() if _event_scanner else None
        except Exception:
            events = None
        runtime_snapshot = runtime_snapshot or _build_runtime_health_snapshot()
        drift_analytics = compute_live_paper_drift(
            closed_trades=all_closed,
            open_trades=open_trades,
            audit_rows=audit_rows,
            live_source_orders_today=((runtime_snapshot.get("live_trader", {}) or {}).get("source_orders_today", {}) or {}),
            source_limit=10,
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account": {
                "balance": balance,
                "total_pnl": account.get("total_pnl", 0),
                "gross_pnl_estimate": account.get("total_pnl", 0) + total_fees,
                "lifetime_fees": round(total_fees, 2),
                "lifetime_slippage": round(total_slippage, 2),
                "lifetime_execution_cost": round(total_fees + total_slippage, 2),
                "recent_fees": round(recent_fees, 2),
                "recent_slippage": round(recent_slippage, 2),
                "roi_pct": round(roi, 2),
                "total_trades": account.get("total_trades", 0),
                "winning_trades": account.get("winning_trades", 0),
                "win_rate": round(account["winning_trades"] / account["total_trades"] * 100, 1) if account.get("total_trades", 0) > 0 else 0,
                "initial_balance": initial_balance,
            },
            "traders": traders,
            "strategies": strategies,
            "open_trades": open_trades,
            "closed_trades": closed_trades,
            "copy_trades": copy_trades,
            "type_distribution": type_dist,
            "research_logs": logs,
            "score_history": score_history,
            "events": events,
            "trade_analytics": trade_analytics,
            "drift_analytics": drift_analytics,
            "v2": _get_v2_metrics(conn),
        }


def _get_v2_metrics(conn) -> Dict:
    """Collect V2 pipeline metrics (firewall, agent scores, regime)."""
    v2 = {
        "firewall": {},
        "agent_scores": [],
        "source_scorecard": [],
        "regime": {},
    }

    try:
        # Firewall stats - pulled from the global firewall instance if available
        pass
        # We'll populate this from the /api/data handler if firewall is set
    except Exception:
        pass

    try:
        # Agent scores from DB
        rows = conn.execute(
            "SELECT source_key, total_signals, correct_signals, total_pnl, "
            "accuracy, sharpe, dynamic_weight, last_updated "
            "FROM agent_scores ORDER BY dynamic_weight DESC LIMIT 20"
        ).fetchall()
        v2["agent_scores"] = [dict(r) for r in rows]
    except Exception:
        pass  # Table may not exist yet

    try:
        if _agent_scorer:
            v2["source_scorecard"] = _agent_scorer.get_scorecard()[:20]
    except Exception:
        pass

    try:
        if _shadow_tracker:
            v2["shadow_summary"] = _shadow_tracker.get_summary(days=30)
    except Exception:
        pass

    return v2


# Module-level references for V2 + V2.5 + V3 components (set by set_v2_components)
_firewall = None
_regime_detector = None
_arena = None
_agent_scorer = None
_kelly_sizer = None
_trade_memory = None
_calibration = None
_llm_filter = None
_liquidation_strategy = None
_signal_processor = None
_arena_incubator = None
_decision_engine = None
_multi_scanner = None
_event_scanner = None
_health_registry = None
_shadow_tracker = None
_copy_trader = None

# Module-level live trader reference for closing positions on exchange
_live_trader = None


def set_live_trader(trader):
    """Set the live trader reference so the dashboard can close live positions."""
    global _live_trader  # noqa: PLW0603
    _live_trader = trader


def _close_paper_trade_at_market(trade_id: int) -> dict:
    """
    Close a single paper trade at the current market price.
    If live trading is active, also closes the position on exchange.

    Returns dict with status and details.
    """
    from src.data.hyperliquid_client import get_all_mids

    # Fetch the trade
    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM paper_trades WHERE id = ? AND status = 'open'", (trade_id,)
        ).fetchone()

    if not row:
        return {"error": f"Trade {trade_id} not found or already closed", "status": "error"}

    trade = dict(row)
    coin = trade.get("coin", "")
    side = trade.get("side", "")
    entry_price = float(trade.get("entry_price", 0) or 0)
    size = float(trade.get("size", 0) or 0)
    leverage = float(trade.get("leverage", 1) or 1)

    # Get current market price
    mids = get_all_mids() or {}
    current_price = float(mids.get(coin, 0) or 0)
    if current_price <= 0:
        return {"error": f"Cannot get market price for {coin}", "status": "error"}

    # Calculate PnL
    if side == "long":
        pnl = (current_price - entry_price) * size * leverage
    else:
        pnl = (entry_price - current_price) * size * leverage
    pnl = round(pnl, 2)

    # Update metadata
    try:
        existing_meta = trade.get("metadata", {})
        if isinstance(existing_meta, str):
            existing_meta = json.loads(existing_meta or "{}")
        existing_meta = dict(existing_meta or {})
    except Exception:
        existing_meta = {}
    existing_meta["manual_close"] = True
    existing_meta["close_source"] = "dashboard"
    db.update_paper_trade_metadata(trade_id, existing_meta)

    # Ensure paper_account singleton exists before the atomic close+credit,
    # otherwise the combined function will roll back (account row required).
    if not db.get_paper_account():
        db.init_paper_account(config.PAPER_TRADING_INITIAL_BALANCE)

    # CRIT-FIX C2: atomic close + account credit in one transaction.
    if not db.close_paper_trade_and_credit_account(trade_id, current_price, pnl):
        return {"error": f"Failed to close trade {trade_id} in DB", "status": "error"}

    # If live trading is active, also close the position on exchange
    live_close_result = None
    if _live_trader:
        try:
            if hasattr(_live_trader, 'is_live_enabled') and _live_trader.is_live_enabled():
                if hasattr(_live_trader, 'is_deployable') and _live_trader.is_deployable():
                    live_close_result = _live_trader.close_position(coin)
                    logger.info("Live position closed for %s: %s", coin, live_close_result)
        except Exception as e:
            logger.error("Failed to close live position for %s: %s", coin, e)
            live_close_result = {"status": "error", "message": str(e)}

    logger.info(
        "Manually closed trade #%d: %s %s @ $%.2f -> $%.2f, PnL: $%.2f",
        trade_id, side.upper(), coin, entry_price, current_price, pnl,
    )
    return {
        "status": "ok",
        "trade_id": trade_id,
        "coin": coin,
        "side": side,
        "entry_price": entry_price,
        "exit_price": current_price,
        "pnl": pnl,
        "live_close": live_close_result,
    }


def set_v2_components(firewall=None, regime_detector=None, arena=None,
                       agent_scorer=None, kelly_sizer=None, trade_memory=None,
                       calibration=None, llm_filter=None,
                       liquidation_strategy=None, signal_processor=None,
                       arena_incubator=None, decision_engine=None,
                       multi_scanner=None, event_scanner=None, shadow_tracker=None,
                       health_registry=None, copy_trader=None):
    """Set V2 + V2.5 + V3 + V4 component references for dashboard metrics."""
    global _firewall, _regime_detector, _arena  # noqa: PLW0603
    global _agent_scorer, _shadow_tracker  # noqa: PLW0603
    global _kelly_sizer, _trade_memory, _calibration, _llm_filter, _liquidation_strategy  # noqa
    global _signal_processor, _arena_incubator, _decision_engine  # noqa
    global _multi_scanner, _event_scanner, _health_registry, _copy_trader  # noqa
    _firewall = firewall
    _regime_detector = regime_detector
    _arena = arena
    _agent_scorer = agent_scorer
    _kelly_sizer = kelly_sizer
    _trade_memory = trade_memory
    _calibration = calibration
    _llm_filter = llm_filter
    _liquidation_strategy = liquidation_strategy
    _signal_processor = signal_processor
    _arena_incubator = arena_incubator
    _decision_engine = decision_engine
    _multi_scanner = multi_scanner
    _event_scanner = event_scanner
    _shadow_tracker = shadow_tracker
    _health_registry = health_registry
    _copy_trader = copy_trader


def _build_runtime_health_snapshot() -> Dict:
    """Build a compact runtime health payload for dashboard/API consumers."""
    now = datetime.now(timezone.utc)
    snapshot: Dict[str, object] = {
        "timestamp": now.isoformat(),
        "overall": "unknown",
        "all_trading_safe": None,
        "subsystem_counts": {},
        "stale_subsystems": [],
        "at_risk_subsystems": [],
        "subsystems": {},
    }
    stale_timeout_s = max(
        30,
        int(os.environ.get("DASHBOARD_HEALTH_STALE_SECONDS", "600")),
    )

    try:
        if _health_registry:
            statuses = _health_registry.get_all()
            state_counts: Dict[str, int] = {}
            stale_names = []
            at_risk = []
            subsystem_map = {}
            for name, status in statuses.items():
                state = status.state.value
                state_counts[state] = state_counts.get(state, 0) + 1

                heartbeat_age_s = None
                if status.last_heartbeat:
                    heartbeat_age_s = (now - status.last_heartbeat).total_seconds()
                    if heartbeat_age_s > stale_timeout_s:
                        stale_names.append(name)

                if state in {"FAILED", "DEGRADED"}:
                    at_risk.append(name)

                subsystem_map[name] = {
                    "state": state,
                    "reason": status.reason,
                    "dependency_ready": bool(status.dependency_ready),
                    "affects_trading": bool(status.affects_trading),
                    "heartbeat_age_s": round(heartbeat_age_s, 1)
                    if heartbeat_age_s is not None
                    else None,
                }

            all_safe = bool(_health_registry.is_all_trading_safe())
            overall = "ok" if all_safe and not at_risk else "at_risk"
            snapshot.update(
                {
                    "overall": overall,
                    "all_trading_safe": all_safe,
                    "subsystem_counts": state_counts,
                    "stale_subsystems": stale_names,
                    "at_risk_subsystems": at_risk,
                    "subsystems": subsystem_map,
                }
            )
    except Exception as exc:
        snapshot["error"] = f"health_registry_unavailable: {exc}"

    try:
        class _Container:
            live_trader = _live_trader

        snapshot["readiness"] = evaluate_readiness(
            container=_Container(),
            health_registry=_health_registry,
            stale_seconds=stale_timeout_s,
        )
    except Exception as exc:
        snapshot["readiness"] = {"ready": False, "live_ready": False, "error": str(exc)}

    try:
        if _firewall:
            fw = _firewall.get_stats()
            snapshot["firewall"] = {
                "total_signals": int(fw.get("total_signals", 0) or 0),
                "passed": int(fw.get("passed", 0) or 0),
                "top_rejection_reason": fw.get("top_rejection_reason", "none"),
                "daily_losses": float(fw.get("daily_losses", 0.0) or 0.0),
                "source_policies": fw.get("source_policies", [])[:10],
                "short_side_policy": fw.get("short_side_policy", {}),
            }
    except Exception:
        pass

    try:
        if _live_trader:
            stats = _live_trader.get_stats()
            snapshot["live_trader"] = {
                "kill_switch_active": bool(stats.get("kill_switch_active", False)),
                "kill_switch_reason": stats.get("kill_switch_reason"),
                "canary_mode": bool(stats.get("canary_mode", False)),
                "max_order_usd": stats.get("max_order_usd"),
                "daily_pnl": stats.get("daily_pnl"),
                "daily_pnl_limit": stats.get("daily_pnl_limit"),
                "entry_signals_today": stats.get("total_entry_signals_today"),
                "attempted_entry_signals": stats.get("attempted_entry_signals"),
                "executed_entry_signals": stats.get("executed_entry_signals"),
                "max_orders_per_source_per_day": stats.get("max_orders_per_source_per_day"),
                "source_orders_today": stats.get("source_orders_today", {}),
                "source_policies": stats.get("source_policies", [])[:10],
                "short_side_policy": stats.get("short_side_policy", {}),
                "entry_metrics": stats.get("entry_metrics", {}),
                "min_order_rejects_today": stats.get("min_order_rejects_today"),
                "min_order_floorups_today": stats.get("min_order_floorups_today"),
                "min_order_top_tier_floorups_today": stats.get("min_order_top_tier_floorups_today"),
                "min_order_same_side_merges_today": stats.get("min_order_same_side_merges_today"),
                "approved_but_not_executable_today": stats.get("approved_but_not_executable_today"),
                "canary_headroom_ratio": stats.get("canary_headroom_ratio"),
                "crash_safe_canary_order_usd": stats.get("crash_safe_canary_order_usd"),
            }
    except Exception:
        pass

    try:
        if _copy_trader and hasattr(_copy_trader, "get_stats"):
            snapshot["copy_trader"] = _copy_trader.get_stats()
    except Exception:
        pass

    return snapshot


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hyperliquid Trading Cockpit</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --bg:#f6f2ea;
  --panel:#fffdf9;
  --panel-soft:#f9f4ec;
  --ink:#181512;
  --muted:#6b645c;
  --line:#e3d8c7;
  --teal:#1f6f5f;
  --teal-soft:#dcebe6;
  --blue:#2f5b9f;
  --amber:#b9771f;
  --red:#b54d3f;
  --green:#207f59;
  --shadow:0 14px 34px rgba(40,31,20,.04);
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:
    radial-gradient(circle at top, rgba(31,111,95,.06), transparent 32%),
    linear-gradient(180deg,#fbf8f1 0%,var(--bg) 100%);
  color:var(--ink);
  font-family:'IBM Plex Sans',sans-serif;
  padding:28px 18px 44px;
}
a{color:inherit}
.page-shell{max-width:1460px;margin:0 auto}
.masthead{
  display:flex;justify-content:space-between;align-items:flex-start;gap:18px;
  margin-bottom:26px;padding:0 2px 18px;border-bottom:1px solid rgba(227,216,199,.9);
}
.eyebrow,.section-tag{
  font-size:.74rem;text-transform:uppercase;letter-spacing:.16em;color:var(--teal);font-weight:700;
}
h1{
  font-family:'Fraunces',serif;font-size:clamp(2.15rem,3.4vw,3.7rem);line-height:.96;margin:8px 0 10px;color:var(--ink);
}
.subtitle{max-width:700px;color:var(--muted);font-size:.95rem;line-height:1.6}
.status-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}
.status-chip{
  display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;
  background:var(--panel);border:1px solid var(--line);font-size:.82rem;font-weight:600;color:var(--ink);
}
.ws-dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--red)}
.nav-cluster{display:flex;flex-wrap:wrap;gap:10px;justify-content:flex-end}
.nav-pill{
  min-width:138px;padding:11px 14px;border-radius:14px;text-decoration:none;border:1px solid var(--line);
  background:rgba(255,255,255,.72);display:flex;flex-direction:column;gap:2px;transition:border-color .15s ease,transform .15s ease,background .15s ease;
}
.nav-pill:hover{transform:translateY(-1px);border-color:#c6b8a1;background:#fff}
.nav-pill strong{font-size:.9rem;color:var(--ink)}
.nav-pill span{font-size:.73rem;color:var(--muted);line-height:1.35}
.metric-band{margin-bottom:18px}
.metric-band .grid{grid-template-columns:repeat(4,minmax(0,1fr))}
.section-head{display:flex;justify-content:space-between;align-items:flex-end;gap:14px;margin-bottom:12px}
.section-title{font-family:'Fraunces',serif;font-size:1.34rem;line-height:1.08}
.section-note{color:var(--muted);font-size:.88rem;line-height:1.5;max-width:54ch}
.section-actions{display:flex;flex-wrap:wrap;gap:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.card{
  padding:18px;border-radius:18px;border:1px solid var(--line);
  background:linear-gradient(180deg,#fffdf9 0%,#fcf8f1 100%);box-shadow:var(--shadow);
}
.card .label{color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.14em;font-weight:700}
.card .value{font-size:1.95rem;font-weight:700;margin-top:12px;line-height:1;font-family:'IBM Plex Mono',monospace}
.card .sub{color:var(--muted);font-size:.84rem;margin-top:9px;line-height:1.45}
.green{color:var(--green)} .red{color:var(--red)} .yellow{color:var(--amber)} .blue{color:var(--blue)}
.layout{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(360px,.9fr);gap:18px;margin-bottom:18px}
.panel-stack{display:grid;gap:18px}
.section{
  background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:20px;box-shadow:var(--shadow);
}
.table-shell{overflow:auto;border:1px solid var(--line);border-radius:16px;background:#fff}
table{width:100%;border-collapse:collapse;font-size:.88rem;min-width:680px}
th{
  position:sticky;top:0;background:rgba(249,244,236,.96);color:var(--muted);text-align:left;padding:11px 13px;
  border-bottom:1px solid var(--line);font-weight:700;text-transform:uppercase;font-size:.7rem;letter-spacing:.11em;
}
td{padding:11px 13px;border-bottom:1px solid rgba(227,216,199,.75);vertical-align:middle}
tr:hover td{background:rgba(249,244,236,.65)}
.badge{display:inline-flex;align-items:center;gap:6px;padding:4px 9px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.04em;border:1px solid transparent}
.badge-long{background:rgba(32,127,89,.1);color:var(--green);border-color:rgba(32,127,89,.18)}
.badge-short{background:rgba(181,77,63,.1);color:var(--red);border-color:rgba(181,77,63,.18)}
.badge-type{background:rgba(47,91,159,.08);color:var(--blue);border-color:rgba(47,91,159,.14)}
code{font-family:'IBM Plex Mono',monospace;background:rgba(31,111,95,.07);padding:2px 6px;border-radius:8px;font-size:.82rem}
canvas{width:100%!important;height:304px!important}
#type-chart{height:304px!important}
#equity-chart{height:304px!important}
.btn{
  padding:9px 13px;border-radius:999px;border:none;cursor:pointer;font-size:.8rem;font-weight:700;transition:transform .15s ease,opacity .15s ease;
}
.btn:hover{opacity:.92;transform:translateY(-1px)}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}
.btn-close,.btn-close-all{background:rgba(181,77,63,.10);color:var(--red);border:1px solid rgba(181,77,63,.18)}
.btn-reset{background:rgba(185,119,31,.10);color:var(--amber);border:1px solid rgba(185,119,31,.2)}
.btn-success{background:rgba(32,127,89,.1);color:var(--green);border:1px solid rgba(32,127,89,.2)}
.detail-list{color:var(--muted);font-size:.88rem;line-height:1.65;margin-top:10px}
.detail-list strong{color:var(--ink)}
.split-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-bottom:18px}
.details-grid{display:grid;gap:14px}
details.section{padding:0;overflow:hidden}
details.section[open]{background:var(--panel)}
details summary{
  list-style:none;cursor:pointer;padding:18px;display:flex;justify-content:space-between;align-items:center;gap:10px;
}
details summary::-webkit-details-marker{display:none}
details summary::after{
  content:'+';display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:999px;
  border:1px solid var(--line);color:var(--muted);font-size:1rem;flex:0 0 auto;background:var(--panel-soft);
}
details[open] summary::after{content:'-'}
.summary-copy{color:var(--muted);font-size:.84rem}
.details-body{padding:0 18px 18px}
.empty-row{padding:20px 14px!important;color:var(--muted);text-align:center;font-style:italic}
.table-note{display:block;margin-top:4px;color:var(--muted);font-size:.76rem;line-height:1.45}
.reason-tag{
  display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;margin:0 8px 8px 0;
  font-size:.7rem;font-weight:700;letter-spacing:.04em;background:rgba(181,77,63,.08);color:var(--red);
  border:1px solid rgba(181,77,63,.14)
}
.regime-card{padding:18px;text-align:center}
.regime-card .value{font-size:1rem}
.consensus-item{padding:12px 13px;border-radius:14px;background:var(--panel-soft);border:1px solid var(--line)}
.consensus-meta{display:block;margin-top:6px;color:var(--muted);font-size:.8rem;line-height:1.45}
.agent-key{display:inline-block;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.runtime-grid{display:grid;gap:12px}
@media (max-width: 1180px){
  .layout,.split-grid{grid-template-columns:1fr}
  .metric-band .grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width: 820px){
  body{padding:16px 12px 28px}
  .masthead{flex-direction:column}
  .nav-cluster{width:100%}
  .nav-pill{flex:1}
  h1{font-size:2rem}
  .section,.card{border-radius:16px}
  .metric-band .grid{grid-template-columns:1fr}
  canvas,#type-chart,#equity-chart{height:250px!important}
}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head>
<body>
<div class="page-shell">
  <header class="masthead">
    <div>
      <p class="eyebrow">Trading Desk</p>
      <h1>Hyperliquid Cockpit</h1>
      <p class="subtitle">A cleaner operator view for readiness, positions, allocator quality, and live trading pressure. The goal is to make the next decision obvious, not to show every metric equally.</p>
      <div class="status-row">
        <span class="status-chip">Overview</span>
        <span class="status-chip"><span id="update-time">loading...</span></span>
        <span class="status-chip"><span id="ws-status" class="ws-dot" title="WebSocket disconnected"></span> live mids</span>
      </div>
    </div>
    <div class="nav-cluster">
      <a href="/options" class="nav-pill"><strong>Options</strong><span>Flow and unusual prints</span></a>
      <a href="/backtest" class="nav-pill"><strong>Backtest</strong><span>Replay changes safely</span></a>
      <a href="/stress" class="nav-pill"><strong>Stress</strong><span>Scenario pressure testing</span></a>
    </div>
  </header>

  <section class="metric-band">
    <div class="grid" id="stats-cards"></div>
  </section>

  <section class="layout">
    <article class="section">
      <div class="section-head">
        <div>
          <p class="section-tag">Execution</p>
          <h2 class="section-title">Open Positions</h2>
        </div>
        <div class="section-actions">
          <button class="btn btn-close-all" onclick="closeAllTrades()" id="btn-close-all" title="Close all open positions at market price">Close all</button>
          <button class="btn btn-reset" onclick="resetPaperTrading()" id="btn-reset" title="Reset paper trading history (keeps discovered traders and strategies)">Reset history</button>
        </div>
      </div>
      <div class="table-shell">
        <table>
          <thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Current</th><th>Size</th><th>Lev</th><th>Unrealized</th><th>SL</th><th>TP</th><th>Strategy</th><th>Action</th></tr></thead>
          <tbody id="open-trades"></tbody>
        </table>
      </div>
    </article>
    <aside class="panel-stack">
      <section class="section">
        <div class="section-head">
          <div>
            <p class="section-tag">Runtime</p>
            <h2 class="section-title">Readiness</h2>
          </div>
        </div>
        <div class="runtime-grid">
          <div class="grid" id="runtime-cards"></div>
          <div id="runtime-health-detail" class="detail-list">Loading runtime health...</div>
        </div>
      </section>
      <section class="section">
        <div class="section-head">
          <div>
            <p class="section-tag">Copy Flow</p>
            <h2 class="section-title">Mirrored Trades</h2>
          </div>
        </div>
        <div class="table-shell">
          <table>
            <thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Size</th><th>Source</th><th>Status</th><th>PnL</th></tr></thead>
            <tbody id="copy-trades"></tbody>
          </table>
        </div>
      </section>
    </aside>
  </section>

  <section class="split-grid">
    <article class="section">
      <div class="section-head">
        <div>
          <p class="section-tag">Performance</p>
          <h2 class="section-title">Equity Curve</h2>
        </div>
      </div>
      <canvas id="equity-chart"></canvas>
    </article>
    <article class="section">
      <div class="section-head">
        <div>
          <p class="section-tag">Mix</p>
          <h2 class="section-title">Strategy Distribution</h2>
        </div>
      </div>
      <canvas id="type-chart"></canvas>
    </article>
  </section>

  <section class="split-grid">
    <article class="section">
      <div class="section-head">
        <div>
          <p class="section-tag">Decision Layer</p>
          <h2 class="section-title">Pipeline</h2>
        </div>
      </div>
      <div class="grid" id="v2-cards"></div>
      <div style="margin-top:14px" id="firewall-stats" class="detail-list">Loading...</div>
    </article>
    <article class="section">
      <div class="section-head">
        <div>
          <p class="section-tag">Source Quality</p>
          <h2 class="section-title">Scorecard</h2>
        </div>
      </div>
      <div class="table-shell">
        <table>
          <thead><tr><th>Source</th><th>Trades</th><th>Accuracy</th><th>Sharpe</th><th>Weight</th><th>PnL</th></tr></thead>
          <tbody id="agent-scores"></tbody>
        </table>
      </div>
    </article>
  </section>

  <section class="details-grid">
    <details class="section" open>
      <summary>
        <div>
          <p class="section-tag">Audit Trail</p>
          <h2 class="section-title">Closed Trades</h2>
        </div>
        <span class="summary-copy">Realized outcomes and execution costs</span>
      </summary>
      <div class="details-body">
        <div class="table-shell">
          <table>
            <thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Exit</th><th>Gross PnL</th><th>Fees</th><th>Slippage</th><th>Net PnL</th><th>Lev</th><th>Closed</th></tr></thead>
            <tbody id="closed-trades"></tbody>
          </table>
        </div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Execution Analytics</p>
          <h2 class="section-title">Sides, Sources, and Canary</h2>
        </div>
        <span class="summary-copy">Recent side performance, realized source results, and live execution friction</span>
      </summary>
      <div class="details-body">
        <div id="analytics-summary" class="detail-list">Loading execution analytics...</div>
        <div class="split-grid" style="margin-top:14px;margin-bottom:0">
          <div class="table-shell">
            <table>
              <thead><tr><th>Side</th><th>Trades</th><th>Win Rate</th><th>Fees</th><th>Net PnL</th></tr></thead>
              <tbody id="analytics-sides"></tbody>
            </table>
          </div>
          <div class="table-shell">
            <table>
              <thead><tr><th>Source</th><th>Trades</th><th>Win Rate</th><th>Fees</th><th>Net PnL</th></tr></thead>
              <tbody id="analytics-sources"></tbody>
            </table>
          </div>
          <div class="table-shell">
            <table>
              <thead><tr><th>Coin / Side</th><th>Trades</th><th>Win Rate</th><th>Fees</th><th>Net PnL</th></tr></thead>
              <tbody id="analytics-coin-sides"></tbody>
            </table>
          </div>
          <div class="table-shell">
            <table>
              <thead><tr><th>Live Metric</th><th>Value</th></tr></thead>
              <tbody id="analytics-live"></tbody>
            </table>
          </div>
        </div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Drift Monitor</p>
          <h2 class="section-title">Live vs Paper</h2>
        </div>
        <span class="summary-copy">Source-level drift between paper outcomes, audit approvals, and live entry flow</span>
      </summary>
      <div class="details-body">
        <div id="drift-summary" class="detail-list">Loading live vs paper drift...</div>
        <div class="split-grid" style="margin-top:14px;margin-bottom:0">
          <div class="table-shell">
            <table>
              <thead><tr><th>Source</th><th>Paper Open</th><th>Paper Closed</th><th>Paper PnL</th><th>Approved</th><th>Rejected</th><th>Live Today</th><th>Top Reject</th></tr></thead>
              <tbody id="drift-source-table"></tbody>
            </table>
          </div>
          <div class="table-shell">
            <table>
              <thead><tr><th>Side</th><th>Paper Open</th><th>Paper Closed</th><th>Paper PnL</th><th>Approved</th><th>Rejected</th></tr></thead>
              <tbody id="drift-side-table"></tbody>
            </table>
          </div>
        </div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Regime</p>
          <h2 class="section-title">Per-Coin Context</h2>
        </div>
        <span class="summary-copy">Market state, ADX, ATR, and confidence by coin</span>
      </summary>
      <div class="details-body">
        <div id="regime-grid" class="grid"></div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Research</p>
          <h2 class="section-title">Strategy Rankings</h2>
        </div>
        <span class="summary-copy">Top strategies by current score and realized outcomes</span>
      </summary>
      <div class="details-body">
        <div class="table-shell">
          <table>
            <thead><tr><th>#</th><th>Strategy</th><th>Type</th><th>Score</th><th>PnL</th><th>Win Rate</th><th>Trades</th></tr></thead>
            <tbody id="strategies-table"></tbody>
          </table>
        </div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Discovery</p>
          <h2 class="section-title">Tracked Traders</h2>
        </div>
        <span class="summary-copy">Addresses, account values, and copy-trading candidates</span>
      </summary>
      <div class="details-body">
        <div class="table-shell">
          <table>
            <thead><tr><th>#</th><th>Address</th><th>Account Value</th><th>PnL</th><th>Win Rate</th><th>Trades</th></tr></thead>
            <tbody id="traders-table"></tbody>
          </table>
        </div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Events</p>
          <h2 class="section-title">Macro and Policy Calendar</h2>
        </div>
        <span class="summary-copy">Upcoming core data and recent official releases</span>
      </summary>
      <div class="details-body">
        <div id="events-summary" class="detail-list">Loading event scanner...</div>
        <div class="split-grid" style="margin-top:14px;margin-bottom:0">
          <div class="table-shell">
            <table>
              <thead><tr><th>Time</th><th>Source</th><th>Severity</th><th>Event</th></tr></thead>
              <tbody id="events-upcoming"></tbody>
            </table>
          </div>
          <div class="table-shell">
            <table>
              <thead><tr><th>Published</th><th>Source</th><th>Category</th><th>Release</th></tr></thead>
              <tbody id="events-recent"></tbody>
            </table>
          </div>
          <div class="table-shell">
            <table>
              <thead><tr><th>Source</th><th>Impact</th><th>Assets</th><th>Incident</th></tr></thead>
              <tbody id="events-incidents"></tbody>
            </table>
          </div>
        </div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Cycle Feed</p>
          <h2 class="section-title">Research Activity</h2>
        </div>
        <span class="summary-copy">Recent scans, discoveries, and cycle summaries</span>
      </summary>
      <div class="details-body">
        <div class="table-shell">
          <table>
            <thead><tr><th>Time</th><th>Type</th><th>Summary</th><th>Traders</th><th>Strategies</th></tr></thead>
            <tbody id="logs-table"></tbody>
          </table>
        </div>
      </div>
    </details>

    <details class="section">
      <summary>
        <div>
          <p class="section-tag">Arena</p>
          <h2 class="section-title">Alpha Arena</h2>
        </div>
        <span class="summary-copy">Leaderboard and recent consensus votes</span>
      </summary>
      <div class="details-body">
        <div class="grid" id="arena-cards" style="margin-bottom:14px"></div>
        <div class="table-shell" style="margin-bottom:14px">
          <table>
            <thead><tr><th>#</th><th>Agent</th><th>Strategy</th><th>Status</th><th>ELO</th><th>Fitness</th><th>Capital</th><th>PnL</th><th>Trades</th><th>Win%</th><th>Sharpe</th><th>Gen</th></tr></thead>
            <tbody id="arena-leaderboard"></tbody>
          </table>
        </div>
        <div id="consensus-log" class="detail-list"></div>
      </div>
    </details>
  </section>
</div>
<script>
// WebSocket live price feed
let livePrices = {};
let ws = null;
let wsReconnectTimer = null;

function connectWebSocket(){
  try {
    ws = new WebSocket('wss://api.hyperliquid.xyz/ws');
    ws.onopen = () => {
      console.log('WebSocket connected');
      document.getElementById('ws-status').style.background = '#207f59';
      document.getElementById('ws-status').title = 'WebSocket connected';
      const msg = JSON.stringify({method: 'subscribe', subscription: {type: 'allMids'}});
      ws.send(msg);
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if(data.channel === 'allMids' && data.data && data.data.mids) {
          livePrices = data.data.mids;
          renderOpenTrades(window.currentOpenTrades || []);
        }
      } catch(e) {
        console.error('WebSocket message error:', e);
      }
    };
    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      document.getElementById('ws-status').style.background = '#b54d3f';
      document.getElementById('ws-status').title = 'WebSocket error';
    };
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      document.getElementById('ws-status').style.background = '#b54d3f';
      document.getElementById('ws-status').title = 'WebSocket disconnected';
      clearTimeout(wsReconnectTimer);
      wsReconnectTimer = setTimeout(connectWebSocket, 5000);
    };
  } catch(e) {
    console.error('WebSocket connection error:', e);
    document.getElementById('ws-status').style.background = '#b54d3f';
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = setTimeout(connectWebSocket, 5000);
  }
}

let typeChart = null;
let liveUpdateInterval = null;

function fmt(n, d=2){ return n != null ? Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) : 'n/a' }
function fmtUsd(n){ return n != null ? '$' + fmt(n) : 'n/a' }
function pnlClass(n){ return n > 0 ? 'green' : n < 0 ? 'red' : '' }
function shortAddr(a){ return a ? a.slice(0,6)+'...'+a.slice(-4) : 'n/a' }

function renderCards(d){
  const a = d.account;
  const cards = [
    {label:'Paper Balance', value:fmtUsd(a.balance), cls:pnlClass(a.total_pnl)},
    {label:'Total PnL', value:fmtUsd(a.total_pnl), cls:pnlClass(a.total_pnl), sub:`ROI: ${a.roi_pct>0?'+':''}${a.roi_pct}%`},
    {label:'Gross PnL (Est.)', value:fmtUsd(a.gross_pnl_estimate), cls:pnlClass(a.gross_pnl_estimate)},
    {label:'Execution Costs', value:fmtUsd(-(a.lifetime_execution_cost||0)), cls:'red', sub:`Fees ${fmtUsd(a.lifetime_fees||0)} + Slip ${fmtUsd(a.lifetime_slippage||0)}`},
    {label:'Win Rate', value:a.win_rate+'%', cls:a.win_rate>=50?'green':'yellow', sub:`${a.winning_trades}/${a.total_trades} trades`},
    {label:'Tracked Traders', value:d.traders.length, cls:'blue'},
    {label:'Active Strategies', value:d.strategies.length, cls:'green'},
    {label:'Open Positions', value:d.open_trades.length, cls:'yellow'},
  ];
  document.getElementById('stats-cards').innerHTML = cards.map(c=>`
    <div class="card"><div class="label">${c.label}</div>
    <div class="value ${c.cls||''}">${c.value}</div>
    ${c.sub?`<div class="sub">${c.sub}</div>`:''}</div>`).join('');
}

function renderOpenTrades(trades){
  document.getElementById('open-trades').innerHTML = trades.length ? trades.map(t=>{
    const currentPrice = livePrices[t.coin] ? parseFloat(livePrices[t.coin]) : null;
    let unrealPnL = 'n/a';
    let pnlCls = '';
    if(currentPrice) {
      const diff = t.side === 'long' ? (currentPrice - t.entry_price) : (t.entry_price - currentPrice);
      unrealPnL = diff * t.size * t.leverage;
      pnlCls = unrealPnL > 0 ? 'green' : unrealPnL < 0 ? 'red' : '';
    }
    return `<tr><td>${t.coin}</td>
    <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
    <td>${fmtUsd(t.entry_price)}</td><td>${currentPrice ? fmtUsd(currentPrice) : 'n/a'}</td>
    <td>${fmt(t.size,4)}</td><td>${t.leverage}x</td>
    <td class="${pnlCls}">${unrealPnL !== 'n/a' ? fmtUsd(unrealPnL) : 'n/a'}</td>
    <td>${fmtUsd(t.stop_loss)}</td><td>${fmtUsd(t.take_profit)}</td>
    <td>${t.strategy_id||'n/a'}</td>
    <td><button class="btn btn-close" onclick="closeTrade(${t.id},'${t.coin}')" id="close-btn-${t.id}" title="Close this position at market price">Close</button></td></tr>`;
  }).join('') : '<tr><td colspan="11" class="empty-row">No open positions</td></tr>';
  const closeAllBtn = document.getElementById('btn-close-all');
  if(closeAllBtn) closeAllBtn.style.display = trades.length ? '' : 'none';
}

function renderStrategies(strats){
  document.getElementById('strategies-table').innerHTML = strats.slice(0,20).map((s,i)=>`
    <tr><td>${i+1}</td><td>${s.name}</td>
    <td><span class="badge badge-type">${s.strategy_type}</span></td>
    <td>${fmt(s.current_score,4)}</td>
    <td class="${pnlClass(s.total_pnl)}">${fmtUsd(s.total_pnl)}</td>
    <td>${fmt(s.win_rate*100,1)}%</td><td>${s.trade_count}</td></tr>`).join('');
}

function renderTraders(traders){
  document.getElementById('traders-table').innerHTML = traders.slice(0,30).map((t,i)=>`
    <tr><td>${i+1}</td><td><code>${shortAddr(t.address)}</code></td>
    <td>${fmtUsd(t.account_value)}</td>
    <td class="${pnlClass(t.total_pnl)}">${fmtUsd(t.total_pnl)}</td>
    <td>${fmt(t.win_rate*100,1)}%</td><td>${t.trade_count}</td></tr>`).join('');
}

function renderClosedTrades(trades){
  document.getElementById('closed-trades').innerHTML = trades.length ? trades.slice(0,20).map(t=>`
    <tr><td>${t.coin}</td>
    <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
    <td>${fmtUsd(t.entry_price)}</td><td>${fmtUsd(t.exit_price)}</td>
    <td class="${pnlClass(t.gross_pnl)}">${fmtUsd(t.gross_pnl)}</td>
    <td class="red">${fmtUsd(-(t.fees_paid||0))}</td>
    <td class="red">${fmtUsd(-(t.slippage_cost||0))}</td>
    <td class="${pnlClass(t.pnl)}">${fmtUsd(t.pnl)}</td>
    <td>${t.leverage}x</td><td>${t.closed_at?t.closed_at.slice(0,16):''}</td></tr>`).join('')
    : '<tr><td colspan="10" class="empty-row">No closed trades yet</td></tr>';
}

function renderTradeAnalytics(analytics, runtime){
  const summaryEl = document.getElementById('analytics-summary');
  const sidesEl = document.getElementById('analytics-sides');
  const sourcesEl = document.getElementById('analytics-sources');
  const coinSidesEl = document.getElementById('analytics-coin-sides');
  const liveEl = document.getElementById('analytics-live');
  if(!analytics){
    summaryEl.textContent = 'Execution analytics not available yet.';
    sidesEl.innerHTML = '<tr><td colspan="5" class="empty-row">No side analytics</td></tr>';
    sourcesEl.innerHTML = '<tr><td colspan="5" class="empty-row">No source analytics</td></tr>';
    coinSidesEl.innerHTML = '<tr><td colspan="5" class="empty-row">No coin-side analytics</td></tr>';
    liveEl.innerHTML = '<tr><td colspan="2" class="empty-row">No live analytics</td></tr>';
    return;
  }

  const summary = analytics.summary || {};
  const comp = analytics.short_vs_long || {};
  const runtimeLive = (runtime || {}).live_trader || {};
  const shortPolicy = runtimeLive.short_side_policy || ((runtime || {}).firewall || {}).short_side_policy || {};
  const policyStatus = String(shortPolicy.status || 'unknown').toUpperCase();
  const policyReason = shortPolicy.reason || 'No short-side policy data yet.';
  summaryEl.innerHTML =
    `<div>Closed trades: <strong>${summary.count || 0}</strong> | Net PnL: <strong class="${pnlClass(summary.net_pnl || 0)}">${fmtUsd(summary.net_pnl || 0)}</strong> | Win rate: <strong>${((summary.win_rate || 0) * 100).toFixed(1)}%</strong></div>` +
    `<div>Short vs long: <strong class="${pnlClass(comp.short_net_pnl || 0)}">short ${fmtUsd(comp.short_net_pnl || 0)}</strong> vs <strong class="${pnlClass(comp.long_net_pnl || 0)}">long ${fmtUsd(comp.long_net_pnl || 0)}</strong></div>` +
    `<div>Short guardrail: <strong>${policyStatus}</strong> — ${policyReason}</div>`;

  const sides = analytics.by_side || [];
  sidesEl.innerHTML = sides.length ? sides.map(row => `
    <tr>
      <td><span class="badge badge-${row.label}">${row.label.toUpperCase()}</span></td>
      <td>${row.count}</td>
      <td>${((row.win_rate || 0) * 100).toFixed(1)}%</td>
      <td class="red">${fmtUsd(-(row.fees || 0))}</td>
      <td class="${pnlClass(row.net_pnl || 0)}">${fmtUsd(row.net_pnl || 0)}</td>
    </tr>`).join('') : '<tr><td colspan="5" class="empty-row">No side analytics yet</td></tr>';

  const sources = analytics.by_source || [];
  sourcesEl.innerHTML = sources.length ? sources.map(row => `
    <tr>
      <td><code class="agent-key">${row.label}</code></td>
      <td>${row.count}</td>
      <td>${((row.win_rate || 0) * 100).toFixed(1)}%</td>
      <td class="red">${fmtUsd(-(row.fees || 0))}</td>
      <td class="${pnlClass(row.net_pnl || 0)}">${fmtUsd(row.net_pnl || 0)}</td>
    </tr>`).join('') : '<tr><td colspan="5" class="empty-row">No realized source results yet</td></tr>';

  const coinSides = analytics.by_coin_side || [];
  coinSidesEl.innerHTML = coinSides.length ? coinSides.map(row => `
    <tr>
      <td><code class="agent-key">${row.coin} ${String(row.side || '').toUpperCase()}</code></td>
      <td>${row.count}</td>
      <td>${((row.win_rate || 0) * 100).toFixed(1)}%</td>
      <td class="red">${fmtUsd(-(row.fees || 0))}</td>
      <td class="${pnlClass(row.net_pnl || 0)}">${fmtUsd(row.net_pnl || 0)}</td>
    </tr>`).join('') : '<tr><td colspan="5" class="empty-row">No coin-side analytics yet</td></tr>';

  const entryMetrics = runtimeLive.entry_metrics || {};
  const attempted = Number(runtimeLive.attempted_entry_signals || 0);
  const executed = Number(runtimeLive.executed_entry_signals || 0);
  const acceptRate = attempted > 0 ? `${((executed / attempted) * 100).toFixed(1)}%` : 'n/a';
  const liveRows = [
    ['Canary mode', runtimeLive.canary_mode ? 'ON' : 'OFF'],
    ['Attempted entries', attempted],
    ['Executed entries', executed],
    ['Execution hit rate', acceptRate],
    ['Min-order rejects', Number(runtimeLive.min_order_rejects_today || 0)],
    ['Min-order floor-ups', Number(runtimeLive.min_order_floorups_today || 0)],
    ['Top-tier floor-ups', Number(runtimeLive.min_order_top_tier_floorups_today || 0)],
    ['Same-side merge fills', Number(runtimeLive.min_order_same_side_merges_today || 0)],
    ['Approved but not executable', Number(runtimeLive.approved_but_not_executable_today || 0)],
    ['Canary headroom', runtimeLive.canary_headroom_ratio != null ? `${runtimeLive.canary_headroom_ratio}x` : 'n/a'],
    ['Crash-safe cap', runtimeLive.crash_safe_canary_order_usd != null ? fmtUsd(runtimeLive.crash_safe_canary_order_usd) : 'n/a'],
    ['Top reject', (runtime || {}).firewall?.top_rejection_reason || 'none'],
  ];
  Object.entries(entryMetrics)
    .filter(([key, value]) => String(key).startsWith('rejected_') && Number(value) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 5)
    .forEach(([key, value]) => liveRows.push([key.replaceAll('_', ' '), value]));

  liveEl.innerHTML = liveRows.map(([label, value]) => `
    <tr>
      <td>${label}</td>
      <td>${value}</td>
    </tr>`).join('');
}

function renderDriftAnalytics(drift, runtime){
  const summaryEl = document.getElementById('drift-summary');
  const sourceEl = document.getElementById('drift-source-table');
  const sideEl = document.getElementById('drift-side-table');
  if(!drift){
    summaryEl.textContent = 'Live vs paper drift is not available yet.';
    sourceEl.innerHTML = '<tr><td colspan="8" class="empty-row">No source drift data</td></tr>';
    sideEl.innerHTML = '<tr><td colspan="6" class="empty-row">No side drift data</td></tr>';
    return;
  }

  const summary = drift.summary || {};
  const copyStats = (runtime || {}).copy_trader || {};
  const copyGuardrail = copyStats.guardrail || {};
  summaryEl.innerHTML =
    `<div>Paper open: <strong>${summary.paper_open_positions || 0}</strong> | paper closed: <strong>${summary.paper_closed_trades || 0}</strong> | realized paper PnL: <strong class="${pnlClass(summary.paper_net_pnl || 0)}">${fmtUsd(summary.paper_net_pnl || 0)}</strong></div>` +
    `<div>Audit approvals: <strong>${summary.audit_approved || 0}</strong> | audit rejects: <strong>${summary.audit_rejected || 0}</strong> | live entries today: <strong>${summary.live_entries_today || 0}</strong> | approval gap: <strong>${summary.approval_gap || 0}</strong></div>` +
    `<div>Realized paper sources: <strong>${summary.paper_sources_realized || 0}</strong> | live-active sources: <strong>${summary.live_sources_active || 0}</strong></div>` +
    `<div>Copy-trade guardrail: <strong>${String(copyGuardrail.status || 'unknown').toUpperCase()}</strong> | ${copyGuardrail.reason || 'No copy-trader guardrail data yet.'}</div>`;

  const bySource = drift.by_source || [];
  sourceEl.innerHTML = bySource.length ? bySource.map(row => `
    <tr>
      <td><code class="agent-key">${row.label}</code></td>
      <td>${row.paper_open}</td>
      <td>${row.paper_closed}</td>
      <td class="${pnlClass(row.paper_net_pnl || 0)}">${fmtUsd(row.paper_net_pnl || 0)}</td>
      <td>${row.audit_approved}</td>
      <td>${row.audit_rejected}</td>
      <td>${row.live_entries_today}</td>
      <td>${row.top_reject_reason || 'n/a'}</td>
    </tr>`).join('') : '<tr><td colspan="8" class="empty-row">No source drift data yet</td></tr>';

  const bySide = drift.by_side || [];
  sideEl.innerHTML = bySide.length ? bySide.map(row => `
    <tr>
      <td><span class="badge badge-${row.label}">${String(row.label || 'unknown').toUpperCase()}</span></td>
      <td>${row.paper_open}</td>
      <td>${row.paper_closed}</td>
      <td class="${pnlClass(row.paper_net_pnl || 0)}">${fmtUsd(row.paper_net_pnl || 0)}</td>
      <td>${row.audit_approved}</td>
      <td>${row.audit_rejected}</td>
    </tr>`).join('') : '<tr><td colspan="6" class="empty-row">No side drift data yet</td></tr>';
}

function renderTypeChart(dist){
  const ctx = document.getElementById('type-chart').getContext('2d');
  const labels = dist.map(d=>d.strategy_type);
  const counts = dist.map(d=>d.count);
  const colors = ['#1f6f5f','#2f5b9f','#b9771f','#b54d3f','#7f5f9f','#4d8f84','#c28a35','#6c8ab8','#8a5b46','#4f7d54'];
  if(typeChart) typeChart.destroy();
  typeChart = new Chart(ctx, {
    type:'doughnut',
    data:{labels, datasets:[{data:counts, backgroundColor:colors.slice(0,labels.length), borderWidth:0, hoverOffset:6, cutout:'64%'}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{color:'#6f655b',font:{size:11},usePointStyle:true,boxWidth:8,padding:16}}}}
  });
}

function renderLogs(logs){
  document.getElementById('logs-table').innerHTML = logs.map(l=>`
    <tr><td>${l.timestamp?l.timestamp.slice(0,16):''}</td>
    <td><span class="badge badge-type">${l.cycle_type}</span></td>
    <td>${l.summary}</td><td>${l.traders_analyzed||0}</td><td>${l.strategies_found||0}</td></tr>`).join('');
}

function renderEvents(events){
  if(!events){
    document.getElementById('events-summary').textContent = 'Event scanner not initialized.';
    document.getElementById('events-upcoming').innerHTML = '<tr><td colspan="4" class="empty-row">No event data</td></tr>';
    document.getElementById('events-recent').innerHTML = '<tr><td colspan="4" class="empty-row">No recent official releases</td></tr>';
    document.getElementById('events-incidents').innerHTML = '<tr><td colspan="4" class="empty-row">No active crypto incidents</td></tr>';
    return;
  }

  const summary = events.summary || {};
  const nextLabel = summary.next_event_title
    ? `${summary.next_event_title} at ${(summary.next_event_time || '').slice(0,16).replace('T',' ')}`
    : 'No upcoming core events in the lookahead window.';
  document.getElementById('events-summary').innerHTML =
    `<div>Sources healthy: <strong>${summary.sources_ok || 0}/${summary.sources_total || 0}</strong></div>` +
    `<div>High impact next 24h: <strong>${summary.high_impact_next_24h || 0}</strong></div>` +
    `<div>Active incidents: <strong>${summary.incident_count || 0}</strong></div>` +
    `<div>Next event: <strong>${nextLabel}</strong></div>`;

  const upcoming = events.upcoming || [];
  document.getElementById('events-upcoming').innerHTML = upcoming.length ? upcoming.map(event=>`
    <tr>
      <td>${event.event_time ? event.event_time.slice(5,16).replace('T',' ') : 'n/a'}</td>
      <td>${event.source}</td>
      <td><span class="badge badge-type">${event.severity}</span></td>
      <td>${event.title}</td>
    </tr>`).join('') : '<tr><td colspan="4" class="empty-row">No upcoming core events</td></tr>';

  const recent = events.recent || [];
  document.getElementById('events-recent').innerHTML = recent.length ? recent.map(event=>`
    <tr>
      <td>${(event.published_time || event.event_time || '').slice(5,16).replace('T',' ') || 'n/a'}</td>
      <td>${event.source}</td>
      <td><span class="badge badge-type">${event.category}</span></td>
      <td>${event.title}</td>
    </tr>`).join('') : '<tr><td colspan="4" class="empty-row">No recent official releases</td></tr>';

  const incidents = events.incidents || [];
  document.getElementById('events-incidents').innerHTML = incidents.length ? incidents.map(event=>`
    <tr>
      <td>${event.source}</td>
      <td><span class="badge badge-type">${event.severity}</span></td>
      <td>${(event.assets || []).join(', ') || 'ALL'}</td>
      <td>${event.title}</td>
    </tr>`).join('') : '<tr><td colspan="4" class="empty-row">No active crypto incidents</td></tr>';
}

function renderCopyTrades(trades){
  document.getElementById('copy-trades').innerHTML = trades.length ? trades.slice(0,15).map(t=>{
    let meta = {};
    try { meta = JSON.parse(t.metadata || '{}'); } catch(e) {}
    const isGolden = meta.is_golden || meta.golden_wallet;
    const traderLabel = meta.source_trader
      ? `<code>${meta.source_trader}</code>${isGolden ? ' <span class="table-note" style="display:inline;margin-left:6px;color:#b9771f">gold</span>' : ''}`
      : 'n/a';
    const typeLabel = meta.type || 'copy_open';
    return `<tr><td>${t.coin}</td>
    <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
    <td>${fmtUsd(t.entry_price)}</td><td>${fmt(t.size,4)}</td>
    <td>${traderLabel}</td>
    <td><span class="badge badge-type">${t.status}</span> <span class="table-note" style="display:inline;margin-left:6px">${typeLabel}</span></td>
    <td class="${pnlClass(t.pnl)}">${t.pnl?fmtUsd(t.pnl):'n/a'}</td></tr>`;
  }).join('') : '<tr><td colspan="7" class="empty-row">No copy trades yet. Position cache is still warming up.</td></tr>';
}

let equityChart = null;
function renderEquityChart(closed){
  const ctx = document.getElementById('equity-chart').getContext('2d');
  if(!closed || closed.length === 0) return;
  const sorted = [...closed].reverse();
  let cumPnl = 0;
  const labels = [];
  const data = [];
  sorted.forEach((t,i)=>{
    cumPnl += (t.pnl || 0);
    labels.push(t.closed_at ? t.closed_at.slice(5,16) : `#${i+1}`);
    data.push(cumPnl);
  });
  const gradient = ctx.createLinearGradient(0, 0, 0, 320);
  gradient.addColorStop(0, 'rgba(31,111,95,0.22)');
  gradient.addColorStop(1, 'rgba(31,111,95,0.02)');
  if(equityChart) equityChart.destroy();
  equityChart = new Chart(ctx, {
    type:'line',
    data:{labels, datasets:[{label:'Cumulative PnL ($)',data,
      borderColor:'#1f6f5f',backgroundColor:gradient,fill:true,tension:0.34,pointRadius:0,
      pointHoverRadius:4,borderWidth:2.2}]},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{
        x:{ticks:{color:'#6f655b',maxTicksLimit:10},grid:{display:false}},
        y:{ticks:{color:'#6f655b'},grid:{color:'rgba(111,101,91,0.12)'}}
      },
      plugins:{legend:{display:false}}}
  });
}

async function closeTrade(tradeId, coin){
  if(!confirm(`Close ${coin} position at market price?`)) return;
  const btn = document.getElementById('close-btn-'+tradeId);
  if(btn){btn.disabled=true;btn.textContent='Closing...';}
  try {
    const resp = await fetch('/api/trade/close', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({trade_id: tradeId})
    });
    const d = await resp.json();
    if(d.status === 'ok'){
      if(btn){btn.textContent='Closed';btn.classList.remove('btn-close');btn.classList.add('btn-success');}
      setTimeout(refresh, 500);
    } else {
      alert('Close failed: '+(d.error||d.message||'Unknown error'));
      if(btn){btn.disabled=false;btn.textContent='Close';}
    }
  } catch(e){
    alert('Close error: '+e.message);
    if(btn){btn.disabled=false;btn.textContent='Close';}
  }
}

async function closeAllTrades(){
  const count = (window.currentOpenTrades||[]).length;
  if(!count) return alert('No open positions to close.');
  if(!confirm(`Close ALL ${count} open position(s) at market price?`)) return;
  const btn = document.getElementById('btn-close-all');
  if(btn){btn.disabled=true;btn.textContent='Closing all...';}
  try {
    const resp = await fetch('/api/trade/close-all', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({})
    });
    const d = await resp.json();
    if(d.status === 'ok'){
      if(btn){btn.textContent=`Closed ${d.closed||0} positions`;btn.classList.remove('btn-close-all');btn.classList.add('btn-success');}
      setTimeout(()=>{refresh();if(btn){btn.textContent='Close All';btn.classList.add('btn-close-all');btn.classList.remove('btn-success');btn.disabled=false;}}, 1500);
    } else {
      alert('Close all failed: '+(d.error||'Unknown error'));
      if(btn){btn.disabled=false;btn.textContent='Close All';}
    }
  } catch(e){
    alert('Close all error: '+e.message);
    if(btn){btn.disabled=false;btn.textContent='Close All';}
  }
}

async function resetPaperTrading(){
  if(!confirm('Reset paper trading history?\\n\\nThis will:\\n- Close & delete all paper trades\\n- Reset balance to initial ($10,000)\\n- Reset PnL, win rate, and trade count\\n\\nDiscovered traders, strategies, and research data will be KEPT.')) return;
  const btn = document.getElementById('btn-reset');
  if(btn){btn.disabled=true;btn.textContent='Resetting...';}
  try {
    const resp = await fetch('/api/paper/reset', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({})
    });
    const d = await resp.json();
    if(d.status === 'ok'){
      if(btn){btn.textContent=`Reset done (${d.open_deleted||0}+${d.closed_deleted||0} trades cleared)`;btn.classList.remove('btn-reset');btn.classList.add('btn-success');}
      setTimeout(()=>{refresh();if(btn){btn.textContent='Reset History';btn.classList.add('btn-reset');btn.classList.remove('btn-success');btn.disabled=false;}}, 2000);
    } else {
      alert('Reset failed: '+(d.error||'Unknown error'));
      if(btn){btn.disabled=false;btn.textContent='Reset History';}
    }
  } catch(e){
    alert('Reset error: '+e.message);
    if(btn){btn.disabled=false;btn.textContent='Reset History';}
  }
}

async function refresh(){
  try {
    const resp = await fetch('/api/data');
    const d = await resp.json();
    window.currentOpenTrades = d.open_trades;
    const wsStatus = ws && ws.readyState === WebSocket.OPEN ? '(live prices via WebSocket)' : '(WebSocket connecting...)';
    document.getElementById('update-time').textContent = new Date().toLocaleTimeString() + ' ' + wsStatus;
    renderCards(d);
    renderOpenTrades(d.open_trades);
    renderStrategies(d.strategies);
    renderTraders(d.traders);
    renderClosedTrades(d.closed_trades);
    renderTradeAnalytics(d.trade_analytics, d.runtime_health);
    renderDriftAnalytics(d.drift_analytics, d.runtime_health);
    renderCopyTrades(d.copy_trades || []);
    renderEquityChart(d.closed_trades);
    renderTypeChart(d.type_distribution);
    renderLogs(d.research_logs);
    renderEvents(d.events);
    if(d.v2) renderV2(d.v2);
    if(d.runtime_health) renderRuntimeHealth(d.runtime_health);
    if(d.v2 && d.v2.arena) renderArena(d.v2.arena);
  } catch(e) {
    console.error('Refresh error:', e);
  }
}

function renderV2(v2) {
  const fw = v2.firewall || {};
  const scorecard = (v2.source_scorecard && v2.source_scorecard.length) ? v2.source_scorecard : (v2.agent_scores || []);
  const shadow = v2.shadow_summary || {};
  const passRate = fw.total_signals > 0 ? (fw.passed / fw.total_signals * 100).toFixed(1) : 'n/a';
  const regimeCoins = Object.keys(v2.regime || {});
  const mainRegime = regimeCoins.length > 0 ? (v2.regime[regimeCoins[0]] || {}).regime || '?' : 'unknown';

  const cards = [
    {label:'Firewall Pass Rate', value: fw.total_signals > 0 ? passRate + '%' : 'n/a', cls: fw.total_signals > 0 ? (Number(passRate) >= 50 ? 'green' : 'yellow') : 'blue', sub: `${fw.passed||0}/${fw.total_signals||0} signals`},
    {label:'Signals Rejected', value: Math.max((fw.total_signals||0) - (fw.passed||0), 0), cls:'red'},
    {label:'Scored Sources', value: scorecard.length, cls:'blue'},
    {label:'Market Regime', value: mainRegime.replace('_',' ').toUpperCase(), cls: mainRegime.includes('up') ? 'green' : mainRegime.includes('down') ? 'red' : 'yellow'},
  ];
  document.getElementById('v2-cards').innerHTML = cards.map(c=>`
    <div class="card"><div class="label">${c.label}</div>
    <div class="value ${c.cls||''}">${c.value}</div>
    ${c.sub?`<div class="sub">${c.sub}</div>`:''}</div>`).join('');

  const reasons = ['confidence','risk','regime','conflict','cooldown','accuracy','drawdown','schema','source_policy','source_cap','event_risk','side_policy','exposure','funding']
    .filter(r => fw['rejected_'+r] > 0)
    .map(r => `<span class="reason-tag">${r}: ${fw['rejected_'+r]}</span>`)
    .join('');
  const shadowLine = shadow.total_trades
    ? `<div>Shadow ${shadow.period_days || 30}d: <strong class="${pnlClass(shadow.total_pnl || 0)}">${fmtUsd(shadow.total_pnl || 0)}</strong> across ${shadow.total_trades} trades. Best ${shadow.best_source || 'n/a'}, worst ${shadow.worst_source || 'n/a'}.</div>`
    : '<div class="table-note">Shadow attribution will populate as more trades close.</div>';
  const shortPolicy = fw.short_side_policy || {};
  const shortPolicyLine = shortPolicy.status
    ? `<div>Short guardrail: <strong>${String(shortPolicy.status).toUpperCase()}</strong> — ${shortPolicy.reason || 'n/a'}</div>`
    : '';
  document.getElementById('firewall-stats').innerHTML =
    `<div>Total: <strong>${fw.total_signals || 0}</strong> | Passed: <span class="green">${fw.passed || 0}</span> | Rejected: <span class="red">${Math.max((fw.total_signals || 0) - (fw.passed || 0), 0)}</span></div>` +
    `<div>${reasons || '<span class="green">No rejection pressure right now.</span>'}</div>` +
    shortPolicyLine +
    shadowLine;

  document.getElementById('agent-scores').innerHTML = scorecard.length ? scorecard.map(s=>{
    const trades = s.completed_trades ?? s.total_signals ?? 0;
    const accuracy = s.weighted_accuracy ?? s.accuracy ?? 0;
    const sharpe = s.sharpe;
    const weight = s.dynamic_weight ?? 0;
    const accCls = accuracy >= 0.5 ? 'green' : accuracy >= 0.3 ? 'yellow' : 'red';
    const wCls = weight >= 0.6 ? 'green' : weight >= 0.3 ? 'yellow' : 'red';
    const status = String(s.status || 'active').toUpperCase();
    const rankLabel = s.rank ? `Rank #${s.rank}` : status;
    return `<tr><td><code class="agent-key">${s.source_key}</code><span class="table-note">${rankLabel}</span></td><td>${trades}</td>
      <td class="${accCls}">${(accuracy*100).toFixed(1)}%</td>
      <td>${typeof sharpe === 'number' ? sharpe.toFixed(2) : 'n/a'}</td>
      <td class="${wCls}">${(weight*100).toFixed(0)}%</td>
      <td class="${pnlClass(s.total_pnl || 0)}">${fmtUsd(s.total_pnl || 0)}</td></tr>`;
  }).join('') : '<tr><td colspan="6" class="empty-row">No source scorecard yet. It will build as trades settle.</td></tr>';

  const regime = v2.regime || {};
  const coins = Object.entries(regime);
  document.getElementById('regime-grid').innerHTML = coins.length ? coins.map(([coin, r])=>{
    const cls = r.regime.includes('up') ? 'green' : r.regime.includes('down') ? 'red' : r.regime === 'volatile' ? 'yellow' : 'blue';
    return `<div class="card regime-card">
      <div class="label">${coin}</div>
      <div class="value ${cls}">${r.regime.replace('_',' ').toUpperCase()}</div>
      <div class="sub">ADX: ${r.adx} | ATR: ${(r.atr_pct*100).toFixed(1)}% | Conf: ${(r.confidence*100).toFixed(0)}%</div>
    </div>`;
  }).join('') : '<div class="empty-row" style="grid-column:1/-1">Regime data becomes useful after the first full cycle.</div>';
}

function renderRuntimeHealth(runtime) {
  const counts = runtime.subsystem_counts || {};
  const staleCount = (runtime.stale_subsystems || []).length;
  const atRiskCount = (runtime.at_risk_subsystems || []).length;
  const live = runtime.live_trader || {};
  const fw = runtime.firewall || {};
  const readiness = runtime.readiness || {};

  const cards = [
    {label:'Overall', value:(runtime.overall || 'unknown').toUpperCase(), cls: runtime.overall === 'ok' ? 'green' : 'red'},
    {label:'Trading Safe', value: runtime.all_trading_safe === true ? 'YES' : runtime.all_trading_safe === false ? 'NO' : 'N/A', cls: runtime.all_trading_safe ? 'green' : 'yellow'},
    {label:'Ready', value: readiness.ready ? 'YES' : 'NO', cls: readiness.ready ? 'green' : 'red'},
    {label:'Live Ready', value: readiness.live_ready ? 'YES' : 'NO', cls: readiness.live_ready ? 'green' : 'yellow'},
    {label:'At-Risk Subsystems', value: atRiskCount, cls: atRiskCount > 0 ? 'red' : 'green'},
    {label:'Stale Heartbeats', value: staleCount, cls: staleCount > 0 ? 'yellow' : 'green'},
    {label:'Failed', value: counts.FAILED || 0, cls: (counts.FAILED || 0) > 0 ? 'red' : 'green'},
    {label:'Degraded', value: counts.DEGRADED || 0, cls: (counts.DEGRADED || 0) > 0 ? 'yellow' : 'green'},
    {label:'Firewall Rejections', value: Math.max((fw.total_signals || 0) - (fw.passed || 0), 0), cls:'yellow'},
    {label:'Kill Switch', value: live.kill_switch_active ? 'ON' : 'OFF', cls: live.kill_switch_active ? 'red' : 'green'},
  ];
  document.getElementById('runtime-cards').innerHTML = cards.map(c=>`
    <div class="card"><div class="label">${c.label}</div>
    <div class="value ${c.cls||''}">${c.value}</div></div>`).join('');

  const atRiskList = (runtime.at_risk_subsystems || []).slice(0, 8).join(', ') || 'none';
  const staleList = (runtime.stale_subsystems || []).slice(0, 8).join(', ') || 'none';
  const killReason = live.kill_switch_reason ? `<span class="red">${live.kill_switch_reason}</span>` : 'none';
  const topReject = fw.top_rejection_reason || 'none';
  const sourceUsage = live.source_orders_today || {};
  const sourceCap = Number(live.max_orders_per_source_per_day || 0);
  const sourcePolicies = (live.source_policies || fw.source_policies || []).slice(0, 5);
  const sourceUsageSummary = Object.keys(sourceUsage).length
    ? Object.entries(sourceUsage)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([key, used]) => sourceCap > 0 ? `${key}: ${used}/${sourceCap}` : `${key}: ${used}`)
        .join(', ')
    : 'none';
  const sourcePolicySummary = sourcePolicies.length
    ? sourcePolicies
        .map(p => `${p.source_key}: ${String(p.status || 'unknown').toUpperCase()} (${Math.round((p.dynamic_weight || 0) * 100)}%)`)
        .join(', ')
    : 'none';
  const readinessReasons = (readiness.reasons || []).slice(0, 5).join(', ') || 'none';
  const attempted = Number(live.attempted_entry_signals || 0);
  const executed = Number(live.executed_entry_signals || 0);
  const hitRate = attempted > 0 ? `${((executed / attempted) * 100).toFixed(1)}%` : 'n/a';
  const shortPolicy = live.short_side_policy || fw.short_side_policy || {};
  const copyGuardrail = (runtime.copy_trader || {}).guardrail || {};
  const copyGuardrailLine = copyGuardrail.reason
    ? `<div>Copy-trade guardrail: <strong>${String(copyGuardrail.status || 'unknown').toUpperCase()}</strong> | ${copyGuardrail.reason}</div>`
    : '';
  document.getElementById('runtime-health-detail').innerHTML =
    `<div>At-risk: <strong>${atRiskList}</strong></div>` +
    `<div>Stale: <strong>${staleList}</strong></div>` +
    `<div>Readiness blockers: <strong>${readinessReasons}</strong></div>` +
    `<div>Firewall top reject: <strong>${topReject}</strong></div>` +
    `<div>Kill-switch reason: ${killReason}</div>` +
    `<div>Canary execution: <strong>${executed}/${attempted}</strong> (${hitRate}) | min-order rejects <strong>${live.min_order_rejects_today || 0}</strong> | floor-ups <strong>${live.min_order_floorups_today || 0}</strong></div>` +
    `<div>Execution rescue: top-tier floor-ups <strong>${live.min_order_top_tier_floorups_today || 0}</strong> | same-side merges <strong>${live.min_order_same_side_merges_today || 0}</strong> | approved but not executable <strong>${live.approved_but_not_executable_today || 0}</strong></div>` +
    `<div>Canary headroom: <strong>${live.canary_headroom_ratio != null ? live.canary_headroom_ratio + 'x' : 'n/a'}</strong> | crash-safe cap <strong>${live.crash_safe_canary_order_usd != null ? fmtUsd(live.crash_safe_canary_order_usd) : 'n/a'}</strong></div>` +
    `<div>Short guardrail: <strong>${String(shortPolicy.status || 'unknown').toUpperCase()}</strong> — ${shortPolicy.reason || 'n/a'}</div>` +
    copyGuardrailLine +
    `<div>Live source usage: <strong>${sourceUsageSummary}</strong></div>` +
    `<div>Source allocator: <strong>${sourcePolicySummary}</strong></div>`;
}

function renderArena(arena) {
  const s = arena.stats || {};
  const cards = [
    {label:'Arena Agents', value: s.active_agents||0, cls:'blue', sub:`${s.champions||0} champions`},
    {label:'Incubating', value: s.incubating||0, cls:'yellow'},
    {label:'Eliminated', value: s.eliminated||0, cls:'red'},
    {label:'Arena PnL', value: fmtUsd(s.total_arena_pnl||0), cls:pnlClass(s.total_arena_pnl)},
    {label:'Max ELO', value: Math.round(s.max_elo||0), cls:'green'},
    {label:'Rounds Played', value: s.total_rounds||0, cls:'blue'},
  ];
  document.getElementById('arena-cards').innerHTML = cards.map(c=>`
    <div class="card"><div class="label">${c.label}</div>
    <div class="value ${c.cls||''}">${c.value}</div>
    ${c.sub?`<div class="sub">${c.sub}</div>`:''}</div>`).join('');

  const lb = arena.leaderboard || [];
  document.getElementById('arena-leaderboard').innerHTML = lb.length ? lb.map(a=>{
    const statusCls = a.status==='champion'?'green':a.status==='active'?'blue':a.status==='probation'?'yellow':'red';
    return `<tr>
      <td>${a.rank}</td>
      <td><code>${a.name}</code></td>
      <td><span class="badge badge-type">${a.strategy}</span></td>
      <td><span class="${statusCls}">${a.status.toUpperCase()}</span></td>
      <td>${a.elo}</td>
      <td>${a.fitness.toFixed(3)}</td>
      <td>${fmtUsd(a.capital)}</td>
      <td class="${pnlClass(a.pnl)}">${fmtUsd(a.pnl)}</td>
      <td>${a.trades}</td>
      <td>${a.win_rate}%</td>
      <td>${a.sharpe}</td>
      <td>${a.generation > 0 ? 'Gen '+a.generation : 'Seed'}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="12" class="empty-row">Arena data will appear after the first competitive cycle.</td></tr>';

  const votes = (s.recent_votes || []).reverse();
  document.getElementById('consensus-log').innerHTML = votes.length ? votes.map(v=>{
    const cls = v.approved ? 'green' : 'red';
    return `<div class="consensus-item">
      <span class="${cls}" style="font-weight:700">${v.approved?'APPROVED':'REJECTED'}</span>
      <span style="margin-left:8px">${v.side.toUpperCase()} ${v.coin}</span>
      <span class="consensus-meta">${v.votes_for} for / ${v.votes_against} against | approval ${(v.approval_ratio*100).toFixed(0)}%</span>
    </div>`;
  }).join('') : '<div class="empty-row">No consensus votes yet</div>';
}

// Connect to WebSocket for live prices
connectWebSocket();

// Live update open trades from WebSocket prices (1-2 second interval)
liveUpdateInterval = setInterval(() => {
  if(window.currentOpenTrades && Object.keys(livePrices).length > 0) {
    renderOpenTrades(window.currentOpenTrades);
  }
}, 1500);

// Full data refresh every 30 seconds
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the unified dashboard (main + options flow)."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _handle_client_disconnect(self, exc: BaseException) -> None:
        logger.debug(
            "Dashboard client disconnected before response completed: %s %s (%s)",
            getattr(self, "command", "?"),
            getattr(self, "path", "?"),
            exc,
        )
        self.close_connection = True

    def _redirect(self, location: str, code: int = 303):
        self.send_response(code)
        self.send_header("Location", location)
        self._send_no_cache_headers()
        self.end_headers()

    def _send_no_cache_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        pending_auth_cookie = getattr(self, "_pending_auth_cookie", "")
        if pending_auth_cookie:
            secure_attr = "; Secure" if _is_hosted_dashboard_environment() else ""
            self.send_header(
                "Set-Cookie",
                f"{_AUTH_COOKIE_NAME}={pending_auth_cookie}; Path=/; HttpOnly; SameSite=Lax{secure_attr}",
            )

    def _cookie_auth_token(self) -> str:
        """Return the dashboard auth token from the Cookie header if present."""
        cookie_header = self.headers.get("Cookie", "")
        if not cookie_header:
            return ""
        try:
            cookie = SimpleCookie()
            cookie.load(cookie_header)
            morsel = cookie.get(_AUTH_COOKIE_NAME)
            return morsel.value if morsel else ""
        except Exception:
            return ""

    def _check_auth(self) -> bool:
        """Return True if request is authenticated or auth is not required."""
        auth_token = _dashboard_auth_token()
        parsed = urlparse(self.path)
        if parsed.path in _AUTH_EXEMPT_PATHS:
            return True  # Health probe - always open
        if not auth_token:
            if self.command == "POST" and parsed.path in _AUTH_REQUIRED_POST_PATHS:
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self._send_no_cache_headers()
                self.end_headers()
                self.wfile.write(
                    b'{"error": "dashboard_write_auth_not_configured", '
                    b'"hint": "Set DASHBOARD_AUTH_TOKEN before enabling dashboard write actions"}'
                )
                return False
            return True  # Read-only dashboard remains open when auth is not configured.
        # Check Authorization header: "Bearer <token>"
        # Constant-time comparison defends against remote timing attacks that
        # leak the token one byte at a time.
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and _secure_token_eq(
            auth_header[7:].strip(), auth_token
        ):
            self._pending_auth_cookie = auth_token
            return True
        if _secure_token_eq(self._cookie_auth_token(), auth_token):
            return True
        if self.command == "GET" and not parsed.path.startswith("/api/"):
            next_path = parsed.path or "/"
            if parsed.query:
                next_path = f"{next_path}?{parsed.query}"
            self._redirect(f"/login?next={quote(next_path, safe='/?=&')}")
            return False
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Bearer realm="dashboard"')
        self._send_no_cache_headers()
        self.end_headers()
        self.wfile.write(b'{"error": "unauthorized", "hint": "Set Authorization: Bearer <token> header"}')
        return False

    def _serve_login(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        next_path = params.get("next", ["/"])[0] or "/"
        error_message = ""
        if params.get("error", [""])[0] == "invalid":
            error_message = "Token was not accepted. Double-check the Railway variable and try again."
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self._send_no_cache_headers()
        self.end_headers()
        self.wfile.write(_login_html(error_message=error_message, next_path=next_path).encode())

    def _handle_login(self):
        auth_token = _dashboard_auth_token()
        content_len = int(self.headers.get("Content-Length", 0) or 0)
        raw_body = self.rfile.read(content_len) if content_len > 0 else b""
        form = parse_qs(raw_body.decode("utf-8", errors="ignore"))
        token = (form.get("token", [""])[0] or "").strip()
        next_path = (form.get("next", ["/"])[0] or "/").strip()
        if not next_path.startswith("/"):
            next_path = "/"
        if not auth_token:
            self._json_response({"error": "dashboard_auth_not_configured"}, code=503)
            return
        if not _secure_token_eq(token, auth_token):
            self._redirect(f"/login?error=invalid&next={quote(next_path, safe='/?=&')}")
            return
        self._pending_auth_cookie = auth_token
        self._redirect(next_path)

    def do_GET(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)

        if parsed.path == "/login":
            self._serve_login()

        elif parsed.path == "/" or parsed.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())

        elif parsed.path == "/options":
            # Serve the options flow dashboard
            self._serve_options_html()

        elif parsed.path == "/api/data":
            try:
                runtime_snapshot = _build_runtime_health_snapshot()
                data = get_dashboard_data(runtime_snapshot=runtime_snapshot)
                # Inject live V2 metrics
                if _firewall:
                    data["v2"]["firewall"] = _firewall.get_stats()
                if _regime_detector and _regime_detector._cache:
                    data["v2"]["regime"] = {
                        coin: state.to_dict()
                        for coin, state in _regime_detector._cache.items()
                    }
                if _arena:
                    data["v2"]["arena"] = {
                        "stats": _arena.get_stats(),
                        "leaderboard": _arena.get_leaderboard(top_n=15),
                    }

                # V2.5 metrics
                v25 = {}
                try:
                    if _kelly_sizer:
                        v25["kelly"] = _kelly_sizer.get_all_sizing_stats()
                except Exception:
                    pass
                try:
                    if _trade_memory:
                        v25["memory"] = _trade_memory.get_stats()
                except Exception:
                    pass
                try:
                    if _calibration:
                        _ece = _calibration.get_ece("global")
                        v25["calibration"] = {
                            "global_ece": _ece if _ece is not None else "N/A",
                            "quality": _calibration._quality_label(_ece),
                            "curve": _calibration.get_calibration_curve("global"),
                            "sources": _calibration.get_all_stats(),
                        }
                except Exception:
                    pass
                try:
                    if _llm_filter:
                        v25["llm_filter"] = _llm_filter.get_stats()
                except Exception:
                    pass
                try:
                    if _liquidation_strategy:
                        v25["lcrs"] = _liquidation_strategy.get_stats()
                except Exception:
                    pass
                try:
                    if _signal_processor:
                        v25["signal_processor"] = _signal_processor.get_stats()
                except Exception:
                    pass
                try:
                    if _arena_incubator:
                        v25["incubator"] = _arena_incubator.get_stats()
                except Exception:
                    pass
                try:
                    if _decision_engine:
                        v25["decision_engine"] = _decision_engine.get_stats()
                except Exception:
                    pass
                try:
                    if _multi_scanner:
                        v25["multi_scanner"] = _multi_scanner.get_stats()
                except Exception:
                    pass
                try:
                    from src.data.hyperliquid_client import get_api_stats
                    v25["api_manager"] = get_api_stats()
                except Exception:
                    pass
                try:
                    if _live_trader:
                        live_stats = _live_trader.get_stats()
                        # Refresh balance cache so dashboard never shows stale
                        # numbers when the fast cycle hasn't run recently.
                        if hasattr(_live_trader, "snapshot_balance"):
                            try:
                                live_stats["wallet_balance"] = (
                                    _live_trader.snapshot_balance(log=False)
                                )
                            except Exception:
                                pass
                        v25["live_trader"] = live_stats
                except Exception:
                    pass
                if v25:
                    data["v25"] = v25

                data["runtime_health"] = runtime_snapshot
                self._json_response(data)
            except Exception as e:
                self._json_response({"error": str(e)}, code=500)

        elif parsed.path == "/api/flow":
            # Options flow data endpoint
            self._serve_flow_data()

        elif parsed.path == "/api/health":
            self._json_response({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

        elif parsed.path in {"/api/ready", "/api/live_ready"}:
            class _Container:
                live_trader = _live_trader

            readiness = evaluate_readiness(
                container=_Container(),
                health_registry=_health_registry,
            )
            if parsed.path == "/api/live_ready":
                live_ready = bool(readiness.get("live_ready", False))
                payload = {
                    **readiness,
                    "ready": live_ready,
                    "status": "ready" if live_ready else "not_ready",
                }
                if not readiness.get("checks", {}).get("live_requested", False):
                    payload["reasons"] = list(payload.get("reasons", [])) + ["live_trading_disabled"]
                self._json_response(payload, code=200 if live_ready else 503)
            else:
                self._json_response(
                    readiness,
                    code=200 if bool(readiness.get("ready", False)) else 503,
                )

        elif parsed.path == "/api/health_report":
            # Full structured health report for Claude monitoring
            import os
            import json as _json
            report_path = "/data/health_report.json"
            if os.path.exists(report_path):
                with open(report_path) as f:
                    self._json_response(_json.load(f))
            else:
                self._json_response({"error": "no report yet"}, code=404)

        elif parsed.path == "/stress":
            self._serve_stress_html()

        elif parsed.path == "/api/stress":
            self._serve_stress_data()

        elif parsed.path == "/backtest":
            self._serve_backtest_html()

        elif parsed.path == "/api/backtest":
            self._serve_backtest_data()

        elif parsed.path == "/api/candle-backtest/cache":
            self._serve_cache_list()

        elif parsed.path == "/api/backtest/wallet":
            params = parse_qs(parsed.query)
            address = params.get("address", [None])[0]
            if address:
                self._serve_wallet_detail(address)
            else:
                self._json_response({"error": "address param required"}, code=400)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/auth/login":
            self._handle_login()
        elif parsed.path == "/api/order":
            self._handle_order()
        elif parsed.path == "/api/backtest/run":
            self._handle_backtest_run()
        elif parsed.path == "/api/paper/reset":
            self._handle_paper_reset()
        elif parsed.path == "/api/trade/close":
            self._handle_close_trade()
        elif parsed.path == "/api/trade/close-all":
            self._handle_close_all_trades()
        elif parsed.path == "/api/candle-backtest/run":
            self._handle_candle_backtest()
        elif parsed.path == "/api/candle-backtest/fetch":
            self._handle_candle_fetch()
        elif parsed.path == "/api/candle-backtest/cache/clear":
            self._handle_cache_clear()
        elif parsed.path == "/api/stress/run":
            self._handle_stress_run()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_paper_reset(self):
        """Reset all paper trades and balance."""
        try:
            from src.data.database import reset_paper_trades
            import config as _cfg
            # Read optional balance from request body
            balance = _cfg.PAPER_TRADING_INITIAL_BALANCE
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len > 0:
                body = json.loads(self.rfile.read(content_len))
                balance = float(body.get("balance", balance))
            result = reset_paper_trades(balance)
            self._json_response({"status": "ok", **result})
        except Exception as e:
            self._json_response({"error": str(e)}, code=500)

    def _handle_close_trade(self):
        """Close a single paper trade at current market price."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            trade_id = body.get("trade_id")
            if not trade_id:
                self._json_response({"error": "trade_id required"}, code=400)
                return

            result = _close_paper_trade_at_market(trade_id)
            self._json_response(result, code=200 if result.get("status") == "ok" else 400)
        except Exception as e:
            logger.error("Close trade error: %s", e)
            self._json_response({"error": str(e)}, code=500)

    def _handle_close_all_trades(self):
        """Close all open paper trades at current market prices."""
        try:
            from src.data import database as db
            open_trades = db.get_open_paper_trades()
            if not open_trades:
                self._json_response({"status": "ok", "closed": 0, "message": "No open trades"})
                return

            closed = 0
            errors = []
            for trade in open_trades:
                result = _close_paper_trade_at_market(trade["id"])
                if result.get("status") == "ok":
                    closed += 1
                else:
                    errors.append(f"{trade.get('coin','?')}: {result.get('error','unknown')}")

            self._json_response({
                "status": "ok",
                "closed": closed,
                "total": len(open_trades),
                "errors": errors if errors else None,
            })
        except Exception as e:
            logger.error("Close all trades error: %s", e)
            self._json_response({"error": str(e)}, code=500)

    def _serve_stress_html(self):
        """Serve the stress test dashboard HTML."""
        try:
            from src.ui.stress_dashboard import STRESS_HTML
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(STRESS_HTML.encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Stress dashboard error: {e}".encode())

    def _serve_stress_data(self):
        """Serve stress test data as JSON."""
        try:
            from src.ui.stress_dashboard import get_stress_dashboard_data
            data = get_stress_dashboard_data()
            self._json_response(data)
        except Exception as e:
            self._json_response({"error": str(e), "has_data": False})

    def _handle_stress_run(self):
        """Run a stress test on demand."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            scenarios = body.get("scenarios", None)
            use_seed = body.get("use_seed", False)

            from src.ui.stress_dashboard import run_stress_test
            result = run_stress_test(scenarios=scenarios, use_seed=use_seed)
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e), "has_data": False}, code=500)

    def _serve_cache_list(self):
        """Return cached candle data summary."""
        try:
            from src.backtest.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            self._json_response({
                "cached": fetcher.list_cached(),
                "stats": fetcher.get_cache_stats(),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, code=500)

    def _handle_candle_fetch(self):
        """Fetch candle data from Hyperliquid API (and cache it)."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

            from src.backtest.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            coin = body.get("coin", "BTC").upper()
            timeframe = body.get("timeframe", "1h")
            start = body.get("start")
            end = body.get("end")
            no_cache = body.get("no_cache", False)

            candles = fetcher.fetch_candles(coin, timeframe, start=start, end=end, use_cache=not no_cache)
            self._json_response({
                "status": "ok",
                "coin": coin,
                "timeframe": timeframe,
                "candles": len(candles),
                "start": candles[0].timestamp_ms if candles else None,
                "end": candles[-1].timestamp_ms if candles else None,
            })
        except Exception as e:
            self._json_response({"error": str(e)}, code=500)

    def _handle_candle_backtest(self):
        """Run a candle-based backtest and return results."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

            from src.backtest.data_fetcher import DataFetcher
            from src.backtest.candle_backtester import CandleBacktester, CandleBacktestConfig

            fetcher = DataFetcher()
            coin = body.get("coin", "BTC").upper()
            timeframe = body.get("timeframe", "1h")
            strategy = body.get("strategy", "momentum")
            start = body.get("start")
            end = body.get("end")

            # Build config from body params
            cfg_params = {}
            for key in ("position_size_pct", "max_leverage", "stop_loss_pct",
                         "take_profit_pct", "trailing_stop_pct", "fast_period",
                         "slow_period", "rsi_period", "rsi_overbought", "rsi_oversold",
                         "bb_period", "bb_std"):
                if key in body:
                    cfg_params[key] = float(body[key]) if "." in str(body[key]) else int(body[key])
            cfg_params["strategy"] = strategy
            cfg = CandleBacktestConfig(**cfg_params)

            # Fetch data
            candles = fetcher.fetch_candles(coin, timeframe, start=start, end=end)
            if not candles:
                self._json_response({"error": f"No candle data for {coin} {timeframe}"}, code=400)
                return

            # Run backtest
            bt = CandleBacktester(cfg)
            result = bt.run(candles, strategy=strategy, coin=coin)

            # Return result (limit equity curve to 500 points for JSON size)
            ec = result.equity_curve
            dd = result.drawdown_curve
            step = max(1, len(ec) // 500)
            self._json_response({
                "status": "ok",
                "summary": result.summary(),
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "total_pnl_pct": result.total_pnl_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "profit_factor": result.profit_factor,
                "avg_trade_pnl": result.avg_trade_pnl,
                "best_trade_pnl": result.best_trade_pnl,
                "worst_trade_pnl": result.worst_trade_pnl,
                "total_fees": result.total_fees,
                "duration_seconds": result.duration_seconds,
                "candles_per_second": result.candles_per_second,
                "equity_curve": ec[::step],
                "drawdown_curve": dd[::step],
                "trades": result.trades[:200],  # Limit for JSON size
            })
        except Exception as e:
            self._json_response({"error": str(e)}, code=500)

    def _handle_cache_clear(self):
        """Clear the candle data cache."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            from src.backtest.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            coin = body.get("coin")
            timeframe = body.get("timeframe")
            fetcher.clear_cache(coin=coin, timeframe=timeframe)
            self._json_response({"status": "ok"})
        except Exception as e:
            self._json_response({"error": str(e)}, code=500)

    def _json_response(self, data: dict, code: int = 200):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data, default=_safe_json).encode())
        except OSError as exc:
            if _is_client_disconnect(exc):
                self._handle_client_disconnect(exc)
                return
            raise

    def _serve_options_html(self):
        """Serve the options flow dashboard HTML."""
        try:
            from src.ui.options_dashboard import _get_dashboard_html
            html = _get_dashboard_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(html.encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Options dashboard error: {e}".encode())

    def _serve_flow_data(self):
        """Serve options flow data from the scanner."""
        global _options_scanner
        if _options_scanner:
            data = _options_scanner.get_dashboard_data()
        else:
            data = {"error": "Scanner not initialized", "convictions": [],
                    "heatmap": [], "flow_bars": [], "unusual_prints": [],
                    "spot_prices": {}, "summary": {"total_unusual": 0},
                    "timestamp": ""}
        self._json_response(data)

    def _handle_order(self):
        """Handle options order placement (needs Deribit API keys)."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            self._json_response({
                "status": "Order queued (Deribit API keys required for live execution)",
                "order": body,
            })
        except Exception as e:
            self._json_response({"status": f"Error: {e}"}, code=400)

    def _serve_backtest_html(self):
        """Serve the backtest dashboard HTML."""
        try:
            from src.ui.backtest_dashboard import BACKTEST_HTML
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(BACKTEST_HTML.encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Backtest dashboard error: {e}".encode())

    def _serve_backtest_data(self):
        """Serve backtest overview data as JSON."""
        try:
            from src.ui.backtest_dashboard import get_backtest_dashboard_data
            data = get_backtest_dashboard_data()
            self._json_response(data)
        except Exception as e:
            self._json_response({"error": str(e), "wallets": [],
                                 "golden_count": 0, "total_evaluated": 0,
                                 "timeframe_summaries": {}, "backtest_addresses": []})

    def _serve_wallet_detail(self, address: str):
        """Serve detailed backtest data for one wallet."""
        try:
            from src.ui.backtest_dashboard import get_wallet_detail
            data = get_wallet_detail(address)
            if data:
                self._json_response(data)
            else:
                self._json_response({"error": "Wallet not found"}, code=404)
        except Exception as e:
            self._json_response({"error": str(e)}, code=500)

    def _handle_backtest_run(self):
        """Trigger a golden wallet scan + backtest run."""
        try:
            import threading
            from src.discovery.golden_wallet import run_golden_scan, init_golden_tables
            from src.backtest.backtest_engine import run_all_backtests, save_backtest_result, init_backtest_tables

            def _run():
                try:
                    init_golden_tables()
                    init_backtest_tables()
                    run_golden_scan(max_wallets=30)
                    results = run_all_backtests()
                    for r in results:
                        save_backtest_result(r)
                except Exception as e:
                    import logging
                    logging.getLogger("backtest").error(f"Background scan error: {e}")

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._json_response({"status": "started", "message": "Golden scan + backtest running in background. Refresh in a few minutes."})
        except Exception as e:
            self._json_response({"error": str(e)}, code=500)


def set_options_scanner(scanner):
    """Set the options scanner reference for the /options and /api/flow routes."""
    global _options_scanner
    _options_scanner = scanner


def start_dashboard(port=None, options_scanner=None):
    """Start the unified dashboard server in a background thread.

    Serves both the main bot dashboard (/) and options flow (/options)
    on a single port for Railway compatibility.
    """
    if options_scanner:
        set_options_scanner(options_scanner)

    port = port or int(os.environ.get("PORT", 8080))
    host = _resolve_dashboard_host()
    _validate_dashboard_auth_configuration(host)
    base_url = _resolve_dashboard_base_url(host, port)
    server = HTTPServer((host, port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Dashboard bind address: http://%s:%d", host, port)
    logger.info("Dashboard running at %s", base_url)
    logger.info("  Main dashboard:    %s/", base_url)
    logger.info("  Options flow:      %s/options", base_url)
    logger.info("  Backtest:          %s/backtest", base_url)
    logger.info("  Stress test:       %s/stress", base_url)
    return server


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.data.database import init_db
    logger.warning(
        "Standalone dashboard mode only shows persisted DB state. "
        "Run main.py to attach live subsystem metrics."
    )
    init_db()
    port = int(os.environ.get("PORT", 8080))
    host = _resolve_dashboard_host()
    _validate_dashboard_auth_configuration(host)
    base_url = _resolve_dashboard_base_url(host, port)
    print(f"Starting dashboard on {host}:{port}...")
    print(f"Dashboard URL: {base_url}/")
    server = HTTPServer((host, port), DashboardHandler)
    server.serve_forever()
