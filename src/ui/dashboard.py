"""
Live Simulation Dashboard (V2)
Serves a unified web dashboard on port 8080 with:
  - Main bot dashboard (strategies, paper trades, trader data)
  - Options flow dashboard (convictions, heatmap, unusual prints)

Both dashboards served from the same port for Railway compatibility.
"""
import json
import os
import sys
import sqlite3
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Dict
from urllib.parse import urlparse, parse_qs

import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Module-level options scanner reference (set by set_options_scanner)
_options_scanner = None

# ─── Dashboard Authentication ─────────────────────────────────
# Set DASHBOARD_AUTH_TOKEN to require Bearer-token authentication on all
# API endpoints (except /api/health which is left open for Railway probes).
# When the token is not set, all endpoints are unauthenticated.
_DASHBOARD_AUTH_TOKEN: str = os.environ.get("DASHBOARD_AUTH_TOKEN", "").strip()

# Paths that are always open (health probes, Railway readiness checks)
_AUTH_EXEMPT_PATHS = {"/api/health"}


def _resolve_dashboard_host() -> str:
    """Resolve dashboard bind host with secure localhost default."""
    explicit = os.environ.get("DASHBOARD_HOST", "").strip()
    if explicit:
        return explicit
    public = os.environ.get("DASHBOARD_BIND_PUBLIC", "false").strip().lower() in {
        "1", "true", "yes"
    }
    if public and not _DASHBOARD_AUTH_TOKEN:
        logger.warning(
            "DASHBOARD_BIND_PUBLIC=true but DASHBOARD_AUTH_TOKEN is not set. "
            "The dashboard will be publicly accessible WITHOUT authentication. "
            "Set DASHBOARD_AUTH_TOKEN to secure it."
        )
    return "0.0.0.0" if public else "127.0.0.1"


def _get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_json(obj):
    """JSON serializer that handles numpy/sqlite types."""
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode()
    return str(obj)


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


def get_dashboard_data():
    """Collect all data needed for the dashboard."""
    conn = _get_db()
    try:
        # Account
        account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        account = dict(account) if account else {"balance": 0, "total_pnl": 0, "total_trades": 0, "winning_trades": 0}

        # Traders
        traders = [dict(r) for r in conn.execute(
            "SELECT address, account_value, total_pnl, roi_pct, win_rate, trade_count, last_updated "
            "FROM traders WHERE active = 1 ORDER BY total_pnl DESC LIMIT 100"
        ).fetchall()]

        # Strategies
        strategies = [dict(r) for r in conn.execute(
            "SELECT id, name, strategy_type, current_score, total_pnl, win_rate, trade_count, discovered_at "
            "FROM strategies WHERE active = 1 ORDER BY current_score DESC"
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
            "SELECT id, pnl, entry_price, exit_price, size, leverage, metadata FROM paper_trades WHERE status = 'closed'"
        ).fetchall()]
        total_fees = 0.0
        total_slippage = 0.0
        for trade in all_closed:
            costs = _parse_trade_costs(trade)
            total_fees += costs["fees_paid"]
            total_slippage += costs["slippage_cost"]
        recent_fees = sum(float(t.get("fees_paid", 0) or 0) for t in closed_trades)
        recent_slippage = sum(float(t.get("slippage_cost", 0) or 0) for t in closed_trades)

        # Copy trades (from metadata)
        copy_trades = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades WHERE metadata LIKE '%copy_trade%' OR metadata LIKE '%golden_wallet%' ORDER BY opened_at DESC LIMIT 30"
        ).fetchall()]

        # Strategy type distribution
        type_dist = [dict(r) for r in conn.execute(
            "SELECT strategy_type, COUNT(*) as count, AVG(current_score) as avg_score, "
            "SUM(total_pnl) as total_pnl FROM strategies WHERE active = 1 "
            "GROUP BY strategy_type ORDER BY count DESC"
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
            "v2": _get_v2_metrics(conn),
        }
    finally:
        conn.close()


def _get_v2_metrics(conn) -> Dict:
    """Collect V2 pipeline metrics (firewall, agent scores, regime)."""
    v2 = {
        "firewall": {},
        "agent_scores": [],
        "regime": {},
    }

    try:
        # Firewall stats — pulled from the global firewall instance if available
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

    return v2


# Module-level references for V2 + V2.5 + V3 components (set by set_v2_components)
_firewall = None
_regime_detector = None
_arena = None
_kelly_sizer = None
_trade_memory = None
_calibration = None
_llm_filter = None
_liquidation_strategy = None
_signal_processor = None
_arena_incubator = None
_decision_engine = None
_multi_scanner = None

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
    from src.data import database as db
    from src.data.hyperliquid_client import get_all_mids

    # Fetch the trade
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM paper_trades WHERE id = ? AND status = 'open'", (trade_id,)
        ).fetchone()
    finally:
        conn.close()

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

    # Close the paper trade in DB
    if not db.close_paper_trade(trade_id, current_price, pnl):
        return {"error": f"Failed to close trade {trade_id} in DB", "status": "error"}

    # Update paper account balance
    account = db.get_paper_account()
    if not account:
        # Account row missing — initialize with default balance
        db.init_paper_account(config.PAPER_TRADING_INITIAL_BALANCE)
        account = db.get_paper_account()
    if account:
        new_balance = float(account.get("balance", 0) or 0) + pnl
        total_pnl = float(account.get("total_pnl", 0) or 0) + pnl
        total_trades = int(account.get("total_trades", 0) or 0) + 1
        winning = int(account.get("winning_trades", 0) or 0) + (1 if pnl > 0 else 0)
        db.update_paper_account(new_balance, total_pnl, total_trades, winning)

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
                       kelly_sizer=None, trade_memory=None, calibration=None,
                       llm_filter=None, liquidation_strategy=None,
                       signal_processor=None, arena_incubator=None,
                       decision_engine=None, multi_scanner=None):
    """Set V2 + V2.5 + V3 + V4 component references for dashboard metrics."""
    global _firewall, _regime_detector, _arena  # noqa: PLW0603
    global _kelly_sizer, _trade_memory, _calibration, _llm_filter, _liquidation_strategy  # noqa
    global _signal_processor, _arena_incubator, _decision_engine  # noqa
    global _multi_scanner  # noqa
    _firewall = firewall
    _regime_detector = regime_detector
    _arena = arena
    _kelly_sizer = kelly_sizer
    _trade_memory = trade_memory
    _calibration = calibration
    _llm_filter = llm_filter
    _liquidation_strategy = liquidation_strategy
    _signal_processor = signal_processor
    _arena_incubator = arena_incubator
    _decision_engine = decision_engine
    _multi_scanner = multi_scanner


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hyperliquid Research Bot - Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e17;color:#e1e5ee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,monospace;padding:20px}
h1{color:#00d4aa;font-size:1.6em;margin-bottom:4px}
h2{color:#7b8ab8;font-size:1.1em;margin:20px 0 10px;text-transform:uppercase;letter-spacing:1px}
.subtitle{color:#555;font-size:0.85em;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}
.card{background:#141b2d;border:1px solid #1e2a45;border-radius:8px;padding:16px}
.card .label{color:#7b8ab8;font-size:0.75em;text-transform:uppercase;letter-spacing:1px}
.card .value{font-size:1.8em;font-weight:700;margin-top:4px}
.card .sub{color:#555;font-size:0.8em;margin-top:2px}
.green{color:#00d4aa}.red{color:#ff4757}.yellow{color:#ffa502}.blue{color:#3498db}
table{width:100%;border-collapse:collapse;margin-bottom:20px;font-size:0.85em}
th{background:#141b2d;color:#7b8ab8;text-align:left;padding:8px 12px;border-bottom:2px solid #1e2a45;font-weight:600;text-transform:uppercase;font-size:0.75em;letter-spacing:1px}
td{padding:8px 12px;border-bottom:1px solid #1a2235}
tr:hover{background:#141b2d}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.75em;font-weight:600}
.badge-long{background:#00d4aa22;color:#00d4aa}
.badge-short{background:#ff475722;color:#ff4757}
.badge-type{background:#3498db22;color:#3498db}
.section{background:#0d1320;border:1px solid #1e2a45;border-radius:8px;padding:16px;margin-bottom:16px}
.bar{height:6px;background:#1e2a45;border-radius:3px;overflow:hidden;margin-top:4px}
.bar-fill{height:100%;border-radius:3px;transition:width 0.5s}
.refresh-info{color:#555;font-size:0.75em;text-align:right}
.flex-row{display:flex;gap:16px;flex-wrap:wrap}
.flex-row>*{flex:1;min-width:300px}
canvas{width:100%!important;height:200px!important}
.btn{padding:4px 10px;border-radius:4px;border:none;cursor:pointer;font-size:0.75em;font-weight:600;transition:opacity 0.2s}
.btn:hover{opacity:0.8}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.btn-close{background:#ff475733;color:#ff4757;border:1px solid #ff4757}
.btn-close-all{background:#ff475722;color:#ff4757;border:1px solid #ff4757;padding:6px 14px;font-size:0.8em}
.btn-reset{background:#ffa50222;color:#ffa502;border:1px solid #ffa502;padding:6px 14px;font-size:0.8em}
.btn-success{background:#00d4aa22;color:#00d4aa;border:1px solid #00d4aa}
.action-bar{display:flex;gap:10px;align-items:center;margin-bottom:10px}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center">
<div><h1>HYPERLIQUID RESEARCH BOT</h1>
<p class="subtitle">Live Simulation Dashboard &mdash; <span id="update-time">loading...</span> <span id="ws-status" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff4757;margin-left:8px" title="WebSocket disconnected"></span></p></div>
<div style="display:flex;gap:10px">
<a href="/options" style="color:#00d4aa;text-decoration:none;border:1px solid #00d4aa;padding:8px 16px;border-radius:6px;font-size:0.85em;font-weight:600">OPTIONS FLOW</a>
<a href="/backtest" style="color:#94a3b8;text-decoration:none;border:1px solid #334155;padding:8px 16px;border-radius:6px;font-size:0.85em;font-weight:600">BACKTEST</a>
<a href="/stress" style="color:#eab308;text-decoration:none;border:1px solid #eab308;padding:8px 16px;border-radius:6px;font-size:0.85em;font-weight:600">STRESS TEST</a>
</div>
</div>

<div class="grid" id="stats-cards"></div>

<div class="flex-row">
<div class="section">
<div style="display:flex;justify-content:space-between;align-items:center">
<h2>Open Positions</h2>
<div class="action-bar">
<button class="btn btn-close-all" onclick="closeAllTrades()" id="btn-close-all" title="Close all open positions at market price">Close All</button>
<button class="btn btn-reset" onclick="resetPaperTrading()" id="btn-reset" title="Reset paper trading history (keeps discovered traders & strategies)">Reset History</button>
</div>
</div>
<table><thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Current</th><th>Size</th><th>Lev</th><th>Unreal. PnL</th><th>SL</th><th>TP</th><th>Strategy</th><th>Action</th></tr></thead>
<tbody id="open-trades"></tbody></table>
</div>
</div>

<div class="flex-row">
<div class="section">
<h2>Copy Trades (Mirroring Top Traders)</h2>
<table><thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Size</th><th>Source</th><th>Status</th><th>PnL</th></tr></thead>
<tbody id="copy-trades"></tbody></table>
</div>
</div>

<div class="flex-row">
<div class="section">
<h2>Strategy Rankings</h2>
<table><thead><tr><th>#</th><th>Strategy</th><th>Type</th><th>Score</th><th>PnL</th><th>Win Rate</th><th>Trades</th></tr></thead>
<tbody id="strategies-table"></tbody></table>
</div>
<div class="section">
<h2>Strategy Type Distribution</h2>
<canvas id="type-chart"></canvas>
</div>
<div class="section">
<h2>Equity Curve</h2>
<canvas id="equity-chart"></canvas>
</div>
</div>

<div class="flex-row">
<div class="section">
<h2>Top Tracked Traders</h2>
<table><thead><tr><th>#</th><th>Address</th><th>Account Value</th><th>PnL</th><th>Win Rate</th><th>Trades</th></tr></thead>
<tbody id="traders-table"></tbody></table>
</div>
</div>

<div class="flex-row">
<div class="section">
<h2>Closed Trades History</h2>
<table><thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Exit</th><th>Gross PnL</th><th>Fees</th><th>Slippage</th><th>Net PnL</th><th>Lev</th><th>Closed</th></tr></thead>
<tbody id="closed-trades"></tbody></table>
</div>
</div>

<div class="section">
<h2>Research Activity Log</h2>
<table><thead><tr><th>Time</th><th>Type</th><th>Summary</th><th>Traders</th><th>Strategies</th></tr></thead>
<tbody id="logs-table"></tbody></table>
</div>

<div class="section" style="border-top:2px solid #00d4aa;padding-top:16px;margin-top:24px">
<h2 style="color:#00d4aa">V2 Pipeline Metrics</h2>
<div class="grid" id="v2-cards"></div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<h2>Decision Firewall</h2>
<div id="firewall-stats" style="font-size:0.85em;color:#7b8ab8">Loading...</div>
</div>
<div>
<h2>Agent Scores (Signal Sources)</h2>
<table><thead><tr><th>Source</th><th>Signals</th><th>Accuracy</th><th>Sharpe</th><th>Weight</th><th>PnL</th></tr></thead>
<tbody id="agent-scores"></tbody></table>
</div>
</div>

<h2>Per-Coin Regime</h2>
<div id="regime-grid" class="grid"></div>
</div>

<div class="section" style="border-top:2px solid #ff6b35;padding-top:16px;margin-top:24px">
<h2 style="color:#ff6b35">Alpha Arena</h2>
<div class="grid" id="arena-cards"></div>

<h2>Arena Leaderboard</h2>
<table><thead><tr><th>#</th><th>Agent</th><th>Strategy</th><th>Status</th><th>ELO</th><th>Fitness</th><th>Capital</th><th>PnL</th><th>Trades</th><th>Win%</th><th>Sharpe</th><th>Gen</th></tr></thead>
<tbody id="arena-leaderboard"></tbody></table>

<h2>Recent Consensus Votes</h2>
<div id="consensus-log" style="font-size:0.85em;color:#7b8ab8"></div>
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
      document.getElementById('ws-status').style.background = '#00d4aa';
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
      document.getElementById('ws-status').style.background = '#ff4757';
      document.getElementById('ws-status').title = 'WebSocket error';
    };
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      document.getElementById('ws-status').style.background = '#ff4757';
      document.getElementById('ws-status').title = 'WebSocket disconnected';
      clearTimeout(wsReconnectTimer);
      wsReconnectTimer = setTimeout(connectWebSocket, 5000);
    };
  } catch(e) {
    console.error('WebSocket connection error:', e);
    document.getElementById('ws-status').style.background = '#ff4757';
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = setTimeout(connectWebSocket, 5000);
  }
}

let typeChart = null;
let liveUpdateInterval = null;

function fmt(n, d=2){ return n != null ? Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) : '—' }
function fmtUsd(n){ return n != null ? '$' + fmt(n) : '—' }
function pnlClass(n){ return n > 0 ? 'green' : n < 0 ? 'red' : '' }
function shortAddr(a){ return a ? a.slice(0,6)+'...'+a.slice(-4) : '—' }

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
    let unrealPnL = '—';
    let pnlCls = '';
    if(currentPrice) {
      const diff = t.side === 'long' ? (currentPrice - t.entry_price) : (t.entry_price - currentPrice);
      unrealPnL = diff * t.size * t.leverage;
      pnlCls = unrealPnL > 0 ? 'green' : unrealPnL < 0 ? 'red' : '';
    }
    return `<tr><td>${t.coin}</td>
    <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
    <td>${fmtUsd(t.entry_price)}</td><td>${currentPrice ? fmtUsd(currentPrice) : '—'}</td>
    <td>${fmt(t.size,4)}</td><td>${t.leverage}x</td>
    <td class="${pnlCls}">${unrealPnL !== '—' ? fmtUsd(unrealPnL) : '—'}</td>
    <td>${fmtUsd(t.stop_loss)}</td><td>${fmtUsd(t.take_profit)}</td>
    <td>${t.strategy_id||'—'}</td>
    <td><button class="btn btn-close" onclick="closeTrade(${t.id},'${t.coin}')" id="close-btn-${t.id}" title="Close this position at market price">Close</button></td></tr>`;
  }).join('') : '<tr><td colspan="11" style="color:#555">No open positions</td></tr>';
  // Show/hide close-all button based on open trades
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
    : '<tr><td colspan="10" style="color:#555">No closed trades yet</td></tr>';
}

function renderTypeChart(dist){
  const ctx = document.getElementById('type-chart').getContext('2d');
  const labels = dist.map(d=>d.strategy_type);
  const counts = dist.map(d=>d.count);
  const colors = ['#00d4aa','#3498db','#ffa502','#ff4757','#9b59b6','#1abc9c','#e74c3c','#f39c12','#2ecc71','#e67e22'];
  if(typeChart) typeChart.destroy();
  typeChart = new Chart(ctx, {
    type:'doughnut',
    data:{labels, datasets:[{data:counts, backgroundColor:colors.slice(0,labels.length), borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'right',labels:{color:'#7b8ab8',font:{size:11}}}}}
  });
}

function renderLogs(logs){
  document.getElementById('logs-table').innerHTML = logs.map(l=>`
    <tr><td>${l.timestamp?l.timestamp.slice(0,16):''}</td>
    <td><span class="badge badge-type">${l.cycle_type}</span></td>
    <td>${l.summary}</td><td>${l.traders_analyzed||0}</td><td>${l.strategies_found||0}</td></tr>`).join('');
}

function renderCopyTrades(trades){
  document.getElementById('copy-trades').innerHTML = trades.length ? trades.slice(0,15).map(t=>{
    let meta = {};
    try { meta = JSON.parse(t.metadata || '{}'); } catch(e) {}
    const isGolden = meta.is_golden || meta.golden_wallet;
    const traderLabel = meta.source_trader ? `<code>${meta.source_trader}</code>${isGolden ? ' <span style="color:#ffd700;font-size:0.8em" title="Golden Wallet">★</span>' : ''}` : '—';
    const typeLabel = meta.type || 'copy_open';
    return `<tr><td>${t.coin}</td>
    <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
    <td>${fmtUsd(t.entry_price)}</td><td>${fmt(t.size,4)}</td>
    <td>${traderLabel}</td>
    <td><span class="badge badge-type">${t.status}</span> <span style="color:#555;font-size:0.75em">${typeLabel}</span></td>
    <td class="${pnlClass(t.pnl)}">${t.pnl?fmtUsd(t.pnl):'—'}</td></tr>`;
  }).join('') : '<tr><td colspan="7" style="color:#555">No copy trades yet — warming up position cache...</td></tr>';
}

let equityChart = null;
function renderEquityChart(closed){
  const ctx = document.getElementById('equity-chart').getContext('2d');
  if(!closed || closed.length === 0) return;
  // Build cumulative PnL from closed trades (oldest first)
  const sorted = [...closed].reverse();
  let cumPnl = 0;
  const labels = [];
  const data = [];
  sorted.forEach((t,i)=>{
    cumPnl += (t.pnl || 0);
    labels.push(t.closed_at ? t.closed_at.slice(5,16) : `#${i+1}`);
    data.push(cumPnl);
  });
  if(equityChart) equityChart.destroy();
  equityChart = new Chart(ctx, {
    type:'line',
    data:{labels, datasets:[{label:'Cumulative PnL ($)',data,
      borderColor:'#00d4aa',backgroundColor:'rgba(0,212,170,0.1)',fill:true,tension:0.3,pointRadius:2}]},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{x:{ticks:{color:'#555',maxTicksLimit:10}},y:{ticks:{color:'#7b8ab8'}}},
      plugins:{legend:{labels:{color:'#7b8ab8'}}}}
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
    renderCopyTrades(d.copy_trades || []);
    renderEquityChart(d.closed_trades);
    renderTypeChart(d.type_distribution);
    renderLogs(d.research_logs);
    if(d.v2) renderV2(d.v2);
    if(d.v2 && d.v2.arena) renderArena(d.v2.arena);
  } catch(e) {
    console.error('Refresh error:', e);
  }
}

function renderV2(v2) {
  // V2 summary cards
  const fw = v2.firewall || {};
  const passRate = fw.total_signals > 0 ? (fw.passed / fw.total_signals * 100).toFixed(1) : '—';
  const regimeCoins = Object.keys(v2.regime || {});
  const mainRegime = regimeCoins.length > 0 ? (v2.regime[regimeCoins[0]] || {}).regime || '?' : 'unknown';

  const cards = [
    {label:'Firewall Pass Rate', value: passRate + '%', cls: Number(passRate) >= 50 ? 'green' : 'yellow', sub: `${fw.passed||0}/${fw.total_signals||0} signals`},
    {label:'Signals Rejected', value: (fw.total_signals||0) - (fw.passed||0), cls:'red'},
    {label:'Agent Sources', value: (v2.agent_scores||[]).length, cls:'blue'},
    {label:'Market Regime', value: mainRegime.replace('_',' ').toUpperCase(), cls: mainRegime.includes('up') ? 'green' : mainRegime.includes('down') ? 'red' : 'yellow'},
  ];
  document.getElementById('v2-cards').innerHTML = cards.map(c=>`
    <div class="card"><div class="label">${c.label}</div>
    <div class="value ${c.cls||''}">${c.value}</div>
    ${c.sub?`<div class="sub">${c.sub}</div>`:''}</div>`).join('');

  // Firewall breakdown
  if(fw.total_signals > 0) {
    const reasons = ['confidence','risk','regime','conflict','cooldown','accuracy','drawdown','schema']
      .filter(r => fw['rejected_'+r] > 0)
      .map(r => `<span style="margin-right:12px"><span class="badge" style="background:#ff475733;color:#ff4757">${r}</span> ${fw['rejected_'+r]}</span>`);
    document.getElementById('firewall-stats').innerHTML =
      `<div style="margin-bottom:8px">Total: ${fw.total_signals} | Passed: <span class="green">${fw.passed}</span> | Rejected: <span class="red">${fw.total_signals - fw.passed}</span></div>` +
      (reasons.length ? `<div>Rejection reasons: ${reasons.join('')}</div>` : '<div class="green">No rejections</div>');
  }

  // Agent scores table
  const scores = v2.agent_scores || [];
  document.getElementById('agent-scores').innerHTML = scores.length ? scores.map(s=>{
    const accCls = s.accuracy >= 0.5 ? 'green' : s.accuracy >= 0.3 ? 'yellow' : 'red';
    const wCls = s.dynamic_weight >= 0.6 ? 'green' : s.dynamic_weight >= 0.3 ? 'yellow' : 'red';
    return `<tr><td><code>${s.source_key}</code></td><td>${s.total_signals}</td>
      <td class="${accCls}">${(s.accuracy*100).toFixed(1)}%</td>
      <td>${s.sharpe ? s.sharpe.toFixed(2) : '—'}</td>
      <td class="${wCls}">${(s.dynamic_weight*100).toFixed(0)}%</td>
      <td class="${pnlClass(s.total_pnl)}">${fmtUsd(s.total_pnl)}</td></tr>`;
  }).join('') : '<tr><td colspan="6" style="color:#555">No agent data yet — scores build as trades complete</td></tr>';

  // Regime per coin
  const regime = v2.regime || {};
  const coins = Object.entries(regime);
  document.getElementById('regime-grid').innerHTML = coins.length ? coins.map(([coin, r])=>{
    const cls = r.regime.includes('up') ? 'green' : r.regime.includes('down') ? 'red' : r.regime === 'volatile' ? 'yellow' : 'blue';
    return `<div class="card" style="text-align:center">
      <div class="label">${coin}</div>
      <div class="value ${cls}" style="font-size:1em">${r.regime.replace('_',' ').toUpperCase()}</div>
      <div class="sub">ADX: ${r.adx} | ATR: ${(r.atr_pct*100).toFixed(1)}% | Conf: ${(r.confidence*100).toFixed(0)}%</div>
    </div>`;
  }).join('') : '<div style="color:#555;grid-column:1/-1">Regime data available after first full cycle</div>';
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

  // Leaderboard
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
  }).join('') : '<tr><td colspan="12" style="color:#555">Arena populating — agents compete after first full cycle</td></tr>';

  // Consensus votes
  const votes = (s.recent_votes || []).reverse();
  document.getElementById('consensus-log').innerHTML = votes.length ? votes.map(v=>{
    const cls = v.approved ? 'green' : 'red';
    return `<div style="margin-bottom:6px;padding:6px;background:#141b2d;border-radius:4px">
      <span class="${cls}" style="font-weight:bold">${v.approved?'APPROVED':'REJECTED'}</span>
      <span>${v.side.toUpperCase()} ${v.coin}</span> —
      <span style="color:#888">${v.votes_for} for / ${v.votes_against} against (${(v.approval_ratio*100).toFixed(0)}%)</span>
    </div>`;
  }).join('') : '<div style="color:#555">No consensus votes yet</div>';
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

    def _check_auth(self) -> bool:
        """Return True if request is authenticated or auth is not required."""
        if not _DASHBOARD_AUTH_TOKEN:
            return True  # Auth not configured — allow all
        parsed = urlparse(self.path)
        if parsed.path in _AUTH_EXEMPT_PATHS:
            return True  # Health probe — always open
        # Check Authorization header: "Bearer <token>"
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and auth_header[7:].strip() == _DASHBOARD_AUTH_TOKEN:
            return True
        # Check query parameter: ?token=<token>
        qs = parse_qs(parsed.query)
        if qs.get("token", [None])[0] == _DASHBOARD_AUTH_TOKEN:
            return True
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Bearer realm="dashboard"')
        self.end_headers()
        self.wfile.write(b'{"error": "unauthorized", "hint": "Set Authorization: Bearer <token> header"}')
        return False

    def do_GET(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())

        elif parsed.path == "/options":
            # Serve the options flow dashboard
            self._serve_options_html()

        elif parsed.path == "/api/data":
            try:
                data = get_dashboard_data()
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

                self._json_response(data)
            except Exception as e:
                self._json_response({"error": str(e)}, code=500)

        elif parsed.path == "/api/flow":
            # Options flow data endpoint
            self._serve_flow_data()

        elif parsed.path == "/api/health":
            self._json_response({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

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
        parsed = urlparse(self.path)
        if parsed.path == "/api/order":
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
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=_safe_json).encode())

    def _serve_options_html(self):
        """Serve the options flow dashboard HTML."""
        try:
            from src.ui.options_dashboard import _get_dashboard_html
            html = _get_dashboard_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
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
    server = HTTPServer((host, port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Dashboard running at http://%s:%d", host, port)
    logger.info("  Main dashboard:    http://%s:%d/", host, port)
    logger.info("  Options flow:      http://%s:%d/options", host, port)
    logger.info("  Backtest:          http://%s:%d/backtest", host, port)
    logger.info("  Stress test:       http://%s:%d/stress", host, port)
    return server


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.data.database import init_db
    init_db()
    port = int(os.environ.get("PORT", 8080))
    host = _resolve_dashboard_host()
    print(f"Starting dashboard on {host}:{port}...")
    server = HTTPServer((host, port), DashboardHandler)
    server.serve_forever()
