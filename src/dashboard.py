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
from datetime import datetime
from typing import Dict
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Module-level options scanner reference (set by set_options_scanner)
_options_scanner = None


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

        # Copy trades (from metadata)
        copy_trades = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades WHERE metadata LIKE '%copy_trade%' ORDER BY opened_at DESC LIMIT 30"
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
            "timestamp": datetime.utcnow().isoformat(),
            "account": {
                "balance": balance,
                "total_pnl": account.get("total_pnl", 0),
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
        from src.decision_firewall import DecisionFirewall
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


# Module-level references for V2 components (set by set_v2_components)
_firewall = None
_regime_detector = None
_arena = None


def set_v2_components(firewall=None, regime_detector=None, arena=None,
                       kelly_sizer=None, trade_memory=None, calibration=None,
                       llm_filter=None, liquidation_strategy=None):
    """Set V2 + V2.5 component references for dashboard metrics."""
    global _firewall, _regime_detector, _arena
    global _kelly_sizer, _trade_memory, _calibration, _llm_filter, _liquidation_strategy
    _firewall = firewall
    _regime_detector = regime_detector
    _arena = arena
    _kelly_sizer = kelly_sizer
    _trade_memory = trade_memory
    _calibration = calibration
    _llm_filter = llm_filter
    _liquidation_strategy = liquidation_strategy


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
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center">
<div><h1>HYPERLIQUID RESEARCH BOT</h1>
<p class="subtitle">Live Simulation Dashboard &mdash; <span id="update-time">loading...</span></p></div>
<a href="/options" style="color:#00d4aa;text-decoration:none;border:1px solid #00d4aa;padding:8px 16px;border-radius:6px;font-size:0.85em;font-weight:600">OPTIONS FLOW &rarr;</a>
</div>

<div class="grid" id="stats-cards"></div>

<div class="flex-row">
<div class="section">
<h2>Open Positions</h2>
<table><thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Size</th><th>Lev</th><th>SL</th><th>TP</th><th>Strategy</th></tr></thead>
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
<table><thead><tr><th>Coin</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Lev</th><th>Closed</th></tr></thead>
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
let typeChart = null;

function fmt(n, d=2){ return n != null ? Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) : '—' }
function fmtUsd(n){ return n != null ? '$' + fmt(n) : '—' }
function pnlClass(n){ return n > 0 ? 'green' : n < 0 ? 'red' : '' }
function shortAddr(a){ return a ? a.slice(0,6)+'...'+a.slice(-4) : '—' }

function renderCards(d){
  const a = d.account;
  const cards = [
    {label:'Paper Balance', value:fmtUsd(a.balance), cls:pnlClass(a.total_pnl)},
    {label:'Total PnL', value:fmtUsd(a.total_pnl), cls:pnlClass(a.total_pnl), sub:`ROI: ${a.roi_pct>0?'+':''}${a.roi_pct}%`},
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
  document.getElementById('open-trades').innerHTML = trades.length ? trades.map(t=>`
    <tr><td>${t.coin}</td>
    <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
    <td>${fmtUsd(t.entry_price)}</td><td>${fmt(t.size,4)}</td>
    <td>${t.leverage}x</td><td>${fmtUsd(t.stop_loss)}</td><td>${fmtUsd(t.take_profit)}</td>
    <td>${t.strategy_id||'—'}</td></tr>`).join('') : '<tr><td colspan="8" style="color:#555">No open positions</td></tr>';
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
    <td class="${pnlClass(t.pnl)}">${fmtUsd(t.pnl)}</td>
    <td>${t.leverage}x</td><td>${t.closed_at?t.closed_at.slice(0,16):''}</td></tr>`).join('')
    : '<tr><td colspan="7" style="color:#555">No closed trades yet</td></tr>';
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
    return `<tr><td>${t.coin}</td>
    <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
    <td>${fmtUsd(t.entry_price)}</td><td>${fmt(t.size,4)}</td>
    <td><code>${meta.source_trader||'—'}</code></td>
    <td><span class="badge badge-type">${t.status}</span></td>
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

async function refresh(){
  try {
    const resp = await fetch('/api/data');
    const d = await resp.json();
    document.getElementById('update-time').textContent = new Date().toLocaleTimeString() + ' (auto-refreshes every 30s)';
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

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the unified dashboard (main + options flow)."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
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
                        v25["calibration"] = {
                            "global_ece": _calibration.get_ece("global"),
                            "quality": _calibration._quality_label(_calibration.get_ece("global")),
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
                if v25:
                    data["v25"] = v25

                self._json_response(data)
            except Exception as e:
                self._json_response({"error": str(e)}, code=500)

        elif parsed.path == "/api/flow":
            # Options flow data endpoint
            self._serve_flow_data()

        elif parsed.path == "/api/health":
            self._json_response({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/order":
            self._handle_order()
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data: dict, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=_safe_json).encode())

    def _serve_options_html(self):
        """Serve the options flow dashboard HTML."""
        try:
            from src.options_dashboard import _get_dashboard_html
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
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Dashboard running at http://0.0.0.0:{port}")
    print(f"  Main dashboard:    http://0.0.0.0:{port}/")
    print(f"  Options flow:      http://0.0.0.0:{port}/options")
    return server


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.database import init_db
    init_db()
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting dashboard on port {port}...")
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    server.serve_forever()
