"""
Backtest Dashboard
==================
Separate dashboard served at /backtest on the same port 8080.
Shows golden wallet analysis, equity curves, multi-timeframe heatmaps,
per-coin breakdowns, and hour/weekday performance patterns.

Provides both HTML and JSON API endpoints:
  GET /backtest         → HTML dashboard
  GET /api/backtest     → JSON data for all wallets
  GET /api/backtest/wallet?address=0x... → detail for one wallet
  POST /api/backtest/run  → trigger a golden scan + backtest run
"""
import json
import sqlite3
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_backtest_dashboard_data() -> Dict:
    """Collect all backtest data for the dashboard."""
    conn = _get_db()
    try:
        # Golden wallets summary
        wallets = [dict(r) for r in conn.execute(
            "SELECT address, bot_score, total_fills, raw_pnl, penalised_pnl, "
            "max_drawdown_pct, penalised_max_drawdown_pct, sharpe_ratio, "
            "win_rate, trades_per_day, is_golden, coins_traded, best_coin, "
            "worst_coin, evaluated_at, connected_to_live "
            "FROM golden_wallets ORDER BY penalised_pnl DESC"
        ).fetchall()]

        # Parse JSON fields
        for w in wallets:
            try:
                w["coins_traded"] = json.loads(w.get("coins_traded", "[]"))
            except:
                w["coins_traded"] = []

        golden_count = len([w for w in wallets if w["is_golden"]])

        # Backtest results per timeframe (aggregated across wallets)
        timeframe_summaries = {}
        for tf in ["1d", "1h", "15m"]:
            rows = [dict(r) for r in conn.execute(
                "SELECT address, total_periods, active_periods, total_pnl, "
                "total_penalised_pnl, avg_period_pnl, std_period_pnl, "
                "best_period_pnl, worst_period_pnl, profitable_periods, "
                "profitable_pct, consistency_score "
                "FROM backtest_results WHERE timeframe = ? "
                "ORDER BY total_penalised_pnl DESC", (tf,)
            ).fetchall()]
            timeframe_summaries[tf] = rows

        # Per-wallet detailed periods (for selected wallet — client picks)
        # Just return the list of addresses that have backtest data
        bt_addresses = [r["address"] for r in conn.execute(
            "SELECT DISTINCT address FROM backtest_results"
        ).fetchall()]

        return {
            "wallets": wallets,
            "golden_count": golden_count,
            "total_evaluated": len(wallets),
            "timeframe_summaries": timeframe_summaries,
            "backtest_addresses": bt_addresses,
        }
    finally:
        conn.close()


def get_wallet_detail(address: str) -> Optional[Dict]:
    """Get full backtest detail for one wallet."""
    conn = _get_db()
    try:
        # Wallet info
        wallet = conn.execute(
            "SELECT * FROM golden_wallets WHERE address = ?", (address,)
        ).fetchone()
        if not wallet:
            return None
        wallet = dict(wallet)

        # Parse curves
        for field in ["raw_equity_curve", "penalised_equity_curve", "equity_timestamps", "coins_traded"]:
            try:
                wallet[field] = json.loads(wallet.get(field, "[]"))
            except:
                wallet[field] = []

        # Timeframe results with period detail
        timeframes = {}
        for tf in ["1d", "1h", "15m"]:
            row = conn.execute(
                "SELECT * FROM backtest_results WHERE address = ? AND timeframe = ?",
                (address, tf)
            ).fetchone()
            if row:
                row = dict(row)
                try:
                    row["periods"] = json.loads(row.get("periods_json", "[]"))
                except:
                    row["periods"] = []
                del row["periods_json"]
                timeframes[tf] = row

        # Coin performance
        coins = [dict(r) for r in conn.execute(
            "SELECT * FROM backtest_coin_perf WHERE address = ? ORDER BY pen_pnl DESC",
            (address,)
        ).fetchall()]

        # Hourly PnL
        hourly = {r["bucket"]: r["pnl"] for r in conn.execute(
            "SELECT bucket, pnl FROM backtest_time_analysis "
            "WHERE address = ? AND analysis_type = 'hour'", (address,)
        ).fetchall()}

        # Weekday PnL
        weekday = {r["bucket"]: r["pnl"] for r in conn.execute(
            "SELECT bucket, pnl FROM backtest_time_analysis "
            "WHERE address = ? AND analysis_type = 'weekday'", (address,)
        ).fetchall()}

        return {
            "wallet": wallet,
            "timeframes": timeframes,
            "coin_performance": coins,
            "hourly_pnl": hourly,
            "weekday_pnl": weekday,
        }
    finally:
        conn.close()


# ─── Dashboard HTML ──────────────────────────────────────────────

BACKTEST_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden Wallet Backtest Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --gold: #d29922;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  h2 { font-size: 18px; margin: 24px 0 12px; color: var(--accent); }
  h3 { font-size: 15px; margin: 16px 0 8px; color: var(--dim); }
  .subtitle { color: var(--dim); margin-bottom: 20px; }
  .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-gold { background: rgba(210,153,34,.15); color: var(--gold); }
  .badge-green { background: rgba(63,185,80,.15); color: var(--green); }
  .badge-red { background: rgba(248,81,73,.15); color: var(--red); }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .stat-label { font-size: 11px; text-transform: uppercase; color: var(--dim); letter-spacing: .5px; }
  .stat-value { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .stat-value.positive { color: var(--green); }
  .stat-value.negative { color: var(--red); }
  .stat-value.gold { color: var(--gold); }
  table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; }
  th { background: #1c2128; padding: 10px 12px; text-align: left; font-size: 11px; text-transform: uppercase; color: var(--dim); letter-spacing: .5px; }
  td { padding: 10px 12px; border-top: 1px solid var(--border); }
  tr:hover td { background: rgba(88,166,255,.04); }
  .clickable { cursor: pointer; }
  .clickable:hover td { background: rgba(88,166,255,.08); }
  .pnl-pos { color: var(--green); font-weight: 600; }
  .pnl-neg { color: var(--red); font-weight: 600; }
  .chart-container { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 12px 0; }
  canvas { width: 100% !important; height: 300px !important; }
  .heatmap { display: grid; gap: 2px; margin: 8px 0; }
  .heatmap-cell { padding: 6px 4px; text-align: center; font-size: 11px; border-radius: 4px; min-width: 40px; }
  .tabs { display: flex; gap: 4px; margin: 16px 0; }
  .tab { padding: 8px 16px; border-radius: 6px; cursor: pointer; background: var(--card); border: 1px solid var(--border); color: var(--dim); font-size: 13px; }
  .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .detail-panel { display: none; }
  .detail-panel.active { display: block; }
  .back-btn { background: var(--card); border: 1px solid var(--border); color: var(--accent); padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .back-btn:hover { background: var(--border); }
  .run-btn { background: var(--green); color: #000; border: none; padding: 8px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }
  .run-btn:hover { opacity: .85; }
  .run-btn:disabled { opacity: .4; cursor: not-allowed; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #fff3; border-top-color: #fff; border-radius: 50%; animation: spin .8s linear infinite; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .flex-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .flex-row > * { flex: 1; min-width: 300px; }
  #status-msg { color: var(--dim); font-style: italic; margin: 8px 0; }
</style>
</head>
<body>
<div class="container">
  <div class="top-bar">
    <div>
      <h1>Golden Wallet Backtest</h1>
      <div class="subtitle">Multi-timeframe analysis with +100ms delay &amp; -0.045% fee penalty</div>
    </div>
    <div>
      <button class="run-btn" id="runBtn" onclick="runScan()">Run Golden Scan</button>
      <a href="/" style="margin-left:12px;color:var(--accent);text-decoration:none;">&larr; Main Dashboard</a>
    </div>
  </div>
  <div id="status-msg"></div>

  <!-- Overview page -->
  <div id="overview-page">
    <div class="stats-grid" id="summary-stats"></div>
    <h2>Evaluated Wallets</h2>
    <table id="wallets-table">
      <thead><tr>
        <th>Address</th><th>Status</th><th>Raw PnL</th><th>Penalised PnL</th>
        <th>Max DD</th><th>Sharpe</th><th>Win Rate</th><th>Fills</th><th>TPD</th>
      </tr></thead>
      <tbody></tbody>
    </table>

    <h2>Timeframe Comparison</h2>
    <div class="tabs" id="tf-tabs">
      <div class="tab active" data-tf="1d">Daily</div>
      <div class="tab" data-tf="1h">Hourly</div>
      <div class="tab" data-tf="15m">15-Min</div>
    </div>
    <table id="tf-table">
      <thead><tr>
        <th>Address</th><th>Active Periods</th><th>Pen. PnL</th><th>Avg/Period</th>
        <th>Best Period</th><th>Worst Period</th><th>Win %</th><th>Consistency</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <!-- Wallet detail page -->
  <div id="detail-page" class="detail-panel">
    <button class="back-btn" onclick="showOverview()">&larr; Back to Overview</button>
    <h2 id="detail-title"></h2>
    <div class="stats-grid" id="detail-stats"></div>

    <h3>Equity Curve (Raw vs Penalised)</h3>
    <div class="chart-container"><canvas id="equityChart"></canvas></div>

    <div class="flex-row">
      <div>
        <h3>PnL by Hour of Day (UTC)</h3>
        <div class="chart-container"><canvas id="hourlyChart"></canvas></div>
      </div>
      <div>
        <h3>PnL by Day of Week</h3>
        <div class="chart-container"><canvas id="weekdayChart"></canvas></div>
      </div>
    </div>

    <h3>Per-Coin Performance</h3>
    <table id="coin-table">
      <thead><tr><th>Coin</th><th>Fills</th><th>Raw PnL</th><th>Pen. PnL</th><th>Volume</th><th>Win Rate</th></tr></thead>
      <tbody></tbody>
    </table>

    <h3>Period Heatmap</h3>
    <div class="tabs" id="detail-tf-tabs">
      <div class="tab active" data-tf="1d">Daily</div>
      <div class="tab" data-tf="1h">Hourly</div>
      <div class="tab" data-tf="15m">15-Min</div>
    </div>
    <div id="heatmap-container" class="chart-container"></div>
    <table id="periods-table">
      <thead><tr><th>Period</th><th>Fills</th><th>PnL</th><th>Raw PnL</th><th>Win Rate</th><th>Coins</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
let DATA = null;
let DETAIL = null;
let selectedTf = '1d';
let detailTf = '1d';
let charts = {};

async function load() {
  try {
    const r = await fetch('/api/backtest');
    DATA = await r.json();
    renderOverview();
  } catch(e) {
    document.getElementById('status-msg').textContent = 'No backtest data yet. Click "Run Golden Scan" to start.';
  }
}

function pnlClass(v) { return v >= 0 ? 'pnl-pos' : 'pnl-neg'; }
function fmt(v, d=0) { return v >= 0 ? '+$' + v.toFixed(d).replace(/\\B(?=(\\d{3})+(?!\\d))/g,',') : '-$' + Math.abs(v).toFixed(d).replace(/\\B(?=(\\d{3})+(?!\\d))/g,','); }
function shortAddr(a) { return a.slice(0,6) + '...' + a.slice(-4); }

function renderOverview() {
  if (!DATA) return;
  const s = document.getElementById('summary-stats');
  s.innerHTML = `
    <div class="stat-card"><div class="stat-label">Wallets Evaluated</div><div class="stat-value">${DATA.total_evaluated}</div></div>
    <div class="stat-card"><div class="stat-label">Golden Wallets</div><div class="stat-value gold">${DATA.golden_count}</div></div>
    <div class="stat-card"><div class="stat-label">Best Penalised PnL</div><div class="stat-value positive">${DATA.wallets.length ? fmt(DATA.wallets[0].penalised_pnl) : 'N/A'}</div></div>
    <div class="stat-card"><div class="stat-label">Avg Sharpe (Golden)</div><div class="stat-value">${avgSharpe()}</div></div>
  `;

  // Wallets table
  const tbody = document.querySelector('#wallets-table tbody');
  tbody.innerHTML = DATA.wallets.map(w => `
    <tr class="clickable" onclick="showDetail('${w.address}')">
      <td><code>${shortAddr(w.address)}</code></td>
      <td>${w.is_golden ? '<span class=\\"badge badge-gold\\">GOLDEN</span>' : '<span class=\\"badge badge-red\\">Not Golden</span>'}</td>
      <td class="${pnlClass(w.raw_pnl)}">${fmt(w.raw_pnl)}</td>
      <td class="${pnlClass(w.penalised_pnl)}">${fmt(w.penalised_pnl)}</td>
      <td>${w.penalised_max_drawdown_pct.toFixed(1)}%</td>
      <td>${w.sharpe_ratio.toFixed(2)}</td>
      <td>${w.win_rate.toFixed(0)}%</td>
      <td>${w.total_fills}</td>
      <td>${w.trades_per_day.toFixed(1)}</td>
    </tr>
  `).join('');

  renderTfTable();
}

function avgSharpe() {
  if (!DATA) return '0';
  const golden = DATA.wallets.filter(w => w.is_golden);
  if (!golden.length) return '0';
  return (golden.reduce((s,w) => s + w.sharpe_ratio, 0) / golden.length).toFixed(2);
}

function renderTfTable() {
  const rows = DATA.timeframe_summaries[selectedTf] || [];
  const tbody = document.querySelector('#tf-table tbody');
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><code>${shortAddr(r.address)}</code></td>
      <td>${r.active_periods}</td>
      <td class="${pnlClass(r.total_penalised_pnl)}">${fmt(r.total_penalised_pnl)}</td>
      <td class="${pnlClass(r.avg_period_pnl)}">${fmt(r.avg_period_pnl,2)}</td>
      <td class="pnl-pos">${fmt(r.best_period_pnl)}</td>
      <td class="pnl-neg">${fmt(r.worst_period_pnl)}</td>
      <td>${r.profitable_pct.toFixed(0)}%</td>
      <td>${r.consistency_score.toFixed(0)}%</td>
    </tr>
  `).join('');
}

// Tab switching
document.getElementById('tf-tabs').addEventListener('click', e => {
  if (!e.target.dataset.tf) return;
  selectedTf = e.target.dataset.tf;
  document.querySelectorAll('#tf-tabs .tab').forEach(t => t.classList.remove('active'));
  e.target.classList.add('active');
  renderTfTable();
});

// Detail view
async function showDetail(address) {
  try {
    const r = await fetch('/api/backtest/wallet?address=' + address);
    DETAIL = await r.json();
    renderDetail();
    document.getElementById('overview-page').style.display = 'none';
    document.getElementById('detail-page').classList.add('active');
  } catch(e) { console.error(e); }
}

function showOverview() {
  document.getElementById('detail-page').classList.remove('active');
  document.getElementById('overview-page').style.display = 'block';
}

function renderDetail() {
  if (!DETAIL) return;
  const w = DETAIL.wallet;
  document.getElementById('detail-title').textContent =
    (w.is_golden ? '★ ' : '') + shortAddr(w.address) + (w.is_golden ? ' (GOLDEN)' : '');

  const s = document.getElementById('detail-stats');
  s.innerHTML = `
    <div class="stat-card"><div class="stat-label">Raw PnL</div><div class="stat-value ${w.raw_pnl>=0?'positive':'negative'}">${fmt(w.raw_pnl)}</div></div>
    <div class="stat-card"><div class="stat-label">Penalised PnL</div><div class="stat-value ${w.penalised_pnl>=0?'positive':'negative'}">${fmt(w.penalised_pnl)}</div></div>
    <div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value negative">${w.penalised_max_drawdown_pct.toFixed(1)}%</div></div>
    <div class="stat-card"><div class="stat-label">Sharpe</div><div class="stat-value">${w.sharpe_ratio.toFixed(2)}</div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">${w.win_rate.toFixed(0)}%</div></div>
    <div class="stat-card"><div class="stat-label">Trades/Day</div><div class="stat-value">${w.trades_per_day.toFixed(1)}</div></div>
    <div class="stat-card"><div class="stat-label">Best Coin</div><div class="stat-value gold">${w.best_coin || 'N/A'}</div></div>
    <div class="stat-card"><div class="stat-label">Fills (90d)</div><div class="stat-value">${w.total_fills}</div></div>
  `;

  // Equity chart
  renderEquityChart(w);
  // Hourly + weekday charts
  renderHourlyChart(DETAIL.hourly_pnl);
  renderWeekdayChart(DETAIL.weekday_pnl);
  // Coin table
  renderCoinTable(DETAIL.coin_performance);
  // Period heatmap
  renderPeriods(detailTf);
}

function renderEquityChart(w) {
  const ctx = document.getElementById('equityChart').getContext('2d');
  if (charts.equity) charts.equity.destroy();

  const raw = w.raw_equity_curve || [];
  const pen = w.penalised_equity_curve || [];
  const ts = w.equity_timestamps || [];
  const labels = ts.map(t => new Date(t).toLocaleDateString());

  charts.equity = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'Raw Equity', data: raw, borderColor: '#58a6ff', borderWidth: 1.5, pointRadius: 0, fill: false },
        { label: 'Penalised Equity', data: pen, borderColor: '#d29922', borderWidth: 2, pointRadius: 0, fill: false },
        { label: 'Baseline ($10k)', data: raw.map(()=>10000), borderColor: '#30363d', borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 12 }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e', callback: v => '$'+v.toLocaleString() }, grid: { color: '#21262d' } },
      }
    }
  });
}

function renderHourlyChart(data) {
  const ctx = document.getElementById('hourlyChart').getContext('2d');
  if (charts.hourly) charts.hourly.destroy();
  const labels = Array.from({length:24}, (_,i) => i+'h');
  const values = labels.map((_,i) => data[i] || 0);
  const colors = values.map(v => v >= 0 ? 'rgba(63,185,80,.7)' : 'rgba(248,81,73,.7)');

  charts.hourly = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e', callback: v => '$'+v }, grid: { color: '#21262d' } },
      }
    }
  });
}

function renderWeekdayChart(data) {
  const ctx = document.getElementById('weekdayChart').getContext('2d');
  if (charts.weekday) charts.weekday.destroy();
  const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const values = days.map((_,i) => data[i] || 0);
  const colors = values.map(v => v >= 0 ? 'rgba(63,185,80,.7)' : 'rgba(248,81,73,.7)');

  charts.weekday = new Chart(ctx, {
    type: 'bar',
    data: { labels: days, datasets: [{ data: values, backgroundColor: colors }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e', callback: v => '$'+v }, grid: { color: '#21262d' } },
      }
    }
  });
}

function renderCoinTable(coins) {
  const tbody = document.querySelector('#coin-table tbody');
  tbody.innerHTML = coins.map(c => `
    <tr>
      <td><strong>${c.coin}</strong></td>
      <td>${c.fills}</td>
      <td class="${pnlClass(c.raw_pnl)}">${fmt(c.raw_pnl)}</td>
      <td class="${pnlClass(c.pen_pnl)}">${fmt(c.pen_pnl)}</td>
      <td>$${(c.volume/1000).toFixed(0)}k</td>
      <td>${c.win_rate.toFixed(0)}%</td>
    </tr>
  `).join('');
}

function renderPeriods(tf) {
  const tfData = DETAIL.timeframes[tf];
  if (!tfData || !tfData.periods) {
    document.querySelector('#periods-table tbody').innerHTML = '<tr><td colspan="6">No data</td></tr>';
    return;
  }
  const periods = tfData.periods;

  // Heatmap
  const container = document.getElementById('heatmap-container');
  const maxAbs = Math.max(...periods.map(p => Math.abs(p.pnl)), 1);
  container.innerHTML = '<div class="heatmap" style="grid-template-columns:repeat(auto-fill,minmax(50px,1fr));">' +
    periods.slice(-90).map(p => {
      const intensity = Math.min(Math.abs(p.pnl) / maxAbs, 1);
      const bg = p.pnl >= 0
        ? `rgba(63,185,80,${0.1 + intensity * 0.6})`
        : `rgba(248,81,73,${0.1 + intensity * 0.6})`;
      return `<div class="heatmap-cell" style="background:${bg}" title="${p.label}: $${p.pnl.toFixed(0)}">
        ${p.label.split(' ').pop() || p.label}<br><small>${p.pnl >= 0 ? '+' : ''}${p.pnl.toFixed(0)}</small></div>`;
    }).join('') + '</div>';

  // Periods table
  const tbody = document.querySelector('#periods-table tbody');
  tbody.innerHTML = periods.slice().reverse().slice(0,50).map(p => `
    <tr>
      <td>${p.label}</td>
      <td>${p.fills}</td>
      <td class="${pnlClass(p.pnl)}">${fmt(p.pnl,2)}</td>
      <td class="${pnlClass(p.raw_pnl)}">${fmt(p.raw_pnl,2)}</td>
      <td>${p.wr.toFixed(0)}%</td>
      <td>${(p.coins||[]).join(', ')}</td>
    </tr>
  `).join('');
}

// Detail tab switching
document.getElementById('detail-tf-tabs').addEventListener('click', e => {
  if (!e.target.dataset.tf) return;
  detailTf = e.target.dataset.tf;
  document.querySelectorAll('#detail-tf-tabs .tab').forEach(t => t.classList.remove('active'));
  e.target.classList.add('active');
  renderPeriods(detailTf);
});

// Run scan
async function runScan() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Scanning...';
  document.getElementById('status-msg').textContent = 'Running golden wallet scan + backtest... this may take several minutes.';

  try {
    const r = await fetch('/api/backtest/run', { method: 'POST' });
    const result = await r.json();
    document.getElementById('status-msg').textContent =
      `Scan complete: ${result.golden || 0} golden wallets found out of ${result.scanned || 0} evaluated.`;
    await load();
  } catch(e) {
    document.getElementById('status-msg').textContent = 'Scan failed: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = 'Run Golden Scan';
}

// Auto-refresh every 30s
setInterval(load, 30000);
load();
</script>
</body>
</html>"""
