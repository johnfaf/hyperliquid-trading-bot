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
    --bg: #0a0e14; --card: #12171e; --card-hover: #161d26; --border: #1e2a3a;
    --text: #d0d7e0; --dim: #6b7a8d; --accent: #4da6ff; --accent-dim: rgba(77,166,255,.12);
    --green: #00d68f; --green-dim: rgba(0,214,143,.12);
    --red: #ff5c5c; --red-dim: rgba(255,92,92,.12);
    --gold: #f0b429; --gold-dim: rgba(240,180,41,.12);
    --purple: #a78bfa; --orange: #ff9f43;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; line-height: 1.5; }
  .container { max-width: 1440px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; font-weight: 700; letter-spacing: -.3px; }
  h2 { font-size: 16px; margin: 28px 0 14px; color: var(--accent); font-weight: 600; }
  h3 { font-size: 14px; margin: 20px 0 10px; color: var(--dim); font-weight: 500; }
  .subtitle { color: var(--dim); font-size: 12px; margin-top: 4px; }
  .top-bar { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
  .btn-group { display: flex; gap: 8px; align-items: center; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .badge-gold { background: var(--gold-dim); color: var(--gold); }
  .badge-green { background: var(--green-dim); color: var(--green); }
  .badge-red { background: var(--red-dim); color: var(--red); }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 20px; }
  .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; transition: background .15s; }
  .stat-card:hover { background: var(--card-hover); }
  .stat-label { font-size: 10px; text-transform: uppercase; color: var(--dim); letter-spacing: .8px; font-weight: 500; }
  .stat-value { font-size: 20px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .stat-sub { font-size: 11px; color: var(--dim); margin-top: 2px; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .gold { color: var(--gold); }
  table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 10px; overflow: hidden; font-size: 13px; }
  th { background: #0e1319; padding: 10px 14px; text-align: left; font-size: 10px; text-transform: uppercase; color: var(--dim); letter-spacing: .8px; font-weight: 500; }
  td { padding: 10px 14px; border-top: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  tr:hover td { background: rgba(77,166,255,.03); }
  .clickable { cursor: pointer; }
  .clickable:hover td { background: rgba(77,166,255,.06); }
  .pnl-pos { color: var(--green); font-weight: 600; }
  .pnl-neg { color: var(--red); font-weight: 600; }
  .chart-container { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin: 12px 0; }
  canvas { width: 100% !important; height: 280px !important; }
  .heatmap { display: grid; gap: 3px; margin: 8px 0; }
  .heatmap-cell { padding: 6px 4px; text-align: center; font-size: 10px; border-radius: 4px; min-width: 40px; font-weight: 500; }
  .tabs { display: flex; gap: 4px; margin: 16px 0; }
  .tab { padding: 7px 16px; border-radius: 8px; cursor: pointer; background: var(--card); border: 1px solid var(--border); color: var(--dim); font-size: 12px; font-weight: 500; transition: all .15s; }
  .tab:hover { color: var(--text); border-color: var(--dim); }
  .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .detail-panel { display: none; }
  .detail-panel.active { display: block; }
  .back-btn { background: var(--card); border: 1px solid var(--border); color: var(--accent); padding: 7px 16px; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 500; transition: all .15s; }
  .back-btn:hover { background: var(--card-hover); }
  .btn { border: none; padding: 8px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 12px; transition: all .15s; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .btn-green { background: var(--green); color: #000; }
  .btn-green:hover:not(:disabled) { opacity: .85; }
  .btn-red { background: var(--red-dim); color: var(--red); border: 1px solid rgba(255,92,92,.3); }
  .btn-red:hover:not(:disabled) { background: rgba(255,92,92,.2); }
  .btn-outline { background: transparent; color: var(--accent); border: 1px solid var(--border); }
  .btn-outline:hover { border-color: var(--accent); background: var(--accent-dim); }
  .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #fff3; border-top-color: #fff; border-radius: 50%; animation: spin .8s linear infinite; margin-right: 6px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .flex-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .flex-row > * { flex: 1; min-width: 300px; }
  #status-msg { color: var(--dim); font-size: 12px; font-style: italic; margin: 8px 0; min-height: 18px; }
  .paper-section { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 24px; }
  .paper-section h2 { margin-top: 0; }
  .section-row { display: flex; justify-content: space-between; align-items: center; }
  code { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; }
  .tooltip { position: relative; }
  .tooltip:hover::after { content: attr(data-tip); position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: #1a2332; color: var(--text); padding: 4px 8px; border-radius: 4px; font-size: 11px; white-space: nowrap; z-index: 10; }
</style>
</head>
<body>
<div class="container">
  <div class="top-bar">
    <div>
      <h1>Golden Wallet Backtest</h1>
      <div class="subtitle">Multi-timeframe analysis &middot; +100ms copy delay &middot; 4.5bps slippage &middot; funding rate sim</div>
    </div>
    <div class="btn-group">
      <button class="btn btn-green" id="runBtn" onclick="runScan()">Run Golden Scan</button>
      <button class="btn btn-red" id="resetBtn" onclick="resetPaper()">Reset Paper Trades</button>
      <button class="btn btn-outline" onclick="exportCSV()">Export CSV</button>
      <a href="/" style="margin-left:8px;color:var(--accent);text-decoration:none;font-size:12px;">&larr; Main</a>
    </div>
  </div>
  <div id="status-msg"></div>

  <!-- Paper Trading Summary -->
  <div class="paper-section" id="paper-section">
    <div class="section-row">
      <h2 style="margin:0">Paper Trading Account</h2>
      <span id="paper-status" class="badge badge-green">Active</span>
    </div>
    <div class="stats-grid" id="paper-stats" style="margin-top:14px"></div>
  </div>

  <!-- Candle Backtest Section -->
  <div class="paper-section" id="candle-bt-section">
    <div class="section-row">
      <h2 style="margin:0">Candle Backtest</h2>
      <span id="cbt-speed" class="badge badge-green" style="display:none"></span>
    </div>
    <div class="subtitle" style="margin-bottom:14px">Run strategies against historical OHLCV data from Hyperliquid</div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px;">
      <div>
        <label class="stat-label" style="display:block;margin-bottom:4px;">Coin</label>
        <select id="cbt-coin" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
          <option>BTC</option><option>ETH</option><option>SOL</option><option>DOGE</option>
          <option>ARB</option><option>OP</option><option>AVAX</option><option>SUI</option>
          <option>WIF</option><option>PEPE</option><option>ONDO</option><option>LINK</option>
          <option>WLD</option><option>TIA</option><option>INJ</option><option>SEI</option>
        </select>
      </div>
      <div>
        <label class="stat-label" style="display:block;margin-bottom:4px;">Timeframe</label>
        <select id="cbt-tf" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
          <option value="1m">1 Min</option><option value="5m">5 Min</option>
          <option value="15m">15 Min</option><option value="1h" selected>1 Hour</option>
          <option value="4h">4 Hour</option><option value="1d">1 Day</option>
        </select>
      </div>
      <div>
        <label class="stat-label" style="display:block;margin-bottom:4px;">Strategy</label>
        <select id="cbt-strategy" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
          <optgroup label="Trend Following">
            <option value="momentum">Momentum (EMA Cross)</option>
            <option value="supertrend">SuperTrend</option>
            <option value="adx_trend">ADX Trend (+DI/-DI)</option>
            <option value="ichimoku">Ichimoku Cloud</option>
            <option value="breakout">Breakout (N-Period)</option>
            <option value="volume_breakout">Volume Breakout</option>
          </optgroup>
          <optgroup label="Mean Reversion">
            <option value="mean_reversion">Mean Reversion (BB)</option>
            <option value="rsi">RSI (Overbought/Oversold)</option>
            <option value="stochastic">Stochastic (%K/%D)</option>
            <option value="vwap_reversion">VWAP Reversion</option>
          </optgroup>
          <optgroup label="Oscillator / Combo">
            <option value="macd">MACD Crossover</option>
            <option value="macd_histogram">MACD Histogram</option>
            <option value="ema_rsi_combo">EMA + RSI Combo</option>
          </optgroup>
        </select>
      </div>
      <div>
        <label class="stat-label" style="display:block;margin-bottom:4px;">Start Date</label>
        <input type="date" id="cbt-start" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
      </div>
      <div>
        <label class="stat-label" style="display:block;margin-bottom:4px;">End Date</label>
        <input type="date" id="cbt-end" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
      </div>
    </div>

    <details style="margin-bottom:14px;">
      <summary style="cursor:pointer;color:var(--accent);font-size:12px;font-weight:500;">Advanced Parameters</summary>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:10px;">
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">Position Size %</label>
          <input type="number" id="cbt-possize" value="5" min="1" max="50" step="1" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">Leverage</label>
          <input type="number" id="cbt-leverage" value="3" min="1" max="20" step="0.5" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">Stop Loss %</label>
          <input type="number" id="cbt-sl" value="2" min="0.5" max="20" step="0.5" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">Take Profit %</label>
          <input type="number" id="cbt-tp" value="4" min="0.5" max="50" step="0.5" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">Fast MA Period</label>
          <input type="number" id="cbt-fast" value="10" min="2" max="100" step="1" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">Slow MA Period</label>
          <input type="number" id="cbt-slow" value="30" min="5" max="200" step="1" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">RSI Period</label>
          <input type="number" id="cbt-rsi" value="14" min="2" max="50" step="1" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
        <div>
          <label class="stat-label" style="display:block;margin-bottom:4px;">Trailing Stop %</label>
          <input type="number" id="cbt-trail" value="1.5" min="0" max="10" step="0.5" style="width:100%;padding:7px 10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;">
        </div>
      </div>
    </details>

    <div class="btn-group">
      <button class="btn btn-green" id="cbtRunBtn" onclick="runCandleBacktest()">Run Backtest</button>
      <button class="btn btn-outline" id="cbtFetchBtn" onclick="fetchCandleData()">Fetch Data Only</button>
      <button class="btn btn-outline" id="cbtCacheBtn" onclick="showCacheInfo()">Cache Info</button>
      <button class="btn btn-red" id="cbtClearBtn" onclick="clearCache()">Clear Cache</button>
    </div>
    <div id="cbt-status" style="color:var(--dim);font-size:12px;font-style:italic;margin-top:8px;min-height:18px;"></div>

    <!-- Results -->
    <div id="cbt-results" style="display:none;margin-top:20px;">
      <h3>Results</h3>
      <div class="stats-grid" id="cbt-stats"></div>

      <h3>Equity Curve</h3>
      <div class="chart-container"><canvas id="cbtEquityChart"></canvas></div>

      <h3>Drawdown</h3>
      <div class="chart-container"><canvas id="cbtDrawdownChart"></canvas></div>

      <h3>Trade Log</h3>
      <table id="cbt-trades-table">
        <thead><tr>
          <th>#</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>PnL%</th><th>Exit Reason</th><th>Hold</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

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

    <h3>Drawdown (%)</h3>
    <div class="chart-container"><canvas id="drawdownChart"></canvas></div>

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
let PAPER = null;
let selectedTf = '1d';
let detailTf = '1d';
let charts = {};

async function load() {
  try {
    const [btRes, paperRes] = await Promise.all([
      fetch('/api/backtest'),
      fetch('/api/status').catch(() => null),
    ]);
    DATA = await btRes.json();
    if (paperRes && paperRes.ok) {
      const status = await paperRes.json();
      PAPER = status.paper_account;
    }
    renderOverview();
    renderPaperStats();
  } catch(e) {
    document.getElementById('status-msg').textContent = 'No backtest data yet. Click "Run Golden Scan" to start.';
  }
}

function renderPaperStats() {
  const el = document.getElementById('paper-stats');
  if (!PAPER) {
    el.innerHTML = '<div class="stat-card"><div class="stat-label">Status</div><div class="stat-value" style="color:var(--dim)">No data</div></div>';
    return;
  }
  const balance = PAPER.balance || 0;
  const pnl = PAPER.total_pnl || 0;
  const trades = PAPER.total_trades || 0;
  const wins = PAPER.winning_trades || 0;
  const wr = trades > 0 ? (wins/trades*100).toFixed(1) : '0';
  const roi = balance > 0 ? ((pnl / 10000) * 100).toFixed(2) : '0';
  el.innerHTML = `
    <div class="stat-card"><div class="stat-label">Balance</div><div class="stat-value">$${balance.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
    <div class="stat-card"><div class="stat-label">Total PnL</div><div class="stat-value ${pnl>=0?'positive':'negative'}">${fmt(pnl,2)}</div></div>
    <div class="stat-card"><div class="stat-label">ROI</div><div class="stat-value ${pnl>=0?'positive':'negative'}">${roi}%</div></div>
    <div class="stat-card"><div class="stat-label">Trades</div><div class="stat-value">${trades}</div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">${wr}%</div></div>
    <div class="stat-card"><div class="stat-label">Wins / Losses</div><div class="stat-value"><span class="positive">${wins}</span> / <span class="negative">${trades-wins}</span></div></div>
  `;
}

async function resetPaper() {
  if (!confirm('This will DELETE all paper trades and reset balance to $10,000. Continue?')) return;
  const btn = document.getElementById('resetBtn');
  btn.disabled = true;
  btn.textContent = 'Resetting...';
  try {
    const r = await fetch('/api/paper/reset', {method:'POST'});
    const res = await r.json();
    document.getElementById('status-msg').textContent =
      'Paper trades reset: ' + (res.open_deleted||0) + ' open + ' + (res.closed_deleted||0) + ' closed cleared.';
    await load();
  } catch(e) {
    document.getElementById('status-msg').textContent = 'Reset failed: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = 'Reset Paper Trades';
}

function exportCSV() {
  if (!DATA || !DATA.wallets || !DATA.wallets.length) {
    alert('No data to export');
    return;
  }
  let csv = 'address,is_golden,raw_pnl,penalised_pnl,max_dd_pct,sharpe,win_rate,fills,trades_per_day\\n';
  DATA.wallets.forEach(w => {
    csv += [w.address,w.is_golden,w.raw_pnl.toFixed(2),w.penalised_pnl.toFixed(2),
            w.penalised_max_drawdown_pct.toFixed(2),w.sharpe_ratio.toFixed(3),
            w.win_rate.toFixed(1),w.total_fills,w.trades_per_day.toFixed(2)].join(',') + '\\n';
  });
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'backtest_wallets_' + new Date().toISOString().slice(0,10) + '.csv';
  a.click();
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

  // Equity chart + drawdown
  renderEquityChart(w);
  renderDrawdownChart(w);
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

function renderDrawdownChart(w) {
  const ctx = document.getElementById('drawdownChart').getContext('2d');
  if (charts.drawdown) charts.drawdown.destroy();

  const pen = w.penalised_equity_curve || [];
  const ts = w.equity_timestamps || [];
  const labels = ts.map(t => new Date(t).toLocaleDateString());

  // Compute drawdown series
  let peak = pen[0] || 10000;
  const dd = pen.map(v => {
    if (v > peak) peak = v;
    return peak > 0 ? ((v - peak) / peak * 100) : 0;
  });

  charts.drawdown = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Drawdown %',
        data: dd,
        borderColor: 'rgba(255,92,92,.8)',
        backgroundColor: 'rgba(255,92,92,.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#6b7a8d' } } },
      scales: {
        x: { ticks: { color: '#6b7a8d', maxTicksLimit: 12 }, grid: { color: '#1e2a3a' } },
        y: { ticks: { color: '#6b7a8d', callback: v => v.toFixed(1)+'%' }, grid: { color: '#1e2a3a' } },
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

// ─── Candle Backtest Functions ─────────────────────────────

let CBT_RESULT = null;

function _getCbtParams() {
  return {
    coin: document.getElementById('cbt-coin').value,
    timeframe: document.getElementById('cbt-tf').value,
    strategy: document.getElementById('cbt-strategy').value,
    start: document.getElementById('cbt-start').value || undefined,
    end: document.getElementById('cbt-end').value || undefined,
    position_size_pct: parseFloat(document.getElementById('cbt-possize').value) / 100,
    max_leverage: parseFloat(document.getElementById('cbt-leverage').value),
    stop_loss_pct: parseFloat(document.getElementById('cbt-sl').value) / 100,
    take_profit_pct: parseFloat(document.getElementById('cbt-tp').value) / 100,
    trailing_stop_pct: parseFloat(document.getElementById('cbt-trail').value) / 100,
    fast_period: parseInt(document.getElementById('cbt-fast').value),
    slow_period: parseInt(document.getElementById('cbt-slow').value),
    rsi_period: parseInt(document.getElementById('cbt-rsi').value),
  };
}

async function runCandleBacktest() {
  const btn = document.getElementById('cbtRunBtn');
  const status = document.getElementById('cbt-status');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running...';
  status.textContent = 'Fetching data and running backtest...';

  try {
    const params = _getCbtParams();
    const r = await fetch('/api/candle-backtest/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    const data = await r.json();

    if (data.error) {
      status.textContent = 'Error: ' + data.error;
      btn.disabled = false;
      btn.textContent = 'Run Backtest';
      return;
    }

    CBT_RESULT = data;
    status.textContent = `Done in ${data.duration_seconds.toFixed(2)}s — ` +
      `${(data.candles_per_second||0).toLocaleString()} candles/sec`;

    const badge = document.getElementById('cbt-speed');
    badge.style.display = 'inline-block';
    badge.textContent = `${(data.candles_per_second||0).toLocaleString()} candles/s`;

    renderCbtResults(data);
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }

  btn.disabled = false;
  btn.textContent = 'Run Backtest';
}

async function fetchCandleData() {
  const btn = document.getElementById('cbtFetchBtn');
  const status = document.getElementById('cbt-status');
  btn.disabled = true;
  btn.textContent = 'Fetching...';
  status.textContent = 'Downloading candle data from Hyperliquid...';

  try {
    const params = _getCbtParams();
    const r = await fetch('/api/candle-backtest/fetch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    const data = await r.json();
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
    } else {
      status.textContent = `Fetched ${data.candles.toLocaleString()} ${data.coin} ${data.timeframe} candles (cached locally)`;
    }
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }

  btn.disabled = false;
  btn.textContent = 'Fetch Data Only';
}

async function showCacheInfo() {
  const status = document.getElementById('cbt-status');
  try {
    const r = await fetch('/api/candle-backtest/cache');
    const data = await r.json();
    if (!data.cached || !data.cached.length) {
      status.textContent = 'Cache is empty. Fetch some data first.';
      return;
    }
    const lines = data.cached.map(c =>
      `${c.coin} ${c.timeframe}: ${c.candles.toLocaleString()} candles (${c.start} → ${c.end})`
    );
    status.innerHTML = `<strong>Cache</strong> (${data.stats.db_size_mb} MB): ` + lines.join(' | ');
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }
}

async function clearCache() {
  if (!confirm('Clear all cached candle data?')) return;
  const status = document.getElementById('cbt-status');
  try {
    await fetch('/api/candle-backtest/cache/clear', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    status.textContent = 'Cache cleared.';
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }
}

function renderCbtResults(data) {
  document.getElementById('cbt-results').style.display = 'block';

  const s = document.getElementById('cbt-stats');
  const pnlCls = data.total_pnl >= 0 ? 'positive' : 'negative';
  s.innerHTML = `
    <div class="stat-card"><div class="stat-label">Total PnL</div><div class="stat-value ${pnlCls}">${fmt(data.total_pnl,2)}</div><div class="stat-sub">${data.total_pnl_pct >= 0 ? '+' : ''}${data.total_pnl_pct.toFixed(1)}%</div></div>
    <div class="stat-card"><div class="stat-label">Trades</div><div class="stat-value">${data.total_trades}</div><div class="stat-sub"><span class="positive">${data.winning_trades}W</span> / <span class="negative">${data.losing_trades}L</span></div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">${data.win_rate.toFixed(1)}%</div></div>
    <div class="stat-card"><div class="stat-label">Sharpe</div><div class="stat-value">${data.sharpe_ratio.toFixed(3)}</div></div>
    <div class="stat-card"><div class="stat-label">Sortino</div><div class="stat-value">${data.sortino_ratio.toFixed(3)}</div></div>
    <div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value negative">${data.max_drawdown_pct.toFixed(1)}%</div></div>
    <div class="stat-card"><div class="stat-label">Profit Factor</div><div class="stat-value">${data.profit_factor.toFixed(2)}</div></div>
    <div class="stat-card"><div class="stat-label">Best Trade</div><div class="stat-value positive">${fmt(data.best_trade_pnl,2)}</div></div>
    <div class="stat-card"><div class="stat-label">Worst Trade</div><div class="stat-value negative">${fmt(data.worst_trade_pnl,2)}</div></div>
    <div class="stat-card"><div class="stat-label">Total Fees</div><div class="stat-value">${fmt(-data.total_fees,2)}</div></div>
  `;

  // Equity chart
  renderCbtEquityChart(data.equity_curve);
  renderCbtDrawdownChart(data.drawdown_curve);
  renderCbtTradeTable(data.trades || []);
}

function renderCbtEquityChart(equity) {
  const ctx = document.getElementById('cbtEquityChart').getContext('2d');
  if (charts.cbtEquity) charts.cbtEquity.destroy();

  const labels = equity.map((_, i) => i);
  charts.cbtEquity = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Equity', data: equity, borderColor: '#4da6ff', borderWidth: 1.5, pointRadius: 0, fill: false },
        { label: 'Initial ($10k)', data: equity.map(() => 10000), borderColor: '#30363d', borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        x: { display: false },
        y: { ticks: { color: '#8b949e', callback: v => '$'+v.toLocaleString() }, grid: { color: '#21262d' } },
      }
    }
  });
}

function renderCbtDrawdownChart(dd) {
  const ctx = document.getElementById('cbtDrawdownChart').getContext('2d');
  if (charts.cbtDrawdown) charts.cbtDrawdown.destroy();

  charts.cbtDrawdown = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dd.map((_, i) => i),
      datasets: [{
        label: 'Drawdown %',
        data: dd.map(v => -v),
        borderColor: 'rgba(255,92,92,.8)',
        backgroundColor: 'rgba(255,92,92,.08)',
        borderWidth: 1.5, pointRadius: 0, fill: true,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#6b7a8d' } } },
      scales: {
        x: { display: false },
        y: { ticks: { color: '#6b7a8d', callback: v => v.toFixed(1)+'%' }, grid: { color: '#1e2a3a' } },
      }
    }
  });
}

function renderCbtTradeTable(trades) {
  const tbody = document.querySelector('#cbt-trades-table tbody');
  tbody.innerHTML = trades.slice(0, 100).map((t, i) => {
    const sideClass = t.side === 'long' ? 'positive' : 'negative';
    const pnlClass2 = t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    return `<tr>
      <td>${i+1}</td>
      <td class="${sideClass}" style="font-weight:600;text-transform:uppercase;">${t.side}</td>
      <td>$${t.entry_price.toFixed(2)}</td>
      <td>$${t.exit_price.toFixed(2)}</td>
      <td class="${pnlClass2}">${fmt(t.pnl,2)}</td>
      <td class="${pnlClass2}">${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%</td>
      <td><span class="badge ${t.exit_reason === 'take_profit' ? 'badge-green' : t.exit_reason === 'stop_loss' ? 'badge-red' : 'badge-gold'}">${t.exit_reason}</span></td>
      <td>${t.hold_candles} candles</td>
    </tr>`;
  }).join('');
}

// Set default dates (90 days ago → today)
(function setDefaultDates() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 90);
  document.getElementById('cbt-end').value = end.toISOString().slice(0, 10);
  document.getElementById('cbt-start').value = start.toISOString().slice(0, 10);
})();

// Auto-refresh every 30s
setInterval(load, 30000);
load();
</script>
</body>
</html>"""
