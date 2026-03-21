"""
Options Flow Dashboard — Standalone web dashboard on port 8081.
Three-panel layout: Account/Convictions | Flow Heatmap | Unusual Prints Tape
"""
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import os

logger = logging.getLogger(__name__)

# Module-level scanner reference (set by start_options_dashboard)
_scanner = None


def _get_dashboard_html() -> str:
    """Return the full HTML/CSS/JS for the options flow dashboard."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Options Flow Scanner</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Courier New', monospace; background: #0a0a0f; color: #e0e0e0; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2a2a4a; }
.header h1 { font-size: 18px; color: #00ff88; letter-spacing: 2px; }
.header .status { font-size: 12px; color: #888; }
.header .live-dot { display: inline-block; width: 8px; height: 8px; background: #00ff88; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

.main { display: grid; grid-template-columns: 280px 1fr 420px; height: calc(100vh - 50px); gap: 1px; background: #1a1a2e; }

/* Left Panel */
.left-panel { background: #0d0d15; padding: 16px; overflow-y: auto; }
.panel-title { font-size: 13px; color: #00ff88; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; border-bottom: 1px solid #2a2a4a; padding-bottom: 6px; }
.account-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 12px; border-bottom: 1px solid #151520; }
.account-row .label { color: #888; }
.account-row .value { color: #fff; font-weight: bold; }
.account-row .value.positive { color: #00ff88; }
.account-row .value.negative { color: #ff4444; }

.conviction-card { background: #12121e; border: 1px solid #2a2a4a; border-radius: 6px; padding: 10px; margin-bottom: 8px; }
.conviction-card .ticker { font-size: 16px; font-weight: bold; }
.conviction-card .ticker.bullish { color: #00ff88; }
.conviction-card .ticker.bearish { color: #ff4444; }
.conviction-card .flow-detail { font-size: 11px; color: #888; margin-top: 4px; }
.conviction-card .net-flow { font-size: 14px; font-weight: bold; margin-top: 4px; }
.conviction-bar { height: 4px; border-radius: 2px; margin-top: 6px; }
.conviction-bar.bullish { background: linear-gradient(90deg, #00ff88, #004422); }
.conviction-bar.bearish { background: linear-gradient(90deg, #ff4444, #440000); }

/* Center Panel */
.center-panel { background: #0d0d15; padding: 16px; overflow-y: auto; }
.heatmap-container { margin-bottom: 24px; }
.heatmap-grid { display: grid; grid-template-columns: 80px repeat(4, 1fr); gap: 2px; margin-top: 8px; }
.heatmap-header { font-size: 10px; color: #888; text-align: center; padding: 6px; text-transform: uppercase; }
.heatmap-label { font-size: 12px; color: #fff; padding: 6px; display: flex; align-items: center; font-weight: bold; }
.heatmap-cell { border-radius: 4px; padding: 8px; text-align: center; font-size: 11px; font-weight: bold; min-height: 40px; display: flex; align-items: center; justify-content: center; }
.flow-chart-container { height: 200px; margin-top: 16px; }

/* Right Panel */
.right-panel { background: #0d0d15; overflow-y: auto; }
.tape-header { display: grid; grid-template-columns: 55px 45px 90px 70px 65px 55px 80px 70px; padding: 8px 12px; font-size: 10px; color: #888; text-transform: uppercase; border-bottom: 1px solid #2a2a4a; position: sticky; top: 0; background: #0d0d15; z-index: 10; }
.tape-row { display: grid; grid-template-columns: 55px 45px 90px 70px 65px 55px 80px 70px; padding: 6px 12px; font-size: 11px; border-bottom: 1px solid #0f0f1a; cursor: pointer; transition: background 0.15s; }
.tape-row:hover { background: #1a1a2e; }
.tape-row .tier { font-weight: bold; padding: 1px 4px; border-radius: 3px; font-size: 9px; text-align: center; }
.tier-MEGA_BLOCK { background: #ff00ff33; color: #ff88ff; border: 1px solid #ff00ff; }
.tier-BLOCK { background: #ffaa0033; color: #ffcc44; border: 1px solid #ffaa00; }
.tier-SWEEP { background: #00aaff33; color: #44ccff; border: 1px solid #00aaff; }
.tier-LARGE { background: #88888833; color: #aaa; border: 1px solid #888; }
.dir-bullish { color: #00ff88; }
.dir-bearish { color: #ff4444; }

/* Order Entry */
.order-panel { background: #12121e; border-top: 2px solid #2a2a4a; padding: 12px; }
.order-panel .panel-title { margin-bottom: 8px; }
.order-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.order-field { display: flex; flex-direction: column; }
.order-field label { font-size: 10px; color: #888; margin-bottom: 2px; }
.order-field input, .order-field select { background: #1a1a2e; border: 1px solid #2a2a4a; color: #fff; padding: 6px 8px; border-radius: 4px; font-size: 12px; font-family: 'Courier New', monospace; }
.order-field .computed { font-size: 11px; color: #ffcc44; font-weight: bold; }
.btn-order { background: #00ff88; color: #000; border: none; padding: 10px 20px; font-weight: bold; font-size: 13px; border-radius: 4px; cursor: pointer; margin-top: 8px; font-family: 'Courier New', monospace; }
.btn-order:hover { background: #00cc66; }
.btn-order.sell { background: #ff4444; color: #fff; }
.btn-order.sell:hover { background: #cc3333; }

.empty-state { text-align: center; color: #555; padding: 40px; font-size: 14px; }
</style>
</head>
<body>

<div class="header">
  <h1><span class="live-dot"></span>OPTIONS FLOW SCANNER</h1>
  <div class="status">
    <span id="last-update">Loading...</span> |
    <span id="print-count">0</span> unusual prints tracked
  </div>
</div>

<div class="main">
  <!-- LEFT PANEL: Account + Convictions -->
  <div class="left-panel">
    <div class="panel-title">Spot Prices</div>
    <div id="spot-prices"></div>

    <div class="panel-title" style="margin-top: 16px;">Risk Limits</div>
    <div class="account-row"><span class="label">Per Trade</span><span class="value">1%</span></div>
    <div class="account-row"><span class="label">Total Risk</span><span class="value">5%</span></div>
    <div class="account-row"><span class="label">Max Positions</span><span class="value">5</span></div>

    <div class="panel-title" style="margin-top: 16px;">Top Convictions</div>
    <div id="convictions-list">
      <div class="empty-state">Scanning...</div>
    </div>
  </div>

  <!-- CENTER PANEL: Heatmap + Flow Chart -->
  <div class="center-panel">
    <div class="panel-title">Flow Heatmap (Net Notional by Expiry)</div>
    <div class="heatmap-container">
      <div class="heatmap-grid" id="heatmap-grid">
        <!-- Generated by JS -->
      </div>
    </div>

    <div class="panel-title">Net Directional Flow by Ticker</div>
    <div class="flow-chart-container">
      <canvas id="flow-chart"></canvas>
    </div>
  </div>

  <!-- RIGHT PANEL: Tape + Order Entry -->
  <div class="right-panel">
    <div class="panel-title" style="padding: 8px 12px;">Unusual Prints Tape</div>
    <div class="tape-header">
      <span>Time</span><span>Ticker</span><span>Strike/Type</span><span>Expiry</span>
      <span>Vol/OI</span><span>Tier</span><span>Notional</span><span>Flow</span>
    </div>
    <div id="tape-body">
      <div class="empty-state">Waiting for unusual prints...</div>
    </div>

    <!-- Order Entry -->
    <div class="order-panel" id="order-panel" style="display:none;">
      <div class="panel-title">Order Entry</div>
      <div class="order-grid">
        <div class="order-field">
          <label>Instrument</label>
          <input type="text" id="order-instrument" readonly>
        </div>
        <div class="order-field">
          <label>Side</label>
          <select id="order-side">
            <option value="buy">BUY</option>
            <option value="sell">SELL</option>
          </select>
        </div>
        <div class="order-field">
          <label>Quantity</label>
          <input type="number" id="order-qty" value="1" min="0.1" step="0.1">
        </div>
        <div class="order-field">
          <label>Limit Price</label>
          <input type="number" id="order-price" step="0.0001">
        </div>
        <div class="order-field">
          <label>Breakeven</label>
          <span class="computed" id="order-breakeven">—</span>
        </div>
        <div class="order-field">
          <label>% Move Needed</label>
          <span class="computed" id="order-pct-move">—</span>
        </div>
      </div>
      <div style="display:flex; gap:8px; margin-top:8px;">
        <button class="btn-order" onclick="placeOrder('buy')">BUY</button>
        <button class="btn-order sell" onclick="placeOrder('sell')">SELL</button>
      </div>
    </div>
  </div>
</div>

<script>
let flowChart = null;
let currentData = null;

async function fetchData() {
  try {
    const resp = await fetch('/api/flow');
    if (!resp.ok) return;
    currentData = await resp.json();
    renderAll(currentData);
  } catch(e) { console.error('Fetch error:', e); }
}

function formatUSD(val) {
  if (Math.abs(val) >= 1e6) return '$' + (val/1e6).toFixed(2) + 'M';
  if (Math.abs(val) >= 1e3) return '$' + (val/1e3).toFixed(1) + 'K';
  return '$' + val.toFixed(0);
}

function renderAll(data) {
  document.getElementById('last-update', '').textContent = 'Updated: ' + new Date(data.timestamp).toLocaleTimeString();
  document.getElementById('print-count').textContent = data.summary.total_unusual;

  renderSpotPrices(data.spot_prices);
  renderConvictions(data.convictions);
  renderHeatmap(data.heatmap, data.spot_prices);
  renderFlowChart(data.flow_bars);
  renderTape(data.unusual_prints);
}

function renderSpotPrices(prices) {
  const el = document.getElementById('spot-prices');
  let html = '';
  for (const [ticker, price] of Object.entries(prices)) {
    html += '<div class="account-row"><span class="label">' + ticker + '</span><span class="value">' + formatUSD(price) + '</span></div>';
  }
  el.innerHTML = html;
}

function renderConvictions(convictions) {
  const el = document.getElementById('convictions-list');
  if (!convictions.length) { el.innerHTML = '<div class="empty-state">No strong convictions yet</div>'; return; }
  let html = '';
  convictions.slice(0, 5).forEach(c => {
    const dir = c.direction === 'BULLISH' ? 'bullish' : 'bearish';
    const netStr = formatUSD(Math.abs(c.net_flow));
    html += '<div class="conviction-card">' +
      '<div class="ticker ' + dir + '">' + c.ticker + ' ' + c.direction + '</div>' +
      '<div class="net-flow ' + dir + '">Net ' + netStr + ' across ' + c.total_prints + ' prints</div>' +
      '<div class="flow-detail">Bullish: ' + formatUSD(c.bullish_notional) + ' | Bearish: ' + formatUSD(c.bearish_notional) + '</div>' +
      '<div class="flow-detail">Conviction: ' + c.conviction_pct + '% | Spot: ' + formatUSD(c.spot_price) + '</div>' +
      '<div class="conviction-bar ' + dir + '" style="width:' + c.conviction_pct + '%"></div>' +
      '</div>';
  });
  el.innerHTML = html;
}

function renderHeatmap(heatmap, spotPrices) {
  const grid = document.getElementById('heatmap-grid');
  const windows = ['weekly', 'monthly', 'quarterly', 'leap'];
  const tickers = [...new Set(heatmap.map(h => h.ticker))];
  if (!tickers.length) { grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1">No heatmap data</div>'; return; }

  // Find max value for color scaling
  const maxVal = Math.max(...heatmap.map(h => Math.abs(h.net_notional)), 1);

  let html = '<div class="heatmap-header"></div>';
  windows.forEach(w => { html += '<div class="heatmap-header">' + w + '</div>'; });

  tickers.forEach(ticker => {
    html += '<div class="heatmap-label">' + ticker + '</div>';
    windows.forEach(w => {
      const cell = heatmap.find(h => h.ticker === ticker && h.window === w);
      const val = cell ? cell.net_notional : 0;
      const intensity = Math.min(Math.abs(val) / maxVal, 1);
      let bg;
      if (val > 0) bg = 'rgba(0, 255, 136, ' + (0.1 + intensity * 0.6) + ')';
      else if (val < 0) bg = 'rgba(255, 68, 68, ' + (0.1 + intensity * 0.6) + ')';
      else bg = 'rgba(30, 30, 50, 0.5)';
      const text = val !== 0 ? formatUSD(val) : '—';
      html += '<div class="heatmap-cell" style="background:' + bg + '">' + text + '</div>';
    });
  });
  grid.innerHTML = html;
}

function renderFlowChart(flowBars) {
  const ctx = document.getElementById('flow-chart');
  if (!flowBars.length) return;

  const labels = flowBars.map(f => f.ticker);
  const values = flowBars.map(f => f.net_flow);
  const colors = values.map(v => v >= 0 ? '#00ff88' : '#ff4444');

  if (flowChart) flowChart.destroy();
  flowChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Net Flow',
        data: values,
        backgroundColor: colors,
        borderWidth: 0,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(ctx) { return 'Net: ' + formatUSD(ctx.parsed.x); }
          }
        }
      },
      scales: {
        x: {
          grid: { color: '#1a1a2e' },
          ticks: { color: '#888', callback: v => formatUSD(v) }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#fff', font: { weight: 'bold' } }
        }
      }
    }
  });
}

function renderTape(prints) {
  const el = document.getElementById('tape-body');
  if (!prints.length) { el.innerHTML = '<div class="empty-state">Waiting for unusual prints...</div>'; return; }

  let html = '';
  prints.forEach((p, i) => {
    const dirClass = p.direction === 'bullish' ? 'dir-bullish' : 'dir-bearish';
    const tierClass = 'tier-' + p.tier;
    html += '<div class="tape-row" onclick="selectPrint(' + i + ')">' +
      '<span>' + p.time + '</span>' +
      '<span style="font-weight:bold">' + p.ticker + '</span>' +
      '<span>' + p.strike + ' ' + p.option_type.toUpperCase() + '</span>' +
      '<span>' + p.expiry + '</span>' +
      '<span style="color:#ffcc44">' + p.vol_oi_ratio + 'x</span>' +
      '<span><span class="tier ' + tierClass + '">' + p.tier.replace('_', ' ') + '</span></span>' +
      '<span style="font-weight:bold">' + formatUSD(p.notional) + '</span>' +
      '<span class="' + dirClass + '">' + p.direction.toUpperCase() + '</span>' +
      '</div>';
  });
  el.innerHTML = html;
}

function selectPrint(idx) {
  if (!currentData || !currentData.unusual_prints[idx]) return;
  const p = currentData.unusual_prints[idx];
  document.getElementById('order-panel').style.display = 'block';
  document.getElementById('order-instrument').value = p.instrument;
  document.getElementById('order-side').value = p.direction === 'bullish' ? 'buy' : 'sell';

  // Compute breakeven
  const spot = p.spot_price || 0;
  const strike = p.strike || 0;
  if (spot > 0 && strike > 0) {
    const pctMove = ((strike - spot) / spot * 100).toFixed(2);
    document.getElementById('order-breakeven').textContent = formatUSD(strike);
    document.getElementById('order-pct-move').textContent = pctMove + '%';
  }
}

function placeOrder(side) {
  const instrument = document.getElementById('order-instrument').value;
  const qty = document.getElementById('order-qty').value;
  const price = document.getElementById('order-price').value;
  if (!instrument) { alert('Select a print first'); return; }
  // POST to /api/order
  fetch('/api/order', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ instrument, side, qty: parseFloat(qty), price: parseFloat(price) })
  })
  .then(r => r.json())
  .then(data => { alert(data.status || 'Order submitted'); })
  .catch(e => { alert('Order error: ' + e); });
}

// Auto-refresh every 30 seconds
fetchData();
setInterval(fetchData, 30000);
</script>
</body>
</html>'''


class FlowDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the options flow dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logs

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self._serve_html()
        elif self.path == "/api/flow":
            self._serve_flow_data()
        elif self.path == "/api/health":
            self._json_response({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/order":
            self._handle_order()
        else:
            self.send_error(404)

    def _serve_html(self):
        html = _get_dashboard_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_flow_data(self):
        global _scanner
        if _scanner:
            data = _scanner.get_dashboard_data()
        else:
            data = {"error": "Scanner not initialized", "convictions": [],
                    "heatmap": [], "flow_bars": [], "unusual_prints": [],
                    "spot_prices": {}, "summary": {"total_unusual": 0, "currencies_tracked": 0},
                    "timestamp": ""}
        self._json_response(data)

    def _handle_order(self):
        """Handle order placement (placeholder — needs Deribit API keys)."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            logger.info(f"Order request: {body}")
            # TODO: Implement Deribit order execution with API keys
            self._json_response({
                "status": "Order queued (Deribit API keys required for live execution)",
                "order": body,
            })
        except Exception as e:
            self._json_response({"status": f"Error: {e}"}, code=400)

    def _json_response(self, data: dict, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())


def start_options_dashboard(scanner, port: int = None):
    """Start the options flow dashboard in a background thread."""
    global _scanner
    _scanner = scanner

    port = port or int(os.environ.get("OPTIONS_DASHBOARD_PORT", 8081))

    server = HTTPServer(("0.0.0.0", port), FlowDashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Options Flow Dashboard running on port {port}")
    return server
