"""
Options Flow Dashboard - Standalone web dashboard on port 8081.
Three-panel layout: Account/Convictions | Flow Heatmap | Unusual Prints Tape
"""
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
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
<title>Options Flow Desk</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
  --bg: #f6f2ea;
  --panel: #fffdf9;
  --panel-soft: #f9f4ec;
  --ink: #181512;
  --muted: #6b645c;
  --line: #e3d8c7;
  --teal: #1f6f5f;
  --blue: #2f5b9f;
  --amber: #b9771f;
  --red: #b54d3f;
  --green: #207f59;
  --shadow: 0 14px 34px rgba(40,31,20,.04);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  min-height: 100vh;
  background:
    radial-gradient(circle at top, rgba(31,111,95,.06), transparent 32%),
    linear-gradient(180deg, #fbf8f1 0%, var(--bg) 100%);
  color: var(--ink);
  font-family: 'IBM Plex Sans', sans-serif;
  padding: 28px 18px 38px;
}
a { color: inherit; }
.shell { max-width: 1420px; margin: 0 auto; }
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
  margin-bottom: 22px;
  padding-bottom: 18px;
  border-bottom: 1px solid rgba(227,216,199,.9);
}
.eyebrow {
  font-size: .74rem;
  text-transform: uppercase;
  letter-spacing: .16em;
  color: var(--teal);
  font-weight: 700;
}
h1 {
  margin: 8px 0 10px;
  font-family: 'Fraunces', serif;
  font-size: clamp(2rem, 3vw, 3.2rem);
  line-height: .98;
}
.subtitle { max-width: 62ch; color: var(--muted); line-height: 1.6; font-size: .94rem; }
.header-actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
.pill {
  display: inline-flex; align-items: center; gap: 8px; padding: 9px 12px;
  border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,.72); text-decoration: none;
  font-size: .82rem; font-weight: 600;
}
.live-dot {
  display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  background: var(--green);
}
.main { display: grid; grid-template-columns: 280px minmax(0, 1fr) 400px; gap: 18px; }
.stack,
.workspace,
.tape-stack { display: grid; gap: 18px; align-content: start; }
.panel {
  background: linear-gradient(180deg,#fffdf9 0%,#fcf8f1 100%); border: 1px solid var(--line); border-radius: 22px; padding: 18px; box-shadow: var(--shadow);
}
.panel-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; margin-bottom: 12px; }
.panel-title { font-family: 'Fraunces', serif; font-size: 1.28rem; line-height: 1.08; }
.panel-copy { color: var(--muted); font-size: .86rem; line-height: 1.5; }
.metric-row {
  display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 10px 0;
  border-bottom: 1px solid rgba(227,216,199,.75);
}
.metric-row:last-child { border-bottom: 0; }
.metric-row .label { color: var(--muted); font-size: .85rem; }
.metric-row .value { color: var(--ink); font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.conviction-card {
  background: var(--panel-soft); border: 1px solid var(--line); border-radius: 16px; padding: 13px; margin-bottom: 10px;
}
.conviction-card .ticker { font-size: .96rem; font-weight: 700; }
.conviction-card .ticker.bullish,
.conviction-card .net-flow.bullish { color: var(--green); }
.conviction-card .ticker.bearish,
.conviction-card .net-flow.bearish { color: var(--red); }
.conviction-card .flow-detail { font-size: .79rem; color: var(--muted); margin-top: 5px; line-height: 1.45; }
.conviction-card .net-flow { font-size: .96rem; font-weight: 700; margin-top: 6px; }
.conviction-bar { height: 5px; border-radius: 999px; margin-top: 9px; }
.conviction-bar.bullish { background: linear-gradient(90deg, rgba(32,127,89,.14), var(--green)); }
.conviction-bar.bearish { background: linear-gradient(90deg, rgba(181,77,63,.14), var(--red)); }
.heatmap-grid {
  display: grid; grid-template-columns: 84px repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 8px;
}
.heatmap-header {
  font-size: .7rem; color: var(--muted); text-align: center; padding: 8px; text-transform: uppercase; letter-spacing: .12em;
}
.heatmap-label {
  display: flex; align-items: center; font-size: .8rem; color: var(--ink); padding: 8px; font-weight: 700;
}
.heatmap-cell {
  min-height: 54px; border-radius: 14px; padding: 10px; display: flex; align-items: center; justify-content: center;
  text-align: center; font-size: .76rem; font-weight: 700; border: 1px solid rgba(227,216,199,.75);
}
.flow-chart-container { height: 334px; }
.tape-header,
.tape-row {
  display: grid; grid-template-columns: 58px 46px 96px 72px 70px 64px 82px 72px; gap: 8px; align-items: center;
}
.tape-header {
  padding: 10px 2px; border-bottom: 1px solid var(--line); font-size: .67rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .12em;
}
.tape-body { max-height: 620px; overflow: auto; }
.tape-row {
  padding: 11px 2px; border-bottom: 1px solid rgba(227,216,199,.75); font-size: .79rem; cursor: pointer;
  transition: background .15s ease;
}
.tape-row:hover { background: rgba(249,244,236,.85); }
.tier {
  display: inline-flex; align-items: center; justify-content: center; padding: 3px 7px; border-radius: 999px;
  font-size: .64rem; font-weight: 700; border: 1px solid transparent;
}
.tier-MEGA_BLOCK { background: rgba(127,95,159,.10); color: #7f5f9f; border-color: rgba(127,95,159,.15); }
.tier-BLOCK { background: rgba(185,119,31,.10); color: var(--amber); border-color: rgba(185,119,31,.15); }
.tier-SWEEP { background: rgba(47,91,159,.08); color: var(--blue); border-color: rgba(47,91,159,.12); }
.tier-LARGE { background: rgba(107,100,92,.08); color: var(--muted); border-color: rgba(107,100,92,.12); }
.dir-bullish { color: var(--green); }
.dir-bearish { color: var(--red); }
.order-panel { display: grid; gap: 14px; }
.order-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.order-field { display: flex; flex-direction: column; gap: 6px; }
.order-field label { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; }
.order-field input,
.order-field select {
  width: 100%; border: 1px solid var(--line); border-radius: 12px; padding: 11px 13px; background: var(--panel-soft);
  color: var(--ink); font-size: .88rem; font-family: 'IBM Plex Mono', monospace;
}
.order-field .computed {
  min-height: 42px; display: flex; align-items: center; padding: 11px 13px; border-radius: 12px;
  background: var(--panel-soft); border: 1px solid var(--line); color: var(--amber); font-family: 'IBM Plex Mono', monospace; font-weight: 600;
}
.order-actions { display: flex; gap: 10px; }
.btn-order {
  flex: 1; border: 0; border-radius: 999px; padding: 11px 15px; font-weight: 700; font-size: .88rem; cursor: pointer;
}
.btn-order.buy { background: rgba(32,127,89,.1); color: var(--green); border: 1px solid rgba(32,127,89,.18); }
.btn-order.sell { background: rgba(181,77,63,.1); color: var(--red); border: 1px solid rgba(181,77,63,.18); }
.empty-state {
  text-align: center; color: var(--muted); padding: 24px 16px; font-size: .88rem; line-height: 1.45;
}
@media (max-width: 1260px) {
  .main { grid-template-columns: 280px minmax(0, 1fr); }
  .tape-stack { grid-column: 1 / -1; }
}
@media (max-width: 900px) {
  body { padding: 16px 12px 26px; }
  .topbar { flex-direction: column; }
  .main { grid-template-columns: 1fr; }
  .order-grid { grid-template-columns: 1fr; }
  .tape-header,
  .tape-row { grid-template-columns: 58px 48px 1.2fr 1fr 70px 66px 80px 76px; font-size: .73rem; }
}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div>
      <p class="eyebrow">Options Desk</p>
      <h1>Flow Overview</h1>
      <p class="subtitle">A simpler read on options pressure: spot board on the left, directional concentration in the center, and the raw tape with a staging ticket on the right.</p>
    </div>
    <div class="header-actions">
      <a class="pill" href="/">Back to cockpit</a>
      <span class="pill"><span class="live-dot"></span> live scan</span>
      <span class="pill" id="last-update">Loading...</span>
      <span class="pill"><span id="print-count">0</span></span>
    </div>
  </header>

  <section class="main">
    <aside class="stack">
      <article class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Spot</p>
            <h2 class="panel-title">Underlyings</h2>
          </div>
        </div>
        <div id="spot-prices"></div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Risk</p>
            <h2 class="panel-title">Guardrails</h2>
          </div>
        </div>
        <div class="metric-row"><span class="label">Per trade</span><span class="value">1%</span></div>
        <div class="metric-row"><span class="label">Total risk</span><span class="value">5%</span></div>
        <div class="metric-row"><span class="label">Max positions</span><span class="value">5</span></div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Conviction</p>
            <h2 class="panel-title">Top Pressure</h2>
          </div>
        </div>
        <div id="convictions-list"><div class="empty-state">Scanning options flow...</div></div>
      </article>
    </aside>

    <main class="workspace">
      <article class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Heatmap</p>
            <h2 class="panel-title">Net Notional by Expiry</h2>
          </div>
        </div>
        <div class="heatmap-grid" id="heatmap-grid"></div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Bias</p>
            <h2 class="panel-title">Ticker Flow Balance</h2>
          </div>
        </div>
        <div class="flow-chart-container">
          <canvas id="flow-chart"></canvas>
        </div>
      </article>
    </main>

    <aside class="tape-stack">
      <article class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Tape</p>
            <h2 class="panel-title">Unusual Prints</h2>
          </div>
        </div>
        <div class="tape-header">
          <span>Time</span><span>Ticker</span><span>Strike/Type</span><span>Expiry</span>
          <span>Vol/OI</span><span>Tier</span><span>Notional</span><span>Flow</span>
        </div>
        <div class="tape-body" id="tape-body">
          <div class="empty-state">Waiting for unusual prints...</div>
        </div>
      </article>

      <article class="panel order-panel" id="order-panel" style="display:none;">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Ticket</p>
            <h2 class="panel-title">Order Stub</h2>
          </div>
        </div>
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
            <label>Limit price</label>
            <input type="number" id="order-price" step="0.0001">
          </div>
          <div class="order-field">
            <label>Breakeven</label>
            <span class="computed" id="order-breakeven">--</span>
          </div>
          <div class="order-field">
            <label>Move needed</label>
            <span class="computed" id="order-pct-move">--</span>
          </div>
        </div>
        <div class="order-actions">
          <button class="btn-order buy" onclick="placeOrder('buy')">Buy</button>
          <button class="btn-order sell" onclick="placeOrder('sell')">Sell</button>
        </div>
      </article>
    </aside>
  </section>
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
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

function formatUSD(val) {
  if (Math.abs(val) >= 1e6) return '$' + (val / 1e6).toFixed(2) + 'M';
  if (Math.abs(val) >= 1e3) return '$' + (val / 1e3).toFixed(1) + 'K';
  return '$' + Number(val || 0).toFixed(0);
}

function renderAll(data) {
  const summary = data.summary || {};
  const underlyings = summary.currencies_tracked || Object.keys(data.spot_prices || {}).length;
  document.getElementById('last-update').textContent = 'Updated ' + new Date(data.timestamp).toLocaleTimeString();
  document.getElementById('print-count').textContent = `${summary.total_unusual || 0} prints | ${underlyings} names`;
  renderSpotPrices(data.spot_prices || {});
  renderConvictions(data.convictions || []);
  renderHeatmap(data.heatmap || []);
  renderFlowChart(data.flow_bars || []);
  renderTape(data.unusual_prints || []);
}

function renderSpotPrices(prices) {
  const el = document.getElementById('spot-prices');
  const entries = Object.entries(prices || {});
  if (!entries.length) {
    el.innerHTML = '<div class="empty-state">No spot prices yet.</div>';
    return;
  }
  el.innerHTML = entries.map(([ticker, price]) => (
    '<div class="metric-row"><span class="label">' + ticker + '</span><span class="value">' + formatUSD(price) + '</span></div>'
  )).join('');
}

function renderConvictions(convictions) {
  const el = document.getElementById('convictions-list');
  if (!convictions.length) {
    el.innerHTML = '<div class="empty-state">No strong convictions yet.</div>';
    return;
  }
  el.innerHTML = convictions.slice(0, 5).map(c => {
    const dir = c.direction === 'BULLISH' ? 'bullish' : 'bearish';
    const netStr = formatUSD(Math.abs(c.net_flow));
    return '<div class="conviction-card">' +
      '<div class="ticker ' + dir + '">' + c.ticker + ' ' + c.direction + '</div>' +
      '<div class="net-flow ' + dir + '">Net ' + netStr + ' across ' + c.total_prints + ' prints</div>' +
      '<div class="flow-detail">Bullish: ' + formatUSD(c.bullish_notional) + ' | Bearish: ' + formatUSD(c.bearish_notional) + '</div>' +
      '<div class="flow-detail">Conviction: ' + c.conviction_pct + '% | Spot: ' + formatUSD(c.spot_price) + '</div>' +
      '<div class="conviction-bar ' + dir + '" style="width:' + Math.max(8, c.conviction_pct) + '%"></div>' +
      '</div>';
  }).join('');
}

function renderHeatmap(heatmap) {
  const grid = document.getElementById('heatmap-grid');
  const windows = ['weekly', 'monthly', 'quarterly', 'leap'];
  const tickers = [...new Set(heatmap.map(h => h.ticker))];
  if (!tickers.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1">No heatmap data yet.</div>';
    return;
  }

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
      if (val > 0) bg = 'rgba(32, 127, 89, ' + (0.10 + intensity * 0.50) + ')';
      else if (val < 0) bg = 'rgba(181, 77, 63, ' + (0.10 + intensity * 0.50) + ')';
      else bg = 'rgba(107, 100, 92, 0.08)';
      const text = val !== 0 ? formatUSD(val) : '--';
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
  const colors = values.map(v => v >= 0 ? '#1f6f5f' : '#b54d3f');
  const borders = values.map(v => v >= 0 ? '#17584b' : '#943b30');

  if (flowChart) flowChart.destroy();
  flowChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Net Flow',
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1,
        borderRadius: 8,
        barThickness: 18,
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
            label: function(ctx) { return 'Net ' + formatUSD(ctx.parsed.x); }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(107,100,92,0.12)' },
          ticks: { color: '#6b645c', callback: v => formatUSD(v) }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#181512', font: { weight: '700' } }
        }
      }
    }
  });
}

function renderTape(prints) {
  const el = document.getElementById('tape-body');
  if (!prints.length) {
    el.innerHTML = '<div class="empty-state">Waiting for unusual prints...</div>';
    return;
  }

  el.innerHTML = prints.map((p, i) => {
    const dirClass = p.direction === 'bullish' ? 'dir-bullish' : 'dir-bearish';
    const tierClass = 'tier-' + p.tier;
    return '<div class="tape-row" onclick="selectPrint(' + i + ')">' +
      '<span>' + p.time + '</span>' +
      '<span style="font-weight:700">' + p.ticker + '</span>' +
      '<span>' + p.strike + ' ' + p.option_type.toUpperCase() + '</span>' +
      '<span>' + p.expiry + '</span>' +
      '<span style="color:#b9771f;font-weight:700">' + p.vol_oi_ratio + 'x</span>' +
      '<span><span class="tier ' + tierClass + '">' + p.tier.replace('_', ' ') + '</span></span>' +
      '<span style="font-weight:700">' + formatUSD(p.notional) + '</span>' +
      '<span class="' + dirClass + '">' + p.direction.toUpperCase() + '</span>' +
      '</div>';
  }).join('');
}

function selectPrint(idx) {
  if (!currentData || !currentData.unusual_prints[idx]) return;
  const p = currentData.unusual_prints[idx];
  document.getElementById('order-panel').style.display = 'grid';
  document.getElementById('order-instrument').value = p.instrument;
  document.getElementById('order-side').value = p.direction === 'bullish' ? 'buy' : 'sell';
  document.getElementById('order-price').value = p.price || '';

  const spot = p.spot_price || 0;
  const strike = p.strike || 0;
  if (spot > 0 && strike > 0) {
    const pctMove = ((strike - spot) / spot * 100).toFixed(2);
    document.getElementById('order-breakeven').textContent = formatUSD(strike);
    document.getElementById('order-pct-move').textContent = pctMove + '%';
  } else {
    document.getElementById('order-breakeven').textContent = '--';
    document.getElementById('order-pct-move').textContent = '--';
  }
  document.getElementById('order-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function placeOrder(side) {
  const instrument = document.getElementById('order-instrument').value;
  const qty = document.getElementById('order-qty').value;
  const price = document.getElementById('order-price').value;
  if (!instrument) {
    alert('Select a print first');
    return;
  }
  fetch('/api/order', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ instrument, side, qty: parseFloat(qty), price: parseFloat(price) })
  })
    .then(r => r.json())
    .then(data => { alert(data.status || 'Order submitted'); })
    .catch(e => { alert('Order error: ' + e); });
}

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
        """Handle order placement (placeholder - needs Deribit API keys)."""
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
