"""
Stress Test Dashboard
======================
Interactive dashboard for running and viewing stress test results.
Integrates with the main dashboard on the /stress route.
"""

import json
import logging
import os
import glob
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports"

logger = logging.getLogger("stress_dashboard")


# ─── Data layer ───────────────────────────────────────────────

def get_stress_reports() -> list:
    """Load all stress test JSON reports, newest first."""
    reports = []
    pattern = str(REPORTS_DIR / "stress_test_*.json")
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_file"] = os.path.basename(path)
            reports.append(data)
        except Exception as e:
            logger.warning("Could not load %s: %s", path, e)
    return reports


def get_latest_stress_report() -> dict:
    """Get the most recent stress test report."""
    reports = get_stress_reports()
    return reports[0] if reports else {}


def get_stress_dashboard_data() -> dict:
    """Return all stress test data for the API endpoint."""
    reports = get_stress_reports()
    latest = reports[0] if reports else None

    return {
        "has_data": latest is not None,
        "latest": latest,
        "history": [
            {
                "file": r["_file"],
                "timestamp": r.get("timestamp", ""),
                "score": r.get("composite_stress_score", 0),
                "survived": r.get("scenarios_survived", 0),
                "total": len(r.get("scenarios", [])),
                "worst": r.get("worst_scenario", ""),
                "baseline_pnl": r.get("baseline_pnl", 0),
                "duration": r.get("duration_seconds", 0),
            }
            for r in reports[:20]
        ],
        "total_reports": len(reports),
    }


def run_stress_test(scenarios=None, use_seed=False) -> dict:
    """Run a stress test and return the report."""
    import sys
    sys.path.insert(0, str(ROOT))

    from src.backtest.stress_test import StressTestEngine, _load_fills_from_db, generate_html_report
    from src.backtest.stress_scenarios import StressScenarioSuite

    # Load fills
    fills_raw = []
    if use_seed:
        from scripts.seed_and_replay import seed_database
        seed_database()
        fills_raw = _load_fills_from_db()
    else:
        fills_raw = _load_fills_from_db()

    if not fills_raw:
        return {"error": "No fills found. Seed the database first.", "has_data": False}

    # Run stress test
    engine = StressTestEngine()
    report = engine.run(fills_raw, scenarios=scenarios)

    # Save reports
    os.makedirs(str(REPORTS_DIR), exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_path = str(REPORTS_DIR / f"stress_test_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    html_path = str(REPORTS_DIR / f"stress_test_{ts}.html")
    generate_html_report(report, html_path)

    return {
        "has_data": True,
        "report": report.to_dict(),
        "json_path": json_path,
        "html_path": html_path,
    }


# ─── Dashboard HTML ──────────────────────────────────────────

STRESS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stress Test Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0a0e17; color: #e2e8f0; padding: 20px; max-width: 1200px; margin: 0 auto;
  }
  h1 { font-size: 24px; font-weight: 700; }
  h2 { font-size: 18px; margin: 20px 0 10px; color: #94a3b8; font-weight: 600; }
  .subtitle { color: #64748b; font-size: 13px; margin-bottom: 20px; }
  a { color: #00d4aa; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* Nav */
  .nav { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }
  .nav a {
    border: 1px solid #334155; padding: 6px 14px; border-radius: 6px;
    font-size: 13px; font-weight: 600; color: #94a3b8; transition: all 0.2s;
  }
  .nav a:hover { border-color: #00d4aa; color: #00d4aa; text-decoration: none; }
  .nav a.active { border-color: #00d4aa; color: #00d4aa; background: rgba(0,212,170,0.1); }

  /* Score */
  .score-ring {
    width: 180px; height: 180px; border-radius: 50%; margin: 0 auto 16px;
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    position: relative;
  }
  .score-ring::before {
    content: ''; position: absolute; inset: 0; border-radius: 50%;
    border: 6px solid #1e293b;
  }
  .score-value { font-size: 52px; font-weight: 800; line-height: 1; }
  .score-label { font-size: 14px; font-weight: 600; margin-top: 4px; }
  .score-sub { font-size: 12px; color: #64748b; margin-top: 2px; }

  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card {
    background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; text-align: center;
  }
  .card-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .card-value { font-size: 22px; font-weight: 700; }
  .card-sub { font-size: 11px; color: #475569; margin-top: 4px; }

  /* Scenario cards */
  .scenario {
    background: #111827; border: 1px solid #1e293b; border-radius: 12px;
    padding: 20px; margin-bottom: 14px; transition: border-color 0.2s;
  }
  .scenario:hover { border-color: #334155; }
  .scenario-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .scenario-name { font-size: 16px; font-weight: 700; }
  .badge {
    font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .badge-survived { background: rgba(34,197,94,0.15); color: #22c55e; }
  .badge-damaged { background: rgba(234,179,8,0.15); color: #eab308; }
  .badge-blown { background: rgba(239,68,68,0.15); color: #ef4444; }

  /* Severity bar */
  .severity-track { background: #1e293b; border-radius: 4px; height: 6px; margin: 8px 0 14px; }
  .severity-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

  /* Metrics grid */
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
  .metric {
    text-align: center; padding: 10px; background: #0f172a; border-radius: 8px;
    border: 1px solid #1e293b;
  }
  .metric-label { font-size: 10px; color: #64748b; text-transform: uppercase; }
  .metric-value { font-size: 18px; font-weight: 700; margin-top: 2px; }
  .metric-delta { font-size: 10px; color: #475569; margin-top: 2px; }

  /* Meta tags */
  .meta-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
  .meta-tag {
    font-size: 11px; background: #1e293b; padding: 2px 8px; border-radius: 4px; color: #94a3b8;
  }

  /* History table */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; color: #64748b; border-bottom: 1px solid #1e293b; font-size: 11px; text-transform: uppercase; }
  td { padding: 8px 12px; border-bottom: 1px solid #111827; }
  tr:hover td { background: #111827; }

  /* Buttons */
  .btn {
    padding: 8px 20px; border-radius: 8px; font-size: 13px; font-weight: 600;
    cursor: pointer; border: none; transition: all 0.2s;
  }
  .btn-primary { background: #00d4aa; color: #0a0e17; }
  .btn-primary:hover { background: #00b894; }
  .btn-primary:disabled { background: #334155; color: #64748b; cursor: not-allowed; }
  .btn-secondary { background: transparent; border: 1px solid #334155; color: #94a3b8; }
  .btn-secondary:hover { border-color: #00d4aa; color: #00d4aa; }

  .controls { display: flex; gap: 10px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }

  #loading { display: none; text-align: center; padding: 40px; color: #64748b; }
  #loading .spinner {
    width: 32px; height: 32px; border: 3px solid #1e293b; border-top: 3px solid #00d4aa;
    border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .empty-state { text-align: center; padding: 60px 20px; color: #475569; }
  .empty-state h3 { font-size: 18px; color: #94a3b8; margin-bottom: 8px; }

  .green { color: #22c55e; } .red { color: #ef4444; } .yellow { color: #eab308; }
</style>
</head>
<body>

<!-- Nav -->
<div class="nav">
  <a href="/">DASHBOARD</a>
  <a href="/options">OPTIONS FLOW</a>
  <a href="/backtest">BACKTEST</a>
  <a href="/stress" class="active">STRESS TEST</a>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
  <h1>STRESS TEST PLATFORM</h1>
</div>
<p class="subtitle">Extreme-market scenario testing &mdash; flash crashes, funding squeezes, liquidity drains, cascade liquidations</p>

<!-- Controls -->
<div class="controls">
  <button class="btn btn-primary" id="btn-run" onclick="runStressTest(false)">Run Stress Test</button>
  <button class="btn btn-secondary" id="btn-seed" onclick="runStressTest(true)">Run with Sample Data</button>
  <select id="scenario-filter" style="background:#111827;border:1px solid #334155;color:#e2e8f0;padding:6px 12px;border-radius:6px;font-size:13px">
    <option value="">All Scenarios</option>
    <option value="flash_crash">Flash Crash</option>
    <option value="funding_squeeze">Funding Squeeze</option>
    <option value="liquidity_drain">Liquidity Drain</option>
    <option value="cascade_liquidation">Cascade Liquidation</option>
    <option value="black_swan">Black Swan</option>
  </select>
  <span id="status-msg" style="font-size:12px;color:#64748b"></span>
</div>

<div id="loading">
  <div class="spinner"></div>
  <div>Running stress scenarios...</div>
</div>

<div id="content">
  <div id="empty-state" class="empty-state">
    <h3>No stress test results yet</h3>
    <p>Click "Run Stress Test" to test your bot against extreme market scenarios,<br>
    or "Run with Sample Data" if you don't have fills in the database.</p>
  </div>
</div>

<script>
let currentData = null;

async function loadData() {
  try {
    const res = await fetch('/api/stress');
    currentData = await res.json();
    render(currentData);
  } catch (e) {
    console.error('Failed to load stress data:', e);
  }
}

async function runStressTest(useSeed) {
  const scenario = document.getElementById('scenario-filter').value;
  const body = { use_seed: useSeed };
  if (scenario) body.scenarios = [scenario];

  document.getElementById('btn-run').disabled = true;
  document.getElementById('btn-seed').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('status-msg').textContent = 'Running...';

  try {
    const res = await fetch('/api/stress/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) {
      document.getElementById('status-msg').textContent = data.error;
    } else {
      document.getElementById('status-msg').textContent = 'Complete!';
      await loadData();
    }
  } catch (e) {
    document.getElementById('status-msg').textContent = 'Error: ' + e.message;
  } finally {
    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-seed').disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}

function render(data) {
  const el = document.getElementById('content');
  if (!data || !data.has_data) {
    document.getElementById('empty-state').style.display = 'block';
    return;
  }

  const r = data.latest;
  const score = r.composite_stress_score || 0;
  const scoreColor = score >= 70 ? '#22c55e' : score >= 40 ? '#eab308' : '#ef4444';
  const scoreLabel = score >= 70 ? 'RESILIENT' : score >= 40 ? 'MODERATE' : 'FRAGILE';
  const scenarios = r.scenarios || [];

  let html = '';

  // ─── Score + baseline cards ─────────────────────────────
  html += `
  <div style="display:grid;grid-template-columns:200px 1fr;gap:20px;margin-bottom:24px;align-items:start">
    <div style="text-align:center">
      <div class="score-ring" style="border: 6px solid ${scoreColor}40">
        <div class="score-value" style="color:${scoreColor}">${score.toFixed(0)}</div>
        <div class="score-label" style="color:${scoreColor}">${scoreLabel}</div>
        <div class="score-sub">${r.scenarios_survived}/${scenarios.length} survived</div>
      </div>
      <div style="font-size:11px;color:#475569;margin-top:4px">Resilience Score</div>
    </div>
    <div>
      <div class="cards">
        <div class="card">
          <div class="card-label">Baseline PnL</div>
          <div class="card-value ${r.baseline_pnl >= 0 ? 'green' : 'red'}">$${num(r.baseline_pnl)}</div>
        </div>
        <div class="card">
          <div class="card-label">Baseline DD</div>
          <div class="card-value">${(r.baseline_dd||0).toFixed(1)}%</div>
        </div>
        <div class="card">
          <div class="card-label">Baseline Sharpe</div>
          <div class="card-value">${(r.baseline_sharpe||0).toFixed(3)}</div>
        </div>
        <div class="card">
          <div class="card-label">Baseline Trades</div>
          <div class="card-value">${r.baseline_trades || 0}</div>
        </div>
        <div class="card">
          <div class="card-label">Worst Scenario</div>
          <div class="card-value" style="font-size:14px;color:#ef4444">${r.worst_scenario || 'N/A'}</div>
        </div>
        <div class="card">
          <div class="card-label">Runtime</div>
          <div class="card-value">${(r.duration_seconds||0).toFixed(1)}s</div>
        </div>
      </div>
    </div>
  </div>`;

  // ─── Scenario cards ─────────────────────────────────────
  html += '<h2>Scenario Results</h2>';
  for (const s of scenarios) {
    const survived = s.survived;
    const blown = s.blown;
    const badgeClass = blown ? 'badge-blown' : survived ? 'badge-survived' : 'badge-damaged';
    const badgeText = blown ? 'BLOWN' : survived ? 'SURVIVED' : 'DAMAGED';
    const sevColor = s.severity_score < 30 ? '#22c55e' : s.severity_score < 60 ? '#eab308' : '#ef4444';

    let metaTags = '';
    if (s.scenario_meta) {
      for (const [k, v] of Object.entries(s.scenario_meta)) {
        if (k === 'components') continue;
        metaTags += `<span class="meta-tag">${k}: <b>${v}</b></span>`;
      }
    }

    html += `
    <div class="scenario">
      <div class="scenario-header">
        <span class="scenario-name">${s.scenario_name}</span>
        <span class="badge ${badgeClass}">${badgeText}</span>
      </div>
      <div class="severity-track">
        <div class="severity-fill" style="width:${Math.min(100,s.severity_score)}%;background:${sevColor}"></div>
      </div>
      <div style="text-align:right;font-size:11px;color:#475569;margin-top:-10px;margin-bottom:12px">
        Severity: ${s.severity_score.toFixed(0)}/100
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">PnL</div>
          <div class="metric-value ${s.total_pnl >= 0 ? 'green' : 'red'}">$${num(s.total_pnl)}</div>
          <div class="metric-delta">${num(s.pnl_delta)} vs base</div>
        </div>
        <div class="metric">
          <div class="metric-label">Max DD</div>
          <div class="metric-value red">${(s.max_drawdown_pct||0).toFixed(1)}%</div>
          <div class="metric-delta">${s.dd_delta >= 0 ? '+' : ''}${(s.dd_delta||0).toFixed(1)}%</div>
        </div>
        <div class="metric">
          <div class="metric-label">Win Rate</div>
          <div class="metric-value">${(s.win_rate||0).toFixed(1)}%</div>
          <div class="metric-delta">${s.total_trades} trades</div>
        </div>
        <div class="metric">
          <div class="metric-label">Sharpe</div>
          <div class="metric-value">${(s.sharpe_ratio||0).toFixed(3)}</div>
          <div class="metric-delta">${s.sharpe_delta >= 0 ? '+' : ''}${(s.sharpe_delta||0).toFixed(4)}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Liquidations</div>
          <div class="metric-value ${s.liquidations > 0 ? 'red' : ''}">${s.liquidations}</div>
          <div class="metric-delta">${s.stop_losses_hit} SL hits</div>
        </div>
        <div class="metric">
          <div class="metric-label">Final Balance</div>
          <div class="metric-value">$${num(s.final_balance)}</div>
          <div class="metric-delta">${s.balance_change_pct >= 0 ? '+' : ''}${(s.balance_change_pct||0).toFixed(1)}%</div>
        </div>
      </div>
      ${metaTags ? '<div class="meta-tags">' + metaTags + '</div>' : ''}
    </div>`;
  }

  // ─── History table ──────────────────────────────────────
  if (data.history && data.history.length > 1) {
    html += '<h2>Run History</h2><table><thead><tr>';
    html += '<th>Time</th><th>Score</th><th>Survived</th><th>Worst</th><th>Baseline PnL</th><th>Duration</th>';
    html += '</tr></thead><tbody>';
    for (const h of data.history) {
      const ts = h.timestamp ? h.timestamp.substring(0, 19).replace('T', ' ') : h.file;
      const hScore = h.score || 0;
      const hColor = hScore >= 70 ? '#22c55e' : hScore >= 40 ? '#eab308' : '#ef4444';
      html += `<tr>
        <td>${ts}</td>
        <td style="color:${hColor};font-weight:700">${hScore.toFixed(0)}/100</td>
        <td>${h.survived}/${h.total}</td>
        <td>${h.worst}</td>
        <td class="${h.baseline_pnl >= 0 ? 'green' : 'red'}">$${num(h.baseline_pnl)}</td>
        <td>${h.duration.toFixed(1)}s</td>
      </tr>`;
    }
    html += '</tbody></table>';
  }

  el.innerHTML = html;
}

function num(v) {
  if (v == null) return '0';
  const n = Number(v);
  return (n >= 0 ? '+' : '') + n.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

// Auto-load on page open
loadData();
</script>
</body>
</html>"""
