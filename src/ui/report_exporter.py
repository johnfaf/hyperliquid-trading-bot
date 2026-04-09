"""
Report Exporter (V6)
=====================
Generates exportable performance reports in HTML and CSV formats.
- HTML: visual dashboard with charts (using inline Chart.js)
- CSV: raw data export for spreadsheet analysis
"""
import os
import csv
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import database as db

logger = logging.getLogger(__name__)


def _ensure_reports_dir(output_dir: Optional[str] = None) -> str:
    """Ensure output directory exists."""
    if output_dir is None:
        output_dir = config.REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _calculate_sharpe_ratio(trades: List[Dict]) -> float:
    """Calculate Sharpe ratio from trade history."""
    if not trades or len(trades) < 2:
        return 0.0

    pnls = [t.get("pnl", 0) for t in trades]
    if not pnls:
        return 0.0

    import statistics
    try:
        std_dev = statistics.stdev(pnls)
        if std_dev == 0:
            return 0.0
        avg_pnl = statistics.mean(pnls)
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        return avg_pnl / std_dev if std_dev > 0 else 0.0
    except:
        return 0.0


def _calculate_max_drawdown(closed_trades: List[Dict]) -> float:
    """Calculate max drawdown from closed trades."""
    if not closed_trades:
        return 0.0

    cumulative_pnl = 0
    peak = 0
    max_dd = 0

    for trade in sorted(closed_trades, key=lambda t: t.get("closed_at", "")):
        cumulative_pnl += trade.get("pnl", 0)
        if cumulative_pnl > peak:
            peak = cumulative_pnl
        drawdown = peak - cumulative_pnl
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def _calculate_equity_curve(closed_trades: List[Dict]) -> List[Dict]:
    """Generate equity curve data from cumulative PnL."""
    curve = []
    cumulative = 0
    initial_balance = config.PAPER_TRADING_INITIAL_BALANCE

    for trade in sorted(closed_trades, key=lambda t: t.get("closed_at", "")):
        cumulative += trade.get("pnl", 0)
        curve.append({
            "date": trade.get("closed_at", "")[:10],
            "balance": initial_balance + cumulative
        })

    return curve


def export_html_report(output_dir: str = None) -> str:
    """
    Generates a self-contained HTML file with inline CSS and Chart.js CDN.

    Sections:
    - Portfolio Overview: balance, total PnL, ROI, win rate, Sharpe
    - Equity Curve: line chart of balance over time
    - Strategy Performance Table: active strategies sorted by score
    - Trade History Table: last 50 closed trades
    - Strategy Type Breakdown: pie chart by count
    - Top Traders Tracked: top 20 traders
    - Golden Wallets: summary (if available)
    - Risk Metrics: max drawdown, avg leverage, concentration

    Returns: filepath to generated HTML
    """
    output_dir = _ensure_reports_dir(output_dir)

    # Fetch data
    account = db.get_paper_account()
    strategies = db.get_active_strategies()
    traders = db.get_active_traders()[:20]
    closed_trades = db.get_paper_trade_history(limit=100)
    open_trades = db.get_open_paper_trades()

    # Calculate metrics
    total_pnl = account.get("total_pnl", 0) if account else 0
    balance = account.get("balance", config.PAPER_TRADING_INITIAL_BALANCE) if account else config.PAPER_TRADING_INITIAL_BALANCE
    total_trades = account.get("total_trades", 0) if account else 0
    winning_trades = account.get("winning_trades", 0) if account else 0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    roi = ((balance / config.PAPER_TRADING_INITIAL_BALANCE) - 1) * 100
    sharpe = _calculate_sharpe_ratio(closed_trades)
    max_drawdown = _calculate_max_drawdown(closed_trades)
    equity_curve = _calculate_equity_curve(closed_trades)

    # Strategy type breakdown
    type_counts = {}
    for s in strategies:
        st = s.get("strategy_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1

    # Average leverage from trades
    avg_leverage = 0
    if closed_trades:
        avg_leverage = sum(t.get("leverage", 1) for t in closed_trades) / len(closed_trades)

    # HTML structure
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("  <meta charset='UTF-8'>")
    html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append("  <title>Hyperliquid Trading Bot - Performance Report</title>")
    html.append("  <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>")
    html.append("  <style>")
    html.append("""
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: #e0e0e0;
        line-height: 1.6;
        padding: 20px;
    }

    .container {
        max-width: 1400px;
        margin: 0 auto;
        background: #16213e;
        border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        overflow: hidden;
    }

    header {
        background: linear-gradient(135deg, #1f3a5f 0%, #16213e 100%);
        padding: 30px;
        border-bottom: 2px solid #0f3460;
    }

    h1 {
        font-size: 2.5em;
        color: #00d4ff;
        margin-bottom: 10px;
    }

    .timestamp {
        color: #b0b0b0;
        font-size: 0.9em;
    }

    .content {
        padding: 30px;
    }

    .section {
        margin-bottom: 40px;
    }

    h2 {
        font-size: 1.8em;
        color: #00d4ff;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #0f3460;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    .metric-card {
        background: #0f3460;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
    }

    .metric-card.positive {
        border-left-color: #0dd657;
    }

    .metric-card.negative {
        border-left-color: #ff4444;
    }

    .metric-label {
        color: #a0a0a0;
        font-size: 0.85em;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #00d4ff;
        font-family: 'Courier New', monospace;
    }

    .metric-card.positive .metric-value {
        color: #0dd657;
    }

    .metric-card.negative .metric-value {
        color: #ff4444;
    }

    .chart-container {
        background: #0f3460;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 30px;
        position: relative;
        height: 400px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        background: #0f3460;
        border-radius: 8px;
        overflow: hidden;
    }

    thead {
        background: #1a2847;
    }

    th {
        padding: 15px;
        text-align: left;
        color: #00d4ff;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85em;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #0f3460;
    }

    td {
        padding: 12px 15px;
        border-bottom: 1px solid #0f3460;
        color: #e0e0e0;
    }

    tbody tr:hover {
        background: #1a2847;
    }

    .text-positive {
        color: #0dd657;
    }

    .text-negative {
        color: #ff4444;
    }

    .text-muted {
        color: #a0a0a0;
    }

    footer {
        background: #0a0e27;
        padding: 20px 30px;
        text-align: center;
        color: #a0a0a0;
        font-size: 0.85em;
        border-top: 1px solid #0f3460;
    }

    .empty-state {
        color: #a0a0a0;
        padding: 20px;
        text-align: center;
    }
    """)
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")

    # Header
    html.append("<div class='container'>")
    html.append("  <header>")
    html.append("    <h1>📊 Hyperliquid Trading Bot</h1>")
    html.append("    <p class='timestamp'>Performance Report</p>")
    html.append(f"    <p class='timestamp'>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>")
    html.append("  </header>")

    # Content
    html.append("  <div class='content'>")

    # 1. Portfolio Overview
    html.append("    <div class='section'>")
    html.append("      <h2>Portfolio Overview</h2>")
    html.append("      <div class='metrics-grid'>")

    balance_class = "positive" if total_pnl >= 0 else "negative"
    html.append(f"        <div class='metric-card {balance_class}'>")
    html.append("          <div class='metric-label'>Current Balance</div>")
    html.append(f"          <div class='metric-value'>${balance:,.2f}</div>")
    html.append("        </div>")

    pnl_class = "positive" if total_pnl >= 0 else "negative"
    html.append(f"        <div class='metric-card {pnl_class}'>")
    html.append("          <div class='metric-label'>Total PnL</div>")
    html.append(f"          <div class='metric-value'>${total_pnl:+,.2f}</div>")
    html.append("        </div>")

    roi_class = "positive" if roi >= 0 else "negative"
    html.append(f"        <div class='metric-card {roi_class}'>")
    html.append("          <div class='metric-label'>ROI</div>")
    html.append(f"          <div class='metric-value'>{roi:+.2f}%</div>")
    html.append("        </div>")

    html.append("        <div class='metric-card'>")
    html.append("          <div class='metric-label'>Win Rate</div>")
    html.append(f"          <div class='metric-value'>{win_rate:.1f}%</div>")
    html.append("        </div>")

    html.append("        <div class='metric-card'>")
    html.append("          <div class='metric-label'>Sharpe Ratio</div>")
    html.append(f"          <div class='metric-value'>{sharpe:.3f}</div>")
    html.append("        </div>")

    html.append("        <div class='metric-card'>")
    html.append("          <div class='metric-label'>Total Trades</div>")
    html.append(f"          <div class='metric-value'>{total_trades}</div>")
    html.append("        </div>")

    html.append("      </div>")
    html.append("    </div>")

    # 2. Equity Curve Chart
    if equity_curve:
        html.append("    <div class='section'>")
        html.append("      <h2>Equity Curve</h2>")
        html.append("      <div class='chart-container'>")
        html.append("        <canvas id='equityChart'></canvas>")
        html.append("      </div>")
        html.append("    </div>")

    # 3. Strategy Performance Table
    if strategies:
        html.append("    <div class='section'>")
        html.append("      <h2>Strategy Performance</h2>")
        html.append("      <table>")
        html.append("        <thead>")
        html.append("          <tr>")
        html.append("            <th>Name</th>")
        html.append("            <th>Type</th>")
        html.append("            <th>Score</th>")
        html.append("            <th>PnL</th>")
        html.append("            <th>Win Rate</th>")
        html.append("            <th>Trades</th>")
        html.append("            <th>Sharpe</th>")
        html.append("          </tr>")
        html.append("        </thead>")
        html.append("        <tbody>")
        for s in strategies[:20]:
            pnl_class = "text-positive" if s.get("total_pnl", 0) >= 0 else "text-negative"
            html.append("          <tr>")
            html.append(f"            <td>{s.get('name', 'N/A')[:40]}</td>")
            html.append(f"            <td>{s.get('strategy_type', 'unknown')}</td>")
            html.append(f"            <td>{s.get('current_score', 0):.4f}</td>")
            html.append(f"            <td class='{pnl_class}'>${s.get('total_pnl', 0):+,.2f}</td>")
            html.append(f"            <td>{s.get('win_rate', 0)*100:.1f}%</td>")
            html.append(f"            <td>{s.get('trade_count', 0)}</td>")
            html.append(f"            <td>{s.get('sharpe_ratio', 0):.3f}</td>")
            html.append("          </tr>")
        html.append("        </tbody>")
        html.append("      </table>")
        html.append("    </div>")

    # 4. Trade History Table
    if closed_trades:
        html.append("    <div class='section'>")
        html.append("      <h2>Recent Trade History (Last 50)</h2>")
        html.append("      <table>")
        html.append("        <thead>")
        html.append("          <tr>")
        html.append("            <th>Coin</th>")
        html.append("            <th>Side</th>")
        html.append("            <th>Entry Price</th>")
        html.append("            <th>Exit Price</th>")
        html.append("            <th>PnL</th>")
        html.append("            <th>Leverage</th>")
        html.append("            <th>Duration</th>")
        html.append("          </tr>")
        html.append("        </thead>")
        html.append("        <tbody>")
        for t in closed_trades[:50]:
            pnl_class = "text-positive" if t.get("pnl", 0) >= 0 else "text-negative"
            opened = datetime.fromisoformat(t.get("opened_at", ""))
            closed = datetime.fromisoformat(t.get("closed_at", ""))
            duration = (closed - opened).total_seconds() / 3600
            html.append("          <tr>")
            html.append(f"            <td>{t.get('coin', 'N/A')}</td>")
            html.append(f"            <td>{t.get('side', 'N/A').upper()}</td>")
            html.append(f"            <td>${t.get('entry_price', 0):,.2f}</td>")
            html.append(f"            <td>${t.get('exit_price', 0):,.2f}</td>")
            html.append(f"            <td class='{pnl_class}'>${t.get('pnl', 0):+,.2f}</td>")
            html.append(f"            <td>{t.get('leverage', 1):.1f}x</td>")
            html.append(f"            <td>{duration:.1f}h</td>")
            html.append("          </tr>")
        html.append("        </tbody>")
        html.append("      </table>")
        html.append("    </div>")

    # 5. Strategy Type Breakdown
    if type_counts:
        html.append("    <div class='section'>")
        html.append("      <h2>Strategy Type Distribution</h2>")
        html.append("      <div class='chart-container'>")
        html.append("        <canvas id='typeChart'></canvas>")
        html.append("      </div>")
        html.append("    </div>")

    # 6. Top Traders
    if traders:
        html.append("    <div class='section'>")
        html.append("      <h2>Top Traders Tracked</h2>")
        html.append("      <table>")
        html.append("        <thead>")
        html.append("          <tr>")
        html.append("            <th>Rank</th>")
        html.append("            <th>Address</th>")
        html.append("            <th>Account Value</th>")
        html.append("            <th>Total PnL</th>")
        html.append("            <th>Win Rate</th>")
        html.append("            <th>Trade Count</th>")
        html.append("          </tr>")
        html.append("        </thead>")
        html.append("        <tbody>")
        for i, t in enumerate(traders, 1):
            addr = f"{t['address'][:6]}...{t['address'][-4:]}"
            pnl_class = "text-positive" if t.get("total_pnl", 0) >= 0 else "text-negative"
            html.append("          <tr>")
            html.append(f"            <td>{i}</td>")
            html.append(f"            <td><code>{addr}</code></td>")
            html.append(f"            <td>${t.get('account_value', 0):,.0f}</td>")
            html.append(f"            <td class='{pnl_class}'>${t.get('total_pnl', 0):+,.0f}</td>")
            html.append(f"            <td>{t.get('win_rate', 0)*100:.1f}%</td>")
            html.append(f"            <td>{t.get('trade_count', 0)}</td>")
            html.append("          </tr>")
        html.append("        </tbody>")
        html.append("      </table>")
        html.append("    </div>")

    # 7. Risk Metrics
    html.append("    <div class='section'>")
    html.append("      <h2>Risk Metrics</h2>")
    html.append("      <div class='metrics-grid'>")

    html.append("        <div class='metric-card'>")
    html.append("          <div class='metric-label'>Max Drawdown</div>")
    html.append(f"          <div class='metric-value'>${max_drawdown:,.2f}</div>")
    html.append("        </div>")

    html.append("        <div class='metric-card'>")
    html.append("          <div class='metric-label'>Avg Leverage</div>")
    html.append(f"          <div class='metric-value'>{avg_leverage:.2f}x</div>")
    html.append("        </div>")

    if strategies:
        top_strategy_pnl = strategies[0].get("total_pnl", 0)
        total_strategy_pnl = sum(s.get("total_pnl", 0) for s in strategies)
        concentration = (top_strategy_pnl / total_strategy_pnl * 100) if total_strategy_pnl != 0 else 0
        html.append("        <div class='metric-card'>")
        html.append("          <div class='metric-label'>Strategy Concentration</div>")
        html.append(f"          <div class='metric-value'>{concentration:.1f}%</div>")
        html.append("        </div>")

    html.append("      </div>")
    html.append("    </div>")

    html.append("  </div>")
    html.append("</div>")

    # JavaScript for charts
    html.append("<script>")

    # Equity Curve Chart
    if equity_curve:
        dates = [item["date"] for item in equity_curve]
        balances = [item["balance"] for item in equity_curve]
        html.append("const equityCtx = document.getElementById('equityChart').getContext('2d');")
        html.append("new Chart(equityCtx, {")
        html.append("  type: 'line',")
        html.append(f"  data: {{")
        html.append(f"    labels: {json.dumps(dates)},")
        html.append(f"    datasets: [{{")
        html.append(f"      label: 'Account Balance',")
        html.append(f"      data: {json.dumps(balances)},")
        html.append(f"      borderColor: '#00d4ff',")
        html.append(f"      backgroundColor: 'rgba(0, 212, 255, 0.1)',")
        html.append(f"      borderWidth: 2,")
        html.append(f"      tension: 0.4,")
        html.append(f"      fill: true")
        html.append(f"    }}]")
        html.append(f"  }},")
        html.append(f"  options: {{")
        html.append(f"    responsive: true,")
        html.append(f"    maintainAspectRatio: false,")
        html.append(f"    plugins: {{")
        html.append(f"      legend: {{ labels: {{ color: '#a0a0a0' }} }}")
        html.append(f"    }},")
        html.append(f"    scales: {{")
        html.append(f"      y: {{ ticks: {{ color: '#a0a0a0' }} }},")
        html.append(f"      x: {{ ticks: {{ color: '#a0a0a0' }} }}")
        html.append(f"    }}")
        html.append(f"  }}")
        html.append("});")

    # Strategy Type Chart
    if type_counts:
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = ["#00d4ff", "#0dd657", "#ff4444", "#ffd700", "#00ff88", "#ff00ff"]
        html.append("const typeCtx = document.getElementById('typeChart').getContext('2d');")
        html.append("new Chart(typeCtx, {")
        html.append("  type: 'doughnut',")
        html.append(f"  data: {{")
        html.append(f"    labels: {json.dumps(types)},")
        html.append(f"    datasets: [{{")
        html.append(f"      data: {json.dumps(counts)},")
        html.append(f"      backgroundColor: {json.dumps(colors[:len(types)])}")
        html.append(f"    }}]")
        html.append(f"  }},")
        html.append(f"  options: {{")
        html.append(f"    responsive: true,")
        html.append(f"    maintainAspectRatio: false,")
        html.append(f"    plugins: {{")
        html.append(f"      legend: {{ labels: {{ color: '#a0a0a0' }} }}")
        html.append(f"    }}")
        html.append(f"  }}")
        html.append("});")

    html.append("</script>")

    # Footer
    html.append("<footer>")
    html.append("  <p>Hyperliquid Trading Research Bot &copy; 2024-2025</p>")
    html.append("</footer>")

    html.append("</body>")
    html.append("</html>")

    # Write file
    filename = f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write("\n".join(html))

    logger.info(f"HTML report exported to {filepath}")
    return filepath


def export_csv_trades(output_dir: str = None) -> str:
    """
    Exports all closed trades as CSV.

    Columns: id, strategy_type, coin, side, entry_price, exit_price, pnl,
             leverage, opened_at, closed_at, duration_hours

    Returns: filepath to generated CSV
    """
    output_dir = _ensure_reports_dir(output_dir)

    closed_trades = db.get_paper_trade_history(limit=10000)

    filename = f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="") as csvfile:
        fieldnames = [
            "id", "strategy_type", "coin", "side", "entry_price",
            "exit_price", "pnl", "leverage", "opened_at", "closed_at", "duration_hours"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for trade in closed_trades:
            strategy_id = trade.get("strategy_id")
            strategy_type = "N/A"
            if strategy_id:
                strategy = db.get_strategy(strategy_id)
                if strategy:
                    strategy_type = strategy.get("strategy_type", "N/A")

            opened = datetime.fromisoformat(trade.get("opened_at", ""))
            closed = datetime.fromisoformat(trade.get("closed_at", ""))
            duration = (closed - opened).total_seconds() / 3600

            writer.writerow({
                "id": trade.get("id", ""),
                "strategy_type": strategy_type,
                "coin": trade.get("coin", ""),
                "side": trade.get("side", ""),
                "entry_price": trade.get("entry_price", 0),
                "exit_price": trade.get("exit_price", 0),
                "pnl": trade.get("pnl", 0),
                "leverage": trade.get("leverage", 1),
                "opened_at": trade.get("opened_at", ""),
                "closed_at": trade.get("closed_at", ""),
                "duration_hours": f"{duration:.2f}"
            })

    logger.info(f"Trades CSV exported to {filepath}")
    return filepath


def export_csv_strategies(output_dir: str = None) -> str:
    """
    Exports all strategies as CSV.

    Columns: id, name, type, score, pnl, win_rate, trade_count, sharpe,
             discovered_at, active

    Returns: filepath to generated CSV
    """
    output_dir = _ensure_reports_dir(output_dir)

    # Get both active and inactive strategies
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY current_score DESC"
        ).fetchall()
    strategies = [dict(r) for r in rows]

    filename = f"strategies_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="") as csvfile:
        fieldnames = [
            "id", "name", "type", "score", "pnl", "win_rate",
            "trade_count", "sharpe", "discovered_at", "active"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for s in strategies:
            writer.writerow({
                "id": s.get("id", ""),
                "name": s.get("name", ""),
                "type": s.get("strategy_type", ""),
                "score": f"{s.get('current_score', 0):.6f}",
                "pnl": f"{s.get('total_pnl', 0):.2f}",
                "win_rate": f"{s.get('win_rate', 0):.4f}",
                "trade_count": s.get("trade_count", 0),
                "sharpe": f"{s.get('sharpe_ratio', 0):.6f}",
                "discovered_at": s.get("discovered_at", ""),
                "active": "Yes" if s.get("active", 0) else "No"
            })

    logger.info(f"Strategies CSV exported to {filepath}")
    return filepath


def export_csv_traders(output_dir: str = None) -> str:
    """
    Exports all tracked traders as CSV.

    Columns: address, account_value, total_pnl, roi_pct, win_rate,
             trade_count, first_seen, last_updated, active

    Returns: filepath to generated CSV
    """
    output_dir = _ensure_reports_dir(output_dir)

    traders = db.get_all_traders_including_bots()

    filename = f"traders_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="") as csvfile:
        fieldnames = [
            "address", "account_value", "total_pnl", "roi_pct", "win_rate",
            "trade_count", "first_seen", "last_updated", "active"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for t in traders:
            writer.writerow({
                "address": t.get("address", ""),
                "account_value": f"{t.get('account_value', 0):.2f}",
                "total_pnl": f"{t.get('total_pnl', 0):.2f}",
                "roi_pct": f"{t.get('roi_pct', 0):.4f}",
                "win_rate": f"{t.get('win_rate', 0):.4f}",
                "trade_count": t.get("trade_count", 0),
                "first_seen": t.get("first_seen", ""),
                "last_updated": t.get("last_updated", ""),
                "active": "Yes" if t.get("active", 1) else "No"
            })

    logger.info(f"Traders CSV exported to {filepath}")
    return filepath


def export_full_package(output_dir: str = None) -> Dict[str, str]:
    """
    Calls all export functions and returns a dictionary of filepaths.

    Returns: {
        "html": "/path/to/report.html",
        "trades_csv": "/path/to/trades.csv",
        "strategies_csv": "/path/to/strategies.csv",
        "traders_csv": "/path/to/traders.csv"
    }
    """
    output_dir = _ensure_reports_dir(output_dir)

    result = {
        "html": export_html_report(output_dir),
        "trades_csv": export_csv_trades(output_dir),
        "strategies_csv": export_csv_strategies(output_dir),
        "traders_csv": export_csv_traders(output_dir),
    }

    logger.info(f"Full export package completed: {result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()

    print("Generating full report package...")
    filepaths = export_full_package()

    for report_type, filepath in filepaths.items():
        print(f"  {report_type}: {filepath}")
