"""
Reporting Module
Generates human-readable reports on trader research, strategy performance,
paper trading results, and self-improvement metrics.
"""
import logging
import os
import json
from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import database as db

logger = logging.getLogger(__name__)


class Reporter:
    """Generates reports and summaries."""

    def __init__(self):
        os.makedirs(config.REPORTS_DIR, exist_ok=True)

    def generate_daily_report(self) -> str:
        """Generate a comprehensive daily report as markdown."""
        now = datetime.now(timezone.utc)
        report = []
        report.append("# Hyperliquid Research Bot - Daily Report")
        report.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}\n")

        # 1. Trader Overview
        report.append("## Tracked Traders")
        traders = db.get_active_traders()
        report.append(f"Currently tracking **{len(traders)}** traders.\n")

        if traders:
            report.append("| Rank | Address | Account Value | Total PnL | Win Rate | Trades |")
            report.append("|------|---------|--------------|-----------|----------|--------|")
            for i, t in enumerate(traders[:20], 1):
                addr = f"{t['address'][:6]}...{t['address'][-4:]}"
                report.append(
                    f"| {i} | `{addr}` | ${t['account_value']:,.0f} | "
                    f"${t['total_pnl']:,.0f} | {t['win_rate']*100:.1f}% | {t['trade_count']} |"
                )
        report.append("")

        # 2. Strategy Rankings
        report.append("## Strategy Rankings")
        strategies = db.get_active_strategies()
        report.append(f"**{len(strategies)}** active strategies being tracked.\n")

        if strategies:
            report.append("| Rank | Strategy | Type | Score | PnL | Win Rate |")
            report.append("|------|----------|------|-------|-----|----------|")
            for i, s in enumerate(strategies[:15], 1):
                report.append(
                    f"| {i} | {s['name'][:30]} | {s['strategy_type']} | "
                    f"{s['current_score']:.3f} | ${s['total_pnl']:,.0f} | {s['win_rate']*100:.1f}% |"
                )
        report.append("")

        # 3. Paper Trading Performance
        report.append("## Paper Trading Performance")
        account = db.get_paper_account()
        if account:
            roi = ((account['balance'] / config.PAPER_TRADING_INITIAL_BALANCE) - 1) * 100
            wr = (account['winning_trades'] / account['total_trades'] * 100
                  if account['total_trades'] > 0 else 0)
            report.append(f"- **Balance:** ${account['balance']:,.2f}")
            report.append(f"- **Total PnL:** ${account['total_pnl']:,.2f}")
            report.append(f"- **ROI:** {roi:+.2f}%")
            report.append(f"- **Total Trades:** {account['total_trades']}")
            report.append(f"- **Win Rate:** {wr:.1f}%")
        else:
            report.append("Paper trading not yet initialized.")
        report.append("")

        # Open positions
        open_trades = db.get_open_paper_trades()
        if open_trades:
            report.append("### Open Positions")
            report.append("| Coin | Side | Entry | Size | Leverage | SL | TP |")
            report.append("|------|------|-------|------|----------|----|----|")
            for t in open_trades:
                report.append(
                    f"| {t['coin']} | {t['side'].upper()} | ${t['entry_price']:,.2f} | "
                    f"{t['size']:.4f} | {t['leverage']}x | ${t['stop_loss'] or 0:,.2f} | "
                    f"${t['take_profit'] or 0:,.2f} |"
                )
            report.append("")

        # Recent closed trades
        closed = db.get_paper_trade_history(limit=10)
        if closed:
            report.append("### Recent Closed Trades")
            report.append("| Coin | Side | Entry | Exit | PnL | Leverage |")
            report.append("|------|------|-------|------|-----|----------|")
            for t in closed:
                pnl_str = f"${t['pnl']:+,.2f}" if t['pnl'] else "$0"
                report.append(
                    f"| {t['coin']} | {t['side'].upper()} | ${t['entry_price']:,.2f} | "
                    f"${t['exit_price'] or 0:,.2f} | {pnl_str} | {t['leverage']}x |"
                )
            report.append("")

        # 4. Strategy type distribution
        report.append("## Strategy Type Distribution")
        if strategies:
            type_counts = {}
            type_avg_scores = {}
            for s in strategies:
                st = s["strategy_type"]
                type_counts[st] = type_counts.get(st, 0) + 1
                type_avg_scores.setdefault(st, []).append(s["current_score"])

            report.append("| Strategy Type | Count | Avg Score |")
            report.append("|--------------|-------|-----------|")
            for st, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                avg = sum(type_avg_scores[st]) / len(type_avg_scores[st])
                report.append(f"| {st} | {count} | {avg:.3f} |")
        report.append("")

        # 5. Self-improvement metrics
        report.append("## Self-Improvement Metrics")
        report.append(
            "The bot tracks strategy scores over time. Strategies that consistently "
            "perform well get higher scores, while underperformers are automatically "
            "deactivated."
        )
        report.append(f"- **Score decay rate:** {config.SCORE_DECAY_RATE} per day")
        report.append(f"- **Minimum active score:** {config.MIN_STRATEGY_SCORE}")
        report.append("")

        report_text = "\n".join(report)

        # Save to file
        filename = f"report_{now.strftime('%Y%m%d_%H%M')}.md"
        filepath = os.path.join(config.REPORTS_DIR, filename)
        with open(filepath, "w") as f:
            f.write(report_text)

        logger.info(f"Daily report saved to {filepath}")
        return report_text

    def generate_strategy_detail_report(self, strategy_id: int) -> str:
        """Generate detailed report for a single strategy."""
        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return "Strategy not found."

        scores = db.get_strategy_score_history(strategy_id, limit=30)
        report = []
        report.append(f"# Strategy Detail: {strategy['name']}")
        report.append(f"**Type:** {strategy['strategy_type']}")
        report.append(f"**Discovered:** {strategy['discovered_at']}")
        report.append(f"**Current Score:** {strategy['current_score']:.4f}")
        report.append(f"**Active:** {'Yes' if strategy['active'] else 'No'}")
        report.append("")
        report.append(f"**Total PnL:** ${strategy['total_pnl']:,.2f}")
        report.append(f"**Win Rate:** {strategy['win_rate']*100:.1f}%")
        report.append(f"**Trade Count:** {strategy['trade_count']}")
        report.append("")

        if strategy['parameters']:
            try:
                params = json.loads(strategy['parameters']) if isinstance(strategy['parameters'], str) else strategy['parameters']
                report.append("## Parameters")
                for k, v in params.items():
                    report.append(f"- **{k}:** {v}")
                report.append("")
            except (json.JSONDecodeError, TypeError):
                pass

        if scores:
            report.append("## Score History (last 30 entries)")
            report.append("| Date | Score | PnL | Win Rate | Sharpe | Consistency | Risk Adj |")
            report.append("|------|-------|-----|----------|--------|-------------|----------|")
            for s in scores[:30]:
                report.append(
                    f"| {s['timestamp'][:16]} | {s['score']:.4f} | {s['pnl_score']:.3f} | "
                    f"{s['win_rate_score']:.3f} | {s['sharpe_score']:.3f} | "
                    f"{s['consistency_score']:.3f} | {s['risk_adj_score']:.3f} |"
                )
        report.append("")

        return "\n".join(report)

    def print_live_status(self) -> str:
        """Print a compact live status summary for terminal output."""
        account = db.get_paper_account()
        traders = db.get_active_traders()
        strategies = db.get_active_strategies()
        open_trades = db.get_open_paper_trades()

        lines = []
        lines.append("=" * 60)
        lines.append("  HYPERLIQUID RESEARCH BOT - LIVE STATUS")
        lines.append(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 60)

        if account:
            roi = ((account['balance'] / config.PAPER_TRADING_INITIAL_BALANCE) - 1) * 100
            lines.append(f"  Paper Balance: ${account['balance']:,.2f} ({roi:+.1f}% ROI)")
            lines.append(f"  Paper PnL:     ${account['total_pnl']:,.2f}")

        lines.append(f"  Tracked Traders:    {len(traders)}")
        lines.append(f"  Active Strategies:  {len(strategies)}")
        lines.append(f"  Open Positions:     {len(open_trades)}")

        if strategies:
            top = strategies[0]
            lines.append(f"  Top Strategy:       {top['strategy_type']} (score={top['current_score']:.3f})")

        lines.append("=" * 60)

        status = "\n".join(lines)
        return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    reporter = Reporter()
    print(reporter.print_live_status())
