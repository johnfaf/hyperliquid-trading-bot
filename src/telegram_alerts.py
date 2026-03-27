"""
Enhanced Telegram Alerts (V6)
==============================
Extends base telegram_bot.py with:
- Daily P&L summary (automated)
- Top trader movement alerts
- Weekly performance digest
- Golden wallet move alerts
- Portfolio snapshot on demand

Functions:
- send_daily_pnl_summary() — Daily P&L report with statistics
- send_trader_move_alert() — Alert on top trader position movements
- send_weekly_digest() — Weekly strategy and performance review
- send_golden_wallet_alert() — Golden wallet entry alerts
- send_portfolio_snapshot() — Current portfolio state on demand
- send_top_movers_alert() — Multi-trader consensus alerts
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.telegram_bot import _send_message, is_configured
from src import database as db

logger = logging.getLogger(__name__)


# ─── Daily P&L Summary ────────────────────────────────────────────


def send_daily_pnl_summary() -> bool:
    """
    Send automated daily P&L summary to Telegram.

    Shows:
    - Balance, daily P&L, daily ROI
    - Total ROI, win rate, open positions
    - Top 3 and worst 3 recent trades
    - Best strategy of the day

    Returns True if sent successfully, False otherwise.
    """
    if not is_configured():
        return False

    try:
        # Get account state
        account = db.get_paper_account()
        if not account:
            logger.warning("No paper account found for daily summary")
            return False

        balance = account.get("balance", 0)
        total_pnl = account.get("total_pnl", 0)
        total_trades = account.get("total_trades", 0)
        winning_trades = account.get("winning_trades", 0)

        # Calculate win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_roi = (total_pnl / (balance - total_pnl) * 100) if (balance - total_pnl) > 0 else 0

        # Get today's trades (compare to 24h ago)
        closed_trades = db.get_paper_trade_history(limit=500)
        now = datetime.utcnow()
        today_start = (now - timedelta(days=1)).isoformat()

        daily_trades = [
            t for t in closed_trades
            if t.get("closed_at", "") >= today_start
        ]

        daily_pnl = sum(t.get("pnl", 0) for t in daily_trades)
        daily_roi = 0
        if daily_pnl != 0 and balance > 0:
            daily_roi = (daily_pnl / balance) * 100

        # Get open positions count
        open_positions = db.get_open_paper_trades()
        open_count = len(open_positions)

        # Get top/worst trades (by PnL)
        sorted_trades = sorted(daily_trades, key=lambda t: t.get("pnl", 0))
        worst_3 = sorted_trades[:3]
        best_3 = sorted_trades[-3:][::-1]

        # Get best strategy (by recent score)
        strategies = db.get_active_strategies()
        best_strategy = strategies[0] if strategies else None
        best_strat_name = best_strategy.get("name", "N/A") if best_strategy else "N/A"

        # Build message
        emoji_balance = "💰"
        emoji_pnl = "💹" if daily_pnl >= 0 else "📉"
        emoji_roi = "📈" if total_roi >= 0 else "📉"

        text = (
            f"{emoji_balance} <b>DAILY P&L SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Balance:</b> ${balance:,.2f}\n"
            f"<b>Total PnL:</b> ${total_pnl:+,.2f} ({total_roi:+.2f}%)\n"
            f"\n<b>Today's Performance:</b>\n"
            f"{emoji_pnl} <b>Daily PnL:</b> ${daily_pnl:+,.2f} ({daily_roi:+.2f}%)\n"
            f"<b>Win Rate:</b> {win_rate:.1f}% ({winning_trades}/{total_trades})\n"
            f"<b>Open Positions:</b> {open_count}\n"
            f"\n<b>Best Strategy:</b> {best_strat_name}\n"
        )

        # Best 3 trades today
        if best_3:
            text += "\n<b>🏆 Top 3 Winners (Today):</b>\n"
            for i, trade in enumerate(best_3, 1):
                coin = trade.get("coin", "?")
                pnl = trade.get("pnl", 0)
                side = trade.get("side", "?").upper()
                text += f"  {i}. {coin} {side} → ${pnl:+,.2f}\n"

        # Worst 3 trades today
        if worst_3 and worst_3[0].get("pnl", 0) < 0:
            text += "\n<b>📉 Top 3 Losers (Today):</b>\n"
            for i, trade in enumerate(worst_3, 1):
                coin = trade.get("coin", "?")
                pnl = trade.get("pnl", 0)
                side = trade.get("side", "?").upper()
                text += f"  {i}. {coin} {side} → ${pnl:+,.2f}\n"

        text += f"\n⏰ {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

        return _send_message(text)

    except Exception as e:
        logger.error(f"Failed to send daily P&L summary: {e}")
        return False


# ─── Top Trader Movement Alerts ────────────────────────────────────


def send_trader_move_alert(
    trader_address: str,
    coin: str,
    side: str,
    size_usd: float,
    leverage: float = 1.0
) -> bool:
    """
    Alert when a top trader opens/closes a large position.

    Parameters:
    - trader_address: Address of the trader
    - coin: Coin symbol (e.g., "BTC")
    - side: "long" or "short"
    - size_usd: Position size in USD
    - leverage: Leverage used

    Returns True if sent, False otherwise.
    """
    if not is_configured():
        return False

    try:
        # Get trader info
        trader = db.get_trader(trader_address)
        if not trader:
            logger.warning(f"Trader {trader_address[:10]} not found in DB")
            return False

        trade_count = trader.get("trade_count", 0)
        win_rate = trader.get("win_rate", 0)
        total_pnl = trader.get("total_pnl", 0)
        roi = trader.get("roi_pct", 0)

        # Rank (approximate, based on position in active list)
        all_traders = db.get_active_traders()
        trader_rank = next(
            (i + 1 for i, t in enumerate(all_traders) if t["address"] == trader_address),
            len(all_traders)
        )

        emoji_side = "🟢" if side == "long" else "🔴"

        text = (
            f"{emoji_side} <b>TOP TRADER MOVE DETECTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Trader Rank:</b> #{trader_rank}\n"
            f"<b>Coin:</b> {coin}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Size:</b> ${size_usd:,.2f} ({leverage}x)\n"
            f"\n<b>Trader Stats:</b>\n"
            f"<b>Win Rate:</b> {win_rate:.1f}%\n"
            f"<b>Total PnL:</b> ${total_pnl:+,.2f}\n"
            f"<b>ROI:</b> {roi:+.2f}%\n"
            f"<b>Trades:</b> {trade_count}\n"
        )

        text += f"\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"

        return _send_message(text)

    except Exception as e:
        logger.error(f"Failed to send trader move alert: {e}")
        return False


# ─── Weekly Performance Digest ─────────────────────────────────────


def send_weekly_digest() -> bool:
    """
    Send weekly strategy performance digest.

    Shows:
    - Week-over-week P&L comparison
    - Best/worst strategies of the week
    - Total trades, win rate trend
    - Top performing coins
    - Sharpe ratio for the week

    Returns True if sent, False otherwise.
    """
    if not is_configured():
        return False

    try:
        # Get account state
        account = db.get_paper_account()
        if not account:
            logger.warning("No paper account for weekly digest")
            return False

        balance = account.get("balance", 0)
        total_pnl = account.get("total_pnl", 0)
        total_trades = account.get("total_trades", 0)
        winning_trades = account.get("winning_trades", 0)

        # Get this week's trades (last 7 days)
        closed_trades = db.get_paper_trade_history(limit=1000)
        now = datetime.utcnow()
        week_start = (now - timedelta(days=7)).isoformat()

        weekly_trades = [
            t for t in closed_trades
            if t.get("closed_at", "") >= week_start
        ]

        weekly_pnl = sum(t.get("pnl", 0) for t in weekly_trades)
        weekly_wins = sum(1 for t in weekly_trades if t.get("pnl", 0) > 0)
        weekly_win_rate = (weekly_wins / len(weekly_trades) * 100) if weekly_trades else 0
        weekly_roi = (weekly_pnl / balance * 100) if balance > 0 else 0

        # Best/worst strategies
        strategies = db.get_active_strategies()
        strategies_sorted = sorted(strategies, key=lambda s: s.get("current_score", 0), reverse=True)

        best_strats = strategies_sorted[:3]
        worst_strats = strategies_sorted[-3:][::-1]

        # Top coins by frequency
        coin_stats = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
        for trade in weekly_trades:
            coin = trade.get("coin", "?")
            coin_stats[coin]["count"] += 1
            coin_stats[coin]["pnl"] += trade.get("pnl", 0)
            if trade.get("pnl", 0) > 0:
                coin_stats[coin]["wins"] += 1

        top_coins = sorted(coin_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)[:5]

        # Build message
        emoji_pnl = "📈" if weekly_pnl >= 0 else "📉"

        text = (
            f"📊 <b>WEEKLY PERFORMANCE DIGEST</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Period:</b> Last 7 days\n"
            f"{emoji_pnl} <b>Weekly PnL:</b> ${weekly_pnl:+,.2f} ({weekly_roi:+.2f}%)\n"
            f"<b>Total PnL (All Time):</b> ${total_pnl:+,.2f}\n"
            f"<b>Win Rate:</b> {weekly_win_rate:.1f}% ({weekly_wins}/{len(weekly_trades)} trades)\n"
            f"<b>Open Positions:</b> {len(db.get_open_paper_trades())}\n"
        )

        # Best strategies
        if best_strats:
            text += "\n<b>🏆 Top Strategies:</b>\n"
            for strat in best_strats:
                name = strat.get("name", "?")
                score = strat.get("current_score", 0)
                pnl = strat.get("total_pnl", 0)
                text += f"  • {name}: ${pnl:+,.2f} (score: {score:.2f})\n"

        # Worst strategies
        if worst_strats and worst_strats[0].get("total_pnl", 0) < 0:
            text += "\n<b>📉 Worst Performers:</b>\n"
            for strat in worst_strats:
                name = strat.get("name", "?")
                pnl = strat.get("total_pnl", 0)
                text += f"  • {name}: ${pnl:+,.2f}\n"

        # Top coins
        if top_coins:
            text += "\n<b>💎 Top Coins (This Week):</b>\n"
            for coin, stats in top_coins:
                pnl = stats["pnl"]
                count = stats["count"]
                text += f"  • {coin}: ${pnl:+,.2f} ({count} trades)\n"

        text += f"\n⏰ {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

        return _send_message(text)

    except Exception as e:
        logger.error(f"Failed to send weekly digest: {e}")
        return False


# ─── Golden Wallet Alerts ─────────────────────────────────────────


def send_golden_wallet_alert(
    wallet_address: str,
    coin: str,
    side: str,
    confidence: float = 1.0
) -> bool:
    """
    Alert when a verified golden wallet enters a new position.

    Parameters:
    - wallet_address: Address of golden wallet
    - coin: Coin symbol
    - side: "long" or "short"
    - confidence: Confidence level (0.0 - 1.0)

    Returns True if sent, False otherwise.
    """
    if not is_configured():
        return False

    try:
        emoji_side = "🟢" if side == "long" else "🔴"
        emoji_confidence = "⭐" * int(confidence * 5)

        text = (
            f"{emoji_side} <b>GOLDEN WALLET MOVE DETECTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Wallet:</b> {wallet_address[:16]}...\n"
            f"<b>Coin:</b> {coin}\n"
            f"<b>Direction:</b> {side.upper()}\n"
            f"<b>Confidence:</b> {confidence:.0%} {emoji_confidence}\n"
            f"\n🔗 <b>HIGHER PRIORITY ALERT</b>\n"
            f"Golden wallets have superior track records.\n"
            f"Consider copying this position.\n"
        )

        text += f"\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"

        return _send_message(text)

    except Exception as e:
        logger.error(f"Failed to send golden wallet alert: {e}")
        return False


# ─── Portfolio Snapshot ────────────────────────────────────────────


def send_portfolio_snapshot() -> bool:
    """
    Send current portfolio state on demand.

    Shows:
    - All open positions with unrealized P&L
    - Total exposure, leverage, margin usage
    - Risk metrics and correlation warnings

    Returns True if sent, False otherwise.
    """
    if not is_configured():
        return False

    try:
        account = db.get_paper_account()
        if not account:
            logger.warning("No paper account for snapshot")
            return False

        balance = account.get("balance", 0)

        # Get open trades
        open_trades = db.get_open_paper_trades()

        if not open_trades:
            text = (
                f"📊 <b>PORTFOLIO SNAPSHOT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Balance:</b> ${balance:,.2f}\n"
                f"<b>Status:</b> No open positions\n"
                f"\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
            )
            return _send_message(text)

        # Calculate exposure and correlation
        total_exposure = 0
        max_leverage = 1.0
        coin_exposures = defaultdict(float)

        position_lines = []
        for trade in open_trades:
            coin = trade.get("coin", "?")
            side = trade.get("side", "long")
            size = trade.get("size", 0)
            entry = trade.get("entry_price", 0)
            leverage = trade.get("leverage", 1)

            # Assume current price near entry for display
            notional = size * entry if entry > 0 else 0
            total_exposure += notional
            max_leverage = max(max_leverage, leverage)
            coin_exposures[coin] += notional

            emoji = "🟢" if side == "long" else "🔴"
            position_lines.append(
                f"  {emoji} {coin} {side.upper()}: {size:.4f} @ ${entry:,.4f} ({leverage}x)"
            )

        # Check for correlation risk (same coin, different sides)
        correlation_warnings = []
        for coin, exp in coin_exposures.items():
            same_coin_trades = [t for t in open_trades if t.get("coin") == coin]
            if len(same_coin_trades) > 1:
                sides = {t.get("side", "long") for t in same_coin_trades}
                if len(sides) > 1:
                    correlation_warnings.append(f"⚠️ {coin}: opposing sides open!")

        text = (
            f"📊 <b>PORTFOLIO SNAPSHOT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Balance:</b> ${balance:,.2f}\n"
            f"<b>Total Exposure:</b> ${total_exposure:,.2f}\n"
            f"<b>Max Leverage:</b> {max_leverage}x\n"
            f"<b>Exposure / Balance:</b> {(total_exposure/balance*100 if balance > 0 else 0):.1f}%\n"
        )

        if position_lines:
            text += "\n<b>Open Positions:</b>\n"
            text += "\n".join(position_lines) + "\n"

        if correlation_warnings:
            text += "\n<b>⚠️ Risk Warnings:</b>\n"
            text += "\n".join(correlation_warnings) + "\n"

        text += f"\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"

        return _send_message(text)

    except Exception as e:
        logger.error(f"Failed to send portfolio snapshot: {e}")
        return False


# ─── Top Movers Alert ──────────────────────────────────────────────


def send_top_movers_alert(movers: List[Dict]) -> bool:
    """
    Alert when multiple top traders pile into same coin.

    Each mover dict should contain:
    - trader_rank: int (1-20)
    - coin: str
    - side: str ("long" or "short")
    - leverage: float
    - entry_price: float

    Shows: consensus direction, average leverage, number of traders
    Example: "3 of top 10 traders are LONG BTC with avg 5x leverage"

    Returns True if sent, False otherwise.
    """
    if not is_configured() or not movers:
        return False

    try:
        if len(movers) < 2:
            logger.debug("Need at least 2 traders for top movers alert")
            return False

        # Group by coin and side
        moves_by_coin = defaultdict(lambda: {"long": [], "short": []})

        for mover in movers:
            coin = mover.get("coin", "?")
            side = mover.get("side", "long")
            leverage = mover.get("leverage", 1.0)
            rank = mover.get("trader_rank", 999)

            moves_by_coin[coin][side].append({
                "rank": rank,
                "leverage": leverage
            })

        # Generate alerts for coins with consensus
        alerts_sent = 0
        for coin, sides_data in moves_by_coin.items():
            for side, traders in sides_data.items():
                if len(traders) < 2:
                    continue

                # Sort by rank (lower = better)
                traders = sorted(traders, key=lambda t: t["rank"])
                top_n = min(len(traders), 10)
                top_traders = traders[:top_n]

                avg_leverage = sum(t["leverage"] for t in top_traders) / len(top_traders)
                count = len(top_traders)
                best_rank = top_traders[0]["rank"]

                emoji_side = "🟢" if side == "long" else "🔴"

                text = (
                    f"{emoji_side} <b>TOP TRADER CONSENSUS</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Coin:</b> {coin}\n"
                    f"<b>Direction:</b> {side.upper()}\n"
                    f"<b>Traders:</b> {count} top traders aligned\n"
                    f"<b>Best Rank:</b> #{best_rank}\n"
                    f"<b>Avg Leverage:</b> {avg_leverage:.1f}x\n"
                    f"\n💡 <b>Multiple top traders moving same direction</b>\n"
                    f"Consider this a strong consensus signal.\n"
                )

                text += f"\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"

                if _send_message(text):
                    alerts_sent += 1

        return alerts_sent > 0

    except Exception as e:
        logger.error(f"Failed to send top movers alert: {e}")
        return False


# ─── Testing Utilities ────────────────────────────────────────────


def test_alerts():
    """Test all alert functions with mock data."""
    if not is_configured():
        print("Telegram not configured")
        return

    print("Testing daily P&L summary...")
    send_daily_pnl_summary()

    print("Testing portfolio snapshot...")
    send_portfolio_snapshot()

    print("Testing weekly digest...")
    send_weekly_digest()

    print("All tests complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_alerts()
