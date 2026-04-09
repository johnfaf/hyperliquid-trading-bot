"""
Telegram Bot for Trade Signal Notifications
Sends alerts when the bot opens/closes trades, detects strong signals,
or finds interesting market conditions.
Uses Telegram Bot API directly — no extra packages needed.
"""
import os
import logging
import requests
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_TELEGRAM_SUBSYSTEM = "telegram"
_MAX_SEND_RETRIES = max(1, int(os.environ.get("TELEGRAM_SEND_RETRIES", "3")))
_RETRY_BASE_DELAY_S = max(0.2, float(os.environ.get("TELEGRAM_SEND_RETRY_BASE_S", "0.8")))
_failure_streak = 0
_MAX_FAILURE_STREAK = 100  # Cap to prevent unbounded growth
_last_failure_ts: float = 0.0  # Auto-reset streak after extended silence
_FAILURE_RESET_WINDOW_S = 300.0  # 5 minutes without sends → reset streak


def _load_credentials() -> tuple:
    """Load Telegram credentials lazily from environment."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    api = f"https://api.telegram.org/bot{token}" if token else ""
    return token, chat_id, api


def _report_health(success: bool, reason: str = "") -> None:
    """Report Telegram subsystem health to the shared registry (best effort)."""
    try:
        from src.core.health_registry import registry, SubsystemState
    except Exception:
        return

    try:
        registry.register(_TELEGRAM_SUBSYSTEM, affects_trading=False)
        if success:
            registry.set_status(
                _TELEGRAM_SUBSYSTEM,
                SubsystemState.HEALTHY,
                reason="",
                dependency_ready=True,
                startup_status="READY",
            )
            registry.heartbeat(_TELEGRAM_SUBSYSTEM)
            return
        state = SubsystemState.DEGRADED if _failure_streak < 3 else SubsystemState.FAILED
        registry.set_status(
            _TELEGRAM_SUBSYSTEM,
            state,
            reason=reason[:200],
            dependency_ready=False,
            startup_status="ERROR",
        )
    except Exception:
        return


def is_configured() -> bool:
    """Check if Telegram credentials are set."""
    token, chat_id, _ = _load_credentials()
    return bool(token and chat_id)


def _send_message(text: str, parse_mode: str = "HTML", disable_preview: bool = True) -> bool:
    """Send a message via Telegram Bot API."""
    global _failure_streak, _last_failure_ts

    token, chat_id, telegram_api = _load_credentials()
    if not (token and chat_id and telegram_api):
        return False

    last_error = "unknown"
    for attempt in range(1, _MAX_SEND_RETRIES + 1):
        try:
            resp = requests.post(
                f"{telegram_api}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": disable_preview,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                _failure_streak = 0
                _report_health(success=True)
                return True

            body_preview = (resp.text or "")[:120]
            last_error = f"http_{resp.status_code}:{body_preview}"
            transient = resp.status_code == 429 or resp.status_code >= 500
            if transient and attempt < _MAX_SEND_RETRIES:
                wait = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    "Telegram send transient failure (%s) attempt %d/%d; retrying in %.1fs",
                    resp.status_code,
                    attempt,
                    _MAX_SEND_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue

            logger.warning("Telegram send failed: %s", last_error)
            break
        except Exception as e:
            last_error = str(e)
            if attempt < _MAX_SEND_RETRIES:
                wait = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    "Telegram send error attempt %d/%d; retrying in %.1fs: %s",
                    attempt,
                    _MAX_SEND_RETRIES,
                    wait,
                    e,
                )
                time.sleep(wait)
                continue
            logger.warning("Telegram send error: %s", e)

    # Cap streak and record timestamp for time-based decay
    now = time.time()
    if _last_failure_ts > 0 and (now - _last_failure_ts) > _FAILURE_RESET_WINDOW_S:
        _failure_streak = 0  # Reset after prolonged silence
    _failure_streak = min(_failure_streak + 1, _MAX_FAILURE_STREAK)
    _last_failure_ts = now  # noqa: F841 – used on next call
    _report_health(success=False, reason=f"send_failure_streak={_failure_streak} last_error={last_error}")
    return False


# ─── Notification Templates ──────────────────────────────────

def notify_trade_opened(trade: Dict, source: str = "strategy"):
    """Notify when a new paper trade is opened."""
    emoji = "🟢" if trade.get("side") == "long" else "🔴"
    leverage = trade.get("leverage", 1)
    confidence = trade.get("confidence", 0)

    text = (
        f"{emoji} <b>NEW TRADE OPENED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coin:</b> {trade.get('coin', '?')}\n"
        f"<b>Side:</b> {trade.get('side', '?').upper()}\n"
        f"<b>Entry:</b> ${trade.get('entry_price', trade.get('price', 0)):,.4f}\n"
        f"<b>Size:</b> {trade.get('size', 0):.4f}\n"
        f"<b>Leverage:</b> {leverage}x\n"
        f"<b>SL:</b> ${trade.get('stop_loss', 0):,.4f}\n"
        f"<b>TP:</b> ${trade.get('take_profit', 0):,.4f}\n"
    )

    if confidence:
        text += f"<b>Confidence:</b> {confidence:.0%}\n"

    if source == "copy":
        text += f"<b>Source:</b> Copy trade from {trade.get('source_trader', '?')}\n"
    elif source == "strategy":
        text += f"<b>Strategy:</b> {trade.get('strategy_type', '?')}\n"

    text += f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"

    _send_message(text)


def notify_trade_closed(trade: Dict, exit_price: float, pnl: float, reason: str = ""):
    """Notify when a trade is closed."""
    emoji = "💰" if pnl > 0 else "💸"
    pnl_pct = ""
    if trade.get("entry_price") and trade["entry_price"] > 0:
        leverage = trade.get("leverage", 1)
        if trade.get("side") == "long":
            pct = ((exit_price - trade["entry_price"]) / trade["entry_price"]) * 100 * leverage
        else:
            pct = ((trade["entry_price"] - exit_price) / trade["entry_price"]) * 100 * leverage
        pnl_pct = f" ({pct:+.2f}%)"

    reason_labels = {
        "stop_loss": "🛑 Stop Loss",
        "take_profit": "🎯 Take Profit",
        "trailing_stop": "📈 Trailing Stop",
        "time_exit_24h": "⏰ Time Exit (24h)",
        "copy_close": "👤 Source Exited",
    }
    reason_text = reason_labels.get(reason, reason or "Manual")

    text = (
        f"{emoji} <b>TRADE CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coin:</b> {trade.get('coin', '?')}\n"
        f"<b>Side:</b> {trade.get('side', '?').upper()}\n"
        f"<b>Entry:</b> ${trade.get('entry_price', 0):,.4f}\n"
        f"<b>Exit:</b> ${exit_price:,.4f}\n"
        f"<b>PnL:</b> ${pnl:+,.2f}{pnl_pct}\n"
        f"<b>Reason:</b> {reason_text}\n"
        f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
    )

    _send_message(text)


def notify_market_bias(overview: Dict):
    """Notify about overall market directional bias."""
    bias = overview.get("overall_bias", "neutral")
    score = overview.get("overall_bias_score", 0)
    volume = overview.get("total_market_volume", 0)

    if bias == "neutral":
        return  # Don't spam on neutral

    emoji = "🐂" if bias == "bullish" else "🐻"

    bullish = overview.get("bullish_coins", [])
    bearish = overview.get("bearish_coins", [])

    text = (
        f"{emoji} <b>MARKET BIAS: {bias.upper()}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Bias Score:</b> {score:+.4f}\n"
        f"<b>24h Volume:</b> ${volume:,.0f}\n"
    )

    if bullish:
        text += f"<b>Bullish:</b> {', '.join(bullish)}\n"
    if bearish:
        text += f"<b>Bearish:</b> {', '.join(bearish)}\n"

    # Per-coin breakdown
    coins = overview.get("coins", {})
    if coins:
        text += "\n<b>Coin Details:</b>\n"
        for coin, data in list(coins.items())[:7]:
            cb = data.get("directional_bias", "?")
            chg = data.get("avg_price_change_pct", 0)
            vol = data.get("total_volume_24h", 0)
            icon = "🟢" if cb == "bullish" else "🔴" if cb == "bearish" else "⚪"
            text += f"{icon} {coin}: {chg:+.2f}% | Vol: ${vol:,.0f}\n"

    text += f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"

    _send_message(text)


def notify_cycle_summary(summary: Dict):
    """Notify with a cycle summary (sent every full research cycle)."""
    text = (
        f"📊 <b>CYCLE SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Balance:</b> ${summary.get('balance', 0):,.2f}\n"
        f"<b>Total PnL:</b> ${summary.get('total_pnl', 0):+,.2f}\n"
        f"<b>Unrealized:</b> ${summary.get('unrealized_pnl', 0):+,.2f}\n"
        f"<b>Equity:</b> ${summary.get('total_equity', 0):,.2f}\n"
        f"<b>ROI:</b> {summary.get('roi_pct', 0):+.2f}%\n"
        f"<b>Win Rate:</b> {summary.get('win_rate', 0):.0%}\n"
        f"<b>Open Positions:</b> {summary.get('open_positions', 0)}\n"
        f"<b>Total Trades:</b> {summary.get('total_trades', 0)}\n"
    )

    if summary.get("top_strategy"):
        text += f"\n<b>Top Strategy:</b> {summary['top_strategy']}\n"

    if summary.get("market_bias"):
        text += f"<b>Market Bias:</b> {summary['market_bias']}\n"

    text += f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"

    _send_message(text)


def notify_strong_signal(coin: str, side: str, reasons: List[str], confidence: float):
    """Notify about a high-confidence convergence signal."""
    emoji = "⚡🟢" if side == "long" else "⚡🔴"

    text = (
        f"{emoji} <b>STRONG SIGNAL DETECTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coin:</b> {coin}\n"
        f"<b>Direction:</b> {side.upper()}\n"
        f"<b>Confidence:</b> {confidence:.0%}\n"
        f"\n<b>Confluence Reasons:</b>\n"
    )

    for reason in reasons:
        text += f"  • {reason}\n"

    text += f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"

    _send_message(text)


def notify_trailing_stop_update(coin: str, side: str, old_sl: float, new_sl: float):
    """Notify when a trailing stop is updated."""
    text = (
        f"📈 <b>TRAILING STOP UPDATED</b>\n"
        f"<b>{coin}</b> {side.upper()}\n"
        f"SL: ${old_sl:,.4f} → ${new_sl:,.4f}\n"
    )
    _send_message(text)


def send_startup_message():
    """Send a message when the bot starts."""
    text = (
        f"🤖 <b>Hyperliquid Research Bot Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Monitoring top traders, strategies, and\n"
        f"multi-exchange volume data.\n"
        f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    _send_message(text)


def send_test_message():
    """Send a test message to verify configuration."""
    if not is_configured():
        print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        return False
    return _send_message("✅ <b>Hyperliquid Bot</b> — Telegram notifications working!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if is_configured():
        print("Sending test message...")
        send_test_message()
    else:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to test.")
