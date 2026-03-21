"""
Configuration for the Hyperliquid Trading Research Bot.
"""
import os

# ─── API Endpoints ─────────────────────────────────────────────
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_INFO_URL = f"{HYPERLIQUID_API_URL}/info"

# ─── Database ──────────────────────────────────────────────────
# Use environment variable or default to local data/ directory
# On some systems the mounted folder may have restricted permissions,
# so you can override with: export HL_BOT_DB=/path/to/bot.db
_DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bot.db")
DB_PATH = os.environ.get("HL_BOT_DB", _DEFAULT_DB)

# ─── Trader Discovery ─────────────────────────────────────────
# Minimum PnL (USD) to consider a trader "top"
# Set low initially so seed addresses get picked up; raise once the bot is mature
MIN_PNL_THRESHOLD = 0
# Maximum number of top traders to track at any time
MAX_TRACKED_TRADERS = 100
# How often to refresh the leaderboard (seconds)
LEADERBOARD_REFRESH_INTERVAL = 3600  # 1 hour

# ─── Strategy Analysis ────────────────────────────────────────
# Minimum number of trades to classify a strategy
MIN_TRADES_FOR_STRATEGY = 5
# Time windows for analysis
TIME_WINDOWS = {
    "short": 24,     # hours
    "medium": 168,    # 1 week
    "long": 720,      # 30 days
}

# ─── Strategy Scoring ─────────────────────────────────────────
# Weight decay factor for older strategy scores (per day)
SCORE_DECAY_RATE = 0.95
# Minimum score to keep a strategy active
MIN_STRATEGY_SCORE = 0.1
# Scoring weights
SCORING_WEIGHTS = {
    "pnl": 0.30,
    "win_rate": 0.25,
    "sharpe_ratio": 0.20,
    "consistency": 0.15,
    "risk_adjusted_return": 0.10,
}

# ─── Paper Trading ─────────────────────────────────────────────
PAPER_TRADING_INITIAL_BALANCE = 10_000  # USD
PAPER_TRADING_MAX_POSITION_PCT = 0.08   # 8% of balance per trade (smaller = more concurrent trades)
PAPER_TRADING_MAX_LEVERAGE = 5
PAPER_TRADING_STOP_LOSS_PCT = 0.05      # 5%
PAPER_TRADING_TAKE_PROFIT_PCT = 0.10    # 10%

# ─── Scheduling ────────────────────────────────────────────────
# Main loop interval (seconds)
MAIN_LOOP_INTERVAL = 300  # 5 minutes
# Full research cycle interval (seconds)
RESEARCH_CYCLE_INTERVAL = 3600  # 1 hour
# Strategy re-scoring interval (seconds)
SCORING_INTERVAL = 86400  # daily

# ─── Logging ───────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_LEVEL = "INFO"

# ─── Reports ───────────────────────────────────────────────────
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
