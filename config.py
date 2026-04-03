"""
Configuration for the Hyperliquid Trading Research Bot.
"""
import os

# ─── API Endpoints ─────────────────────────────────────────────
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_INFO_URL = f"{HYPERLIQUID_API_URL}/info"

# ─── Database ──────────────────────────────────────────────────
# Priority: HL_BOT_DB env var > /data/ volume > local ./data/
# On Railway: set HL_BOT_DB=/data/bot.db in Variables tab, or the code
# auto-detects the /data volume if it exists and is writable.
def _resolve_db_path() -> str:
    # 1. Explicit env var always wins
    env_db = os.environ.get("HL_BOT_DB")
    if env_db:
        return env_db
    # 2. Check if /data exists AND is writable (Railway persistent volume)
    try:
        os.makedirs("/data", exist_ok=True)
        probe = "/data/.write_test"
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return "/data/bot.db"
    except OSError:
        pass
    # 3. Fallback to local ./data/
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bot.db")

DB_PATH = _resolve_db_path()
_HAS_PERSISTENT_VOLUME = DB_PATH.startswith("/data")

# ─── Trader Discovery ─────────────────────────────────────────
# Minimum PnL (USD) to consider a trader "top"
# Set low initially so seed addresses get picked up; raise once the bot is mature
MIN_PNL_THRESHOLD = 0
# Maximum number of top traders to track at any time
MAX_TRACKED_TRADERS = 2000  # Scan top 2000 — bots are skipped via DB, APIManager handles rate limits
# How often to refresh the leaderboard (seconds)
LEADERBOARD_REFRESH_INTERVAL = 3600  # 1 hour

# ─── Bot Detection (tunable thresholds) ──────────────────────
BOT_HARD_CUTOFF_TRADES = int(os.environ.get("BOT_HARD_CUTOFF_TRADES", 100))   # >N trades/day = instant bot
BOT_THRESHOLD = int(os.environ.get("BOT_THRESHOLD", 3))                        # signal score >= N = bot
BOT_MM_PNL_THRESHOLD = float(os.environ.get("BOT_MM_PNL_THRESHOLD", 0.0))     # median PnL < N = spread/MM
BOT_ELEVATED_FREQ = int(os.environ.get("BOT_ELEVATED_FREQ", 50))              # trades/day for elevated freq signal

# ─── Strategy Analysis ────────────────────────────────────────
# Minimum number of trades to classify a strategy
MIN_TRADES_FOR_STRATEGY = 10
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
MIN_STRATEGY_SCORE = 0.15
# Max active strategies in DB — prune lowest-scoring beyond this
MAX_ACTIVE_STRATEGIES = int(os.environ.get("MAX_ACTIVE_STRATEGIES", 200))
# Max strategies per trading cycle fed to decision engine
MAX_STRATEGIES_PER_CYCLE = int(os.environ.get("MAX_STRATEGIES_PER_CYCLE", 15))
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

# Portfolio rotation for paper trading: keep the book flexible without
# removing safety rails entirely.
PORTFOLIO_TARGET_POSITIONS = int(os.environ.get("PORTFOLIO_TARGET_POSITIONS", 8))
PORTFOLIO_HARD_MAX_POSITIONS = int(os.environ.get("PORTFOLIO_HARD_MAX_POSITIONS", 10))
PORTFOLIO_RESERVED_HIGH_CONVICTION_SLOTS = int(
    os.environ.get("PORTFOLIO_RESERVED_HIGH_CONVICTION_SLOTS", 2)
)
PORTFOLIO_HIGH_CONVICTION_THRESHOLD = float(
    os.environ.get("PORTFOLIO_HIGH_CONVICTION_THRESHOLD", 0.78)
)
PORTFOLIO_MIN_HOLD_MINUTES = int(os.environ.get("PORTFOLIO_MIN_HOLD_MINUTES", 60))
PORTFOLIO_REPLACEMENT_THRESHOLD = float(
    os.environ.get("PORTFOLIO_REPLACEMENT_THRESHOLD", 0.15)
)
PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE = int(
    os.environ.get("PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE", 1)
)

# ─── Scheduling ────────────────────────────────────────────────
# 3-tier scheduling:
#   Tier 1 — Fast cycle:   position checks, SL/TP, copy-trade scan
#   Tier 2 — Trading cycle: regime detection, scoring, paper trading, arena
#   Tier 3 — Discovery:     leaderboard scan, bot detection, strategy ID
FAST_CYCLE_INTERVAL = 60           # 1 minute — position management
TRADING_CYCLE_INTERVAL = int(os.environ.get("TRADING_CYCLE_INTERVAL", 300))   # 5 minutes — regime + trading
DISCOVERY_CYCLE_INTERVAL = int(os.environ.get("DISCOVERY_CYCLE_INTERVAL", 86400))  # 24 hours — leaderboard scan
# Env-overridable so you can change on Railway without redeploying code:
#   TRADING_CYCLE_INTERVAL=180  → trade every 3 min (high vol)
#   DISCOVERY_CYCLE_INTERVAL=43200  → discover every 12h

# Legacy (kept for backward compat, not used by new scheduler)
MAIN_LOOP_INTERVAL = 300
RESEARCH_CYCLE_INTERVAL = TRADING_CYCLE_INTERVAL
SCORING_INTERVAL = 86400

# ─── Multi-Exchange Scanner ────────────────────────────────────
# Enable/disable secondary venues (Hyperliquid is always primary)
LIGHTER_ENABLED = os.environ.get("LIGHTER_ENABLED", "true").lower() in ("true", "1", "yes")

# ─── Predictive Regime Forecaster ──────────────────────────────
ENABLE_PREDICTIVE_FORECASTER = os.environ.get("ENABLE_PREDICTIVE_FORECASTER", "true").lower() in ("true", "1", "yes")
FORECASTER_CRASH_THRESHOLD = float(os.environ.get("FORECASTER_CRASH_THRESHOLD", -0.15))
ARKHAM_API_KEY = os.environ.get("ARKHAM_API_KEY")  # Optional: platform.arkhamintelligence.com

# ─── XGBoost Forecaster (optional ML upgrade) ─────────────────
ENABLE_XGBOOST_FORECASTER = os.environ.get("ENABLE_XGBOOST_FORECASTER", "true").lower() in ("true", "1", "yes")
XGBOOST_MODEL_PATH = "models/regime_xgboost.json"
XGBOOST_CRASH_THRESHOLD = float(os.environ.get("XGBOOST_CRASH_THRESHOLD", -0.18))
XGBOOST_MIN_CONFIDENCE = float(os.environ.get("XGBOOST_MIN_CONFIDENCE", 0.52))
XGBOOST_RETRAIN_INTERVAL = int(os.environ.get("XGBOOST_RETRAIN_INTERVAL", 86400))  # 24h walk-forward

# ─── Kelly Sizing ─────────────────────────────────────────────
# Multiplier: 1.0=full, 0.5=half, 0.25=quarter (recommended for crypto)
KELLY_MULTIPLIER = float(os.environ.get("KELLY_MULTIPLIER", 0.25))
KELLY_VOL_ADJUSTED = os.environ.get("KELLY_VOL_ADJUSTED", "true").lower() in ("true", "1", "yes")

# ─── Funding Rate Risk ────────────────────────────────────────
FUNDING_RISK_ENABLED = os.environ.get("FUNDING_RISK_ENABLED", "true").lower() in ("true", "1", "yes")
FUNDING_NEGATIVE_THRESHOLD = float(os.environ.get("FUNDING_NEGATIVE_THRESHOLD", -0.001))
FUNDING_POSITIVE_THRESHOLD = float(os.environ.get("FUNDING_POSITIVE_THRESHOLD", 0.003))

# ─── Polymarket Integration ──────────────────────────────────
POLYMARKET_ENABLED = os.environ.get("POLYMARKET_ENABLED", "true").lower() in ("true", "1", "yes")
POLYMARKET_SCAN_INTERVAL = int(os.environ.get("POLYMARKET_SCAN_INTERVAL", 180))  # 3 minutes
POLYMARKET_MIN_VOLUME = float(os.environ.get("POLYMARKET_MIN_VOLUME", 10000))    # $10k min volume

# ─── Options Flow Integration ───────────────────────────────
OPTIONS_FLOW_ENABLED = os.environ.get("OPTIONS_FLOW_ENABLED", "true").lower() in ("true", "1", "yes")
OPTIONS_FLOW_SCAN_INTERVAL = int(os.environ.get("OPTIONS_FLOW_SCAN_INTERVAL", 120))  # 2 minutes

# ─── Forecaster External Data ───────────────────────────────
# How long before external data (Polymarket, Options) is considered stale
FORECASTER_EXTERNAL_DATA_TTL = int(os.environ.get("FORECASTER_EXTERNAL_DATA_TTL", 600))  # 10 min

# ─── Monte-Carlo Stress Testing ──────────────────────────────
MONTE_CARLO_PATHS = int(os.environ.get("MONTE_CARLO_PATHS", 5000))
MONTE_CARLO_INCLUDE_CRASHES = True

# ─── Logging ───────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_LEVEL = "INFO"

# ─── Reports ───────────────────────────────────────────────────
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
