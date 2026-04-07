"""
Configuration for the Hyperliquid Trading Research Bot.
"""
import os

# ─── API Endpoints ─────────────────────────────────────────────
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_INFO_URL = f"{HYPERLIQUID_API_URL}/info"
HYPERLIQUID_EXCHANGE_URL = f"{HYPERLIQUID_API_URL}/exchange"

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
PAPER_TRADING_MAX_LEVERAGE = float(os.environ.get("PAPER_TRADING_MAX_LEVERAGE", 5))
# Stop-loss is applied BEFORE leverage scaling in paper_trader.py (SL% / leverage).
# At 5x leverage a 5% SL = 1% raw move — too tight for volatile crypto.
# Raise this so the raw per-position SL allows normal intra-day noise:
#   e.g. PAPER_TRADING_STOP_LOSS_PCT=0.15 at 5x → 3% raw move before exit.
PAPER_TRADING_STOP_LOSS_PCT = float(os.environ.get("PAPER_TRADING_STOP_LOSS_PCT", 0.15))
PAPER_TRADING_TAKE_PROFIT_PCT = float(os.environ.get("PAPER_TRADING_TAKE_PROFIT_PCT", 0.30))
PAPER_TRADING_MAKER_FEE_BPS = float(os.environ.get("PAPER_TRADING_MAKER_FEE_BPS", 1.5))
PAPER_TRADING_TAKER_FEE_BPS = float(os.environ.get("PAPER_TRADING_TAKER_FEE_BPS", 4.5))
PAPER_TRADING_DEFAULT_EXECUTION_ROLE = os.environ.get(
    "PAPER_TRADING_DEFAULT_EXECUTION_ROLE", "taker"
).lower()

# Live trading wallet / secret-management controls.
# Agent-wallet-only mode: signer key must be for a delegated agent wallet, and
# HL_PUBLIC_ADDRESS points to the trading account (master/vault) being managed.
LIVE_TRADING_ENABLED = os.environ.get(
    "LIVE_TRADING_ENABLED", "false"
).lower() in ("true", "1", "yes")

# ─── Live Order Caps (cautious bootstrap) ─────────────────────
# Hyperliquid enforces a $10 minimum notional per order on both perps and
# spot.  Any order below this silently does not fill (the matching engine
# drops it and clearinghouseState never shows the position) — you only
# notice because fill-verification times out with "FILL NOT VERIFIED".
# This floor is PHYSICALLY enforced by the exchange and cannot be lowered
# by config.
LIVE_MIN_ORDER_USD = float(os.environ.get("LIVE_MIN_ORDER_USD", 11.0))

# Hard ceiling on the notional ($ USDC) of any single live order.  This is a
# safety net while the bot is ramping on a small live balance — even if paper
# sizing, rescaling, or the firewall suggest a larger trade, nothing above
# LIVE_MAX_ORDER_USD is ever sent to the exchange.
# The default is set slightly above the exchange minimum so fresh bootstraps
# can actually execute; set a higher value via env var as confidence grows.
# NOTE: a value below LIVE_MIN_ORDER_USD is impossible to honor — the
# LiveTrader will raise it to LIVE_MIN_ORDER_USD at startup with a warning.
LIVE_MAX_ORDER_USD = float(os.environ.get("LIVE_MAX_ORDER_USD", 12.0))
# Daily loss limit for the live account in USD (forwarded to LiveTrader).
LIVE_MAX_DAILY_LOSS_USD = float(os.environ.get("LIVE_MAX_DAILY_LOSS_USD", 5.0))
HL_WALLET_MODE = os.environ.get("HL_WALLET_MODE", "agent_only").strip().lower()
SECRET_MANAGER_PROVIDER = os.environ.get(
    "SECRET_MANAGER_PROVIDER", "none"
).strip().lower()
AWS_KMS_REGION = os.environ.get("AWS_KMS_REGION", "")
AWS_KMS_KEY_ID = os.environ.get("AWS_KMS_KEY_ID", "")
AWS_KMS_CIPHERTEXT_B64 = os.environ.get("AWS_KMS_CIPHERTEXT_B64", "")
VAULT_ADDR = os.environ.get("VAULT_ADDR", "")
VAULT_TOKEN = os.environ.get("VAULT_TOKEN", "")
VAULT_SECRET_PATH = os.environ.get("VAULT_SECRET_PATH", "")
VAULT_SECRET_KEY = os.environ.get("VAULT_SECRET_KEY", "hl_agent_private_key")
VAULT_KV_VERSION = int(os.environ.get("VAULT_KV_VERSION", "2"))

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
PORTFOLIO_TRANSACTION_COST_WEIGHT = float(
    os.environ.get("PORTFOLIO_TRANSACTION_COST_WEIGHT", 8.0)
)
PORTFOLIO_CHURN_PENALTY = float(os.environ.get("PORTFOLIO_CHURN_PENALTY", 0.02))
PORTFOLIO_EXPECTED_SLIPPAGE_BPS = float(
    os.environ.get("PORTFOLIO_EXPECTED_SLIPPAGE_BPS", 3.0)
)
PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR = int(
    os.environ.get("PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR", 4)
)
PORTFOLIO_MAX_REPLACEMENTS_PER_DAY = int(
    os.environ.get("PORTFOLIO_MAX_REPLACEMENTS_PER_DAY", 12)
)
PORTFOLIO_FORCED_EXIT_COOLDOWN_MINUTES = int(
    os.environ.get("PORTFOLIO_FORCED_EXIT_COOLDOWN_MINUTES", 45)
)
PORTFOLIO_ROUND_TRIP_BLOCK_MINUTES = int(
    os.environ.get("PORTFOLIO_ROUND_TRIP_BLOCK_MINUTES", 20)
)
PORTFOLIO_MAX_COIN_EXPOSURE_PCT = float(
    os.environ.get("PORTFOLIO_MAX_COIN_EXPOSURE_PCT", 0.45)
)
PORTFOLIO_MAX_SIDE_EXPOSURE_PCT = float(
    os.environ.get("PORTFOLIO_MAX_SIDE_EXPOSURE_PCT", 0.65)
)
PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT = float(
    os.environ.get("PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT", 0.55)
)
PORTFOLIO_SIZER_ENABLED = os.environ.get(
    "PORTFOLIO_SIZER_ENABLED", "true"
).lower() in ("true", "1", "yes")
PORTFOLIO_MIN_POSITION_PCT = float(os.environ.get("PORTFOLIO_MIN_POSITION_PCT", 0.01))
PORTFOLIO_TARGET_VOLATILITY_PCT = float(
    os.environ.get("PORTFOLIO_TARGET_VOLATILITY_PCT", 0.025)
)
PORTFOLIO_MAX_BETA_ABS = float(os.environ.get("PORTFOLIO_MAX_BETA_ABS", 1.40))
PORTFOLIO_STOP_LOSS_VOL_MULTIPLIER = float(
    os.environ.get("PORTFOLIO_STOP_LOSS_VOL_MULTIPLIER", 0.85)
)
PORTFOLIO_TREND_REWARD_RISK = float(os.environ.get("PORTFOLIO_TREND_REWARD_RISK", 2.4))
PORTFOLIO_BASE_REWARD_RISK = float(os.environ.get("PORTFOLIO_BASE_REWARD_RISK", 1.9))
PORTFOLIO_VOLATILE_REWARD_RISK = float(
    os.environ.get("PORTFOLIO_VOLATILE_REWARD_RISK", 1.35)
)
PORTFOLIO_TREND_TIME_LIMIT_HOURS = float(
    os.environ.get("PORTFOLIO_TREND_TIME_LIMIT_HOURS", 36.0)
)
PORTFOLIO_BASE_TIME_LIMIT_HOURS = float(
    os.environ.get("PORTFOLIO_BASE_TIME_LIMIT_HOURS", 24.0)
)
PORTFOLIO_VOLATILE_TIME_LIMIT_HOURS = float(
    os.environ.get("PORTFOLIO_VOLATILE_TIME_LIMIT_HOURS", 12.0)
)
ROTATION_ENGINE_ENABLED = os.environ.get(
    "ROTATION_ENGINE_ENABLED", "false"
).lower() in ("true", "1", "yes")
ROTATION_DRY_RUN_TELEMETRY = os.environ.get(
    "ROTATION_DRY_RUN_TELEMETRY", "true"
).lower() in ("true", "1", "yes")
ROTATION_SHADOW_MODE_DAYS = int(os.environ.get("ROTATION_SHADOW_MODE_DAYS", "7"))
ROTATION_REQUIRE_EXPLICIT_THRESHOLDS = os.environ.get(
    "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", "true"
).lower() in ("true", "1", "yes")

# ─── Decision Firewall ─────────────────────────────────────────
# Minimum signal confidence to pass the firewall.
# 0.15 (15%) is far too permissive — nearly any signal passes.
# Raise to 0.45+ to ensure only well-confirmed signals are traded.
FIREWALL_MIN_CONFIDENCE = float(os.environ.get("FIREWALL_MIN_CONFIDENCE", 0.45))

# Decision engine tightening: these weights and floors make the ranking layer
# care about calibrated confidence, source quality, and net expectancy after costs.
DECISION_W_SCORE = float(os.environ.get("DECISION_W_SCORE", 0.18))
DECISION_W_REGIME = float(os.environ.get("DECISION_W_REGIME", 0.14))
DECISION_W_DIVERSITY = float(os.environ.get("DECISION_W_DIVERSITY", 0.10))
DECISION_W_FRESHNESS = float(os.environ.get("DECISION_W_FRESHNESS", 0.05))
DECISION_W_CONSENSUS = float(os.environ.get("DECISION_W_CONSENSUS", 0.04))
DECISION_W_CONFIDENCE = float(os.environ.get("DECISION_W_CONFIDENCE", 0.13))
DECISION_W_SOURCE_QUALITY = float(os.environ.get("DECISION_W_SOURCE_QUALITY", 0.09))
DECISION_W_CONFIRMATION = float(os.environ.get("DECISION_W_CONFIRMATION", 0.07))
DECISION_W_EXPECTED_VALUE = float(os.environ.get("DECISION_W_EXPECTED_VALUE", 0.20))
DECISION_W_CONFLUENCE = float(os.environ.get("DECISION_W_CONFLUENCE", 0.10))
DECISION_W_CONTEXT = float(os.environ.get("DECISION_W_CONTEXT", 0.08))
DECISION_W_CALIBRATION = float(os.environ.get("DECISION_W_CALIBRATION", 0.06))
DECISION_W_MEMORY = float(os.environ.get("DECISION_W_MEMORY", 0.07))
DECISION_W_DIVERGENCE = float(os.environ.get("DECISION_W_DIVERGENCE", 0.08))
DECISION_W_CAPITAL_GOVERNOR = float(os.environ.get("DECISION_W_CAPITAL_GOVERNOR", 0.08))
DECISION_MIN_SCORE = float(os.environ.get("DECISION_MIN_SCORE", 0.34))
DECISION_MIN_CONFIDENCE = float(os.environ.get("DECISION_MIN_CONFIDENCE", 0.58))
DECISION_MIN_SOURCE_WEIGHT = float(os.environ.get("DECISION_MIN_SOURCE_WEIGHT", 0.35))
DECISION_MIN_EXPECTED_VALUE_PCT = float(
    os.environ.get("DECISION_MIN_EXPECTED_VALUE_PCT", 0.0015)
)
DECISION_CONFLUENCE_ENABLED = os.environ.get(
    "DECISION_CONFLUENCE_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_CONFLUENCE_BASELINE = float(
    os.environ.get("DECISION_CONFLUENCE_BASELINE", 0.30)
)
DECISION_CONFLUENCE_FULL_WEIGHT = float(
    os.environ.get("DECISION_CONFLUENCE_FULL_WEIGHT", 1.50)
)
DECISION_CONFLUENCE_TARGET_SUPPORT_SOURCES = int(
    os.environ.get("DECISION_CONFLUENCE_TARGET_SUPPORT_SOURCES", 2)
)
DECISION_CONFLUENCE_CONFLICT_BLOCK_THRESHOLD = float(
    os.environ.get("DECISION_CONFLUENCE_CONFLICT_BLOCK_THRESHOLD", 0.65)
)
DECISION_CONFLUENCE_CONFLICT_FLOOR = float(
    os.environ.get("DECISION_CONFLUENCE_CONFLICT_FLOOR", 0.35)
)
DECISION_CONTEXT_PERFORMANCE_ENABLED = os.environ.get(
    "DECISION_CONTEXT_PERFORMANCE_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_CONTEXT_PERFORMANCE_LOOKBACK_HOURS = float(
    os.environ.get("DECISION_CONTEXT_PERFORMANCE_LOOKBACK_HOURS", 24 * 30)
)
DECISION_CONTEXT_PERFORMANCE_MIN_TRADES = int(
    os.environ.get("DECISION_CONTEXT_PERFORMANCE_MIN_TRADES", 3)
)
DECISION_CONTEXT_PERFORMANCE_RETURN_SCALE = float(
    os.environ.get("DECISION_CONTEXT_PERFORMANCE_RETURN_SCALE", 0.03)
)
DECISION_CONTEXT_PERFORMANCE_BLOCK_WIN_RATE = float(
    os.environ.get("DECISION_CONTEXT_PERFORMANCE_BLOCK_WIN_RATE", 0.30)
)
DECISION_CONTEXT_PERFORMANCE_BLOCK_AVG_RETURN_PCT = float(
    os.environ.get("DECISION_CONTEXT_PERFORMANCE_BLOCK_AVG_RETURN_PCT", -0.015)
)
DECISION_CONTEXT_PERFORMANCE_BOOST_WIN_RATE = float(
    os.environ.get("DECISION_CONTEXT_PERFORMANCE_BOOST_WIN_RATE", 0.60)
)
DECISION_CALIBRATION_ENABLED = os.environ.get(
    "DECISION_CALIBRATION_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_CALIBRATION_MIN_RECORDS = int(
    os.environ.get("DECISION_CALIBRATION_MIN_RECORDS", 20)
)
DECISION_CALIBRATION_TARGET_ECE = float(
    os.environ.get("DECISION_CALIBRATION_TARGET_ECE", 0.05)
)
DECISION_CALIBRATION_MAX_ECE = float(
    os.environ.get("DECISION_CALIBRATION_MAX_ECE", 0.20)
)
DECISION_MEMORY_ENABLED = os.environ.get(
    "DECISION_MEMORY_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_MEMORY_MIN_TRADES = int(
    os.environ.get("DECISION_MEMORY_MIN_TRADES", 3)
)
DECISION_MEMORY_MIN_SIMILARITY = float(
    os.environ.get("DECISION_MEMORY_MIN_SIMILARITY", 0.55)
)
DECISION_MEMORY_TOP_K = int(
    os.environ.get("DECISION_MEMORY_TOP_K", 8)
)
DECISION_MEMORY_BLOCK_ON_AVOID = os.environ.get(
    "DECISION_MEMORY_BLOCK_ON_AVOID",
    "true",
).lower() in ("true", "1", "yes")
DECISION_DIVERGENCE_ENABLED = os.environ.get(
    "DECISION_DIVERGENCE_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_DIVERGENCE_BLOCK_ON_STATUS = os.environ.get(
    "DECISION_DIVERGENCE_BLOCK_ON_STATUS",
    "true",
).lower() in ("true", "1", "yes")
DECISION_MAX_TRADES_PER_CYCLE = int(os.environ.get("DECISION_MAX_TRADES_PER_CYCLE", 2))
DECISION_MAKER_FEE_BPS = float(
    os.environ.get("DECISION_MAKER_FEE_BPS", PAPER_TRADING_MAKER_FEE_BPS)
)
DECISION_TAKER_FEE_BPS = float(
    os.environ.get("DECISION_TAKER_FEE_BPS", PAPER_TRADING_TAKER_FEE_BPS)
)
DECISION_EXPECTED_SLIPPAGE_BPS = float(
    os.environ.get("DECISION_EXPECTED_SLIPPAGE_BPS", PORTFOLIO_EXPECTED_SLIPPAGE_BPS)
)
DECISION_CHURN_PENALTY_BPS = float(
    os.environ.get("DECISION_CHURN_PENALTY_BPS", 2.0)
)
DECISION_DEFAULT_EXECUTION_ROLE = os.environ.get(
    "DECISION_DEFAULT_EXECUTION_ROLE",
    PAPER_TRADING_DEFAULT_EXECUTION_ROLE,
)
DECISION_PERSIST_RESEARCH = os.environ.get(
    "DECISION_PERSIST_RESEARCH",
    "true",
).lower() in ("true", "1", "yes")
DECISION_EXECUTION_QUALITY_ENABLED = os.environ.get(
    "DECISION_EXECUTION_QUALITY_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_EXECUTION_QUALITY_LOOKBACK_HOURS = float(
    os.environ.get("DECISION_EXECUTION_QUALITY_LOOKBACK_HOURS", 168)
)
DECISION_EXECUTION_QUALITY_MIN_EVENTS = int(
    os.environ.get("DECISION_EXECUTION_QUALITY_MIN_EVENTS", 3)
)
DECISION_EXECUTION_REJECTION_PENALTY_BPS = float(
    os.environ.get("DECISION_EXECUTION_REJECTION_PENALTY_BPS", 12.0)
)
DECISION_EXECUTION_FILL_GAP_PENALTY_BPS = float(
    os.environ.get("DECISION_EXECUTION_FILL_GAP_PENALTY_BPS", 8.0)
)
DECISION_EXECUTION_PROTECTIVE_FAILURE_PENALTY_BPS = float(
    os.environ.get("DECISION_EXECUTION_PROTECTIVE_FAILURE_PENALTY_BPS", 18.0)
)
DECISION_EXECUTION_POLICY_ENABLED = os.environ.get(
    "DECISION_EXECUTION_POLICY_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_EXECUTION_POLICY_LOOKBACK_HOURS = float(
    os.environ.get("DECISION_EXECUTION_POLICY_LOOKBACK_HOURS", 168)
)
DECISION_EXECUTION_POLICY_MIN_EVENTS = int(
    os.environ.get("DECISION_EXECUTION_POLICY_MIN_EVENTS", 3)
)
DECISION_EXECUTION_POLICY_MAKER_REJECTION_CEILING = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAKER_REJECTION_CEILING", 0.18)
)
DECISION_EXECUTION_POLICY_MAKER_FILL_FLOOR = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAKER_FILL_FLOOR", 0.72)
)
DECISION_EXECUTION_POLICY_MAKER_CONFIDENCE_CEILING = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAKER_CONFIDENCE_CEILING", 0.78)
)
DECISION_EXECUTION_POLICY_MAKER_SOURCE_QUALITY_FLOOR = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAKER_SOURCE_QUALITY_FLOOR", 0.52)
)
DECISION_EXECUTION_POLICY_MAKER_OFFSET_BPS = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAKER_OFFSET_BPS", 1.5)
)
DECISION_EXECUTION_POLICY_MIN_MAKER_OFFSET_BPS = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MIN_MAKER_OFFSET_BPS", 0.5)
)
DECISION_EXECUTION_POLICY_MAX_MAKER_OFFSET_BPS = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAX_MAKER_OFFSET_BPS", 5.0)
)
DECISION_EXECUTION_POLICY_MAKER_TIMEOUT_SECONDS = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAKER_TIMEOUT_SECONDS", 5.0)
)
DECISION_EXECUTION_POLICY_FALLBACK_CONFIDENCE_THRESHOLD = float(
    os.environ.get("DECISION_EXECUTION_POLICY_FALLBACK_CONFIDENCE_THRESHOLD", 0.72)
)
DECISION_EXECUTION_POLICY_FALLBACK_SOURCE_QUALITY_THRESHOLD = float(
    os.environ.get("DECISION_EXECUTION_POLICY_FALLBACK_SOURCE_QUALITY_THRESHOLD", 0.62)
)
DECISION_EXECUTION_POLICY_MARKET_SLIPPAGE_MULTIPLIER = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MARKET_SLIPPAGE_MULTIPLIER", 1.0)
)
DECISION_EXECUTION_POLICY_MAKER_SLIPPAGE_FLOOR_BPS = float(
    os.environ.get("DECISION_EXECUTION_POLICY_MAKER_SLIPPAGE_FLOOR_BPS", 0.4)
)
DECISION_EXECUTION_POLICY_DEFAULT_MARKET_SLIPPAGE_BPS = float(
    os.environ.get("DECISION_EXECUTION_POLICY_DEFAULT_MARKET_SLIPPAGE_BPS", 3.0)
)
DECISION_EXECUTION_POLICY_URGENT_SOURCES = [
    item.strip()
    for item in os.environ.get(
        "DECISION_EXECUTION_POLICY_URGENT_SOURCES",
        "options_flow,polymarket,liquidation_strategy,arena_champion",
    ).split(",")
    if item.strip()
]
EXPERIMENT_REPORT_LIMIT_CYCLES = int(
    os.environ.get("EXPERIMENT_REPORT_LIMIT_CYCLES", 120)
)
EXPERIMENT_OOS_RATIO = float(os.environ.get("EXPERIMENT_OOS_RATIO", 0.30))
EXPERIMENT_MIN_OOS_CYCLES = int(os.environ.get("EXPERIMENT_MIN_OOS_CYCLES", 10))
EXPERIMENT_MIN_EV_DELTA_PCT = float(
    os.environ.get("EXPERIMENT_MIN_EV_DELTA_PCT", 0.0005)
)
EXPERIMENT_ATTRIBUTION_LOOKBACK_HOURS = float(
    os.environ.get("EXPERIMENT_ATTRIBUTION_LOOKBACK_HOURS", 168)
)
EXPERIMENT_DIVERGENCE_LOOKBACK_HOURS = float(
    os.environ.get("EXPERIMENT_DIVERGENCE_LOOKBACK_HOURS", 24)
)
RUNTIME_DIVERGENCE_CONTROL_ENABLED = os.environ.get(
    "RUNTIME_DIVERGENCE_CONTROL_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
RUNTIME_DIVERGENCE_REFRESH_INTERVAL_SECONDS = float(
    os.environ.get("RUNTIME_DIVERGENCE_REFRESH_INTERVAL_SECONDS", 300)
)
RUNTIME_DIVERGENCE_MIN_LIVE_EVENTS = int(
    os.environ.get("RUNTIME_DIVERGENCE_MIN_LIVE_EVENTS", 3)
)
RUNTIME_DIVERGENCE_SOURCE_MIN_SELECTED = int(
    os.environ.get("RUNTIME_DIVERGENCE_SOURCE_MIN_SELECTED", 3)
)
RUNTIME_DIVERGENCE_CAUTION_MULTIPLIER = float(
    os.environ.get("RUNTIME_DIVERGENCE_CAUTION_MULTIPLIER", 0.65)
)
RUNTIME_DIVERGENCE_BLOCKED_MULTIPLIER = float(
    os.environ.get("RUNTIME_DIVERGENCE_BLOCKED_MULTIPLIER", 0.0)
)
RUNTIME_DIVERGENCE_BLOCK_ON_STATUS = os.environ.get(
    "RUNTIME_DIVERGENCE_BLOCK_ON_STATUS",
    "true",
).lower() in ("true", "1", "yes")
RUNTIME_DIVERGENCE_GLOBAL_CAUTION_OPEN_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_CAUTION_OPEN_GAP_RATIO", 0.20)
)
RUNTIME_DIVERGENCE_GLOBAL_BLOCK_OPEN_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_BLOCK_OPEN_GAP_RATIO", 0.40)
)
RUNTIME_DIVERGENCE_GLOBAL_CAUTION_EXECUTION_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_CAUTION_EXECUTION_GAP_RATIO", 0.25)
)
RUNTIME_DIVERGENCE_GLOBAL_BLOCK_EXECUTION_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_BLOCK_EXECUTION_GAP_RATIO", 0.50)
)
RUNTIME_DIVERGENCE_GLOBAL_CAUTION_PNL_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_CAUTION_PNL_GAP_RATIO", 0.20)
)
RUNTIME_DIVERGENCE_GLOBAL_BLOCK_PNL_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_BLOCK_PNL_GAP_RATIO", 0.45)
)
RUNTIME_DIVERGENCE_GLOBAL_CAUTION_REJECTION_RATE = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_CAUTION_REJECTION_RATE", 0.18)
)
RUNTIME_DIVERGENCE_GLOBAL_BLOCK_REJECTION_RATE = float(
    os.environ.get("RUNTIME_DIVERGENCE_GLOBAL_BLOCK_REJECTION_RATE", 0.30)
)
RUNTIME_DIVERGENCE_SOURCE_CAUTION_EXECUTION_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_SOURCE_CAUTION_EXECUTION_GAP_RATIO", 0.30)
)
RUNTIME_DIVERGENCE_SOURCE_BLOCK_EXECUTION_GAP_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_SOURCE_BLOCK_EXECUTION_GAP_RATIO", 0.60)
)
RUNTIME_DIVERGENCE_SOURCE_CAUTION_REJECTION_RATE = float(
    os.environ.get("RUNTIME_DIVERGENCE_SOURCE_CAUTION_REJECTION_RATE", 0.20)
)
RUNTIME_DIVERGENCE_SOURCE_BLOCK_REJECTION_RATE = float(
    os.environ.get("RUNTIME_DIVERGENCE_SOURCE_BLOCK_REJECTION_RATE", 0.35)
)
RUNTIME_DIVERGENCE_SOURCE_CAUTION_FILL_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_SOURCE_CAUTION_FILL_RATIO", 0.70)
)
RUNTIME_DIVERGENCE_SOURCE_BLOCK_FILL_RATIO = float(
    os.environ.get("RUNTIME_DIVERGENCE_SOURCE_BLOCK_FILL_RATIO", 0.45)
)
CAPITAL_GOVERNOR_ENABLED = os.environ.get(
    "CAPITAL_GOVERNOR_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
CAPITAL_GOVERNOR_LOOKBACK_HOURS = float(
    os.environ.get("CAPITAL_GOVERNOR_LOOKBACK_HOURS", 24 * 14)
)
CAPITAL_GOVERNOR_REFRESH_INTERVAL_SECONDS = float(
    os.environ.get("CAPITAL_GOVERNOR_REFRESH_INTERVAL_SECONDS", 300)
)
CAPITAL_GOVERNOR_MIN_PAPER_TRADES = int(
    os.environ.get("CAPITAL_GOVERNOR_MIN_PAPER_TRADES", 5)
)
CAPITAL_GOVERNOR_MIN_LIVE_SNAPSHOTS = int(
    os.environ.get("CAPITAL_GOVERNOR_MIN_LIVE_SNAPSHOTS", 12)
)
CAPITAL_GOVERNOR_MIN_SOURCE_PROFILES = int(
    os.environ.get("CAPITAL_GOVERNOR_MIN_SOURCE_PROFILES", 3)
)
CAPITAL_GOVERNOR_CAUTION_MULTIPLIER = float(
    os.environ.get("CAPITAL_GOVERNOR_CAUTION_MULTIPLIER", 0.75)
)
CAPITAL_GOVERNOR_RISK_OFF_MULTIPLIER = float(
    os.environ.get("CAPITAL_GOVERNOR_RISK_OFF_MULTIPLIER", 0.40)
)
CAPITAL_GOVERNOR_BLOCKED_MULTIPLIER = float(
    os.environ.get("CAPITAL_GOVERNOR_BLOCKED_MULTIPLIER", 0.0)
)
CAPITAL_GOVERNOR_BLOCK_ON_RISK_OFF = os.environ.get(
    "CAPITAL_GOVERNOR_BLOCK_ON_RISK_OFF",
    "false",
).lower() in ("true", "1", "yes")
CAPITAL_GOVERNOR_CAUTION_PAPER_DRAWDOWN_PCT = float(
    os.environ.get("CAPITAL_GOVERNOR_CAUTION_PAPER_DRAWDOWN_PCT", 0.08)
)
CAPITAL_GOVERNOR_RISK_OFF_PAPER_DRAWDOWN_PCT = float(
    os.environ.get("CAPITAL_GOVERNOR_RISK_OFF_PAPER_DRAWDOWN_PCT", 0.15)
)
CAPITAL_GOVERNOR_BLOCK_PAPER_DRAWDOWN_PCT = float(
    os.environ.get("CAPITAL_GOVERNOR_BLOCK_PAPER_DRAWDOWN_PCT", 0.22)
)
CAPITAL_GOVERNOR_CAUTION_LIVE_DRAWDOWN_PCT = float(
    os.environ.get("CAPITAL_GOVERNOR_CAUTION_LIVE_DRAWDOWN_PCT", 0.05)
)
CAPITAL_GOVERNOR_RISK_OFF_LIVE_DRAWDOWN_PCT = float(
    os.environ.get("CAPITAL_GOVERNOR_RISK_OFF_LIVE_DRAWDOWN_PCT", 0.10)
)
CAPITAL_GOVERNOR_BLOCK_LIVE_DRAWDOWN_PCT = float(
    os.environ.get("CAPITAL_GOVERNOR_BLOCK_LIVE_DRAWDOWN_PCT", 0.16)
)
CAPITAL_GOVERNOR_CAUTION_PAPER_SHARPE = float(
    os.environ.get("CAPITAL_GOVERNOR_CAUTION_PAPER_SHARPE", 0.00)
)
CAPITAL_GOVERNOR_RISK_OFF_PAPER_SHARPE = float(
    os.environ.get("CAPITAL_GOVERNOR_RISK_OFF_PAPER_SHARPE", -0.35)
)
CAPITAL_GOVERNOR_CAUTION_LIVE_SHARPE = float(
    os.environ.get("CAPITAL_GOVERNOR_CAUTION_LIVE_SHARPE", 0.00)
)
CAPITAL_GOVERNOR_RISK_OFF_LIVE_SHARPE = float(
    os.environ.get("CAPITAL_GOVERNOR_RISK_OFF_LIVE_SHARPE", -0.25)
)
CAPITAL_GOVERNOR_CAUTION_DEGRADED_SOURCE_RATIO = float(
    os.environ.get("CAPITAL_GOVERNOR_CAUTION_DEGRADED_SOURCE_RATIO", 0.35)
)
CAPITAL_GOVERNOR_RISK_OFF_BLOCKED_SOURCE_RATIO = float(
    os.environ.get("CAPITAL_GOVERNOR_RISK_OFF_BLOCKED_SOURCE_RATIO", 0.30)
)
CAPITAL_GOVERNOR_LOW_REGIME_CONFIDENCE = float(
    os.environ.get("CAPITAL_GOVERNOR_LOW_REGIME_CONFIDENCE", 0.45)
)
CAPITAL_GOVERNOR_DIVERGENCE_BLOCKS = os.environ.get(
    "CAPITAL_GOVERNOR_DIVERGENCE_BLOCKS",
    "true",
).lower() in ("true", "1", "yes")
DECISION_CAPITAL_GOVERNOR_ENABLED = os.environ.get(
    "DECISION_CAPITAL_GOVERNOR_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
DECISION_CAPITAL_GOVERNOR_BLOCK_ON_STATUS = os.environ.get(
    "DECISION_CAPITAL_GOVERNOR_BLOCK_ON_STATUS",
    "true",
).lower() in ("true", "1", "yes")
EXPERIMENT_BENCHMARK_REPORT_PATH = os.environ.get(
    "EXPERIMENT_BENCHMARK_REPORT_PATH",
    "reports/experiment_benchmark_pack.json",
)
ADAPTIVE_LEARNING_ENABLED = os.environ.get(
    "ADAPTIVE_LEARNING_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
ADAPTIVE_LEARNING_LOOKBACK_HOURS = float(
    os.environ.get("ADAPTIVE_LEARNING_LOOKBACK_HOURS", 24 * 30)
)
ADAPTIVE_LEARNING_RECENT_LOOKBACK_HOURS = float(
    os.environ.get("ADAPTIVE_LEARNING_RECENT_LOOKBACK_HOURS", 24 * 3)
)
ADAPTIVE_LEARNING_REFRESH_INTERVAL_CYCLES = int(
    os.environ.get(
        "ADAPTIVE_LEARNING_REFRESH_INTERVAL_CYCLES",
        max(24, int(86400 / max(int(os.environ.get("TRADING_CYCLE_INTERVAL", 900)), 1))),
    )
)
ADAPTIVE_LEARNING_MIN_CLOSED_TRADES = int(
    os.environ.get("ADAPTIVE_LEARNING_MIN_CLOSED_TRADES", 5)
)
ADAPTIVE_LEARNING_MIN_RECENT_CLOSED_TRADES = int(
    os.environ.get("ADAPTIVE_LEARNING_MIN_RECENT_CLOSED_TRADES", 3)
)
ADAPTIVE_LEARNING_MIN_SELECTED_CANDIDATES = int(
    os.environ.get("ADAPTIVE_LEARNING_MIN_SELECTED_CANDIDATES", 5)
)
ADAPTIVE_LEARNING_CAUTION_HEALTH_FLOOR = float(
    os.environ.get("ADAPTIVE_LEARNING_CAUTION_HEALTH_FLOOR", 0.46)
)
ADAPTIVE_LEARNING_PROMOTION_HEALTH_FLOOR = float(
    os.environ.get("ADAPTIVE_LEARNING_PROMOTION_HEALTH_FLOOR", 0.70)
)
ADAPTIVE_LEARNING_CAUTION_DRIFT_THRESHOLD = float(
    os.environ.get("ADAPTIVE_LEARNING_CAUTION_DRIFT_THRESHOLD", 0.18)
)
ADAPTIVE_LEARNING_BLOCK_DRIFT_THRESHOLD = float(
    os.environ.get("ADAPTIVE_LEARNING_BLOCK_DRIFT_THRESHOLD", 0.33)
)
ADAPTIVE_LEARNING_MAX_CALIBRATION_ECE = float(
    os.environ.get("ADAPTIVE_LEARNING_MAX_CALIBRATION_ECE", 0.20)
)
ADAPTIVE_LEARNING_MIN_WEIGHT_MULTIPLIER = float(
    os.environ.get("ADAPTIVE_LEARNING_MIN_WEIGHT_MULTIPLIER", 0.12)
)
ADAPTIVE_LEARNING_RETURN_SCALE = float(
    os.environ.get("ADAPTIVE_LEARNING_RETURN_SCALE", 0.04)
)
ADAPTIVE_LEARNING_BLOCK_ON_STATUS = os.environ.get(
    "ADAPTIVE_LEARNING_BLOCK_ON_STATUS",
    "true",
).lower() in ("true", "1", "yes")
ADAPTIVE_LEARNING_MIN_HEALTH_SCORE = float(
    os.environ.get("ADAPTIVE_LEARNING_MIN_HEALTH_SCORE", 0.42)
)
ADAPTIVE_ARENA_MIN_TRADES = int(
    os.environ.get("ADAPTIVE_ARENA_MIN_TRADES", 10)
)
ADAPTIVE_ARENA_MIN_WIN_RATE = float(
    os.environ.get("ADAPTIVE_ARENA_MIN_WIN_RATE", 0.52)
)
ADAPTIVE_ARENA_MIN_SHARPE = float(
    os.environ.get("ADAPTIVE_ARENA_MIN_SHARPE", 0.05)
)
ADAPTIVE_ARENA_MAX_DRAWDOWN = float(
    os.environ.get("ADAPTIVE_ARENA_MAX_DRAWDOWN", 0.22)
)
SOURCE_BUDGET_ALLOCATOR_ENABLED = os.environ.get(
    "SOURCE_BUDGET_ALLOCATOR_ENABLED",
    "true",
).lower() in ("true", "1", "yes")
SOURCE_BUDGET_LOOKBACK_HOURS = float(
    os.environ.get("SOURCE_BUDGET_LOOKBACK_HOURS", 24 * 30)
)
SOURCE_BUDGET_REFRESH_INTERVAL_SECONDS = float(
    os.environ.get("SOURCE_BUDGET_REFRESH_INTERVAL_SECONDS", 300)
)
SOURCE_BUDGET_MIN_POSITION_PCT = float(
    os.environ.get("SOURCE_BUDGET_MIN_POSITION_PCT", PORTFOLIO_MIN_POSITION_PCT)
)
SOURCE_BUDGET_MIN_MULTIPLIER = float(
    os.environ.get("SOURCE_BUDGET_MIN_MULTIPLIER", 0.10)
)
SOURCE_BUDGET_MAX_MULTIPLIER = float(
    os.environ.get("SOURCE_BUDGET_MAX_MULTIPLIER", 1.20)
)
SOURCE_BUDGET_ACTIVE_MULTIPLIER = float(
    os.environ.get("SOURCE_BUDGET_ACTIVE_MULTIPLIER", 1.00)
)
SOURCE_BUDGET_WARMING_MULTIPLIER = float(
    os.environ.get("SOURCE_BUDGET_WARMING_MULTIPLIER", 0.78)
)
SOURCE_BUDGET_CAUTION_MULTIPLIER = float(
    os.environ.get("SOURCE_BUDGET_CAUTION_MULTIPLIER", 0.50)
)
SOURCE_BUDGET_BLOCKED_MULTIPLIER = float(
    os.environ.get("SOURCE_BUDGET_BLOCKED_MULTIPLIER", 0.0)
)
SOURCE_BUDGET_ACTIVE_CAP_PCT = float(
    os.environ.get("SOURCE_BUDGET_ACTIVE_CAP_PCT", 0.30)
)
SOURCE_BUDGET_WARMING_CAP_PCT = float(
    os.environ.get("SOURCE_BUDGET_WARMING_CAP_PCT", 0.16)
)
SOURCE_BUDGET_CAUTION_CAP_PCT = float(
    os.environ.get("SOURCE_BUDGET_CAUTION_CAP_PCT", 0.10)
)
SOURCE_BUDGET_BLOCKED_CAP_PCT = float(
    os.environ.get("SOURCE_BUDGET_BLOCKED_CAP_PCT", 0.0)
)
SOURCE_BUDGET_MIN_HEALTH_SCORE = float(
    os.environ.get("SOURCE_BUDGET_MIN_HEALTH_SCORE", 0.42)
)
SOURCE_BUDGET_MIN_CLOSED_TRADES = int(
    os.environ.get("SOURCE_BUDGET_MIN_CLOSED_TRADES", 3)
)
SOURCE_BUDGET_RETURN_SCALE = float(
    os.environ.get("SOURCE_BUDGET_RETURN_SCALE", 0.04)
)
SOURCE_BUDGET_LIVE_REJECTION_CEILING = float(
    os.environ.get("SOURCE_BUDGET_LIVE_REJECTION_CEILING", 0.25)
)
SOURCE_BUDGET_LIVE_FILL_FLOOR = float(
    os.environ.get("SOURCE_BUDGET_LIVE_FILL_FLOOR", 0.60)
)
SOURCE_BUDGET_BLOCK_ON_STATUS = os.environ.get(
    "SOURCE_BUDGET_BLOCK_ON_STATUS",
    "true",
).lower() in ("true", "1", "yes")
SOURCE_BUDGET_CAPITAL_GOVERNOR_ENABLED = os.environ.get(
    "SOURCE_BUDGET_CAPITAL_GOVERNOR_ENABLED",
    "true",
).lower() in ("true", "1", "yes")

# External-signal thresholds used before signals enter the decision engine.
POLYMARKET_MIN_DECISION_CONFIDENCE = float(
    os.environ.get("POLYMARKET_MIN_DECISION_CONFIDENCE", 0.60)
)
OPTIONS_FLOW_INJECTION_MIN_CONVICTION = float(
    os.environ.get("OPTIONS_FLOW_INJECTION_MIN_CONVICTION", 75.0)
)
OPTIONS_FLOW_DIRECT_MIN_CONVICTION = float(
    os.environ.get("OPTIONS_FLOW_DIRECT_MIN_CONVICTION", 80.0)
)
ARENA_MIN_FITNESS = float(os.environ.get("ARENA_MIN_FITNESS", 0.20))
ARENA_MIN_TRADES = int(os.environ.get("ARENA_MIN_TRADES", 12))

# ─── Scheduling ────────────────────────────────────────────────
# 3-tier scheduling:
#   Tier 1 — Fast cycle:   position checks, SL/TP, copy-trade scan
#   Tier 2 — Trading cycle: regime detection, scoring, paper trading, arena
#   Tier 3 — Discovery:     leaderboard scan, bot detection, strategy ID
FAST_CYCLE_INTERVAL = 60           # 1 minute — position management
TRADING_CYCLE_INTERVAL = int(os.environ.get("TRADING_CYCLE_INTERVAL", 900))   # 15 minutes — regime + trading (was 5 min, too frequent)
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
