"""
Configuration for the Hyperliquid Trading Research Bot.
"""
import os
import math


def _parse_coin_list(raw_value: str) -> list[str]:
    return [
        coin.strip().upper()
        for coin in (raw_value or "").split(",")
        if coin and coin.strip()
    ]

# ─── API Endpoints ─────────────────────────────────────────────
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_INFO_URL = f"{HYPERLIQUID_API_URL}/info"
HYPERLIQUID_EXCHANGE_URL = f"{HYPERLIQUID_API_URL}/exchange"

# ─── Database ──────────────────────────────────────────────────
# Priority: HL_BOT_DB env var > /data/ volume > local ./data/
# On Railway: set HL_BOT_DB=/data/bot.db in Variables tab, or the code
# auto-detects the /data volume if it exists and is writable.
def _can_use_persistent_volume() -> bool:
    """Return True when Railway-style /data persistence is actually available."""
    if os.name == "nt":
        return False

    data_dir = "/data"
    if not os.path.isdir(data_dir):
        return False

    try:
        probe = os.path.join(data_dir, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def _resolve_db_path() -> str:
    # 1. Explicit env var always wins
    env_db = os.environ.get("HL_BOT_DB")
    if env_db:
        return env_db
    # 2. Use Railway-style persistent volume only on supported platforms.
    if _can_use_persistent_volume():
        return "/data/bot.db"
    # 3. Fallback to local ./data/
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bot.db")

DB_PATH = _resolve_db_path()
_HAS_PERSISTENT_VOLUME = DB_PATH.startswith("/data")

# ─── Database Backend ────────────────────────────────────────
# "sqlite"    — all reads/writes go to SQLite (default, current behavior)
# "dualwrite" — writes to both SQLite and Postgres, reads from SQLite
# "postgres"  — all reads/writes go to Postgres
_raw_db_backend = os.environ.get("DB_BACKEND", "sqlite").strip().lower()
POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "").strip()
# Auto-downgrade to sqlite if Postgres backends are requested but no DSN is set.
# This prevents spamming warnings on deployments that don't have Postgres yet.
if _raw_db_backend in ("dualwrite", "postgres") and not POSTGRES_DSN:
    DB_BACKEND = "sqlite"
else:
    DB_BACKEND = _raw_db_backend
POSTGRES_POOL_MIN = int(os.environ.get("POSTGRES_POOL_MIN", 2))
POSTGRES_POOL_MAX = int(os.environ.get("POSTGRES_POOL_MAX", 10))
POSTGRES_STATEMENT_TIMEOUT_MS = int(os.environ.get("POSTGRES_STATEMENT_TIMEOUT_MS", 5000))
POSTGRES_APP_NAME = os.environ.get("POSTGRES_APP_NAME", "hyperliquid-bot").strip()

# ─── Feature Store (Postgres-only, auto-enabled when POSTGRES_DSN set) ─
FEATURE_STORE_COINS = os.environ.get("FEATURE_STORE_COINS", "").strip()
FEATURE_STORE_MAX_COINS = int(os.environ.get("FEATURE_STORE_MAX_COINS", 30))
FEATURE_STORE_BACKFILL_5M_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_5M_DAYS", 7))
FEATURE_STORE_BACKFILL_1H_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_1H_DAYS", 30))
FEATURE_STORE_BACKFILL_4H_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_4H_DAYS", 90))
FEATURE_STORE_BACKFILL_1D_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_1D_DAYS", 365))

# Dynamic risk policy engine
RISK_POLICY_DEFAULT_REWARD_MULTIPLE = float(
    os.environ.get("RISK_POLICY_DEFAULT_REWARD_MULTIPLE", 3.25)
)
RISK_POLICY_MIN_REWARD_MULTIPLE = float(
    os.environ.get("RISK_POLICY_MIN_REWARD_MULTIPLE", 1.75)
)
RISK_POLICY_MAX_REWARD_MULTIPLE = float(
    os.environ.get("RISK_POLICY_MAX_REWARD_MULTIPLE", 4.5)
)
RISK_POLICY_ATR_STOP_MULTIPLIER = float(
    os.environ.get("RISK_POLICY_ATR_STOP_MULTIPLIER", 1.0)
)
RISK_POLICY_MIN_STOP_ROE_PCT = float(
    os.environ.get("RISK_POLICY_MIN_STOP_ROE_PCT", 0.01)
)
RISK_POLICY_MAX_STOP_ROE_PCT = float(
    os.environ.get("RISK_POLICY_MAX_STOP_ROE_PCT", 0.15)
)
RISK_POLICY_MIN_STOP_PRICE_PCT = float(
    os.environ.get("RISK_POLICY_MIN_STOP_PRICE_PCT", 0.004)
)
RISK_POLICY_MAX_STOP_PRICE_PCT = float(
    os.environ.get("RISK_POLICY_MAX_STOP_PRICE_PCT", 0.025)
)
RISK_POLICY_MAX_TAKE_PROFIT_PRICE_PCT = float(
    os.environ.get("RISK_POLICY_MAX_TAKE_PROFIT_PRICE_PCT", 0.07)
)
RISK_POLICY_STOP_VOL_CAP_MULTIPLIER = float(
    os.environ.get("RISK_POLICY_STOP_VOL_CAP_MULTIPLIER", 2.5)
)
RISK_POLICY_TARGET_VOL_CAP_MULTIPLIER = float(
    os.environ.get("RISK_POLICY_TARGET_VOL_CAP_MULTIPLIER", 6.0)
)
RISK_POLICY_DEFAULT_TIME_LIMIT_HOURS = float(
    os.environ.get("RISK_POLICY_DEFAULT_TIME_LIMIT_HOURS", 18.0)
)
RISK_POLICY_DEFAULT_BREAKEVEN_AT_R = float(
    os.environ.get("RISK_POLICY_DEFAULT_BREAKEVEN_AT_R", 0.85)
)
RISK_POLICY_DEFAULT_BREAKEVEN_BUFFER_ROE_PCT = float(
    os.environ.get("RISK_POLICY_DEFAULT_BREAKEVEN_BUFFER_ROE_PCT", 0.005)
)
RISK_POLICY_DEFAULT_TRAIL_AFTER_R = float(
    os.environ.get("RISK_POLICY_DEFAULT_TRAIL_AFTER_R", 1.35)
)
RISK_POLICY_DEFAULT_TRAILING_DISTANCE_RATIO = float(
    os.environ.get("RISK_POLICY_DEFAULT_TRAILING_DISTANCE_RATIO", 0.65)
)
RISK_POLICY_SOURCE_PROFILES_JSON = os.environ.get(
    "RISK_POLICY_SOURCE_PROFILES_JSON",
    "",
).strip()

# ─── Macro Regime Overlay ────────────────────────────────────────
# Protective regime that scrapes external macro sources and adjusts risk posture
MACRO_REGIME_ENABLED = os.environ.get("MACRO_REGIME_ENABLED", "true").lower() in ("true", "1", "yes")
MACRO_REGIME_REFRESH_SECONDS = int(os.environ.get("MACRO_REGIME_REFRESH_SECONDS", 900))
MACRO_REGIME_BLOCK_AT_LEVEL = os.environ.get("MACRO_REGIME_BLOCK_AT_LEVEL", "extreme").strip()

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
MIN_STRATEGY_SCORE = 0.05
# Keep at least top-N strategies active even when all scores are weak, to avoid
# complete strategy starvation during cold-start or rough regimes.
MIN_ACTIVE_STRATEGIES = int(os.environ.get("MIN_ACTIVE_STRATEGIES", 5))
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
# Paper-trading risk is defined in ROE space, then converted back into raw
# trigger prices by dividing by leverage. Take-profit is always kept at 5x the
# configured stop-loss so every paper signal uses the same reward-to-risk shape.
PAPER_TRADING_STOP_LOSS_PCT = float(os.environ.get("PAPER_TRADING_STOP_LOSS_PCT", 0.15))
PAPER_TRADING_TAKE_PROFIT_PCT = PAPER_TRADING_STOP_LOSS_PCT * 5.0
PAPER_TRADING_MAKER_FEE_BPS = float(os.environ.get("PAPER_TRADING_MAKER_FEE_BPS", 0.2))
PAPER_TRADING_TAKER_FEE_BPS = float(os.environ.get("PAPER_TRADING_TAKER_FEE_BPS", 2.5))
PAPER_TRADING_DEFAULT_EXECUTION_ROLE = os.environ.get(
    "PAPER_TRADING_DEFAULT_EXECUTION_ROLE", "taker"
).lower()
# Simulated slippage range applied to paper market orders (basis points).
PAPER_TRADING_SLIPPAGE_MIN_BPS = float(os.environ.get("PAPER_TRADING_SLIPPAGE_MIN_BPS", 1.0))
PAPER_TRADING_SLIPPAGE_MAX_BPS = float(os.environ.get("PAPER_TRADING_SLIPPAGE_MAX_BPS", 5.0))
# Accrue Hyperliquid 8h funding payments on open paper positions.
PAPER_TRADING_FUNDING_ENABLED = os.environ.get(
    "PAPER_TRADING_FUNDING_ENABLED", "true"
).lower() in ("true", "1", "yes")

# Live trading wallet / secret-management controls.
# Agent-wallet-only mode: signer key must be for a delegated agent wallet, and
# HL_PUBLIC_ADDRESS points to the trading account (master/vault) being managed.
LIVE_TRADING_ENABLED = os.environ.get(
    "LIVE_TRADING_ENABLED", "false"
).lower() in ("true", "1", "yes")
LIVE_TRADING_DUAL_CONTROL_CONFIRM = os.environ.get(
    "LIVE_TRADING_DUAL_CONTROL_CONFIRM", "false"
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
LIVE_MAX_ORDER_USD = float(os.environ.get("LIVE_MAX_ORDER_USD", 100.0))
LIVE_MAX_POSITION_SIZE_USD = float(
    os.environ.get(
        "LIVE_MAX_POSITION_SIZE_USD",
        os.environ.get("HL_MAX_POSITION_SIZE", str(LIVE_MAX_ORDER_USD)),
    )
)
# Daily loss limit for the live account in USD (forwarded to LiveTrader).
LIVE_MAX_DAILY_LOSS_USD = float(os.environ.get("LIVE_MAX_DAILY_LOSS_USD", 100.0))
LIVE_CANARY_MODE = os.environ.get(
    "LIVE_CANARY_MODE", "false"
).lower() in ("true", "1", "yes")
LIVE_CANARY_MAX_ORDER_USD = float(os.environ.get("LIVE_CANARY_MAX_ORDER_USD", 25.0))
LIVE_CANARY_MAX_SIGNALS_PER_DAY = int(os.environ.get("LIVE_CANARY_MAX_SIGNALS_PER_DAY", 25))
LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY = int(
    os.environ.get("LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 0)
)
LIVE_MIN_ORDER_TOP_TIER_ENABLED = os.environ.get(
    "LIVE_MIN_ORDER_TOP_TIER_ENABLED", "true"
).lower() in ("true", "1", "yes")
LIVE_MIN_ORDER_TOP_TIER_MIN_CONFIDENCE = float(
    os.environ.get("LIVE_MIN_ORDER_TOP_TIER_MIN_CONFIDENCE", 0.72)
)
LIVE_MIN_ORDER_TOP_TIER_MAX_BUMP_MULTIPLIER = float(
    os.environ.get("LIVE_MIN_ORDER_TOP_TIER_MAX_BUMP_MULTIPLIER", 1.35)
)
LIVE_MIN_ORDER_ALLOW_DEGRADED_SOURCES = os.environ.get(
    "LIVE_MIN_ORDER_ALLOW_DEGRADED_SOURCES", "false"
).lower() in ("true", "1", "yes")
LIVE_MIN_ORDER_SAME_SIDE_MERGE_ENABLED = os.environ.get(
    "LIVE_MIN_ORDER_SAME_SIDE_MERGE_ENABLED", "true"
).lower() in ("true", "1", "yes")
LIVE_MIN_ORDER_SAME_SIDE_MAX_BUMP_MULTIPLIER = float(
    os.environ.get("LIVE_MIN_ORDER_SAME_SIDE_MAX_BUMP_MULTIPLIER", 2.5)
)
LIVE_ANALYTICS_LOOKBACK_TRADES = int(os.environ.get("LIVE_ANALYTICS_LOOKBACK_TRADES", 200))
COPY_TRADER_ENABLED = os.environ.get(
    "COPY_TRADER_ENABLED", "true"
).lower() in ("true", "1", "yes")
COPY_TRADER_MAX_CONCURRENT_TRADES = int(
    os.environ.get("COPY_TRADER_MAX_CONCURRENT_TRADES", 2)
)
COPY_TRADER_MAX_NEW_TRADES_PER_CYCLE = int(
    os.environ.get("COPY_TRADER_MAX_NEW_TRADES_PER_CYCLE", 1)
)
COPY_TRADER_AUTO_PAUSE_MIN_CLOSED_TRADES = int(
    os.environ.get("COPY_TRADER_AUTO_PAUSE_MIN_CLOSED_TRADES", 6)
)
COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE = float(
    os.environ.get("COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE", 0.40)
)
COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE = float(
    os.environ.get("COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE", 0.25)
)
COPY_TRADER_AUTO_PAUSE_BLOCK_NET_PNL = float(
    os.environ.get("COPY_TRADER_AUTO_PAUSE_BLOCK_NET_PNL", -25.0)
)
LIVE_EXTERNAL_KILL_SWITCH_FILE = os.environ.get("LIVE_EXTERNAL_KILL_SWITCH_FILE", "").strip()
LIVE_KILL_SWITCH_STATE_FILE = os.environ.get("LIVE_KILL_SWITCH_STATE_FILE", "/data/live_kill_switch_state.json").strip()
RUNTIME_CONFIG_OVERRIDE_FILE = os.environ.get("RUNTIME_CONFIG_OVERRIDE_FILE", "/data/config.json").strip()
RUNTIME_CONFIG_POLL_SECONDS = int(os.environ.get("RUNTIME_CONFIG_POLL_SECONDS", 10))
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
ROTATION_ENGINE_ENABLED = os.environ.get(
    "ROTATION_ENGINE_ENABLED", "true"
).lower() in ("true", "1", "yes")
# Dry-run telemetry: when true alongside ROTATION_ENGINE_ENABLED, rotations
# are simulated (logged but not executed).  Default off so rotations are live.
ROTATION_DRY_RUN_TELEMETRY = os.environ.get(
    "ROTATION_DRY_RUN_TELEMETRY", "false"
).lower() in ("true", "1", "yes")
ROTATION_SHADOW_MODE_DAYS = int(os.environ.get("ROTATION_SHADOW_MODE_DAYS", "0"))
ROTATION_REQUIRE_EXPLICIT_THRESHOLDS = os.environ.get(
    "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", "false"
).lower() in ("true", "1", "yes")

# ─── Decision Firewall ─────────────────────────────────────────
# Minimum signal confidence to pass the firewall.
# 0.15 (15%) is far too permissive — nearly any signal passes.
# Raise to 0.45+ to ensure only well-confirmed signals are traded.
FIREWALL_MIN_CONFIDENCE = float(os.environ.get("FIREWALL_MIN_CONFIDENCE", 0.45))
FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY = int(
    os.environ.get("FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY", 0)
)
SHORT_HARDENING_ENABLED = os.environ.get("SHORT_HARDENING_ENABLED", "true").lower() in ("true", "1", "yes")
SHORT_HARDENING_LOOKBACK_TRADES = int(os.environ.get("SHORT_HARDENING_LOOKBACK_TRADES", 120))
SHORT_HARDENING_MIN_CLOSED_TRADES = int(os.environ.get("SHORT_HARDENING_MIN_CLOSED_TRADES", 12))
SHORT_HARDENING_DEGRADE_WIN_RATE = float(os.environ.get("SHORT_HARDENING_DEGRADE_WIN_RATE", 0.48))
SHORT_HARDENING_BLOCK_WIN_RATE = float(os.environ.get("SHORT_HARDENING_BLOCK_WIN_RATE", 0.40))
SHORT_HARDENING_BLOCK_NET_PNL = float(os.environ.get("SHORT_HARDENING_BLOCK_NET_PNL", -0.5))
SHORT_HARDENING_CONFIDENCE_MULTIPLIER = float(
    os.environ.get("SHORT_HARDENING_CONFIDENCE_MULTIPLIER", 0.80)
)
SHORT_HARDENING_SIZE_MULTIPLIER = float(os.environ.get("SHORT_HARDENING_SIZE_MULTIPLIER", 0.50))
FIREWALL_COIN_COOLDOWN_SECONDS = int(os.environ.get("FIREWALL_COIN_COOLDOWN_SECONDS", 180))
FIREWALL_SAME_SIDE_COOLDOWN_SECONDS = int(
    os.environ.get("FIREWALL_SAME_SIDE_COOLDOWN_SECONDS", 900)
)
FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN = int(
    os.environ.get("FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN", 2)
)
FIREWALL_CANARY_MODE = os.environ.get(
    "FIREWALL_CANARY_MODE", "false"
).lower() in ("true", "1", "yes")
FIREWALL_CANARY_MAX_POSITIONS = int(
    os.environ.get("FIREWALL_CANARY_MAX_POSITIONS", 2)
)

# Per-source capital allocator / throttling.
SOURCE_POLICY_ENABLED = os.environ.get(
    "SOURCE_POLICY_ENABLED", "true"
).lower() in ("true", "1", "yes")
SOURCE_POLICY_MIN_CLOSED_TRADES = int(
    os.environ.get("SOURCE_POLICY_MIN_CLOSED_TRADES", 3)
)
SOURCE_POLICY_KEEP_TOP_N = int(os.environ.get("SOURCE_POLICY_KEEP_TOP_N", 5))
SOURCE_POLICY_PAUSE_WEIGHT = float(
    os.environ.get("SOURCE_POLICY_PAUSE_WEIGHT", 0.12)
)
SOURCE_POLICY_DEGRADE_WEIGHT = float(
    os.environ.get("SOURCE_POLICY_DEGRADE_WEIGHT", 0.32)
)
SOURCE_POLICY_WARMUP_MAX_SIGNALS_PER_DAY = int(
    os.environ.get("SOURCE_POLICY_WARMUP_MAX_SIGNALS_PER_DAY", 1)
)
SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY = int(
    os.environ.get("SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY", 1)
)
SOURCE_POLICY_WARMUP_SIZE_MULTIPLIER = float(
    os.environ.get("SOURCE_POLICY_WARMUP_SIZE_MULTIPLIER", 0.75)
)
SOURCE_POLICY_DEGRADED_SIZE_MULTIPLIER = float(
    os.environ.get("SOURCE_POLICY_DEGRADED_SIZE_MULTIPLIER", 0.60)
)
SOURCE_POLICY_WARMUP_MIN_CONFIDENCE = float(
    os.environ.get("SOURCE_POLICY_WARMUP_MIN_CONFIDENCE", 0.45)
)
SOURCE_POLICY_DEGRADED_MIN_CONFIDENCE = float(
    os.environ.get("SOURCE_POLICY_DEGRADED_MIN_CONFIDENCE", 0.55)
)

# Runtime readiness / incident monitoring.
READINESS_STALE_SECONDS = int(os.environ.get("READINESS_STALE_SECONDS", 600))
READINESS_DB_WRITE_TTL_S = int(os.environ.get("READINESS_DB_WRITE_TTL_S", 60))
READINESS_ALERT_COOLDOWN_S = int(
    os.environ.get("READINESS_ALERT_COOLDOWN_S", 900)
)

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

# Arena champion bootstrap controls.
ARENA_CHAMPION_MIN_FITNESS = float(os.environ.get("ARENA_CHAMPION_MIN_FITNESS", 0.15))
ARENA_CHAMPION_MIN_TRADES = int(os.environ.get("ARENA_CHAMPION_MIN_TRADES", 5))
ARENA_CHAMPION_MIN_WIN_RATE = float(os.environ.get("ARENA_CHAMPION_MIN_WIN_RATE", 0.45))
ARENA_COIN_UNIVERSE = _parse_coin_list(
    os.environ.get("ARENA_COIN_UNIVERSE", "").strip() or FEATURE_STORE_COINS
)
if not ARENA_COIN_UNIVERSE:
    ARENA_COIN_UNIVERSE = ["BTC", "ETH", "SOL"]
ARENA_MAX_COINS = int(os.environ.get("ARENA_MAX_COINS", 3))
ARENA_INTERVAL = os.environ.get("ARENA_INTERVAL", "1h").strip() or "1h"
ARENA_LOOKBACK_HOURS = int(os.environ.get("ARENA_LOOKBACK_HOURS", 720))

# Options-flow conviction gate (0-100).
OPTIONS_FLOW_MIN_CONVICTION_PCT = float(
    os.environ.get("OPTIONS_FLOW_MIN_CONVICTION_PCT", 30.0)
)

# ─── XGBoost Forecaster (optional ML upgrade) ─────────────────
ENABLE_XGBOOST_FORECASTER = os.environ.get("ENABLE_XGBOOST_FORECASTER", "true").lower() in ("true", "1", "yes")
XGBOOST_MODEL_PATH = "models/regime_xgboost.json"
XGBOOST_CRASH_THRESHOLD = float(os.environ.get("XGBOOST_CRASH_THRESHOLD", -0.18))
XGBOOST_MIN_CONFIDENCE = float(os.environ.get("XGBOOST_MIN_CONFIDENCE", 0.52))
XGBOOST_RETRAIN_INTERVAL = int(os.environ.get("XGBOOST_RETRAIN_INTERVAL", 86400))  # 24h walk-forward

# --- Feature Store Alpha Pipeline (Phase B) ---
ENABLE_ALPHA_PIPELINE = os.environ.get("ENABLE_ALPHA_PIPELINE", "true").lower() in ("true", "1", "yes")
ALPHA_TIMEFRAME = os.environ.get("ALPHA_TIMEFRAME", "1h")
ALPHA_LOOKBACK_DAYS = int(os.environ.get("ALPHA_LOOKBACK_DAYS", 120))
ALPHA_MIN_TRAINING_SAMPLES = int(os.environ.get("ALPHA_MIN_TRAINING_SAMPLES", 250))
ALPHA_RETRAIN_INTERVAL = int(os.environ.get("ALPHA_RETRAIN_INTERVAL", 21600))
ALPHA_WALK_FORWARD_SPLITS = int(os.environ.get("ALPHA_WALK_FORWARD_SPLITS", 5))
ALPHA_LABEL_MIN_ABS_RETURN = float(os.environ.get("ALPHA_LABEL_MIN_ABS_RETURN", 0.0005))
ALPHA_SIGNAL_MIN_CONFIDENCE = float(os.environ.get("ALPHA_SIGNAL_MIN_CONFIDENCE", 0.58))
ALPHA_MIN_SIGNIFICANT_TRADES = int(os.environ.get("ALPHA_MIN_SIGNIFICANT_TRADES", 60))
ALPHA_MIN_SIGNIFICANCE_PVALUE = float(os.environ.get("ALPHA_MIN_SIGNIFICANCE_PVALUE", 0.10))
ALPHA_MAX_PREDICTION_COINS = int(os.environ.get("ALPHA_MAX_PREDICTION_COINS", 12))
ALPHA_CACHE_TTL = int(os.environ.get("ALPHA_CACHE_TTL", 180))
ALPHA_MODEL_DIR = os.environ.get("ALPHA_MODEL_DIR", "models/alpha_direction")

# ─── LSTM Alpha Agent ────────────────────────────────────────
ENABLE_LSTM_AGENT = os.environ.get("ENABLE_LSTM_AGENT", "true").lower() in ("true", "1", "yes")
LSTM_SEQUENCE_LENGTH = int(os.environ.get("LSTM_SEQUENCE_LENGTH", 30))
LSTM_HIDDEN_SIZE = int(os.environ.get("LSTM_HIDDEN_SIZE", 64))
LSTM_RETRAIN_INTERVAL = int(os.environ.get("LSTM_RETRAIN_INTERVAL", 21600))  # 6 hours
LSTM_MODEL_DIR = os.environ.get("LSTM_MODEL_DIR", "models/lstm_direction")

# ─── RL Position Sizer ──────────────────────────────────────
ENABLE_RL_SIZER = os.environ.get("ENABLE_RL_SIZER", "true").lower() in ("true", "1", "yes")
RL_SIZER_RETRAIN_INTERVAL = int(os.environ.get("RL_SIZER_RETRAIN_INTERVAL", 43200))  # 12 hours
RL_SIZER_TRAINING_EPISODES = int(os.environ.get("RL_SIZER_TRAINING_EPISODES", 500))
RL_SIZER_MODEL_DIR = os.environ.get("RL_SIZER_MODEL_DIR", "models/rl_sizer")

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
POLYMARKET_MIN_LIQUIDITY = float(
    os.environ.get("POLYMARKET_MIN_LIQUIDITY", 1000)
)  # $1k min liquidity
POLYMARKET_MAX_MARKETS_PER_SCAN = int(
    os.environ.get("POLYMARKET_MAX_MARKETS_PER_SCAN", 100)
)

# ─── Options Flow Integration ───────────────────────────────
OPTIONS_FLOW_ENABLED = os.environ.get("OPTIONS_FLOW_ENABLED", "true").lower() in ("true", "1", "yes")
OPTIONS_FLOW_SCAN_INTERVAL = int(os.environ.get("OPTIONS_FLOW_SCAN_INTERVAL", 120))  # 2 minutes

# ─── Structured Event Scanner ──────────────────────────────
EVENT_SCANNER_ENABLED = os.environ.get("EVENT_SCANNER_ENABLED", "true").lower() in ("true", "1", "yes")
EVENT_SCANNER_LOOKAHEAD_DAYS = int(os.environ.get("EVENT_SCANNER_LOOKAHEAD_DAYS", 14))
EVENT_SCANNER_RECENT_HOURS = int(os.environ.get("EVENT_SCANNER_RECENT_HOURS", 72))
EVENT_SCANNER_REFRESH_SECONDS = int(os.environ.get("EVENT_SCANNER_REFRESH_SECONDS", 900))
EVENT_SCANNER_MAX_UPCOMING = int(os.environ.get("EVENT_SCANNER_MAX_UPCOMING", 12))
EVENT_SCANNER_MAX_RECENT = int(os.environ.get("EVENT_SCANNER_MAX_RECENT", 12))
EVENT_SCANNER_INCLUDE_MEDIUM = os.environ.get(
    "EVENT_SCANNER_INCLUDE_MEDIUM", "true"
).lower() in ("true", "1", "yes")
EVENT_SCANNER_ENABLE_CRYPTO_INCIDENTS = os.environ.get(
    "EVENT_SCANNER_ENABLE_CRYPTO_INCIDENTS", "true"
).lower() in ("true", "1", "yes")
EVENT_RISK_ENABLED = os.environ.get("EVENT_RISK_ENABLED", "true").lower() in ("true", "1", "yes")
EVENT_RISK_BLOCK_MINUTES = int(os.environ.get("EVENT_RISK_BLOCK_MINUTES", 10))
EVENT_RISK_COOLDOWN_MINUTES = int(os.environ.get("EVENT_RISK_COOLDOWN_MINUTES", 30))
EVENT_RISK_DEGRADE_LOOKAHEAD_MINUTES = int(
    os.environ.get("EVENT_RISK_DEGRADE_LOOKAHEAD_MINUTES", 60)
)
EVENT_RISK_CONFIDENCE_MULTIPLIER = float(
    os.environ.get("EVENT_RISK_CONFIDENCE_MULTIPLIER", 0.65)
)
EVENT_RISK_SIZE_MULTIPLIER = float(os.environ.get("EVENT_RISK_SIZE_MULTIPLIER", 0.60))

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


def _warn_config(msg: str) -> None:
    # Boot logging may not be configured yet.
    print(f"[config] {msg}")


def _validate_numeric_bounds(name: str, min_value: float, max_value: float, fallback):
    value = globals().get(name, fallback)
    if isinstance(value, bool):
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        _warn_config(f"Invalid {name}={value!r}; using fallback {fallback}.")
        globals()[name] = fallback
        return
    if not math.isfinite(numeric):
        _warn_config(f"Non-finite {name}={value!r}; using fallback {fallback}.")
        globals()[name] = fallback
        return
    clamped = min(max(numeric, min_value), max_value)
    if clamped != numeric:
        _warn_config(
            f"{name}={numeric} out of range [{min_value}, {max_value}] "
            f"-> clamped to {clamped}."
        )
    if isinstance(fallback, int):
        globals()[name] = int(clamped)
    else:
        globals()[name] = float(clamped)


def _validate_config_bounds() -> None:
    """Best-effort guardrails for env-configurable numeric settings."""
    if DB_BACKEND not in {"sqlite", "dualwrite", "postgres"}:
        _warn_config(f"Invalid DB_BACKEND={DB_BACKEND!r}; using 'sqlite'.")
        globals()["DB_BACKEND"] = "sqlite"

    rules = [
        ("MIN_STRATEGY_SCORE", 0.0, 1.0, 0.05),
        ("MAX_ACTIVE_STRATEGIES", 1, 5000, 200),
        ("MIN_ACTIVE_STRATEGIES", 1, 500, 5),
        ("MAX_STRATEGIES_PER_CYCLE", 1, 200, 15),
        ("PAPER_TRADING_MAX_LEVERAGE", 1.0, 25.0, 5.0),
        ("PAPER_TRADING_STOP_LOSS_PCT", 0.001, 1.0, 0.15),
        ("PAPER_TRADING_TAKE_PROFIT_PCT", 0.001, 5.0, 0.75),
        ("LIVE_MIN_ORDER_USD", 10.0, 1_000_000.0, 11.0),
        ("LIVE_MAX_ORDER_USD", 10.0, 1_000_000.0, 100.0),
        ("LIVE_MAX_POSITION_SIZE_USD", 10.0, 10_000_000.0, 100.0),
        ("LIVE_MAX_DAILY_LOSS_USD", 1.0, 10_000_000.0, 100.0),
        ("PORTFOLIO_TARGET_POSITIONS", 1, 100, 8),
        ("PORTFOLIO_HARD_MAX_POSITIONS", 1, 200, 10),
        ("PORTFOLIO_RESERVED_HIGH_CONVICTION_SLOTS", 0, 50, 2),
        ("PORTFOLIO_HIGH_CONVICTION_THRESHOLD", 0.0, 1.0, 0.78),
        ("PORTFOLIO_REPLACEMENT_THRESHOLD", 0.0, 1.0, 0.15),
        ("PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE", 0, 50, 1),
        ("PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR", 0, 200, 4),
        ("PORTFOLIO_MAX_REPLACEMENTS_PER_DAY", 0, 500, 12),
        ("PORTFOLIO_MAX_COIN_EXPOSURE_PCT", 0.0, 1.0, 0.45),
        ("PORTFOLIO_MAX_SIDE_EXPOSURE_PCT", 0.0, 1.0, 0.65),
        ("PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT", 0.0, 1.0, 0.55),
        ("FIREWALL_MIN_CONFIDENCE", 0.0, 1.0, 0.45),
        ("FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY", 0, 100_000, 0),
        ("FIREWALL_COIN_COOLDOWN_SECONDS", 0, 86_400, 180),
        ("FIREWALL_SAME_SIDE_COOLDOWN_SECONDS", 0, 86_400, 900),
        ("FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN", 1, 20, 2),
        ("FIREWALL_CANARY_MAX_POSITIONS", 1, 100, 2),
        ("SOURCE_POLICY_MIN_CLOSED_TRADES", 1, 1000, 3),
        ("SOURCE_POLICY_KEEP_TOP_N", 1, 1000, 5),
        ("SOURCE_POLICY_PAUSE_WEIGHT", 0.0, 1.0, 0.12),
        ("SOURCE_POLICY_DEGRADE_WEIGHT", 0.0, 1.0, 0.32),
        ("SOURCE_POLICY_WARMUP_MAX_SIGNALS_PER_DAY", 0, 100_000, 1),
        ("SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY", 0, 100_000, 1),
        ("SOURCE_POLICY_WARMUP_SIZE_MULTIPLIER", 0.0, 1.0, 0.75),
        ("SOURCE_POLICY_DEGRADED_SIZE_MULTIPLIER", 0.0, 1.0, 0.60),
        ("SOURCE_POLICY_WARMUP_MIN_CONFIDENCE", 0.0, 1.0, 0.45),
        ("SOURCE_POLICY_DEGRADED_MIN_CONFIDENCE", 0.0, 1.0, 0.55),
        ("TRADING_CYCLE_INTERVAL", 10, 86_400, 900),
        ("DISCOVERY_CYCLE_INTERVAL", 60, 2_592_000, 86400),
        ("POLYMARKET_SCAN_INTERVAL", 10, 3600, 180),
        ("POLYMARKET_MAX_MARKETS_PER_SCAN", 10, 10_000, 100),
        ("OPTIONS_FLOW_SCAN_INTERVAL", 10, 3600, 120),
        ("RISK_POLICY_DEFAULT_REWARD_MULTIPLE", 0.5, 20.0, 3.25),
        ("RISK_POLICY_MIN_REWARD_MULTIPLE", 0.1, 10.0, 1.75),
        ("RISK_POLICY_MAX_REWARD_MULTIPLE", 0.1, 50.0, 4.5),
        ("RISK_POLICY_ATR_STOP_MULTIPLIER", 0.1, 10.0, 1.0),
        ("RISK_POLICY_MIN_STOP_ROE_PCT", 0.0001, 1.0, 0.01),
        ("RISK_POLICY_MAX_STOP_ROE_PCT", 0.0001, 5.0, 0.15),
        ("RISK_POLICY_MIN_STOP_PRICE_PCT", 0.0001, 1.0, 0.004),
        ("RISK_POLICY_MAX_STOP_PRICE_PCT", 0.0001, 1.0, 0.025),
        ("RISK_POLICY_MAX_TAKE_PROFIT_PRICE_PCT", 0.0001, 5.0, 0.07),
        ("RISK_POLICY_STOP_VOL_CAP_MULTIPLIER", 0.1, 20.0, 2.5),
        ("RISK_POLICY_TARGET_VOL_CAP_MULTIPLIER", 0.1, 50.0, 6.0),
        ("RISK_POLICY_DEFAULT_TIME_LIMIT_HOURS", 0.25, 24 * 30, 18.0),
        ("RISK_POLICY_DEFAULT_BREAKEVEN_AT_R", 0.0, 20.0, 0.85),
        ("RISK_POLICY_DEFAULT_BREAKEVEN_BUFFER_ROE_PCT", 0.0, 1.0, 0.005),
        ("RISK_POLICY_DEFAULT_TRAIL_AFTER_R", 0.0, 20.0, 1.35),
        ("RISK_POLICY_DEFAULT_TRAILING_DISTANCE_RATIO", 0.0, 5.0, 0.65),
        ("EVENT_SCANNER_LOOKAHEAD_DAYS", 1, 90, 14),
        ("EVENT_SCANNER_RECENT_HOURS", 1, 720, 72),
        ("EVENT_SCANNER_REFRESH_SECONDS", 60, 86_400, 900),
        ("EVENT_SCANNER_MAX_UPCOMING", 1, 200, 12),
        ("EVENT_SCANNER_MAX_RECENT", 1, 200, 12),
        ("EVENT_RISK_BLOCK_MINUTES", 0, 1_440, 10),
        ("EVENT_RISK_COOLDOWN_MINUTES", 0, 1_440, 30),
        ("EVENT_RISK_DEGRADE_LOOKAHEAD_MINUTES", 0, 2_880, 60),
        ("EVENT_RISK_CONFIDENCE_MULTIPLIER", 0.0, 1.0, 0.65),
        ("EVENT_RISK_SIZE_MULTIPLIER", 0.0, 1.0, 0.60),
        ("FORECASTER_EXTERNAL_DATA_TTL", 10, 86_400, 600),
        ("ARENA_CHAMPION_MIN_FITNESS", 0.0, 1.0, 0.15),
        ("ARENA_CHAMPION_MIN_TRADES", 1, 500, 5),
        ("ARENA_CHAMPION_MIN_WIN_RATE", 0.0, 1.0, 0.45),
        ("ARENA_MAX_COINS", 1, 100, 3),
        ("ARENA_LOOKBACK_HOURS", 24, 8760, 720),
        ("OPTIONS_FLOW_MIN_CONVICTION_PCT", 0.0, 100.0, 30.0),
        ("XGBOOST_MIN_CONFIDENCE", 0.0, 1.0, 0.52),
        ("XGBOOST_RETRAIN_INTERVAL", 60, 2_592_000, 86400),
        ("ALPHA_LOOKBACK_DAYS", 7, 3650, 120),
        ("ALPHA_MIN_TRAINING_SAMPLES", 50, 100_000, 250),
        ("ALPHA_RETRAIN_INTERVAL", 300, 2_592_000, 21600),
        ("ALPHA_WALK_FORWARD_SPLITS", 2, 20, 5),
        ("ALPHA_LABEL_MIN_ABS_RETURN", 0.0, 1.0, 0.0005),
        ("ALPHA_SIGNAL_MIN_CONFIDENCE", 0.0, 1.0, 0.58),
        ("ALPHA_MIN_SIGNIFICANT_TRADES", 1, 100_000, 60),
        ("ALPHA_MIN_SIGNIFICANCE_PVALUE", 0.0, 1.0, 0.10),
        ("ALPHA_MAX_PREDICTION_COINS", 1, 500, 12),
        ("ALPHA_CACHE_TTL", 5, 86_400, 180),
        ("KELLY_MULTIPLIER", 0.0, 1.0, 0.25),
        ("MONTE_CARLO_PATHS", 100, 200_000, 5000),
        # Previously unvalidated float/int env vars:
        ("PAPER_TRADING_MAKER_FEE_BPS", 0.0, 100.0, 0.2),
        ("PAPER_TRADING_TAKER_FEE_BPS", 0.0, 100.0, 2.5),
        ("BOT_MM_PNL_THRESHOLD", -1e6, 1e6, 0.0),
        ("BOT_HARD_CUTOFF_TRADES", 1, 100_000, 100),
        ("BOT_THRESHOLD", 1, 100, 3),
        ("BOT_ELEVATED_FREQ", 1, 100_000, 50),
        ("PORTFOLIO_CHURN_PENALTY", 0.0, 1.0, 0.02),
        ("PORTFOLIO_MIN_HOLD_MINUTES", 0, 525_600, 60),
        ("ROTATION_SHADOW_MODE_DAYS", 0, 365, 7),
        ("FORECASTER_CRASH_THRESHOLD", -1.0, 0.0, -0.15),
        ("XGBOOST_CRASH_THRESHOLD", -1.0, 0.0, -0.18),
        ("FUNDING_NEGATIVE_THRESHOLD", -1.0, 0.0, -0.001),
        ("FUNDING_POSITIVE_THRESHOLD", 0.0, 1.0, 0.003),
        ("POLYMARKET_MIN_VOLUME", 0.0, 1e9, 10_000.0),
        ("POLYMARKET_MIN_LIQUIDITY", 0.0, 1e9, 1_000.0),
        ("LIVE_CANARY_MAX_ORDER_USD", 10.0, 1_000_000.0, 25.0),
        ("LIVE_CANARY_MAX_SIGNALS_PER_DAY", 1, 100_000, 25),
        ("LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 0, 100_000, 0),
        ("LIVE_MIN_ORDER_TOP_TIER_MIN_CONFIDENCE", 0.0, 1.0, 0.72),
        ("LIVE_MIN_ORDER_TOP_TIER_MAX_BUMP_MULTIPLIER", 1.0, 10.0, 1.35),
        ("LIVE_MIN_ORDER_SAME_SIDE_MAX_BUMP_MULTIPLIER", 1.0, 10.0, 2.5),
        ("LIVE_ANALYTICS_LOOKBACK_TRADES", 10, 5_000, 200),
        ("COPY_TRADER_MAX_CONCURRENT_TRADES", 0, 100, 2),
        ("COPY_TRADER_MAX_NEW_TRADES_PER_CYCLE", 0, 100, 1),
        ("COPY_TRADER_AUTO_PAUSE_MIN_CLOSED_TRADES", 1, 5_000, 6),
        ("COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE", 0.0, 1.0, 0.40),
        ("COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE", 0.0, 1.0, 0.25),
        ("COPY_TRADER_AUTO_PAUSE_BLOCK_NET_PNL", -1_000_000.0, 1_000_000.0, -25.0),
        ("SHORT_HARDENING_LOOKBACK_TRADES", 10, 5_000, 120),
        ("SHORT_HARDENING_MIN_CLOSED_TRADES", 1, 1_000, 12),
        ("SHORT_HARDENING_DEGRADE_WIN_RATE", 0.0, 1.0, 0.48),
        ("SHORT_HARDENING_BLOCK_WIN_RATE", 0.0, 1.0, 0.40),
        ("SHORT_HARDENING_BLOCK_NET_PNL", -1_000_000.0, 1_000_000.0, -0.5),
        ("SHORT_HARDENING_CONFIDENCE_MULTIPLIER", 0.0, 1.0, 0.80),
        ("SHORT_HARDENING_SIZE_MULTIPLIER", 0.0, 1.0, 0.50),
        ("READINESS_STALE_SECONDS", 30, 86_400, 600),
        ("READINESS_DB_WRITE_TTL_S", 1, 3_600, 60),
        ("READINESS_ALERT_COOLDOWN_S", 30, 86_400, 900),
        ("RUNTIME_CONFIG_POLL_SECONDS", 1, 3_600, 10),
        ("VAULT_KV_VERSION", 1, 2, 2),
        ("MACRO_REGIME_REFRESH_SECONDS", 60, 86_400, 900),
    ]
    for name, min_value, max_value, fallback in rules:
        _validate_numeric_bounds(name, min_value, max_value, fallback)

    if ALPHA_TIMEFRAME not in {"1h"}:
        _warn_config(f"Invalid ALPHA_TIMEFRAME={ALPHA_TIMEFRAME!r}; using '1h'.")
        globals()["ALPHA_TIMEFRAME"] = "1h"

    if LIVE_MAX_ORDER_USD < LIVE_MIN_ORDER_USD:
        _warn_config(
            f"LIVE_MAX_ORDER_USD ({LIVE_MAX_ORDER_USD}) is below LIVE_MIN_ORDER_USD "
            f"({LIVE_MIN_ORDER_USD}); raising max to min."
        )
        globals()["LIVE_MAX_ORDER_USD"] = float(LIVE_MIN_ORDER_USD)

    if LIVE_MAX_POSITION_SIZE_USD < LIVE_MAX_ORDER_USD:
        _warn_config(
            "LIVE_MAX_POSITION_SIZE_USD is below LIVE_MAX_ORDER_USD; "
            "raising position cap to order cap."
        )
        globals()["LIVE_MAX_POSITION_SIZE_USD"] = float(LIVE_MAX_ORDER_USD)

    if PORTFOLIO_HARD_MAX_POSITIONS < PORTFOLIO_TARGET_POSITIONS:
        _warn_config(
            "PORTFOLIO_HARD_MAX_POSITIONS is below PORTFOLIO_TARGET_POSITIONS; "
            "raising hard max to target."
        )
        globals()["PORTFOLIO_HARD_MAX_POSITIONS"] = int(PORTFOLIO_TARGET_POSITIONS)

    if SOURCE_POLICY_PAUSE_WEIGHT > SOURCE_POLICY_DEGRADE_WEIGHT:
        _warn_config(
            "SOURCE_POLICY_PAUSE_WEIGHT is above SOURCE_POLICY_DEGRADE_WEIGHT; "
            "clamping pause threshold down to the degrade threshold."
        )
        globals()["SOURCE_POLICY_PAUSE_WEIGHT"] = float(SOURCE_POLICY_DEGRADE_WEIGHT)

    if COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE > COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE:
        _warn_config(
            "COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE is above "
            "COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE; clamping block threshold "
            "down to the degrade threshold."
        )
        globals()["COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE"] = float(
            COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE
        )


_validate_config_bounds()
