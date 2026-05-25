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
_POSTGRES_DSN_CANDIDATES = (
    ("POSTGRES_DSN", os.environ.get("POSTGRES_DSN", "")),
    # Railway exposes Postgres as DATABASE_URL by default.  Accept it so
    # operators do not accidentally run SQLite after selecting Postgres.
    ("DATABASE_URL", os.environ.get("DATABASE_URL", "")),
    ("DATABASE_PRIVATE_URL", os.environ.get("DATABASE_PRIVATE_URL", "")),
)
POSTGRES_DSN_SOURCE = ""
POSTGRES_DSN = ""
for _dsn_name, _dsn_value in _POSTGRES_DSN_CANDIDATES:
    _dsn_value = str(_dsn_value or "").strip()
    if _dsn_value:
        POSTGRES_DSN_SOURCE = _dsn_name
        POSTGRES_DSN = _dsn_value
        break
# Auto-downgrade to sqlite if Postgres backends are requested but no DSN is set.
# H3 (audit): we still downgrade so dev environments boot, but we record
# the downgrade and emit a visible warning.  When live trading is enabled,
# live_trader's init raises on this flag so the operator can't scale
# capital believing Postgres is the ledger while SQLite is actually active.
DB_BACKEND_DOWNGRADED = False
_DB_BACKEND_REQUESTED = _raw_db_backend
if _raw_db_backend in ("dualwrite", "postgres") and not POSTGRES_DSN:
    DB_BACKEND = "sqlite"
    DB_BACKEND_DOWNGRADED = True
    import sys as _sys
    print(
        f"[config] WARNING: DB_BACKEND={_raw_db_backend!r} requested but "
        f"POSTGRES_DSN is empty -- downgrading to sqlite.  Live trading "
        f"will REFUSE to start in this state (set POSTGRES_DSN, "
        f"DATABASE_URL, or DB_BACKEND=sqlite).",
        file=_sys.stderr,
    )
else:
    DB_BACKEND = _raw_db_backend
POSTGRES_POOL_MIN = int(os.environ.get("POSTGRES_POOL_MIN", 2))
POSTGRES_POOL_MAX = int(os.environ.get("POSTGRES_POOL_MAX", 10))
POSTGRES_POOL_TIMEOUT_SEC = float(os.environ.get("POSTGRES_POOL_TIMEOUT_SEC", 3.0))
POSTGRES_CONNECT_TIMEOUT_SEC = int(float(os.environ.get("POSTGRES_CONNECT_TIMEOUT_SEC", 3)))
POSTGRES_STATEMENT_TIMEOUT_MS = int(os.environ.get("POSTGRES_STATEMENT_TIMEOUT_MS", 5000))
POSTGRES_APP_NAME = os.environ.get("POSTGRES_APP_NAME", "hyperliquid-bot").strip()

# Runtime DB audit/readiness guardrails. The audit is read-only; readiness
# blocks when findings at or above READINESS_DB_AUDIT_BLOCK_SEVERITY are found.
READINESS_DB_AUDIT_ENABLED = os.environ.get(
    "READINESS_DB_AUDIT_ENABLED", "true"
).lower() in ("true", "1", "yes")
READINESS_DB_AUDIT_AUTO_REPAIR = os.environ.get(
    "READINESS_DB_AUDIT_AUTO_REPAIR", "true"
).lower() in ("true", "1", "yes")
READINESS_DB_AUDIT_TTL_S = int(os.environ.get("READINESS_DB_AUDIT_TTL_S", 300))
READINESS_DB_AUDIT_BLOCK_SEVERITY = os.environ.get(
    "READINESS_DB_AUDIT_BLOCK_SEVERITY", "high"
).strip().lower()
if READINESS_DB_AUDIT_BLOCK_SEVERITY not in {"low", "medium", "high", "critical"}:
    READINESS_DB_AUDIT_BLOCK_SEVERITY = "high"
DB_AUDIT_CANDLE_CACHE_MISSING_ACTIVE_SEVERITY = os.environ.get(
    "DB_AUDIT_CANDLE_CACHE_MISSING_ACTIVE_SEVERITY", "medium"
).strip().lower()
if DB_AUDIT_CANDLE_CACHE_MISSING_ACTIVE_SEVERITY not in {"low", "medium", "high", "critical"}:
    DB_AUDIT_CANDLE_CACHE_MISSING_ACTIVE_SEVERITY = "medium"
DB_AUDIT_PENDING_DECISION_MAX_AGE_MINUTES = float(
    os.environ.get("DB_AUDIT_PENDING_DECISION_MAX_AGE_MINUTES", 30.0)
)
DB_AUDIT_REGIME_MAX_AGE_HOURS = float(
    os.environ.get("DB_AUDIT_REGIME_MAX_AGE_HOURS", 24.0)
)
DB_AUDIT_NON_ACTIVE_REGIME_RETENTION_DAYS = float(
    os.environ.get("DB_AUDIT_NON_ACTIVE_REGIME_RETENTION_DAYS", 7.0)
)
DB_AUDIT_SOURCE_STALE_MULTIPLIER = float(
    os.environ.get("DB_AUDIT_SOURCE_STALE_MULTIPLIER", 2.0)
)
SOURCE_HEALTH_SNAPSHOT_INTERVAL_S = int(
    os.environ.get("SOURCE_HEALTH_SNAPSHOT_INTERVAL_S", 60)
)
DB_AUDIT_MIN_CANDLE_COINS = int(os.environ.get("DB_AUDIT_MIN_CANDLE_COINS", 2))
DB_AUDIT_DUALWRITE_HEALTH_WINDOW_S = float(
    os.environ.get("DB_AUDIT_DUALWRITE_HEALTH_WINDOW_S", 300.0)
)
DB_AUDIT_DUALWRITE_MAX_FAILURES = int(
    os.environ.get("DB_AUDIT_DUALWRITE_MAX_FAILURES", 5)
)
DB_SAFE_AUTO_REPAIR_ON_BOOT = os.environ.get(
    "DB_SAFE_AUTO_REPAIR_ON_BOOT", "true"
).lower() in ("true", "1", "yes")
DB_REPAIR_KEEP_MISSING_SOURCE_STRATEGIES = int(
    os.environ.get("DB_REPAIR_KEEP_MISSING_SOURCE_STRATEGIES", 500)
)
DB_REPAIR_STARTUP_STRATEGY_PRUNE_LIMIT = int(
    os.environ.get("DB_REPAIR_STARTUP_STRATEGY_PRUNE_LIMIT", 1000)
)
BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE = os.environ.get(
    "BOOT_DB_AUDIT_INCLUDE_CANDLE_CACHE", "false"
).lower() in ("true", "1", "yes")

# ─── Feature Store (Postgres-only, auto-enabled when POSTGRES_DSN set) ─
FEATURE_STORE_COINS = os.environ.get("FEATURE_STORE_COINS", "").strip()
FEATURE_STORE_MAX_COINS = int(os.environ.get("FEATURE_STORE_MAX_COINS", 30))
FEATURE_STORE_BOOTSTRAP_TOP_COINS = int(os.environ.get("FEATURE_STORE_BOOTSTRAP_TOP_COINS", 8))
FEATURE_STORE_BACKFILL_5M_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_5M_DAYS", 7))
FEATURE_STORE_BACKFILL_1H_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_1H_DAYS", 30))
FEATURE_STORE_BACKFILL_4H_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_4H_DAYS", 90))
FEATURE_STORE_BACKFILL_1D_DAYS = int(os.environ.get("FEATURE_STORE_BACKFILL_1D_DAYS", 365))

# Runtime backup size guard.  Wallet fills are the largest backup component;
# keeping the newest rows preserves redeploy continuity without writing a
# hundreds-of-MB JSON file on every reporting cycle.  Set 0 to disable capping.
HL_BOT_BACKUP_MAX_WALLET_FILLS = int(os.environ.get("HL_BOT_BACKUP_MAX_WALLET_FILLS", 5000))
HL_BOT_BACKUP_MAX_GOLDEN_WALLETS = int(os.environ.get("HL_BOT_BACKUP_MAX_GOLDEN_WALLETS", 200))
HL_BOT_BACKUP_INCLUDE_EQUITY_CURVES = os.environ.get(
    "HL_BOT_BACKUP_INCLUDE_EQUITY_CURVES", "false"
).lower() in ("true", "1", "yes")

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

# H11 (audit): explicit reward/risk mode selector.  Surfaces the policy
# to operators so it's configurable per tier instead of an implicit
# dynamic adjustment.  Valid values:
#   - fixed_5r        (TP target = fixed 5R of the stop, skip dynamic
#                      regime/confidence adjustments — predictable for
#                      canary/T0 tiers where we're still calibrating)
#   - dynamic_bounded (legacy behavior — adjust R based on regime,
#                      confidence, source quality, expected move; bounded
#                      by min_reward_multiple/max_reward_multiple)
#   - hybrid_min_5r   (run dynamic adjustments but floor the final R at
#                      hybrid_min_r_floor — best of both worlds for
#                      advanced tiers with stable edge)
RISK_POLICY_RR_MODE = os.environ.get("RISK_POLICY_RR_MODE", "dynamic_bounded").strip().lower()
if RISK_POLICY_RR_MODE not in {"fixed_5r", "dynamic_bounded", "hybrid_min_5r"}:
    # Fail loud at import time — don't silently run with a broken mode.
    raise ValueError(
        f"RISK_POLICY_RR_MODE={RISK_POLICY_RR_MODE!r} is not a valid mode. "
        f"Expected one of: fixed_5r, dynamic_bounded, hybrid_min_5r"
    )
RISK_POLICY_FIXED_R_TARGET = float(
    os.environ.get("RISK_POLICY_FIXED_R_TARGET", 5.0)
)
RISK_POLICY_HYBRID_MIN_R_FLOOR = float(
    os.environ.get("RISK_POLICY_HYBRID_MIN_R_FLOOR", 5.0)
)
RISK_POLICY_SHORT_CAUTION_ENABLED = os.environ.get(
    "RISK_POLICY_SHORT_CAUTION_ENABLED", "true"
).lower() in ("true", "1", "yes")
RISK_POLICY_SHORT_CAUTION_CONFIDENCE_THRESHOLD = float(
    os.environ.get("RISK_POLICY_SHORT_CAUTION_CONFIDENCE_THRESHOLD", 0.60)
)
RISK_POLICY_SHORT_CAUTION_MAX_REWARD_MULTIPLE = float(
    os.environ.get("RISK_POLICY_SHORT_CAUTION_MAX_REWARD_MULTIPLE", 3.0)
)
RISK_POLICY_SHORT_CAUTION_TIME_LIMIT_MULTIPLIER = float(
    os.environ.get("RISK_POLICY_SHORT_CAUTION_TIME_LIMIT_MULTIPLIER", 0.75)
)
RISK_POLICY_SHORT_CAUTION_BREAKEVEN_AT_R = float(
    os.environ.get("RISK_POLICY_SHORT_CAUTION_BREAKEVEN_AT_R", 0.65)
)

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
# ★ AUDIT FIX (May 2026): hard cutoff tightened from 100 -> 80
# trades/day.  Observed cases with 100-150 trades/day classified as
# "Uncertain" instead of bot; 80/day is still aggressive but well
# above human plausibility (a discretionary human placing manual
# orders 10 hours/day at peak attention does ~30-40 trades).  Elevated
# frequency signal threshold dropped 50 -> 30 to align.
BOT_HARD_CUTOFF_TRADES = int(os.environ.get("BOT_HARD_CUTOFF_TRADES", 80))    # >N trades/day = instant bot
BOT_THRESHOLD = int(os.environ.get("BOT_THRESHOLD", 3))                        # signal score >= N = bot
BOT_MM_PNL_THRESHOLD = float(os.environ.get("BOT_MM_PNL_THRESHOLD", 0.0))     # median PnL < N = spread/MM
BOT_ELEVATED_FREQ = int(os.environ.get("BOT_ELEVATED_FREQ", 30))              # trades/day for elevated freq signal
# Statistical-anomaly separation (catches accounts the frequency-based
# detectors miss): a ~100% win rate sustained over a meaningful closed
# sample is statistically implausible for a real directional trader -- it
# is wash trading, a vault/MM, or "selective close" (never realizes a
# loss). Such accounts (and zero-evidence junk wallets) are uncopyable and
# must be separated from humans rather than displayed at "100% winrate".
BOT_PERFECT_WINRATE = float(os.environ.get("BOT_PERFECT_WINRATE", 0.98))       # win rate >= N (with min trades) = anomaly
BOT_PERFECT_WINRATE_MIN_TRADES = int(
    os.environ.get("BOT_PERFECT_WINRATE_MIN_TRADES", 15)
)  # min closed trades before a perfect/near-perfect record counts as a bot signal
# Minimum statistical evidence for a trader to be "copyable" (shown on the
# dashboard, eligible as a copy source).  A trader below this bar is NOT a
# bot -- it just has too little realized history to act on, so it is hidden
# as insufficient-evidence (and stays re-discoverable: discovery re-evaluates
# it and it returns automatically once it has a real track record).  Bar:
# >= TRADER_MIN_CLOSED_TRADES realized closed trades AND non-zero realized
# PnL/ROI (a $0-pnl/0%-ROI row is degenerate junk, e.g. the trivial
# "100% winrate / 0% ROI" accounts).
TRADER_MIN_CLOSED_TRADES = int(os.environ.get("TRADER_MIN_CLOSED_TRADES", 10))

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
# Minimum composite score for a strategy to remain active. 0.05 was effectively
# a no-op (every strategy passed); 0.20 forces real selectivity while still
# leaving a comfortable margin above noise.
MIN_STRATEGY_SCORE = float(os.environ.get("MIN_STRATEGY_SCORE", 0.20))
# Keep at least top-N strategies active even when all scores are weak, to avoid
# complete strategy starvation during cold-start or rough regimes.
MIN_ACTIVE_STRATEGIES = int(os.environ.get("MIN_ACTIVE_STRATEGIES", 5))
# Hard cap on active strategies in DB — lowest-scoring are deactivated beyond
# this. Default 25 gives MAX_STRATEGIES_PER_CYCLE (15) headroom for rotation
# without an unbounded `strategies` table. Previous default of 200 happened to
# match the discovered-strategy population so the cap never fired.
MAX_ACTIVE_STRATEGIES = int(os.environ.get("MAX_ACTIVE_STRATEGIES", 25))
# Max strategies per trading cycle fed to decision engine
MAX_STRATEGIES_PER_CYCLE = int(os.environ.get("MAX_STRATEGIES_PER_CYCLE", 15))
# If the live active-valid strategy pool is smaller than the per-cycle feed,
# recover valid inactive rows before scoring. This prevents the bot from
# getting trapped with only a few momentum rows while valid range/copy-derived
# strategies sit inactive after a prior quarantine/scoring pass.
STRATEGY_RECOVERY_TARGET_ACTIVE_VALID = int(
    os.environ.get(
        "STRATEGY_RECOVERY_TARGET_ACTIVE_VALID",
        max(MIN_ACTIVE_STRATEGIES, MAX_STRATEGIES_PER_CYCLE),
    )
)
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
# Max number of coins to pre-compute candle features for per strategy
# cycle. Core BTC/ETH/SOL are always included; the rest come from the
# coins this cycle's strategies target. Cap protects the candle API
# from a large strategy fan-out.
PAPER_FEATURE_PRECOMPUTE_MAX_COINS = int(
    os.environ.get("PAPER_FEATURE_PRECOMPUTE_MAX_COINS", 12) or 12
)
# Paper-trading risk is defined in ROE space, then converted back into raw
# trigger prices by dividing by leverage.
#
# Take-profit was historically 5x the configured stop-loss (a 5:1 R:R shape).
# The 30d audit (May 2026) found TP fires only 4 times in 308 trades (1.3%)
# while time_limit fires 28 times (+$188 net) -- winners reach an R or two
# of profit but then drift back before reaching the 5R TP. Halved to 2.5x
# so the bot captures the move it actually gets, rather than holding for a
# target almost never hit.
#
# Override via env var if you want a different shape:
#   PAPER_TRADING_TAKE_PROFIT_MULTIPLE=3.0
PAPER_TRADING_STOP_LOSS_PCT = float(os.environ.get("PAPER_TRADING_STOP_LOSS_PCT", 0.15))
PAPER_TRADING_TAKE_PROFIT_MULTIPLE = float(
    os.environ.get("PAPER_TRADING_TAKE_PROFIT_MULTIPLE", 2.5)
)
PAPER_TRADING_TAKE_PROFIT_PCT = PAPER_TRADING_STOP_LOSS_PCT * PAPER_TRADING_TAKE_PROFIT_MULTIPLE
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
PAPER_EXECUTION_MAX_TRADES_PER_CYCLE = int(
    os.environ.get("PAPER_EXECUTION_MAX_TRADES_PER_CYCLE", 3)
)
TRADE_QUALITY_FEE_EV_GATE_ENABLED = os.environ.get(
    "TRADE_QUALITY_FEE_EV_GATE_ENABLED", "true"
).lower() in ("true", "1", "yes")
TRADE_QUALITY_MIN_EDGE_COST_MULTIPLE = float(
    os.environ.get("TRADE_QUALITY_MIN_EDGE_COST_MULTIPLE", 1.5)
)
TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS = float(
    os.environ.get("TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS", PAPER_TRADING_SLIPPAGE_MAX_BPS)
)
TRADE_QUALITY_SHORT_MIN_CONFIDENCE = float(
    os.environ.get("TRADE_QUALITY_SHORT_MIN_CONFIDENCE", 0.55)
)
TRADE_QUALITY_STRONG_SHORT_CONFIRMATION = os.environ.get(
    "TRADE_QUALITY_STRONG_SHORT_CONFIRMATION", "true"
).lower() in ("true", "1", "yes")
# When true, paper_trader's trade-quality gate sources its edge from
# the firewall EV gate's already-computed breakdown (signal context
# ev_breakdown) instead of the legacy confidence proxy, and treats a
# solidly-positive firewall EV as short-confirmation. Reconciles the
# two EV gates into one source of truth and breaks the cold-start
# deadlock where confidence pinned at 0.50 forced the proxy edge to
# ~0 and silently vetoed every signal the firewall EV gate accepted.
# NOTE: during calibration cold-start the firewall EV uses an assumed
# p_win=0.50 with the strategy's R-multiples, so this lets regime-
# aligned trades through on assumption-driven EV until per-bucket
# calibration matures -- a deliberate tradeoff to get outcome data
# flowing. Set false to revert to the confidence-proxy behaviour.
TRADE_QUALITY_USE_FIREWALL_EV = os.environ.get(
    "TRADE_QUALITY_USE_FIREWALL_EV", "true"
).lower() in ("true", "1", "yes")

# Live trading wallet / secret-management controls.
# Agent-wallet-only mode: signer key must be for a delegated agent wallet, and
# HL_PUBLIC_ADDRESS points to the trading account (master/vault) being managed.
LIVE_TRADING_ENABLED = os.environ.get(
    "LIVE_TRADING_ENABLED", "false"
).strip().lower() in ("true", "1", "yes")
LIVE_TRADING_DUAL_CONTROL_CONFIRM = os.environ.get(
    "LIVE_TRADING_DUAL_CONTROL_CONFIRM", "false"
).strip().lower() in ("true", "1", "yes")

# ─── Live Order Caps (cautious bootstrap) ─────────────────────
# Hyperliquid enforces a $10 minimum notional per order on both perps and
# spot.  Any order below this silently does not fill (the matching engine
# drops it and clearinghouseState never shows the position) — you only
# notice because fill-verification times out with "FILL NOT VERIFIED".
# This floor is PHYSICALLY enforced by the exchange and cannot be lowered
# by config.
# H8 (audit): route live-capital-critical env vars through safe_env_*
# so a typo at redeploy can't crash boot and leave positions unmanaged.
# The helper logs a warning and falls back to the module default rather
# than raising ValueError at import time.
from src.core.env_utils import (  # noqa: E402 -- must follow sys.path setup above
    safe_env_bool as _safe_env_bool,
    safe_env_float as _safe_env_float,
    safe_env_int as _safe_env_int,
)

LIVE_MIN_ORDER_USD = _safe_env_float("LIVE_MIN_ORDER_USD", 11.0, lo=1.0, hi=10_000.0)

# Hard ceiling on the notional ($ USDC) of any single live order.  This is a
# safety net while the bot is ramping on a small live balance — even if paper
# sizing, rescaling, or the firewall suggest a larger trade, nothing above
# LIVE_MAX_ORDER_USD is ever sent to the exchange.
# The default is set slightly above the exchange minimum so fresh bootstraps
# can actually execute; set a higher value via env var as confidence grows.
# NOTE: a value below LIVE_MIN_ORDER_USD is impossible to honor — the
# LiveTrader will raise it to LIVE_MIN_ORDER_USD at startup with a warning.
LIVE_MAX_ORDER_USD = _safe_env_float("LIVE_MAX_ORDER_USD", 150.0, lo=1.0, hi=1_000_000.0)
_live_max_position_default = os.environ.get(
    "HL_MAX_POSITION_SIZE", str(LIVE_MAX_ORDER_USD)
)
_raw_live_max_position = os.environ.get("LIVE_MAX_POSITION_SIZE_USD", _live_max_position_default)
try:
    LIVE_MAX_POSITION_SIZE_USD = float(_raw_live_max_position)
    if LIVE_MAX_POSITION_SIZE_USD < 1.0:
        raise ValueError("below floor")
    if LIVE_MAX_POSITION_SIZE_USD > 10_000_000.0:
        raise ValueError("above ceiling")
except (TypeError, ValueError):
    import sys as _sys
    print(
        f"[config] WARNING: LIVE_MAX_POSITION_SIZE_USD={_raw_live_max_position!r} "
        f"out of [1,1e7] range or non-numeric; falling back to "
        f"LIVE_MAX_ORDER_USD=${LIVE_MAX_ORDER_USD}.",
        file=_sys.stderr,
    )
    LIVE_MAX_POSITION_SIZE_USD = float(LIVE_MAX_ORDER_USD)
# Daily loss limit for the live account in USD (forwarded to LiveTrader).
LIVE_MAX_DAILY_LOSS_USD = _safe_env_float(
    "LIVE_MAX_DAILY_LOSS_USD", 100.0, lo=0.01, hi=10_000_000.0,
)
LIVE_CANARY_MODE = os.environ.get(
    "LIVE_CANARY_MODE", "false"
).lower() in ("true", "1", "yes")
LIVE_CANARY_MAX_ORDER_USD = _safe_env_float(
    "LIVE_CANARY_MAX_ORDER_USD", 25.0, lo=1.0, hi=1_000_000.0,
)
LIVE_CANARY_MAX_SIGNALS_PER_DAY = _safe_env_int(
    "LIVE_CANARY_MAX_SIGNALS_PER_DAY", 25, lo=0, hi=100_000,
)
LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY = int(
    os.environ.get("LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 0)
)
LIVE_RISK_SIZING_ENABLED = _safe_env_bool("LIVE_RISK_SIZING_ENABLED", True)
LIVE_RISK_PER_TRADE_PCT = _safe_env_float(
    "LIVE_RISK_PER_TRADE_PCT", 0.0075, lo=0.0, hi=0.25,
)
LIVE_MAX_MARGIN_PER_ORDER_PCT = _safe_env_float(
    "LIVE_MAX_MARGIN_PER_ORDER_PCT", 0.12, lo=0.0, hi=1.0,
)
LIVE_MIN_MARGIN_PER_ORDER_USD = _safe_env_float(
    "LIVE_MIN_MARGIN_PER_ORDER_USD", 0.0, lo=0.0, hi=1_000_000.0,
)
LIVE_DYNAMIC_SOURCE_CAPS_ALLOW_STATIC_EXPANSION = _safe_env_bool(
    "LIVE_DYNAMIC_SOURCE_CAPS_ALLOW_STATIC_EXPANSION", False,
)
LIVE_ORDER_HYGIENE_AUDIT_INTERVAL_CYCLES = _safe_env_int(
    "LIVE_ORDER_HYGIENE_AUDIT_INTERVAL_CYCLES", 5, lo=1, hi=100_000,
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
LIVE_MIN_ORDER_ALLOW_POLICY_ERROR_FLOORUP = os.environ.get(
    "LIVE_MIN_ORDER_ALLOW_POLICY_ERROR_FLOORUP", "false"
).lower() in ("true", "1", "yes")
LIVE_MIN_ORDER_SHORT_MIN_CONFIDENCE = float(
    os.environ.get("LIVE_MIN_ORDER_SHORT_MIN_CONFIDENCE", 0.75)
)
LIVE_MIN_ORDER_SAME_SIDE_MERGE_ENABLED = os.environ.get(
    "LIVE_MIN_ORDER_SAME_SIDE_MERGE_ENABLED", "true"
).lower() in ("true", "1", "yes")
LIVE_MIN_ORDER_SAME_SIDE_MAX_BUMP_MULTIPLIER = float(
    os.environ.get("LIVE_MIN_ORDER_SAME_SIDE_MAX_BUMP_MULTIPLIER", 2.5)
)
LIVE_ANALYTICS_LOOKBACK_TRADES = int(os.environ.get("LIVE_ANALYTICS_LOOKBACK_TRADES", 200))
LIVE_ENTRY_EXECUTION_MODE = os.environ.get(
    "LIVE_ENTRY_EXECUTION_MODE", "maker_then_market"
).strip().lower()
LIVE_MAKER_ENTRY_OFFSET_BPS = _safe_env_float(
    "LIVE_MAKER_ENTRY_OFFSET_BPS", 1.0, lo=0.0, hi=100.0,
)
LIVE_MAKER_ENTRY_TIMEOUT_S = _safe_env_float(
    "LIVE_MAKER_ENTRY_TIMEOUT_S", 2.5, lo=0.0, hi=30.0,
)
LIVE_MAKER_ENTRY_FALLBACK_TO_MARKET = _safe_env_bool(
    "LIVE_MAKER_ENTRY_FALLBACK_TO_MARKET", True,
)
LIVE_SCHEDULE_CANCEL_ENABLED = _safe_env_bool(
    "LIVE_SCHEDULE_CANCEL_ENABLED", False,
)
LIVE_SCHEDULE_CANCEL_ENTRY_TIMEOUT_S = _safe_env_float(
    "LIVE_SCHEDULE_CANCEL_ENTRY_TIMEOUT_S", 60.0, lo=5.0, hi=86_400.0,
)
LIVE_SCHEDULE_CANCEL_WORKING_TIMEOUT_S = _safe_env_float(
    "LIVE_SCHEDULE_CANCEL_WORKING_TIMEOUT_S", 300.0, lo=5.0, hi=86_400.0,
)

# ── A1: ATR-aware stop-loss floor ────────────────────────────────────────
# Why this exists: with the default 4% ROE stop and high leverage (e.g. 25x)
# the *price* stop becomes 4%/25 = 16 bps — well inside normal 5m noise on
# most coins. Recent audit found 5 of last week's 8 losses were stop-outs
# triggered by adverse moves as small as -0.03%, some on the same bar as the
# entry. This widens the stop (never tightens it) to at least
# max(ATR_STOP_ATR_MULTIPLIER * recent_ATR, ATR_STOP_NOISE_FLOOR_BPS).
# TP is widened proportionally so the reward:risk ratio is preserved.
#
# DEFAULT OFF: this changes real-money trigger prices, so it ships dark and
# must be backtest-validated on the 90d window before flipping default ON.
# Once ATR_STOP_FLOOR_ENABLED=true, no signal can ever be stopped by a move
# tighter than max(2.5 * ATR, 50 bps) — that's the noise band.
ATR_STOP_FLOOR_ENABLED = _safe_env_bool("ATR_STOP_FLOOR_ENABLED", False)
ATR_STOP_ATR_MULTIPLIER = _safe_env_float(
    "ATR_STOP_ATR_MULTIPLIER", 2.5, lo=0.5, hi=10.0,
)
ATR_STOP_NOISE_FLOOR_BPS = _safe_env_float(
    "ATR_STOP_NOISE_FLOOR_BPS", 50.0, lo=0.0, hi=1000.0,
)

# ── A4: HL ↔ CEX funding-carry shadow telemetry ─────────────────────────
# When enabled, every cross-venue confirmation call that sees both an HL
# and a CEX funding rate runs evaluate_carry() and logs the resulting
# CarryOpportunity (actionable / vetoed + edge). PURE TELEMETRY — never
# mutates the cross-venue signal or routes a real order. Designed for
# the 2-week shadow-mode validation before any execution wiring lands.
#
# DEFAULT OFF. Flip true to populate logs for the shadow evaluator.
FUNDING_CARRY_SHADOW_ENABLED = _safe_env_bool("FUNDING_CARRY_SHADOW_ENABLED", False)
FUNDING_CARRY_SHADOW_MIN_EDGE_BPS = _safe_env_float(
    "FUNDING_CARRY_SHADOW_MIN_EDGE_BPS", 8.0, lo=0.0, hi=500.0,
)
FUNDING_CARRY_SHADOW_HOLD_HOURS = _safe_env_float(
    "FUNDING_CARRY_SHADOW_HOLD_HOURS", 4.0, lo=0.25, hi=24.0,
)

# ── A6: maker-first execution policy SHADOW telemetry ───────────────────
# When enabled, every live entry order placement also runs the
# MakerExecutionPolicy.decide() function against a synthesized BBO
# (mid ± MAKER_FIRST_SHADOW_SPREAD_BPS / 2) and the source-class
# default policy, logging the recommended action: POST_ALO / HOLD /
# REPOST_AT_BBO / TAKER_FALLBACK / ABANDON / FILLED.
#
# PURE TELEMETRY -- the actual order placement path is unchanged.
# Designed to populate a few thousand decisions before any live
# wiring lands, so we can validate:
#   1. Are signals reaching the entry order WITHIN max_signal_age_s
#      under the per-source default policies? (If not, raise the
#      timeout or admit the lane is too slow for taker fallback.)
#   2. What's the action histogram per source class? (If it's all
#      ABANDON-stale, copy_trade is too slow for maker-only.)
#
# DEFAULT OFF. Flip true to start populating MAKER_SHADOW logs.
MAKER_FIRST_SHADOW_ENABLED = _safe_env_bool("MAKER_FIRST_SHADOW_ENABLED", False)
MAKER_FIRST_SHADOW_SPREAD_BPS = _safe_env_float(
    "MAKER_FIRST_SHADOW_SPREAD_BPS", 1.0, lo=0.1, hi=100.0,
)

# Regime reversal supervision for open LIVE positions.
# Default mode is intentionally staged: detect confirmed opposite regimes and
# tighten protection, but do not flatten/reverse real capital unless the
# operator explicitly enables those higher-impact gates.
REGIME_REVERSAL_ENABLED = _safe_env_bool("REGIME_REVERSAL_ENABLED", True)
REGIME_REVERSAL_TIGHTEN_ENABLED = _safe_env_bool("REGIME_REVERSAL_TIGHTEN_ENABLED", True)
REGIME_REVERSAL_CLOSE_ENABLED = _safe_env_bool("REGIME_REVERSAL_CLOSE_ENABLED", False)
REGIME_REVERSAL_REVERSE_ENABLED = _safe_env_bool("REGIME_REVERSAL_REVERSE_ENABLED", False)
REGIME_REVERSAL_REVERSE_ON_CRASH = _safe_env_bool("REGIME_REVERSAL_REVERSE_ON_CRASH", False)
# Regime hysteresis (#7): the overall regime label is consumed as hard
# truth by ~12 gates; in the logs it flipped bullish/crash/neutral cycle
# to cycle and poisoned the market-side guard. When enabled, a *changed*
# label must persist REGIME_HYSTERESIS_MIN_STREAK consecutive cycles (or
# arrive with >= REGIME_HYSTERESIS_OVERRIDE_CONF confidence, so a genuine
# crash still flips instantly) before the gates see the new label.
# DEFAULT OFF -> behavior is byte-identical until an operator opts in.
REGIME_HYSTERESIS_ENABLED = _safe_env_bool("REGIME_HYSTERESIS_ENABLED", False)
REGIME_HYSTERESIS_MIN_STREAK = int(
    os.environ.get("REGIME_HYSTERESIS_MIN_STREAK", 2)
)
REGIME_HYSTERESIS_OVERRIDE_CONF = _safe_env_float(
    "REGIME_HYSTERESIS_OVERRIDE_CONF", 0.85, lo=0.0, hi=1.0
)
# Recent-side block escape (#1): long/short hardening blocks a whole side
# from a count-based lookback of the last N closed trades. When a side is
# blocked it stops trading -> no new closes -> the count-based window
# never refreshes -> the block is PERMANENT (the live deadlock: 0 trades
# in 6h, "Recent longs are underperforming x35"). After a side has been
# continuously blocked this many hours, downgrade the hard block to
# "degraded" (reduced-size probe) so the sample can refresh and the gate
# re-evaluates on fresh data. ~one reduced probe per cooldown. 0 disables
# (legacy permanent block).
FIREWALL_RECENT_SIDE_BLOCK_MAX_HOURS = _safe_env_float(
    "FIREWALL_RECENT_SIDE_BLOCK_MAX_HOURS", 24.0, lo=0.0, hi=720.0
)
# Feature precompute coverage (#3): cap how many recent copy-trade
# candidate coins to fold into the watched/feature-precompute universe so
# copy signals on the broad tracked-trader coin set stop being dropped
# with data_readiness_missing:candles,feature_vector.
FEATURE_COPY_CANDIDATE_COINS_MAX = int(
    os.environ.get("FEATURE_COPY_CANDIDATE_COINS_MAX", 25)
)
# Copy-source-floor regime relaxation (#2): DEFAULT ON.
#
# History: this lever shipped default-OFF on the assumption that the
# 45%-floor rejections were AgentScorer correctly down-weighting
# unproven copy sources. 30 days of audit_trail forensics disproved
# that. Across 2 different traders on 8 different coins, the rejected
# confidence value was EXACTLY 43% every time -- a constant. Tracing
# the cascade against the log:
#
#   raw copy confidence (per trader / coin): 0.64, 0.85, 0.89, ..., 0.95
#   -> all capped to 0.50 by "synthetic regime non-authoritative"
#   -> source-side guard x 0.75 = 0.375
#   -> agent-scorer weight-blend with default 0.5 (cold-start constant)
#     0.375 * 0.6 + 0.5 * 0.4 = 0.425 ~= 0.43
#   -> below the 0.45 source-allocator warmup floor -> REJECTED
#   -> source never accrues closed trades -> stays "unproven" forever
#
# The merit signal is erased upstream at the synthetic cap, so the 0.45
# floor isn't filtering bad sources -- it's rejecting a constant. This
# is a STRUCTURAL deadlock of the same shape as #1, not a tuning
# preference. Default-OFF was wrong; flipping to default-ON.
#
# The relax only fires when (a) source_key starts with "copy_trade" AND
# (b) regime_data.forecaster_synthetic_warm_start is True -- i.e. only
# when the cascade can structurally crush a high-confidence trader
# signal. Non-synthetic-regime merit gating is unchanged.
#
# Operator escape hatch: set
# COPY_SOURCE_FLOOR_SYNTHETIC_RELAX_ENABLED=false in Railway env vars
# to revert to the prior conservative behaviour without redeploying.
COPY_SOURCE_FLOOR_SYNTHETIC_RELAX_ENABLED = _safe_env_bool(
    "COPY_SOURCE_FLOOR_SYNTHETIC_RELAX_ENABLED", True
)
COPY_SOURCE_FLOOR_SYNTHETIC_RELAX = _safe_env_float(
    "COPY_SOURCE_FLOOR_SYNTHETIC_RELAX", 0.07, lo=0.0, hi=0.30
)
# Copy-source-floor synthetic EXEMPTION (#1, the real fix). Evidence
# (logs.1779171559187): every copy signal -- raw 0.64..0.95, all coins,
# all source traders -- is flattened to 0.50 by the synthetic-regime cap,
# then blended to a CONSTANT 0.43 that is deterministically 2pts under
# the 0.45 source floor -> 100% of copy signals rejected forever while
# the forecaster stays synthetic. That is NOT the AgentScorer grading
# source merit (the merit signal was erased upstream by the cap); it is a
# structural dead-zone. So when a copy signal's confidence was capped by
# a synthetic / non-authoritative regime read, SKIP the source-confidence
# floor entirely (other source-policy checks -- paused/blocked/
# quarantine/day-cap -- still apply). DEFAULT ON to break the deadlock;
# flag-gated so an operator can revert to the legacy floor if desired.
COPY_SOURCE_FLOOR_SYNTHETIC_EXEMPT_ENABLED = _safe_env_bool(
    "COPY_SOURCE_FLOOR_SYNTHETIC_EXEMPT_ENABLED", True
)
REGIME_REVERSAL_MIN_CONFIDENCE = _safe_env_float(
    "REGIME_REVERSAL_MIN_CONFIDENCE", 0.70, lo=0.0, hi=1.0,
)
REGIME_REVERSAL_REVERSE_CONFIDENCE = _safe_env_float(
    "REGIME_REVERSAL_REVERSE_CONFIDENCE", 0.82, lo=0.0, hi=1.0,
)
REGIME_REVERSAL_CONFIRM_CYCLES = _safe_env_int(
    "REGIME_REVERSAL_CONFIRM_CYCLES", 3, lo=1, hi=100,
)
REGIME_REVERSAL_MIN_POSITION_AGE_SECONDS = _safe_env_int(
    "REGIME_REVERSAL_MIN_POSITION_AGE_SECONDS", 180, lo=0, hi=86_400,
)
REGIME_REVERSAL_COOLDOWN_SECONDS = _safe_env_int(
    "REGIME_REVERSAL_COOLDOWN_SECONDS", 900, lo=0, hi=86_400,
)
REGIME_REVERSAL_MAX_ACTIONS_PER_COIN_PER_DAY = _safe_env_int(
    "REGIME_REVERSAL_MAX_ACTIONS_PER_COIN_PER_DAY", 2, lo=0, hi=100,
)
REGIME_REVERSAL_TIGHTEN_STOP_R_MULTIPLE = _safe_env_float(
    "REGIME_REVERSAL_TIGHTEN_STOP_R_MULTIPLE", 0.35, lo=0.01, hi=2.0,
)
REGIME_REVERSAL_REVERSE_POSITION_PCT = _safe_env_float(
    "REGIME_REVERSAL_REVERSE_POSITION_PCT", 0.03, lo=0.001, hi=0.50,
)
COPY_TRADER_ENABLED = os.environ.get(
    "COPY_TRADER_ENABLED", "true"
).lower() in ("true", "1", "yes")
COPY_TRADER_MAX_CONCURRENT_TRADES = int(
    os.environ.get("COPY_TRADER_MAX_CONCURRENT_TRADES", 5)
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
COPY_TRADER_SOURCE_SIDE_GUARD_ENABLED = os.environ.get(
    "COPY_TRADER_SOURCE_SIDE_GUARD_ENABLED", "true"
).lower() in ("true", "1", "yes")
COPY_TRADER_SOURCE_SIDE_MIN_CLOSED_TRADES = int(
    os.environ.get("COPY_TRADER_SOURCE_SIDE_MIN_CLOSED_TRADES", 3)
)
COPY_TRADER_SOURCE_SIDE_DEGRADE_WIN_RATE = float(
    os.environ.get("COPY_TRADER_SOURCE_SIDE_DEGRADE_WIN_RATE", 0.45)
)
COPY_TRADER_SOURCE_SIDE_BLOCK_WIN_RATE = float(
    os.environ.get("COPY_TRADER_SOURCE_SIDE_BLOCK_WIN_RATE", 0.35)
)
COPY_TRADER_SOURCE_SIDE_BLOCK_NET_PNL = float(
    os.environ.get("COPY_TRADER_SOURCE_SIDE_BLOCK_NET_PNL", -0.25)
)
COPY_TRADER_SOURCE_SIDE_CONFIDENCE_MULTIPLIER = float(
    os.environ.get("COPY_TRADER_SOURCE_SIDE_CONFIDENCE_MULTIPLIER", 0.75)
)
COPY_TRADER_SOURCE_SIDE_SIZE_MULTIPLIER = float(
    os.environ.get("COPY_TRADER_SOURCE_SIDE_SIZE_MULTIPLIER", 0.50)
)
# Hard blocklist for copy-trade sources. Comma-separated 0x-addresses (case-
# insensitive). Any signal whose source_trader matches is dropped immediately,
# bypassing all guards. Use for confirmed bad actors or sources you want
# permanently disabled.
COPY_TRADER_BLOCKED_SOURCES = tuple(
    addr.strip().lower()
    for addr in os.environ.get(
        "COPY_TRADER_BLOCKED_SOURCES",
        "0x350e33a7",
    ).split(",")
    if addr.strip()
)
# Aggregate per-source hard cutoff. After at least
# COPY_TRADER_HARD_CUTOFF_MIN_TRADES closed trades, if a source's overall
# win rate falls below COPY_TRADER_HARD_CUTOFF_WIN_RATE the source is auto-
# disabled (different from auto_pause: this is per-source, not per-side, and
# evaluates regardless of net PnL).
COPY_TRADER_HARD_CUTOFF_ENABLED = os.environ.get(
    "COPY_TRADER_HARD_CUTOFF_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
COPY_TRADER_HARD_CUTOFF_MIN_TRADES = int(
    os.environ.get("COPY_TRADER_HARD_CUTOFF_MIN_TRADES", 30)
)
COPY_TRADER_HARD_CUTOFF_WIN_RATE = float(
    os.environ.get("COPY_TRADER_HARD_CUTOFF_WIN_RATE", 0.50)
)
LIVE_EXTERNAL_KILL_SWITCH_FILE = os.environ.get("LIVE_EXTERNAL_KILL_SWITCH_FILE", "").strip()
LIVE_KILL_SWITCH_STATE_FILE = os.environ.get("LIVE_KILL_SWITCH_STATE_FILE", "/data/live_kill_switch_state.json").strip()
# Master kill-switch override. When true, every kill-switch trip is
# logged but ignored: live entries continue to flow regardless of
# daily-loss limit, dualwrite health, daily-PnL refresh failures,
# external file flag, drawdown limit, or any other safety chain that
# would normally stop trading. Use only when you are absolutely sure
# (operator maintenance, infra outage you've already mitigated, etc.)
# -- per-source firewall checks, position caps, and order-size limits
# still apply, but the *sticky* gate is bypassed entirely.
LIVE_KILL_SWITCH_DISABLED = os.environ.get(
    "LIVE_KILL_SWITCH_DISABLED", "false"
).strip().lower() in ("1", "true", "yes", "on")
# Number of consecutive userFills failures tolerated before the daily-PnL
# refresh trips the sticky kill switch. A single transient API blip should
# not permanently disable live trading; sustained outages still fail closed.
LIVE_DAILY_PNL_REFRESH_FAILURE_THRESHOLD = max(
    1, int(os.environ.get("LIVE_DAILY_PNL_REFRESH_FAILURE_THRESHOLD", "3"))
)

# Confidence calibration controls.
# Older outcomes get exponentially down-weighted with this half-life so
# the calibrator tracks the current strategy stack rather than the
# all-time history. Set to 0 to disable decay entirely.
CALIBRATION_HALF_LIFE_DAYS = float(
    os.environ.get("CALIBRATION_HALF_LIFE_DAYS", "30")
)
# Per-source minimum outcomes before we trust calibrated confidence at
# all. Below this threshold, predictions are capped to the cold-start
# prior (CALIBRATION_COLDSTART_PRIOR) so an uncalibrated source cannot
# emit aggressive confidences.
CALIBRATION_MIN_OUTCOMES = int(
    os.environ.get("CALIBRATION_MIN_OUTCOMES", "30")
)
CALIBRATION_COLDSTART_PRIOR = float(
    os.environ.get("CALIBRATION_COLDSTART_PRIOR", "0.50")
)
# When true (default), the bucketed-threshold firewall floor does NOT
# raise above the operator's FIREWALL_MIN_CONFIDENCE just because a
# (source|side|regime) bucket lacks evidence (no-data / thin /
# global-fallback). Absence of evidence is not evidence of badness;
# inventing a cold-start confidence tax there deadlocked bootstrap
# (regime-aligned, strongly +EV signals at ~0.4x cold-start
# confidence were all rejected as "below bucket floor 50%"). A
# *measured* bad-ECE bucket still hard-quarantines. Cold-start risk is
# controlled by the leverage clamp + reduced size + positive-EV
# requirement, not a redundant confidence tax. Set false to restore
# the old max(min_confidence, coldstart_prior) cold-start floor.
CALIBRATION_COLDSTART_USES_GLOBAL_MIN = os.environ.get(
    "CALIBRATION_COLDSTART_USES_GLOBAL_MIN", "true"
).lower() in ("true", "1", "yes")
# Above this minimum we still apply Bayesian shrinkage; we only trust
# the empirical isotonic fit once a source crosses this many outcomes.
CALIBRATION_ISOTONIC_MIN_OUTCOMES = int(
    os.environ.get("CALIBRATION_ISOTONIC_MIN_OUTCOMES", "100")
)
# Auto-quarantine sources whose ECE crosses this threshold once they
# have at least CALIBRATION_QUARANTINE_MIN_SAMPLES outcomes. Quarantined
# sources are routed to shadow only until ECE recovers.
CALIBRATION_QUARANTINE_ECE = float(
    os.environ.get("CALIBRATION_QUARANTINE_ECE", "0.25")
)
CALIBRATION_QUARANTINE_MIN_SAMPLES = int(
    os.environ.get("CALIBRATION_QUARANTINE_MIN_SAMPLES", "50")
)
# When the global calibrator goes off the rails (ECE >= this), pause
# live entries entirely. Paper trades continue to feed the calibrator.
CALIBRATION_LIVE_PAUSE_ECE = float(
    os.environ.get("CALIBRATION_LIVE_PAUSE_ECE", "0.50")
)
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
# 0.45 turned out to be TOO strict in canary: observed log shows
# copy/options signals landing at 41–43% conviction, consistently
# just below the gate, so nothing ever reaches live.  0.40 still
# rejects low-quality signals while letting well-formed, macro-drag-
# adjusted signals through.  Raise back to 0.45+ once we have
# meaningful live-trade history to score sources against.
FIREWALL_MIN_CONFIDENCE = float(os.environ.get("FIREWALL_MIN_CONFIDENCE", 0.40))
# Block signals whose source resolves to ``unknown`` / ``strategy:unknown`` /
# ``strategy:untagged`` at the firewall layer. Default-on because those
# buckets dominate the calibration table when upstream signal generators
# fail to tag strategy_type/source -- if calibration data is mostly one
# fat untagged bucket, per-source thresholds have nothing to gate on.
# Disable via ``FIREWALL_BLOCK_UNKNOWN_SOURCES=false`` to allow them
# through (e.g. during a deliberate paper-collection backfill).
FIREWALL_BLOCK_UNKNOWN_SOURCES = os.environ.get(
    "FIREWALL_BLOCK_UNKNOWN_SOURCES", "true"
).lower() in ("true", "1", "yes")

# ─── Cold-start leverage clamp ────────────────────────────────
# While a (source|side|regime) calibration bucket has fewer than
# COLDSTART_CALIBRATION_MIN_SAMPLES real outcomes, the EV gate runs on
# the assumed p_win=0.50 prior -- the trade is unproven. Clamp leverage
# to COLDSTART_MAX_LEVERAGE so an unproven bucket can't (a) over-
# concentrate capital-at-risk, or (b) saturate the leveraged-notional
# aggregate-exposure cap and lock out diversified signals (observed in
# prod: one 8x cold-start short → 23 exposure rejections / 6h).
COLDSTART_LEVERAGE_CLAMP_ENABLED = os.environ.get(
    "COLDSTART_LEVERAGE_CLAMP_ENABLED", "true"
).lower() in ("true", "1", "yes")
COLDSTART_MAX_LEVERAGE = _safe_env_float(
    "COLDSTART_MAX_LEVERAGE", 3.0, lo=1.0, hi=25.0
)
COLDSTART_CALIBRATION_MIN_SAMPLES = int(
    os.environ.get("COLDSTART_CALIBRATION_MIN_SAMPLES", 30) or 30
)

# ─── Live promotion gate ──────────────────────────────────────
# Walk-forward gate that blocks paper -> live mirror until a strategy /
# source has accumulated enough out-of-sample evidence. Layered on top
# of synthetic-strategy quarantine -- quarantine catches broken rows,
# this catches well-formed but unproven sources.
LIVE_PROMOTION_GATE_ENABLED = os.environ.get(
    "LIVE_PROMOTION_GATE_ENABLED", "true"
).lower() in ("true", "1", "yes")
LIVE_PROMOTION_MIN_TRADES = int(
    os.environ.get("LIVE_PROMOTION_MIN_TRADES", 30)
)
LIVE_PROMOTION_MIN_WIN_RATE = _safe_env_float(
    "LIVE_PROMOTION_MIN_WIN_RATE", 0.45, lo=0.0, hi=1.0
)
LIVE_PROMOTION_MIN_SCORE = _safe_env_float(
    "LIVE_PROMOTION_MIN_SCORE", 0.20, lo=0.0, hi=1.0
)

# Promotion bootstrap tier (DEFAULT OFF).
# ────────────────────────────────────────────────────────────────
# The standard promotion gate requires 30 paper trades + 45% win rate to
# qualify a source for live mirror. In a defensive posture where the
# firewall blocks ~95% of candidates, sources accumulate paper trades
# slowly enough that nothing ever reaches live (~weeks per source).
# The bootstrap tier provides an alternative path: a smaller sample
# (default 5 trades) at a HIGHER accuracy bar (default 60%) earns a
# fractional-size live mirror (default 0.25× standard size). The source
# can graduate to the full tier later by accumulating 30 trades.
#
# Off by default because mirroring live with less evidence is a real
# risk trade-off — operator must opt in by setting
# PROMOTION_BOOTSTRAP_TIER_ENABLED=true.
PROMOTION_BOOTSTRAP_TIER_ENABLED = _safe_env_bool(
    "PROMOTION_BOOTSTRAP_TIER_ENABLED", False,
)
PROMOTION_BOOTSTRAP_MIN_TRADES = int(
    os.environ.get("PROMOTION_BOOTSTRAP_MIN_TRADES", 5)
)
PROMOTION_BOOTSTRAP_MIN_WIN_RATE = _safe_env_float(
    "PROMOTION_BOOTSTRAP_MIN_WIN_RATE", 0.60, lo=0.0, hi=1.0
)
PROMOTION_BOOTSTRAP_SIZE_SCALE = _safe_env_float(
    "PROMOTION_BOOTSTRAP_SIZE_SCALE", 0.25, lo=0.01, hi=1.0
)

# A5: optional Deflated-Sharpe gate on strategy promotion. DEFAULT OFF.
# Only ever BLOCKS a promotion the base gate already approved -- it never
# unblocks. When on, the strategy's recent paper-trade P&L Sharpe must be
# statistically significant after deflating for selection bias
# (num_trials). Missing/insufficient history or any compute error fails
# OPEN (defer to the base gate) so this can only tighten, never break
# promotion. P&L is a valid input -- Sharpe is scale-invariant.
PROMOTION_REQUIRE_DSR = _safe_env_bool("PROMOTION_REQUIRE_DSR", False)
PROMOTION_DSR_NUM_TRIALS = int(
    os.environ.get("PROMOTION_DSR_NUM_TRIALS", 50)
)
PROMOTION_DSR_MIN_OBS = int(
    os.environ.get("PROMOTION_DSR_MIN_OBS", 20)
)
# Drift-aware promotion gate: consult learning_drift_reports before
# approving a paper-to-live promotion. When enabled, any recent
# DriftReport with blocks_promotion=TRUE (created within the last
# PROMOTION_DRIFT_MAX_AGE_HOURS) downgrades the promotion. This wires
# the FeatureDriftMonitor's blocks_promotion flag -- previously the
# monitor computed and persisted the flag but NO code consumed it, so
# the protection was dead.
#
# Strictly downgrade-only: never approves a promotion the base gate
# rejected. Any error (DB, parse, schema) fails OPEN. Default OFF so
# behavior is byte-identical until an operator opts in.
PROMOTION_REQUIRE_DRIFT_OK = _safe_env_bool("PROMOTION_REQUIRE_DRIFT_OK", False)
PROMOTION_DRIFT_MAX_AGE_HOURS = _safe_env_float(
    "PROMOTION_DRIFT_MAX_AGE_HOURS", 24.0, lo=0.1, hi=720.0,
)
# A2: optional Thompson-sampling source allocator blended into
# AgentScorer.get_weight(). DEFAULT OFF -> get_weight is byte-identical
# to the legacy dynamic-weight path. When on, the allocator is fed the
# same per-source win/loss outcomes and its posterior sample is blended
# with the legacy weight by AGENT_BANDIT_BLEND (1.0 = pure Thompson,
# 0.0 = legacy). Lazy: no allocator object/state exists unless enabled.
AGENT_BANDIT_ALLOCATOR_ENABLED = _safe_env_bool(
    "AGENT_BANDIT_ALLOCATOR_ENABLED", False
)
AGENT_BANDIT_BLEND = _safe_env_float(
    "AGENT_BANDIT_BLEND", 1.0, lo=0.0, hi=1.0
)

# ── Loss attribution: don't penalise sources for the bot's tight stops ──
# When the bot's stop-loss triggers on a sub-noise move (the historical
# 5-of-8 noise stop-outs from last week, some on -0.03% moves), it isn't
# the source trader's fault -- it's our own too-tight SL. Without this
# guard, the bandit (A2) records a loss against the source, lowering
# its future allocation. This is the structural risk we called out
# when designing A1: A1 fixes the cause, A2 needs to be aware so it
# doesn't poison its posterior with our own noise stops while A1 rolls
# out (or for any signal where ATR data is unavailable / A1 is off).
#
# When enabled: a close classified as NOISE_STOP (or RECONCILED) by
# src.signals.loss_attribution.classify_close() is *skipped* by the
# bandit feed -- the source's posterior is untouched. SIGNAL_LOSS
# (a real adverse move beyond the noise band) still feeds the bandit
# as a loss. TAKE_PROFIT still feeds as a win.
#
# DEFAULT OFF: bandit behavior is byte-identical until an operator
# opts in. Pair this with A1 (ATR_STOP_FLOOR_ENABLED) once both have
# soaked in shadow.
BANDIT_SKIP_NOISE_STOPS_ENABLED = _safe_env_bool(
    "BANDIT_SKIP_NOISE_STOPS_ENABLED", False
)

# ─── Funding-rate divergence brake ─────────────────────────────
# Cross-market safety brake: when BTC/ETH funding is meaningfully
# positive AND price is below the 4h moving average (crowded longs
# paying premium into a selloff), block new longs. Symmetric for
# crowded shorts into rallies. Asymmetric -- never confirms or
# boosts a trade, only blocks.
FUNDING_DIVERGENCE_ENABLED = os.environ.get(
    "FUNDING_DIVERGENCE_ENABLED", "true"
).lower() in ("true", "1", "yes")
FUNDING_DIVERGENCE_FUNDING_THRESHOLD = _safe_env_float(
    "FUNDING_DIVERGENCE_FUNDING_THRESHOLD", 0.00015, lo=0.0, hi=0.01
)
FUNDING_DIVERGENCE_PRICE_DEV_THRESHOLD = _safe_env_float(
    "FUNDING_DIVERGENCE_PRICE_DEV_THRESHOLD", 0.005, lo=0.0, hi=0.5
)
FUNDING_DIVERGENCE_CACHE_TTL_S = _safe_env_float(
    "FUNDING_DIVERGENCE_CACHE_TTL_S", 300.0, lo=10.0, hi=3600.0
)

# ─── Per-bucket firewall confidence thresholds ─────────────────
# When enabled, the firewall consults the calibration tracker to
# derive a per-(source, side, regime) min-confidence floor instead of
# using a single global value. Asymmetric: never lowers below
# FIREWALL_MIN_CONFIDENCE, only raises for thin/miscalibrated buckets.
# Disable while the calibration table is being repopulated.
FIREWALL_USE_BUCKETED_THRESHOLDS = os.environ.get(
    "FIREWALL_USE_BUCKETED_THRESHOLDS", "true"
).lower() in ("true", "1", "yes")

# ─── Orphan position reaper ───────────────────────────────────
# An orphan is a live position the bot found on the exchange but
# didn't open itself. The reconciliation path creates a synthetic
# paper trade so PnL accounting stays consistent, but the bot has
# no thesis for these positions and never closes them. The reaper
# is opt-in: set ``ORPHAN_REAPER_ENABLED=true`` to have it close
# orphans past ``ORPHAN_REAPER_MAX_AGE_HOURS`` old. The break-even
# gate (default on) holds positions whose mid-price is worse than
# entry so the reaper doesn't realise losses on positions the
# operator might want to manage manually.
ORPHAN_REAPER_ENABLED = os.environ.get(
    "ORPHAN_REAPER_ENABLED", "false"
).lower() in ("true", "1", "yes")
ORPHAN_REAPER_MAX_AGE_HOURS = _safe_env_float(
    "ORPHAN_REAPER_MAX_AGE_HOURS", 24.0, lo=0.1, hi=720.0
)
ORPHAN_REAPER_REQUIRE_BREAKEVEN = os.environ.get(
    "ORPHAN_REAPER_REQUIRE_BREAKEVEN", "true"
).lower() in ("true", "1", "yes")

# ─── Expected-value gate ──────────────────────────────────────
# Replace confidence-only thresholds with a post-cost EV check. A signal
# at modest confidence with 3R/1R asymmetry can clear; a signal at high
# confidence with 1R/1R after fees+slippage+funding can be rejected.
# Live trades additionally need the lower-confidence-bound positive.
EV_GATE_ENABLED = os.environ.get("EV_GATE_ENABLED", "true").lower() in ("true", "1", "yes")
EV_GATE_MIN_BPS = _safe_env_float("EV_GATE_MIN_BPS", 10.0, lo=0.0, hi=10_000.0)
EV_GATE_MIN_COST_RATIO = _safe_env_float("EV_GATE_MIN_COST_RATIO", 1.5, lo=1.0, hi=10.0)
EV_GATE_LIVE_SIGMA_MULT = _safe_env_float("EV_GATE_LIVE_SIGMA_MULT", 2.0, lo=0.0, hi=10.0)

# ─── Trade-cost estimator ─────────────────────────────────────
TRADE_COSTS_DEFAULT_HOLDING_HOURS = _safe_env_float(
    "TRADE_COSTS_DEFAULT_HOLDING_HOURS", 24.0, lo=0.1, hi=720.0
)
TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS = _safe_env_float(
    "TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS", 5.0, lo=0.0, hi=500.0
)

# ─── Data-readiness gate ──────────────────────────────────────
# Reject signals whose data inputs are incomplete. Off-by-default
# components (oi, source_health) are logged but don't block; the
# required set blocks. Set DATA_READINESS_REQUIRED_COMPONENTS to a
# comma-separated list to customise.
DATA_READINESS_GATE_ENABLED = os.environ.get(
    "DATA_READINESS_GATE_ENABLED", "true"
).lower() in ("true", "1", "yes")
DATA_READINESS_REQUIRED_COMPONENTS = os.environ.get(
    "DATA_READINESS_REQUIRED_COMPONENTS",
    "candles,funding,spread,feature_vector",
)

# ─── Calibration trend monitor ────────────────────────────────
CALIBRATION_TREND_ENABLED = os.environ.get(
    "CALIBRATION_TREND_ENABLED", "true"
).lower() in ("true", "1", "yes")
CALIBRATION_TREND_WINDOW_DAYS = int(
    os.environ.get("CALIBRATION_TREND_WINDOW_DAYS", 3) or 3
)
CALIBRATION_TREND_DETERIORATION_BRIER_PER_DAY = _safe_env_float(
    "CALIBRATION_TREND_DETERIORATION_BRIER_PER_DAY", 0.02, lo=0.0, hi=1.0
)
CALIBRATION_TREND_MIN_SAMPLES_PER_DAY = int(
    os.environ.get("CALIBRATION_TREND_MIN_SAMPLES_PER_DAY", 5) or 5
)
CALIBRATION_TREND_DERISK_MULTIPLIER = _safe_env_float(
    "CALIBRATION_TREND_DERISK_MULTIPLIER", 0.75, lo=0.1, hi=1.0
)
FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY = int(
    os.environ.get("FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY", 0)
)
# Long-side hardening — symmetric mirror of SHORT_HARDENING_*.
# The firewall's _apply_long_hardening already supports per-knob tuning;
# this block wires them to env vars so operators can override the same
# way the short-side ones are overridable.  Defaults track the values
# baked into ``DecisionFirewall.__init__`` so existing behaviour is
# unchanged when nothing is set.
#
# Disable entirely with LONG_HARDENING_ENABLED=false when the recent
# long-trade history is contaminated (e.g. after a deploy thrash) and
# the operator wants to let new longs back through so the rolling
# 120-trade lookback can refresh.
LONG_HARDENING_ENABLED = os.environ.get("LONG_HARDENING_ENABLED", "true").lower() in ("true", "1", "yes")
LONG_HARDENING_LOOKBACK_TRADES = int(os.environ.get("LONG_HARDENING_LOOKBACK_TRADES", 120))
LONG_HARDENING_MIN_CLOSED_TRADES = int(os.environ.get("LONG_HARDENING_MIN_CLOSED_TRADES", 12))
LONG_HARDENING_DEGRADE_WIN_RATE = float(os.environ.get("LONG_HARDENING_DEGRADE_WIN_RATE", 0.48))
LONG_HARDENING_BLOCK_WIN_RATE = float(os.environ.get("LONG_HARDENING_BLOCK_WIN_RATE", 0.40))
LONG_HARDENING_BLOCK_NET_PNL = float(os.environ.get("LONG_HARDENING_BLOCK_NET_PNL", -0.5))
LONG_HARDENING_CONFIDENCE_MULTIPLIER = float(
    os.environ.get("LONG_HARDENING_CONFIDENCE_MULTIPLIER", 0.80)
)
LONG_HARDENING_SIZE_MULTIPLIER = float(os.environ.get("LONG_HARDENING_SIZE_MULTIPLIER", 0.50))

SHORT_HARDENING_ENABLED = os.environ.get("SHORT_HARDENING_ENABLED", "true").lower() in ("true", "1", "yes")
SHORT_HARDENING_LOOKBACK_TRADES = int(os.environ.get("SHORT_HARDENING_LOOKBACK_TRADES", 120))
SHORT_HARDENING_MIN_CLOSED_TRADES = int(os.environ.get("SHORT_HARDENING_MIN_CLOSED_TRADES", 12))
SHORT_HARDENING_DEGRADE_WIN_RATE = float(os.environ.get("SHORT_HARDENING_DEGRADE_WIN_RATE", 0.48))
SHORT_HARDENING_BLOCK_WIN_RATE = float(os.environ.get("SHORT_HARDENING_BLOCK_WIN_RATE", 0.30))
SHORT_HARDENING_BLOCK_NET_PNL = float(os.environ.get("SHORT_HARDENING_BLOCK_NET_PNL", -25.0))
SHORT_HARDENING_CONFIDENCE_MULTIPLIER = float(
    os.environ.get("SHORT_HARDENING_CONFIDENCE_MULTIPLIER", 0.80)
)
SHORT_HARDENING_SIZE_MULTIPLIER = float(os.environ.get("SHORT_HARDENING_SIZE_MULTIPLIER", 0.50))
SHORT_HARDENING_BLOCK_OVERRIDE_ENABLED = os.environ.get(
    "SHORT_HARDENING_BLOCK_OVERRIDE_ENABLED", "true"
).lower() in ("true", "1", "yes")
SHORT_HARDENING_BLOCK_OVERRIDE_MIN_CONFIDENCE = float(
    os.environ.get("SHORT_HARDENING_BLOCK_OVERRIDE_MIN_CONFIDENCE", 0.70)
)
SHORT_HARDENING_BLOCK_OVERRIDE_MIN_REGIME_CONFIDENCE = float(
    os.environ.get("SHORT_HARDENING_BLOCK_OVERRIDE_MIN_REGIME_CONFIDENCE", 0.60)
)
SHORT_HARDENING_BLOCK_OVERRIDE_SIZE_MULTIPLIER = float(
    os.environ.get("SHORT_HARDENING_BLOCK_OVERRIDE_SIZE_MULTIPLIER", 0.35)
)
SHORT_HARDENING_MARKET_ADAPTIVE_OVERRIDE_ENABLED = os.environ.get(
    "SHORT_HARDENING_MARKET_ADAPTIVE_OVERRIDE_ENABLED", "true"
).lower() in ("true", "1", "yes")
SHORT_HARDENING_MARKET_ADAPTIVE_MIN_MOMENTUM = float(
    os.environ.get("SHORT_HARDENING_MARKET_ADAPTIVE_MIN_MOMENTUM", 0.003)
)
SHORT_HARDENING_MARKET_ADAPTIVE_SCOPED_SIZE_MULTIPLIER = float(
    os.environ.get("SHORT_HARDENING_MARKET_ADAPTIVE_SCOPED_SIZE_MULTIPLIER", 0.25)
)
SHORT_HARDENING_SOURCE_GUARD_ENABLED = os.environ.get(
    "SHORT_HARDENING_SOURCE_GUARD_ENABLED", "true"
).lower() in ("true", "1", "yes")
SHORT_HARDENING_SOURCE_MIN_CLOSED_TRADES = int(
    os.environ.get("SHORT_HARDENING_SOURCE_MIN_CLOSED_TRADES", 3)
)
SHORT_HARDENING_SOURCE_BLOCK_NET_PNL = float(
    os.environ.get("SHORT_HARDENING_SOURCE_BLOCK_NET_PNL", -0.25)
)
SHORT_HARDENING_COIN_GUARD_ENABLED = os.environ.get(
    "SHORT_HARDENING_COIN_GUARD_ENABLED", "true"
).lower() in ("true", "1", "yes")
SHORT_HARDENING_COIN_MIN_CLOSED_TRADES = int(
    os.environ.get("SHORT_HARDENING_COIN_MIN_CLOSED_TRADES", 4)
)
SHORT_HARDENING_COIN_BLOCK_NET_PNL = float(
    os.environ.get("SHORT_HARDENING_COIN_BLOCK_NET_PNL", -0.25)
)
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
# AUDIT M1 — aggregate exposure caps (two independent metrics).
# FIREWALL_MAX_AGGREGATE_EXPOSURE caps *leveraged notional* (sum of
# size × price × leverage) against balance.  Default 1.50 lets ~4-5
# concurrent 5x leveraged paper positions co-exist for strategy
# evaluation; drop to 0.60-0.80 for live.
FIREWALL_MAX_AGGREGATE_EXPOSURE = float(
    os.environ.get("FIREWALL_MAX_AGGREGATE_EXPOSURE", 1.50)
)
# FIREWALL_MAX_AGGREGATE_MARGIN_PCT caps sum of *margin actually
# locked* (notional / leverage).  Leverage-agnostic capital-at-risk
# view.  Default 0.60 = "no more than 60% of equity locked across all
# positions at once".  Set to 0 to disable.
FIREWALL_MAX_AGGREGATE_MARGIN_PCT = float(
    os.environ.get("FIREWALL_MAX_AGGREGATE_MARGIN_PCT", 0.60)
)
# FIREWALL_AGGREGATE_EXPOSURE_FLOOR_USD — absolute dollar floor for the
# *leveraged-notional* aggregate cap.  FIREWALL_MAX_AGGREGATE_EXPOSURE scales
# with balance, which structurally deadlocks a very small live wallet: 150%
# of a $102 account is only $153 of leveraged notional, so a single mirrored
# position blows it and every subsequent live entry is hard-rejected.  When
# ``balance * FIREWALL_MAX_AGGREGATE_EXPOSURE`` falls below this floor the
# floor is used instead, so a tiny account can still run its intended
# positions.  This does NOT loosen real risk: the leverage-agnostic
# FIREWALL_MAX_AGGREGATE_MARGIN_PCT cap (margin actually locked vs balance)
# remains the true capital-at-risk control and binds first on a small wallet.
# At normal balances the percentage cap already exceeds this floor so behavior
# is unchanged (e.g. $10k paper -> 150% = $15k > $5k floor).  Set 0 to
# disable the floor entirely (pure percentage cap, legacy behavior).
FIREWALL_AGGREGATE_EXPOSURE_FLOOR_USD = float(
    os.environ.get("FIREWALL_AGGREGATE_EXPOSURE_FLOOR_USD", 5000.0)
)
FIREWALL_BLOCK_LOSING_AVERAGING = _safe_env_bool(
    "FIREWALL_BLOCK_LOSING_AVERAGING", True
)
FIREWALL_AVERAGING_MAX_LOSS_ROE_PCT = float(
    os.environ.get("FIREWALL_AVERAGING_MAX_LOSS_ROE_PCT", 0.015)
)
FIREWALL_ENTRY_LOCATION_FILTER_ENABLED = os.environ.get(
    "FIREWALL_ENTRY_LOCATION_FILTER_ENABLED", "true"
).lower() in ("true", "1", "yes")
FIREWALL_ENTRY_MAX_ATR_EXTENSION = float(
    os.environ.get("FIREWALL_ENTRY_MAX_ATR_EXTENSION", 1.8)
)
FIREWALL_ENTRY_MAX_PRICE_EXTENSION_PCT = float(
    os.environ.get("FIREWALL_ENTRY_MAX_PRICE_EXTENSION_PCT", 0.035)
)
FIREWALL_SIDE_IMBALANCE_GUARD_ENABLED = _safe_env_bool(
    "FIREWALL_SIDE_IMBALANCE_GUARD_ENABLED", True
)
FIREWALL_SIDE_IMBALANCE_LOOKBACK_TRADES = int(
    os.environ.get("FIREWALL_SIDE_IMBALANCE_LOOKBACK_TRADES", 60)
)
FIREWALL_SIDE_IMBALANCE_MIN_SAMPLES = int(
    os.environ.get("FIREWALL_SIDE_IMBALANCE_MIN_SAMPLES", 12)
)
FIREWALL_SIDE_IMBALANCE_MAX_SHARE = float(
    os.environ.get("FIREWALL_SIDE_IMBALANCE_MAX_SHARE", 0.80)
)
FIREWALL_SIDE_IMBALANCE_CONFIDENCE_BUMP = float(
    os.environ.get("FIREWALL_SIDE_IMBALANCE_CONFIDENCE_BUMP", 0.15)
)
FIREWALL_SIDE_IMBALANCE_SIZE_MULTIPLIER = float(
    os.environ.get("FIREWALL_SIDE_IMBALANCE_SIZE_MULTIPLIER", 0.50)
)

# Regime-flip exit on live positions (DEFAULT OFF).
# ────────────────────────────────────────────────────────────────
# Closes a live position via reduce-only market order when the bot's
# regime detector AND forecaster both flip against the position's
# direction with high confidence for a sustained number of cycles.
# In addition to (not a replacement for) the SL/TP brackets already
# placed at entry.  When OFF: no positions are closed by this code
# path; existing SL/TP behavior is byte-identical to before.
#
# Layered gates (ALL must pass before close):
#   1. min hold time (anti-whipsaw on fresh entries)
#   2. coin's regime is opposite to position side
#   3. coin's regime confidence >= REGIME_FLIP_EXIT_MIN_CONFIDENCE
#   4. forecaster signal points against position (optional, default ON)
#   5. against-direction has persisted >= MIN_CONSECUTIVE_CYCLES
#
# When DRY_RUN=true (default), the module logs what it WOULD close
# but never sends the order -- safe to enable for observation.
REGIME_FLIP_EXIT_ENABLED = _safe_env_bool(
    "REGIME_FLIP_EXIT_ENABLED", False,
)
REGIME_FLIP_EXIT_DRY_RUN = _safe_env_bool(
    "REGIME_FLIP_EXIT_DRY_RUN", True,
)
REGIME_FLIP_EXIT_MIN_CONFIDENCE = _safe_env_float(
    "REGIME_FLIP_EXIT_MIN_CONFIDENCE", 0.70, lo=0.0, hi=1.0,
)
REGIME_FLIP_EXIT_MIN_CONSECUTIVE_CYCLES = int(
    os.environ.get("REGIME_FLIP_EXIT_MIN_CONSECUTIVE_CYCLES", 2)
)
REGIME_FLIP_EXIT_MIN_HOLD_SECONDS = int(
    os.environ.get("REGIME_FLIP_EXIT_MIN_HOLD_SECONDS", 300)
)
REGIME_FLIP_EXIT_REQUIRE_FORECASTER_AGREE = _safe_env_bool(
    "REGIME_FLIP_EXIT_REQUIRE_FORECASTER_AGREE", True,
)
REGIME_FLIP_EXIT_FORECASTER_MIN_SIGNAL = _safe_env_float(
    "REGIME_FLIP_EXIT_FORECASTER_MIN_SIGNAL", 0.20, lo=0.0, hi=1.0,
)

# Break-even stop policy (DEFAULT OFF).
# ────────────────────────────────────────────────────────────────
# When a live position is profitable by >= BREAK_EVEN_STOP_TRIGGER_PCT,
# move its stop-loss to the entry price (plus a small buffer to cover
# fees).  The position then has a guaranteed non-loss floor while
# remaining open to capture additional upside.
#
# This is the safest of the layered SL policies because it can only
# IMPROVE outcomes -- it never moves SL further from price.  The
# sl_is_tighter guard in src/trading/sl_management.py refuses any
# attempt to loosen the SL.
BREAK_EVEN_STOP_ENABLED = _safe_env_bool("BREAK_EVEN_STOP_ENABLED", False)
BREAK_EVEN_STOP_DRY_RUN = _safe_env_bool("BREAK_EVEN_STOP_DRY_RUN", True)
BREAK_EVEN_STOP_TRIGGER_PCT = _safe_env_float(
    "BREAK_EVEN_STOP_TRIGGER_PCT", 0.01, lo=0.001, hi=0.50,
)
BREAK_EVEN_STOP_BUFFER_PCT = _safe_env_float(
    "BREAK_EVEN_STOP_BUFFER_PCT", 0.001, lo=0.0, hi=0.05,
)

# Time-decay SL tightening policy (DEFAULT OFF).
# ────────────────────────────────────────────────────────────────
# The longer a position is held without resolution (no SL/TP/reconcile/
# break-even promotion), the tighter its SL becomes.  Caps slow-bleed
# losses on positions whose trade thesis has gone stale.
#
# Discrete bands, not continuous trailing -- each cancel/replace is a
# meaningful step (not whipsaw-prone) and API rate stays bounded.
#
# Band schedule expressed as (age_seconds, distance_factor).  Factor is
# the fraction of the CURRENT SL distance the new SL keeps after
# tightening.  Smaller factor = tighter SL.  At factor=0.25 a position
# whose SL was 3% away ends up at ~0.75% from current price.
TIME_DECAY_SL_ENABLED = _safe_env_bool("TIME_DECAY_SL_ENABLED", False)
TIME_DECAY_SL_DRY_RUN = _safe_env_bool("TIME_DECAY_SL_DRY_RUN", True)
# Band 1: 30 min -> 75% of current SL distance
TIME_DECAY_SL_BAND1_SECONDS = int(
    os.environ.get("TIME_DECAY_SL_BAND1_SECONDS", 1800)
)
TIME_DECAY_SL_BAND1_FACTOR = _safe_env_float(
    "TIME_DECAY_SL_BAND1_FACTOR", 0.75, lo=0.10, hi=1.00,
)
# Band 2: 90 min -> 50%
TIME_DECAY_SL_BAND2_SECONDS = int(
    os.environ.get("TIME_DECAY_SL_BAND2_SECONDS", 5400)
)
TIME_DECAY_SL_BAND2_FACTOR = _safe_env_float(
    "TIME_DECAY_SL_BAND2_FACTOR", 0.50, lo=0.05, hi=1.00,
)
# Band 3: 180 min -> 25%
TIME_DECAY_SL_BAND3_SECONDS = int(
    os.environ.get("TIME_DECAY_SL_BAND3_SECONDS", 10800)
)
TIME_DECAY_SL_BAND3_FACTOR = _safe_env_float(
    "TIME_DECAY_SL_BAND3_FACTOR", 0.25, lo=0.05, hi=1.00,
)
# Band 4: 240 min -> 25% (no further tightening beyond this)
TIME_DECAY_SL_BAND4_SECONDS = int(
    os.environ.get("TIME_DECAY_SL_BAND4_SECONDS", 14400)
)
TIME_DECAY_SL_BAND4_FACTOR = _safe_env_float(
    "TIME_DECAY_SL_BAND4_FACTOR", 0.25, lo=0.05, hi=1.00,
)

# Trailing stop policy (DEFAULT OFF).
# ────────────────────────────────────────────────────────────────
# Trail SL behind the favourable-direction high/low-water mark by a
# fixed offset.  Captures profit on positions that run past the
# original TP distance.  Activates only after a position is already
# in profit by >= TRAILING_STOP_ACTIVATION_PROFIT_PCT so it doesn't
# fight the break-even path on early moves.
#
# Min-step throttle prevents tiny SL adjustments on noise -- only
# moves SL when the proposed change is at least MIN_STEP_PCT past
# the current SL.
TRAILING_STOP_ENABLED = _safe_env_bool("TRAILING_STOP_ENABLED", False)
TRAILING_STOP_DRY_RUN = _safe_env_bool("TRAILING_STOP_DRY_RUN", True)
# Profit % required before the trail activates.  Below this we leave
# SL alone (break-even handles the 0-1% zone).
TRAILING_STOP_ACTIVATION_PROFIT_PCT = _safe_env_float(
    "TRAILING_STOP_ACTIVATION_PROFIT_PCT", 0.01, lo=0.0, hi=0.50,
)
# Distance from HWM/LWM at which the trailing SL sits.
TRAILING_STOP_OFFSET_PCT = _safe_env_float(
    "TRAILING_STOP_OFFSET_PCT", 0.01, lo=0.001, hi=0.20,
)
# Minimum SL move (% of current SL) before we cancel+replace.  Stops
# the policy from spamming the exchange on every tiny tick.
TRAILING_STOP_MIN_STEP_PCT = _safe_env_float(
    "TRAILING_STOP_MIN_STEP_PCT", 0.002, lo=0.0, hi=0.05,
)

# Cross-asset momentum override: when core majors break out together, block
# countertrend entries and pause mean-reversion-style fades. Auto-closing
# countertrend live positions is available but off by default.
GLOBAL_MOMENTUM_OVERRIDE_ENABLED = _safe_env_bool(
    "GLOBAL_MOMENTUM_OVERRIDE_ENABLED", True
)
GLOBAL_MOMENTUM_CORE_COINS = _parse_coin_list(
    os.environ.get("GLOBAL_MOMENTUM_CORE_COINS", "BTC,ETH,SOL")
)
GLOBAL_MOMENTUM_MIN_AGREEING_COINS = int(
    os.environ.get("GLOBAL_MOMENTUM_MIN_AGREEING_COINS", 2)
)
GLOBAL_MOMENTUM_MIN_CONFIDENCE = float(
    os.environ.get("GLOBAL_MOMENTUM_MIN_CONFIDENCE", 0.58)
)
GLOBAL_MOMENTUM_MIN_MOMENTUM = float(
    os.environ.get("GLOBAL_MOMENTUM_MIN_MOMENTUM", 0.006)
)
GLOBAL_MOMENTUM_MIN_VOLUME_RATIO = float(
    os.environ.get("GLOBAL_MOMENTUM_MIN_VOLUME_RATIO", 0.75)
)
GLOBAL_MOMENTUM_CLOSE_COUNTERTREND = _safe_env_bool(
    "GLOBAL_MOMENTUM_CLOSE_COUNTERTREND", False
)
BTC_MARKET_LEADER_GUARD_ENABLED = _safe_env_bool(
    "BTC_MARKET_LEADER_GUARD_ENABLED", True
)
BTC_MARKET_LEADER_COIN = os.environ.get("BTC_MARKET_LEADER_COIN", "BTC").strip().upper() or "BTC"
BTC_MARKET_LEADER_MIN_CONFIDENCE = float(
    os.environ.get("BTC_MARKET_LEADER_MIN_CONFIDENCE", GLOBAL_MOMENTUM_MIN_CONFIDENCE)
)
BTC_MARKET_LEADER_MIN_MOMENTUM = float(
    os.environ.get("BTC_MARKET_LEADER_MIN_MOMENTUM", 0.003)
)
BTC_MARKET_LEADER_MIN_VOLUME_RATIO = float(
    os.environ.get("BTC_MARKET_LEADER_MIN_VOLUME_RATIO", GLOBAL_MOMENTUM_MIN_VOLUME_RATIO)
)

# Directional market-side guard: blocks/de-risks entries fighting a strong
# current market read. This closes the old asymmetry where short-side history
# hardening could make shorts harder than longs during bearish BTC momentum.
FIREWALL_MARKET_SIDE_GUARD_ENABLED = _safe_env_bool(
    "FIREWALL_MARKET_SIDE_GUARD_ENABLED", True
)
FIREWALL_MARKET_SIDE_GUARD_MIN_CONFIDENCE = float(
    os.environ.get("FIREWALL_MARKET_SIDE_GUARD_MIN_CONFIDENCE", 0.60)
)
# Market-read inputs for the market-side guard.  Without these a lone
# regime label (e.g. trending_down @74%) silently vetoes a high-conviction
# counter-regime entry -- the "BULLISH options flow shown but LONG blocked,
# only SHORTs persist" bug.  Letting strong fresh options-flow conviction
# (and the down-weighted synthetic forecaster) count as market-read
# candidates lets independent bullish/bearish confluence satisfy the guard
# instead of being unilaterally overridden.  A hard crash/panic carve-out
# below still refuses to let one options print buy into a confirmed crash.
FIREWALL_MARKET_READ_USES_OPTIONS_FLOW = _safe_env_bool(
    "FIREWALL_MARKET_READ_USES_OPTIONS_FLOW", True
)
# Minimum options-flow conviction (0-1) before it counts as a market read.
FIREWALL_OPTIONS_FLOW_READ_MIN_CONVICTION = float(
    os.environ.get("FIREWALL_OPTIONS_FLOW_READ_MIN_CONVICTION", 0.70)
)
# If an opposite-side crash/panic regime read is at/above this confidence,
# options flow may NOT grant alignment (never buy a confirmed crash on one
# print).  Options flow can still override a *moderate* trending_down.
FIREWALL_OPTIONS_FLOW_OVERRIDE_MAX_REGIME_CONF = float(
    os.environ.get("FIREWALL_OPTIONS_FLOW_OVERRIDE_MAX_REGIME_CONF", 0.85)
)
# Synthetic warm-start forecaster is no longer discarded outright; it
# contributes at this confidence weight (0 = ignore, 1 = full weight).
# Default 0.5 keeps it able to *align* but not single-handedly *block*.
FIREWALL_FORECASTER_SYNTHETIC_WEIGHT = float(
    os.environ.get("FIREWALL_FORECASTER_SYNTHETIC_WEIGHT", 0.5)
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
    os.environ.get("SOURCE_POLICY_WARMUP_MAX_SIGNALS_PER_DAY", 4)
)
SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY = int(
    os.environ.get("SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY", 1)
)
# Options-flow per-day cap graduation. Warmup/degraded fixed caps
# throttle an options_flow directional source to ~1 signal/day until
# it has a track record (prod 6h scan: 23 of 93 decisions rejected on
# this cap). Once a source whose key starts with ``options_flow`` has
# produced MORE THAN OPTIONS_FLOW_CAP_MIN_TRADES closed trades, its
# per-day cap is lifted to OPTIONS_FLOW_GRADUATED_CAP. Never overrides
# a paused/blocked source (hard safety stop stays hard).
OPTIONS_FLOW_CAP_GRADUATION_ENABLED = os.environ.get(
    "OPTIONS_FLOW_CAP_GRADUATION_ENABLED", "true"
).lower() in ("true", "1", "yes")
OPTIONS_FLOW_CAP_MIN_TRADES = int(
    os.environ.get("OPTIONS_FLOW_CAP_MIN_TRADES", 3)
)
OPTIONS_FLOW_GRADUATED_CAP = int(
    os.environ.get("OPTIONS_FLOW_GRADUATED_CAP", 4)
)
# Regime-aware LLM exhaustion guard. The exhaustion-trap block (no
# shorting RSI<22 / no longing RSI>78) protects against reversal in
# ranging/volatile/contra-regime contexts, but in a CONFIRMED strong
# trend it inverts -- shorting RSI<22 while regime==TRENDING_DOWN is
# trend continuation, the highest-conviction setup. A blanket hard
# block there deadlocks the core strategy (observed: LLM pass_rate 4%,
# 0 orders while the bot wanted to short a trending_down market). When
# enabled, trend-aligned signals are de-risked (confidence *=
# LLM_EXHAUSTION_TREND_ALIGNED_CONF_MULT) instead of hard-blocked;
# non-aligned contexts keep the hard block.
LLM_EXHAUSTION_REGIME_AWARE = os.environ.get(
    "LLM_EXHAUSTION_REGIME_AWARE", "true"
).lower() in ("true", "1", "yes")
LLM_EXHAUSTION_TREND_ALIGNED_CONF_MULT = _safe_env_float(
    "LLM_EXHAUSTION_TREND_ALIGNED_CONF_MULT", 0.85, lo=0.1, hi=1.0
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
SOURCE_POLICY_DYNAMIC_CAPS_ENABLED = _safe_env_bool(
    "SOURCE_POLICY_DYNAMIC_CAPS_ENABLED", True,
)
SOURCE_POLICY_ACTIVE_MIN_SIGNALS_PER_DAY = _safe_env_int(
    "SOURCE_POLICY_ACTIVE_MIN_SIGNALS_PER_DAY", 3, lo=0, hi=100_000,
)
SOURCE_POLICY_ACTIVE_MAX_SIGNALS_PER_DAY = _safe_env_int(
    "SOURCE_POLICY_ACTIVE_MAX_SIGNALS_PER_DAY", 8, lo=0, hi=100_000,
)
SOURCE_POLICY_STRONG_MIN_CLOSED_TRADES = _safe_env_int(
    "SOURCE_POLICY_STRONG_MIN_CLOSED_TRADES", 12, lo=1, hi=100_000,
)
SOURCE_POLICY_STRONG_WIN_RATE = _safe_env_float(
    "SOURCE_POLICY_STRONG_WIN_RATE", 0.55, lo=0.0, hi=1.0,
)
SOURCE_POLICY_STRONG_RECENT_PNL_FLOOR = _safe_env_float(
    "SOURCE_POLICY_STRONG_RECENT_PNL_FLOOR", 0.0, lo=-1_000_000.0, hi=1_000_000.0,
)

# Runtime readiness / incident monitoring.
READINESS_STALE_SECONDS = int(os.environ.get("READINESS_STALE_SECONDS", 600))
READINESS_DB_WRITE_TTL_S = int(os.environ.get("READINESS_DB_WRITE_TTL_S", 60))
READINESS_REQUIRE_HEALTH_REGISTRY = os.environ.get(
    "READINESS_REQUIRE_HEALTH_REGISTRY", "false"
).lower() in ("true", "1", "yes")
READINESS_ALERT_COOLDOWN_S = int(
    os.environ.get("READINESS_ALERT_COOLDOWN_S", 900)
)

# ─── Decision-engine signal sources (arena/polymarket/options_flow) ───
# These three subsystems all log activity but were producing zero signals in
# the May logs because their thresholds were tuned for steady-state, not
# bootstrap. Defaults below are looser than before but still real filters.
OPTIONS_FLOW_MIN_NOTIONAL = int(
    os.environ.get("OPTIONS_FLOW_MIN_NOTIONAL", 10_000)
)
OPTIONS_FLOW_MIN_VOL_OI_RATIO = float(
    os.environ.get("OPTIONS_FLOW_MIN_VOL_OI_RATIO", 0.08)
)
OPTIONS_FLOW_MIN_CONVICTION_PCT = float(
    os.environ.get("OPTIONS_FLOW_MIN_CONVICTION_PCT", 30.0)
)
ARENA_CHAMPION_MIN_FITNESS = float(
    os.environ.get("ARENA_CHAMPION_MIN_FITNESS", 0.10)
)
ARENA_CHAMPION_MIN_TRADES = int(
    os.environ.get("ARENA_CHAMPION_MIN_TRADES", 3)
)
ARENA_CHAMPION_MIN_WIN_RATE = float(
    os.environ.get("ARENA_CHAMPION_MIN_WIN_RATE", 0.45)
)
POLYMARKET_MIN_VOLUME_THRESHOLD = int(
    os.environ.get("POLYMARKET_MIN_VOLUME_THRESHOLD", 5_000)
)
POLYMARKET_MIN_LIQUIDITY_THRESHOLD = int(
    os.environ.get("POLYMARKET_MIN_LIQUIDITY_THRESHOLD", 500)
)

# ─── Cross-Venue Hedger ────────────────────────────────────────
# Default hedge venue is Kraken Futures (futures.kraken.com). Binance/Bybit
# code paths exist but live execution is NOT implemented for those — leave
# disabled. dry_run defaults to True; flip to False only after credentials
# are set and you've reviewed at least one [DRY-RUN] log cycle.
HEDGER_DRY_RUN = os.environ.get(
    "HEDGER_DRY_RUN", "true"
).strip().lower() in ("1", "true", "yes", "on")
HEDGER_HEDGE_RATIO = float(os.environ.get("HEDGER_HEDGE_RATIO", 0.5))
HEDGER_CRASH_CONFIDENCE = float(os.environ.get("HEDGER_CRASH_CONFIDENCE", 0.5))
HEDGER_KRAKEN_ENABLED = os.environ.get(
    "HEDGER_KRAKEN_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
HEDGER_BINANCE_ENABLED = os.environ.get(
    "HEDGER_BINANCE_ENABLED", "false"
).strip().lower() in ("1", "true", "yes", "on")
HEDGER_BYBIT_ENABLED = os.environ.get(
    "HEDGER_BYBIT_ENABLED", "false"
).strip().lower() in ("1", "true", "yes", "on")
HEDGER_KRAKEN_SYMBOL_TEMPLATE = os.environ.get(
    "HEDGER_KRAKEN_SYMBOL_TEMPLATE", "PF_{COIN}USD"
)
HEDGER_KRAKEN_ORDER_TYPE = os.environ.get("HEDGER_KRAKEN_ORDER_TYPE", "mkt")
HEDGER_RATE_LIMIT_MS = int(os.environ.get("HEDGER_RATE_LIMIT_MS", 100))

# ─── PositionMonitor (WebSocket subscriptions) ─────────────────
# When True, only bootstrap and subscribe to tracked wallets that have shown
# positions or fills within the last POSITION_MONITOR_ACTIVITY_LOOKBACK_S
# seconds. Inactive wallets are skipped until the next periodic full refresh
# (POSITION_MONITOR_FULL_REFRESH_S). This cuts wasted REST calls and
# Hyperliquid "Inactive" reconnect churn when tracking large lists.
POSITION_MONITOR_ACTIVITY_FILTER_ENABLED = os.environ.get(
    "POSITION_MONITOR_ACTIVITY_FILTER_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
POSITION_MONITOR_ACTIVITY_LOOKBACK_S = int(
    os.environ.get("POSITION_MONITOR_ACTIVITY_LOOKBACK_S", 21600)  # 6 hours
)
POSITION_MONITOR_FULL_REFRESH_S = int(
    os.environ.get("POSITION_MONITOR_FULL_REFRESH_S", 86400)  # 24 hours
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
LIVE_EXECUTION_VENUE = os.environ.get("LIVE_EXECUTION_VENUE", "hyperliquid").strip().lower()
if LIVE_EXECUTION_VENUE not in {"hyperliquid", "lighter"}:
    LIVE_EXECUTION_VENUE = "hyperliquid"
LIGHTER_ENABLED = os.environ.get("LIGHTER_ENABLED", "true").strip().lower() in ("true", "1", "yes")
LIGHTER_STRATEGY_INJECTION_ENABLED = os.environ.get(
    "LIGHTER_STRATEGY_INJECTION_ENABLED", "false"
).strip().lower() in ("true", "1", "yes")
LIGHTER_STRATEGY_INJECTION_LIMIT = _safe_env_int("LIGHTER_STRATEGY_INJECTION_LIMIT", 25, lo=1, hi=250)
LIGHTER_STRATEGY_MIN_VOLUME_USD = _safe_env_float(
    "LIGHTER_STRATEGY_MIN_VOLUME_USD", 10_000.0, lo=0.0, hi=100_000_000.0
)
LIGHTER_LIVE_TRADING_ENABLED = os.environ.get(
    "LIGHTER_LIVE_TRADING_ENABLED", "false"
).strip().lower() in ("true", "1", "yes")
LIGHTER_LIVE_TRADING_DUAL_CONTROL_CONFIRM = os.environ.get(
    "LIGHTER_LIVE_TRADING_DUAL_CONTROL_CONFIRM", "false"
).strip().lower() in ("true", "1", "yes")
LIGHTER_BASE_URL = os.environ.get("LIGHTER_BASE_URL", "https://mainnet.zklighter.elliot.ai").strip()
LIGHTER_ACCOUNT_INDEX = _safe_env_int("LIGHTER_ACCOUNT_INDEX", -1, lo=-1, hi=10_000_000)
LIGHTER_API_KEY_INDEX = _safe_env_int("LIGHTER_API_KEY_INDEX", 0, lo=0, hi=10_000)
LIGHTER_PRIVATE_KEY = os.environ.get("LIGHTER_PRIVATE_KEY", "").strip()
LIGHTER_L1_ADDRESS = os.environ.get("LIGHTER_L1_ADDRESS", "").strip()
LIGHTER_MIN_ORDER_USD = _safe_env_float("LIGHTER_MIN_ORDER_USD", 1.0, lo=0.0, hi=10_000.0)
LIGHTER_MAX_ORDER_USD = _safe_env_float("LIGHTER_MAX_ORDER_USD", 100.0, lo=1.0, hi=10_000_000.0)
LIGHTER_DEFAULT_LEVERAGE = _safe_env_float("LIGHTER_DEFAULT_LEVERAGE", 5.0, lo=1.0, hi=50.0)
LIGHTER_MAX_SLIPPAGE_BPS = _safe_env_float("LIGHTER_MAX_SLIPPAGE_BPS", 20.0, lo=0.0, hi=1000.0)
LIGHTER_SIZE_DECIMALS_DEFAULT = _safe_env_int("LIGHTER_SIZE_DECIMALS_DEFAULT", 4, lo=0, hi=18)
LIGHTER_PRICE_DECIMALS_DEFAULT = _safe_env_int("LIGHTER_PRICE_DECIMALS_DEFAULT", 2, lo=0, hi=18)

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
ARENA_REQUIRE_CONTRARIAN_VALIDATION = _safe_env_bool(
    "ARENA_REQUIRE_CONTRARIAN_VALIDATION", True
)
ARENA_HIGH_CONFIDENCE_THRESHOLD = float(
    os.environ.get("ARENA_HIGH_CONFIDENCE_THRESHOLD", 0.80)
)
ARENA_UNVALIDATED_CONFIDENCE_CAP = float(
    os.environ.get("ARENA_UNVALIDATED_CONFIDENCE_CAP", 0.74)
)
# Total virtual capital allocated across all Alpha Arena agents. Per-agent share
# is TOTAL / N_active. Default matches the paper trader starting balance so the
# Arena scoreboard stays consistent with paper equity instead of inflating to
# $90K against a $10K paper account.
ARENA_TOTAL_POOL_USD = _safe_env_float(
    "ARENA_TOTAL_POOL_USD", 10_000.0, lo=10.0, hi=10_000_000.0
)

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
XGBOOST_SYNTHETIC_MAX_CONFIDENCE = _safe_env_float(
    "XGB_SYNTHETIC_MAX_CONFIDENCE", 0.45, lo=0.0, hi=0.60,
)
# Forward-return labeler: turns past predictions into observed training labels
# by inspecting price moves over a forward window. Without this, the model
# only ever sees synthetic warm-start data (source=synthetic in logs).
XGBOOST_LABELER_ENABLED = os.environ.get(
    "XGBOOST_LABELER_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
XGBOOST_LABELER_FORWARD_MINUTES = int(
    os.environ.get("XGBOOST_LABELER_FORWARD_MINUTES", 60)
)
# Forward % move thresholds. Tighter than typical signal thresholds because
# regime labels are about market state, not trade direction.
XGBOOST_LABELER_CRASH_PCT = float(
    os.environ.get("XGBOOST_LABELER_CRASH_PCT", -0.015)  # -1.5%
)
XGBOOST_LABELER_BULLISH_PCT = float(
    os.environ.get("XGBOOST_LABELER_BULLISH_PCT", 0.015)  # +1.5%
)
XGBOOST_LABELER_BATCH_SIZE = int(
    os.environ.get("XGBOOST_LABELER_BATCH_SIZE", 200)
)
# Minimum age of a prediction before we'll label it (lets the forward window
# fully mature; should be >= forward_minutes).
XGBOOST_LABELER_MIN_AGE_MINUTES = int(
    os.environ.get("XGBOOST_LABELER_MIN_AGE_MINUTES", 65)
)

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
RL_SIZER_APPLY_TO_ORDERS = _safe_env_bool("RL_SIZER_APPLY_TO_ORDERS", False)
RL_SIZER_RETRAIN_INTERVAL = int(os.environ.get("RL_SIZER_RETRAIN_INTERVAL", 43200))  # 12 hours
RL_SIZER_TRAINING_EPISODES = int(os.environ.get("RL_SIZER_TRAINING_EPISODES", 500))
RL_SIZER_MODEL_DIR = os.environ.get("RL_SIZER_MODEL_DIR", "models/rl_sizer")
# Minimum closed-trade count required before the RL sizer trains. Lowered
# from 100 to 50 so the model bootstraps faster; shadow trades are added to
# the dataset (see RL_SIZER_USE_SHADOW_DATA) to widen the sample.
RL_SIZER_MIN_TRAINING_TRADES = int(os.environ.get("RL_SIZER_MIN_TRAINING_TRADES", 50))
# When true, shadow_tracker pnl_pct values are concatenated with Kelly's
# realized returns at training time. This lets the sizer learn from signals
# the firewall blocked, expanding the dataset without taking risk.
RL_SIZER_USE_SHADOW_DATA = os.environ.get(
    "RL_SIZER_USE_SHADOW_DATA", "true"
).strip().lower() in ("1", "true", "yes", "on")
RL_SIZER_SHADOW_LOOKBACK_DAYS = int(
    os.environ.get("RL_SIZER_SHADOW_LOOKBACK_DAYS", 90)
)

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
POLYMARKET_MIN_VOLUME = float(os.environ.get("POLYMARKET_MIN_VOLUME", 5000))    # $5k min volume
POLYMARKET_MIN_LIQUIDITY = float(
    os.environ.get("POLYMARKET_MIN_LIQUIDITY", 500)
)  # $500 min liquidity
POLYMARKET_MAX_MARKETS_PER_SCAN = int(
    os.environ.get("POLYMARKET_MAX_MARKETS_PER_SCAN", 100)
)
POLYMARKET_TRADE_BACKFILL_SOURCE = str(
    os.environ.get("POLYMARKET_TRADE_BACKFILL_SOURCE", "data_api") or "data_api"
).strip().lower()
if POLYMARKET_TRADE_BACKFILL_SOURCE not in {"data_api", "clob"}:
    POLYMARKET_TRADE_BACKFILL_SOURCE = "data_api"
POLYMARKET_TRADE_BACKFILL_TAKER_ONLY = os.environ.get(
    "POLYMARKET_TRADE_BACKFILL_TAKER_ONLY", "false"
).lower() in ("true", "1", "yes")
POLYMARKET_TRADE_BACKFILL_LIMIT_PER_MARKET = int(
    os.environ.get("POLYMARKET_TRADE_BACKFILL_LIMIT_PER_MARKET", 200)
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
        ("MIN_STRATEGY_SCORE", 0.0, 1.0, 0.20),
        ("MAX_ACTIVE_STRATEGIES", 1, 5000, 25),
        ("MIN_ACTIVE_STRATEGIES", 1, 500, 5),
        ("MAX_STRATEGIES_PER_CYCLE", 1, 200, 15),
        ("STRATEGY_RECOVERY_TARGET_ACTIVE_VALID", 1, 500, 15),
        ("PAPER_TRADING_MAX_LEVERAGE", 1.0, 25.0, 5.0),
        ("PAPER_TRADING_STOP_LOSS_PCT", 0.001, 1.0, 0.15),
        ("PAPER_TRADING_TAKE_PROFIT_PCT", 0.001, 5.0, 0.75),
        ("LIVE_MIN_ORDER_USD", 10.0, 1_000_000.0, 11.0),
        ("LIVE_MAX_ORDER_USD", 10.0, 1_000_000.0, 150.0),
        ("LIVE_MAX_POSITION_SIZE_USD", 10.0, 10_000_000.0, 150.0),
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
        ("FIREWALL_MIN_CONFIDENCE", 0.0, 1.0, 0.40),
        ("FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY", 0, 100_000, 0),
        ("FIREWALL_COIN_COOLDOWN_SECONDS", 0, 86_400, 180),
        ("FIREWALL_SAME_SIDE_COOLDOWN_SECONDS", 0, 86_400, 900),
        ("FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN", 1, 20, 2),
        ("FIREWALL_CANARY_MAX_POSITIONS", 1, 100, 2),
        # AUDIT M1 — leveraged notional and margin caps
        ("FIREWALL_MAX_AGGREGATE_EXPOSURE", 0.0, 20.0, 1.50),
        ("FIREWALL_MAX_AGGREGATE_MARGIN_PCT", 0.0, 5.0, 0.60),
        ("FIREWALL_AGGREGATE_EXPOSURE_FLOOR_USD", 0.0, 10_000_000.0, 5000.0),
        ("SOURCE_POLICY_MIN_CLOSED_TRADES", 1, 1000, 3),
        ("SOURCE_POLICY_KEEP_TOP_N", 1, 1000, 5),
        ("SOURCE_POLICY_PAUSE_WEIGHT", 0.0, 1.0, 0.12),
        ("SOURCE_POLICY_DEGRADE_WEIGHT", 0.0, 1.0, 0.32),
        ("SOURCE_POLICY_WARMUP_MAX_SIGNALS_PER_DAY", 0, 100_000, 4),
        ("SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY", 0, 100_000, 1),
        ("SOURCE_POLICY_WARMUP_SIZE_MULTIPLIER", 0.0, 1.0, 0.75),
        ("SOURCE_POLICY_DEGRADED_SIZE_MULTIPLIER", 0.0, 1.0, 0.60),
        ("SOURCE_POLICY_WARMUP_MIN_CONFIDENCE", 0.0, 1.0, 0.45),
        ("SOURCE_POLICY_DEGRADED_MIN_CONFIDENCE", 0.0, 1.0, 0.55),
        ("SOURCE_POLICY_ACTIVE_MIN_SIGNALS_PER_DAY", 0, 100_000, 3),
        ("SOURCE_POLICY_ACTIVE_MAX_SIGNALS_PER_DAY", 0, 100_000, 8),
        ("SOURCE_POLICY_STRONG_MIN_CLOSED_TRADES", 1, 100_000, 12),
        ("SOURCE_POLICY_STRONG_WIN_RATE", 0.0, 1.0, 0.55),
        ("SOURCE_POLICY_STRONG_RECENT_PNL_FLOOR", -1_000_000.0, 1_000_000.0, 0.0),
        ("TRADING_CYCLE_INTERVAL", 10, 86_400, 900),
        ("DISCOVERY_CYCLE_INTERVAL", 60, 2_592_000, 86400),
        ("POLYMARKET_SCAN_INTERVAL", 10, 3600, 180),
        ("POLYMARKET_MAX_MARKETS_PER_SCAN", 10, 10_000, 100),
        ("POLYMARKET_TRADE_BACKFILL_LIMIT_PER_MARKET", 1, 10_000, 200),
        ("OPTIONS_FLOW_SCAN_INTERVAL", 10, 3600, 120),
        ("FEATURE_STORE_BOOTSTRAP_TOP_COINS", 0, 100, 8),
        ("HL_BOT_BACKUP_MAX_WALLET_FILLS", 0, 1_000_000, 5000),
        ("HL_BOT_BACKUP_MAX_GOLDEN_WALLETS", 0, 100_000, 200),
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
        ("BOT_HARD_CUTOFF_TRADES", 1, 100_000, 80),
        ("BOT_THRESHOLD", 1, 100, 3),
        ("BOT_ELEVATED_FREQ", 1, 100_000, 30),
        ("BOT_PERFECT_WINRATE", 0.50, 1.0, 0.98),
        ("BOT_PERFECT_WINRATE_MIN_TRADES", 1, 100_000, 15),
        ("TRADER_MIN_CLOSED_TRADES", 0, 100_000, 10),
        ("PORTFOLIO_CHURN_PENALTY", 0.0, 1.0, 0.02),
        ("PORTFOLIO_MIN_HOLD_MINUTES", 0, 525_600, 60),
        ("ROTATION_SHADOW_MODE_DAYS", 0, 365, 7),
        ("FORECASTER_CRASH_THRESHOLD", -1.0, 0.0, -0.15),
        ("XGBOOST_CRASH_THRESHOLD", -1.0, 0.0, -0.18),
        ("XGB_SYNTHETIC_MAX_CONFIDENCE", 0.0, 0.60, 0.45),
        ("FUNDING_NEGATIVE_THRESHOLD", -1.0, 0.0, -0.001),
        ("FUNDING_POSITIVE_THRESHOLD", 0.0, 1.0, 0.003),
        ("DB_AUDIT_NON_ACTIVE_REGIME_RETENTION_DAYS", 0.0, 3650.0, 7.0),
        ("POLYMARKET_MIN_VOLUME", 0.0, 1e9, 5_000.0),
        ("POLYMARKET_MIN_LIQUIDITY", 0.0, 1e9, 500.0),
        ("LIVE_CANARY_MAX_ORDER_USD", 10.0, 1_000_000.0, 25.0),
        ("LIVE_CANARY_MAX_SIGNALS_PER_DAY", 1, 100_000, 25),
        ("LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 0, 100_000, 0),
        ("LIVE_RISK_PER_TRADE_PCT", 0.0, 0.25, 0.0075),
        ("LIVE_MAX_MARGIN_PER_ORDER_PCT", 0.0, 1.0, 0.12),
        ("LIVE_MIN_MARGIN_PER_ORDER_USD", 0.0, 1_000_000.0, 0.0),
        ("LIVE_ORDER_HYGIENE_AUDIT_INTERVAL_CYCLES", 1, 100_000, 5),
        ("LIVE_MIN_ORDER_TOP_TIER_MIN_CONFIDENCE", 0.0, 1.0, 0.72),
        ("LIVE_MIN_ORDER_TOP_TIER_MAX_BUMP_MULTIPLIER", 1.0, 10.0, 1.35),
        ("LIVE_MIN_ORDER_SHORT_MIN_CONFIDENCE", 0.0, 1.0, 0.75),
        ("LIVE_MIN_ORDER_SAME_SIDE_MAX_BUMP_MULTIPLIER", 1.0, 10.0, 2.5),
        ("LIVE_ANALYTICS_LOOKBACK_TRADES", 10, 5_000, 200),
        ("LIVE_MAKER_ENTRY_OFFSET_BPS", 0.0, 100.0, 1.0),
        ("LIVE_MAKER_ENTRY_TIMEOUT_S", 0.0, 30.0, 2.5),
        ("LIVE_SCHEDULE_CANCEL_ENTRY_TIMEOUT_S", 5.0, 86_400.0, 60.0),
        ("LIVE_SCHEDULE_CANCEL_WORKING_TIMEOUT_S", 5.0, 86_400.0, 300.0),
        ("REGIME_REVERSAL_MIN_CONFIDENCE", 0.0, 1.0, 0.70),
        ("REGIME_REVERSAL_REVERSE_CONFIDENCE", 0.0, 1.0, 0.82),
        ("REGIME_REVERSAL_CONFIRM_CYCLES", 1, 100, 3),
        ("REGIME_REVERSAL_MIN_POSITION_AGE_SECONDS", 0, 86_400, 180),
        ("REGIME_REVERSAL_COOLDOWN_SECONDS", 0, 86_400, 900),
        ("REGIME_REVERSAL_MAX_ACTIONS_PER_COIN_PER_DAY", 0, 100, 2),
        ("REGIME_REVERSAL_TIGHTEN_STOP_R_MULTIPLE", 0.01, 2.0, 0.35),
        ("REGIME_REVERSAL_REVERSE_POSITION_PCT", 0.001, 0.50, 0.03),
        ("COPY_TRADER_MAX_CONCURRENT_TRADES", 0, 100, 5),
        ("COPY_TRADER_MAX_NEW_TRADES_PER_CYCLE", 0, 100, 1),
        ("COPY_TRADER_AUTO_PAUSE_MIN_CLOSED_TRADES", 1, 5_000, 6),
        ("COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE", 0.0, 1.0, 0.40),
        ("COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE", 0.0, 1.0, 0.25),
        ("COPY_TRADER_AUTO_PAUSE_BLOCK_NET_PNL", -1_000_000.0, 1_000_000.0, -25.0),
        ("COPY_TRADER_SOURCE_SIDE_MIN_CLOSED_TRADES", 1, 5_000, 3),
        ("COPY_TRADER_SOURCE_SIDE_DEGRADE_WIN_RATE", 0.0, 1.0, 0.45),
        ("COPY_TRADER_SOURCE_SIDE_BLOCK_WIN_RATE", 0.0, 1.0, 0.35),
        ("COPY_TRADER_SOURCE_SIDE_BLOCK_NET_PNL", -1_000_000.0, 1_000_000.0, -0.25),
        ("COPY_TRADER_SOURCE_SIDE_CONFIDENCE_MULTIPLIER", 0.0, 1.0, 0.75),
        ("COPY_TRADER_SOURCE_SIDE_SIZE_MULTIPLIER", 0.0, 1.0, 0.50),
        ("COPY_TRADER_COUNTERTREND_BLOCK_MIN_CONFIDENCE", 0.0, 1.0, 0.58),
        ("COPY_TRADER_SYNTHETIC_REGIME_CONFIDENCE_CAP", 0.0, 1.0, 0.50),
        ("SHORT_HARDENING_LOOKBACK_TRADES", 10, 5_000, 120),
        ("SHORT_HARDENING_MIN_CLOSED_TRADES", 1, 1_000, 12),
        ("SHORT_HARDENING_DEGRADE_WIN_RATE", 0.0, 1.0, 0.48),
        ("SHORT_HARDENING_BLOCK_WIN_RATE", 0.0, 1.0, 0.30),
        ("SHORT_HARDENING_BLOCK_NET_PNL", -1_000_000.0, 1_000_000.0, -25.0),
        ("SHORT_HARDENING_CONFIDENCE_MULTIPLIER", 0.0, 1.0, 0.80),
        ("SHORT_HARDENING_SIZE_MULTIPLIER", 0.0, 1.0, 0.50),
        ("SHORT_HARDENING_BLOCK_OVERRIDE_MIN_CONFIDENCE", 0.0, 1.0, 0.70),
        ("SHORT_HARDENING_BLOCK_OVERRIDE_MIN_REGIME_CONFIDENCE", 0.0, 1.0, 0.60),
        ("SHORT_HARDENING_BLOCK_OVERRIDE_SIZE_MULTIPLIER", 0.0, 1.0, 0.35),
        ("SHORT_HARDENING_MARKET_ADAPTIVE_MIN_MOMENTUM", 0.0, 1.0, 0.003),
        ("SHORT_HARDENING_MARKET_ADAPTIVE_SCOPED_SIZE_MULTIPLIER", 0.0, 1.0, 0.25),
        ("SHORT_HARDENING_SOURCE_MIN_CLOSED_TRADES", 1, 1_000, 3),
        ("SHORT_HARDENING_SOURCE_BLOCK_NET_PNL", -1_000_000.0, 1_000_000.0, -0.25),
        ("SHORT_HARDENING_COIN_MIN_CLOSED_TRADES", 1, 1_000, 4),
        ("SHORT_HARDENING_COIN_BLOCK_NET_PNL", -1_000_000.0, 1_000_000.0, -0.25),
        ("FIREWALL_AVERAGING_MAX_LOSS_ROE_PCT", 0.0, 1.0, 0.015),
        ("FIREWALL_ENTRY_MAX_ATR_EXTENSION", 0.0, 20.0, 1.8),
        ("FIREWALL_ENTRY_MAX_PRICE_EXTENSION_PCT", 0.0, 1.0, 0.035),
        ("FIREWALL_SIDE_IMBALANCE_LOOKBACK_TRADES", 10, 5_000, 60),
        ("FIREWALL_SIDE_IMBALANCE_MIN_SAMPLES", 5, 5_000, 12),
        ("FIREWALL_SIDE_IMBALANCE_MAX_SHARE", 0.50, 0.98, 0.80),
        ("FIREWALL_SIDE_IMBALANCE_CONFIDENCE_BUMP", 0.0, 0.50, 0.15),
        ("FIREWALL_SIDE_IMBALANCE_SIZE_MULTIPLIER", 0.05, 1.0, 0.50),
        ("GLOBAL_MOMENTUM_MIN_AGREEING_COINS", 1, 20, 2),
        ("GLOBAL_MOMENTUM_MIN_CONFIDENCE", 0.0, 1.0, 0.58),
        ("GLOBAL_MOMENTUM_MIN_MOMENTUM", 0.0, 1.0, 0.006),
        ("GLOBAL_MOMENTUM_MIN_VOLUME_RATIO", 0.0, 100.0, 0.75),
        ("BTC_MARKET_LEADER_MIN_CONFIDENCE", 0.0, 1.0, 0.58),
        ("BTC_MARKET_LEADER_MIN_MOMENTUM", 0.0, 1.0, 0.003),
        ("BTC_MARKET_LEADER_MIN_VOLUME_RATIO", 0.0, 100.0, 0.75),
        ("FIREWALL_MARKET_SIDE_GUARD_MIN_CONFIDENCE", 0.0, 1.0, 0.60),
        ("FIREWALL_OPTIONS_FLOW_READ_MIN_CONVICTION", 0.0, 1.0, 0.70),
        ("FIREWALL_OPTIONS_FLOW_OVERRIDE_MAX_REGIME_CONF", 0.0, 1.0, 0.85),
        ("FIREWALL_FORECASTER_SYNTHETIC_WEIGHT", 0.0, 1.0, 0.5),
        ("PAPER_EXECUTION_MAX_TRADES_PER_CYCLE", 0, 100, 3),
        ("TRADE_QUALITY_MIN_EDGE_COST_MULTIPLE", 0.0, 100.0, 1.5),
        ("TRADE_QUALITY_EXPECTED_SLIPPAGE_BPS", 0.0, 1_000.0, PAPER_TRADING_SLIPPAGE_MAX_BPS),
        ("TRADE_QUALITY_SHORT_MIN_CONFIDENCE", 0.0, 1.0, 0.55),
        ("RISK_POLICY_SHORT_CAUTION_CONFIDENCE_THRESHOLD", 0.0, 1.0, 0.60),
        ("RISK_POLICY_SHORT_CAUTION_MAX_REWARD_MULTIPLE", 1.0, 20.0, 3.0),
        ("RISK_POLICY_SHORT_CAUTION_TIME_LIMIT_MULTIPLIER", 0.1, 2.0, 0.75),
        ("RISK_POLICY_SHORT_CAUTION_BREAKEVEN_AT_R", 0.1, 5.0, 0.65),
        ("ARENA_HIGH_CONFIDENCE_THRESHOLD", 0.0, 1.0, 0.80),
        ("ARENA_UNVALIDATED_CONFIDENCE_CAP", 0.0, 1.0, 0.74),
        ("ARENA_TOTAL_POOL_USD", 10.0, 10_000_000.0, 10_000.0),
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

    if LIVE_ENTRY_EXECUTION_MODE not in {"market", "maker_only", "maker_then_market"}:
        _warn_config(
            f"Invalid LIVE_ENTRY_EXECUTION_MODE={LIVE_ENTRY_EXECUTION_MODE!r}; "
            "using 'maker_then_market'."
        )
        globals()["LIVE_ENTRY_EXECUTION_MODE"] = "maker_then_market"

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

    if SOURCE_POLICY_ACTIVE_MAX_SIGNALS_PER_DAY < SOURCE_POLICY_ACTIVE_MIN_SIGNALS_PER_DAY:
        _warn_config(
            "SOURCE_POLICY_ACTIVE_MAX_SIGNALS_PER_DAY is below "
            "SOURCE_POLICY_ACTIVE_MIN_SIGNALS_PER_DAY; raising active max to active min."
        )
        globals()["SOURCE_POLICY_ACTIVE_MAX_SIGNALS_PER_DAY"] = int(
            SOURCE_POLICY_ACTIVE_MIN_SIGNALS_PER_DAY
        )

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
