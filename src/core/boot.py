"""
Boot & Logging Setup
====================
Everything that must happen before ANY subsystem is imported or instantiated:

1. Configure structured JSON / text logging
2. Run boot-time dependency validation (fail-fast)
3. Initialize the database
4. Restore from backup if needed

Extracted from the old monolithic main.py so that:
* CLI entrypoints can call ``setup_logging()`` independently
* The dependency check runs before heavy imports
* Logging configuration lives in exactly one place
"""
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import config
from src.core.time_utils import utc_date_str


# ─── Secret scrubbing ─────────────────────────────────────────

_SECRET_PATTERNS = [
    re.compile(
        r'(api[_-]?key|api[_-]?secret|private[_-]?key|secret[_-]?key'
        r'|password|token|authorization)\s*[=:]\s*\S+',
        re.IGNORECASE,
    ),
    re.compile(r'0x[a-fA-F0-9]{64}'),                                  # ETH private keys
    re.compile(r'(Bearer|Basic)\s+[A-Za-z0-9+/=_-]{20,}', re.IGNORECASE),  # Auth headers
]


def _scrub_secrets(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production.
    Railway, Datadog, ELK, and most log aggregators parse JSON natively.
    Includes secret-scrubbing.
    """
    def format(self, record):
        import json as _json
        msg = _scrub_secrets(record.getMessage())
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = _scrub_secrets(self.formatException(record.exc_info))
        for key in ("wallet", "coin", "action", "latency_ms", "status_code"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return _json.dumps(entry, default=str)


def setup_logging() -> logging.Logger:
    """Configure root logger with file + console handlers. Returns module logger."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, f"bot_{utc_date_str('%Y%m%d')}.log")

    json_fmt = JSONFormatter()
    text_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(text_fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, config.LOG_LEVEL))
    use_json = os.environ.get("LOG_FORMAT", "json").lower() != "text"
    ch.setFormatter(json_fmt if use_json else text_fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    return logging.getLogger("boot")


def validate_dependencies(logger: logging.Logger) -> None:
    """
    Run boot-time dependency validation.
    Fails fast with RuntimeError if any enabled feature's packages are missing.
    """
    from src.core.dependency_validator import validate_or_fail, get_boot_report
    logger.info(get_boot_report(config))
    validate_or_fail(config_module=config)
    validate_runtime_profile_controls(logger)
    validate_operational_controls(logger)
    validate_wallet_security_controls(logger)


def validate_runtime_profile_controls(logger: logging.Logger) -> None:
    """Validate runtime profile selection and explicit live-mode controls."""
    summary = getattr(config, "get_runtime_profile_summary", lambda: {})() or {}
    requested_profile = str(summary.get("requested_profile", "") or "").strip().lower()
    profile_valid = bool(summary.get("profile_valid", True))
    profile = str(getattr(config, "RUNTIME_PROFILE", "paper") or "paper").strip().lower()
    allowed = set(summary.get("allowed_profiles", []) or ("paper", "shadow", "live"))
    effective_execution_mode = str(summary.get("effective_execution_mode", profile) or profile).strip().lower()
    override_controls = list(summary.get("override_controls", []) or [])
    if not profile_valid:
        raise RuntimeError(
            f"Unsupported BOT_RUNTIME_PROFILE '{requested_profile}'. Expected one of: {', '.join(sorted(allowed))}."
        )
    if profile not in allowed:
        raise RuntimeError(
            f"Unsupported BOT_RUNTIME_PROFILE '{profile}'. Expected one of: {', '.join(sorted(allowed))}."
        )

    if override_controls:
        logger.warning(
            "Runtime profile resolved: %s (effective execution mode: %s, env overrides: %s)",
            profile,
            effective_execution_mode,
            ", ".join(override_controls),
        )
    else:
        logger.info(
            "Runtime profile resolved: %s (effective execution mode: %s)",
            profile,
            effective_execution_mode,
        )

    if profile != "live":
        return

    required_envs = [
        "LIVE_TRADING_ENABLED",
        "LIVE_MIN_ORDER_USD",
        "LIVE_MAX_ORDER_USD",
        "LIVE_MAX_DAILY_LOSS_USD",
        "LIVE_PREFLIGHT_REQUIRED",
        "LIVE_ACTIVATION_GUARD_ENABLED",
        "LIVE_ACTIVATION_APPROVED_AT",
        "LIVE_ACTIVATION_APPROVED_BY",
        "LIVE_ACTIVATION_MAX_AGE_HOURS",
        "HL_WALLET_MODE",
        "SECRET_MANAGER_PROVIDER",
    ]
    missing = [env_name for env_name in required_envs if not str(os.environ.get(env_name, "")).strip()]
    if missing:
        raise RuntimeError(
            "BOT_RUNTIME_PROFILE=live requires explicit env vars for live controls: "
            + ", ".join(missing)
        )

    if not getattr(config, "LIVE_TRADING_ENABLED", False):
        raise RuntimeError(
            "BOT_RUNTIME_PROFILE=live resolved LIVE_TRADING_ENABLED=false. "
            "Set LIVE_TRADING_ENABLED=true explicitly or switch profiles."
        )

    if not getattr(config, "LIVE_PREFLIGHT_REQUIRED", False):
        raise RuntimeError(
            "BOT_RUNTIME_PROFILE=live requires LIVE_PREFLIGHT_REQUIRED=true."
        )

    if not getattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", False):
        raise RuntimeError(
            "BOT_RUNTIME_PROFILE=live requires LIVE_ACTIVATION_GUARD_ENABLED=true."
        )

    logger.info("Runtime profile controls validated for live deployment.")


def validate_operational_controls(logger: logging.Logger) -> None:
    """
    Enforce explicit rotation thresholds before enabling rotation engine.
    This prevents accidental live-ish operation on hidden defaults.
    """
    if not getattr(config, "ROTATION_ENGINE_ENABLED", False):
        return
    if not getattr(config, "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", True):
        logger.warning("Rotation engine enabled without explicit-threshold enforcement.")
        return

    required_envs = [
        "PORTFOLIO_REPLACEMENT_THRESHOLD",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_DAY",
        "PORTFOLIO_FORCED_EXIT_COOLDOWN_MINUTES",
        "PORTFOLIO_ROUND_TRIP_BLOCK_MINUTES",
        "PORTFOLIO_MAX_COIN_EXPOSURE_PCT",
        "PORTFOLIO_MAX_SIDE_EXPOSURE_PCT",
        "PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT",
        "PORTFOLIO_TRANSACTION_COST_WEIGHT",
        "PORTFOLIO_CHURN_PENALTY",
        "PORTFOLIO_EXPECTED_SLIPPAGE_BPS",
    ]
    missing = [env for env in required_envs if not str(os.environ.get(env, "")).strip()]
    shadow_mode = bool(getattr(config, "ROTATION_DRY_RUN_TELEMETRY", True))
    if missing:
        if shadow_mode:
            logger.warning(
                "Rotation shadow mode enabled with missing explicit threshold env vars: %s. "
                "Using config defaults for shadow simulation; set these env vars before "
                "live replacement mode.",
                ", ".join(missing),
            )
        else:
            raise RuntimeError(
                "Rotation engine is enabled but explicit threshold env vars are missing: "
                + ", ".join(missing)
            )
    else:
        logger.info("Rotation operational controls validated (explicit thresholds present).")

    if shadow_mode:
        shadow_days = max(int(getattr(config, "ROTATION_SHADOW_MODE_DAYS", 7)), 1)
        end_at = datetime.now(timezone.utc) + timedelta(days=shadow_days)
        logger.info(
            "Rotation shadow mode active: replacements will be simulated only for %sd "
            "(window end target %s).",
            shadow_days,
            end_at.isoformat(),
        )


def validate_wallet_security_controls(logger: logging.Logger) -> None:
    """Enforce wallet-mode and secret-manager safety constraints."""
    wallet_mode = str(getattr(config, "HL_WALLET_MODE", "agent_only")).lower().strip()
    provider = str(getattr(config, "SECRET_MANAGER_PROVIDER", "none")).lower().strip()
    if wallet_mode not in {"agent_only"}:
        raise RuntimeError(
            f"Unsupported HL_WALLET_MODE '{wallet_mode}'. Only 'agent_only' is permitted."
        )
    if os.environ.get("HL_PRIVATE_KEY"):
        raise RuntimeError(
            "Legacy HL_PRIVATE_KEY is disallowed in agent-only mode. "
            "Use HL_AGENT_PRIVATE_KEY or SECRET_MANAGER_PROVIDER."
        )
    if provider not in {"none", "aws_kms", "hashicorp", "hashicorp_vault", "vault"}:
        raise RuntimeError(
            f"Unsupported SECRET_MANAGER_PROVIDER '{provider}'. "
            "Expected one of: none, aws_kms, hashicorp."
        )

    if provider == "aws_kms":
        missing = []
        if not os.environ.get("AWS_KMS_CIPHERTEXT_B64"):
            missing.append("AWS_KMS_CIPHERTEXT_B64")
        if not os.environ.get("AWS_KMS_REGION"):
            missing.append("AWS_KMS_REGION")
        if missing:
            raise RuntimeError(
                "SECRET_MANAGER_PROVIDER=aws_kms but required env vars are missing: "
                + ", ".join(missing)
            )
        try:
            import boto3  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SECRET_MANAGER_PROVIDER=aws_kms requires boto3. "
                "Install it or switch provider."
            ) from exc

    if provider in {"hashicorp", "hashicorp_vault", "vault"}:
        missing = []
        for env_name in ("VAULT_ADDR", "VAULT_TOKEN", "VAULT_SECRET_PATH"):
            if not os.environ.get(env_name):
                missing.append(env_name)
        if missing:
            raise RuntimeError(
                "SECRET_MANAGER_PROVIDER=hashicorp but required env vars are missing: "
                + ", ".join(missing)
            )

    logger.info(
        "Wallet security controls validated (mode=%s, secret_provider=%s).",
        wallet_mode,
        provider,
    )


def init_database(logger: logging.Logger) -> None:
    """Initialize DB and restore from backup if needed."""
    from src.data.database import init_db, restore_from_json
    init_db()
    if restore_from_json():
        logger.info("Restored DB from backup (post-deploy recovery)")


def log_persistence_info(logger: logging.Logger) -> None:
    """Log persistence paths for Railway debugging."""
    logger.info(
        "[PERSISTENCE] persistent_volume=%s DB_PATH=%s HL_BOT_DB_env=%s uid=%s /data_exists=%s",
        config._HAS_PERSISTENT_VOLUME, config.DB_PATH,
        os.environ.get("HL_BOT_DB", "NOT SET"),
        os.getuid(), os.path.isdir("/data"),
    )
