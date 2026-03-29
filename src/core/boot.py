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
from datetime import datetime

import config


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
    log_file = os.path.join(config.LOG_DIR, f"bot_{datetime.utcnow().strftime('%Y%m%d')}.log")

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
