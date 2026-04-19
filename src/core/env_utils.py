"""
Env-var parsing utilities.

Trading-critical env vars (loss caps, cooldowns, slippage, thresholds) must
never raise on malformed input or silently accept an unreasonable value.
``safe_env_float`` parses with try/except, falls back to a default on
unparseable input, and clamps to an optional ``[lo, hi]`` range with a
warning so operators learn immediately instead of discovering it when a
trade behaves unexpectedly.

This module intentionally has no dependencies on trading/business logic
so it can be imported from anywhere in the codebase.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def safe_env_float(
    name: str,
    default: float,
    *,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> float:
    """Parse a numeric env var with try/except + optional range clamp."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Env var %s=%r is not numeric; falling back to default %s",
            name, raw, default,
        )
        return float(default)
    if lo is not None and value < lo:
        logger.warning(
            "Env var %s=%s is below lower bound %s; clamping up.",
            name, value, lo,
        )
        value = float(lo)
    if hi is not None and value > hi:
        logger.warning(
            "Env var %s=%s exceeds upper bound %s; clamping down.",
            name, value, hi,
        )
        value = float(hi)
    return float(value)


def safe_env_int(
    name: str,
    default: int,
    *,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
) -> int:
    """Parse an integer env var with the same safety guarantees."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(float(raw))  # tolerate "10.0" style inputs
    except (TypeError, ValueError):
        logger.warning(
            "Env var %s=%r is not integer; falling back to default %s",
            name, raw, default,
        )
        return int(default)
    if lo is not None and value < lo:
        logger.warning(
            "Env var %s=%s is below lower bound %s; clamping up.",
            name, value, lo,
        )
        value = int(lo)
    if hi is not None and value > hi:
        logger.warning(
            "Env var %s=%s exceeds upper bound %s; clamping down.",
            name, value, hi,
        )
        value = int(hi)
    return int(value)


def safe_env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var; truthy values: 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in ("1", "true", "yes", "on", "t", "y"):
        return True
    if val in ("0", "false", "no", "off", "f", "n", ""):
        return False
    logger.warning(
        "Env var %s=%r is not boolean; falling back to default %s",
        name, raw, default,
    )
    return bool(default)
