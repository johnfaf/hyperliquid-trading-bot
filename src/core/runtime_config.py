"""
Runtime config override loader.

This provides a small hot-reload layer for settings that are safe to change
without restarting the bot. It reads a JSON override file and updates both the
module-level config globals and the instantiated runtime components that copied
those values at startup.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)


RUNTIME_OVERRIDE_SPECS: Dict[str, Dict[str, Any]] = {
    "FIREWALL_MIN_CONFIDENCE": {"type": "float", "min": 0.0, "max": 1.0},
    "FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY": {"type": "int", "min": 0, "max": 100_000},
    "SHORT_HARDENING_ENABLED": {"type": "bool"},
    "SHORT_HARDENING_LOOKBACK_TRADES": {"type": "int", "min": 10, "max": 5_000},
    "SHORT_HARDENING_MIN_CLOSED_TRADES": {"type": "int", "min": 1, "max": 1_000},
    "SHORT_HARDENING_DEGRADE_WIN_RATE": {"type": "float", "min": 0.0, "max": 1.0},
    "SHORT_HARDENING_BLOCK_WIN_RATE": {"type": "float", "min": 0.0, "max": 1.0},
    "SHORT_HARDENING_BLOCK_NET_PNL": {
        "type": "float",
        "min": -1_000_000.0,
        "max": 1_000_000.0,
    },
    "SHORT_HARDENING_CONFIDENCE_MULTIPLIER": {"type": "float", "min": 0.0, "max": 1.0},
    "SHORT_HARDENING_SIZE_MULTIPLIER": {"type": "float", "min": 0.0, "max": 1.0},
    "SHORT_HARDENING_SOURCE_GUARD_ENABLED": {"type": "bool"},
    "SHORT_HARDENING_SOURCE_MIN_CLOSED_TRADES": {"type": "int", "min": 1, "max": 1_000},
    "SHORT_HARDENING_SOURCE_BLOCK_NET_PNL": {
        "type": "float",
        "min": -1_000_000.0,
        "max": 1_000_000.0,
    },
    "SHORT_HARDENING_COIN_GUARD_ENABLED": {"type": "bool"},
    "SHORT_HARDENING_COIN_MIN_CLOSED_TRADES": {"type": "int", "min": 1, "max": 1_000},
    "SHORT_HARDENING_COIN_BLOCK_NET_PNL": {
        "type": "float",
        "min": -1_000_000.0,
        "max": 1_000_000.0,
    },
    "EVENT_RISK_ENABLED": {"type": "bool"},
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


class RuntimeConfigManager:
    """Poll a JSON override file and apply supported runtime-safe settings."""

    def __init__(
        self,
        path: Optional[str] = None,
        poll_seconds: Optional[float] = None,
    ):
        self.path = str(path or getattr(config, "RUNTIME_CONFIG_OVERRIDE_FILE", "/data/config.json"))
        self.poll_seconds = max(
            1.0,
            float(poll_seconds or getattr(config, "RUNTIME_CONFIG_POLL_SECONDS", 10)),
        )
        self._base_values = {
            key: getattr(config, key, None)
            for key in RUNTIME_OVERRIDE_SPECS
        }
        self._last_poll_at = 0.0
        self._last_applied_signature: Optional[str] = None
        self._last_applied_overrides: Dict[str, Any] = {}

    def poll(self, container=None, force: bool = False) -> bool:
        now = time.time()
        if not force and (now - self._last_poll_at) < self.poll_seconds:
            return False
        self._last_poll_at = now

        overrides = self._load_overrides()
        signature = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
        if signature == self._last_applied_signature:
            return False

        applied = dict(self._base_values)
        applied.update(overrides)
        self._apply(applied, container)
        self._last_applied_signature = signature
        self._last_applied_overrides = overrides

        if overrides:
            logger.info("Applied runtime config overrides from %s: %s", self.path, sorted(overrides))
        else:
            logger.info("Runtime config overrides cleared; reverted to base values")
        return True

    def current_overrides(self) -> Dict[str, Any]:
        return dict(self._last_applied_overrides)

    def _load_overrides(self) -> Dict[str, Any]:
        if not self.path or not os.path.exists(self.path):
            return {}

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load runtime config overrides from %s: %s", self.path, exc)
            return {}

        if not isinstance(payload, dict):
            logger.warning("Ignoring runtime config override file %s: top-level JSON must be an object", self.path)
            return {}

        if isinstance(payload.get("overrides"), dict):
            payload = payload["overrides"]

        overrides: Dict[str, Any] = {}
        for key, spec in RUNTIME_OVERRIDE_SPECS.items():
            if key not in payload:
                continue
            try:
                overrides[key] = self._coerce_value(key, payload[key], spec)
            except ValueError as exc:
                logger.warning("Ignoring runtime override %s from %s: %s", key, self.path, exc)
        return overrides

    def _coerce_value(self, name: str, raw_value: Any, spec: Dict[str, Any]) -> Any:
        value_type = spec.get("type")
        if value_type == "bool":
            return _parse_bool(raw_value)
        if value_type == "int":
            try:
                numeric = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"expected integer, got {raw_value!r}") from exc
        elif value_type == "float":
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"expected float, got {raw_value!r}") from exc
        else:
            raise ValueError(f"unsupported override type for {name}: {value_type}")

        minimum = spec.get("min")
        maximum = spec.get("max")
        if minimum is not None and numeric < minimum:
            raise ValueError(f"value {numeric} below minimum {minimum}")
        if maximum is not None and numeric > maximum:
            raise ValueError(f"value {numeric} above maximum {maximum}")
        return numeric

    def _apply(self, values: Dict[str, Any], container=None) -> None:
        for key, value in values.items():
            setattr(config, key, value)

        firewall = getattr(container, "firewall", None) if container else None
        if firewall and hasattr(firewall, "apply_runtime_overrides"):
            firewall.apply_runtime_overrides(values)
