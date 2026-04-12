"""
Shared runtime health registry for non-critical market data sources.

This is intentionally lighter-weight than the subsystem health registry. It
tracks whether optional signal inputs such as Polymarket or Deribit are
currently available so downstream consumers can distinguish "neutral data" from
"missing data".
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional


class DataSourceRegistry:
    """Thread-safe source state tracker with coarse UP/DOWN/DEGRADED states."""

    VALID_STATES = {"UNKNOWN", "UP", "DOWN", "DEGRADED"}

    def __init__(self):
        self._lock = threading.RLock()
        self._sources: Dict[str, Dict] = {}
        self._version = 0

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name or "").strip().lower() or "unknown"

    def register_source(self, name: str, state: str = "UNKNOWN") -> None:
        normalized = self._normalize_name(name)
        with self._lock:
            self._sources.setdefault(
                normalized,
                {
                    "state": state if state in self.VALID_STATES else "UNKNOWN",
                    "reason": "",
                    "updated_at": 0.0,
                    "last_success_at": 0.0,
                    "metadata": {},
                },
            )

    def mark_up(self, name: str, reason: str = "", metadata: Optional[Dict] = None) -> None:
        self._mark(name, "UP", reason=reason, metadata=metadata, success=True)

    def mark_down(self, name: str, reason: str = "", metadata: Optional[Dict] = None) -> None:
        self._mark(name, "DOWN", reason=reason, metadata=metadata)

    def mark_degraded(
        self,
        name: str,
        reason: str = "",
        metadata: Optional[Dict] = None,
    ) -> None:
        self._mark(name, "DEGRADED", reason=reason, metadata=metadata)

    def _mark(
        self,
        name: str,
        state: str,
        reason: str = "",
        metadata: Optional[Dict] = None,
        success: bool = False,
    ) -> None:
        if state not in self.VALID_STATES:
            raise ValueError(f"Unsupported data source state: {state}")
        normalized = self._normalize_name(name)
        now = time.time()
        meta = dict(metadata or {})
        with self._lock:
            current = self._sources.setdefault(
                normalized,
                {
                    "state": "UNKNOWN",
                    "reason": "",
                    "updated_at": 0.0,
                    "last_success_at": 0.0,
                    "metadata": {},
                },
            )
            changed = (
                current.get("state") != state
                or str(current.get("reason", "")) != str(reason or "")
                or dict(current.get("metadata") or {}) != meta
            )
            current["state"] = state
            current["reason"] = str(reason or "")
            current["updated_at"] = now
            current["metadata"] = meta
            if success:
                current["last_success_at"] = now
            if changed:
                self._version += 1

    def get(self, name: str) -> Dict:
        normalized = self._normalize_name(name)
        with self._lock:
            current = self._sources.get(normalized)
            if not current:
                return {
                    "state": "UNKNOWN",
                    "reason": "",
                    "updated_at": 0.0,
                    "last_success_at": 0.0,
                    "metadata": {},
                }
            return {
                "state": current.get("state", "UNKNOWN"),
                "reason": current.get("reason", ""),
                "updated_at": float(current.get("updated_at", 0.0) or 0.0),
                "last_success_at": float(current.get("last_success_at", 0.0) or 0.0),
                "metadata": dict(current.get("metadata") or {}),
            }

    def snapshot(self) -> Dict[str, Dict]:
        with self._lock:
            return {name: self.get(name) for name in list(self._sources)}

    def version(self) -> int:
        with self._lock:
            return int(self._version)
