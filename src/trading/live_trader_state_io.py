"""
Live Trader State I/O Helpers
==============================
Atomic JSON read/write helpers used by every persisted-state file in
live_trader.py (kill-switch state, dedup cache, source-orders counter,
protective-churn quarantine, attempt history).

Extracted so each persist method on LiveTrader is no longer 25 lines of
mkdir+tmp+replace boilerplate — the LiveTrader code can call these helpers
and the corner cases (parent dir missing when no env override is set,
permission errors, malformed JSON) are handled in one place.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    require_dir_or_env: Optional[str] = None,
    component: str = "state",
) -> bool:
    """Write JSON to ``path`` atomically (tmp + os.replace).

    Args:
        path: Destination file path. No-op if empty/None.
        payload: JSON-serialisable object.
        require_dir_or_env: When set, the parent dir must exist OR the named
            env var must be set (operator opt-in). Used by the kill-switch
            file to avoid silently creating /data/ on machines that don't
            mount that volume.
        component: Tag used in log messages.

    Returns:
        True on success, False on any failure.
    """
    if not path:
        return False
    try:
        directory = os.path.dirname(path)
        if directory:
            if not os.path.exists(directory):
                if require_dir_or_env and require_dir_or_env not in os.environ:
                    logger.warning(
                        "%s state directory %s does not exist; skipping persistence "
                        "until %s is explicitly set",
                        component, directory, require_dir_or_env,
                    )
                    return False
                os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        logger.warning("Failed to persist %s state to %s: %s", component, path, exc)
        return False


def read_json(path: str, *, component: str = "state") -> Optional[Any]:
    """Read JSON from ``path`` if it exists. Returns parsed payload or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load %s state from %s: %s", component, path, exc)
        return None


def kill_switch_payload(active: bool, reason: str) -> Dict[str, Any]:
    """Canonical kill-switch state payload."""
    return {
        "active": bool(active),
        "reason": str(reason or ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
