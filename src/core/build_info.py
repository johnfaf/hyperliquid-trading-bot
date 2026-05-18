"""Deployed-build identity.

A recurring blind spot during incidents was *"is the running bot even on
the commit I merged?"* — we could not tell whether Railway had
redeployed. Railway injects the deploy's git metadata as ``RAILWAY_GIT_*``
/ ``RAILWAY_DEPLOYMENT_ID`` env vars; this surfaces them (with a local
``git`` fallback for dev) so the running commit is visible on
``/api/health`` and in the startup log.

Everything here is best-effort and dependency-free — it must never raise.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Captured once at process import → uptime / "when did this build start".
PROCESS_STARTED_AT = datetime.now(timezone.utc).isoformat()

_UNKNOWN = "unknown"
_cached: Optional[Dict[str, Any]] = None


def _env(name: str) -> str:
    return str(os.environ.get(name, "") or "").strip()


def _local_git_short() -> str:
    """Dev fallback only — the container image has no .git, so this is a
    no-op there (returns ``unknown``); Railway env vars are authoritative."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or _UNKNOWN
    except Exception:
        pass
    return _UNKNOWN


def get_build_info(refresh: bool = False) -> Dict[str, Any]:
    """Return the deployed build's identity.

    Keys: commit (full), short, branch, message, author, deployment_id,
    service, environment, source ("railway" | "git" | "unknown"),
    process_started_at.
    """
    global _cached
    if _cached is not None and not refresh:
        return dict(_cached)

    commit = _env("RAILWAY_GIT_COMMIT_SHA")
    source = "railway" if commit else _UNKNOWN
    if not commit:
        local = _local_git_short()
        if local != _UNKNOWN:
            commit = local
            source = "git"

    short = commit[:10] if commit and commit != _UNKNOWN else _UNKNOWN
    info: Dict[str, Any] = {
        "commit": commit or _UNKNOWN,
        "short": short,
        "branch": _env("RAILWAY_GIT_BRANCH") or _UNKNOWN,
        "message": _env("RAILWAY_GIT_COMMIT_MESSAGE") or "",
        "author": _env("RAILWAY_GIT_AUTHOR") or "",
        "deployment_id": _env("RAILWAY_DEPLOYMENT_ID") or _UNKNOWN,
        "service": _env("RAILWAY_SERVICE_NAME") or _UNKNOWN,
        "environment": _env("RAILWAY_ENVIRONMENT_NAME")
        or _env("RAILWAY_ENVIRONMENT")
        or _UNKNOWN,
        "source": source,
        "process_started_at": PROCESS_STARTED_AT,
    }
    _cached = dict(info)
    return info


def build_banner() -> str:
    """One-line human banner for the startup log."""
    b = get_build_info()
    _lines = (b.get("message") or "").splitlines()
    msg = _lines[0][:80] if _lines else ""
    return (
        f"BUILD commit={b['short']} branch={b['branch']} "
        f"deploy={b['deployment_id'][:12]} src={b['source']} "
        f"started={b['process_started_at']}"
        + (f' msg="{msg}"' if msg else "")
    )
