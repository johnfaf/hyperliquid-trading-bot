"""Cookie-based auth for the v2 dashboard.

Reuses the existing ``DASHBOARD_AUTH_TOKEN`` env contract so the v2
app drops in alongside v1 without operators re-issuing tokens.

We sign session cookies with HMAC-SHA256 over the token + an issued-at
timestamp so a leaked cookie expires with the configured TTL even if
the server's token doesn't rotate. Constant-time comparison prevents
timing oracles.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Optional

from fastapi import Request, Response
from starlette.responses import RedirectResponse

logger = logging.getLogger(__name__)

COOKIE_NAME = "dashboard_v2_auth"
DEFAULT_TTL_S = 86400


def _server_token() -> str:
    return os.environ.get("DASHBOARD_AUTH_TOKEN", "").strip()


def auth_configured() -> bool:
    return bool(_server_token())


def _session_ttl_s() -> int:
    try:
        raw = int(os.environ.get("DASHBOARD_SESSION_TTL_S", str(DEFAULT_TTL_S)))
    except (TypeError, ValueError):
        raw = DEFAULT_TTL_S
    return max(300, min(raw, 30 * 86400))


def _public_read_enabled() -> bool:
    val = os.environ.get("DASHBOARD_PUBLIC_READ", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def public_read_enabled() -> bool:
    return _public_read_enabled()


def _sign(payload: str, token: str) -> str:
    return hmac.new(
        token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def issue_cookie(response: Response, *, secure: bool = True) -> None:
    token = _server_token()
    if not token:
        return
    ttl = _session_ttl_s()
    issued_at = int(time.time())
    payload = f"{issued_at}:{ttl}"
    signature = _sign(payload, token)
    cookie_value = f"{payload}:{signature}"
    response.set_cookie(
        COOKIE_NAME,
        cookie_value,
        max_age=ttl,
        httponly=True,
        secure=secure,
        samesite="lax",
    )


def clear_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME)


def verify_cookie(request: Request) -> bool:
    token = _server_token()
    if not token:
        return False
    raw = request.cookies.get(COOKIE_NAME)
    if not raw:
        return False
    try:
        issued_at_str, ttl_str, signature = raw.rsplit(":", 2)
    except ValueError:
        return False
    payload = f"{issued_at_str}:{ttl_str}"
    expected = _sign(payload, token)
    if not hmac.compare_digest(signature, expected):
        return False
    try:
        issued_at = int(issued_at_str)
        ttl = int(ttl_str)
    except ValueError:
        return False
    if time.time() > issued_at + ttl:
        return False
    return True


def read_access_allowed(request: Any) -> bool:
    """Return whether a read-only dashboard channel may be opened.

    HTTP GET routes use :func:`require_auth`, which intentionally allows
    reads when ``DASHBOARD_PUBLIC_READ=true``. WebSockets are read-only in
    v2 as well, so they must follow the same policy; otherwise the page can
    render while the live event stream silently closes with 4401.
    """
    if _public_read_enabled():
        return True
    return verify_cookie(request)


def login_with_token(submitted_token: str) -> bool:
    """Constant-time compare submitted token against the server token."""
    expected = _server_token()
    if not expected:
        return False
    return hmac.compare_digest(submitted_token, expected)


def require_auth(request: Request) -> Optional[RedirectResponse]:
    """FastAPI dependency: returns a redirect-to-login if not authed.

    Returning a Response from a route is the natural FastAPI way to
    short-circuit. Routes that need authed-only behaviour should call
    this and return its result if non-None.
    """
    if _public_read_enabled() and request.method == "GET":
        return None
    if verify_cookie(request):
        return None
    return RedirectResponse(url="/login", status_code=303)
