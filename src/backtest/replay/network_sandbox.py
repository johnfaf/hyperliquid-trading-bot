"""Outbound-HTTP guard for replay mode.

The api_manager_shim intercepts everything routed through `APIManager.post()`.
But several modules in the codebase reach for `requests.get/post` directly:
the macro scraper, polymarket clients, options flow, etc. Stubs handle those
at the subsystem level -- but the network sandbox is a belt-and-suspenders
catch for any path we missed.

Install with `engage()` and you get a `ReplayNetworkBlocked` raise on:
  - requests.Session.send  (used by requests.get/post/put/...)
  - urllib.request.urlopen
  - http.client.HTTPConnection.request / HTTPSConnection.request

`disengage()` restores the originals. Always pair them with try/finally.

Loopback (127.0.0.1) traffic is allowed so tests / local SQLite-over-HTTP
tooling don't break. To enforce strictest possible isolation pass
`allow_loopback=False`.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)


class ReplayNetworkBlocked(RuntimeError):
    """Raised when something tried to make a real outbound HTTP request during replay."""


_ENGAGED = False
_ORIGINALS: dict = {}
_ALLOW_LOOPBACK = True


def _is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    return host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127.")


def _blocked(target: str, reason: str) -> ReplayNetworkBlocked:
    return ReplayNetworkBlocked(
        f"Outbound HTTP blocked in replay: {target} ({reason}). "
        "Either route this call through api_manager (which the replay shim "
        "intercepts) or shim the caller at the subsystem level."
    )


def engage(*, allow_loopback: bool = True) -> None:
    """Activate the sandbox. Idempotent."""
    global _ENGAGED, _ALLOW_LOOPBACK
    if _ENGAGED:
        return
    _ALLOW_LOOPBACK = allow_loopback

    # 1. requests.Session.send
    try:
        import requests
        _ORIGINALS["requests.Session.send"] = requests.Session.send

        def _blocked_send(self, request, **kwargs):
            url = getattr(request, "url", "?")
            host = urllib.parse.urlparse(url).hostname
            if _ALLOW_LOOPBACK and _is_loopback(host):
                return _ORIGINALS["requests.Session.send"](self, request, **kwargs)
            raise _blocked(url, "requests.Session.send")

        requests.Session.send = _blocked_send
    except ImportError:
        pass

    # 2. urllib.request.urlopen
    try:
        import urllib.request as ur
        _ORIGINALS["urllib.request.urlopen"] = ur.urlopen

        def _blocked_urlopen(url, *args, **kwargs):
            target = url if isinstance(url, str) else getattr(url, "full_url", "?")
            host = urllib.parse.urlparse(target).hostname if isinstance(target, str) else None
            if _ALLOW_LOOPBACK and _is_loopback(host):
                return _ORIGINALS["urllib.request.urlopen"](url, *args, **kwargs)
            raise _blocked(target, "urllib.request.urlopen")

        ur.urlopen = _blocked_urlopen
    except ImportError:
        pass

    # 3. http.client low-level (catches things bypassing requests + urllib)
    try:
        import http.client as hc
        _ORIGINALS["http.client.HTTPConnection.request"] = hc.HTTPConnection.request
        _ORIGINALS["http.client.HTTPSConnection.request"] = hc.HTTPSConnection.request

        def _blocked_request(self, method, url, *args, **kwargs):
            host = getattr(self, "host", "?")
            if _ALLOW_LOOPBACK and _is_loopback(host):
                return _ORIGINALS["http.client.HTTPConnection.request"](self, method, url, *args, **kwargs)
            raise _blocked(f"{host}{url}", f"http.client {method}")

        hc.HTTPConnection.request = _blocked_request
        hc.HTTPSConnection.request = _blocked_request
    except ImportError:
        pass

    _ENGAGED = True
    logger.info("Replay network sandbox engaged (allow_loopback=%s)", allow_loopback)


def disengage() -> None:
    """Restore the originals. Idempotent."""
    global _ENGAGED
    if not _ENGAGED:
        return
    try:
        import requests
        requests.Session.send = _ORIGINALS["requests.Session.send"]
    except (ImportError, KeyError):
        pass
    try:
        import urllib.request as ur
        ur.urlopen = _ORIGINALS["urllib.request.urlopen"]
    except (ImportError, KeyError):
        pass
    try:
        import http.client as hc
        hc.HTTPConnection.request = _ORIGINALS["http.client.HTTPConnection.request"]
        hc.HTTPSConnection.request = _ORIGINALS["http.client.HTTPSConnection.request"]
    except (ImportError, KeyError):
        pass
    _ORIGINALS.clear()
    _ENGAGED = False
    logger.info("Replay network sandbox disengaged")


def is_engaged() -> bool:
    return _ENGAGED
