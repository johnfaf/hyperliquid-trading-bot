"""Verify the network sandbox actually blocks outbound HTTP."""
import pytest
import requests

from src.backtest.replay.network_sandbox import (
    engage, disengage, is_engaged, ReplayNetworkBlocked,
)


@pytest.fixture(autouse=True)
def _ensure_disengaged():
    """Make sure we never leak the sandbox into another test."""
    yield
    disengage()


def test_requests_get_blocked():
    engage()
    assert is_engaged()
    with pytest.raises(ReplayNetworkBlocked):
        requests.get("https://api.hyperliquid.xyz/info", timeout=1)


def test_requests_post_blocked():
    engage()
    with pytest.raises(ReplayNetworkBlocked):
        requests.post("https://api.hyperliquid.xyz/info", json={"type": "allMids"}, timeout=1)


def test_urllib_blocked():
    import urllib.request as ur
    engage()
    with pytest.raises(ReplayNetworkBlocked):
        ur.urlopen("https://example.com", timeout=1)


def test_disengage_restores_network():
    engage()
    disengage()
    assert not is_engaged()
    # We don't actually hit the network in a unit test, but verify the
    # function is restored (i.e. not our blocking stub).
    import requests as rq
    assert "_blocked" not in rq.Session.send.__name__


def test_double_engage_is_idempotent():
    engage()
    engage()  # should not blow up
    assert is_engaged()


def test_loopback_allowed_by_default():
    """Local SQLite-over-HTTP / dashboard tests should still work."""
    engage(allow_loopback=True)
    # We don't actually open a server; just verify the path doesn't raise
    # at the sandbox layer. ConnectionError from no server is expected.
    try:
        requests.get("http://127.0.0.1:1/", timeout=0.5)
    except ReplayNetworkBlocked:
        pytest.fail("Loopback should be allowed when allow_loopback=True")
    except Exception:
        pass  # Connection refused / timeout is the expected outcome
