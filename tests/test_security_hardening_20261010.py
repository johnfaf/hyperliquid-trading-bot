"""Regression checks for previously identified security weaknesses."""
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from src.ui import dashboard
from src.ui.v2 import auth as v2_auth
from src.notifications.telegram_bot import _safe_telegram_error


def test_login_page_escapes_untrusted_return_url():
    attack = '/"><img src=x onerror=alert(1)>'
    rendered = dashboard._login_html(next_path=attack)
    assert '<img src=x' not in rendered
    assert '&quot;&gt;&lt;img' in rendered


def test_dashboard_rejects_external_redirect_targets():
    assert dashboard._safe_dashboard_next("//evil.example/path") == "/"
    assert dashboard._safe_dashboard_next("/\\evil.example") == "/"
    assert dashboard._safe_dashboard_next("https://evil.example") == "/"
    assert dashboard._safe_dashboard_next("/positions?view=all") == "/positions?view=all"


def test_v2_rejects_login_without_server_token(monkeypatch):
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    request = SimpleNamespace(cookies={})
    assert v2_auth.verify_cookie(request) is False
    assert v2_auth.login_with_token("") is False


def test_v2_rate_limits_login_guesses(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-secret")
    v2_auth._LOGIN_ATTEMPTS.clear()
    for _ in range(10):
        assert v2_auth.login_attempt_allowed("security-test-ip")
    assert not v2_auth.login_attempt_allowed("security-test-ip")


def test_telegram_error_redacts_bot_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "sensitive-testing-token")
    msg = _safe_telegram_error(
        "connection failed to https://api.telegram.org/botsensitive-testing-token/sendMessage"
    )
    assert "sensitive-testing-token" not in msg
    assert "<redacted>" in msg
