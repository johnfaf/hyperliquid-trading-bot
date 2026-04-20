import pytest
from io import BytesIO
from datetime import datetime, timedelta, timezone

from src.core.health_registry import SubsystemHealthRegistry, SubsystemState
from src.ui import dashboard


class _FakeFirewall:
    def get_stats(self):
        return {
            "total_signals": 10,
            "passed": 6,
            "top_rejection_reason": "rejected_confidence",
            "daily_losses": 12.5,
        }


class _FakeLiveTrader:
    def get_stats(self):
        return {
            "kill_switch_active": True,
            "kill_switch_reason": "env:LIVE_EXTERNAL_KILL_SWITCH",
            "canary_mode": True,
            "max_order_usd": 25.0,
            "daily_pnl": -7.5,
            "daily_pnl_limit": 100.0,
            "total_entry_signals_today": 3,
            "attempted_entry_signals": 5,
            "executed_entry_signals": 2,
            "max_orders_per_source_per_day": 2,
            "source_orders_today": {"copy_trade:0xabc": 1},
            "entry_metrics": {"approved_but_not_executable": 1},
            "min_order_rejects_today": 4,
            "min_order_floorups_today": 2,
            "min_order_top_tier_floorups_today": 1,
            "min_order_same_side_merges_today": 1,
            "approved_but_not_executable_today": 1,
            "canary_headroom_ratio": 2.27,
            "crash_safe_canary_order_usd": 55.0,
        }


class _FakeCopyTrader:
    def get_stats(self):
        return {
            "enabled": True,
            "open_copy_trades": 1,
            "guardrail": {
                "status": "blocked",
                "reason": "Recent copy_trade trades are underperforming",
            },
        }


class _BrokenPipeWriter(BytesIO):
    def write(self, data):
        raise BrokenPipeError(32, "Broken pipe")


def test_build_runtime_health_snapshot_includes_subsystems_and_safety(monkeypatch):
    registry = SubsystemHealthRegistry()
    registry.register("decision_firewall", affects_trading=True)
    registry.register("live_trader", affects_trading=True)
    registry.set_status(
        "decision_firewall",
        SubsystemState.HEALTHY,
        dependency_ready=True,
        startup_status="READY",
    )
    registry.set_status(
        "live_trader",
        SubsystemState.DEGRADED,
        reason="ws_lag",
        dependency_ready=True,
        startup_status="READY",
    )
    # Force one stale heartbeat.
    registry._subsystems["live_trader"].last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=900)

    monkeypatch.setattr(dashboard, "_health_registry", registry)
    monkeypatch.setattr(dashboard, "_firewall", _FakeFirewall())
    monkeypatch.setattr(dashboard, "_live_trader", _FakeLiveTrader())
    monkeypatch.setattr(dashboard, "_copy_trader", _FakeCopyTrader())
    monkeypatch.setenv("DASHBOARD_HEALTH_STALE_SECONDS", "600")

    snapshot = dashboard._build_runtime_health_snapshot()

    assert snapshot["overall"] == "at_risk"
    assert snapshot["all_trading_safe"] is True
    assert "live_trader" in snapshot["at_risk_subsystems"]
    assert "live_trader" in snapshot["stale_subsystems"]
    assert snapshot["firewall"]["top_rejection_reason"] == "rejected_confidence"
    assert snapshot["live_trader"]["kill_switch_active"] is True
    assert snapshot["live_trader"]["source_orders_today"] == {"copy_trade:0xabc": 1}
    assert snapshot["live_trader"]["min_order_top_tier_floorups_today"] == 1
    assert snapshot["live_trader"]["approved_but_not_executable_today"] == 1
    assert snapshot["copy_trader"]["guardrail"]["status"] == "blocked"


def test_dashboard_host_defaults_to_localhost_when_not_hosted(monkeypatch):
    for name in (
        "DASHBOARD_HOST",
        "DASHBOARD_BIND_PUBLIC",
        "DASHBOARD_PUBLIC_URL",
        "DASHBOARD_AUTH_TOKEN",
        "RAILWAY_PUBLIC_DOMAIN",
        "RAILWAY_STATIC_URL",
        "RENDER_EXTERNAL_URL",
        "FLY_APP_NAME",
        "K_SERVICE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert dashboard._resolve_dashboard_host() == "127.0.0.1"
    assert dashboard._resolve_dashboard_base_url("127.0.0.1", 8080) == "http://127.0.0.1:8080"


def test_dashboard_host_auto_binds_publicly_on_railway(monkeypatch):
    for name in (
        "DASHBOARD_HOST",
        "DASHBOARD_BIND_PUBLIC",
        "DASHBOARD_PUBLIC_URL",
        "DASHBOARD_AUTH_TOKEN",
        "RAILWAY_STATIC_URL",
        "RENDER_EXTERNAL_URL",
        "FLY_APP_NAME",
        "K_SERVICE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "bot.up.railway.app")

    assert dashboard._resolve_dashboard_host() == "0.0.0.0"
    assert dashboard._resolve_dashboard_base_url("0.0.0.0", 8080) == "https://bot.up.railway.app"


def test_dashboard_public_url_override_wins(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://dash.example.com/")
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "bot.up.railway.app")

    assert dashboard._resolve_dashboard_base_url("0.0.0.0", 8080) == "https://dash.example.com"


def test_hosted_public_dashboard_requires_auth_token(monkeypatch):
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "bot.up.railway.app")
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="DASHBOARD_AUTH_TOKEN"):
        dashboard._validate_dashboard_auth_configuration("0.0.0.0")


def test_local_public_dashboard_can_warn_without_auth_token(monkeypatch):
    for name in ("RAILWAY_PUBLIC_DOMAIN", "RAILWAY_STATIC_URL", "RENDER_EXTERNAL_URL", "FLY_APP_NAME", "K_SERVICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)

    dashboard._validate_dashboard_auth_configuration("0.0.0.0")


def test_dashboard_write_endpoints_require_configured_auth_token(monkeypatch):
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    handler = _make_dashboard_handler(path="/api/trade/close-all", command="POST")

    allowed = handler._check_auth()

    assert allowed is False
    assert ("status", 403) in handler._responses
    assert b"dashboard_write_auth_not_configured" in handler.wfile.getvalue()


@pytest.mark.parametrize(
    "path",
    [
        "/api/order",
        "/api/paper/reset",
        "/api/trade/close",
        "/api/trade/close-all",
        # H9 (audit): compute-intensive, destructive, and external-API
        # POST endpoints must also be gated when auth is not configured.
        "/api/backtest/run",
        "/api/candle-backtest/run",
        "/api/candle-backtest/fetch",
        "/api/candle-backtest/cache/clear",
        "/api/stress/run",
    ],
)
def test_dashboard_all_mutating_post_endpoints_require_auth_token(monkeypatch, path):
    """H9: every mutating / compute-intensive / destructive POST endpoint
    must refuse to execute when DASHBOARD_AUTH_TOKEN is unset, not just
    the finance endpoints.  Without this gate, a Railway-hosted dashboard
    with auth disabled exposes backtest spinners, cache wipes, and
    exchange-API fetch loops to the public internet."""
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    handler = _make_dashboard_handler(path=path, command="POST")

    allowed = handler._check_auth()

    assert allowed is False, (
        f"{path} must be gated when DASHBOARD_AUTH_TOKEN is unset"
    )
    assert ("status", 403) in handler._responses
    assert b"dashboard_write_auth_not_configured" in handler.wfile.getvalue()


def test_dashboard_all_mutating_post_endpoints_accept_valid_bearer(monkeypatch):
    """H9: when DASHBOARD_AUTH_TOKEN is set, all gated POST endpoints must
    still accept a valid Bearer token — the expanded gate should not break
    authenticated operators."""
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    for path in (
        "/api/order",
        "/api/paper/reset",
        "/api/backtest/run",
        "/api/candle-backtest/run",
        "/api/candle-backtest/fetch",
        "/api/candle-backtest/cache/clear",
        "/api/stress/run",
    ):
        handler = _make_dashboard_handler(
            path=path,
            headers={"Authorization": "Bearer secret-token"},
            command="POST",
        )
        assert handler._check_auth() is True, (
            f"{path} should authorize with valid Bearer token"
        )


def _make_dashboard_handler(path="/", headers=None, command="GET"):
    handler = dashboard.DashboardHandler.__new__(dashboard.DashboardHandler)
    handler.path = path
    handler.headers = headers or {}
    handler.command = command
    handler.wfile = BytesIO()
    handler.rfile = BytesIO()
    handler._responses = []
    handler.send_response = lambda code: handler._responses.append(("status", code))
    handler.send_header = lambda key, value: handler._responses.append(("header", key, value))
    handler.end_headers = lambda: handler._responses.append(("end",))
    return handler


def test_dashboard_rejects_query_param_token(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    handler = _make_dashboard_handler(path="/?token=secret-token")

    allowed = handler._check_auth()

    assert allowed is False
    assert ("status", 303) in handler._responses
    assert ("header", "Location", "/login?next=/?token=secret-token") in handler._responses


def test_dashboard_bearer_auth_sets_cookie(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    handler = _make_dashboard_handler(
        headers={"Authorization": "Bearer secret-token"},
    )

    allowed = handler._check_auth()

    assert allowed is True
    assert handler._pending_auth_cookie == "secret-token"


def test_dashboard_redirects_page_requests_to_login(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    handler = _make_dashboard_handler(path="/options")

    allowed = handler._check_auth()

    assert allowed is False
    assert ("status", 303) in handler._responses
    assert ("header", "Location", "/login?next=/options") in handler._responses


def test_dashboard_login_post_sets_cookie_and_redirects(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    handler = _make_dashboard_handler(
        path="/api/auth/login",
        headers={"Content-Length": "29"},
    )
    handler.rfile = BytesIO(b"token=secret-token&next=%2F")

    handler._handle_login()

    assert handler._pending_auth_cookie == "secret-token"
    assert ("status", 303) in handler._responses
    assert ("header", "Location", "/") in handler._responses


def test_dashboard_login_post_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    handler = _make_dashboard_handler(
        path="/api/auth/login",
        headers={"Content-Length": "22"},
    )
    handler.rfile = BytesIO(b"token=wrong&next=%2F")

    handler._handle_login()

    assert ("status", 303) in handler._responses
    assert ("header", "Location", "/login?error=invalid&next=/") in handler._responses


def test_dashboard_json_response_swallows_client_disconnect_on_body_write():
    handler = _make_dashboard_handler(path="/api/data")
    handler.wfile = _BrokenPipeWriter()

    handler._json_response({"ok": True})

    assert handler.close_connection is True


def test_dashboard_json_response_swallows_client_disconnect_during_headers():
    handler = _make_dashboard_handler(path="/api/data")

    def _broken_end_headers():
        raise BrokenPipeError(32, "Broken pipe")

    handler.end_headers = _broken_end_headers

    handler._json_response({"ok": True})

    assert handler.close_connection is True
