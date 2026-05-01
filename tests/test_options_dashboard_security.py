from io import BytesIO

import pytest

from src.ui import options_dashboard


def _make_handler(path="/", headers=None, command="GET"):
    handler = options_dashboard.FlowDashboardHandler.__new__(options_dashboard.FlowDashboardHandler)
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


def test_options_dashboard_refuses_public_bind_without_auth(monkeypatch):
    monkeypatch.delenv("OPTIONS_DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("OPTIONS_DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.delenv("OPTIONS_DASHBOARD_ALLOW_UNAUTH_PUBLIC", raising=False)

    with pytest.raises(RuntimeError, match="public options dashboard without auth"):
        options_dashboard.start_options_dashboard(scanner=None, port=0)


def test_options_dashboard_order_endpoint_disabled_even_when_authenticated(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    monkeypatch.delenv("OPTIONS_DASHBOARD_ENABLE_ORDER_ENDPOINT", raising=False)
    handler = _make_handler(
        path="/api/order",
        headers={"Authorization": "Bearer secret-token", "Content-Length": "2"},
        command="POST",
    )
    handler.rfile = BytesIO(b"{}")

    handler.do_POST()

    assert ("status", 403) in handler._responses
    assert b"options_order_endpoint_disabled" in handler.wfile.getvalue()


def test_options_dashboard_rejects_oversized_order_body(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret-token")
    monkeypatch.setenv("OPTIONS_DASHBOARD_ENABLE_ORDER_ENDPOINT", "true")
    monkeypatch.setenv("OPTIONS_DASHBOARD_MAX_REQUEST_BODY_BYTES", "1024")
    handler = _make_handler(
        path="/api/order",
        headers={"Authorization": "Bearer secret-token", "Content-Length": "1025"},
        command="POST",
    )

    handler.do_POST()

    assert ("status", 413) in handler._responses
    assert b"request_body_too_large" in handler.wfile.getvalue()


def test_options_dashboard_does_not_emit_wildcard_cors(monkeypatch):
    handler = _make_handler(
        path="/api/health",
        headers={"Origin": "https://evil.example"},
    )

    handler.do_GET()

    assert not any(
        item[0] == "header" and item[1] == "Access-Control-Allow-Origin" and item[2] == "*"
        for item in handler._responses
    )
