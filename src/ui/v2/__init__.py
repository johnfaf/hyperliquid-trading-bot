"""Dashboard v2 — FastAPI + HTMX.

Coexists with the legacy stdlib dashboard ([src/ui/dashboard.py]). Opt
in by setting ``DASHBOARD_V2_ENABLED=true`` and pointing the launcher
at the bot's subsystem container via :func:`set_components`.
"""
from src.ui.v2.app import create_app, start_server  # noqa: F401
from src.ui.v2.state import set_components  # noqa: F401
