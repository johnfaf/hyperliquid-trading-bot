"""Deployed-build identity surface (answers "is the bot on the merged
commit?" without shelling into the container)."""
from __future__ import annotations

import src.core.build_info as bi


def test_railway_env_is_authoritative(monkeypatch):
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "303ce5c0d31adb03c9a88c399172377dca6cbdc6")
    monkeypatch.setenv("RAILWAY_GIT_BRANCH", "main")
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_MESSAGE", "Merge X regime fix\nsecond line")
    monkeypatch.setenv("RAILWAY_DEPLOYMENT_ID", "ff3cbab8-5daf-40c3-b33b-7effa9065c14")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_NAME", "production")

    info = bi.get_build_info(refresh=True)
    assert info["commit"].startswith("303ce5c")
    assert info["short"] == "303ce5c0d3"
    assert info["branch"] == "main"
    assert info["environment"] == "production"
    assert info["source"] == "railway"
    assert info["process_started_at"]

    banner = bi.build_banner()
    assert "commit=303ce5c0d3" in banner
    assert "branch=main" in banner
    # multi-line commit message must collapse to first line in the banner
    assert "second line" not in banner


def test_no_env_falls_back_to_unknown(monkeypatch):
    for k in (
        "RAILWAY_GIT_COMMIT_SHA", "RAILWAY_GIT_BRANCH", "RAILWAY_GIT_COMMIT_MESSAGE",
        "RAILWAY_DEPLOYMENT_ID", "RAILWAY_SERVICE_NAME", "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_ENVIRONMENT",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(bi, "_local_git_short", lambda: bi._UNKNOWN)

    info = bi.get_build_info(refresh=True)
    assert info["commit"] == "unknown"
    assert info["short"] == "unknown"
    assert info["source"] == "unknown"
    assert "BUILD" in bi.build_banner()  # never raises
