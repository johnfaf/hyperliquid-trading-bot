from types import SimpleNamespace
import logging

from src.core.cycles import research_cycle
from src.data import hyperliquid_client as hl
from src.discovery import golden_wallet


class _DummyBucket:
    def acquire(self, priority=None, timeout=None):
        return True


class _DummyResponse:
    status_code = 200

    def json(self):
        return {
            "leaderboardRows": [
                {
                    "ethAddress": "0x" + "1" * 40,
                    "accountValue": "123.45",
                }
            ]
        }


class _FakeDiscovery:
    def run_discovery_cycle(self):
        return {
            "traders_discovered": 7,
            "traders_analyzed": 3,
        }


def test_get_leaderboard_works_without_manager_session(monkeypatch):
    manager = SimpleNamespace(bucket=_DummyBucket())
    monkeypatch.setattr(hl, "get_manager", lambda: manager)
    monkeypatch.setattr(hl.requests, "get", lambda url, timeout=30: _DummyResponse())

    data = hl.get_leaderboard()

    assert data["leaderboardRows"][0]["ethAddress"] == "0x" + "1" * 40


def test_run_discovery_logs_golden_scan_summary_keys(monkeypatch, caplog):
    monkeypatch.setattr(golden_wallet, "purge_non_golden_wallets", lambda: 0)
    monkeypatch.setattr(
        golden_wallet,
        "run_golden_scan",
        lambda max_wallets=200: {"golden": 2, "scanned": 5},
    )

    container = SimpleNamespace(
        discovery=_FakeDiscovery(),
        identifier=None,
    )

    with caplog.at_level(logging.INFO):
        research_cycle.run_discovery(container)

    assert "Golden scan: 2 golden wallets out of 5 evaluated" in caplog.text
