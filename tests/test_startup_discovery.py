import os

os.environ["LIVE_CANARY_MODE"] = "false"

import main


def test_strategy_pool_with_no_valid_rows_requires_startup_discovery(monkeypatch):
    monkeypatch.setattr(
        main.db,
        "get_strategy_runtime_status",
        lambda: {
            "total": 14424,
            "active_valid": 0,
            "inactive_valid": 0,
            "invalid_reasons": {"missing_source_wallet": 2557},
        },
    )

    needed, status = main._strategy_pool_requires_startup_discovery()

    assert needed is True
    assert status["invalid_reasons"]["missing_source_wallet"] == 2557


def test_strategy_pool_with_recoverable_valid_rows_does_not_force_discovery(monkeypatch):
    monkeypatch.setattr(
        main.db,
        "get_strategy_runtime_status",
        lambda: {"total": 10, "active_valid": 0, "inactive_valid": 3},
    )

    needed, status = main._strategy_pool_requires_startup_discovery()

    assert needed is False
    assert status["inactive_valid"] == 3


def test_strategy_pool_status_error_does_not_force_discovery(monkeypatch):
    def fail():
        raise RuntimeError("db busy")

    monkeypatch.setattr(main.db, "get_strategy_runtime_status", fail)

    needed, status = main._strategy_pool_requires_startup_discovery()

    assert needed is False
    assert status == {"error": "db busy"}
