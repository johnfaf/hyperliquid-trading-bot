import json

import config

from src.core.runtime_config import RuntimeConfigManager
from src.signals.decision_firewall import DecisionFirewall


def test_runtime_config_hot_reload_applies_and_reverts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIREWALL_MIN_CONFIDENCE", 0.45, raising=False)
    monkeypatch.setattr(
        config,
        "FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY",
        0,
        raising=False,
    )

    firewall = DecisionFirewall({"min_confidence": 0.45, "forecaster": object()})
    container = type("Container", (), {"firewall": firewall})()
    override_path = tmp_path / "config.json"
    manager = RuntimeConfigManager(path=str(override_path), poll_seconds=10)

    override_path.write_text(
        json.dumps({"FIREWALL_MIN_CONFIDENCE": 0.62}),
        encoding="utf-8",
    )
    assert manager.poll(container, force=True) is True
    assert config.FIREWALL_MIN_CONFIDENCE == 0.62
    assert firewall.min_confidence == 0.62

    override_path.unlink()
    assert manager.poll(container, force=True) is True
    assert config.FIREWALL_MIN_CONFIDENCE == 0.45
    assert firewall.min_confidence == 0.45
