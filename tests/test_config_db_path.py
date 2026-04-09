import os

import config


def test_resolve_db_path_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("HL_BOT_DB", "C:\\custom\\bot.db")

    assert config._resolve_db_path() == "C:\\custom\\bot.db"


def test_resolve_db_path_falls_back_to_repo_data_when_no_persistent_volume(monkeypatch):
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    monkeypatch.setattr(config, "_can_use_persistent_volume", lambda: False)

    expected = os.path.join(
        os.path.dirname(os.path.abspath(config.__file__)),
        "data",
        "bot.db",
    )

    assert config._resolve_db_path() == expected


def test_windows_never_auto_uses_posix_data_volume(monkeypatch):
    monkeypatch.setattr(config.os, "name", "nt", raising=False)
    monkeypatch.setattr(config.os.path, "isdir", lambda _: True)

    assert config._can_use_persistent_volume() is False
