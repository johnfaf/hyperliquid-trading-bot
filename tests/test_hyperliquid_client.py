import src.data.hyperliquid_client as hl


def test_get_all_mids_returns_parsed_floats(monkeypatch):
    monkeypatch.setattr(
        hl,
        "_post",
        lambda payload, priority=None, retries=3: {
            "BTC": "123.45",
            "ETH": "oops",
            "SOL": "250.0",
        },
    )

    mids = hl.get_all_mids()

    assert mids == {"BTC": 123.45, "SOL": 250.0}
    assert isinstance(mids["BTC"], float)


def test_get_user_state_warns_once_for_invalid_address(monkeypatch, caplog):
    hl._WARNED_INVALID_ADDRESSES.clear()

    with caplog.at_level("WARNING"):
        assert hl.get_user_state("not-an-address") is None
        assert hl.get_user_state("not-an-address") is None

    warnings = [record.message for record in caplog.records if "skipping malformed address" in record.message]
    assert len(warnings) == 1
