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


def test_get_user_state_returns_negative_equity_state(monkeypatch, caplog):
    monkeypatch.setattr(
        hl,
        "_post",
        lambda payload, priority=None, retries=3: {
            "assetPositions": [],
            "marginSummary": {
                "accountValue": "-123.45",
                "totalMarginUsed": "50",
                "totalNtlPos": "200",
            },
            "withdrawable": "-5",
        },
    )

    with caplog.at_level("WARNING"):
        state = hl.get_user_state("0x1234567890abcdef1234567890abcdef12345678")

    assert state is not None
    assert state["account_value"] == -123.45
    assert state["account_status"] == "negative_equity"
    assert state["liquidation_risk"] is True
    assert "negative account_value" in caplog.text


def test_get_user_state_skips_positions_with_invalid_entry_price(monkeypatch, caplog):
    hl._WARNED_INVALID_ENTRY_PRICES.clear()
    monkeypatch.setattr(
        hl,
        "_post",
        lambda payload, priority=None, retries=3: {
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "1.5",
                        "entryPx": None,
                        "leverage": {"value": "3"},
                        "unrealizedPnl": "10",
                        "returnOnEquity": "0.1",
                        "marginUsed": "25",
                    }
                }
            ],
            "marginSummary": {"accountValue": "1000", "totalMarginUsed": "25", "totalNtlPos": "100"},
            "withdrawable": "900",
        },
    )

    with caplog.at_level("WARNING"):
        state = hl.get_user_state("0x1234567890abcdef1234567890abcdef12345678")

    assert state is not None
    assert state["positions"] == []
    assert "entryPx" in caplog.text
