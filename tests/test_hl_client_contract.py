"""Integration-boundary contract tests for the Hyperliquid client.

The ``closedPnl``/``dir`` discovery bug shipped green because the *mock*
returned the wrong shape — nothing pinned the real client's NORMALIZED
output that discovery/firewall/calibration actually consume. These tests
lock that contract: feed a RAW (camelCase) HL payload through the real
``_post`` seam and assert the normalized snake_case schema, and that raw
keys can never leak through again.
"""
from __future__ import annotations

import pytest

import src.data.hyperliquid_client as hl

ADDR = "0x" + "a" * 40

# Raw Hyperliquid `userFills` shape (camelCase, as the public API returns).
_RAW_FILLS = [
    {
        "coin": "BTC", "side": "B", "dir": "Close Long", "px": "65000.5",
        "sz": "0.1", "time": 1700000000000, "fee": "0.02",
        "liquidation": False, "startPosition": "0.1", "closedPnl": "12.34",
        "hash": "0xhash", "oid": 99, "crossed": True,
    }
]

# Raw `clearinghouseState` shape.
_RAW_STATE = {
    "assetPositions": [
        {"position": {
            "coin": "ETH", "szi": "2.0", "entryPx": "3000",
            "leverage": {"value": 5}, "unrealizedPnl": "10",
            "returnOnEquity": "0.1", "marginUsed": "1200",
            "liquidationPx": "2500",
        }}
    ],
    "marginSummary": {
        "accountValue": "10234.5", "totalMarginUsed": "1200",
        "totalNtlPos": "6000",
    },
    "withdrawable": "9000",
}

_FILL_CONTRACT = {
    "coin", "side", "price", "size", "time", "fee", "is_liquidation",
    "start_position", "direction", "closed_pnl", "hash", "oid", "crossed",
}
_RAW_FILL_KEYS_FORBIDDEN = {"closedPnl", "dir", "px", "sz", "liquidation", "startPosition"}

_POS_CONTRACT = {
    "coin", "side", "size", "entry_price", "leverage", "unrealized_pnl",
    "return_on_equity", "margin_used", "liquidation_price",
}


def test_get_user_fills_emits_normalized_contract(monkeypatch):
    monkeypatch.setattr(hl, "_post", lambda *a, **k: _RAW_FILLS)
    fills = hl.get_user_fills(ADDR)
    assert len(fills) == 1
    f = fills[0]
    assert set(f.keys()) == _FILL_CONTRACT
    # The exact regression: raw camelCase must NOT survive normalization.
    assert not (_RAW_FILL_KEYS_FORBIDDEN & set(f.keys()))
    assert f["closed_pnl"] == pytest.approx(12.34)
    assert f["direction"] == "Close Long"
    assert f["price"] == pytest.approx(65000.5)
    assert f["size"] == pytest.approx(0.1)
    assert f["is_liquidation"] is False


def test_get_user_state_emits_normalized_contract(monkeypatch):
    monkeypatch.setattr(hl, "_post", lambda *a, **k: _RAW_STATE)
    st = hl.get_user_state(ADDR)
    assert st is not None
    assert {"positions", "account_value", "total_margin_used",
            "total_ntl_pos", "withdrawable"} <= set(st.keys())
    assert st["account_value"] == pytest.approx(10234.5)
    p = st["positions"][0]
    assert _POS_CONTRACT <= set(p.keys())
    assert "szi" not in p and "entryPx" not in p
    assert p["side"] == "long"
    assert p["size"] == pytest.approx(2.0)
    assert p["entry_price"] == pytest.approx(3000.0)
    assert p["leverage"] == pytest.approx(5.0)


def test_profitability_consumer_reads_normalized_keys(monkeypatch):
    """Cross-boundary guard: discovery's 90-day profitability gate must
    score correctly on the NORMALIZED shape the real client returns
    (this is exactly what regressed and inverted the seed filter)."""
    import src.discovery.trader_discovery as td

    norm = (
        [{"direction": "Open Long", "closed_pnl": -0.01, "fee": 0.01},
         {"direction": "Close Long", "closed_pnl": 3.0, "fee": 0.02}] * 12
    )
    monkeypatch.setattr(td.hl, "get_user_fills", lambda *a, **k: norm)
    v = td.evaluate_trader_profitability_window(
        ADDR, window_days=90, min_trades=10, min_net_pnl_usd=0.0
    )
    assert v["verdict"] == "profitable"
    assert v["net_pnl"] > 0
