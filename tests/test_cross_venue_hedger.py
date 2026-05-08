import base64
import hashlib
import hmac

from src.trading.cross_venue_hedger import CrossVenueHedger, HedgeVenue


def test_check_and_hedge_filters_positions_without_coin(monkeypatch):
    hedger = CrossVenueHedger(
        {
            "dry_run": True,
            "kraken_enabled": False,
            "binance_enabled": True,
            "bybit_enabled": False,
            "crash_confidence": 0.5,
            "rate_limit_ms": 0,
        }
    )
    placed = []
    monkeypatch.setattr(
        hedger,
        "_place_hedges",
        lambda coin, position: placed.append((coin, position["size"])) or True,
    )

    result = hedger.check_and_hedge(
        {"regime": "crash", "confidence": 0.9},
        [
            {"size": 1.0, "side": "long"},
            {"coin": "", "size": 2.0, "side": "long"},
            {"coin": " btc ", "size": 3.0, "side": "long"},
        ],
    )

    assert placed == [("BTC", 3.0)]
    assert result["coins_affected"] == ["BTC"]
    assert result["hedges_placed"] == 1


def test_kraken_dry_run_records_active_hedge():
    hedger = CrossVenueHedger(
        {
            "dry_run": True,
            "kraken_enabled": True,
            "binance_enabled": False,
            "bybit_enabled": False,
            "crash_confidence": 0.5,
            "rate_limit_ms": 0,
        }
    )

    result = hedger.check_and_hedge(
        {"regime": "crash", "confidence": 0.9},
        {"BTC": {"side": "long", "size": 1.0}},
    )

    assert result["action"] == "hedged"
    assert result["hedges_placed"] == 1
    active = hedger.get_active_hedges()
    assert active[HedgeVenue.KRAKEN.value], "Kraken active hedge should be tracked"
    assert active[HedgeVenue.KRAKEN.value][0]["coin"] == "BTC"
    assert active[HedgeVenue.KRAKEN.value][0]["side"] == "SELL"


def test_kraken_symbol_maps_btc_to_xbt_and_uses_template():
    hedger = CrossVenueHedger(
        {
            "dry_run": True,
            "kraken_enabled": True,
            "binance_enabled": False,
            "bybit_enabled": False,
        }
    )
    assert hedger._kraken_symbol("BTC") == "PF_XBTUSD"
    assert hedger._kraken_symbol("eth") == "PF_ETHUSD"
    assert hedger._kraken_symbol("SOL") == "PF_SOLUSD"


def test_kraken_sign_matches_known_hmac_construction():
    # Use a known secret/post/nonce/path and verify the signature shape
    # (not against Kraken's actual API, just the documented signing recipe).
    secret_bytes = b"\x01" * 32
    secret_b64 = base64.b64encode(secret_bytes).decode("ascii")
    hedger = CrossVenueHedger({"dry_run": True, "kraken_enabled": True})
    hedger.kraken_api_secret = secret_b64

    post_data = "orderType=mkt&symbol=PF_XBTUSD&side=sell&size=0.01&reduceOnly=true"
    nonce = "1700000000000000"
    path = "/derivatives/api/v3/sendorder"
    expected_sha = hashlib.sha256((post_data + nonce + path).encode("utf-8")).digest()
    expected_mac = hmac.new(secret_bytes, expected_sha, hashlib.sha512).digest()
    expected_b64 = base64.b64encode(expected_mac).decode("ascii")

    assert hedger._kraken_sign(post_data, nonce, path) == expected_b64
