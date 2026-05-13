import io
import sqlite3
import zipfile
from datetime import date

from scripts import backfill_multi_coin_klines as backfill


class _Resp:
    status_code = 200

    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


def _zip_payload():
    csv_body = "1000,100,101,99,100.5,12,0,0,0,0,0,0\n2000,101,102,100,101.5,13,0,0,0,0,0,0\n"
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("BTCUSDC-1s-2026-05-01.csv", csv_body)
    return bio.getvalue()


def test_backfill_binance_vision_stores_timeframe_and_source(tmp_path, monkeypatch):
    seen = {}

    def fake_get(url, timeout=30):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Resp(_zip_payload())

    monkeypatch.setattr(backfill.requests, "get", fake_get)
    db_path = tmp_path / "candle_cache.db"

    stats = backfill.backfill(
        coins=["BTC"],
        start=date(2026, 5, 1),
        end=date(2026, 5, 1),
        timeframe="1s",
        cache_db=str(db_path),
        quote="USDC",
        request_sleep_s=0,
    )

    assert "BTCUSDC/1s/BTCUSDC-1s-2026-05-01.zip" in seen["url"]
    assert stats["inserted"] == 2
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT coin, timeframe, source, close FROM candles WHERE timestamp_ms = 1000"
    ).fetchone()
    conn.close()
    assert row == ("BTC", "1s", "binance_vision", 100.5)

