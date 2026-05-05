import pytest

from src.data import feature_store
from src.ui import backtest_dashboard, report_exporter, stress_dashboard


class _FakeCursor:
    def __init__(self):
        self.executed_rows = []

    def executemany(self, _sql, rows):
        self.executed_rows.extend(rows)


class _FakePgConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_csv_export_sanitizes_formula_cells():
    assert report_exporter._csv_safe_cell("=HYPERLINK('https://evil')") == "'=HYPERLINK('https://evil')"
    assert report_exporter._csv_safe_cell("+SUM(1,1)") == "'+SUM(1,1)"
    assert report_exporter._csv_safe_cell(" BTC") == "' BTC"
    assert report_exporter._csv_safe_cell("BTC") == "BTC"


def test_backtest_wallet_detail_rejects_invalid_address_before_db(monkeypatch):
    monkeypatch.setattr(
        backtest_dashboard,
        "_get_db",
        lambda: pytest.fail("invalid wallet address should not touch the DB"),
    )

    assert backtest_dashboard.get_wallet_detail("');alert(1);//") is None


def test_store_features_skips_non_finite_values(monkeypatch):
    conn = _FakePgConn()
    monkeypatch.setattr(feature_store, "_pg_conn", lambda: (conn, lambda _conn: None))

    written = feature_store.store_features(
        "BTC",
        "1h",
        123,
        {"good": 1.5, "nan": float("nan"), "inf": float("inf"), "bad": "x"},
    )

    assert written == 1
    assert conn.cursor_obj.executed_rows == [("BTC", "1h", 123, "good", 1.5)]
    assert conn.committed is True


def test_store_candles_skips_malformed_rows(monkeypatch):
    conn = _FakePgConn()
    monkeypatch.setattr(feature_store, "_pg_conn", lambda: (conn, lambda _conn: None))

    written = feature_store.store_candles(
        "ETH",
        "5m",
        [
            {"t": 1, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100},
            {"t": 2, "o": 10, "h": None, "l": 9, "c": 10.5, "v": 100},
            {"o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100},
        ],
    )

    assert written == 1
    assert conn.cursor_obj.executed_rows == [("ETH", "5m", 1, 10.0, 11.0, 9.0, 10.5, 100.0)]


def test_stress_dashboard_rejects_unknown_scenarios():
    with pytest.raises(ValueError, match="invalid stress scenario"):
        stress_dashboard._validate_scenarios(["flash_crash", "not_real"])
