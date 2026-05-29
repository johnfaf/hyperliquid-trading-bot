import sqlite3
from datetime import datetime, timedelta, timezone

from scripts.investor_report import generate_investor_report
from src.analysis.investor_evidence import (
    build_baselines,
    build_live_evidence_pack,
    build_walk_forward_report,
    render_baseline_markdown,
    snapshot_dataset,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            strategy_id INTEGER,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            size REAL NOT NULL,
            leverage REAL DEFAULT 1,
            pnl REAL DEFAULT 0,
            status TEXT DEFAULT 'open',
            stop_loss REAL,
            take_profit REAL,
            client_order_id TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE wallet_fills (
            id INTEGER PRIMARY KEY,
            wallet_address TEXT NOT NULL,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            original_price REAL NOT NULL,
            penalised_price REAL NOT NULL,
            size REAL NOT NULL,
            time_ms INTEGER NOT NULL,
            delayed_time_ms INTEGER NOT NULL,
            closed_pnl REAL DEFAULT 0,
            penalised_pnl REAL DEFAULT 0,
            fee REAL DEFAULT 0,
            is_liquidation INTEGER DEFAULT 0,
            direction TEXT DEFAULT ''
        );
        CREATE TABLE candles (
            coin TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        );
        CREATE TABLE audit_trail (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT DEFAULT '{}'
        );
        """
    )
    now = datetime.now(timezone.utc)
    pnls = [10.0, -5.0, 12.0, 3.0, -2.0]
    for idx, pnl in enumerate(pnls, start=1):
        opened = now - timedelta(days=6 - idx, hours=1)
        closed = now - timedelta(days=6 - idx)
        conn.execute(
            """
            INSERT INTO paper_trades
            (id, opened_at, closed_at, coin, side, entry_price, exit_price, size, pnl, status, metadata)
            VALUES (?, ?, ?, 'BTC', 'long', 50000, 50100, 0.01, ?, 'closed', ?)
            """,
            (
                idx,
                opened.isoformat(),
                closed.isoformat(),
                pnl,
                '{"source":"unit","fee":0.25,"funding":-0.05}',
            ),
        )
    start_ms = int((now - timedelta(days=7)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    for coin, start_px, end_px in (("BTC", 50_000, 55_000), ("ETH", 2_000, 1_900)):
        conn.execute(
            "INSERT INTO candles VALUES (?, '1h', ?, ?, ?, ?, ?, 1)",
            (coin, start_ms, start_px, start_px, start_px, start_px),
        )
        conn.execute(
            "INSERT INTO candles VALUES (?, '1h', ?, ?, ?, ?, ?, 1)",
            (coin, end_ms, end_px, end_px, end_px, end_px),
        )
    for wallet, pnl in (("0xaaa", 5.0), ("0xbbb", 8.0), ("0xccc", -4.0)):
        conn.execute(
            """
            INSERT INTO wallet_fills
            (wallet_address, coin, side, original_price, penalised_price, size,
             time_ms, delayed_time_ms, closed_pnl, penalised_pnl, fee, direction)
            VALUES (?, 'BTC', 'buy', 50000, 50010, 0.01, ?, ?, ?, ?, 0.1, 'Close Long')
            """,
            (wallet, end_ms, end_ms, pnl, pnl - 0.1),
        )
    conn.execute(
        "INSERT INTO audit_trail (timestamp, action, details) VALUES (?, 'signal_rejected', '{}')",
        ((now - timedelta(days=1)).isoformat(),),
    )
    conn.commit()
    return conn


def test_baseline_report_contains_required_benchmarks():
    conn = _conn()
    report = build_baselines(conn, candle_conn=conn, window_days=30, random_wallets=2)
    names = {row["benchmark"] for row in report["benchmarks"]}

    assert "Bot closed paper trades" in names
    assert "BTC buy-and-hold" in names
    assert "ETH buy-and-hold" in names
    assert "Top-wallet naive mirror" in names
    assert report["bot_metrics"]["trades"] == 5
    assert "Baseline Benchmark Report" in render_baseline_markdown(report)


def test_snapshot_walkforward_and_live_pack_write_artifacts(tmp_path, monkeypatch):
    conn = _conn()

    # Hermetic: this test asserts the UNSIGNED evidence path, so it must
    # guarantee no agent signing key is reachable -- independent of the
    # ambient shell env or any earlier test that leaked
    # HL_AGENT_PRIVATE_KEY into os.environ.  (build_live_evidence_pack
    # signs the pack whenever that key resolves; CI runs without it but
    # a developer machine / Railway shell may have it set.)
    monkeypatch.delenv("HL_AGENT_PRIVATE_KEY", raising=False)

    manifest = snapshot_dataset(conn, tmp_path / "dataset", window_days=30)
    walk = build_walk_forward_report(conn, window_days=30, starting_capital=10_000)
    pack = build_live_evidence_pack(conn, tmp_path / "live", window_days=30)

    assert manifest["row_counts"]["paper_trades"] == 5
    assert manifest["dataset_sha256"]
    assert walk["row_counts"] == {"train": 3, "validation": 1, "test": 1}
    assert pack["source_sha256"]
    assert pack["signature"]["signed"] is False


def test_generate_investor_report_with_sqlite_path(tmp_path):
    db_path = tmp_path / "bot.db"
    source = _conn()
    disk = sqlite3.connect(db_path)
    source.backup(disk)
    disk.close()
    source.close()

    out = tmp_path / "investor.md"
    result = generate_investor_report(db_path=str(db_path), out=out, window="30d")

    assert out.exists()
    assert "Investor Evidence Report" in out.read_text(encoding="utf-8")
    assert result["dataset_sha256"]
    assert result["trade_csv"]
