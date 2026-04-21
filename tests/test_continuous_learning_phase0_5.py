import contextlib
import sqlite3

from src.data import database as db
from src.data import decision_journal
from src.data.feature_store import compute_features
from src.data.polymarket_history import PolymarketHistoricalProvider, store_markets
from src.learning.policy_registry import CHAMPION_POLICY_ID, ensure_champion_policy
from src.learning.schema import ensure_sqlite_schema
from src.learning.source_inventory import seed_source_inventory
from src.signals.signal_schema import RiskParams, SignalSide, SignalSource, TradeSignal


@contextlib.contextmanager
def _sqlite_ctx(conn):
    yield conn
    conn.commit()


def _memory_db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_sqlite_schema(conn)
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    return conn


def test_phase0_seeds_champion_policy_and_source_inventory(monkeypatch):
    conn = _memory_db(monkeypatch)

    assert ensure_champion_policy() == CHAMPION_POLICY_ID
    assert seed_source_inventory() >= 5

    policy = conn.execute(
        "SELECT * FROM continuous_learning_policies WHERE policy_id = ?",
        (CHAMPION_POLICY_ID,),
    ).fetchone()
    assert policy is not None
    assert policy["status"] == "champion"
    assert "max_live_order_usd" in policy["non_trainable_limits"]

    sources = conn.execute("SELECT source_name FROM source_inventory").fetchall()
    source_names = {row["source_name"] for row in sources}
    assert {"polymarket", "feature_store", "decision_journal"}.issubset(source_names)


def test_polymarket_historical_provider_is_point_in_time(monkeypatch):
    _memory_db(monkeypatch)

    store_markets(
        [
            {
                "id": "btc-up",
                "question": "Will BTC finish green?",
                "active": True,
                "closed": False,
                "volume": 100,
                "outcomePrices": ["0.42", "0.58"],
            }
        ],
        observed_at_ms=1_000,
    )
    store_markets(
        [
            {
                "id": "btc-up",
                "question": "Will BTC finish green?",
                "active": True,
                "closed": False,
                "volume": 200,
                "outcomePrices": ["0.66", "0.34"],
            }
        ],
        observed_at_ms=2_000,
    )

    early = PolymarketHistoricalProvider(as_of_ms=1_500).fetch_markets(limit=10)
    late = PolymarketHistoricalProvider(as_of_ms=2_500).fetch_markets(limit=10)

    assert early[0]["outcomePrices"] == ["0.42", "0.58"]
    assert late[0]["outcomePrices"] == ["0.66", "0.34"]


def test_expanded_feature_vector_accepts_external_inputs():
    candles = []
    price = 100.0
    for idx in range(140):
        price *= 1.001 if idx % 3 else 0.999
        candles.append(
            {
                "t": idx,
                "o": price * 0.999,
                "h": price * 1.01,
                "l": price * 0.99,
                "c": price,
                "v": 1000 + idx * 5,
            }
        )

    features = compute_features(
        "BTC",
        "1h",
        candles,
        funding_rate=0.0002,
        prev_funding_rate=0.0001,
        funding_history=[-0.0001, 0.0, 0.0001, 0.0002],
        open_interest=110.0,
        prev_open_interest=100.0,
        spread_bps=2.5,
        book_imbalance=0.2,
        estimated_slippage_bps=4.0,
        polymarket_features={"probability": 0.61, "probability_delta_1h": 0.03},
        options_features={"iv_rank": 0.72, "skew": -0.1, "flow_direction_score": 1.0},
        source_quality={"source_missing_ratio": 0.1, "max_source_age_seconds": 42},
    )

    assert "return_3" in features
    assert "return_12" in features
    assert "bb_width_20" in features
    assert features["funding_slope"] == 0.0001
    assert features["spread_bps"] == 2.5
    assert features["polymarket_probability"] == 0.61
    assert features["options_iv_rank"] == 0.72
    assert features["source_missing_ratio"] == 0.1


def test_decision_journal_records_updates_and_links_paper_trade(monkeypatch):
    conn = _memory_db(monkeypatch)
    signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.64,
        source=SignalSource.STRATEGY,
        reason="test",
        strategy_id="s1",
        strategy_type="momentum",
        entry_price=100.0,
        leverage=5.0,
        position_pct=0.02,
        risk=RiskParams(stop_loss_pct=0.05, take_profit_pct=0.25, risk_basis="roe"),
        context={"features": {"return_3": 0.01}},
    )

    decision_id = decision_journal.record_decision_snapshot(
        signal,
        account_balance=10_000,
        final_status="candidate",
    )
    assert decision_id == signal.signal_id
    assert decision_journal.update_decision_status(
        decision_id,
        final_status="approved",
        firewall_decision="approved",
        metadata={"stage": "firewall"},
    )
    assert decision_journal.link_paper_trade(decision_id, 123, metadata={"opened": True})

    row = conn.execute(
        "SELECT * FROM decision_snapshots WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()
    assert row["final_status"] == "paper_opened"
    assert row["paper_trade_id"] == 123
    assert row["proposed_size_usd"] == 200.0
    assert row["proposed_tp_roe"] == 0.25
