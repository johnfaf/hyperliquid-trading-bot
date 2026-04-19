"""
Trade Memory System
====================
Stores past trades with their full context (features, decision, outcome)
and retrieves similar past trades before new decisions.

This gives the system "memory" — instead of treating every trade as independent,
it can say: "The last 5 times we saw this exact setup, 4 lost money."

Architecture:
  - Shared runtime database-backed persistent storage
  - Feature vector similarity search (cosine similarity)
  - Retrieval returns most similar past trades with outcomes
  - Summary statistics for similar trade clusters

Not a vector database (no embeddings needed) — we use direct numeric
feature comparison which is more appropriate for structured trading data.
"""
import logging
import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from src.data import database as db

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A complete record of a past trade with context."""
    trade_id: str
    coin: str
    side: str
    strategy_type: str
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    win: bool
    opened_at: str
    closed_at: str

    # Feature context at time of trade
    features: Dict  # The market features when we entered

    # Decision context
    confidence: float
    source: str
    regime: str
    setup_type: str  # e.g. "full_confluence", "funding_extreme"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SimilarityResult:
    """Result of a similarity search."""
    similar_trades: List[Dict]
    total_found: int
    win_rate: float          # Win rate of similar trades
    avg_pnl: float           # Average PnL of similar trades
    avg_return: float        # Average return %
    recommendation: str      # "proceed", "caution", "avoid"
    reason: str
    similarity_scores: List[float]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["similar_trades"] = self.similar_trades[:5]  # Limit for display
        return d


# Feature keys used for similarity comparison
SIMILARITY_FEATURES = [
    "funding_rate", "oi_change", "price_change", "trend_strength",
    "volatility", "volume_ratio", "rsi", "momentum_score",
    "bollinger_position", "overall_score",
]


class TradeMemory:
    """
    Persistent trade memory with similarity-based retrieval.

    Usage:
        memory = TradeMemory()
        memory.record_trade(...)  # After each trade closes
        similar = memory.find_similar(current_features)  # Before new trade
    """

    def __init__(self, db_path: Optional[str] = None):
        import config
        self.db_path = db_path or config.DB_PATH
        if self.db_path != config.DB_PATH:
            logger.warning(
                "TradeMemory db_path override (%s) is no longer used; "
                "using shared runtime database instead.",
                self.db_path,
            )
        self.db_path = config.DB_PATH
        self._init_table()
        self._cache_count = 0
        self._update_cache_count()

        logger.info(f"TradeMemory initialized with {self._cache_count} stored trades")

    def close(self):
        """No-op: TradeMemory now uses the shared database connection layer."""
        return

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _init_table(self):
        """Create the trade_memory table if it doesn't exist."""
        try:
            with db.get_connection() as conn:
                if db.get_backend_name() == "postgres":
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS trade_memory (
                            trade_id TEXT PRIMARY KEY,
                            coin TEXT NOT NULL,
                            side TEXT NOT NULL,
                            strategy_type TEXT,
                            entry_price DOUBLE PRECISION,
                            exit_price DOUBLE PRECISION,
                            pnl DOUBLE PRECISION,
                            return_pct DOUBLE PRECISION,
                            win INTEGER,
                            opened_at TIMESTAMPTZ,
                            closed_at TIMESTAMPTZ,
                            confidence DOUBLE PRECISION,
                            source TEXT,
                            regime TEXT,
                            setup_type TEXT,
                            features_json TEXT,
                            feature_vector TEXT
                        )
                    """)
                else:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS trade_memory (
                            trade_id TEXT PRIMARY KEY,
                            coin TEXT NOT NULL,
                            side TEXT NOT NULL,
                            strategy_type TEXT,
                            entry_price REAL,
                            exit_price REAL,
                            pnl REAL,
                            return_pct REAL,
                            win INTEGER,
                            opened_at TEXT,
                            closed_at TEXT,
                            confidence REAL,
                            source TEXT,
                            regime TEXT,
                            setup_type TEXT,
                            features_json TEXT,
                            feature_vector TEXT
                        )
                    """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trade_memory_coin
                    ON trade_memory(coin)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trade_memory_strategy
                    ON trade_memory(strategy_type)
                """)
        except Exception as e:
            logger.warning(f"Could not init trade_memory table: {e}")

    def _update_cache_count(self):
        try:
            with db.get_connection(for_read=True) as conn:
                row = conn.execute("SELECT COUNT(*) AS c FROM trade_memory").fetchone()
            # psycopg dict_row returns a dict (no integer indexing);
            # sqlite3.Row supports both. Use the named alias for parity.
            self._cache_count = (row["c"] if row else 0) or 0
        except Exception:
            self._cache_count = 0

    def record_trade(self, trade_id: str, coin: str, side: str,
                      strategy_type: str, entry_price: float, exit_price: float,
                      pnl: float, return_pct: float, opened_at: str, closed_at: str,
                      confidence: float = 0, source: str = "",
                      regime: str = "", setup_type: str = "",
                      features: Optional[Dict] = None):
        """
        Record a completed trade with its full context.
        Call this every time a trade closes.
        """
        features = features or {}
        win = 1 if pnl > 0 else 0

        # Extract feature vector for similarity search
        feature_vector = self._extract_feature_vector(features)

        try:
            with db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO trade_memory
                    (trade_id, coin, side, strategy_type, entry_price, exit_price,
                     pnl, return_pct, win, opened_at, closed_at, confidence,
                     source, regime, setup_type, features_json, feature_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        coin = EXCLUDED.coin,
                        side = EXCLUDED.side,
                        strategy_type = EXCLUDED.strategy_type,
                        entry_price = EXCLUDED.entry_price,
                        exit_price = EXCLUDED.exit_price,
                        pnl = EXCLUDED.pnl,
                        return_pct = EXCLUDED.return_pct,
                        win = EXCLUDED.win,
                        opened_at = EXCLUDED.opened_at,
                        closed_at = EXCLUDED.closed_at,
                        confidence = EXCLUDED.confidence,
                        source = EXCLUDED.source,
                        regime = EXCLUDED.regime,
                        setup_type = EXCLUDED.setup_type,
                        features_json = EXCLUDED.features_json,
                        feature_vector = EXCLUDED.feature_vector
                """, (
                    trade_id, coin, side, strategy_type, entry_price, exit_price,
                    pnl, return_pct, win, opened_at, closed_at, confidence,
                    source, regime, setup_type,
                    json.dumps(features), json.dumps(feature_vector),
                ))
            self._cache_count += 1
        except Exception as e:
            logger.debug(f"Could not record trade to memory: {e}")

    def find_similar(self, features: Dict, coin: Optional[str] = None,
                      strategy_type: Optional[str] = None,
                      side: Optional[str] = None,
                      top_k: int = 10,
                      min_similarity: float = 0.5) -> SimilarityResult:
        """
        Find trades with similar market conditions.

        Args:
            features: Current market features dict
            coin: Filter by coin (None = all coins)
            strategy_type: Filter by strategy type (None = all)
            side: Filter by trade side (None = both)
            top_k: Number of similar trades to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            SimilarityResult with similar trades and statistics.
        """
        query_vector = self._extract_feature_vector(features)

        if not any(v != 0 for v in query_vector):
            return SimilarityResult(
                similar_trades=[], total_found=0, win_rate=0, avg_pnl=0,
                avg_return=0, recommendation="proceed",
                reason="No feature data available", similarity_scores=[],
            )

        # Query past trades
        try:
            query = "SELECT * FROM trade_memory WHERE 1=1"
            params = []

            if coin:
                query += " AND coin = ?"
                params.append(coin)
            if strategy_type:
                query += " AND strategy_type = ?"
                params.append(strategy_type)
            if side:
                query += " AND side = ?"
                params.append(side)

            with db.get_connection(for_read=True) as conn:
                rows = conn.execute(query, params).fetchall()
        except Exception as e:
            # MED-FIX MED-8: elevate to WARNING and return "caution" instead of
            # "proceed" — a DB error here means historical loss patterns on this
            # setup are invisible.  Proceeding silently bypasses the memory system
            # entirely; "caution" passes control to the caller but flags the gap.
            logger.warning(
                "Trade memory query error (defaulting to caution -- loss patterns unavailable): %s", e
            )
            return SimilarityResult(
                similar_trades=[], total_found=0, win_rate=0, avg_pnl=0,
                avg_return=0, recommendation="caution",
                reason=f"Query error: {e}", similarity_scores=[],
            )

        if not rows:
            return SimilarityResult(
                similar_trades=[], total_found=0, win_rate=0, avg_pnl=0,
                avg_return=0, recommendation="proceed",
                reason="No past trades found", similarity_scores=[],
            )

        # Calculate similarity for each past trade
        scored_trades = []
        for row in rows:
            row = dict(row)
            try:
                past_vector = json.loads(row.get("feature_vector", "[]"))
                if not past_vector:
                    continue
                sim = self._cosine_similarity(query_vector, past_vector)
                if sim >= min_similarity:
                    scored_trades.append((sim, row))
            except Exception:
                continue

        # Sort by similarity (highest first)
        scored_trades.sort(key=lambda x: x[0], reverse=True)
        top_trades = scored_trades[:top_k]

        if not top_trades:
            return SimilarityResult(
                similar_trades=[], total_found=0, win_rate=0, avg_pnl=0,
                avg_return=0, recommendation="proceed",
                reason="No similar trades found above threshold",
                similarity_scores=[],
            )

        # Calculate statistics
        similar_trades = []
        wins = 0
        total_pnl = 0.0
        total_return = 0.0
        similarity_scores = []

        for sim, trade in top_trades:
            similar_trades.append({
                "trade_id": trade["trade_id"],
                "coin": trade["coin"],
                "side": trade["side"],
                "pnl": trade["pnl"],
                "return_pct": trade["return_pct"],
                "win": bool(trade["win"]),
                "similarity": round(sim, 3),
                "strategy_type": trade["strategy_type"],
                "regime": trade["regime"],
            })
            if trade["win"]:
                wins += 1
            total_pnl += trade["pnl"]
            total_return += trade.get("return_pct", 0)
            similarity_scores.append(sim)

        n = len(top_trades)
        win_rate = wins / n
        avg_pnl = total_pnl / n
        avg_return = total_return / n

        # Generate recommendation
        recommendation, reason = self._generate_recommendation(
            win_rate, avg_pnl, avg_return, n
        )

        return SimilarityResult(
            similar_trades=similar_trades,
            total_found=n,
            win_rate=round(win_rate, 3),
            avg_pnl=round(avg_pnl, 2),
            avg_return=round(avg_return, 4),
            recommendation=recommendation,
            reason=reason,
            similarity_scores=similarity_scores,
        )

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        try:
            with db.get_connection(for_read=True) as conn:
                # Named aliases so psycopg dict_row and sqlite3.Row both work.
                total = conn.execute("SELECT COUNT(*) AS c FROM trade_memory").fetchone()["c"]
                wins = conn.execute("SELECT COUNT(*) AS c FROM trade_memory WHERE win = 1").fetchone()["c"]
                coins = conn.execute("SELECT COUNT(DISTINCT coin) AS c FROM trade_memory").fetchone()["c"]
                strategies = conn.execute("SELECT COUNT(DISTINCT strategy_type) AS c FROM trade_memory").fetchone()["c"]
            return {
                "total_trades": total,
                "win_rate": wins / total if total > 0 else 0,
                "unique_coins": coins,
                "unique_strategies": strategies,
            }
        except Exception:
            return {"total_trades": 0, "win_rate": 0, "unique_coins": 0, "unique_strategies": 0}

    # ─── Internal Methods ────────────────────────────────────────

    def _extract_feature_vector(self, features: Dict) -> List[float]:
        """
        Extract a normalized numeric vector from features dict.
        Used for similarity comparison.
        """
        vector = []
        for key in SIMILARITY_FEATURES:
            val = features.get(key, 0)
            if isinstance(val, (int, float)):
                vector.append(float(val))
            else:
                vector.append(0.0)

        # Normalize to unit length for cosine similarity
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _generate_recommendation(self, win_rate: float, avg_pnl: float,
                                   avg_return: float, n_trades: int) -> Tuple[str, str]:
        """Generate a recommendation based on similar trade statistics."""
        if n_trades < 3:
            return "proceed", f"Only {n_trades} similar trades found — insufficient data"

        if win_rate >= 0.6 and avg_pnl > 0:
            return "proceed", (f"Similar setups won {win_rate:.0%} of the time "
                              f"(avg PnL ${avg_pnl:.2f}, n={n_trades})")
        elif win_rate >= 0.45 and avg_pnl > 0:
            return "proceed", (f"Similar setups slightly profitable "
                              f"(WR={win_rate:.0%}, avg PnL ${avg_pnl:.2f})")
        elif win_rate >= 0.35:
            return "caution", (f"Similar setups have mixed results "
                              f"(WR={win_rate:.0%}, avg PnL ${avg_pnl:.2f})")
        else:
            return "avoid", (f"Similar setups mostly lost money "
                            f"(WR={win_rate:.0%}, avg PnL ${avg_pnl:.2f}, n={n_trades})")
