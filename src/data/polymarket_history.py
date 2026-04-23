"""Polymarket historical storage and point-in-time replay providers."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests

import config

logger = logging.getLogger(__name__)

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _as_list(value: Any) -> List[Any]:
    loaded = _loads(value, value)
    if loaded is None:
        return []
    if isinstance(loaded, list):
        return loaded
    return [loaded]


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "active"}


def _first(market: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in market and market.get(key) not in (None, ""):
            return market.get(key)
    return default


def _stable_id(prefix: str, payload: Dict[str, Any]) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def normalize_market_id(market: Dict[str, Any]) -> str:
    value = _first(
        market,
        ("id", "market_id", "conditionId", "condition_id", "slug", "marketSlug"),
        "",
    )
    value = str(value or "").strip()
    return value or _stable_id("pmkt", market)


def _is_condition_id(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    if len(raw) != 66 or not raw.startswith("0x"):
        return False
    return all(ch in "0123456789abcdef" for ch in raw[2:])


def _condition_id_from_market(market: Dict[str, Any]) -> str:
    value = _first(market, ("conditionId", "condition_id", "market"), "")
    value = str(value or "").strip()
    return value if _is_condition_id(value) else ""


def _market_probability(market: Dict[str, Any]) -> Optional[float]:
    for key in ("probability", "lastTradePrice", "last_price", "price"):
        parsed = _float(market.get(key))
        if parsed is not None:
            return parsed
    prices = [_float(v) for v in _as_list(market.get("outcomePrices"))]
    prices = [v for v in prices if v is not None]
    if prices:
        return prices[0]
    return None


def _extract_tokens(market: Dict[str, Any], market_id: str, observed_at_ms: int) -> List[Dict[str, Any]]:
    raw_tokens = market.get("tokens")
    tokens: List[Dict[str, Any]] = []
    if isinstance(raw_tokens, list):
        for item in raw_tokens:
            if not isinstance(item, dict):
                continue
            token_id = str(_first(item, ("token_id", "tokenId", "id"), "") or "").strip()
            if not token_id:
                continue
            outcome = str(_first(item, ("outcome", "name"), "") or "")
            tokens.append(
                {
                    "token_id": token_id,
                    "market_id": market_id,
                    "outcome": outcome,
                    "side": outcome.lower(),
                    "metadata": item,
                    "first_seen_ms": observed_at_ms,
                    "last_seen_ms": observed_at_ms,
                }
            )
        return tokens

    token_ids = _as_list(
        _first(market, ("clobTokenIds", "clobTokenIDs", "tokenIds", "token_ids"), [])
    )
    outcomes = _as_list(market.get("outcomes"))
    for idx, token_id in enumerate(token_ids):
        token_id = str(token_id or "").strip()
        if not token_id:
            continue
        outcome = str(outcomes[idx] if idx < len(outcomes) else "")
        tokens.append(
            {
                "token_id": token_id,
                "market_id": market_id,
                "outcome": outcome,
                "side": outcome.lower(),
                "metadata": {"index": idx},
                "first_seen_ms": observed_at_ms,
                "last_seen_ms": observed_at_ms,
            }
        )
    return tokens


def store_markets(raw_markets: Iterable[Dict[str, Any]], observed_at_ms: Optional[int] = None) -> int:
    """Upsert market metadata, tokens, and one snapshot per raw market."""
    from src.data import database as db

    observed = int(observed_at_ms or _now_ms())
    count = 0
    with db.get_connection() as conn:
        for market in raw_markets or []:
            if not isinstance(market, dict):
                continue
            market_id = normalize_market_id(market)
            question = str(_first(market, ("question", "title", "description"), "") or "")
            slug = str(_first(market, ("slug", "marketSlug"), "") or "")
            category = str(_first(market, ("category", "categorySlug"), "") or "")
            active = _bool(market.get("active"), True)
            closed = _bool(market.get("closed"), False)
            end_date = _first(market, ("endDate", "end_date", "end_date_iso"), None)
            raw_json = _json(market)
            probability = _market_probability(market)
            volume = _float(_first(market, ("volume", "volumeNum", "totalVolume"), None))
            volume_24h = _float(_first(market, ("volume24hr", "volume24h", "volume_24h"), None))
            liquidity = _float(_first(market, ("liquidity", "liquidityNum"), None))
            best_bid = _float(_first(market, ("bestBid", "best_bid"), None))
            best_ask = _float(_first(market, ("bestAsk", "best_ask"), None))
            spread_bps = None
            if best_bid is not None and best_ask is not None and probability and probability > 0:
                spread_bps = max(0.0, (best_ask - best_bid) / probability * 10_000)

            conn.execute(
                """
                INSERT INTO polymarket_markets
                (market_id, question, slug, category, active, closed, end_date,
                 first_seen_ms, last_seen_ms, raw_market)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    question = EXCLUDED.question,
                    slug = EXCLUDED.slug,
                    category = EXCLUDED.category,
                    active = EXCLUDED.active,
                    closed = EXCLUDED.closed,
                    end_date = EXCLUDED.end_date,
                    last_seen_ms = EXCLUDED.last_seen_ms,
                    raw_market = EXCLUDED.raw_market
                """,
                (
                    market_id,
                    question,
                    slug,
                    category,
                    active,
                    closed,
                    end_date,
                    observed,
                    observed,
                    raw_json,
                ),
            )
            conn.execute(
                """
                INSERT INTO polymarket_market_snapshots
                (market_id, observed_at_ms, probability, volume, volume_24h,
                 liquidity, best_bid, best_ask, spread_bps, raw_market)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id, observed_at_ms) DO UPDATE SET
                    probability = EXCLUDED.probability,
                    volume = EXCLUDED.volume,
                    volume_24h = EXCLUDED.volume_24h,
                    liquidity = EXCLUDED.liquidity,
                    best_bid = EXCLUDED.best_bid,
                    best_ask = EXCLUDED.best_ask,
                    spread_bps = EXCLUDED.spread_bps,
                    raw_market = EXCLUDED.raw_market
                """,
                (
                    market_id,
                    observed,
                    probability,
                    volume,
                    volume_24h,
                    liquidity,
                    best_bid,
                    best_ask,
                    spread_bps,
                    raw_json,
                ),
            )
            for token in _extract_tokens(market, market_id, observed):
                conn.execute(
                    """
                    INSERT INTO polymarket_tokens
                    (token_id, market_id, outcome, side, metadata, first_seen_ms, last_seen_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(token_id) DO UPDATE SET
                        market_id = EXCLUDED.market_id,
                        outcome = EXCLUDED.outcome,
                        side = EXCLUDED.side,
                        metadata = EXCLUDED.metadata,
                        last_seen_ms = EXCLUDED.last_seen_ms
                    """,
                    (
                        token["token_id"],
                        token["market_id"],
                        token.get("outcome", ""),
                        token.get("side", ""),
                        _json(token.get("metadata", {})),
                        token["first_seen_ms"],
                        token["last_seen_ms"],
                    ),
                )
            count += 1
    return count


def store_trades(raw_trades: Iterable[Dict[str, Any]]) -> int:
    """Upsert Polymarket trade prints when available from a downloader."""
    from src.data import database as db

    rows = list(raw_trades or [])
    if not rows:
        return 0
    with db.get_connection() as conn:
        for trade in rows:
            if not isinstance(trade, dict):
                continue
            trade_id = _trade_id(trade)
            ts = int(_float(_first(trade, ("timestamp_ms", "timestampMs", "createdAtMs"), 0), 0) or 0)
            if ts <= 0:
                seconds = _float(_first(trade, ("timestamp", "createdAt"), 0), 0) or 0
                ts = int(seconds * 1000) if seconds < 10_000_000_000 else int(seconds)
            conn.execute(
                """
                INSERT INTO polymarket_trades
                (trade_id, market_id, token_id, timestamp_ms, side, price, size,
                 maker_address, taker_address, raw_trade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    market_id = EXCLUDED.market_id,
                    token_id = EXCLUDED.token_id,
                    timestamp_ms = EXCLUDED.timestamp_ms,
                    side = EXCLUDED.side,
                    price = EXCLUDED.price,
                    size = EXCLUDED.size,
                    maker_address = EXCLUDED.maker_address,
                    taker_address = EXCLUDED.taker_address,
                    raw_trade = EXCLUDED.raw_trade
                """,
                (
                    trade_id,
                    _first(trade, ("market_id", "market", "marketId", "conditionId"), None),
                    _first(trade, ("token_id", "tokenId", "asset_id", "asset"), None),
                    ts,
                    _first(trade, ("side", "takerSide"), None),
                    _float(trade.get("price")),
                    _float(_first(trade, ("size", "amount"), None)),
                    _first(trade, ("maker_address", "maker", "makerAddress"), None),
                    _first(trade, ("taker_address", "taker", "takerAddress", "proxyWallet"), None),
                    _json(trade),
                ),
            )
    return len(rows)


def _trade_id(trade: Dict[str, Any]) -> str:
    explicit_id = str(_first(trade, ("id", "trade_id"), "") or "").strip()
    if explicit_id:
        return explicit_id
    tx_hash = str(_first(trade, ("transactionHash", "transaction_hash"), "") or "").strip()
    if tx_hash:
        suffix = _stable_id(
            "trade",
            {
                "asset": _first(trade, ("asset", "asset_id", "tokenId"), None),
                "conditionId": _first(trade, ("conditionId", "market_id", "market"), None),
                "side": _first(trade, ("side", "takerSide"), None),
                "size": _first(trade, ("size", "amount"), None),
                "price": trade.get("price"),
                "timestamp": _first(
                    trade,
                    ("timestamp_ms", "timestampMs", "timestamp", "createdAtMs", "createdAt"),
                    None,
                ),
                "wallet": _first(trade, ("proxyWallet", "taker", "taker_address"), None),
            },
        )
        return f"{tx_hash}:{suffix.split('_', 1)[-1]}"
    return _stable_id("ptrade", trade)


def store_price_points(rows: Iterable[Dict[str, Any]]) -> int:
    from src.data import database as db

    items = list(rows or [])
    if not items:
        return 0
    with db.get_connection() as conn:
        for row in items:
            conn.execute(
                """
                INSERT INTO polymarket_price_points
                (token_id, timestamp_ms, price, source, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(token_id, timestamp_ms, source) DO UPDATE SET
                    price = EXCLUDED.price,
                    metadata = EXCLUDED.metadata
                """,
                (
                    str(row.get("token_id", "") or ""),
                    int(row.get("timestamp_ms", 0) or 0),
                    float(row.get("price", 0.0) or 0.0),
                    str(row.get("source", "polymarket") or "polymarket"),
                    _json(row.get("metadata", {})),
                ),
            )
    return len(items)


@dataclass
class PolymarketHistoricalProvider:
    """Point-in-time market provider for replay/backtests."""

    as_of_ms: int

    def fetch_markets(
        self,
        limit: int = 100,
        active_only: bool = True,
        as_of_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from src.data import database as db

        replay_ts = int(as_of_ms or self.as_of_ms)
        active_clause = "AND m.active = ? AND m.closed = ?" if active_only else ""
        params: List[Any] = [replay_ts]
        if active_only:
            params.extend([True, False])
        params.append(int(limit))
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(
                f"""
                SELECT s.raw_market, s.probability, s.volume, s.volume_24h,
                       s.liquidity, s.best_bid, s.best_ask, s.spread_bps,
                       m.market_id, m.question, m.slug, m.category, m.active, m.closed
                FROM polymarket_market_snapshots s
                JOIN (
                    SELECT market_id, MAX(observed_at_ms) AS observed_at_ms
                    FROM polymarket_market_snapshots
                    WHERE observed_at_ms <= ?
                    GROUP BY market_id
                ) latest
                  ON latest.market_id = s.market_id
                 AND latest.observed_at_ms = s.observed_at_ms
                JOIN polymarket_markets m ON m.market_id = s.market_id
                WHERE 1 = 1 {active_clause}
                ORDER BY COALESCE(s.volume_24h, s.volume, 0) DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        markets: List[Dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            raw = _loads(record.get("raw_market"), {}) or {}
            if not isinstance(raw, dict):
                raw = {}
            raw.setdefault("id", record.get("market_id"))
            raw.setdefault("question", record.get("question"))
            raw.setdefault("slug", record.get("slug"))
            raw.setdefault("category", record.get("category"))
            raw.setdefault("active", bool(record.get("active")))
            raw.setdefault("closed", bool(record.get("closed")))
            if record.get("probability") is not None:
                raw.setdefault("probability", record.get("probability"))
            if record.get("volume") is not None:
                raw.setdefault("volume", record.get("volume"))
            if record.get("volume_24h") is not None:
                raw.setdefault("volume24hr", record.get("volume_24h"))
            if record.get("liquidity") is not None:
                raw.setdefault("liquidity", record.get("liquidity"))
            markets.append(raw)
        return markets


class PolymarketHistoricalDownloader:
    """Small downloader for seeding historical/replay Polymarket snapshots."""

    def __init__(self, timeout: int = 10, max_retries: int = 3, trade_source: Optional[str] = None):
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.trade_source = str(
            trade_source or getattr(config, "POLYMARKET_TRADE_BACKFILL_SOURCE", "data_api")
        ).strip().lower()
        if self.trade_source not in {"data_api", "clob"}:
            self.trade_source = "data_api"
        self.trade_taker_only = bool(
            getattr(config, "POLYMARKET_TRADE_BACKFILL_TAKER_ONLY", False)
        )

    def _fetch_json(self, url: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    sleep_s = min(8.0, 0.5 * (2 ** attempt))
                    time.sleep(sleep_s)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                logger.debug("Polymarket historical fetch retry after %s", exc)
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        return None

    def fetch_gamma_markets(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        url = (
            f"{GAMMA_API}/markets?limit={int(limit)}&offset={int(offset)}"
            "&active=true&closed=false&order=volume&ascending=false"
        )
        payload = self._fetch_json(url)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("markets", []))
            return data if isinstance(data, list) else []
        return []

    def _resolve_market_reference(self, market_ref: Any) -> tuple[str, str]:
        if isinstance(market_ref, dict):
            market_id = normalize_market_id(market_ref)
            condition_id = _condition_id_from_market(market_ref)
            if not condition_id and _is_condition_id(market_id):
                condition_id = market_id
            return market_id, condition_id

        raw = str(market_ref or "").strip()
        if not raw:
            return "", ""
        if _is_condition_id(raw):
            return raw, raw

        try:
            from src.data import database as db

            with db.get_connection(for_read=True) as conn:
                row = conn.execute(
                    "SELECT market_id, raw_market FROM polymarket_markets WHERE market_id = ? LIMIT 1",
                    (raw,),
                ).fetchone()
            if row is not None:
                market_id = str(row["market_id"] if hasattr(row, "keys") else row[0] or raw)
                raw_market = _loads(row["raw_market"] if hasattr(row, "keys") else row[1], {}) or {}
                return market_id, _condition_id_from_market(raw_market)
        except Exception:
            pass
        return raw, ""

    def _normalize_trade_rows(
        self,
        trades: Any,
        *,
        market_id: str,
        condition_id: str,
    ) -> List[Dict[str, Any]]:
        if isinstance(trades, list):
            raw_rows = trades
        elif isinstance(trades, dict):
            raw_rows = trades.get("data", trades.get("trades", []))
            if not isinstance(raw_rows, list):
                raw_rows = []
        else:
            raw_rows = []

        normalized = []
        for trade in raw_rows:
            if not isinstance(trade, dict):
                continue
            record = dict(trade)
            record["market_id"] = market_id or str(record.get("market_id") or condition_id or "").strip()
            if condition_id:
                record.setdefault("conditionId", condition_id)
            record.setdefault("taker_address", record.get("proxyWallet"))
            normalized.append(record)
        return normalized

    def fetch_market_trades(self, market_ref: Any, limit: int = 100) -> List[Dict[str, Any]]:
        market_id, condition_id = self._resolve_market_reference(market_ref)
        if not condition_id:
            logger.debug("Skipping Polymarket trade backfill without conditionId: %s", market_ref)
            return []

        if self.trade_source == "clob":
            try:
                payload = self._fetch_json(
                    f"{CLOB_API}/trades",
                    params={"market": condition_id, "limit": int(limit)},
                )
                return self._normalize_trade_rows(
                    payload,
                    market_id=market_id,
                    condition_id=condition_id,
                )
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status not in {401, 403}:
                    raise
                logger.info(
                    "Polymarket CLOB trade backfill requires auth for %s; falling back to public data-api /trades",
                    condition_id,
                )

        payload = self._fetch_json(
            f"{DATA_API}/trades",
            params={
                "market": condition_id,
                "limit": int(limit),
                "offset": 0,
                "takerOnly": str(self.trade_taker_only).lower(),
            },
        )
        return self._normalize_trade_rows(
            payload,
            market_id=market_id,
            condition_id=condition_id,
        )

    def backfill_markets(
        self,
        max_markets: int = 500,
        page_size: int = 100,
        observed_at_ms: Optional[int] = None,
    ) -> int:
        observed = int(observed_at_ms or _now_ms())
        total = 0
        offset = 0
        while total < max_markets:
            batch_limit = min(int(page_size), int(max_markets) - total)
            markets = self.fetch_gamma_markets(limit=batch_limit, offset=offset)
            if not markets:
                break
            total += store_markets(markets, observed_at_ms=observed)
            if len(markets) < batch_limit:
                break
            offset += len(markets)
        return total

    def backfill_recent_trades(
        self,
        market_ids: Iterable[Any],
        *,
        per_market: int = 50,
        max_markets: int = 25,
    ) -> int:
        total = 0
        seen = 0
        for market_ref in market_ids:
            if market_ref in (None, ""):
                continue
            if seen >= int(max_markets):
                break
            seen += 1
            trades = self.fetch_market_trades(market_ref, limit=per_market)
            if trades:
                total += store_trades(trades)
        return total

    def snapshot_label(self) -> str:
        return datetime.now(timezone.utc).strftime("polymarket_%Y%m%d_%H%M%S")
