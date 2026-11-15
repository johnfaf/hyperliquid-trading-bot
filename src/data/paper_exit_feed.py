"""Timestamped public BTC observations for an offline paper exit study."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass


SAMPLE_MS = 5_000
BOOK_MAX_AGE_MS = 10_000
MARK_MAX_AGE_MS = 15_000
TRADE_WINDOW_MS = 60_000


def finite(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def positive(value) -> float | None:
    value = finite(value)
    return value if value is not None and value > 0 else None


@dataclass(frozen=True)
class MarketSample:
    at_ms: int
    status: str
    mark: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    depth_imbalance: float | None = None
    signed_flow_imbalance: float | None = None
    funding_rate: float | None = None
    oracle_price: float | None = None
    open_interest: float | None = None
    book_age_ms: int | None = None
    context_age_ms: int | None = None
    depth_age_ms: int | None = None
    quote_source: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.status == "ready" and positive(self.mark) is not None and positive(self.bid) is not None
            and positive(self.ask) is not None and self.ask >= self.bid and positive(self.bid_size) is not None
            and self.book_age_ms is not None and -2_000 <= self.book_age_ms <= BOOK_MAX_AGE_MS
            and self.context_age_ms is not None and 0 <= self.context_age_ms <= MARK_MAX_AGE_MS
        )

    @property
    def spread_bps(self) -> float | None:
        if self.ready:
            return (self.ask - self.bid) / ((self.ask + self.bid) / 2.0) * 10_000
        return None


@dataclass(frozen=True)
class _Book:
    ms: int
    bid: float
    ask: float
    bid_size: float
    depth_imbalance: float | None = None


@dataclass(frozen=True)
class _Context:
    received_ms: int
    mark: float
    funding: float | None
    oracle: float | None
    open_interest: float | None


def epoch_ms() -> int:
    return time.time_ns() // 1_000_000


class PaperExitFeed:
    def __init__(self, coin: str, clock=epoch_ms):
        self._coin = coin
        self._clock = clock
        self._lock = threading.Lock()
        self.reset()

    def make_stream(self):
        from src.core.api_manager import HAS_WEBSOCKET, HyperliquidWebSocket

        if not HAS_WEBSOCKET:
            raise RuntimeError("websocket-client is required for public capture")
        feed = self

        class _ResearchStream(HyperliquidWebSocket):
            def _on_open(self, ws):
                feed.reset()
                super()._on_open(ws)
                for channel in ("bbo", "activeAssetCtx"):
                    ws.send(json.dumps({
                        "method": "subscribe", "subscription": {"type": channel, "coin": feed._coin},
                    }))

        stream = _ResearchStream()
        stream.on_message("l2Book", self.on_book)
        stream.on_message("bbo", self.on_bbo)
        stream.on_message("activeAssetCtx", self.on_context)
        stream.on_message("trades", self.on_trades)
        stream.subscribe_coin(self._coin)
        return stream

    def reset(self):
        with self._lock:
            self._book: _Book | None = None
            self._bbo: _Book | None = None
            self._context: _Context | None = None
            self._trades: dict[tuple[int, int], float] = {}
            self._flow_starts_ms = self._clock()

    def on_book(self, message):
        data = message.get("data")
        if not isinstance(data, dict) or data.get("coin") != self._coin:
            return
        book = None
        try:
            timestamp = data["time"]
            sides = data["levels"]
            if type(timestamp) is int and -2_000 <= self._clock() - timestamp <= BOOK_MAX_AGE_MS:
                bid, ask = (positive(side[0]["px"]) for side in sides)
                bid_size = positive(sides[0][0]["sz"])
                buy_depth, sell_depth = (
                    math.fsum(float(row["px"]) * float(row["sz"]) for row in side[:5]) for side in sides
                )
                if (bid is not None and ask is not None and ask >= bid and bid_size is not None
                        and positive(buy_depth) is not None and positive(sell_depth) is not None):
                    imbalance = (buy_depth - sell_depth) / (buy_depth + sell_depth)
                    book = _Book(timestamp, bid, ask, bid_size, imbalance)
        except (KeyError, IndexError, ValueError, TypeError, OverflowError):
            pass
        with self._lock:
            if book is None:
                self._book = None
            elif self._book is None or book.ms > self._book.ms:
                self._book = book

    def on_bbo(self, message):
        data = message.get("data")
        if not isinstance(data, dict) or data.get("coin") != self._coin:
            return
        book = None
        try:
            timestamp = data["time"]
            top_bid, top_ask = data["bbo"]
            if type(timestamp) is int and -2_000 <= self._clock() - timestamp <= BOOK_MAX_AGE_MS:
                bid, ask, size = positive(top_bid["px"]), positive(top_ask["px"]), positive(top_bid["sz"])
                if bid is not None and ask is not None and ask >= bid and size is not None:
                    book = _Book(timestamp, bid, ask, size)
        except (KeyError, ValueError, TypeError, OverflowError):
            pass
        with self._lock:
            if book is None:
                self._bbo = None
            elif self._bbo is None or book.ms > self._bbo.ms:
                self._bbo = book

    def on_context(self, message):
        data = message.get("data")
        if not isinstance(data, dict) or data.get("coin") != self._coin:
            return
        ctx = data.get("ctx")
        if not isinstance(ctx, dict):
            ctx = {}
        mark = positive(ctx.get("markPx"))
        with self._lock:
            # No venue timestamp is provided on this channel; record local receipt.
            self._context = _Context(
                self._clock(), mark, finite(ctx.get("funding")), positive(ctx.get("oraclePx")),
                positive(ctx.get("openInterest")),
            ) if mark is not None else None

    def on_trades(self, message):
        data = message.get("data")
        if not isinstance(data, list):
            return
        now = self._clock()
        with self._lock:
            for trade in data:
                if not isinstance(trade, dict) or trade.get("coin") != self._coin:
                    continue
                timestamp, tid, side = trade.get("time"), trade.get("tid"), trade.get("side")
                if (type(timestamp) is not int or type(tid) is not int
                        or not 0 <= now - timestamp <= TRADE_WINDOW_MS or side not in ("A", "B")):
                    continue
                price, size = positive(trade.get("px")), positive(trade.get("sz"))
                if price is not None and size is not None and math.isfinite(price * size):
                    self._trades.setdefault((timestamp, tid), price * size * (1 if side == "B" else -1))
            if len(self._trades) > 50_000:
                self._trades.clear()
                self._flow_starts_ms = now

    def snapshot(self, connected: bool) -> MarketSample:
        with self._lock:
            now = self._clock()
            if not connected:
                return MarketSample(now, "disconnected")
            depth, ctx = self._book, self._context
            book, source = max(
                ((depth, "l2Book"), (self._bbo, "bbo")), key=lambda item: item[0].ms if item[0] else -1,
            )
            book_age = now - book.ms if book else None
            depth_age = now - depth.ms if depth else None
            context_age = now - ctx.received_ms if ctx else None
            if book_age is None or not -2_000 <= book_age <= BOOK_MAX_AGE_MS:
                book = None
                source = None
            if depth_age is None or not -2_000 <= depth_age <= BOOK_MAX_AGE_MS:
                depth = None
            if context_age is None or not 0 <= context_age <= MARK_MAX_AGE_MS:
                ctx = None
            self._trades = {key: value for key, value in self._trades.items() if now - key[0] <= TRADE_WINDOW_MS}
            flow = None
            if now - self._flow_starts_ms >= TRADE_WINDOW_MS and self._trades:
                flow = math.fsum(self._trades.values()) / math.fsum(abs(v) for v in self._trades.values())
            return MarketSample(
                at_ms=now,
                status="book_stale" if book is None else "mark_stale" if ctx is None else "ready",
                mark=ctx.mark if ctx else None,
                bid=book.bid if book else None,
                ask=book.ask if book else None,
                bid_size=book.bid_size if book else None,
                depth_imbalance=depth.depth_imbalance if depth else None,
                signed_flow_imbalance=flow,
                funding_rate=ctx.funding if ctx else None,
                oracle_price=ctx.oracle if ctx else None,
                open_interest=ctx.open_interest if ctx else None,
                book_age_ms=book_age,
                context_age_ms=context_age,
                depth_age_ms=depth_age,
                quote_source=source,
            )
