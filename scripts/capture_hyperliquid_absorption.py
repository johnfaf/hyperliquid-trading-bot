#!/usr/bin/env python3
"""Capture public BTC/ETH trades and order-book frames without API credentials."""
from argparse import ArgumentParser
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time

from src.data.absorption_feed import COINS, Capture, storage

FEED = "wss://api.hyperliquid.xyz/ws"
log = logging.getLogger("absorption_capture")


def now_ms():
    return time.time_ns() // 1_000_000


def run(path, seconds=None):
    import websocket

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    deadline = time.monotonic() + seconds if seconds is not None else float("inf")
    with closing(storage(path)) as db:
        backoff = 2
        while not stop.is_set() and time.monotonic() < deadline:
            ws = None
            try:
                ws = websocket.create_connection(FEED, timeout=10)
                ws.settimeout(1)
                capture = Capture(db, now_ms())
                for coin in COINS:
                    for channel in ("l2Book", "trades"):
                        subscription = {"type": channel, "coin": coin}
                        if channel == "l2Book":
                            subscription["fast"] = True
                        ws.send(json.dumps({"method": "subscribe", "subscription": subscription}))
                log.info("Public book/trade session connected for %s", ", ".join(COINS))
                next_ping = time.monotonic() + 20
                last_market = time.monotonic()
                while not stop.is_set() and time.monotonic() < deadline:
                    if time.monotonic() - last_market > 12:
                        raise ConnectionError("public market stream stalled")
                    if time.monotonic() >= next_ping:
                        ws.send('{"method":"ping"}')
                        next_ping = time.monotonic() + 20
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        capture.advance(now_ms())
                        continue
                    received = now_ms()
                    if not raw:
                        raise ConnectionError("public stream closed")
                    message = json.loads(raw)
                    if not isinstance(message, dict) or message.get("channel") == "error":
                        raise ValueError("unexpected stream response")
                    capture.ingest(message, received)
                    if message.get("channel") in {"l2Book", "trades"}:
                        last_market = time.monotonic()
                    backoff = 2
            except (OSError, ValueError, websocket.WebSocketException) as exc:
                log.warning("Public stream reset (%s); new session after reconnect", type(exc).__name__)
            finally:
                if ws is not None:
                    ws.close(timeout=1)
            if time.monotonic() < deadline:
                stop.wait(min(backoff, deadline - time.monotonic()))
            backoff = min(backoff * 2, 30)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--seconds", type=float, help="Run a bounded probe instead of collecting continuously")
    args = parser.parse_args()
    if args.seconds is not None and not 1 <= args.seconds <= 3600:
        parser.error("probe duration must be 1-3600 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.umask(0o077)
    run(args.db.resolve(), args.seconds)


if __name__ == "__main__":
    main()
