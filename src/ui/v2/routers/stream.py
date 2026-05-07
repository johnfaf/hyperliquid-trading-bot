"""WebSocket endpoint that streams cycle events to connected clients.

Anyone visiting the dashboard opens one ``/ws`` connection. The bot's
trading cycle calls :func:`src.ui.v2.events.publish_event` and every
client receives the JSON. We don't replay history; new clients get a
single hello frame and pick up from the next event.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.ui.v2.auth import read_access_allowed
from src.ui.v2.events import get_bus

logger = logging.getLogger(__name__)

router = APIRouter()


def _heartbeat_seconds() -> float:
    raw = os.environ.get("DASHBOARD_V2_WS_HEARTBEAT_SECONDS", "15")
    try:
        return max(5.0, float(raw))
    except (TypeError, ValueError):
        return 15.0


@router.websocket("/ws")
async def stream_events(websocket: WebSocket):
    # Reuse the cookie auth so the WS doesn't get used as an auth bypass.
    if not read_access_allowed(websocket):
        logger.info("dashboard websocket rejected: auth_required")
        await websocket.close(code=4401)
        return
    await websocket.accept()
    queue = await get_bus().subscribe()
    heartbeat_s = _heartbeat_seconds()
    try:
        await websocket.send_json({"kind": "hello", "ok": True, "ts": time.time()})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
            except asyncio.TimeoutError:
                # Heartbeat -- prevents intermediate proxies from idling
                # us out, and lets the client confirm the channel.
                await websocket.send_json({"kind": "heartbeat", "ts": time.time()})
                continue
            await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("ws stream error: %s", exc)
    finally:
        await get_bus().unsubscribe(queue)
