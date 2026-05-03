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

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.ui.v2.auth import verify_cookie
from src.ui.v2.events import get_bus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def stream_events(websocket: WebSocket):
    # Reuse the cookie auth so the WS doesn't get used as an auth bypass.
    if not verify_cookie(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    queue = await get_bus().subscribe()
    try:
        await websocket.send_json({"kind": "hello", "ok": True})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Heartbeat -- prevents intermediate proxies from idling
                # us out, and lets the client confirm the channel.
                await websocket.send_json({"kind": "heartbeat"})
                continue
            await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("ws stream error: %s", exc)
    finally:
        await get_bus().unsubscribe(queue)
