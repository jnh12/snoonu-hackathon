from __future__ import annotations

import asyncio
from typing import Any, Set
from fastapi import WebSocket


class WSManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def send(self, ws: WebSocket, payload: Any) -> None:
        await ws.send_json(payload)

    async def broadcast(self, payload: Any) -> None:
        async with self._lock:
            clients = list(self._clients)

        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    # compat shims
    async def broadcast_json(self, payload: Any) -> None:
        await self.broadcast(payload)

    async def send_json(self, ws: WebSocket, payload: Any) -> None:
        await self.send(ws, payload)


ConnectionManager = WSManager
