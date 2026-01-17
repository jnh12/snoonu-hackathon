from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import asyncio

from .ws_manager import ConnectionManager
from .sim_engine import SimEngine


@dataclass
class RunnerConfig:
    tick_ms: int = 500
    dt_sec: int = 10


class SimRunner:
    def __init__(self, sim: SimEngine, manager: ConnectionManager, sim_lock: asyncio.Lock):
        self.sim = sim
        self.manager = manager
        self.sim_lock = sim_lock

        self.running: bool = False
        self.cfg = RunnerConfig()
        self._task: Optional[asyncio.Task] = None

    async def start(self, tick_ms: Optional[int] = None, dt_sec: Optional[int] = None):
        if tick_ms is not None:
            self.cfg.tick_ms = max(50, int(tick_ms))
        if dt_sec is not None:
            self.cfg.dt_sec = max(1, int(dt_sec))

        if self.running and self._task and not self._task.done():
            return

        self.running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _loop(self):
        while self.running:
            await asyncio.sleep(self.cfg.tick_ms / 1000.0)

            if not self.sim.loaded:
                continue

            async with self.sim_lock:
                self.sim.step(dt_sec=self.cfg.dt_sec)
                snap = self.sim.snapshot()

            await self.manager.broadcast_json(snap)

    async def push_state(self):
        if not self.sim.loaded:
            return
        async with self.sim_lock:
            snap = self.sim.snapshot()
        await self.manager.broadcast_json(snap)
