from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
import asyncio
from typing import Set

from .config import SimConfig
from .loader import load_orders, load_couriers
from .sim_engine import SimEngine, Policy
from .router import make_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Snoonu Hackathon Backend", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = SimConfig()
router = make_router(cfg)
sim = SimEngine(cfg, router)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_ORDERS = os.path.join(DATA_DIR, "doha_test_orders_50.csv")
DEFAULT_COURIERS = os.path.join(DATA_DIR, "doha_couriers_50.csv")

# ---- websocket clients ----
_ws_clients: Set[WebSocket] = set()

# ---- live sim runner ----
_sim_running = False
_last_ws_push = 0.0


class LoadBody(BaseModel):
    orders_path: str | None = None
    couriers_path: str | None = None


class SimStartBody(BaseModel):
    policy: Policy = "baseline"
    start_time_sec: int | None = None


class SimStepBody(BaseModel):
    dt_sec: int = 10


class SimRunBody(BaseModel):
    seconds: int = 300
    dt_sec: int = 10


class SpawnOrdersBody(BaseModel):
    n: int = 5


class SpawnCouriersBody(BaseModel):
    n: int = 3


@app.get("/health")
def health():
    return {"ok": True, "loaded": sim.loaded, "version": app.version}


@app.post("/load")
def load_dataset(body: LoadBody):
    orders_path = body.orders_path or DEFAULT_ORDERS
    couriers_path = body.couriers_path or DEFAULT_COURIERS

    if not os.path.exists(orders_path):
        raise HTTPException(status_code=400, detail=f"Orders CSV not found: {orders_path}")
    if not os.path.exists(couriers_path):
        raise HTTPException(status_code=400, detail=f"Couriers CSV not found: {couriers_path}")

    orders = load_orders(orders_path)
    couriers = load_couriers(couriers_path)
    sim.load(orders, couriers)
    return {"loaded": True, "orders": len(orders), "couriers": len(couriers)}


@app.post("/reset")
def reset():
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    sim.reset(policy="baseline", start_time=None)
    return {"ok": True}


@app.get("/state")
def get_state():
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    return sim.snapshot()


@app.get("/kpis")
def get_kpis():
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    return sim.kpis()


@app.post("/sim/start")
def sim_start(body: SimStartBody):
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    sim.reset(policy=body.policy, start_time=body.start_time_sec)
    return sim.snapshot()


@app.post("/sim/step")
def sim_step(body: SimStepBody):
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    sim.step(dt_sec=body.dt_sec)
    return sim.snapshot()


@app.post("/sim/run")
def sim_run(body: SimRunBody):
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    sim.run(seconds=body.seconds, dt_sec=body.dt_sec)
    return sim.snapshot()


# ---- Live controls ----
@app.post("/sim/play")
def sim_play():
    global _sim_running
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    _sim_running = True
    return {"running": True}


@app.post("/sim/pause")
def sim_pause():
    global _sim_running
    _sim_running = False
    return {"running": False}


# ---- Random spawns ----
@app.post("/spawn/orders")
def spawn_orders(body: SpawnOrdersBody):
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    sim.spawn_random_orders(body.n)
    return {"ok": True, "added": body.n}


@app.post("/spawn/couriers")
def spawn_couriers(body: SpawnCouriersBody):
    if not sim.loaded:
        raise HTTPException(status_code=400, detail="Dataset not loaded")
    sim.spawn_random_couriers(body.n)
    return {"ok": True, "added": body.n}


# ---- WebSocket ----
@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        if sim.loaded:
            await ws.send_json(sim.snapshot())
        else:
            await ws.send_json({"detail": "Dataset not loaded"})
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


async def _broadcast_state():
    if not _ws_clients:
        return
    snap = sim.snapshot() if sim.loaded else {"detail": "Dataset not loaded"}
    dead = []
    for c in list(_ws_clients):
        try:
            await c.send_json(snap)
        except Exception:
            dead.append(c)
    for c in dead:
        _ws_clients.discard(c)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_ticker())


async def _ticker():
    global _last_ws_push
    while True:
        await asyncio.sleep(0.05)
        if sim.loaded and _sim_running:
            sim.step(dt_sec=cfg.live_tick_dt_sec)

        now = asyncio.get_event_loop().time()
        if sim.loaded and (now - _last_ws_push) >= cfg.ws_push_interval_sec:
            _last_ws_push = now
            await _broadcast_state()
