import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import hydra
from omegaconf import OmegaConf
import json
import random as rd
from contextlib import asynccontextmanager
from src.data import DataCollector
from src.pipelines.online_pipeline import OnlinePredictions
from src.data import DataManager
from volume_scanner import VolumeScan
from server.routers import chart, shares
import logging

from src.data.subject import Subject


log = logging.getLogger("tinkoff.invest.logging")
log.setLevel(logging.WARNING)

class TestOnlinePredictions(Subject):
    def __init__(self, cfg, figi):
        super(TestOnlinePredictions, self).__init__()
        self.cfg = cfg
        self.figi = 'BBG004730N88'
    
    async def run(self):
        while True:
            await asyncio.sleep(40)
            self.notify({self.figi: (rd.randint(0, 2), 1)})

@asynccontextmanager
async def lifespan(app: FastAPI):
    hydra.initialize(version_base=None, config_path="config", job_name="test")
    cfg = hydra.compose(config_name="general.yaml", return_hydra_config=True)
    figi = 'TCS00A106YF0'

    all_shares = OmegaConf.load(str(cfg.paths.shares_dict))

    dm = DataManager(cfg)
    dc = DataCollector(cfg)
    op = TestOnlinePredictions(cfg, figi)
    vs = VolumeScan(cfg, dm)

    connections = Connections(all_shares)
    app.state.connections = connections
    app.state.all_shares = all_shares

    dc.attach(op)
    op.attach(connections)

    async def background_tasks():
        await asyncio.gather(dc.run(), op.run(), vs.run())

    app.state.volume_scanner = vs
    app.state.cfg = cfg
    app.state.background_tasks = asyncio.create_task(background_tasks())
    
    yield

    app.state.background_tasks.cancel()

app = FastAPI(lifespan=lifespan)

class Connections:
    def __init__(self, all_shares, max_subscriptions_per_client: int = 3):
        self._subscriptions: dict[WebSocket, set[str]] = {} # websocket to the set of subscribed
        self._subscribed_to_figi: dict[str, set[WebSocket]] = {figi: set([]) for figi in all_shares} # figi to everyone subscribed
        self.max_subscriptions_per_client = max_subscriptions_per_client

    async def add(self, connection: WebSocket, client_id: str) -> None:
        self._subscriptions[connection] = set([])

    async def close(self, websocket) -> None:
        del self._subscriptions[websocket]

    async def update(self, message: dict) -> None:
        for figi, data in message.items():
            for listener in self._subscribed_to_figi[figi]:
                try:
                    await listener.send_json(json.dumps({figi: data}))
                except RuntimeError:
                    pass

    async def subscribe(self, websocket: str, figi: str) -> bool:
        if len(self._subscriptions[websocket]) < self.max_subscriptions_per_client:
            self._subscriptions[websocket].add(figi)
            self._subscribed_to_figi[figi].add(websocket)
            return True
        else:
            return False

    async def unsubscribe(self, websocket: str, figi: str) -> None:
        self._subscriptions[websocket].discard(figi)
        self._subscribed_to_figi[figi].discard(websocket)
        return True

@app.get("/")
async def get():
    return None

@app.websocket("/ws/{client_id}")
async def ws_endpoint(websocket: WebSocket, client_id: str) -> None:
    await websocket.accept()
    await app.state.connections.add(websocket, client_id)
    # prediction_task = asyncio.create_task(recieve_prediction(connections))
    print(f"Client {client_id} connected")
    try:
        while True: 
            data = await websocket.receive_json()
            figi = data.get('FIGI', None)
            state = data.get("STATE", None)
            if figi is not None and state is not None:
                if state:
                    await app.state.connections.subscribe(websocket, figi)
                else:
                    await app.state.connections.unsubscribe(websocket, figi)
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
        await app.state.connections.close(websocket)

app.include_router(chart.router)
app.include_router(shares.router)

