import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Dict, List
import numpy as np
import torch
import os
from polars import DataFrame
from omegaconf import OmegaConf
from server.routers.shares import model_available
from src.model.baseline import LinearNN
from src.model.load_model import ModelLoader
from src.pipelines.data_pipeline import DataPipelineFactory
from src.data.subject import Subject
from src.features.feature_selection import SelectFeatures

class OnlinePredictions(Subject):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.figi = {figi: True for figi in OmegaConf.load(str(cfg.paths.shares_dict)) if model_available(figi, cfg.paths.models)}
        self.df_schema = self.cfg.data.raw_data_trades_columns + [f"{column}_{proc}" for column in self.cfg.data.raw_data_orderbook_columns for proc in (self.cfg.data.data_gather.orderbook_processes + ['last'])]
        self.data_pipeline = DataPipelineFactory.create_online_pipeline(cfg)
        self.feature_selector = SelectFeatures(cfg)
        self.loader = ModelLoader(cfg, k=2)
        self.data_queue = asyncio.Queue()  # Queue to store incoming data points
        self.prediction_queue = queue.Queue()  # Queue to store predictions
        self.max_queue_size = cfg.get('max_queue_size', 10)
        self.batch_size = cfg.get('batch_size', 32)
        self.executor = ThreadPoolExecutor(max_workers=cfg.get('max_workers', 4))
        
    async def run(self):
        await asyncio.gather(
            self._process_data(),
            self._make_predictions()
        )

    def get_model(self, figi):
        return self.loader.load(figi)

    async def update(self, data_point: Dict[str, np.ndarray]):
        if self.data_queue.qsize() < self.max_queue_size:
            recieved_figi = next(iter(data_point))
            if self.figi is None or recieved_figi in self.figi:
                data_point[recieved_figi] = DataFrame(data_point[recieved_figi], schema=self.df_schema)
                await self.data_queue.put(data_point)
        else:
            print("Warning: Data queue is full. Skipping data point.")

    async def _process_data(self):
        while True:
            await asyncio.sleep(1)
            batch = []
            while len(batch) < self.batch_size:
                try:
                    data_point = await asyncio.wait_for(self.data_queue.get(), timeout=0.5)
                    batch.append(data_point)
                except asyncio.TimeoutError:
                    break
            if batch:
                processed_batch = await self._preprocess_batch(batch)
                self.prediction_queue.put(processed_batch)

    async def _preprocess_batch(self, batch: List[Dict[str, np.ndarray]]):
        processed_batch = []
        for data_point in batch:
            for figi, data in data_point.items():
                processed_data = self.data_pipeline.transform(data)
                if processed_data is not None:
                    selected_features = self.feature_selector.fit_transform_online(processed_data, figi)
                    processed_batch.append((figi, selected_features))
        return processed_batch

    async def _make_predictions(self):
        while True:
            await asyncio.sleep(1)
            if not self.prediction_queue.empty():
                batch = self.prediction_queue.get()
                predictions = await self._predict_batch(batch)
                self.notify(predictions)

    async def _predict_batch(self, batch: List[tuple]):
        predictions = {}
        futures = []
        for figi, features in batch:
            model = self.get_model(figi)
            futures.append((figi, self.executor.submit(self._predict_single, model, features)))
        
        for figi, future in futures:
            prediction = await asyncio.to_thread(future.result)
            predictions[figi] = prediction
        
        return predictions

    def _predict_single(self, model, features: DataFrame):
        prediction, proba = model.predict(np.array(features))
        return {'result': prediction.tolist(), 'probability': proba.tolist()}
