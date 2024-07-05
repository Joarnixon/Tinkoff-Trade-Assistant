import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Dict, List
import numpy as np
import torch
import os
from polars import DataFrame
from src.model.load_model import load_nn
from src.model.baseline import LinearNN
from src.pipelines.data_pipeline import DataPipelineFactory
from src.data.subject import Subject
from src.features.feature_selection import SelectFeatures

class OnlinePredictions(Subject):
    def __init__(self, cfg, figi=None):
        super().__init__()
        self.cfg = cfg
        self.figi = figi
        self.df_schema = self.cfg.data.raw_data_trades_columns + [f"{column}_{proc}" for column in self.cfg.data.raw_data_orderbook_columns for proc in (self.cfg.data.data_gather.orderbook_processes + ['last'])]
        self.data_pipeline = DataPipelineFactory.create_online_pipeline(cfg)
        self.feature_selector = SelectFeatures(cfg)
        self.models: Dict[str, object] = {}  # Dictionary to store models for each share
        self.data_queue = asyncio.Queue()  # Queue to store incoming data points
        self.prediction_queue = queue.Queue()  # Queue to store predictions
        self.max_queue_size = cfg.get('max_queue_size', 10)
        self.batch_size = cfg.get('batch_size', 32)
        self.executor = ThreadPoolExecutor(max_workers=cfg.get('max_workers', 4))
        
        self._load_models(self.figi)

    async def run(self):
        await asyncio.gather(
            self._process_data(),
            self._make_predictions()
        )

    def _load_models(self, figi):
        filename = os.path.join(self.cfg.paths.models, figi, 'best.pth')
        model = load_nn(filename)
        self.models[figi] = model

    async def update(self, data_point: Dict[str, np.ndarray]):
        if self.data_queue.qsize() < self.max_queue_size:
            recieved_figi = next(iter(data_point))
            if self.figi is None or recieved_figi == self.figi:
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
                    data_point = await asyncio.wait_for(self.data_queue.get(), timeout=0.1)
                    batch.append(data_point)
                except asyncio.TimeoutError:
                    break  # Process partial batch if no new data arrives within timeout
            
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
                print(f"Predictions: {predictions}")
                self.notify_observers(predictions)

    async def _predict_batch(self, batch: List[tuple]):
        predictions = {}
        futures = []
        for figi, features in batch:
            model = self.models[figi]
            futures.append((figi, self.executor.submit(self._predict_single, model, features)))
        
        for figi, future in futures:
            prediction = await asyncio.to_thread(future.result)
            predictions[figi] = prediction
        
        return predictions

    def _predict_single(self, model, features):
        return (features, model.predict(np.array([features])))

    def notify_observers(self, predictions):
        for observer in self._observers:
            observer.update(predictions)