import os
import time
from collections import defaultdict
import json
import numpy as np
from omegaconf import OmegaConf
from polars import DataFrame
from torch.utils.tensorboard import SummaryWriter
from src.data.data_validation import validate_online
from server.routers.shares import model_available

class OnlineMetric:
    def __init__(self, cfg):
        self.figi = {figi: True for figi in OmegaConf.load(str(cfg.paths.shares_dict)) if model_available(figi, cfg.paths.models)}
        self.path_to_metadata = cfg.paths.online_metric_metadata
        self.percent_change_threshold = float(cfg.data.data_labeling.percent_change_threshold)
        self.time_column = cfg.data.data_labeling.time_column_name
        self.price_column = cfg.data.data_labeling.price_column_name
        self.min_time_interval = int(cfg.data.data_labeling.min_time_interval)
        self.max_time_interval = int(cfg.data.data_labeling.max_time_interval)
        self.df_schema = cfg.data.raw_data_trades_columns + [f"{column}_{proc}" for column in cfg.data.raw_data_orderbook_columns for proc in (cfg.data.data_gather.orderbook_processes + ['last'])]

        self.data_buffer = defaultdict(lambda: {"time": [], "price": []})
        self.prediction_buffer = defaultdict(list) 
        self.scores: dict[str, int] = {figi: 0 for figi in self.figi}
        self.steps: dict[str, int] = {figi: 0 for figi in self.figi}
        # Load previous state from disk
        self._load_state()
        self.writers: dict[SummaryWriter] = {}


    def _load_state(self):
        try:
            path = os.path.join(self.path_to_metadata, 'metadata.json')
            with open(path, 'r') as f:
                state = json.load(f)
                self.scores.update(state.get('scores', self.scores))
                self.steps.update(state.get('steps', self.steps))
        except FileNotFoundError:
            pass

    def save(self):
        try:
            path = os.path.join(self.path_to_metadata, 'metadata.json')
            state = {'scores': self.scores, 'steps': self.steps}
            with open(path, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Error saving state: {e}")


    async def update(self, message: dict):
        for figi, data in message.items():
            if figi in self.figi:
                if len(data) == 1:
                    data = DataFrame(data, schema=self.df_schema)
                    data = validate_online(data)
                    if data is not None:
                        await self._update_data(figi, data)
                else:
                    predicted_class, predicted_proba = data.values()
                    predicted_class = int(predicted_class[0])
                    predicted_proba = predicted_proba[0][predicted_class]
                    await self._update_prediction(figi, predicted_class, predicted_proba)


    async def _update_data(self, figi: str, data: DataFrame):
        """Handles incoming data and stores it."""
        self.data_buffer[figi]["time"].append(data[self.time_column].item())
        self.data_buffer[figi]["price"].append(data[self.price_column].item())

    async def _update_prediction(self, figi: str, prediction: int, prediction_proba: np.ndarray):
        """Handles incoming predictions, calculates labels if possible, and updates scores."""
        self.prediction_buffer[figi].append(prediction)  # Store the prediction
        # Check if enough data exists to calculate a label
        if self.data_buffer[figi]["time"][-1] - self.data_buffer[figi]["time"][0] >= self.min_time_interval: 
            label = self._calculate_label(
                np.array(self.data_buffer[figi]["time"]),
                np.array(self.data_buffer[figi]["price"]),
            )
            if self.prediction_buffer[figi]:
                await self._update_score(label, self.prediction_buffer[figi][0], figi)
            self._trim_data(figi)

        if figi not in self.writers:
            self.writers[figi] = SummaryWriter(log_dir=f'runs/online/{figi}')
      
    def _calculate_label(self, time_array: np.ndarray, price_array: np.ndarray) -> int:
        """Calculates the label based on price changes over time."""
        start_price = price_array[0]
        for i in range(1, len(time_array)):
            time_diff = time_array[i] - time_array[0]
            if self.min_time_interval <= time_diff <= self.max_time_interval:
                end_price = price_array[i]
                percent_change = (end_price - start_price) / start_price * 100
                if percent_change > self.percent_change_threshold:
                    return 1  # Buy
                elif -self.percent_change_threshold <= percent_change <= self.percent_change_threshold:
                    return 0  # Hold
                else:
                    return 2  # Sell
            elif time_diff > self.max_time_interval:
                return 0

    async def _update_score(self, label: int, prediction: int, figi):
        """Updates the score based on the label and prediction."""
        if label == prediction:
            if label != 0:
                self.scores[figi] += 1
            else:
                self.scores[figi] += 0
        else:
            self.scores[figi] -= 1

        self.writers[figi].add_scalar("accuracy", self.scores[figi], self.steps[figi]) 
        self.steps[figi] += 1 
                
    def _trim_data(self, figi: str):
        self.data_buffer[figi]["time"].pop(0)
        self.data_buffer[figi]["price"].pop(0)
        if self.prediction_buffer[figi]:
            self.prediction_buffer[figi].pop(0) 