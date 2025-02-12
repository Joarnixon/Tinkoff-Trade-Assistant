import polars as pl
import numpy as np
from numpy import array
from numba import njit
from typing import Optional
from src.data.data_manager import DataManager, NullDataManager

class DataLabeler:
    def __init__(self, cfg):
        self.percent_change_threshold = float(cfg.data.data_labeling.percent_change_threshold)
        self.time_column = cfg.data.data_labeling.time_column_name
        self.price_column = cfg.data.data_labeling.price_column_name
        self.min_time_interval = int(cfg.data.data_labeling.min_time_interval)
        self.max_time_interval = int(cfg.data.data_labeling.max_time_interval)
        self.label_meaning = dict(cfg.data.data_labeling.label_meaning)

    @staticmethod
    @njit
    def _calculate_price_changes(time_array, price_array, percent_change_threshold, min_time_interval, max_time_interval):
        labels = np.full(len(time_array), np.nan)
        n = len(time_array)
        for i in range(n):
            start_time = time_array[i]
            start_price = price_array[i]
            
            for j in range(i + 1, n):
                time_diff = time_array[j] - start_time
                if time_diff >= max_time_interval:
                    break
                end_price = price_array[j]
                percent_change = (end_price - start_price) / start_price * 100
            
                if abs(percent_change) > percent_change_threshold:
                    labels[i] = 1 if percent_change > 0 else 2
                    break
                elif time_diff >= min_time_interval:
                    labels[i] = 0  # No significant change within min_time_interval
                    break
        return labels

    def fit(self, dataset) -> array:
        """Make labels from the dataset. Size of the dataset would change."""
        time_array = dataset[self.time_column].to_numpy()
        price_array = dataset[self.price_column].to_numpy()
        
        labels = self._calculate_price_changes(time_array, price_array, 
                                               self.percent_change_threshold,
                                               self.min_time_interval, 
                                               self.max_time_interval)
        
        valid_mask = ~np.isnan(labels)
        
        filtered_dataset = dataset.filter(pl.Series(valid_mask))
        labels = labels[valid_mask]
        
        return filtered_dataset, labels