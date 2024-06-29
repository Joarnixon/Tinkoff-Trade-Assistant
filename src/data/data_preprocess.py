import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from src.data.data_manager import DataManager, NullDataManager
from src.data.observer import Observer
from typing import Optional
from scipy import stats
import joblib
import os

class OrderBookPreprocessor:
    def __init__(self, processes):
        self.pipeline = [getattr(self, proc) for proc in processes]
        
    def transform(self, data: list[dict[int, float]]) -> list[float]:
        result = []
        for feature_dict in data:
            for func in self.pipeline:
                result.append(func(feature_dict))
        return result

    def last(self, feature_dict: dict[int, float]) -> float:
        last_timestamp = max(feature_dict.keys())
        return feature_dict[last_timestamp]

    def min(self, feature_dict: dict[int, float]) -> float:
        return min(feature_dict.values())

    def max(self, feature_dict: dict[int, float]) -> float:
        return max(feature_dict.values())
    
    def std(self, feature_dict: dict[int, float]) -> float:
        return np.std(list(feature_dict.values()))
    
class DataPreprocessor:
    def __init__(self, cfg, data_manager: Optional[DataManager]):
        self.cfg = cfg
        self.data_manager = data_manager if data_manager is not None else NullDataManager()
    
    def transform(self, data):
        pass
    
    
