import numpy as np

class OrderBookPreprocessor:
    def __init__(self, processes):
        self.pipeline = [getattr(self, proc) for proc in processes] + [self.last]
        
    def transform(self, data: list[dict[int, float]]) -> list[float]:
        result = []
        for feature_dict in data:
            for func in self.pipeline:
                result.append(func(feature_dict))
        return result

    def last(self, feature_dict: dict[int, float]) -> float:
        return next(reversed(feature_dict.values()), np.nan)

    def min(self, feature_dict: dict[int, float]) -> float:
        return min(feature_dict.values(), default=np.nan)

    def max(self, feature_dict: dict[int, float]) -> float:
        return max(feature_dict.values(), default=np.nan)
    
    def std(self, feature_dict: dict[int, float]) -> float:
        return np.std(list(feature_dict.values())) if feature_dict else np.nan
    
class DataPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, data):
        return data
    
    
