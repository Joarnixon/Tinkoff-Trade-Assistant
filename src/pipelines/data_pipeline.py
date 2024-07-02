import polars as pl
from omegaconf import DictConfig
from typing import Optional

from src.data.data_preprocess import DataPreprocessor
from src.data.data_validation import validate, validate_online
from src.features.feature_engineering import BuildFeatures

class DataPipeline:
    def __init__(self, steps: list):
        self.steps = steps

    def transform(self, data: pl.DataFrame) -> Optional[pl.DataFrame]:
        for transformer in self.steps:
            data = transformer(data)
            if data is None:
                return None
        return data

class DataPipelineFactory:
    @staticmethod
    def create_online_pipeline(cfg: DictConfig) -> DataPipeline:
        return DataPipeline([
            validate_online,
            DataPreprocessor(cfg),
            BuildFeatures(cfg),
        ])
        

    @staticmethod
    def create_offline_pipeline(cfg: DictConfig) -> DataPipeline:
        return DataPipeline([
            validate,
            DataPreprocessor(cfg),
            BuildFeatures(cfg),
        ])
        