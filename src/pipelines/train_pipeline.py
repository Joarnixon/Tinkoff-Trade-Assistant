from src.data.data_manager import DataManager, NullDataManager
from src.data.data_labeling import DataLabeler
from src.features.feature_selection import SelectFeatures
from src.train import train_ml
from src.validate import val_ml
from src.model.base_model import Model
from src.model.baseline import LinearModel
from typing import Union, Optional
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import polars as pl

class TrainPipeline:
    def __init__(self, model: Model, train_func: callable, validate_func: callable, data_manager: Union[DataManager, NullDataManager], cfg: DictConfig, logger, monitor=None):
        self.model = model
        self.train_func = train_func
        self.validate_func = validate_func
        self.data_manager = data_manager
        self.logger = logger
        self.monitor = monitor
        self.random_state = cfg.random_state
        self.test_size = cfg.test_size
        self.feature_selection = SelectFeatures(cfg, data_manager)
        self.data_labeler = DataLabeler(cfg)

    def train(self, data: pl.DataFrame, figi: str):
        data, labels = self.data_labeler.fit(data)
        self.logger.info(f'Data shape: {data.shape}, "Buy" labels count: {len(labels[labels==1])}, "Sell" labels count: {len(labels[labels==-1])}')
        data = self.feature_selection.fit(data, labels, figi)
        self.data_manager.write_processed_share(figi, data, labels)
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=self.test_size, random_state=self.random_state)

        model, logs = self.train_func(self.model, data_train, labels_train, self.logger, self.monitor)

        validation_results = self.validate_func(model, data_val, labels_val, self.logger)

        self.data_manager.save_model(model, figi)

        return model, logs, validation_results


class TrainPipelineFactory:
    @staticmethod
    def create_ml_pipeline(cfg: DictConfig, model: Model, logger, data_manager: Optional[DataManager] = None) -> TrainPipeline:
        return TrainPipeline(
            model,
            train_ml,
            val_ml,
            data_manager if data_manager is not None else NullDataManager(),
            cfg,
            logger,                   
        )
        

    @staticmethod
    def create_nn_pipeline(cfg: DictConfig, data_manager=None) -> TrainPipeline:
        pass