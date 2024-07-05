from src.data.data_manager import DataManager, NullDataManager
from src.data.data_labeling import DataLabeler
from src.features.feature_selection import SelectFeatures
from src.train import train_ml, train_nn
from src.validate import val_ml, val_nn
from src.model.base_model import Model
from src.visualise import plot_predictions
from typing import Union, Optional
from torch import manual_seed
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from hydra.utils import instantiate
import polars as pl

class TrainPipeline:
    def __init__(self, model: Model, train_func: callable, validate_func: callable, data_manager: Union[DataManager, NullDataManager], cfg: DictConfig, model_cfg: DictConfig, train_cfg: DictConfig, logger, monitor):
        self.model = model
        self.train_func = train_func
        self.validate_func = validate_func
        self.data_manager = data_manager
        self.logger = logger
        self.monitor = monitor
        self.select_features = train_cfg.select_features
        self.feature_selection = SelectFeatures(cfg, data_manager)
        self.data_labeler = DataLabeler(cfg)

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

    def train(self, data: pl.DataFrame, figi: str, show_validation=False, save_best=False):
        data, labels = self.data_labeler.fit(data)
        self.logger.info(f'Data shape: {data.shape}, '
                        f'Buy labels count: {len(labels[labels==self.data_labeler.label_meaning["buy"]])}, '
                        f'Sell labels count: {len(labels[labels==self.data_labeler.label_meaning["sell"]])}'
                        )
        if self.select_features:
            data = self.feature_selection.fit_transform(data, labels, figi)
        self.logger.info(f'Data shape after feature selection: {data.shape}')
        self.logger.info(f'Data has colummns: {data.columns}')
        self.data_manager.write_processed_share(figi, data)
        self.data_manager.write_processed_share_labels(figi, labels)
        
        if self.model is None:
            if hasattr(self.model_cfg, 'hidden_layers'):
                manual_seed(self.train_cfg.random_state)
                self.model = instantiate(self.model_cfg, input_size=data.drop('time').shape[1], hidden_layers=self.model_cfg.hidden_layers)
            elif hasattr(self.model_cfg, '_target_'):
                self.model = instantiate(self.model_cfg)

        model, logs, validation_metrics, validation_data = self.train_func(
            self.model, data, labels, self.validate_func,
            self.logger, self.monitor, self.train_cfg
        )
        self.data_manager.save_model(model, figi, save_best=save_best)
        if show_validation:
            data_val, labels_val, predicted, probas = validation_data
            plot_predictions(data_val, predicted, probas)
        return model, logs, validation_metrics, validation_data


class TrainPipelineFactory:
    @staticmethod
    def create_ml_pipeline(cfg: DictConfig, model_cfg: DictConfig, train_cfg: DictConfig, model: Optional[Model], logger, monitor=None, data_manager: Optional[DataManager] = None) -> TrainPipeline:
        return TrainPipeline(
            model,
            train_ml,
            val_ml,
            data_manager if data_manager is not None else NullDataManager(),
            cfg,
            model_cfg,
            train_cfg,
            logger,
            monitor                
        )
        

    @staticmethod
    def create_nn_pipeline(cfg: DictConfig, model_cfg: DictConfig, train_cfg: DictConfig, model: Optional[Model], logger, monitor=None, data_manager: Optional[DataManager] = None) -> TrainPipeline:
        return TrainPipeline(
            model,
            train_nn,
            val_nn,
            data_manager if data_manager is not None else NullDataManager(),
            cfg,
            model_cfg,
            train_cfg,
            logger,    
            monitor if monitor is not None else SummaryWriter('runs/default')       
        )