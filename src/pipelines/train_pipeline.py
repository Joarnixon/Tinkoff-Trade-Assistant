from src.data.data_manager import DataManager, NullDataManager
from src.data.data_labeling import DataLabeler
from src.pipelines.data_pipeline import DataPipelineFactory
from src.features.feature_selection import SelectFeatures
from src.train import train_ml, train_nn
from src.validate import val_ml, val_nn
from src.model.base_model import Model
from src.visualise import plot_predictions
from typing import Union, Optional
from torch import manual_seed
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import polars as pl
import time

class TrainPipeline:
    """
    Used to train the model for the specific figi.
    To train the same model for multiple figi provide figi in train arg.
    If model is none will instantiate the model from config files.
    """
    def __init__(self, model: Model, figi: str, train_func: callable, validate_func: callable, data_manager: Union[DataManager, NullDataManager], cfg: DictConfig, model_cfg: DictConfig, train_cfg: DictConfig, logger, monitor):
        self.model = model
        self.train_func = train_func
        self.validate_func = validate_func
        self.data_manager = data_manager
        self.logger = logger
        self.select_features = train_cfg.select_features
        self.feature_selection = SelectFeatures(cfg, data_manager)
        self.data_labeler = DataLabeler(cfg)
        self.data_pipeline = DataPipelineFactory.create_offline_pipeline(cfg)

        self.model_cfg = model_cfg
        self.train_cfg = OmegaConf.merge(train_cfg, model_cfg, cfg.data.data_labeling)

        self.figi = figi
        self.monitor = monitor if monitor is not None else SummaryWriter(f'runs/offline/{figi}/{str(time.time())}')


    def prepare_data(self, figi):
        share = self.data_manager.load_share(figi)
        share = self.data_pipeline.transform(share)
        data, labels = self.data_labeler.fit(share)
        self.logger.info(f'Data shape: {data.shape}, '
                        f'Buy labels count: {len(labels[labels==self.data_labeler.label_meaning["buy"]])}, '
                        f'Sell labels count: {len(labels[labels==self.data_labeler.label_meaning["sell"]])}'
                        )
        if self.select_features:
            data = self.feature_selection.fit_transform(data, labels, figi)
        self.logger.info(f'Data shape after feature selection: {data.shape}')
        self.logger.info(f'Data has colummns: {data.columns}')
        self.data_manager.write_processed_share(self.figi, data)
        self.data_manager.write_processed_share_labels(self.figi, labels)
        return data, labels

    def train(self, show_validation=False, figi=None):
        if figi is None:
            figi = self.figi
        data, labels = self.prepare_data(figi)
        
        if self.model is None:
            if hasattr(self.model_cfg, 'hidden_layers'):
                manual_seed(self.train_cfg.random_state)
                self.model = instantiate(self.model_cfg, input_size=data.drop('time').shape[1], hidden_layers=self.model_cfg.hidden_layers)
            elif hasattr(self.model_cfg, '_target_'):
                self.model = instantiate(self.model_cfg)

        model, best_score, validation_metrics, validation_data = self.train_func(
            self.model, data, labels, self.validate_func,
            self.logger, self.monitor, self.train_cfg
        )
        self.data_manager.save_model(model, best_score, figi)
        if show_validation:
            data_val, labels_val, predicted, probas = validation_data
            plot_predictions(data_val, predicted, probas)
        return model, best_score, validation_metrics, validation_data


class TrainPipelineFactory:
    @staticmethod
    def create_ml_pipeline(cfg: DictConfig, model_cfg: DictConfig, train_cfg: DictConfig, model: Optional[Model], figi, logger, monitor=None, data_manager: Optional[DataManager] = None) -> TrainPipeline:
        return TrainPipeline(
            model,
            figi,
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
    def create_nn_pipeline(cfg: DictConfig, model_cfg: DictConfig, train_cfg: DictConfig, model: Optional[Model], figi: str, logger, monitor=None, data_manager: Optional[DataManager] = None) -> TrainPipeline:
        return TrainPipeline(
            model,
            figi,
            train_nn,
            val_nn,
            data_manager if data_manager is not None else NullDataManager(),
            cfg,
            model_cfg,
            train_cfg,
            logger,    
            monitor 
        )