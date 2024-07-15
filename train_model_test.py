from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate
from src.pipelines.data_pipeline import DataPipelineFactory
from src.pipelines.train_pipeline import TrainPipelineFactory
from src.data.data_manager import DataManager
from config import update_processed_data_folder
import logging




logger = logging.getLogger("tinkoff.invest.logging")
logger.setLevel(logging.INFO)



@hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
def main(cfg: DictConfig) -> None:
    update_processed_data_folder(cfg)
    figi = 'TCS00A106YF0'
    dm = DataManager(cfg)
    model_cfg = OmegaConf.load('config/models/linear_nn.yaml')
    train_cfg = OmegaConf.load('config/train/train_nn.yaml')
    tp = TrainPipelineFactory.create_nn_pipeline(cfg, model_cfg, train_cfg, None, figi, logger, None, dm)
    model, _, _ , _= tp.train(show_validation=True, save_best=True)

main()